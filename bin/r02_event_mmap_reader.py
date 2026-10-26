"""Read the frozen uncompressed event NPZs without copying unrequested people.

This is an opt-in reader, not a new producer or scientific filter. Each source
is hashed and fully validated once per cache lifetime, with inode/size/time
checks before and after every use. Mount sources read-only. Compressed ZIPs and
object arrays are rejected rather than silently changing the memory strategy.
"""
from contextlib import contextmanager
from pathlib import Path
import struct
import zipfile
import numpy as np

import r02_genomic_event_inputs as original

require = original.require


def signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _metadata_paths(directory, manifest_sha256):
    """Return the authenticated manifest and its complete non-NPZ inventory."""
    root = Path(directory).resolve()
    manifest_path = root / 'manifest.json'
    require(original.sha(manifest_path) == manifest_sha256,
            'Chromosome manifest SHA256 differs')
    manifest = original.load_json(manifest_path)
    outputs = manifest.get('outputs_sha256')
    require(isinstance(outputs, dict), 'Chromosome output inventory differs')
    paths = [manifest_path]
    for name in sorted(name for name in outputs if not name.endswith('.npz')):
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Unsafe chromosome metadata path')
        path = (root / relative).resolve()
        require(path.is_relative_to(root), 'Chromosome metadata escaped its root')
        paths.append(path)
    require(len(paths) == len(set(paths)) and all(path.is_file() for path in paths),
            'Exact non-genotype metadata inventory required')
    return tuple(paths)


def authenticate_event_chromosome(directory, manifest_sha256):
    """Bind a fully authenticated base reader to pre-construction signatures."""
    paths = _metadata_paths(directory, manifest_sha256)
    before = tuple((str(path), signature(path)) for path in paths)
    reader = original.EventChromosome(directory, manifest_sha256)
    after = tuple((str(path), signature(path)) for path in paths)
    require(after == before, 'Chromosome metadata changed during authentication')
    return reader, before


@contextmanager
def mapped_arrays(path):
    """Map stored numeric NPY members; hold no descriptor after this context."""
    arrays = {}
    try:
        with zipfile.ZipFile(path) as archive, Path(path).open('rb') as stream:
            members = archive.infolist()
            require(len(members) == len(original.ARRAY_FIELDS) and
                    {m.filename for m in members} == {k+'.npy' for k in original.ARRAY_FIELDS},
                    'Exact event NPZ member inventory required')
            for member in members:
                require(member.compress_type == zipfile.ZIP_STORED and
                        not member.flag_bits & 1, 'Stored unencrypted NPY required')
                stream.seek(member.header_offset)
                header = stream.read(30)
                require(len(header) == 30 and header[:4] == b'PK\x03\x04', 'Invalid local ZIP header')
                name_len, extra_len = struct.unpack_from('<HH', header, 26)
                stream.seek(name_len+extra_len, 1)
                payload_start = stream.tell()
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported NPY header version')
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream)
                require(not dtype.hasobject and dtype.kind in 'buif' and len(shape) in (1, 2),
                        'Numeric one/two-dimensional event arrays required')
                size = int(np.prod(shape, dtype=object))*dtype.itemsize
                require(stream.tell()-payload_start+size == member.file_size,
                        'NPY extent differs from ZIP member')
                arrays[member.filename[:-4]] = np.memmap(path, dtype=dtype, mode='r',
                    offset=stream.tell(), shape=shape, order='F' if fortran else 'C')
        yield arrays
    finally:
        for array in arrays.values():
            array._mmap.close()


def validate_chunk(data, chunk, n_people):
    """Same source invariants as the frozen reference reader, once per source."""
    n = chunk['n_sites']
    for name, value in data.items():
        require(value.shape == ((n_people, n) if name in {'dosage','observed'} else (n,)),
                'Array shape differs')
    pos, dose, observed = data['position_bp'], data['dosage'], data['observed']
    require(pos.dtype == np.int64 and np.all(np.diff(pos)>0) and
            pos[0] == chunk['start_bp'] and pos[-1] == chunk['end_bp'], 'Chunk positions differ')
    require(dose.dtype == np.uint8 and observed.dtype == np.bool_ and
            np.all(dose[observed] <= 2) and np.all(dose[~observed] == 255),
            'Genotype dosage/mask differs')
    require(np.all(data['ref'] < 4) and np.all(data['alt'] < 4) and
            np.all(data['ref'] != data['alt']) and np.all(data['counted_allele'] <= 1),
            'Nucleotide/count identity differs')
    require(np.all(data['allele_number'] > 0) and np.all(data['allele_count'] >= 0) and
            np.all(data['allele_count'] <= data['allele_number']) and
            np.allclose(data['frequency'], data['allele_count']/data['allele_number'], rtol=0, atol=1e-12),
            'Frequency numerator/denominator disagree')


class VerifiedChunkCache:
    """Bounded metadata only; mappings close at the end of each read."""
    def __init__(self, max_entries):
        require(type(max_entries) is int and max_entries > 0, 'Positive cache entry bound required')
        self.max_entries = max_entries
        self.entries = {}
        self.bytes_hashed = 0

    @contextmanager
    def open(self, path, chunk, n_people, max_array_mb):
        path = Path(path).resolve()
        before = signature(path)
        require(before[2] == chunk['bytes'], 'Genotype chunk size changed')
        key = (str(path), chunk['sha256'], n_people, chunk['n_sites'],
               chunk['start_bp'], chunk['end_bp'])
        known = self.entries.get(key)
        if known is not None:
            require(known == before, 'Previously verified genotype source changed')
        else:
            require(len(self.entries) < self.max_entries, 'Verified-source cache budget exceeded')
            require(original.sha(path) == chunk['sha256'], 'Genotype chunk SHA256 differs')
            self.bytes_hashed += before[2]
        with zipfile.ZipFile(path) as archive:
            require(sum(m.file_size for m in archive.infolist())*4 <= max_array_mb*2**20,
                    'Chunk validation exceeds original memory bound')
        with mapped_arrays(path) as data:
            if known is None:
                validate_chunk(data, chunk, n_people)
            require(signature(path) == before, 'Source changed during validation')
            yield data
            require(signature(path) == before, 'Source changed during read')
        self.entries[key] = before


class MappedEventChromosome(original.EventChromosome):
    def __init__(self, directory, manifest_sha256, *, cache, max_window_sites=None):
        reader, metadata_signatures = authenticate_event_chromosome(
            directory, manifest_sha256)
        self._copy_authenticated_reader(reader)
        self._finish_authenticated_reader(cache, max_window_sites,
                                          metadata_signatures)

    @classmethod
    def from_authenticated(cls, reader, *, metadata_signatures, cache,
                           max_window_sites=None):
        """Convert an already authenticated reader without reopening metadata.

        Attributes are assigned explicitly.  The source reader is not mutated,
        and the only copied mapping is the resource dictionary whose external
        row limit can be raised by the authenticated request.
        """
        require(type(reader) is original.EventChromosome,
                'Exact authenticated EventChromosome required')
        converted = cls.__new__(cls)
        converted._copy_authenticated_reader(reader)
        converted._finish_authenticated_reader(cache, max_window_sites,
                                                metadata_signatures)
        return converted

    def _copy_authenticated_reader(self, reader):
        for name in ('root', 'manifest', 'manifest_sha256', 'cfg', 'source',
                     'persons', 'roles', 'blocks', 'chrom', 'length', 'people',
                     'points'):
            setattr(self, name, getattr(reader, name))
        self.r = dict(reader.r)

    def _finish_authenticated_reader(self, cache, max_window_sites,
                                     metadata_signatures):
        if max_window_sites is not None:
            require(type(max_window_sites) is int and
                    max_window_sites >= self.r['max_window_sites'],
                    'external preparation row limit differs')
            self.r = {**self.r, 'max_window_sites': max_window_sites}
        require(isinstance(cache, VerifiedChunkCache), 'Explicit verified-source cache required')
        self.cache = cache
        expected_paths = tuple(map(str, _metadata_paths(
            self.root, self.manifest_sha256)))
        require(isinstance(metadata_signatures, tuple) and
                tuple(path for path, _ in metadata_signatures) == expected_paths and
                all(signature(path) == expected
                    for path, expected in metadata_signatures),
                'Authenticated chromosome metadata changed before conversion')
        self._metadata_signatures = metadata_signatures

    def _assert_metadata_unchanged(self):
        require(all(signature(path) == expected
                    for path, expected in self._metadata_signatures),
                'Authenticated chromosome metadata changed')

    def load_window(self, *args, **kwargs):
        """Check every authenticated non-NPZ file around each complete view."""
        self._assert_metadata_unchanged()
        try:
            return super().load_window(*args, **kwargs)
        finally:
            self._assert_metadata_unchanged()

    def read(self, kind, ids, lower, upper, excluded):
        columns = {k: [] for k in original.ARRAY_FIELDS}
        n_sites = 0
        for chunk in self.manifest['channels'][kind]:
            if chunk['end_bp'] < lower or chunk['start_bp'] > upper:
                continue
            with self.cache.open(self.safe_path(chunk['path']), chunk, len(self.persons),
                                 self.r['max_array_mb']) as data:
                pos = data['position_bp']
                allowed = (pos >= lower) & (pos <= upper)
                if excluded is not None:
                    allowed &= ~((pos >= excluded[0]) & (pos <= excluded[1]))
                n_sites += int(allowed.sum())
                require(n_sites <= self.r['max_window_sites'], 'Window limit exceeded; no truncation')
                selected = np.flatnonzero(allowed)
                for name in original.ARRAY_FIELDS:
                    value = data[name]
                    # Advanced indexing copies only requested entries before the mmap closes.
                    columns[name].append(value[np.ix_(ids, selected)] if name in {'dosage','observed'}
                                         else value[allowed])
        return {name: np.concatenate(chunks, axis=1 if name in {'dosage','observed'} else 0)
                if chunks else np.empty((len(ids), 0) if name in {'dosage','observed'} else (0,))
                for name, chunks in columns.items()}
