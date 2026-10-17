#!/usr/bin/env python3
"""Authenticate and deduplicate equivalent K04 whole-chromosome common views.

This operational producer never reads targets, chooses predictor columns, or
fits a model. It retains one source NPZ only when the two independently named
part01/part02 views for a chromosome have identical scientific metadata and
exact array content.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

SCHEMA = 'r02_k04_common_view_dedupe_request_v1'
OUTPUT = 'r02_k04_common_view_deduplicated_v1'
INDEX = 'r02_k04_common_view_tar_index_v1'
CATALOGUE = 'r02_k04_common_view_catalogue_v1'
OPERATION = 'VERIFY_EXACT_WHOLE_CHROMOSOME_VIEW_EQUIVALENCE_AND_DEDUPLICATE'
ARRAYS = ('scores','dual_loadings','eigenvalues','fit_weights','fit_indices','observed_counts')
MANIFEST_FIELDS = {
    'schema','stage','request_sha256','source_contract_sha256','implementation_sha256',
    'block_id','exclusion_scope','allowed_input_intervals','partials','n_sites',
    'numerical_rank','numerical_tolerance','weighting','targets_read','outputs_sha256',
    'biological_validation','common_information',
}
SEMANTIC_FIELDS = MANIFEST_FIELDS - {'request_sha256','block_id','outputs_sha256'}
PREFIX = 'gs://projects-usp/dnaBr-lai/datalake/'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_digests(path):
    hs, hm, size = hashlib.sha256(), hashlib.md5(), 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            size += len(block); hs.update(block); hm.update(block)
    return dict(bytes=size, sha256=hs.hexdigest(),
                md5_base64=base64.b64encode(hm.digest()).decode())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def implementation():
    return sha(__file__)


def chromosome(value):
    text = str(value).removeprefix('chr')
    require(text.isdigit() and 1 <= int(text) <= 22, 'Autosome 1..22 required')
    return str(int(text))


def load_request(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and sha(path) == expected, 'Request SHA256 differs')
    value = read_json(path)
    require(set(value) == {'schema','operation','implementation_sha256','archive',
            'chromosomes','resources','scientific_scope'} and
            value['schema'] == SCHEMA and value['operation'] == OPERATION and
            value['implementation_sha256'] == implementation() and
            value['scientific_scope'] == 'SOURCE_EQUIVALENCE_ONLY_NO_PREDICTOR_NO_TARGET_READ',
            'Exact current deduplication request required')
    archive = value['archive']
    require(set(archive) == {'uri','generation','bytes','sha256','md5_base64'} and
            isinstance(archive['uri'], str) and archive['uri'].startswith(PREFIX) and
            not any(c in archive['uri'] for c in '\n\r#?') and
            isinstance(archive['generation'], str) and archive['generation'].isdigit() and
            int(archive['generation']) > 0 and type(archive['bytes']) is int and archive['bytes'] > 0 and
            re.fullmatch('[0-9a-f]{64}', archive['sha256']) and
            len(base64.b64decode(archive['md5_base64'], validate=True)) == 16,
            'Pinned datalake archive identity required')
    resources = value['resources']
    require(set(resources) == {'max_archive_bytes','max_member_bytes','max_array_bytes',
            'max_memory_bytes','max_output_bytes','min_free_bytes'} and
            all(type(x) is int and x > 0 for x in resources.values()) and
            archive['bytes'] <= resources['max_archive_bytes'] and
            2 * resources['max_array_bytes'] <= resources['max_memory_bytes'],
            'Explicit positive resource bounds required')
    require(set(value['chromosomes']) == {str(i) for i in range(1,23)},
            'Exactly autosomes 1..22 required')
    members, logical_paths, member_identities = set(), set(), {}
    for chrom, pair in value['chromosomes'].items():
        require(chromosome(chrom) == chrom and set(pair) == {'part01','part02'},
                'Two candidates required per chromosome')
        for part, desc in pair.items():
            require(set(desc) == {'block_id','manifest','arrays'} and
                    desc['block_id'] == f'chr{int(chrom):02d}_{part}',
                    'Candidate block identity differs')
            for kind in ('manifest','arrays'):
                item = desc[kind]
                require(set(item) == {'path','member','bytes','sha256'} and
                        isinstance(item['path'], str) and not item['path'].startswith('/') and
                        '..' not in Path(item['path']).parts and item['path'] not in logical_paths and
                        isinstance(item['member'], str) and not item['member'].startswith('/') and
                        '..' not in Path(item['member']).parts and
                        type(item['bytes']) is int and 0 < item['bytes'] <= resources['max_member_bytes'] and
                        re.fullmatch('[0-9a-f]{64}', item['sha256']),
                        'Safe unique pinned logical source required')
                logical_paths.add(item['path'])
                identity=(item['bytes'],item['sha256'])
                require(item['member'] not in member_identities or
                        member_identities[item['member']] == identity,
                        'Archive member alias identity differs')
                member_identities[item['member']]=identity
                members.add(item['member'])
    require(len(logical_paths) == 88 and len(members) == 66 and
            len({value['manifest']['member'] for pair in value['chromosomes'].values()
                 for value in pair.values()}) == 44 and
            all(pair['part01']['arrays']['path'] != pair['part02']['arrays']['path'] and
                pair['part01']['arrays']['member'] == pair['part02']['arrays']['member'] and
                pair['part01']['arrays']['bytes'] == pair['part02']['arrays']['bytes'] and
                pair['part01']['arrays']['sha256'] == pair['part02']['arrays']['sha256']
                for pair in value['chromosomes'].values()),
            'Exact 44 logical views backed by 44 manifests and 22 content aliases required')
    return path, value, members


def new_dir(path, reserve):
    path = Path(path).resolve()
    require(not os.path.lexists(path) and path.parent.is_dir() and
            shutil.disk_usage(path.parent).free >= reserve,
            'Fresh output and disk reserve required')
    path.mkdir(mode=0o700)
    return path


def member_bytes(archive, row):
    with Path(archive).open('rb') as stream:
        stream.seek(row['offset_data'])
        data = stream.read(row['bytes'])
    require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
            'Indexed archive member content differs')
    return data


def stage(request_path, request_sha, output):
    request_path, request, wanted = load_request(request_path, request_sha)
    limits, source = request['resources'], request['archive']
    out = new_dir(output, source['bytes'] + limits['min_free_bytes'])
    partial = out/'common-views.tar.partial'; archive = out/'common-views.tar'
    subprocess.run(['gcloud','storage','cp',source['uri']+'#'+source['generation'],str(partial)],
                   check=True, timeout=7200,
                   env={**os.environ,'CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED':'false'})
    observed = file_digests(partial)
    require(all(observed[k] == source[k] for k in ('bytes','sha256','md5_base64')),
            'Downloaded common archive differs')
    os.link(partial, archive); partial.unlink()
    descriptor_by_member = {}
    for pair in request['chromosomes'].values():
        for desc in pair.values():
            for kind in ('manifest','arrays'):
                item=desc[kind]
                physical={'member':item['member'],'bytes':item['bytes'],'sha256':item['sha256']}
                require(item['member'] not in descriptor_by_member or
                        descriptor_by_member[item['member']] == physical,
                        'Archive alias identity differs')
                descriptor_by_member[item['member']]=physical
    rows, seen = {}, set()
    with tarfile.open(archive, 'r:') as bundle:
        for info in bundle:
            if info.name not in wanted:
                continue
            require(info.name not in seen and info.isfile() and not info.issym() and not info.islnk(),
                    'Repeated or unsafe requested tar member')
            seen.add(info.name)
            descriptor = descriptor_by_member[info.name]
            require(info.size == descriptor['bytes'], 'Tar member byte count differs')
            h = hashlib.sha256()
            with bundle.extractfile(info) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == descriptor['sha256'], 'Tar member SHA256 differs')
            rows[info.name] = dict(offset_data=info.offset_data, **descriptor)
    require(seen == wanted, 'Requested common view member absent from archive')
    write_json(out/'archive-index.json', dict(schema=INDEX,request_sha256=request_sha,
        implementation_sha256=implementation(),archive=source,members=rows,
        logical_sources=88,requested_members=len(rows),selective_members_only=True))
    return out


def npz_inventory(path, maximum):
    with zipfile.ZipFile(path) as bundle:
        entries = bundle.infolist()
        require(len(entries) == len(ARRAYS) and
                {x.filename for x in entries} == {x+'.npy' for x in ARRAYS} and
                all(not x.is_dir() and not (x.flag_bits & 1) for x in entries),
                'Continuous NPZ inventory differs')
        total = sum(x.file_size for x in entries)
    require(total <= maximum, 'Continuous NPZ decompression bound exceeded')
    return total


def compare_arrays(left, right, maximum):
    import numpy as np
    npz_inventory(left, maximum); npz_inventory(right, maximum)
    result = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(set(a.files) == set(b.files) == set(ARRAYS), 'Common array names differ')
        for name in ARRAYS:
            x, y = a[name], b[name]
            require(x.shape == y.shape and x.dtype == y.dtype and
                    x.tobytes(order='C') == y.tobytes(order='C'),
                    'Common view array content differs: '+name)
            h = hashlib.sha256()
            h.update(name.encode()); h.update(x.dtype.str.encode())
            h.update(json.dumps(list(x.shape)).encode()); h.update(x.tobytes(order='C'))
            result[name] = dict(shape=list(x.shape),dtype=x.dtype.str,content_sha256=h.hexdigest())
    return result


def dedupe(request_path, request_sha, archive, index_path, chrom, output):
    request_path, request, _ = load_request(request_path, request_sha)
    chrom = chromosome(chrom); limits = request['resources']
    archive, index_path = Path(archive).resolve(), Path(index_path).resolve()
    require(archive.is_file() and archive.stat().st_size == request['archive']['bytes'] and
            index_path.is_file(), 'Staged archive/index missing or changed size')
    index = read_json(index_path)
    require(set(index) == {'schema','request_sha256','implementation_sha256','archive',
                           'members','logical_sources','requested_members','selective_members_only'} and
            index['schema'] == INDEX and index['request_sha256'] == request_sha and
            index['implementation_sha256'] == implementation() and
            index['archive'] == request['archive'] and
            set(index['members']) == {d[k]['member'] for pair in request['chromosomes'].values()
                                      for d in pair.values() for k in ('manifest','arrays')} and
            index['logical_sources'] == 88 and index['requested_members'] == 66 and
            index['selective_members_only'] is True,
            'Authenticated archive index differs')
    candidates = {}
    with tempfile.TemporaryDirectory(prefix='k04-view-', dir=Path(output).resolve().parent) as temp:
        temp = Path(temp)
        for part, desc in request['chromosomes'][chrom].items():
            manifest_bytes = member_bytes(archive, index['members'][desc['manifest']['member']])
            require(hashlib.sha256(manifest_bytes).hexdigest() == desc['manifest']['sha256'],
                    'Candidate manifest changed')
            manifest = json.loads(manifest_bytes)
            require(set(manifest) == MANIFEST_FIELDS and
                    manifest['schema'] == 'r02_structure_materialized_common_merge_v2' and
                    manifest['stage'] == 'FULL_NUMERICAL_COMMON_SPECTRUM_NOT_PREDICTIVE_SELECTION' and
                    manifest['block_id'] == desc['block_id'] and
                    manifest['exclusion_scope'] == 'WHOLE_CHROMOSOME' and
                    manifest['targets_read'] is False and manifest['biological_validation'] is False and
                    manifest['outputs_sha256'] == {'continuous.private.npz':desc['arrays']['sha256']},
                    'Whole-chromosome common manifest scope differs')
            data = member_bytes(archive, index['members'][desc['arrays']['member']])
            path = temp/(part+'.npz'); path.write_bytes(data); path.chmod(0o600)
            npz_inventory(path, limits['max_array_bytes'])
            candidates[part] = (manifest, path, desc)
        a, b = candidates['part01'], candidates['part02']
        require({k:a[0][k] for k in SEMANTIC_FIELDS} ==
                {k:b[0][k] for k in SEMANTIC_FIELDS},
                'part01/part02 scientific manifest content differs')
        arrays = compare_arrays(a[1], b[1], limits['max_array_bytes'])
        require(len(arrays['eigenvalues']['shape']) == 1 and
                arrays['eigenvalues']['shape'][0] == a[0]['numerical_rank'],
                'Common numerical rank/array shape differs')
        out = new_dir(output, limits['min_free_bytes'] + limits['max_output_bytes'])
        canonical = out/'continuous.private.npz'
        shutil.copyfile(a[1], canonical); canonical.chmod(0o600)
        result = dict(schema=OUTPUT,status='EXACT_PART01_PART02_CONTENT_EQUIVALENT_ONE_SOURCE_RETAINED',
            chromosome=chrom,request_sha256=request_sha,implementation_sha256=implementation(),
            source_contract_sha256=a[0]['source_contract_sha256'],exclusion_scope='WHOLE_CHROMOSOME',
            included_partial_ids=sorted(a[0]['partials']),
            n_sites=a[0]['n_sites'],numerical_rank=a[0]['numerical_rank'],weighting=a[0]['weighting'],
            semantic_manifest_sha256=digest({k:a[0][k] for k in SEMANTIC_FIELDS}),
            array_content=arrays,source_candidates={p:dict(block_id=v[2]['block_id'],
                manifest=v[2]['manifest'],arrays=v[2]['arrays']) for p,v in candidates.items()},
            canonical_source='part01',canonical_file_sha256=sha(canonical),
            duplicate_source_retained=False,targets_read=False,predictor_columns_chosen=False,
            biological_validation=False)
        write_json(out/'manifest.json', result)
    require(canonical.stat().st_size <= limits['max_output_bytes'] and
            sha(request_path) == request_sha and implementation() == request['implementation_sha256'],
            'Request/code changed during deduplication')
    return out


def aggregate(request_path, request_sha, directories, output):
    request_path, request, _ = load_request(request_path, request_sha)
    rows = {}
    for directory in directories:
        directory = Path(directory).resolve(); manifest = read_json(directory/'manifest.json')
        chrom = chromosome(manifest['chromosome'])
        require(chrom not in rows and manifest['schema'] == OUTPUT and
                manifest['status'] == 'EXACT_PART01_PART02_CONTENT_EQUIVALENT_ONE_SOURCE_RETAINED' and
                manifest['request_sha256'] == request_sha and
                manifest['implementation_sha256'] == implementation() and
                manifest['duplicate_source_retained'] is False and
                sha(directory/'continuous.private.npz') == manifest['canonical_file_sha256'],
                'Deduplicated chromosome source differs')
        rows[chrom] = dict(source_directory_name=directory.name,
                           manifest_sha256=sha(directory/'manifest.json'),
                           arrays_sha256=manifest['canonical_file_sha256'],
                           n_sites=manifest['n_sites'],numerical_rank=manifest['numerical_rank'])
    require(set(rows) == set(request['chromosomes']), 'Exact 22 deduplicated chromosomes required')
    out = new_dir(output, request['resources']['min_free_bytes'])
    write_json(out/'catalogue.json', dict(schema=CATALOGUE,
        status='ALL22_WHOLE_CHROMOSOME_COMMON_VIEWS_CONTENT_EQUIVALENT_DEDUPLICATED',
        request_sha256=request_sha,implementation_sha256=implementation(),
        chromosomes={c:rows[c] for c in sorted(rows,key=int)},source_views=44,retained_views=22,
        duplicate_views_retained=0,targets_read=False,predictor_columns_chosen=False,
        human_fit_allowed=False,biological_validation=False))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('stage','dedupe','aggregate'),required=True)
    p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--archive');p.add_argument('--index')
    p.add_argument('--chromosome');p.add_argument('--directories',nargs='*')
    a=p.parse_args();os.umask(0o077)
    if a.stage=='stage': result=stage(a.request,a.request_sha256,a.output)
    elif a.stage=='dedupe': result=dedupe(a.request,a.request_sha256,a.archive,a.index,a.chromosome,a.output)
    else: result=aggregate(a.request,a.request_sha256,a.directories,a.output)
    print(json.dumps(dict(stage=a.stage,output=str(result))))


if __name__=='__main__': main()
