"""Authenticated raw-store regions for the masked-genotype objective.

This reader reuses the existing chromosome maps and mmap cache. It does not
choose a mask rate, a model's halo, or a scientific population. EVALUATE is
deliberately unavailable here; final evaluation needs a separately admitted
entry point. Source allele definitions are unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from r02_event_mmap_reader import (MappedEventChromosome, VerifiedChunkCache,
                                   signature, _metadata_paths)
from r02_genomic_event_inputs import (ARRAY_FIELDS, maps, load_contract, bounded_map,
                                     CHROM_SCHEMA, CODE_FILES, material)
from r02_masked_genotypes import CellMaskPlan, mask_both_channels, require


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def role_person_ids(reader, role):
    """Resolve positional role members to authenticated sample identifiers."""
    require(role in reader.roles, 'requested role is absent from authenticated source')
    indices = tuple(reader.roles[role])
    require(all(isinstance(index, (int, np.integer))
                and 0 <= int(index) < len(reader.persons) for index in indices),
            'authenticated role must contain valid person indices')
    identifiers = tuple(reader.persons[int(index)]['sample_id'] for index in indices)
    require(len(identifiers) == len(set(identifiers))
            and all(isinstance(identifier, str) and identifier
                    for identifier in identifiers),
            'authenticated role person identifiers differ')
    return frozenset(identifiers)


def coordinate_mask(person_ids, chromosome, edges_cm, *, seed, epoch, probability):
    """Deterministic per-person mask, independent of batching and all genotypes.

    Probability is an explicit design input, not a recommended value. Random
    streams depend on person/chromosome/seed/epoch and the complete cell axis;
    reshuffling people or splitting a batch does not change their masks.
    """
    require(type(seed) is int and seed >= 0 and type(epoch) is int and epoch >= 0,
            'nonnegative integer seed and epoch required')
    require(type(probability) in (int, float) and math.isfinite(probability)
            and 0 < probability < 1, 'explicit mask probability in (0,1) required')
    rows = []
    for person in person_ids:
        key = json.dumps(['r02_coordinate_mask_v1', person, chromosome, seed, epoch],
                         separators=(',', ':')).encode()
        state = int.from_bytes(hashlib.sha256(key).digest()[:16], 'little')
        rows.append(np.random.Generator(np.random.PCG64(state)).random(len(edges_cm))
                    < probability)
    return CellMaskPlan(tuple(person_ids), chromosome, edges_cm, np.asarray(rows, bool))


def owned_loci(cell_indices, core_start_cell, core_stop_cell):
    """Half-open cell ownership; halos never become extra training labels."""
    return (cell_indices >= core_start_cell) & (cell_indices < core_stop_cell)


class RawMaskedChromosome(MappedEventChromosome):
    """Read an authenticated raw schema without impersonating its producer.

    Producer code hashes remain immutable provenance. A new reader for a new
    objective need not have the same source code as the historical producer.
    The historical pair-task reader and its code-equality rule are untouched.
    """
    def __init__(self, directory, manifest_sha256, *, cache, max_window_sites):
        self.root = Path(directory).resolve()
        paths = _metadata_paths(self.root, manifest_sha256)
        before = tuple((str(path), signature(path)) for path in paths)
        self.manifest_sha256 = manifest_sha256
        self.manifest = m = json.loads((self.root/'manifest.json').read_text())
        require(m.get('schema') == CHROM_SCHEMA and m.get('stage') ==
                'COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING'
                and m.get('training_allowed') is False, 'raw genotype schema differs')
        require(isinstance(m.get('code_sha256'), dict)
                and set(m['code_sha256']) == set(CODE_FILES)
                and all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h)
                        for h in m['code_sha256'].values()),
                'original producer provenance missing')
        required = {'event_inputs.json', 'source_verification.json', 'samples.txt',
                    'map.tsv', 'common_loci.tsv', 'common.bcftools.log', 'rare.bcftools.log',
                    *{'source/'+n for n in ('materialize.json', *material.META)}}
        required |= {c['path'] for chunks in m['channels'].values() for c in chunks}
        cfg_preview = json.loads((self.root/'event_inputs.json').read_text())
        if 'fit_caches' in cfg_preview:
            required |= {'fit_eligible.tsv', 'fit_manifest.json', 'fit_contract.json',
                         'common_targets.tsv'}
        require(set(m['outputs_sha256']) == required, 'raw artifact inventory incomplete')
        for name, digest in m['outputs_sha256'].items():
            path = self.safe_path(name)
            require(path.is_file() and (name.endswith('.npz') or _sha(path) == digest),
                    'raw source metadata missing or changed')
        self.cfg, self.source, self.persons, self.roles, self.blocks, _ = load_contract(
            self.root, m['event_contract_sha256'])
        self.chrom = str(m['chromosome'])
        self.length = self.source['sources'][self.chrom]['length_bp']
        require(m['source_contract_sha256'] == self.cfg['materialize_contract_sha256']
                and m['n_people'] == len(self.persons)
                and m['sample_ids_sha256'] == material.evidence.sample_hash(
                    [p['sample_id'] for p in self.persons]), 'raw source/cohort binding differs')
        self.people = {p['sample_id']: i for i, p in enumerate(self.persons)}
        require(len(self.people) == len(self.persons), 'duplicate raw person identity')
        require(m['role_counts'] == {r: len(ids) for r, ids in self.roles.items()},
                'raw role counts differ from source persons')
        self.r = {**self.cfg['resources'], 'max_window_sites': max_window_sites}
        self.points = bounded_map(self.root/'map.tsv', self.chrom, self.length,
                                   'tsv_chrom_pos_cm', self.r)
        require(set(m['channels']) == {'common', 'rare'}, 'raw channels differ')
        for chunks in m['channels'].values():
            previous = 0
            for c in chunks:
                require(set(c) == {'path', 'sha256', 'n_sites', 'bytes', 'start_bp', 'end_bp'}
                        and c['start_bp'] > previous and c['end_bp'] >= c['start_bp']
                        and 0 < c['n_sites'] <= self.r['chunk_sites']
                        and m['outputs_sha256'].get(c['path']) == c['sha256'],
                        'raw chunk order or identity differs')
                previous = c['end_bp']
        require(all(signature(path) == stamp for path, stamp in before),
                'raw metadata changed during authentication')
        self._metadata_signatures, self.cache = before, cache


class MaskedStoreReader:
    def __init__(self, store, manifest_sha256, axis_directory, axis_receipt_sha256,
                 *, role, max_people, max_window_sites, max_cache_entries):
        require(role in ('FIT', 'SELECT'), 'EVALUATE is not enabled by this reader')
        require(type(max_people) is int and max_people > 0
                and type(max_window_sites) is int and max_window_sites > 0,
                'explicit positive resource bounds required')
        self.role, self.max_people = role, max_people
        self.reader = RawMaskedChromosome(store, manifest_sha256,
            cache=VerifiedChunkCache(max_cache_entries), max_window_sites=max_window_sites)
        self.chromosome = int(self.reader.chrom)
        self.axis_directory = Path(axis_directory).resolve()
        receipt_path = self.axis_directory / 'receipt.json'
        cells_path = self.axis_directory / 'cells.tsv'
        before = (signature(receipt_path), signature(cells_path))
        require(_sha(receipt_path) == axis_receipt_sha256, 'cell receipt changed')
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get('schema') == 'r02_autosomal_cell_axis_receipt_v1'
                and receipt.get('status') == 'COMPLETE'
                and receipt.get('chromosome') == str(self.chromosome)
                and receipt.get('source_binding', {}).get('manifest_sha256') == manifest_sha256
                and receipt.get('map_sha256') == self.reader.manifest['outputs_sha256']['map.tsv'],
                'cell axis is not bound to this chromosome source and map')
        require(_sha(cells_path) == receipt['cells_tsv_sha256'], 'cell coordinates changed')
        with cells_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        self.edges = np.asarray([[float(r['left_edge_cm']), float(r['right_edge_cm'])]
                                 for r in rows], dtype=np.float64)
        # Validate axis geometry without opening genotypes or choosing a mask.
        CellMaskPlan(('coordinate-validation',), self.chromosome, self.edges,
                     np.zeros((1, len(rows)), bool))
        require(receipt.get('cell_count') == len(rows), 'cell count differs from receipt')
        require(before == (signature(receipt_path), signature(cells_path)),
                'cell metadata changed during authentication')
        self.axis_signatures = before
        self.axis_receipt_sha256 = axis_receipt_sha256
        self.cells_sha256 = receipt['cells_tsv_sha256']
        require(self.edges[0, 0] >= self.reader.points[0][1]
                and self.edges[-1, 1] <= self.reader.points[-1][1],
                'cell axis is outside authenticated chromosome map')

    def read(self, plan, *, core_start_cell, core_stop_cell, halo_cm):
        """Read both complete local channels and score only the owned cells.

        halo_cm must be >= the model's full receptive reach (including stacked
        attention); the caller must bind it to the model configuration. This
        method checks the declared geometry, not the architecture's equation.
        """
        require(isinstance(plan, CellMaskPlan) and plan.chromosome == self.chromosome
                and np.array_equal(plan.edges_cm, self.edges), 'mask/cell source differs')
        require(len(plan.person_ids) <= self.max_people, 'person budget exceeded')
        require(type(core_start_cell) is int and type(core_stop_cell) is int
                and 0 <= core_start_cell < core_stop_cell <= len(self.edges),
                'invalid owned cell range')
        require(type(halo_cm) in (int, float) and math.isfinite(halo_cm) and halo_cm >= 0,
                'finite nonnegative model halo in cM required')
        reader = self.reader
        reader._assert_metadata_unchanged()
        require(self.axis_signatures == (signature(self.axis_directory/'receipt.json'),
                                         signature(self.axis_directory/'cells.tsv')),
                'cell metadata changed since authentication')
        require(set(plan.person_ids) <= set(reader.people), 'unknown person')
        ids = [reader.people[p] for p in plan.person_ids]
        require(len(ids) <= reader.r['max_batch_people'] and
                len(ids)*reader.r['max_window_sites']*400
                + reader.r['max_map_points']*256
                + len(reader.persons)*reader.r['chunk_sites']*48
                <= reader.r['max_array_mb']*2**20,
                'regional raw arrays exceed declared source memory budget')
        require(all(reader.persons[i]['role'] == self.role for i in ids),
                'person role differs; mixed roles and EVALUATE forbidden')
        low = max(self.edges[0, 0], self.edges[core_start_cell, 0] - halo_cm)
        high = min(self.edges[-1, 1], self.edges[core_stop_cell-1, 1] + halo_cm)
        lower_bp = math.floor(maps.inverse_cm(reader.points, low, 'left'))
        upper_bp = math.ceil(maps.inverse_cm(reader.points, high, 'right'))
        raw = {}
        for name in ('common', 'rare'):
            data = reader.read(name, ids, lower_bp, upper_bp, None)
            # The historical reader returns float empties; preserve raw schema
            # explicitly even for a region with no loci in one channel.
            dtypes = {'dosage': np.uint8, 'observed': bool, 'position_bp': np.int64,
                      'ref': np.uint8, 'alt': np.uint8, 'counted_allele': np.uint8,
                      'allele_count': np.int64, 'allele_number': np.int64}
            keep = (data['position_cm'] >= low) & (data['position_cm'] <= high)
            raw[name] = {k: np.asarray(v[:, keep] if k in ('dosage', 'observed') else v[keep],
                                      dtype=dtypes.get(k, np.float64))
                         for k, v in data.items()}
            require(set(raw[name]) == ARRAY_FIELDS, 'raw fields differ')
        masked = mask_both_channels(raw['common'], raw['rare'], chromosome=self.chromosome,
                                   person_ids=plan.person_ids, plan=plan)
        owners = {name: owned_loci(value.inputs['cell_index'], core_start_cell, core_stop_cell)
                  for name, value in masked.items()}
        reader._assert_metadata_unchanged()
        return {'channels': masked, 'owned': owners,
                'components': tuple(reader.persons[i]['component'] for i in ids),
                'lineage': {'schema': 'r02_masked_store_region_v1', 'role': self.role,
                    'raw_producer_code_sha256': reader.manifest['code_sha256'],
                    'masked_reader_sha256': _sha(__file__),
                    'chromosome': self.chromosome, 'manifest_sha256': reader.manifest_sha256,
                    'cell_receipt_sha256': self.axis_receipt_sha256,
                    'cells_sha256': self.cells_sha256, 'mask_identity': plan.identity,
                    'core_cells': [core_start_cell, core_stop_cell], 'halo_cm': halo_cm,
                    'context_cm': [float(low), float(high)], 'training_authorized': False}}
