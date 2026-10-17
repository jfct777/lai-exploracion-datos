#!/usr/bin/env python3
"""Exact rare interval counts from authenticated events and unmapped tails.

This module preserves the source-minor catalogue and all analytical persons.
It does not choose communities, graph thresholds, predictive rank or outcomes.
RareCounts is the shared integer I/U/Q algorithm; the batching below changes
only when sparse products are accumulated, never a site or genotype rule.
"""
from __future__ import annotations

import hashlib
import io
import argparse
from bisect import bisect_left
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import zipfile

import numpy as np

import r02_structure_continuous as continuous
import r02_structure_discrete_reuse as reuse

material = continuous.discrete.materialize
evidence = material.evidence
require = material.require
SCHEMA = 'r02_structure_rare_interval_inputs_v1'
CODE = tuple(dict.fromkeys(('r02_structure_rare_inputs.py', *continuous.CODE_FILES, *reuse.CODE_FILES)))


def implementation():
    return {name: material.sha(Path(__file__).with_name(name)) for name in CODE}


class SparseBatchCounter:
    """Coalesce small event chunks before the established sparse count product."""
    def __init__(self, n_people, *, batch_sites, max_sparse_cells):
        require(type(batch_sites) is int and batch_sites > 0 and
                type(max_sparse_cells) is int and max_sparse_cells > 0,
                'Explicit positive sparse batching bounds required')
        self.counts = evidence.RareCounts(n_people)
        self.batch_sites = batch_sites
        self.max_cells = max_sparse_cells
        self.rows = [[], [], [], []]
        self.sites = self.cells = self.flushes = 0

    def flush(self):
        if not self.sites:
            return
        args = [np.concatenate(v) if v else np.empty(0, np.int64) for v in self.rows]
        self.counts.add_sparse(*args, self.sites)
        self.rows = [[], [], [], []]
        self.sites = self.cells = 0
        self.flushes += 1

    def add(self, dosage, observed):
        require(dosage.ndim == 2 and dosage.shape == observed.shape and
                dosage.shape[0] == self.counts.n and dosage.dtype == np.uint8 and
                observed.dtype == np.bool_, 'Rare dose/mask shape or dtype differs')
        require(np.array_equal(dosage == 255, ~observed) and
                np.all(dosage[observed] <= 2), 'Rare missing sentinel or diploid dosage differs')
        # Split on the technical site batch size; a genotype remains unknown.
        for offset in range(0, dosage.shape[1], self.batch_sites):
            d = dosage[:, offset:offset+self.batch_sites]
            o = observed[:, offset:offset+self.batch_sites]
            cr, cc = np.nonzero((d > 0) & o)
            mr, mc = np.nonzero(~o)
            cells = len(cr)+len(mr)
            require(cells <= self.max_cells, 'One rare chunk exceeds sparse-cell bound')
            if self.sites and (self.sites+d.shape[1] > self.batch_sites or
                               self.cells+cells > self.max_cells):
                self.flush()
            for target, value in zip(self.rows, (cr, cc+self.sites, mr, mc+self.sites)):
                if value.size:
                    target.append(value)
            self.sites += d.shape[1]
            self.cells += cells

    def finish(self):
        self.flush()
        return self.counts.finish()


class IntervalCounters:
    """Count the whole chromosome and its exact allowed feature intervals."""
    def __init__(self, n_people, tasks, *, chromosome, length_bp, batch_sites,
                 max_sparse_cells, max_memory_bytes):
        require(tasks and all(t['kind'] == 'features' and t['chromosome'] == chromosome
                              for t in tasks), 'Exact feature tasks for one chromosome required')
        require(len({t['task_id'] for t in tasks}) == len(tasks), 'Duplicate rare task')
        for task in tasks:
            previous = 0
            for start, end in task['intervals']:
                require(type(start) is int and type(end) is int and
                        previous < start <= end <= length_bp, 'Rare intervals overlap or exceed chromosome')
                previous = end
        full = [t for t in tasks if t['intervals'] == [[1, length_bp]]]
        require(len(full) == 1, 'Exactly one full chromosome count required for source equivalence')
        self.full_id = full[0]['task_id']
        self.tasks = {t['task_id']: t for t in tasks}
        # Each accumulator retains three n² int64 matrices. Finish plus the
        # authenticated historical comparison has a separate dense allowance.
        self.memory_bound = (len(tasks)*24+112)*n_people*n_people + \
                            len(tasks)*64*max_sparse_cells + 16*2**20
        require(type(max_memory_bytes) is int and self.memory_bound <= max_memory_bytes,
                'Rare accumulator/sparse/verification memory bound exceeded')
        self.counters = {key: SparseBatchCounter(n_people, batch_sites=batch_sites,
                                                max_sparse_cells=max_sparse_cells)
                         for key in self.tasks}
        self.seen = {key: 0 for key in self.tasks}
        self.ranges = []
        self.length = length_bp

    def add(self, position, dosage, observed):
        require(position.ndim == 1 and position.dtype.kind in 'iu' and len(position) > 0 and
                dosage.shape[1] == len(position) and
                int(position[0]) > 0 and np.all(position[1:] > position[:-1]) and
                int(position[-1]) <= self.length, 'Rare source coordinates duplicated, unordered or out of bounds')
        span = (int(position[0]), int(position[-1]))
        at = bisect_left(self.ranges, span)
        require((at == 0 or self.ranges[at-1][1] < span[0]) and
                (at == len(self.ranges) or span[1] < self.ranges[at][0]),
                'Rare source chunk coordinate ranges overlap')
        self.ranges.insert(at, span)
        for key, task in self.tasks.items():
            keep = np.zeros(len(position), bool)
            for start, end in task['intervals']:
                keep |= (position >= start) & (position <= end)
            if np.any(keep):
                self.counters[key].add(dosage[:, keep], observed[:, keep])
                self.seen[key] += int(keep.sum())

    def finish(self):
        # Yield and release one completed result at a time to bound temporary
        # dense matrices; caller must not retain every yielded array set.
        for key in list(self.counters):
            counter = self.counters.pop(key)
            arrays = counter.finish()
            require(int(arrays['n_sites']) == self.seen[key], 'Interval denominator differs')
            yield key, arrays, counter.flushes


def load_event(data, desc, n_people, source_n_people, *, max_array_bytes):
    """Decode one pinned raw event with fixed source-minor identities."""
    require(hashlib.sha256(data).hexdigest() == desc['sha256'] and len(data) == desc['bytes'],
            'Rare event bytes or SHA256 differs')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        require(len(members) == len(continuous.EVENT_FIELDS) and
                {m.filename for m in members} == {k+'.npy' for k in continuous.EVENT_FIELDS} and
                4*sum(m.file_size for m in members) <= max_array_bytes,
                'Rare event decompression inventory or memory bound differs')
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        a = {key: z[key] for key in continuous.EVENT_FIELDS}
    n = desc['n_sites']; p = a['position_bp']; d = a['dosage']; o = a['observed']
    require(d.shape == o.shape == (n_people, n) and d.dtype == np.uint8 and o.dtype == np.bool_ and
            np.array_equal(d == 255, ~o) and np.all(d[o] <= 2), 'Rare dose/mask/sentinel differs')
    require(p.shape == (n,) and p.dtype.kind in 'iu' and
            int(p[0]) == desc['start_bp'] and int(p[-1]) == desc['end_bp'] and
            np.all(p[1:] > p[:-1]), 'Rare chunk locus order or extent differs')
    for key in continuous.EVENT_FIELDS-{'dosage', 'observed'}:
        require(a[key].shape == (n,), 'Rare site metadata shape differs')
    require(a['ref'].dtype.kind in 'iu' and a['alt'].dtype.kind in 'iu' and
            np.all(a['ref'] <= 3) and np.all(a['alt'] <= 3) and np.all(a['ref'] != a['alt']) and
            a['counted_allele'].dtype.kind in 'iu' and np.all(a['counted_allele'] <= 1),
            'Rare source alleles differ')
    ac, an = a['allele_count'], a['allele_number']
    require(ac.dtype.kind in 'iu' and an.dtype.kind in 'iu' and np.all(ac >= 2) and
            np.all(an > 0) and np.all(an <= 2*source_n_people) and np.all(ac <= an) and
            np.all(100*ac.astype(np.int64) <= an) and
            np.allclose(a['frequency'], ac/an, rtol=0, atol=1e-12), 'Fixed source rare frequency differs')
    require(np.all(np.isfinite(a['position_cm'])), 'Mapped rare event has missing genetic coordinate')
    return a


def iter_archive(path, manifest, *, n_people, source_n_people, max_chunk_bytes,
                 max_array_bytes, deadline):
    """Single bounded tar stream; authenticate each NPZ without extracting trees."""
    chrom = manifest['chromosome']; prefix = f'published/chr{chrom}/'
    entries = manifest['channels']['rare']
    lookup = {prefix+e['path']: e for e in entries}
    require(len(lookup) == len(entries) and entries, 'Rare archive catalogue empty or duplicated')
    previous = 0
    for e in entries:
        require(set(e) == {'path', 'sha256', 'n_sites', 'bytes', 'start_bp', 'end_bp'} and
                Path(e['path']).name == e['path'] and e['path'].startswith('rare.') and
                previous < e['start_bp'] <= e['end_bp'] and e['n_sites'] > 0 and
                e['bytes'] <= max_chunk_bytes and manifest['outputs_sha256'][e['path']] == e['sha256'],
                'Rare catalogue identity/order/byte bound differs')
        previous = e['end_bp']
    seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            require(time.monotonic() < deadline, 'Rare archive original deadline exhausted')
            if member.name not in lookup:
                continue
            require(member.isfile() and member.name not in seen,
                    'Rare archive member duplicate or nonregular')
            desc = lookup[member.name]
            require(member.size == desc['bytes'], 'Rare archive member size differs')
            with archive.extractfile(member) as stream:
                data = stream.read(max_chunk_bytes+1)
            yield load_event(data, desc, n_people, source_n_people, max_array_bytes=max_array_bytes)
            seen.add(member.name)
    require(seen == set(lookup), 'Rare archive lacks required catalogue chunks')


def tail_intervals(event, length_bp):
    """Outside event extent only; an exact count and full-M02 match are required."""
    entries = event['channels']['rare']
    require(entries and event['audit']['rare']['source_rows'] ==
            event['audit']['rare']['retained_sites']+event['audit']['rare'].get('outside_map', 0),
            'Rare catalogue has exclusions beyond the recorded unmapped sites')
    return [[a, b] for a, b in ((1, entries[0]['start_bp']-1),
                               (entries[-1]['end_bp']+1, length_bp)) if a <= b]


def tail_rows(rows, *, source, n_people, expected_sites, batch_sites):
    """Decode original-source rare GT/RD without refitting the minor allele."""
    pos = []; doses = []; masks = []; previous = 0; total = 0
    for row in rows:
        require(len(row) == 9+2*n_people, 'Rare tail GT/RD column count differs')
        p, eligible = material.basic_site(row, source, previous)
        require(p > previous and eligible, 'Rare tail not sorted PASS originally biallelic SNV')
        previous = p
        allele, ac, an = map(int, row[6:9])
        require(allele in (0, 1) and ac >= 2 and an > 0 and 100*ac <= an,
                'Rare tail source-minor frequency differs')
        gt = material.decode_gt(row[9::2]); rd = np.asarray(row[10::2]); mask = np.isfinite(gt)
        require(np.array_equal(rd == '.', ~mask), 'Rare tail RD/GT missingness differs')
        dose = np.full(n_people, 255, np.uint8)
        observed = rd[mask].astype(np.int64)
        require(np.array_equal(observed, (gt if allele else 2-gt)[mask]),
                'Rare tail RD differs from source allele and GT')
        dose[mask] = observed
        pos.append(p); doses.append(dose); masks.append(mask); total += 1
        require(total <= expected_sites, 'Rare tails exceed original unmapped catalogue count')
        if len(pos) == batch_sites:
            yield np.asarray(pos, np.int64), np.column_stack(doses), np.column_stack(masks)
            pos = []; doses = []; masks = []
    require(total == expected_sites, 'Rare tails do not exactly recover original unmapped catalogue count')
    if pos:
        yield np.asarray(pos, np.int64), np.column_stack(doses), np.column_stack(masks)


def assert_exact_counts(actual, expected):
    keys = {'I', 'U', 'Q', 'individual_carrier', 'individual_callable', 'n_sites'}
    require(keys <= set(actual) and keys <= set(expected), 'Missing rare denominator for equivalence')
    for key in sorted(keys):
        require(np.array_equal(actual[key], expected[key]),
                'Event plus tail counts differ from original M02: '+key)


def run(bundle, request, digest, output):
    """Authenticated chromosome task, producing reusable interval count inputs."""
    started = time.monotonic(); pins = {}
    def pin(desc, limit):
        require(set(desc) == {'path', 'sha256'}, 'Exact local file pin required')
        path = continuous.pinned(Path(desc['path']).resolve(), desc['sha256'], limit)
        pins[str(path)] = desc['sha256']; return path
    request = pin(dict(path=str(request), sha256=digest), 2**20)
    r = material.load_json(request)
    require(set(r) == {'schema', 'evidence_scope', 'source_contract_sha256', 'implementation_sha256',
                       'chromosome', 'event_manifest', 'event_archive', 'event_producer_code_sha256',
                       'tails', 'm02', 'partials', 'resources'} and r['schema'] == SCHEMA and
            r['implementation_sha256'] == implementation(), 'Rare request schema/current implementation differs')
    limits = r['resources']
    require(set(limits) == {'max_metadata_bytes', 'max_archive_bytes', 'max_input_bytes', 'max_chunk_bytes',
                           'max_array_bytes', 'max_memory_bytes', 'batch_sites', 'max_sparse_cells',
                           'max_tail_batch_sites', 'max_output_bytes', 'min_free_bytes', 'timeout_seconds'} and
            all(type(v) is int and v > 0 for v in limits.values()), 'Positive explicit rare resource bounds required')
    bundle = Path(bundle).resolve()
    pin(dict(path=str(bundle/'materialize.json'), sha256=r['source_contract_sha256']), limits['max_metadata_bytes'])
    source, persons, roles, blocks, regions = material.load_context(bundle)
    for name, h in source['metadata_sha256'].items():
        pin(dict(path=str(bundle/name), sha256=h), limits['max_metadata_bytes'])
    require(source['roles_state'] == 'EXPLICITLY_FROZEN' and source['source_rare_rule'] ==
            dict(min_mac=2, max_maf='0.01', ascertainment='FIXED_SOURCE_COHORT_NOT_FIT_RESELECTED'),
            'Frozen roles and original source rare ascertainment required')
    require(r['evidence_scope'] in {'DNABR_R02', 'SYNTHETIC_INSTRUMENT_ONLY'}, 'Explicit rare evidence scope required')
    if r['evidence_scope'] == 'DNABR_R02':
        require(len(persons) == 2619 and {k: len(v) for k, v in roles.items()} ==
                {'FIT': 1460, 'SELECT': 476, 'EVALUATE': 683} and len(blocks) == 44,
                'DNABR cohort/role/target dimensions differ')
    chrom = r['chromosome']; src = source['sources'][chrom]
    tasks, _ = material.make_tasks(source, regions)
    tasks = [v for v in tasks.values() if v['kind'] == 'features' and v['chromosome'] == chrom]
    tasks.sort(key=lambda t: (t['intervals'] != [[1, src['length_bp']]], t['task_id']))
    counter = IntervalCounters(len(persons), tasks, chromosome=chrom, length_bp=src['length_bp'],
        batch_sites=limits['batch_sites'], max_sparse_cells=limits['max_sparse_cells'],
        max_memory_bytes=limits['max_memory_bytes'])
    ids = [p['sample_id'] for p in persons]; ids_hash = evidence.sample_hash(ids)
    event_path = pin(r['event_manifest'], limits['max_metadata_bytes']); event = material.load_json(event_path)
    require(event['schema'] == 'r02_genomic_event_chromosome_v1' and
            event['stage'] == 'COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING' and event['chromosome'] == chrom and
            event['source_contract_sha256'] == r['source_contract_sha256'] and event['sample_ids_sha256'] == ids_hash and
            event['n_people'] == len(persons) and event['code_sha256'] == r['event_producer_code_sha256'] and
            event['audit']['rare']['retained_sites'] == sum(e['n_sites'] for e in event['channels']['rare']),
            'Rare event source/people/count/producer differs')
    archive_path = pin(r['event_archive'], limits['max_archive_bytes'])
    tail_ranges = tail_intervals(event, src['length_bp']); tails = r['tails']
    require(set(tails) == {'request', 'manifest', 'vcf'}, 'Explicit original-source tail slice required')
    tr_path = pin(tails['request'], limits['max_metadata_bytes']); tr = material.load_json(tr_path)
    tm = material.load_json(pin(tails['manifest'], limits['max_metadata_bytes']))
    tv = pin(tails['vcf'], limits['max_input_bytes'])
    require(tr['schema'] == tm['schema'] == 'r02_structure_indexed_source_slice_v1' and
            tm['stage'] == 'INDEXED_SOURCE_SLICE_NOT_FIT' and tm['request_sha256'] == material.sha(tr_path) and
            tm['source'] == tr['source'] and tm['implementation_sha256'] == tr['implementation_sha256'] and
            tm['source']['sha256'] == src['rare_vcf']['sha256'] and tm['output_sha256'] == material.sha(tv),
            'Tail slice original-source identity differs')
    ranges_file = pin(tr['regions'], limits['max_metadata_bytes'])
    actual_ranges = [row.split('\t') for row in ranges_file.read_text().splitlines()]
    require(actual_ranges == [[src['contig'], str(a), str(b)] for a, b in tail_ranges],
            'Tail slice regions differ from complete unmapped support')
    sample_file = pin(tr['samples'], limits['max_metadata_bytes'])
    require(sample_file.read_text().splitlines() == ids, 'Tail slice sample roster/order differs')
    tool = material.checked_tool('bcftools', source['tools']['bcftools'])
    require(subprocess.check_output([tool, 'query', '-l', str(tv)], text=True).splitlines() == ids,
            'Tail VCF sample order differs')
    header = subprocess.check_output([tool, 'view', '-h', str(tv)], text=True)
    require('##dnabr_original_alleles=v1\n' in header and '##dnabr_rare_contract=minor_v1\n' in header and
            f"##dnabr_rare_cohort_n_samples={source['source_n_samples']}\n" in header and
            f"##dnabr_rare_cohort_sha256={source['source_samples_sha256']}\n" in header,
            'Tail source cohort/original-allele header differs')
    hist = r['m02']; require(set(hist) == {'manifest', 'npz', 'producer_sha256'}, 'Exact historical rare count pins required')
    hm = material.load_json(pin(hist['manifest'], limits['max_metadata_bytes']))
    hp = pin(hist['npz'], limits['max_input_bytes'])
    require(hm['schema'] == evidence.SCHEMA and hm['kind'] == 'rare' and hm['chromosomes'] == [chrom] and
            hm['samples_sha256'] == ids_hash and hm['n_samples'] == len(ids) and
            hm['source_cohort_sha256'] == source['source_samples_sha256'] and
            hm['source_cohort_n'] == source['source_n_samples'] and
            hm['input_vcf_sha256'] == src['rare_vcf']['sha256'] and
            hm['code_sha256'] == hist['producer_sha256'] and hm['output_npz_sha256'] == material.sha(hp) and
            (hm['mask'], hm['allele'], hm['method']) == (reuse.MASK, reuse.ALLELE, reuse.METHOD),
            'Original M02 count identity/semantics differ')
    with zipfile.ZipFile(hp) as z:
        require(sum(m.file_size for m in z.infolist()) <= limits['max_array_bytes'], 'Historical rare array bound exceeded')
    partials = {}
    for desc in r['partials']:
        require(set(desc) == {'directory', 'manifest_sha256', 'producer_code_sha256'}, 'Original K02 partial pins required')
        directory = Path(desc['directory']).resolve()
        pm = material.load_json(pin(dict(path=str(directory/'manifest.json'), sha256=desc['manifest_sha256']), limits['max_metadata_bytes']))
        task = pm['request']; key = task['task_id']
        require(key not in partials and task in tasks and pm['schema'] == material.PARTIAL and
                pm['stage'] == 'COMPLETE_SLICE' and pm['contract_sha256'] == r['source_contract_sha256'] and
                pm['code_sha256'] == desc['producer_code_sha256'] and pm['sample_ids_sha256'] == ids_hash,
                'K02 original rare partial source/geometry differs')
        pp = pin(dict(path=str(directory/'data.npz'), sha256=pm['outputs_sha256']['data.npz']), limits['max_input_bytes'])
        with zipfile.ZipFile(pp) as z:
            require(sum(m.file_size for m in z.infolist()) <= limits['max_array_bytes'], 'K02 partial array bound exceeded')
        partials[key] = pp
    require(set(partials) == set(counter.tasks), 'Every original K02 rare partial required for equivalence')
    require(sum(Path(p).stat().st_size for p in pins) <= limits['max_input_bytes'], 'Rare total input byte bound exceeded')
    output = material.new_directory(output); fs = material.local_intermediate_filesystem(output)
    deadline = started+limits['timeout_seconds']
    def guard():
        require(time.monotonic() < deadline and shutil.disk_usage(output).free >= limits['min_free_bytes'],
                'Rare original deadline or disk reserve exhausted')
    guard()
    fmt = '%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t%INFO/ORIG_NALLELES\t%INFO/RARE_ALLELE\t%INFO/RARE_AC\t%INFO/RARE_AN[\t%GT\t%RD]\n'
    with (output/'tails.bcftools.log').open('x') as log:
        process = subprocess.Popen([tool, 'query', '-f', fmt, str(tv)], stdout=subprocess.PIPE,
                                   stderr=log, text=True, bufsize=2**20)
        try:
            rows = (line.rstrip('\n').split('\t') for line in process.stdout)
            stream = iter(tail_rows(rows, source=src, n_people=len(persons),
                expected_sites=event['audit']['rare'].get('outside_map', 0), batch_sites=limits['max_tail_batch_sites']))
            lower = event['channels']['rare'][0]['start_bp']; upper = event['channels']['rare'][-1]['end_bp']
            pending = None
            for p, d, o in stream:
                guard(); require(np.all((p < lower) | (p > upper)), 'Tail VCF overlaps mapped catalogue')
                low = p < lower
                if np.any(low): counter.add(p[low], d[:, low], o[:, low])
                if np.any(~low):
                    pending = p[~low], d[:, ~low], o[:, ~low]; break
            for a in iter_archive(archive_path, event, n_people=len(persons), source_n_people=source['source_n_samples'],
                                  max_chunk_bytes=limits['max_chunk_bytes'], max_array_bytes=limits['max_array_bytes'], deadline=deadline):
                guard(); counter.add(a['position_bp'], a['dosage'], a['observed'])
            if pending is not None: counter.add(*pending)
            for p, d, o in stream:
                guard(); require(np.all(p > upper), 'Tail VCF overlaps mapped catalogue or changes order')
                counter.add(p, d, o)
            require(process.wait(timeout=max(1, deadline-time.monotonic())) == 0, 'Rare tail bcftools query failed')
        finally:
            process.stdout.close()
            if process.poll() is None: process.kill(); process.wait()
    result = dict(schema=SCHEMA, stage='EXACT_INTERVAL_COUNTS_NOT_K03_FIT', request_sha256=digest,
        implementation_sha256=implementation(), source_contract_sha256=r['source_contract_sha256'],
        chromosome=chrom, persons_sha256=ids_hash, n_people=len(ids), source_rare_rule=source['source_rare_rule'],
        source_sites=event['audit']['rare']['source_rows'], recovered_unmapped_sites=event['audit']['rare'].get('outside_map', 0),
        source_files_sha256=pins, local_intermediate_filesystem=fs, memory_bound_bytes=counter.memory_bound,
        tasks={}, training_allowed=False, outcomes_read=False, roles_changed=False)
    for key, arrays, flushes in counter.finish():
        guard()
        with np.load(partials[key], allow_pickle=False) as old:
            require(old['sample_ids'].tolist() == ids and int(old['rare_sites']) == int(arrays['n_sites']),
                    'K02 rare partial sample/count differs')
            pair = old['pairs']; require(pair.ndim == 2 and pair.shape[1] == 2 and pair.dtype.kind in 'iu', 'K02 pair index invalid')
            for name in ('I', 'U', 'Q'):
                require(np.array_equal(arrays[name][pair[:, 0], pair[:, 1]], old['rare_'+name]),
                        'Reconstructed rare counts differ from original K02: '+name)
            for name, old_name in [('individual_carrier', 'rare_carriers'), ('individual_callable', 'rare_called')]:
                require(np.array_equal(arrays[name], old[old_name]), 'K02 rare individual denominator differs')
        if key == counter.full_id:
            with np.load(hp, allow_pickle=False) as old:
                require(old['samples'].tolist() == ids and str(old['sample_hash']) == ids_hash and
                        old['chromosomes'].tolist() == [chrom] and str(old['kind']) == 'rare', 'M02 array identity differs')
                assert_exact_counts(arrays, {name: old[name] for name in arrays})
            record = dict(reused_original_m02=hist, candidate_sites=int(arrays['n_sites']), sparse_flushes=flushes)
        else:
            filename = key+'.npz'
            np.savez_compressed(output/filename, samples=np.asarray(ids), sample_hash=np.asarray(ids_hash),
                chromosomes=np.asarray([chrom]), kind=np.asarray('rare'), schema=np.asarray(evidence.SCHEMA), **arrays)
            record = dict(path=filename, sha256=material.sha(output/filename), bytes=(output/filename).stat().st_size,
                          candidate_sites=int(arrays['n_sites']), sparse_flushes=flushes)
        result['tasks'][key] = dict(request=counter.tasks[key], **record)
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= limits['max_output_bytes'],
                'Rare output byte bound exceeded')
    require(all(material.sha(p) == h for p, h in pins.items()) and implementation() == r['implementation_sha256'],
            'Rare inputs or implementation changed during computation')
    result['seconds'] = time.monotonic()-started
    material.write_json(output/'manifest.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path); p.add_argument('--request', required=True, type=Path)
    p.add_argument('--request-sha256', required=True); p.add_argument('--output-dir', required=True, type=Path)
    a = p.parse_args(); os.umask(0o077)
    result = run(a.bundle, a.request, a.request_sha256, a.output_dir)
    print(json.dumps({k: result[k] for k in ('schema', 'stage', 'chromosome', 'seconds')}))


if __name__ == '__main__':
    main()
