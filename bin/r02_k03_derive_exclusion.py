#!/usr/bin/env python3
"""Derive frozen target exclusions from authenticated, complete FIT chromosomes.

Original producer hashes describe the source representation. The additional
derivation record identifies this transformation; it is never passed off as a
new execution of the original genomic decoder.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil

import numpy as np
import r02_k03_event_alt_chunks as raw


def authenticated_record(directory, identity, fit_ids):
    directory = Path(directory).resolve()
    record = raw.load_json(directory / 'chromosome.json')
    raw.require(record.get('schema') == raw.CHROMOSOME_SCHEMA and
                record.get('input_identity') == identity and
                record.get('implementation_sha256') == raw.implementation_sha256() and
                record.get('sample_ids_sha256') == raw._ordered_ids_sha256(fit_ids) and
                record.get('outer_fit_people') == len(fit_ids) and
                record.get('biological_validation') is False,
                'Source chromosome identity, representation or people differ')
    count = total_bytes = previous_end = 0
    for chunk in record['chunks']:
        path = (directory / chunk['path']).resolve()
        raw.require(path.is_relative_to(directory) and path.is_file() and
                    path.stat().st_size == chunk['bytes'] and raw.sha256(path) == chunk['sha256'],
                    'Source chunk path, bytes or hash differs')
        raw.require(chunk['n_sites'] > 0 and previous_end < chunk['start_bp'] <= chunk['end_bp'],
                    'Source chunk ranges overlap or are unordered')
        count += chunk['n_sites']; total_bytes += chunk['bytes']; previous_end = chunk['end_bp']
    raw.require(count == record['included_sites'] and
                count + record['excluded_target_sites'] == record['source_sites'] and
                total_bytes == record['compressed_chunk_bytes'], 'Source chromosome counts differ')
    return record


def derive(base, output, target_identity, fit_ids, *, max_output_bytes, min_free_bytes):
    base, output = Path(base).resolve(), Path(output)
    expected = copy.deepcopy(target_identity); expected['excluded_intervals'] = []
    record = authenticated_record(base, expected, fit_ids)
    raw.require(record['excluded_target_sites'] == 0 and 'derivation' not in record,
                'Derivation requires an original complete, unexcluded chromosome')
    chrom = record['chromosome']; intervals = target_identity['excluded_intervals']
    raw.require(len(intervals) <= 1 and all(i['chromosome'] == chrom and
                1 <= i['start_bp'] <= i['end_bp'] for i in intervals), 'Invalid target interval')
    raw.require(not output.exists() and max_output_bytes > 0 and min_free_bytes > 0,
                'New output directory and positive resource bounds required')
    output.mkdir(parents=True); (output / 'chunks').mkdir()
    retained = excluded = size = linked = sliced = 0
    chunks = []
    for item in record['chunks']:
        start, end = (intervals[0]['start_bp'], intervals[0]['end_bp']) if intervals else (0, 0)
        if intervals and start <= item['start_bp'] and item['end_bp'] <= end:
            excluded += item['n_sites']; continue
        source = base / item['path']; relative = f'chunks/{len(chunks):06d}.npz'
        dest = output / relative
        raw.require(shutil.disk_usage(output).free >= min_free_bytes + item['bytes'],
                    'Derivation disk reserve exhausted')
        if not intervals or item['end_bp'] < start or item['start_bp'] > end:
            try:
                os.link(source, dest)
            except OSError:
                shutil.copyfile(source, dest)
            descriptor = dict(item, path=relative); linked += 1
        else:
            with np.load(source, allow_pickle=False) as z:
                required = {'sample_ids', 'variant_id', 'chromosome', 'position_bp',
                            'ref', 'alt', 'genotype_code'}
                raw.require(set(z.files) == required, 'Boundary array schema differs')
                positions = z['position_bp']; n = len(positions)
                raw.require(n == item['n_sites'] and n > 0 and
                            int(positions[0]) == item['start_bp'] and int(positions[-1]) == item['end_bp']
                            and np.all(positions[1:] > positions[:-1]) and
                            np.array_equal(z['sample_ids'], np.asarray(fit_ids)) and
                            z['genotype_code'].shape == (len(fit_ids), n) and
                            np.all(z['chromosome'] == chrom), 'Boundary coordinates or people differ')
                keep = (positions < start) | (positions > end)
                excluded += int((~keep).sum())
                if not keep.any():
                    continue
                arrays = {k: z[k].copy() if k == 'sample_ids' else
                          z[k][:, keep] if k == 'genotype_code' else z[k][keep] for k in required}
            temporary = dest.with_suffix('.partial.npz')
            np.savez_compressed(temporary, **arrays)
            with np.load(temporary, allow_pickle=False) as z:
                raw.require(all(z[k].dtype == a.dtype and np.array_equal(z[k], a)
                                for k, a in arrays.items()), 'Derived array roundtrip differs')
            os.replace(temporary, dest); sliced += 1
            p = arrays['position_bp']
            descriptor = dict(path=relative, sha256=raw.sha256(dest), bytes=dest.stat().st_size,
                              n_sites=len(p), start_bp=int(p[0]), end_bp=int(p[-1]))
        raw.require(raw.sha256(dest) == descriptor['sha256'], 'Copied derived chunk changed')
        chunks.append(descriptor); retained += descriptor['n_sites']; size += descriptor['bytes']
        raw.require(size <= max_output_bytes, 'Derived output exceeds byte bound')
    raw.require(retained + excluded == record['source_sites'], 'Derived counts do not reconcile')
    result = dict(record, input_identity=copy.deepcopy(target_identity), chunks=chunks,
                  included_sites=retained, excluded_target_sites=excluded, compressed_chunk_bytes=size)
    result['derivation'] = dict(method='EXACT_INCLUSIVE_COORDINATE_EXCLUSION_NO_GENOTYPE_TRANSFORM',
        code_sha256=raw.sha256(Path(__file__)), source_manifest=str(base / 'chromosome.json'),
        source_manifest_sha256=raw.sha256(base / 'chromosome.json'),
        source_producer_implementation_sha256=record['implementation_sha256'],
        complete_chunks_reused=linked, boundary_chunks_filtered=sliced,
        original_decoder_reexecuted=False, scientific_models_admitted=False)
    temp = output / 'chromosome.partial.json'; raw.write_json(temp, result)
    os.replace(temp, output / 'chromosome.json')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True); p.add_argument('--output', required=True)
    p.add_argument('--campaign', required=True); p.add_argument('--chromosome', required=True, type=int)
    a = p.parse_args()
    campaign, _, _, _, fit, people = raw._campaign(Path(a.campaign))
    r = derive(a.base, a.output, raw.chromosome_identity(campaign, a.chromosome),
               [people[i] for i in fit], max_output_bytes=campaign['resources']['max_output_bytes'],
               min_free_bytes=campaign['resources']['min_free_bytes'])
    print(r['included_sites'], r['excluded_target_sites'])


if __name__ == '__main__':
    main()
