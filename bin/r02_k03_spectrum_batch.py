#!/usr/bin/env python3
"""Persist complete inner-FIT spectra from authenticated fixed-catalogue views.

Byte-identical inputs may share a calculation, but all biological view identities
remain in the manifest. No communities, genetic response or rank are selected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def verify_view(record, input_root):
    """Bind every alias to its own territory metadata and shared pair bytes."""
    expected = record['input_sha256']
    for alias in record['aliases']:
        folder = input_root / alias['directory']
        pins = alias['sha256']
        for name, digest in pins.items():
            if sha256(folder / name) != digest:
                raise ValueError('Changed view member: ' + str(folder / name))
        meta = json.loads((folder / 'affinity_inputs.json').read_text())
        exclusion = json.loads((folder / 'exclusion.json').read_text())
        if not (meta['source_contract_sha256'] == record['source_contract_sha256']
                and meta['block_id'] == alias['block_id']
                and meta['exclusion_scope'] == alias['exclusion_scope']
                and meta['exclusion'] == exclusion
                and meta['affinity']['source_measure'] == 'SOURCE_MINOR_JACCARD'
                and meta['affinity']['transformation'] == 'IDENTITY_NONNEGATIVE_INPUT'
                and meta['same_component_fit_edges'] == 'EXCLUDE'
                and meta['missing_edge_policy'] == 'UNKNOWN_NOT_ZERO'
                and all(meta['sha256'][name] == pins[name] for name in
                        ('persons.tsv', 'fit_affinities.private.tsv', 'exclusion.json'))
                and pins['persons.tsv'] == expected['persons.tsv']
                and pins['fit_affinities.private.tsv'] == expected['fit_affinities.private.tsv']):
            raise ValueError('View semantics, source or alias identity differs')
    return input_root / record['aliases'][0]['directory']


def verify_completed_fold(folder, plan, source_digest):
    summary = json.loads((folder / 'summary.json').read_text())
    if (summary['plan_sha256'] != plan['sha256'] or
            summary['result']['affinity_sha256'] != source_digest or
            not summary['full_spectrum_preserved'] or summary['biological_validation']):
        raise ValueError('Completed fold has incompatible provenance')
    artifact = summary['result']['reusable_spectrum_artifacts']
    manifest = json.loads((folder / 'full-spectrum' / 'manifest.json').read_text())
    if manifest != artifact:
        raise ValueError('Completed spectrum manifest changed')
    binary = folder / 'full-spectrum' / artifact['file']
    if binary.stat().st_size != artifact['bytes'] or sha256(binary) != artifact['sha256']:
        raise ValueError('Completed fold spectrum changed')
    return summary


def run_view(record, input_root, plans_root, output, memory_bytes, reserve_bytes):
    from r02_projection_calibration_split import exercise_inner_graph_path
    source = verify_view(record, input_root)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = time.monotonic()
    summaries = []
    for plan in record['plans']:
        plan_file = plans_root / plan['file']
        if sha256(plan_file) != plan['sha256']:
            raise ValueError('Frozen family split changed')
        folder = output / plan['id']
        if (folder / 'summary.json').exists():
            summaries.append(verify_completed_fold(
                folder, plan, record['input_sha256']['fit_affinities.private.tsv']))
            continue
        if shutil.disk_usage(output).free < reserve_bytes:
            raise ValueError('Insufficient disk reserve before next complete spectrum')
        if folder.exists():
            folder.rename(folder.with_name(folder.name + '.incomplete-' + str(time.time_ns())))
        folder.mkdir(mode=0o700)
        tick = time.monotonic()
        result = exercise_inner_graph_path(source / 'persons.tsv', plan_file,
            source / 'fit_affinities.private.tsv',
            record['input_sha256']['fit_affinities.private.tsv'], memory_bytes,
            artifact_directory=folder / 'full-spectrum')
        summary = dict(schema='r02_k03_inner_spectrum_batch_fold_v1',
            status='COMPLETE', physical_view=record['id'], plan_sha256=plan['sha256'],
            result=result, elapsed_seconds=time.monotonic()-tick,
            maximum_resident_set_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            full_spectrum_preserved=True, genetic_response_read=False,
            predictive_rank_selected=False, communities_learned=False,
            biological_validation=False)
        write_json(folder / 'summary.json', summary)
        verify_completed_fold(folder, plan, record['input_sha256']['fit_affinities.private.tsv'])
        summaries.append(summary)
    receipt = dict(schema='r02_k03_complete_physical_view_v1', status='TEN_FOLDS_COMPLETE',
        physical_view=record['id'], aliases=record['aliases'], plans=record['plans'],
        input_sha256=record['input_sha256'], completed_folds=len(summaries),
        elapsed_seconds=time.monotonic()-started, genetic_response_read=False,
        scientific_comparison_released=False)
    if len(summaries) != len(record['plans']) or len(summaries) != 10:
        raise ValueError('Incomplete frozen family split calendar')
    write_json(output / 'VIEW_RECEIPT.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--record-sha256', required=True)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--plans-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--memory-bytes', type=int, required=True)
    p.add_argument('--reserve-bytes', type=int, required=True)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if sha256(a.record) != a.record_sha256:
        raise ValueError('Execution record changed')
    record = json.loads(a.record.read_text())
    if a.verify_only:
        verify_view(record, a.input_root)
        for plan in record['plans']:
            if sha256(a.plans_root / plan['file']) != plan['sha256']:
                raise ValueError('Plan changed')
        print(json.dumps({'status':'VIEW_INPUTS_VERIFIED_NO_FIT','aliases':len(record['aliases'])}))
    else:
        print(json.dumps(run_view(record,a.input_root,a.plans_root,a.output,
                                  a.memory_bytes,a.reserve_bytes)))

if __name__ == '__main__':
    os.umask(0o077)
    main()
