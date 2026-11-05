"""Extract previously unavailable joint counts, reusing authenticated FIT truth.

Source FN summaries remain unchanged. Atomic per-case products permit recovery;
no neural input or genotype dosage is read, and no model is fitted here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from r02_lai_d1_lumped import full_label_counts, combine_counts, CLASSES
from r02_lai_duration_fit_summary import load_authenticated_truth_case, _write_fresh_atomic, _canonical
from r02_lai_duration_input import load_duration_case_inventory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(request_path, output_dir):
    request = json.loads(Path(request_path).read_text())
    if request['scope'] != 'FIT_TRUTH_COUNTS_ONLY_NO_NEURAL_INPUT':
        raise ValueError('scope differs')
    if len(set(request['case_ids'])) != len(request['case_ids']):
        raise ValueError('duplicate cases')
    if sha(request['fn_summary']['path']) != request['fn_summary']['sha256']:
        raise ValueError('FN receipt changed')
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    started, cases, receipts = time.monotonic(), [], []
    for number, case_id in enumerate(request['case_ids']):
        path = output / f'case-{number}.json'
        identity = dict(case_id=case_id, contract=request['contract'],
                        module_sha256=sha(Path(__file__).with_name('r02_lai_d1_lumped.py')),
                        reader_sha256=sha(__file__), request_sha256=sha(request_path))
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt['identity'] != identity:
                raise ValueError('incompatible saved count product')
            if receipt['counts_sha256'] != hashlib.sha256(json.dumps(
                    receipt['counts'], sort_keys=True).encode()).hexdigest():
                raise ValueError('saved counts checksum differs')
            inventory=load_duration_case_inventory(request['contract']['path'],
                request['contract']['sha256'],case_id)
            if inventory.inventory_sha256 != receipt['inventory_sha256']:
                raise ValueError('source inventory changed on recovery')
        else:
            truth = load_authenticated_truth_case(request['contract']['path'],
                request['contract']['sha256'], case_id,
                max_region_logical_bytes=request['max_region_logical_bytes'])
            counts = full_label_counts(truth.query_positions_cm, truth.truth_state, truth.truth_observed)
            receipt = dict(identity=identity, counts=counts,
                counts_sha256=hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest(),
                inventory_sha256=truth.inventory_sha256,
                source_receipts=truth.source_region_receipts)
            _write_fresh_atomic(path, _canonical(receipt))
            del truth
        if receipt['counts']['people'] != request['expected']['per_case_rows']:
            raise ValueError('unexpected per-case mosaic count')
        cases.append(receipt['counts'])
        receipts.append(dict(path=str(path.resolve()), sha256=sha(path)))
    combined = combine_counts(cases)
    expected = request['expected']
    for key in ('intervals', 'changes', 'initial_counts', 'missing_labels', 'people'):
        if combined[key] != expected[key]:
            raise ValueError(f'FN reconciliation failed: {key}')
    matrix = np.asarray(combined['counts'], dtype=np.int64).sum(0)
    if matrix.tolist() != expected['endpoint_matrix']:
        raise ValueError('FN joint endpoint matrix differs')
    final = dict(schema='r02_d1_joint_fit_counts_v1',
        status='VERIFIED_COUNTS_NOT_BIOLOGICAL_IMPROVEMENT',
        classes=CLASSES, data=combined, cases=receipts,
        elapsed_seconds=time.monotonic()-started, training_performed=False,
        shared_donors=True, independent_sample_likelihood=False,
        request_sha256=sha(request_path), fn_summary=request['fn_summary'])
    _write_fresh_atomic(output / 'COUNTS.json', _canonical(final))
    return final


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    result = run(args.request, args.output_dir)
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds', 'training_performed')}))
