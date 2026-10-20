"""Persist the existing full inner-FIT spectrum without choosing predictive rank."""
from pathlib import Path
import hashlib
import json
import numpy as np


def save_spectrum(graph, projections, directory, *, source_sha256, fold_plan_sha256):
    directory = Path(directory)
    if directory.exists() or not directory.parent.is_dir():
        raise ValueError('A new artifact directory is required')
    graph.verify(source_sha256)
    arrays = {name: getattr(graph, name) for name in graph._arrays}
    arrays.update(query_indices=np.asarray([r['query_index'] for r in projections], np.int64),
        query_scores=np.asarray([r['scores'] for r in projections]),
        query_shared_scores=np.asarray([r['shared_component_scores'] for r in projections]),
        query_status=np.asarray([r['status'] for r in projections]),
        query_observed_fit_count=np.asarray([r['observed_fit_count'] for r in projections], np.int64))
    directory.mkdir(mode=0o700)
    path = directory / 'full_spectrum.private.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError('Saved spectrum fields differ')
        for name, expected in arrays.items():
            actual = saved[name]
            equal = (np.array_equal(actual, expected, equal_nan=True)
                     if expected.dtype.kind in 'fc' else np.array_equal(actual, expected))
            if actual.dtype != expected.dtype or not equal:
                raise ValueError('Saved spectrum differs: ' + name)
    graph.verify(source_sha256)
    result = dict(schema='r02_k03_full_inner_spectrum_artifacts_v1',
        file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
        graph_sha256=graph.graph_sha256, source_sha256=source_sha256,
        fold_plan_sha256=fold_plan_sha256, affinity_semantics=graph.affinity_semantics,
        numerical_tolerance=graph.numerical_tolerance, full_spectrum_preserved=True,
        predictive_rank_selected=False, communities_learned=False,
        genetic_response_read=False, biological_validation=False)
    (directory/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
