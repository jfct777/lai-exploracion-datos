"""Bounded, per-planner-call memoization of exact immutable RB line results.

No candidate, provenance, convergence or seed-coverage check is bypassed. The
caller must retain the same authenticated graph throughout the context.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib


@contextmanager
def cached_lines(profile, *, max_entries=128):
    if type(max_entries) is not int or max_entries < 1:
        raise ValueError('Positive cache capacity required')
    original = profile.make_line
    cache = OrderedDict()
    state = {'calls': 0, 'computed': 0, 'hits': 0, 'max_entries': max_entries}
    bound = None
    arrays = []

    def fingerprint(array):
        return hashlib.sha256(array.tobytes(order='C')).hexdigest()

    def wrapped(graph, labels):
        nonlocal bound, arrays
        state['calls'] += 1
        if bound is None:
            bound = graph
            arrays = [(name, getattr(graph, name), getattr(graph, name).flags.writeable,
                       fingerprint(getattr(graph, name)))
                      for name in ('adjacency', 'degree', 'fit_indices', 'active_fit')]
            for _, array, _, _ in arrays:
                array.flags.writeable = False
        if graph is not bound or any(getattr(graph, name) is not array
                                     for name, array, _, _ in arrays):
            raise ValueError('Line cache cannot cross graph identities')
        key = tuple(sorted(labels.items()))
        if key in cache:
            state['hits'] += 1
            cache.move_to_end(key)
            return cache[key]
        result = original(graph, labels)
        cache[key] = result
        state['computed'] += 1
        if len(cache) > max_entries:
            cache.popitem(last=False)
        return result

    profile.make_line = wrapped
    try:
        yield state
    finally:
        profile.make_line = original
        changed = False
        for _, array, writeable, digest in arrays:
            changed |= fingerprint(array) != digest
            array.flags.writeable = writeable
        cache.clear()
        if changed:
            raise ValueError('Authenticated graph mutated during cached planning')
