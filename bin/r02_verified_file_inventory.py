"""Authenticate immutable files without treating a read's atime as a mutation."""
from pathlib import Path
from r02_publish_retained_m02 import signature
from r02_publish_evidence import pipeline


def verified_file(path, cache, expected_sha256=None):
    path = Path(path)
    before = signature(path)
    key = tuple(sorted(before.items()))
    if key not in cache:
        values = pipeline.publication_digests(path)
        if signature(path) != before or values['bytes'] != before['bytes']:
            raise ValueError('File changed while computing checksums: ' + str(path))
        cache[key] = values
    if signature(path) != before:
        raise ValueError('File identity changed during inventory: ' + str(path))
    values = cache[key]
    if expected_sha256 is not None and values['sha256'] != expected_sha256:
        raise ValueError('Content differs from producer SHA256: ' + str(path))
    return dict(**values, signature=before)
