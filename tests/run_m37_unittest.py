#!/usr/bin/env python3
"""Pytest-free focal runner for the fixed M33 PyTorch image."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULES = ("test_m37_trace_core.py", "test_m37_m34_adapter.py", "test_m37_positive_controls.py",
           "test_m37_provenance.py", "test_m37_metric_collection.py", "test_m37_trace_workflow.py",
           "test_m37_compact_sweep.py", "test_m37_compact_workflow.py")


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_suite() -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for name in MODULES:
        module = load(ROOT / name)
        for attribute in sorted(dir(module)):
            if attribute.startswith("test_") and callable(getattr(module, attribute)):
                suite.addTest(unittest.FunctionTestCase(getattr(module, attribute), description=f"{name}:{attribute}"))
    return suite


if __name__ == "__main__":
    outcome = unittest.TextTestRunner(verbosity=2).run(build_suite())
    raise SystemExit(not outcome.wasSuccessful())
