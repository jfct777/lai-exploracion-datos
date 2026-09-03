#!/usr/bin/env python3
"""Pytest-free runner for the isolated M37 SIGHUP recovery tests."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    path = ROOT / "test_m37_compact_recovery.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(
            getattr(module, name), description=f"{path.name}:{name}"
        )
        for name in sorted(dir(module))
        if name.startswith("test_") and callable(getattr(module, name))
    )
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not outcome.wasSuccessful())


if __name__ == "__main__":
    main()
