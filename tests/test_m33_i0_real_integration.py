#!/usr/bin/env python3
"""Synthetic no-mock smoke for the real I0 final VCF/TBI reopen gate."""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REAL = load_module("m33_i0_real_integration", ROOT / "bin" / "m33_i0_real.py")
HELPER = load_module("m33_i0_index_integration", ROOT / "bin" / "m33_i0_index.py")


@unittest.skipUnless(shutil.which("bgzip") and shutil.which("tabix"), "requires htslib runtime")
class M33I0RealIntegrationTests(unittest.TestCase):
    def test_reopens_emitted_index_and_matches_sequential_oracle_without_mock(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "root17.flare.anc.vcf.gz"
            completed = subprocess.run(
                ["bgzip", "--threads", "1", "--stdout"],
                input=HELPER.VCF_TEXT.encode(), check=True, capture_output=True,
            )
            source.write_bytes(completed.stdout)
            subprocess.run(["tabix", "-p", "vcf", source.name], cwd=base, check=True)
            index = Path(f"{source}.tbi")
            source.chmod(0o444)
            index.chmod(0o444)
            count, digest = HELPER.sequential_chr22(source)
            original_count = REAL.EXPECTED_RECORD_COUNT
            try:
                REAL.EXPECTED_RECORD_COUNT = count
                REAL.verify_existing_index(
                    source=source,
                    index=index,
                    descriptor={"size_bytes": source.stat().st_size, "sha256": REAL.sha256_file(source)},
                    expected_query_sha=digest,
                    helpers=HELPER,
                )
                with self.assertRaisesRegex(ValueError, "record digests differ"):
                    REAL.verify_existing_index(
                        source=source,
                        index=index,
                        descriptor={"size_bytes": source.stat().st_size, "sha256": REAL.sha256_file(source)},
                        expected_query_sha="0" * 64,
                        helpers=HELPER,
                    )
            finally:
                REAL.EXPECTED_RECORD_COUNT = original_count


if __name__ == "__main__":
    unittest.main()
