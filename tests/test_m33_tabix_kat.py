#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "m33_tabix_kat.py"
SPEC = importlib.util.spec_from_file_location("m33_tabix_kat", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def build_receipt(replica: str, task_hash: str, work_dir: str) -> dict:
    return {
        "stage": "M33_TABIX_SYNTHETIC_KAT_BUILD",
        "status": "PASS",
        "replica": replica,
        "task_hash": task_hash,
        "task_work_dir": work_dir,
        "tabix_version": MODULE.EXPECTED_VERSION,
        "source_vcf_sha256": MODULE.EXPECTED_SOURCE_VCF_SHA256,
        "tbi_sha256": MODULE.EXPECTED_TBI_SHA256,
        "indexed_record_count": 4,
        "indexed_record_sha256": MODULE.EXPECTED_RECORD_SHA256,
        "sequential_record_count": 4,
        "sequential_record_sha256": MODULE.EXPECTED_RECORD_SHA256,
        "runtime_service_account": "LOCAL_KAT_NOT_CLOUD_AUTHENTICATED",
        "cloud_context_authenticated": False,
        "contains_real_genomic_data": False,
    }


class TabixKatTests(unittest.TestCase):
    def compare(self, a: dict, b: dict):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path_a, path_b = root / "a.json", root / "b.json"
            vcf_a, vcf_b = root / "a.vcf.gz", root / "b.vcf.gz"
            tbi_a, tbi_b = root / "a.vcf.gz.tbi", root / "b.vcf.gz.tbi"
            vcf_bytes = b"fixture"
            tbi_bytes = b"index"
            vcf_a.write_bytes(vcf_bytes)
            vcf_b.write_bytes(vcf_bytes)
            tbi_a.write_bytes(tbi_bytes)
            tbi_b.write_bytes(tbi_bytes)
            path_a.write_text(json.dumps(a), encoding="utf-8")
            path_b.write_text(json.dumps(b), encoding="utf-8")
            output, ready = root / "receipt.json", root / "READY"
            def known_hash(path):
                return (
                    MODULE.EXPECTED_TBI_SHA256
                    if str(path).endswith(".tbi")
                    else MODULE.EXPECTED_SOURCE_VCF_SHA256
                )
            with mock.patch.object(MODULE, "sha256_file", side_effect=known_hash):
                result = MODULE.compare(
                    path_a, path_b, vcf_a, tbi_a, vcf_b, tbi_b, output, ready
                )
            self.assertTrue(output.is_file() and ready.is_file())
            return result

    def test_independent_matching_replicas_pass(self):
        result = self.compare(
            build_receipt("A", "hash-a", "/work/a"),
            build_receipt("B", "hash-b", "/work/b"),
        )
        self.assertEqual(result["status"], "PASS_INDEPENDENT_INDEX_AND_QUERY_PARITY")
        self.assertFalse(result["contains_real_genomic_data"])
        self.assertEqual(
            [row["task_hash"] for row in result["build_receipts"]],
            ["hash-a", "hash-b"],
        )

    def test_shared_task_hash_or_workdir_fails(self):
        cases = [
            (build_receipt("A", "same", "/work/a"), build_receipt("B", "same", "/work/b")),
            (build_receipt("A", "a", "/work/same"), build_receipt("B", "b", "/work/same")),
        ]
        for a, b in cases:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                self.compare(a, b)

    def test_mismatch_and_real_data_flags_fail(self):
        mutations = []
        b = build_receipt("B", "hash-b", "/work/b")
        b["tbi_sha256"] = "d" * 64
        mutations.append(b)
        b = build_receipt("B", "hash-b", "/work/b")
        b["source_vcf_sha256"] = "d" * 64
        mutations.append(b)
        b = build_receipt("B", "hash-b", "/work/b")
        b["contains_real_genomic_data"] = True
        mutations.append(b)
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.compare(build_receipt("A", "hash-a", "/work/a"), changed)

    def test_version_is_exact(self):
        with mock.patch.object(MODULE.subprocess, "run") as run:
            run.return_value.stdout = "tabix (htslib) 1.20\n"
            with self.assertRaisesRegex(ValueError, "version drifted"):
                MODULE.tabix_version()

    def test_cloud_context_requires_exact_prefix_and_runtime_identity(self):
        prefix = "gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/run-1/"
        with mock.patch.object(MODULE, "runtime_service_account", return_value=MODULE.EXPECTED_RUNTIME_SERVICE_ACCOUNT):
            observed = MODULE.validate_cloud_context(
                prefix + "aa/task",
                prefix,
                MODULE.EXPECTED_RUNTIME_SERVICE_ACCOUNT,
            )
        self.assertEqual(observed, MODULE.EXPECTED_RUNTIME_SERVICE_ACCOUNT)
        with self.assertRaisesRegex(ValueError, "escapes"):
            MODULE.validate_cloud_context(
                "gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/other/task",
                prefix,
                MODULE.EXPECTED_RUNTIME_SERVICE_ACCOUNT,
            )

    def test_cloud_compare_rejects_local_receipts(self):
        with self.assertRaisesRegex(ValueError, "cloud-authenticated"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                a = build_receipt("A", "a", "/work/a")
                b = build_receipt("B", "b", "/work/b")
                paths = [root / name for name in ("a.json", "b.json", "a.vcf", "a.tbi", "b.vcf", "b.tbi")]
                paths[0].write_text(json.dumps(a), encoding="utf-8")
                paths[1].write_text(json.dumps(b), encoding="utf-8")
                for path in paths[2:]:
                    path.write_bytes(b"x")
                def known_hash(path):
                    return MODULE.EXPECTED_TBI_SHA256 if path.suffix == ".tbi" else MODULE.EXPECTED_SOURCE_VCF_SHA256
                with mock.patch.object(MODULE, "sha256_file", side_effect=known_hash):
                    MODULE.compare(*paths, root / "result.json", root / "READY", require_cloud=True)


if __name__ == "__main__":
    unittest.main()
