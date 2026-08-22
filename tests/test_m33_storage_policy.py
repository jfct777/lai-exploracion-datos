#!/usr/bin/env python3

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "m33_storage_policy.py"
POLICY = ROOT / "conf" / "m33_storage_namespace_policy.json"
SPEC = importlib.util.spec_from_file_location("m33_storage_policy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StoragePolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = MODULE.load_policy(POLICY)

    @staticmethod
    def descriptor(uri="gs://projects-usp/dnaBr-lai/datalake/source.vcf.gz"):
        return {
            "logical_id": "source",
            "gcs_uri": uri,
            "gcs_generation": "123456789",
            "size_bytes": 1024,
            "sha256_raw": "a" * 64,
            "crc32c": "AAAAAA==",
        }

    def test_policy_is_contract_only(self):
        self.assertFalse(any(
            self.policy["execution_authorization"][key]
            for key in ("create_project_prefix", "real_asset_read", "derived_index_write",
                        "materialization", "training")
        ))

    def test_expected_sinks_are_personal(self):
        sinks = MODULE.expected_persistent_sinks("m33-i0-fixture-20260822a", self.policy)
        self.assertEqual(set(sinks), {"results", "work", "logs", "manifests"})
        self.assertTrue(all(uri.startswith(MODULE.PROJECT_WRITE_ROOT) for uri in sinks.values()))

    def test_exact_authorized_read_descriptors(self):
        approved = [
            self.descriptor(),
            self.descriptor("gs://frozen-data-br/nambr/source.g.vcf.gz"),
            self.descriptor("gs://teams-usp/frank/lai-exploracion-datos/runs/old/object.json"),
        ]
        for descriptor in approved:
            authorized = dict(descriptor)
            authorized["logical_id"] = descriptor["logical_id"]
            with self.subTest(uri=descriptor["gcs_uri"]):
                self.assertEqual(
                    MODULE.validate_read_descriptor(descriptor, authorized, self.policy),
                    descriptor,
                )

    def test_rejects_unapproved_read_locations(self):
        for uri in (
            "gs://teams-usp/nathan/object",
            "gs://unknown-bucket/object",
            "https://storage.googleapis.com/projects-usp/object",
        ):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                descriptor = self.descriptor(uri)
                MODULE.validate_read_descriptor(descriptor, descriptor, self.policy)

    def test_unregistered_truth_and_generation_drift_are_rejected(self):
        authorized = self.descriptor()
        truth = self.descriptor(
            "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/EVAL/truth.tsv"
        )
        with self.assertRaisesRegex(ValueError, "not exactly authorized"):
            MODULE.validate_read_descriptor(truth, authorized, self.policy)
        changed_generation = dict(authorized)
        changed_generation["gcs_generation"] = "123456790"
        with self.assertRaisesRegex(ValueError, "not exactly authorized"):
            MODULE.validate_read_descriptor(changed_generation, authorized, self.policy)

    def test_accepts_each_write_namespace(self):
        examples = {
            "runs": "runs/run-a/result.json",
            "work": "work/nextflow/run-a/chunk",
            "logs": "logs/run-a/trace.txt",
            "manifests": "manifests/run-a/receipt.json",
            "software_receipts": "software/containers/tabix/1.16/receipt.json",
        }
        for namespace, suffix in examples.items():
            uri = MODULE.PROJECT_WRITE_ROOT + suffix
            with self.subTest(namespace=namespace):
                run_id = None if namespace == "software_receipts" else "run-a"
                self.assertEqual(
                    MODULE.validate_write_uri(uri, namespace, self.policy, run_id=run_id), uri
                )

    def test_rejects_lab_and_sibling_writes(self):
        cases = [
            ("runs", "gs://projects-usp/dnaBr-lai/datalake/new.json"),
            ("runs", "gs://teams-usp/nathan/run/result.json"),
            ("runs", "gs://teams-usp/frank/other-project/result.json"),
            ("runs", "gs://teams-usp/frank/lai-exploracion-datos/work/result.json"),
            ("runs", "gs://teams-usp/frank/lai-exploracion-datos/runs/other-run/result.json"),
        ]
        for namespace, uri in cases:
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                MODULE.validate_write_uri(uri, namespace, self.policy, run_id="run-a")

    def test_rejects_ambiguous_or_traversing_uris(self):
        bad = [
            "gs://teams-usp/frank/lai-exploracion-datos/runs/../nathan/x",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/%2e%2e/x",
            "gs://teams-usp/frank/lai-exploracion-datos//runs/x",
            "gs://teams-usp/frank/lai-exploracion-datos/runs\\x",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/*",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/{run_id}/x",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/x?generation=1",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/x#fragment",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/\u212a",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/run-a/$(cmd)",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/run-a/`cmd`",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/run-a/a;b",
            "gs://teams-usp/frank/lai-exploracion-datos/runs/run-a/a|b",
            "GS://teams-usp/frank/lai-exploracion-datos/runs/run-a/x",
        ]
        for uri in bad:
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                MODULE.validate_write_uri(uri, "runs", self.policy, run_id="run-a")

    def test_write_must_name_an_object(self):
        with self.assertRaises(ValueError):
            MODULE.validate_write_uri(
                MODULE.PROJECT_WRITE_ROOT + "runs/", "runs", self.policy, run_id="run-a"
            )

    def test_ready_must_be_last(self):
        self.assertEqual(
            MODULE.validate_publication_order(["part-00000", "manifest.json", "READY"]),
            ["part-00000", "manifest.json", "READY"],
        )
        with self.assertRaises(ValueError):
            MODULE.validate_publication_order(["READY", "part-00000"])
        with self.assertRaises(ValueError):
            MODULE.validate_publication_order([
                MODULE.PROJECT_WRITE_ROOT + "runs/run-a/READY",
                MODULE.PROJECT_WRITE_ROOT + "runs/run-a/part-00000",
            ])

    def test_nested_namespaces_are_rejected(self):
        mutated = json.loads(POLICY.read_text(encoding="utf-8"))
        mutated["persistent_namespaces"]["logs"] = "runs/logs/"
        with self.assertRaisesRegex(ValueError, "nested"):
            MODULE.validate_policy(mutated)

    def test_cli_rejects_ready_before_other_writes(self):
        completed = subprocess.run([
            sys.executable, str(SCRIPT), "--policy", str(POLICY),
            "--run-id", "run-a",
            "--write", "runs", MODULE.PROJECT_WRITE_ROOT + "runs/run-a/READY",
            "--write", "runs", MODULE.PROJECT_WRITE_ROOT + "runs/run-a/part-00000",
        ], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("READY must be written last", completed.stderr)

    def test_real_execution_hard_stop(self):
        with self.assertRaisesRegex(ValueError, "remains blocked"):
            MODULE.require_real_execution_authorized(self.policy)

    def test_run_id_is_fail_closed(self):
        for run_id in ("ab", "M33-run", "../run", "run/name", "run name", "a" * 65):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                MODULE.validate_run_id(run_id)

    def test_duplicate_json_key_is_rejected(self):
        text = POLICY.read_text(encoding="utf-8")
        mutated = text.replace('"schema_version": "1.0.0",',
                               '"schema_version": "1.0.0", "schema_version": "1.0.0",', 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(mutated, encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_policy(path)

    def test_critical_policy_mutations_are_rejected(self):
        mutations = [
            ("persistent_write_contract", "object_creation_precondition", "none"),
            ("google_batch", "resource_labels", {"team": "other"}),
            ("oci", "runtime_reference_must_use_digest", False),
            ("ephemeral_scratch", "must_be_removed_after_verified_publication", False),
            ("execution_authorization", "real_asset_read", True),
        ]
        for section, key, value in mutations:
            mutated = json.loads(POLICY.read_text(encoding="utf-8"))
            mutated[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                MODULE.validate_policy(mutated)

    def test_cli_emits_only_contract_receipt(self):
        completed = subprocess.run([
            sys.executable, str(SCRIPT), "--policy", str(POLICY),
            "--run-id", "m33-storage-fixture-20260822a",
        ], check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "PASS_STORAGE_POLICY_CONTRACT_ONLY")


if __name__ == "__main__":
    unittest.main()
