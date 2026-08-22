#!/usr/bin/env python3

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_i0_real", ROOT / "bin" / "m33_i0_real.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
AUTH = ROOT / "conf" / "m33_i0_real_authorization.json"
CONTRACT = ROOT / "conf" / "m33_m0_materializer_contract.json"
SOURCE_AUTH = ROOT / "conf" / "m33_i0_real_source_auth.json"


class M33I0RealTests(unittest.TestCase):
    def test_authorization_opens_only_exact_i0_boundary(self):
        payload = MODULE.load_authorization(AUTH, CONTRACT)
        self.assertEqual(set(payload["roots"]), {"root17", "root18"})
        self.assertTrue(payload["execution"]["real_asset_read"])
        self.assertTrue(payload["execution"]["derive_index"])
        for key in ("safe_bridge", "materialize", "forward", "backward", "training", "truth", "test", "global_ready"):
            self.assertFalse(payload["execution"][key])
        self.assertFalse(payload["local_output_policy"]["publish_gcs_during_workflow"])
        self.assertIn("@sha256:", payload["tabix_image"])

    def test_authorization_rejects_descriptor_or_downstream_change(self):
        changes = [
            ("descriptor", lambda value: value["roots"]["root17"].__setitem__("generation", "latest")),
            ("training", lambda value: value["execution"].__setitem__("training", True)),
            ("sink", lambda value: value["local_output_policy"].__setitem__("allowed_future_prefix", "gs://projects-usp/")),
        ]
        for label, change in changes:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                payload = json.loads(AUTH.read_text())
                change(payload)
                changed = Path(temporary) / "auth.json"
                changed.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.load_authorization(changed, CONTRACT)

    def test_authorization_rejects_retroactive_base_contract_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed = json.loads(CONTRACT.read_text())
            changed["process_contracts"]["I0_DERIVE_AUTHENTICATE_FLARE_INDEX"]["implemented"] = True
            path = Path(temporary) / "contract.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract hash differs"):
                MODULE.load_authorization(AUTH, path)

    def test_source_auth_inventory_is_exact(self):
        self.assertRegex(MODULE.load_source_auth(SOURCE_AUTH, ROOT), r"^[0-9a-f]{64}$")

    def test_active_account_rejects_credential_override(self):
        with mock.patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/override"}):
            with self.assertRaisesRegex(ValueError, "credential override"):
                MODULE.active_gcloud_account()

    @mock.patch("subprocess.run")
    def test_active_account_requires_one_exact_account(self, run):
        run.return_value = mock.Mock(stdout="wrong@example.org\n")
        self.assertEqual(MODULE.active_gcloud_account(), "wrong@example.org")
        run.return_value = mock.Mock(stdout="a@example.org\nb@example.org\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MODULE.active_gcloud_account()

    @mock.patch("subprocess.run")
    def test_gcloud_impersonation_must_be_unset(self, run):
        run.return_value = mock.Mock(stdout="(unset)\n")
        MODULE.require_no_gcloud_impersonation()
        run.return_value = mock.Mock(stdout="worker@example.org\n")
        with self.assertRaisesRegex(ValueError, "impersonation"):
            MODULE.require_no_gcloud_impersonation()

    def test_derive_rejects_writable_source_before_tabix(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "root17.flare.anc.vcf.gz"
            source.write_bytes(b"not-real")
            source.chmod(0o600)
            (base / "root17.source.receipt.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "read-only"):
                MODULE.derive_index(
                    root_label="root17", source=source,
                    source_receipt=base / "root17.source.receipt.json",
                    output_tbi=base / "root17.flare.anc.vcf.gz.tbi",
                    receipt=base / "root17.i0_real.receipt.json",
                    marker=base / "ROOT17_I0_REAL_PASS_NON_CONSUMABLE",
                    authorization=AUTH, contract=CONTRACT, source_auth=SOURCE_AUTH,
                    helper_script=ROOT / "bin" / "m33_i0_index.py",
                    run_id="m33-i0-test", container_image=MODULE.EXPECTED_IMAGE,
                )

    def _root_payload(self, root_label):
        descriptor = MODULE.EXPECTED_ROOTS[root_label]
        return {
            "schema_version": "1.0.0",
            "stage": "M33_I0_REAL_INDEX",
            "status": "PASS_DOUBLE_INDEX_AND_QUERY_PARITY_TECHNICAL_ONLY",
            "run_id": "m33-i0-test",
            "root_label": root_label,
            "root_seed": descriptor["root_seed"],
            "source_generation": descriptor["generation"],
            "source_flare_sha256": descriptor["sha256"],
            "source_size_bytes": descriptor["size_bytes"],
            "source_auth_sha256": MODULE.sha256_file(SOURCE_AUTH),
            "tabix_version": MODULE.EXPECTED_VERSION,
            "tabix_oci_repository_digest": MODULE.EXPECTED_IMAGE,
            "build_replicates": 2,
            "build_replication_scope": "SAME_TASK_SAME_CONTAINER_SEPARATE_TEMP_DIRECTORIES",
            "independent_tbi_sha256": root_label[-2:] * 32,
            "query_parity_sha256": "a" * 64,
            "indexed_record_count": 79791,
            "sequential_record_count": 79791,
            "output_tbi_sha256": root_label[-2:] * 32,
            "append_only": True,
            "reopen_verified": True,
            "scientific_evidence": False,
            "safe_bridge": False,
            "materialize": False,
            "training": False,
            "truth": False,
            "test": False,
            "global_ready": False,
        }

    def _aggregate_evidence(self, base):
        receipts = []
        markers = []
        sources = []
        indexes = []
        for root_label in ("root17", "root18"):
            source = base / f"{root_label}.flare.anc.vcf.gz"
            source.write_bytes(root_label.encode("ascii"))
            source.chmod(0o400)
            sources.append(source)
            index = base / f"{root_label}.flare.anc.vcf.gz.tbi"
            index.write_bytes(root_label.encode("ascii"))
            index.chmod(0o444)
            indexes.append(index)
            payload = self._root_payload(root_label)
            tbi_sha = MODULE.sha256_file(index)
            payload["independent_tbi_sha256"] = tbi_sha
            payload["output_tbi_sha256"] = tbi_sha
            receipt = base / f"{root_label}.receipt.json"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            marker = base / f"{root_label}.marker.json"
            marker.write_text(json.dumps({
                "stage": "M33_I0_REAL_ROOT_PASS",
                "status": "PASS_TECHNICAL_NON_CONSUMABLE",
                "run_id": "m33-i0-test",
                "root_label": root_label,
                "receipt_sha256": MODULE.sha256_file(receipt),
                "global_ready": False,
            }), encoding="utf-8")
            receipts.append(receipt)
            markers.append(marker)
        return receipts, markers, sources, indexes

    @mock.patch.object(MODULE, "verify_existing_index")
    def test_aggregate_requires_two_bound_roots_and_never_ready(self, verify_index):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            receipts, markers, sources, indexes = self._aggregate_evidence(base)
            payload = MODULE.aggregate(
                receipts=receipts, markers=markers, sources=sources, indexes=indexes,
                manifest=base / "m33_i0_real.manifest.json",
                completion_marker=base / "I0_REAL_PASS_NON_CONSUMABLE",
                authorization=AUTH, contract=CONTRACT, source_auth=SOURCE_AUTH,
                helper_script=ROOT / "bin" / "m33_i0_index.py",
                run_id="m33-i0-test", container_image=MODULE.EXPECTED_IMAGE,
            )
            self.assertEqual(verify_index.call_count, 2)
            self.assertEqual(payload["root_pass_count"], 2)
            self.assertFalse(payload["global_ready"])
            self.assertFalse(payload["gcs_published"])
            self.assertEqual((base / "m33_i0_real.manifest.json").stat().st_mode & 0o777, 0o444)

    def test_aggregate_rejects_mutated_load_bearing_evidence(self):
        mutations = {
            "generation": lambda payload: payload.__setitem__("source_generation", "wrong"),
            "tabix_version": lambda payload: payload.__setitem__("tabix_version", "tabix 9.9"),
            "record_count": lambda payload: payload.__setitem__("indexed_record_count", 0),
            "query_digest": lambda payload: payload.__setitem__("query_parity_sha256", "invalid"),
            "training": lambda payload: payload.__setitem__("training", True),
            "extra_field": lambda payload: payload.__setitem__("unexpected", True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                receipts, markers, sources, indexes = self._aggregate_evidence(base)
                payload = json.loads(receipts[0].read_text())
                mutate(payload)
                receipts[0].write_text(json.dumps(payload), encoding="utf-8")
                marker = json.loads(markers[0].read_text())
                marker["receipt_sha256"] = MODULE.sha256_file(receipts[0])
                markers[0].write_text(json.dumps(marker), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.aggregate(
                        receipts=receipts, markers=markers, sources=sources, indexes=indexes,
                        manifest=base / "m33_i0_real.manifest.json",
                        completion_marker=base / "I0_REAL_PASS_NON_CONSUMABLE",
                        authorization=AUTH, contract=CONTRACT, source_auth=SOURCE_AUTH,
                        helper_script=ROOT / "bin" / "m33_i0_index.py",
                        run_id="m33-i0-test", container_image=MODULE.EXPECTED_IMAGE,
                    )

    def test_aggregate_rejects_bad_marker_or_tampered_index(self):
        for target in ("marker", "index"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                receipts, markers, sources, indexes = self._aggregate_evidence(base)
                if target == "marker":
                    payload = json.loads(markers[0].read_text())
                    payload["status"] = "FAIL"
                    markers[0].write_text(json.dumps(payload), encoding="utf-8")
                else:
                    indexes[0].chmod(0o644)
                    indexes[0].write_bytes(b"tampered")
                    indexes[0].chmod(0o444)
                with self.assertRaises(ValueError):
                    MODULE.aggregate(
                        receipts=receipts, markers=markers, sources=sources, indexes=indexes,
                        manifest=base / "m33_i0_real.manifest.json",
                        completion_marker=base / "I0_REAL_PASS_NON_CONSUMABLE",
                        authorization=AUTH, contract=CONTRACT, source_auth=SOURCE_AUTH,
                        helper_script=ROOT / "bin" / "m33_i0_index.py",
                        run_id="m33-i0-test", container_image=MODULE.EXPECTED_IMAGE,
                    )


if __name__ == "__main__":
    unittest.main()
