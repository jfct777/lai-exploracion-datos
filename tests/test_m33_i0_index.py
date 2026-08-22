#!/usr/bin/env python3

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_i0_index", ROOT / "bin" / "m33_i0_index.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
AUTH = ROOT / "conf" / "m33_i0_fixture_authorization.json"
CONTRACT = ROOT / "conf" / "m33_m0_materializer_contract.json"
SOURCE_AUTH = ROOT / "conf" / "m33_i0_fixture_source_auth.json"


class M33I0IndexTests(unittest.TestCase):
    def test_authorization_is_fixture_only_and_pinned(self):
        payload = MODULE.load_authorization(AUTH, CONTRACT)
        self.assertFalse(payload["real_asset_read"])
        self.assertFalse(payload["safe_bridge"])
        self.assertFalse(payload["materialize"])
        self.assertTrue(payload["global_ready_forbidden"])
        self.assertIn("@sha256:", payload["tabix_image"])

    def test_authorization_rejects_real_asset_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed = json.loads(AUTH.read_text())
            changed["real_asset_read"] = True
            path = Path(temporary) / "authorization.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden stage"):
                MODULE.load_authorization(path, CONTRACT)

    def test_authorization_rejects_descriptor_uri_or_generation(self):
        for field, value in (("uri", "synthetic://wrong"), ("generation", "2")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                changed = json.loads(AUTH.read_text())
                changed["fixture_descriptor"][field] = value
                path = Path(temporary) / "authorization.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "descriptor differs"):
                    MODULE.load_authorization(path, CONTRACT)

    def test_effective_container_digest_must_match_authorization(self):
        payload = MODULE.load_authorization(AUTH, CONTRACT)
        self.assertEqual(MODULE.validate_container_image(MODULE.EXPECTED_IMAGE, payload), MODULE.EXPECTED_IMAGE)
        with self.assertRaisesRegex(ValueError, "effective Tabix OCI image differs"):
            MODULE.validate_container_image("example.invalid/tabix@sha256:" + "0" * 64, payload)

    def test_source_auth_inventory_is_exact(self):
        digest = MODULE.load_source_auth(SOURCE_AUTH, ROOT)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_line_digest_is_byte_exact(self):
        self.assertNotEqual(MODULE.line_digest([b"22\t1\n"]), MODULE.line_digest([b"22\t1\r\n"]))
        with self.assertRaisesRegex(ValueError, "unterminated"):
            MODULE.line_digest([b"22\t1"])

    def test_known_answers_are_frozen(self):
        self.assertEqual(MODULE.EXPECTED_RECORD_COUNT, 4)
        self.assertRegex(MODULE.EXPECTED_TBI_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(MODULE.EXPECTED_RECORD_SHA256, r"^[0-9a-f]{64}$")

    def test_atomic_copy_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.write_bytes(b"new")
            destination.write_bytes(b"old")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                MODULE.atomic_copy_no_overwrite(source, destination)
            self.assertEqual(destination.read_bytes(), b"old")

    def test_atomic_outputs_are_host_readable_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.write_bytes(b"index")
            MODULE.atomic_copy_no_overwrite(source, destination)
            self.assertEqual(destination.read_bytes(), b"index")
            self.assertEqual(destination.stat().st_mode & 0o777, 0o444)
            receipt = root / "receipt.json"
            MODULE.write_json_exclusive(receipt, {"status": "PASS"})
            self.assertEqual(receipt.stat().st_mode & 0o777, 0o444)

    def test_derive_rejects_writable_source_before_tabix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "flare.anc.vcf.gz"
            source.write_bytes(b"not-bgzip")
            source.chmod(0o600)
            for name in ("source.json",):
                (root / name).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "read-only"):
                MODULE.derive(
                    source=source, source_manifest=root / "source.json",
                    authorization=AUTH, contract=CONTRACT,
                    source_auth=SOURCE_AUTH, repo_root=ROOT,
                    output_tbi=root / "out.tbi", receipt=root / "receipt.json",
                    marker=root / "I0_FIXTURE_PASS", run_id="fixture-run",
                    container_image=MODULE.EXPECTED_IMAGE,
                )

    def test_derive_rejects_existing_output_before_tabix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "flare.anc.vcf.gz"
            source.write_bytes(b"not-bgzip")
            source.chmod(0o400)
            (root / "source.json").write_text("{}", encoding="utf-8")
            output = root / "flare.anc.vcf.gz.tbi"
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "already exists"):
                MODULE.derive(
                    source=source, source_manifest=root / "source.json",
                    authorization=AUTH, contract=CONTRACT,
                    source_auth=SOURCE_AUTH, repo_root=ROOT,
                    output_tbi=output, receipt=root / "i0_fixture.receipt.json",
                    marker=root / "I0_FIXTURE_PASS", run_id="fixture-run",
                    container_image=MODULE.EXPECTED_IMAGE,
                )

    def test_source_manifest_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.json"
            payload = {
                "schema_version": "1.0.0",
                "stage": "M33_I0_FIXTURE_SOURCE",
                "status": "PASS_AUTHENTICATED_SYNTHETIC_SOURCE",
                "source_vcf_sha256": MODULE.EXPECTED_FIXTURE_SHA256,
                "source_descriptor": json.loads(AUTH.read_text())["fixture_descriptor"],
                "tabix_version": MODULE.EXPECTED_TABIX_VERSION,
                "tabix_oci_repository_digest": MODULE.EXPECTED_IMAGE,
                "contains_real_genomic_data": False,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            payload["source_auth_sha256"] = MODULE.sha256_file(SOURCE_AUTH)
            path.write_text(json.dumps(payload), encoding="utf-8")
            MODULE.validate_source_manifest(
                path, MODULE.EXPECTED_FIXTURE_SHA256, MODULE.sha256_file(SOURCE_AUTH),
                json.loads(AUTH.read_text())["fixture_descriptor"], MODULE.EXPECTED_IMAGE,
            )
            payload["extra"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                MODULE.validate_source_manifest(
                    path, MODULE.EXPECTED_FIXTURE_SHA256, MODULE.sha256_file(SOURCE_AUTH),
                    json.loads(AUTH.read_text())["fixture_descriptor"], MODULE.EXPECTED_IMAGE,
                )

    def test_derive_rejects_ready_or_external_output_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "flare.anc.vcf.gz"
            source.write_bytes(b"fixture")
            source.chmod(0o400)
            (root / "source.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "index basename"):
                MODULE.derive(
                    source=source, source_manifest=root / "source.json",
                    authorization=AUTH, contract=CONTRACT,
                    source_auth=SOURCE_AUTH, repo_root=ROOT,
                    output_tbi=root / "READY", receipt=root / "i0_fixture.receipt.json",
                    marker=root / "I0_FIXTURE_PASS", run_id="fixture-run",
                    container_image=MODULE.EXPECTED_IMAGE,
                )

    def test_make_fixture_rejects_output_escape_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "fixture VCF basename differs"):
                MODULE.make_fixture(
                    root / "READY", root / "fixture_source.receipt.json", AUTH, CONTRACT,
                    SOURCE_AUTH, ROOT, MODULE.EXPECTED_IMAGE,
                )

    def test_index_evidence_rejects_query_divergence_and_source_mutation(self):
        evidence = dict(
            tbi_a=MODULE.EXPECTED_TBI_SHA256, tbi_b=MODULE.EXPECTED_TBI_SHA256,
            indexed_count_a=4, indexed_count_b=4,
            indexed_sha_a=MODULE.EXPECTED_RECORD_SHA256,
            indexed_sha_b=MODULE.EXPECTED_RECORD_SHA256,
            sequential_count=4, sequential_sha=MODULE.EXPECTED_RECORD_SHA256,
            source_size_before=270, source_size_after=270,
            source_sha_before=MODULE.EXPECTED_FIXTURE_SHA256,
            source_sha_after=MODULE.EXPECTED_FIXTURE_SHA256,
        )
        MODULE.validate_index_evidence(**evidence)
        divergent = dict(evidence, indexed_sha_b="0" * 64)
        with self.assertRaisesRegex(ValueError, "record hashes differ"):
            MODULE.validate_index_evidence(**divergent)
        mutated = dict(evidence, source_sha_after="0" * 64)
        with self.assertRaisesRegex(ValueError, "source VCF changed"):
            MODULE.validate_index_evidence(**mutated)


if __name__ == "__main__":
    unittest.main()
