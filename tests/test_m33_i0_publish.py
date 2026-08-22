#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_i0_publish", ROOT / "bin" / "m33_i0_publish.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
AUTH = ROOT / "conf" / "m33_i0_publication_authorization.json"
SOURCE_AUTH = ROOT / "conf" / "m33_i0_publication_source_auth.json"


class M33I0PublicationTests(unittest.TestCase):
    def test_authorization_is_exactly_append_only_and_no_ready(self):
        payload = MODULE.load_authorization(AUTH)
        self.assertEqual(len(payload["artifacts"]), 8)
        self.assertEqual(set(payload["artifacts"]), set(MODULE.EXPECTED_ARTIFACT_ORDER))
        self.assertEqual(payload["destination_prefix"], MODULE.EXPECTED_PREFIX)
        self.assertEqual(payload["policy"]["if_generation_match"], 0)
        for flag in ("allow_vcf", "allow_overwrite", "allow_delete", "allow_ready",
                     "safe_bridge", "materialize", "training", "global_ready"):
            self.assertFalse(payload["policy"][flag])

    def test_source_auth_inventory_is_exact(self):
        self.assertRegex(MODULE.load_source_auth(SOURCE_AUTH, ROOT), r"^[0-9a-f]{64}$")

    def test_runtime_controls_must_be_canonical_repository_files(self):
        base = ROOT / "conf/m33_storage_namespace_policy.json"
        self.assertEqual(MODULE.validate_control_paths(ROOT, AUTH, SOURCE_AUTH, base), ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / AUTH.name
            external.write_bytes(AUTH.read_bytes())
            with self.assertRaisesRegex(ValueError, "non-canonical control path"):
                MODULE.validate_control_paths(ROOT, external, SOURCE_AUTH, base)

    def test_local_inventory_rejects_vcf_extra_and_writable_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "m33_i0_real.manifest.json"
            artifact.write_bytes(b"x")
            artifact.chmod(0o444)
            digest = MODULE.sha256_file(artifact)
            authorization = {
                "artifacts": {"m33_i0_real.manifest.json": {"size_bytes": 1, "sha256": digest}},
                "source_manifest_sha256": digest,
            }
            MODULE.validate_local_artifacts(root, authorization)
            artifact.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "mode differs"):
                MODULE.validate_local_artifacts(root, authorization)
            artifact.chmod(0o444)
            vcf = root / "forbidden.vcf.gz"
            vcf.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "inventory"):
                MODULE.validate_local_artifacts(root, authorization)

    def test_authorization_rejects_vcf_ready_policy_and_reserved_path(self):
        for label, mutate in (
            ("vcf", lambda p: p["artifacts"].__setitem__("x.vcf.gz", {"size_bytes": 1, "sha256": "0" * 64})),
            ("ready_policy", lambda p: p["policy"].__setitem__("allow_ready", True)),
            ("ready_path", lambda p: (
                p["artifacts"].pop("I0_REAL_PASS_NON_CONSUMABLE"),
                p["artifacts"].__setitem__("READY", {"size_bytes": 1, "sha256": "0" * 64}),
            )),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                payload = json.loads(AUTH.read_text())
                mutate(payload)
                path = Path(temporary) / "auth.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.load_authorization(path)

    @mock.patch("subprocess.run")
    def test_empty_prefix_requires_explicit_no_match_diagnostic(self, run):
        run.return_value = mock.Mock(
            returncode=1, stdout="", stderr=MODULE.EMPTY_PREFIX_MESSAGE + "\n",
        )
        self.assertEqual(MODULE.list_remote_objects(MODULE.EXPECTED_PREFIX), {})
        run.return_value = mock.Mock(
            returncode=1, stdout="", stderr="ERROR: permission denied\n",
        )
        with self.assertRaisesRegex(ValueError, "listing failed"):
            MODULE.list_remote_objects(MODULE.EXPECTED_PREFIX)

    @mock.patch("subprocess.run")
    def test_nonempty_prefix_listing_is_parsed_exactly(self, run):
        name = "frank/lai-exploracion-datos/runs/m33-i0-real-20260822a/x"
        run.return_value = mock.Mock(returncode=0, stderr="", stdout=json.dumps([{
            "url": MODULE.EXPECTED_PREFIX,
            "type": "prefix",
        }, {
            "url": MODULE.EXPECTED_PREFIX + "root17/",
            "type": "prefix",
        }, {
            "url": f"gs://teams-usp/{name}#1",
            "type": "cloud_object",
            "metadata": {"bucket": "teams-usp", "name": name, "generation": "1"},
        }]))
        observed = MODULE.list_remote_objects(MODULE.EXPECTED_PREFIX)
        self.assertEqual(set(observed), {MODULE.EXPECTED_PREFIX + "x"})

    def test_partial_publication_is_recoverable_but_extra_or_early_receipt_stops(self):
        ordered = (MODULE.EXPECTED_PREFIX + "a", MODULE.EXPECTED_PREFIX + "b")
        receipt = MODULE.EXPECTED_PREFIX + "publication.receipt.json"
        for length in range(len(ordered) + 1):
            MODULE.validate_initial_inventory({uri: {} for uri in ordered[:length]}, ordered, receipt)
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            MODULE.validate_initial_inventory({MODULE.EXPECTED_PREFIX + "extra": {}}, ordered, receipt)
        with self.assertRaisesRegex(ValueError, "before all"):
            MODULE.validate_initial_inventory({receipt: {}}, ordered, receipt)
        with self.assertRaisesRegex(ValueError, "valid publication prefix"):
            MODULE.validate_initial_inventory({ordered[1]: {}}, ordered, receipt)
        MODULE.validate_initial_inventory(
            {**{uri: {} for uri in ordered}, receipt: {}}, ordered, receipt,
        )

    def test_local_receipt_can_resume_only_when_exact_and_read_only(self):
        payload = {"status": "PASS"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "publication.receipt.json"
            first = MODULE.prepare_local_receipt(path, payload)
            second = MODULE.prepare_local_receipt(path, payload)
            self.assertEqual(first, second)
            self.assertEqual(path.stat().st_mode & 0o777, 0o444)
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "mode differs"):
                MODULE.prepare_local_receipt(path, payload)

    def test_dependency_order_puts_evidence_before_markers(self):
        self.assertLess(
            MODULE.EXPECTED_ARTIFACT_ORDER.index("root17/root17.i0_real.receipt.json"),
            MODULE.EXPECTED_ARTIFACT_ORDER.index("root17/ROOT17_I0_REAL_PASS_NON_CONSUMABLE"),
        )
        self.assertEqual(MODULE.EXPECTED_ARTIFACT_ORDER[-1], "I0_REAL_PASS_NON_CONSUMABLE")

    @mock.patch("subprocess.run")
    def test_bucket_controls_require_ubla_and_public_access_prevention(self, run):
        run.return_value = mock.Mock(returncode=0, stdout=json.dumps({
            "uniform_bucket_level_access": True,
            "public_access_prevention": "enforced",
        }))
        controls = MODULE.validate_bucket_controls(MODULE.EXPECTED_PREFIX)
        self.assertTrue(controls["uniform_bucket_level_access"])
        run.return_value = mock.Mock(returncode=0, stdout=json.dumps({
            "uniform_bucket_level_access": True,
            "public_access_prevention": "inherited",
        }))
        with self.assertRaisesRegex(ValueError, "public access prevention"):
            MODULE.validate_bucket_controls(MODULE.EXPECTED_PREFIX)

    @mock.patch("subprocess.run")
    def test_untracked_file_blocks_publisher_preflight(self, run):
        run.return_value = mock.Mock(returncode=0, stdout="?? untracked\n")
        with self.assertRaisesRegex(ValueError, "fully clean"):
            MODULE.validate_publisher_commit(ROOT, {"publisher_code_commit": "0" * 40})

    @mock.patch.object(MODULE, "describe_object")
    @mock.patch.object(MODULE, "list_remote_objects")
    def test_final_inventory_rejects_extra_object(self, listing, describe):
        uri = MODULE.EXPECTED_PREFIX + "a"
        extra = MODULE.EXPECTED_PREFIX + "extra"
        listing.return_value = {uri: {}, extra: {}}
        describe.return_value = {"generation": "1", "size": 1}
        expected = {uri: {"size_bytes": 1, "sha256": "0" * 64}}
        verified = {uri: {"generation": "1"}}
        with self.assertRaisesRegex(ValueError, "final remote inventory"):
            MODULE.validate_exact_inventory(MODULE.EXPECTED_PREFIX, expected, verified)

    @mock.patch("subprocess.run")
    def test_publish_uses_create_only_precondition(self, run):
        metadata = {
            "generation": "1", "size": 1, "crc32c_hash": "x", "md5_hash": "y",
        }
        def command_result(args, **kwargs):
            if args[:4] == ["gcloud", "storage", "objects", "describe"]:
                return mock.Mock(returncode=0, stdout=json.dumps(metadata))
            if args[:3] == ["gcloud", "storage", "cp"] and "#1" in args[-2]:
                Path(args[-1]).write_bytes(b"x")
            return mock.Mock(returncode=0, stdout="")
        run.side_effect = command_result
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "x"
            source.write_bytes(b"x")
            with mock.patch.object(MODULE, "sha256_file", return_value="a" * 64):
                MODULE.publish_one(source, MODULE.EXPECTED_PREFIX + "x", {"size_bytes": 1, "sha256": "a" * 64})
        upload = run.call_args_list[0].args[0]
        self.assertIn("--if-generation-match=0", upload)


if __name__ == "__main__":
    unittest.main()
