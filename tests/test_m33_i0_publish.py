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
        self.assertEqual(payload["destination_prefix"], MODULE.EXPECTED_PREFIX)
        self.assertEqual(payload["policy"]["if_generation_match"], 0)
        for flag in ("allow_vcf", "allow_overwrite", "allow_delete", "allow_ready",
                     "safe_bridge", "materialize", "training", "global_ready"):
            self.assertFalse(payload["policy"][flag])

    def test_source_auth_inventory_is_exact(self):
        self.assertRegex(MODULE.load_source_auth(SOURCE_AUTH, ROOT), r"^[0-9a-f]{64}$")

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

    def test_authorization_rejects_vcf_and_ready(self):
        for label, mutate in (
            ("vcf", lambda p: p["artifacts"].__setitem__("x.vcf.gz", {"size_bytes": 1, "sha256": "0" * 64})),
            ("ready", lambda p: p["policy"].__setitem__("allow_ready", True)),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                payload = json.loads(AUTH.read_text())
                mutate(payload)
                path = Path(temporary) / "auth.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.load_authorization(path)

    @mock.patch("subprocess.run")
    def test_empty_prefix_rejects_any_existing_object(self, run):
        run.return_value = mock.Mock(returncode=0, stdout="gs://teams-usp/frank/existing\n")
        with self.assertRaisesRegex(ValueError, "not empty"):
            MODULE.require_empty_prefix(MODULE.EXPECTED_PREFIX)

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
