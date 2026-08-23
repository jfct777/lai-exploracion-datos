#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m33_safe_bridge_technical_publish",
    ROOT / "bin/m33_safe_bridge_technical_publish.py",
)
PUBLISH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PUBLISH)


class TechnicalPublicationTests(unittest.TestCase):
    def test_config_is_minimal_and_non_consumable(self) -> None:
        config = PUBLISH.load_config(
            ROOT / "conf/m33_safe_bridge_technical_publication_config.json")
        self.assertEqual(len(config["source_artifact_order"]), 7)
        self.assertEqual(config["policy"]["final_object_count"], 9)
        self.assertFalse(config["policy"]["allow_npz"])
        self.assertFalse(config["policy"]["consumable"])

    def test_only_receipts_and_provenance_are_allowlisted(self) -> None:
        self.assertEqual(len(PUBLISH.SOURCE_ORDER), 7)
        self.assertEqual(len(PUBLISH.FINAL_ORDER), 9)
        self.assertTrue(all(not name.endswith(".npz") for name in PUBLISH.FINAL_ORDER))
        self.assertTrue(all("READY" not in name for name in PUBLISH.FINAL_ORDER))

    def test_authorization_descriptors_are_exact(self) -> None:
        path = ROOT / "conf/m33_safe_bridge_technical_publication_authorization.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "AUTHORIZED_MINIMAL_APPEND_ONLY_AUDIT_PUBLICATION"
        payload["publication_config_sha256"] = PUBLISH.sha256_file(
            ROOT / "conf/m33_safe_bridge_technical_publication_config.json")
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "authorization.json"
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            observed = PUBLISH.load_authorization(
                candidate, payload["publication_config_sha256"])
        self.assertEqual(set(observed["artifacts"]), set(PUBLISH.SOURCE_ORDER))

    def test_blocked_authorization_cannot_publish(self) -> None:
        path = ROOT / "conf/m33_safe_bridge_technical_publication_authorization.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"].startswith("BLOCKED_"):
            with self.assertRaisesRegex(ValueError, "not enabled"):
                PUBLISH.load_authorization(path, "0" * 64)

    def test_manifest_contract_keeps_npz_ephemeral(self) -> None:
        source = (ROOT / "bin/m33_safe_bridge_technical_publish.py").read_text(encoding="utf-8")
        self.assertIn('"validated_ephemeral_artifacts": ephemeral', source)
        self.assertIn('"npz_published": False', source)
        self.assertIn('"pseudonymous_individual_artifacts_persisted": False', source)

    def test_remote_partial_prefix_must_follow_exact_order(self) -> None:
        ordered = tuple(PUBLISH.PREFIX + relative for relative in PUBLISH.FINAL_ORDER)
        valid = {uri: {} for uri in ordered[:3]}
        invalid = {ordered[2]: {}}
        self.assertEqual(set(valid), set(ordered[:len(valid)]))
        self.assertNotEqual(set(invalid), set(ordered[:len(invalid)]))

    def test_atomic_json_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / PUBLISH.MANIFEST
            descriptor = PUBLISH.atomic_json(path, {"status": "PASS"})
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
            self.assertEqual(descriptor["sha256"], PUBLISH.sha256_file(path))
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                PUBLISH.atomic_json(path, {"status": "PASS"})

    def test_upload_uses_generation_zero_precondition(self) -> None:
        expected = {"size_bytes": 1, "sha256": "0" * 64}
        with mock.patch.object(PUBLISH.subprocess, "run") as run, \
                mock.patch.object(PUBLISH, "verify_remote", return_value={"generation": "1"}):
            PUBLISH.publish_one(Path("local"), "gs://bucket/object", expected)
        command = run.call_args.args[0]
        self.assertIn("--if-generation-match=0", command)


if __name__ == "__main__":
    unittest.main()
