#!/usr/bin/env python3

import importlib.util
import json
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "m33_gcs_append_only.py"
SPEC = importlib.util.spec_from_file_location("m33_gcs_append_only", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeResponse:
    def __init__(self, payload: bytes, headers=None):
        self.payload = payload
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


class AppendOnlyPublisherTests(unittest.TestCase):
    def test_create_uses_generation_zero_and_reopens_exact_generation(self):
        content = b"synthetic\n"
        responses = [
            FakeResponse(json.dumps({"generation": "17"}).encode()),
            FakeResponse(json.dumps({"name": "prefix/object", "generation": "17", "size": str(len(content))}).encode()),
            FakeResponse(content),
        ]
        with mock.patch.object(MODULE.urllib.request, "urlopen", side_effect=responses) as opened:
            record = MODULE.AppendOnlyGCS("token").create_and_verify(
                "bucket", "prefix/object", content, "application/octet-stream"
            )
        self.assertEqual(record["generation"], "17")
        upload_url = opened.call_args_list[0].args[0].full_url
        self.assertIn("ifGenerationMatch=0", upload_url)
        self.assertIn("generation=17", opened.call_args_list[1].args[0].full_url)
        self.assertIn("generation=17", opened.call_args_list[2].args[0].full_url)

    def test_runtime_identity_is_exact(self):
        headers = {"Metadata-Flavor": "Google"}
        good = FakeResponse(MODULE.RUNTIME_SERVICE_ACCOUNT.encode(), headers)
        with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=good):
            self.assertEqual(MODULE.authenticate_runtime(), MODULE.RUNTIME_SERVICE_ACCOUNT)
        bad = FakeResponse(b"wrong@example.iam.gserviceaccount.com", headers)
        with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=bad):
            with self.assertRaisesRegex(ValueError, "runtime service account"):
                MODULE.authenticate_runtime()


if __name__ == "__main__":
    unittest.main()
