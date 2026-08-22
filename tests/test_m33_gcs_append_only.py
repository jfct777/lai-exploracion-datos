#!/usr/bin/env python3

import importlib.util
import json
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import tempfile


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

    def test_worker_publishes_candidate_without_ready(self):
        class Storage:
            @staticmethod
            def load_policy(_): return {}
            @staticmethod
            def validate_run_id(_): return None
            @staticmethod
            def validate_write_uri(uri, *_args, **_kwargs): return uri
            @staticmethod
            def validate_publication_order(uris):
                self.assertFalse(any(uri.endswith("/READY") for uri in uris))
                return uris

        created = []
        class Client:
            def __init__(self, _): pass
            def create_and_verify(self, bucket, object_name, content, _content_type):
                created.append(object_name)
                return {"gcs_uri": f"gs://{bucket}/{object_name}", "generation": "1", "size_bytes": len(content), "sha256": MODULE.sha256_bytes(content)}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kat, controller, output = root / "kat.json", root / "controller.json", root / "out.json"
            kat.write_text("{}", encoding="utf-8"); controller.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                run_id="m33-run", kat_receipt=kat, controller_receipt=controller,
                storage_policy=root / "policy", storage_validator=root / "validator", output=output,
            )
            with mock.patch.object(MODULE, "load_module", return_value=Storage), \
                 mock.patch.object(MODULE, "authenticate_runtime", return_value=MODULE.RUNTIME_SERVICE_ACCOUNT), \
                 mock.patch.object(MODULE, "access_token", return_value="token"), \
                 mock.patch.object(MODULE, "AppendOnlyGCS", Client):
                result = MODULE.publish(args)
        self.assertEqual(result["status"], "PASS_PUBLICATION_CANDIDATE_NO_READY")
        self.assertFalse(result["ready_created"])
        self.assertFalse(any(name.endswith("/READY") for name in created))

    def test_controller_writes_postflight_and_candidate_but_not_ready(self):
        class Storage:
            @staticmethod
            def load_policy(_): return {}
            @staticmethod
            def validate_run_id(_): return None
            @staticmethod
            def validate_write_uri(uri, *_args, **_kwargs): return uri
            @staticmethod
            def validate_publication_order(uris):
                self.assertTrue(uris[-1].endswith("/READY_CANDIDATE"))
                self.assertFalse(any(uri.endswith("/READY") for uri in uris))
                return uris

        created = []
        manifest = MODULE.encode_json({"run_id": "m33-run", "status": "PASS_APPEND_ONLY_REOPENED_AND_HASHED"})
        class Client:
            def __init__(self, _): pass
            def record_for_existing(self, bucket, object_name):
                return {"gcs_uri": f"gs://{bucket}/{object_name}", "generation": "7", "size_bytes": len(manifest), "sha256": MODULE.sha256_bytes(manifest), "content": manifest}
            def reopen_record(self, record): return record["content"]
            def create_and_verify(self, bucket, object_name, content, _content_type):
                created.append(object_name)
                return {"gcs_uri": f"gs://{bucket}/{object_name}", "generation": str(len(created)), "size_bytes": len(content), "sha256": MODULE.sha256_bytes(content)}

        postflight = {
            "run_id": "m33-run", "run_label": "0123456789abcdef",
            "status": "PASS_CHILD_JOBS_TERMINAL_AND_AUTHENTICATED",
            "known_answers": {"record_count": 4},
        }
        with mock.patch.object(MODULE, "load_module", return_value=Storage), \
             mock.patch.object(MODULE, "authenticate_service_account", return_value=MODULE.CONTROLLER_SERVICE_ACCOUNT), \
             mock.patch.object(MODULE, "access_token", return_value="token"), \
             mock.patch.object(MODULE, "AppendOnlyGCS", Client):
            result = MODULE.finalize_candidate(
                run_id="m33-run", storage_policy=Path("policy"),
                storage_validator=Path("validator"), postflight=postflight,
            )
        self.assertEqual(created[-2:], [
            "frank/lai-exploracion-datos/logs/m33-run/postflight.json",
            "frank/lai-exploracion-datos/runs/m33-run/READY_CANDIDATE",
        ])
        self.assertEqual(result["status"], "PASS_CANDIDATE_CREATED_NO_READY")
        self.assertFalse(result["ready_created"])

    def test_external_finalizer_requires_matching_inventory_and_writes_ready_last(self):
        class Storage:
            @staticmethod
            def load_policy(_): return {}
            @staticmethod
            def validate_run_id(_): return None
            @staticmethod
            def validate_write_uri(uri, *_args, **_kwargs): return uri
            @staticmethod
            def validate_publication_order(uris):
                self.assertTrue(uris[-1].endswith("/READY"))
                return uris

        jobs = ["controller", "child-a", "child-b", "compare", "publish"]
        payloads = {
            "publication.manifest.json": {"run_id": "m33-run"},
            "postflight.json": {
                "run_id": "m33-run", "run_label": "0123456789abcdef",
                "known_answers": {"record_count": 4},
                "inventory": {"controller_job": jobs[0], "child_jobs": jobs[1:]},
            },
            "READY_CANDIDATE": {"run_id": "m33-run", "status": "NON_CONSUMABLE_CANDIDATE"},
        }
        created = []
        class Client:
            def __init__(self, _): pass
            def record_for_existing(self, bucket, object_name):
                content = MODULE.encode_json(payloads[object_name.rsplit("/", 1)[-1]])
                return {"gcs_uri": f"gs://{bucket}/{object_name}", "generation": "7", "size_bytes": len(content), "sha256": MODULE.sha256_bytes(content), "content": content}
            def reopen_record(self, record): return record["content"]
            def create_and_verify(self, bucket, object_name, content, _content_type):
                created.append(object_name)
                return {"gcs_uri": f"gs://{bucket}/{object_name}", "generation": str(len(created)), "size_bytes": len(content), "sha256": MODULE.sha256_bytes(content)}

        close = {
            "run_id": "m33-run", "status": "PASS_ALL_RUN_JOBS_SUCCEEDED_ZERO_ACTIVE",
            "job_names": jobs,
        }
        with mock.patch.object(MODULE, "load_module", return_value=Storage), \
             mock.patch.object(MODULE, "token_email", return_value=MODULE.CONTROLLER_SERVICE_ACCOUNT), \
             mock.patch.object(MODULE, "AppendOnlyGCS", Client):
            result = MODULE.finalize_ready(
                run_id="m33-run", storage_policy=Path("policy"),
                storage_validator=Path("validator"), external_close=close, token="token",
            )
        self.assertEqual(created[-2:], [
            "frank/lai-exploracion-datos/logs/m33-run/external_close.json",
            "frank/lai-exploracion-datos/runs/m33-run/READY",
        ])
        self.assertEqual(result["status"], "PASS_READY_CREATED_AFTER_ZERO_ACTIVE")


if __name__ == "__main__":
    unittest.main()
