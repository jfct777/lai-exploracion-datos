"""Synthetic publisher checks: fake gcloud only; no network or scientific run."""
import base64
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import r02_publish_evidence as p


class FakeGcloud:
    def __init__(self):
        self.objects = {}
        self.attempts = []
        self.describes = []
        self.next_generation = 100
        self.after_upload = None
        self.before_describe = None
        self.failure = None

    def add(self, uri, data):
        self.next_generation += 1
        self.objects[uri] = dict(size=str(len(data)), generation=str(self.next_generation),
            md5_hash=base64.b64encode(hashlib.md5(data).digest()).decode(), data=data)

    def run(self, command, **kwargs):
        assert command[:4] == ["gcloud", "storage", "cp", "--if-generation-match=0"]
        assert command[4].startswith("--content-md5=")
        assert kwargs["env"]["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"] == "false"
        source, uri = command[-2:]
        self.attempts.append(uri)
        data = Path(source).read_bytes()
        assert command[4].split("=", 1)[1] == base64.b64encode(hashlib.md5(data).digest()).decode()
        if self.failure:
            return subprocess.CompletedProcess(command, 1, "", self.failure)
        if uri in self.objects:
            return subprocess.CompletedProcess(command, 1, "", "HTTPError 412: Precondition Failed")
        self.add(uri, data)
        if self.after_upload:
            self.after_upload(source, uri)
        return subprocess.CompletedProcess(command, 0, "", "")

    def check_output(self, command, **kwargs):
        assert command[:4] == ["gcloud", "storage", "objects", "describe"]
        assert command[-1] == "--format=json"
        uri = command[4]
        self.describes.append(uri)
        if self.before_describe:
            self.before_describe(uri)
        if uri not in self.objects:
            raise subprocess.CalledProcessError(1, command, stderr="404")
        return json.dumps({k: v for k, v in self.objects[uri].items() if k != "data"})


class EvidencePublisherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "evidence"
        self.output.mkdir()
        (self.output / "data.tsv").write_text("x\ty\n1\t2\n")
        (self.output / "progress.json").write_text('{"status":"COMPLETE"}\n')
        self.manifest = dict(schema="r02_segment_evidence_v1", status="COMPLETE_DESCRIPTIVE_NOT_VALIDATED",
            outputs_sha256={"data.tsv": p.pipeline.sha(self.output / "data.tsv")},
            operational_outputs_sha256={"progress.json": p.pipeline.sha(self.output / "progress.json")})
        self.freeze()
        self.destination = p.PREFIX + "test-run/m142/chr21"
        self.receipt = self.root / "publication.json"
        self.cloud = FakeGcloud()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(p.subprocess, "run", side_effect=self.cloud.run).start()
        mock.patch.object(p.pipeline.subprocess, "check_output", side_effect=self.cloud.check_output).start()

    def freeze(self):
        (self.output / "manifest.json").write_text(json.dumps(self.manifest) + "\n")
        self.expected = p.pipeline.sha(self.output / "manifest.json")

    def publish(self, **changes):
        arguments = dict(output_dir=self.output, expected_manifest_sha256=self.expected,
                         destination=self.destination, receipt=self.receipt)
        arguments.update(changes)
        return p.publish(**arguments)

    def test_exact_inventory_md5_generation_and_manifest_last(self):
        result = self.publish()
        self.assertEqual(result["status"], "PUBLISHED_VERIFIED")
        self.assertEqual([r["name"] for r in result["files"]], ["data.tsv", "progress.json", "manifest.json"])
        self.assertEqual(self.cloud.attempts[-1], self.destination + "/manifest.json")
        self.assertEqual(json.loads(self.receipt.read_text()), result)
        self.assertTrue(result["no_scientific_validation"])
        for row in result["files"]:
            self.assertEqual(row["generation"], self.cloud.objects[row["uri"]]["generation"])
            self.assertEqual(row["sha256"], p.pipeline.sha(row["path"]))
            self.assertGreaterEqual(self.cloud.describes.count(row["uri"]), 2)

    def test_descriptive_community_summary_status_and_inventory(self):
        self.manifest["schema"] = "r02_community_summary_v1"
        self.manifest["status"] = "COMPLETE_DESCRIPTIVE_COMMUNITY_SUMMARY_NOT_VALIDATED"
        self.freeze()
        result = self.publish()
        self.assertEqual(result["evidence_schema"], self.manifest["schema"])
        self.assertTrue(result["no_scientific_validation"])

    def test_biological_evaluation_descriptive_status_and_inventory(self):
        self.manifest["schema"] = "r02_biological_evaluation_v1"
        self.manifest["status"] = "BIOLOGICAL_DESCRIPTIVE_NOT_VALIDATED"
        self.freeze()
        result = self.publish()
        self.assertEqual(result["evidence_schema"], self.manifest["schema"])
        self.assertTrue(result["no_scientific_validation"])
        self.assertEqual(self.cloud.attempts[-1], self.destination+"/manifest.json")

    def test_biological_evaluation_cannot_publish_as_validated(self):
        self.manifest["schema"] = "r02_biological_evaluation_v1"
        self.manifest["status"] = "VALIDATED"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Unsupported schema"):
            self.publish()
        self.assertFalse(self.cloud.attempts)

    def test_catalogue_correspondence_not_genotype_support(self):
        self.manifest["schema"] = "r02_lai_catalogue_correspondence_v1"
        self.manifest["status"] = "COMPLETE_CATALOGUE_CORRESPONDENCE_ONLY_NOT_SUPPORT"
        self.freeze()
        result = self.publish()
        self.assertEqual(result["evidence_schema"], self.manifest["schema"])
        self.assertTrue(result["no_scientific_validation"])

    def test_genotype_audit_not_biological_or_phase_validation(self):
        self.manifest["schema"] = "r02_lai_genotype_support_v1"
        self.manifest["status"] = "COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION"
        self.freeze()
        result = self.publish()
        self.assertEqual(result["evidence_schema"], self.manifest["schema"])
        self.assertTrue(result["no_scientific_validation"])

    def test_genotype_audit_cannot_publish_as_validated_support(self):
        self.manifest["schema"] = "r02_lai_genotype_support_v1"
        self.manifest["status"] = "VALIDATED_SUPPORT"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Unsupported schema"):
            self.publish()
        self.assertFalse(self.cloud.attempts)

    def test_catalogue_cannot_publish_as_completed_support(self):
        self.manifest["schema"] = "r02_lai_catalogue_correspondence_v1"
        self.manifest["status"] = "COMPLETE_SUPPORT"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Unsupported schema"):
            self.publish()

    def test_community_summary_cannot_be_published_as_validated(self):
        self.manifest["schema"] = "r02_community_summary_v1"
        self.manifest["status"] = "VALIDATED"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Unsupported schema"):
            self.publish()
        self.assertFalse(self.cloud.attempts)

    def test_equivalent_http412_reuses_without_replacing_generation(self):
        self.cloud.add(self.destination + "/data.tsv", (self.output / "data.tsv").read_bytes())
        generation = self.cloud.objects[self.destination + "/data.tsv"]["generation"]
        result = self.publish()
        row = result["files"][0]
        self.assertEqual(row["operation"], "REUSED_EQUIVALENT_HTTP_412")
        self.assertEqual(row["generation"], generation)

    def test_all_existing_equivalent_objects_can_resume_with_new_receipt(self):
        first = self.publish()
        second = self.publish(receipt=self.root / "resumed.json")
        self.assertEqual([r["generation"] for r in first["files"]], [r["generation"] for r in second["files"]])
        self.assertTrue(all(r["operation"] == "REUSED_EQUIVALENT_HTTP_412" for r in second["files"]))

    def test_existing_object_with_different_bytes_is_never_overwritten(self):
        uri = self.destination + "/data.tsv"
        self.cloud.add(uri, b"different")
        with self.assertRaisesRegex(ValueError, "GCS size|GCS MD5"):
            self.publish()
        self.assertEqual(self.cloud.objects[uri]["data"], b"different")
        self.assertNotIn(self.destination + "/manifest.json", self.cloud.objects)
        self.assertFalse(self.receipt.exists())

    def test_non412_cloud_failure_is_not_equivalent_success(self):
        self.cloud.add(self.destination + "/data.tsv", (self.output / "data.tsv").read_bytes())
        self.cloud.failure = "HTTPError 403: Forbidden"
        with self.assertRaises(subprocess.CalledProcessError):
            self.publish()
        self.assertFalse(self.receipt.exists())
        self.assertEqual(self.cloud.describes, [])

    def test_remote_integrity_and_generation_are_required(self):
        for field, value in (("size", "999"), ("md5_hash", "wrong"), ("generation", ""), ("generation", True)):
            with self.subTest(field=field, value=value):
                self.cloud.objects.clear()
                self.cloud.after_upload = lambda source, uri: self.cloud.objects[uri].update({field: value})
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse(self.receipt.exists())

    def test_remote_generation_change_stops_before_manifest(self):
        def change_after_second_upload(source, uri):
            if uri.endswith("/progress.json"):
                self.cloud.objects[self.destination + "/data.tsv"]["generation"] = "9999"
        self.cloud.after_upload = change_after_second_upload
        with self.assertRaisesRegex(ValueError, "generation changed"):
            self.publish()
        self.assertNotIn(self.destination + "/manifest.json", self.cloud.objects)
        self.assertFalse(self.receipt.exists())

    def test_recheck_detects_local_mutation_during_upload(self):
        self.cloud.after_upload = lambda source, uri: Path(source).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Local source changed"):
            self.publish()
        self.assertNotIn(self.destination + "/manifest.json", self.cloud.objects)
        self.assertFalse(self.receipt.exists())

    def test_earlier_local_mutation_detected_before_manifest(self):
        def mutate_earlier(source, uri):
            if uri.endswith("/progress.json"):
                (self.output / "data.tsv").write_text("tampered")
        self.cloud.after_upload = mutate_earlier
        with self.assertRaisesRegex(ValueError, "Local source changed|Output SHA256 mismatch"):
            self.publish()
        self.assertNotIn(self.destination + "/manifest.json", self.cloud.objects)

    def test_new_unlisted_file_stops_before_manifest(self):
        def add_file(source, uri):
            if uri.endswith("/progress.json"):
                (self.output / "unexpected.tsv").write_text("unlisted")
        self.cloud.after_upload = add_file
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.publish()
        self.assertNotIn(self.destination + "/manifest.json", self.cloud.objects)

    def test_stale_output_or_manifest_fails_before_any_cloud_action(self):
        (self.output / "data.tsv").write_text("stale")
        with self.assertRaisesRegex(ValueError, "Output SHA256"):
            self.publish()
        with self.assertRaisesRegex(ValueError, "Manifest SHA256"):
            self.publish(expected_manifest_sha256="0" * 64)
        self.assertEqual(self.cloud.attempts, [])

    def test_unsafe_relative_names_rejected_before_cloud(self):
        for name in ("../escape", "/absolute", "a/../escape", "a//b", "a/./b", "gs://elsewhere/x", "x\\y", "-flag", "a#1", "a?x"):
            with self.subTest(name=name):
                self.manifest["outputs_sha256"] = {name: "a" * 64}
                self.freeze()
                with self.assertRaisesRegex(ValueError, "Unsafe relative"):
                    self.publish()
        self.assertEqual(self.cloud.attempts, [])

    def test_unlisted_missing_or_excluded_declared_file_rejected(self):
        (self.output / "extra.txt").write_text("unlisted")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.publish()
        (self.output / "extra.txt").unlink()
        (self.output / "data.tsv").unlink()
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.publish()
        (self.output / "data.tsv").write_text("x\ty\n1\t2\n")
        (self.output / "cache.sqlite").write_bytes(b"cache")
        self.manifest["outputs_sha256"]["cache.sqlite"] = p.pipeline.sha(self.output / "cache.sqlite")
        self.freeze()
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.publish()
        self.assertEqual(self.cloud.attempts, [])

    def test_symlink_output_is_not_followed(self):
        target = self.root / "other.tsv"
        target.write_bytes((self.output / "data.tsv").read_bytes())
        (self.output / "data.tsv").unlink()
        (self.output / "data.tsv").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.publish()
        self.assertEqual(self.cloud.attempts, [])

    def test_destination_restriction_and_traversal(self):
        for destination in ("gs://other/x", p.PREFIX.rstrip("/"), p.PREFIX + "run", p.PREFIX + "../run/part",
                            p.PREFIX + "run/part//", p.PREFIX + "run/part#123", p.PREFIX + "run/part?x"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.publish(destination=destination)
        self.assertEqual(self.cloud.attempts, [])

    def test_receipt_is_new_and_outside_output(self):
        self.receipt.write_text("prior receipt")
        with self.assertRaisesRegex(ValueError, "Receipt already exists"):
            self.publish()
        self.assertEqual(self.receipt.read_text(), "prior receipt")
        self.assertEqual(self.cloud.attempts, [])
        with self.assertRaisesRegex(ValueError, "outside immutable"):
            self.publish(receipt=self.output / "receipt.json")

    def test_receipt_race_never_overwrites_and_cleans_only_owned_temporary(self):
        def competing_receipt(source, uri):
            if uri.endswith("/manifest.json"):
                self.receipt.write_text("other writer")
        self.cloud.after_upload = competing_receipt
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(self.receipt.read_text(), "other writer")
        self.assertEqual(list(self.root.glob(".r02-publication-*")), [])

    def test_receipt_write_failure_leaves_no_completion_receipt(self):
        with mock.patch.object(p.os, "fsync", side_effect=OSError("simulated disk error")):
            with self.assertRaisesRegex(OSError, "simulated disk"):
                self.publish()
        self.assertFalse(self.receipt.exists())
        self.assertEqual(list(self.root.glob(".r02-publication-*")), [])

    def test_only_supported_complete_evidence_is_accepted(self):
        for schema, status in (("r02_segment_preflight_v1", "PREFLIGHT_GEOMETRY_ONLY_NOT_EVIDENCE"),
                               ("r02_segment_evidence_v1", "RUNNING"), ("other", "COMPLETE_DESCRIPTIVE_NOT_VALIDATED")):
            self.manifest.update(schema=schema, status=status)
            self.freeze()
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                self.publish()
        self.assertEqual(self.cloud.attempts, [])

    def test_community_status_and_optional_operational_inventory(self):
        self.manifest.update(schema="r02_community_evidence_v1",
                             status="COMPLETE_DESCRIPTIVE_COMMUNITY_EVIDENCE_NOT_VALIDATED")
        self.manifest["outputs_sha256"].update(self.manifest.pop("operational_outputs_sha256"))
        self.freeze()
        result = self.publish(destination=self.destination + "/")
        self.assertEqual(result["evidence_schema"], "r02_community_evidence_v1")

    def test_duplicate_json_keys_and_inventory_overlap_rejected(self):
        raw = json.dumps(self.manifest).replace('"schema":', '"schema":"other","schema":', 1)
        (self.output / "manifest.json").write_text(raw)
        self.expected = p.pipeline.sha(self.output / "manifest.json")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.publish()
        self.manifest["operational_outputs_sha256"]["data.tsv"] = self.manifest["outputs_sha256"]["data.tsv"]
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.publish()
        self.assertEqual(self.cloud.attempts, [])

    def test_safe_nested_output_names_preserved(self):
        (self.output / "tables").mkdir()
        (self.output / "data.tsv").rename(self.output / "tables/data.tsv")
        self.manifest["outputs_sha256"] = {"tables/data.tsv": p.pipeline.sha(self.output / "tables/data.tsv")}
        self.freeze()
        result = self.publish()
        self.assertIn(self.destination + "/tables/data.tsv", {r["uri"] for r in result["files"]})

    def test_cli_receipt_and_no_overwrite(self):
        argv = ["--output-dir", str(self.output), "--expected-manifest-sha256", self.expected,
                "--destination", self.destination, "--receipt", str(self.receipt)]
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(p.main(argv), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "PUBLISHED_VERIFIED")
        attempts = len(self.cloud.attempts)
        with redirect_stderr(io.StringIO()) as error:
            self.assertEqual(p.main(argv), 2)
        self.assertIn("never overwrite", error.getvalue())
        self.assertEqual(len(self.cloud.attempts), attempts)


if __name__ == "__main__":
    unittest.main()
