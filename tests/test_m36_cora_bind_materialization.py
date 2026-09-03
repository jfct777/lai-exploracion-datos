from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "bin" / "m36_cora_bind_materialization.py"
SPEC = importlib.util.spec_from_file_location("m36_cora_bind_materialization", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BINDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BINDER)


EXECUTED_MANIFEST = (
    b"gnomix_ancestry\tsegment_file\n"
    b"AFR\tanc1.gapfilled_ibd\n"
    b"EUR\tanc2.gapfilled_ibd\n"
    b"NAM\tanc3.gapfilled_ibd\n"
)
CORRECTED_MANIFEST = (
    b"gnomix_ancestry\tsegment_file\n"
    b"EUR\tanc1.gapfilled_ibd\n"
    b"NAM\tanc2.gapfilled_ibd\n"
    b"AFR\tanc3.gapfilled_ibd\n"
)
LOCUS_SENTINEL = b"chrom\tposition\tref\talt\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def target_payload(ratio: int) -> bytes:
    source = "asibd_refined_ibd_gnomix_stratified_exploratory"
    rows = [
        ("A", "B", "outside_chr22_total", source, "between_component", "1", "1", "0.69314718056"),
        ("C", "D", "outside_chr22_total", source, "within_component", "2", "1", "1.09861228867"),
    ]
    between_zeros = [("E", "F"), ("G", "H"), ("I", "J"), ("K", "L"), ("M", "N")]
    rows.extend(
        (left, right, "outside_chr22_total", source, "between_component", "0", "0", "0")
        for left, right in between_zeros[:ratio]
    )
    rows.append(("O", "P", "outside_chr22_total", source, "within_component", "0", "0", "0"))
    lines = ["\t".join(BINDER.TARGET_FIELDS), *("\t".join(row) for row in rows)]
    return ("\n".join(lines) + "\n").encode("utf-8")


class BinderFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.out = root / "bound" / "m36_cora_bound_provenance.json"
        self.executed_manifest = root / "executed_manifest.tab"
        self.corrected_manifest = root / "corrected_manifest.tab"
        self.locus_sentinel = root / "locus_metadata.tsv"
        self.executed_code = ROOT / "bin" / "m36_cora_materialize.py"
        self.materialization_receipt = root / "m36_cora_materialization_receipt.json"
        self.published_descriptors = root / "published_descriptors.json"
        self.invariance_proof = root / "invariance_proof.json"

        self.executed_manifest.write_bytes(EXECUTED_MANIFEST)
        self.corrected_manifest.write_bytes(CORRECTED_MANIFEST)
        self.locus_sentinel.write_bytes(LOCUS_SENTINEL)

        self.remote_payloads = {
            "loci": b"fixture:loci\n",
            "carriers": b"fixture:carriers\n",
            "missing": b"fixture:missing\n",
            "covariates": b"fixture:covariates\n",
            "components": b"fixture:components\n",
            "targets": target_payload(1),
            "targets_zero3": target_payload(3),
            "targets_zero5": target_payload(5),
        }
        base_hashes = {name: digest(self.remote_payloads[name]) for name in BINDER.RECEIPT_ARTIFACTS}
        zero_negative_sampling = {
            str(ratio): {
                "between_component": {
                    "positive": 1, "zero": ratio,
                    "requested_zero_to_positive_ratio": ratio,
                    "achieved_zero_to_positive_ratio": float(ratio),
                    "zero_universe_saturated": False,
                },
                "within_component": {
                    "positive": 1, "zero": 1,
                    "requested_zero_to_positive_ratio": ratio,
                    "achieved_zero_to_positive_ratio": 1.0,
                    "zero_universe_saturated": ratio != 1,
                },
            }
            for ratio in (1, 3, 5)
        }
        self.receipt = {
            "stage": "M36_CORA_MATERIALIZE",
            "status": "MATERIALIZED_PASS",
            "synthetic": False,
            "feature_schema": "m36_factorized_sparse_v1",
            "external_target_schema": "m36_external_common_pairs_log1p_v3_pair_total",
            "zero_negative_ratio_sensitivity": [1, 3, 5],
            "zero_negative_sampling": zero_negative_sampling,
            "input_descriptors": {
                name: {
                    "uri": BINDER.BASE_FILENAMES[name],
                    "generation": "LOCAL_CHAIN",
                    "sha256": base_hashes[name],
                }
                for name in BINDER.RECEIPT_ARTIFACTS
            },
            "source_input_descriptors": {
                "asibd_manifest": {
                    "uri": "m36_cora_asibd_manifest.tab",
                    "sha256": digest(EXECUTED_MANIFEST),
                }
            },
        }
        self._write_json(self.materialization_receipt, self.receipt)
        self.remote_payloads["materialization_receipt"] = self.materialization_receipt.read_bytes()

        self.descriptors = {}
        for index, (name, filename) in enumerate(BINDER.PUBLISHED_FILENAMES.items(), start=1):
            payload = self.remote_payloads[name]
            self.descriptors[name] = {
                "uri": BINDER.EXPECTED_PUBLISHED_PREFIX + filename,
                "generation": str(1788397908000000 + index),
                "size_bytes": len(payload),
                "crc32c_base64": BINDER.crc32c_base64(payload),
                "sha256": digest(payload),
            }
        self._write_json(self.published_descriptors, self.descriptors)
        self.expected_published_metadata = {
            name: {
                field: descriptor[field]
                for field in ("generation", "size_bytes", "crc32c_base64", "sha256")
            }
            for name, descriptor in self.descriptors.items()
        }

        target_for_ratio = {"1": "targets", "3": "targets_zero3", "5": "targets_zero5"}
        self.proof = {
            "schema_version": "1.0.0",
            "stage": "M36_CORA_MANIFEST_INVARIANCE_PROOF",
            "status": "PASS_EXACT_INVARIANCE",
            "run_id": BINDER.RUN_ID,
            "method": "deterministic_synthetic_regression",
            "executed_manifest_sha256": digest(EXECUTED_MANIFEST),
            "corrected_manifest_sha256": digest(CORRECTED_MANIFEST),
            "executed_code_sha256": digest(self.executed_code.read_bytes()),
            "locus_sentinel_sha256": digest(LOCUS_SENTINEL),
            "invariant_field": "segment_file",
            "label_field": "gnomix_ancestry",
            "sensitivity_ratios": [1, 3, 5],
            "ratio_comparisons": {
                ratio: {
                    "exact_equal": True,
                    "comparison_sha256": comparison_hash,
                    "published_sha256": self.descriptors[target_for_ratio[ratio]]["sha256"],
                }
                for ratio, comparison_hash in BINDER.EXPECTED_INVARIANCE_COMPARISON_SHA256.items()
            },
        }
        self._write_json(self.invariance_proof, self.proof)
        self.gcloud_calls: list[list[str]] = []
        self.remote_metadata_overrides: dict[str, dict[str, object]] = {}
        self.extra_inventory_uri: str | None = None

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_descriptors(self) -> None:
        self._write_json(self.published_descriptors, self.descriptors)

    def write_proof(self) -> None:
        self._write_json(self.invariance_proof, self.proof)

    def fake_gcloud_run(self, command, **kwargs):
        del kwargs
        self.gcloud_calls.append(list(command))
        if command[:3] == ["gcloud", "storage", "ls"]:
            entries = []
            for descriptor in self.descriptors.values():
                bucket_and_name = descriptor["uri"].removeprefix("gs://")
                bucket, name = bucket_and_name.split("/", 1)
                entries.append({"type": "cloud_object", "metadata": {"bucket": bucket, "name": name}})
            if self.extra_inventory_uri is not None:
                bucket_and_name = self.extra_inventory_uri.removeprefix("gs://")
                bucket, name = bucket_and_name.split("/", 1)
                entries.append({"type": "cloud_object", "metadata": {"bucket": bucket, "name": name}})
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(entries), stderr="")
        if command[:4] == ["gcloud", "storage", "objects", "describe"]:
            uri = command[4]
            name, descriptor = next(
                (name, item) for name, item in self.descriptors.items() if item["uri"] == uri
            )
            metadata = {
                "generation": descriptor["generation"],
                "size": str(descriptor["size_bytes"]),
                "crc32c_hash": descriptor["crc32c_base64"],
            }
            metadata.update(self.remote_metadata_overrides.get(name, {}))
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(metadata), stderr="")
        if command[:3] == ["gcloud", "storage", "cp"]:
            versioned_uri, destination = command[3], Path(command[4])
            uri, generation = versioned_uri.rsplit("#", 1)
            name, descriptor = next(
                (name, item) for name, item in self.descriptors.items() if item["uri"] == uri
            )
            if generation != descriptor["generation"]:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="generation mismatch")
            destination.write_bytes(self.remote_payloads[name])
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    def bind(self):
        with (
            mock.patch.object(BINDER, "EXPECTED_PUBLISHED_METADATA", self.expected_published_metadata),
            mock.patch.object(BINDER.subprocess, "run", side_effect=self.fake_gcloud_run),
        ):
            return BINDER.bind_materialization(
                self.materialization_receipt,
                self.published_descriptors,
                self.executed_manifest,
                self.corrected_manifest,
                self.invariance_proof,
                self.locus_sentinel,
                self.executed_code,
                self.out,
            )

    def input_paths(self) -> list[Path]:
        return [
            self.materialization_receipt,
            self.published_descriptors,
            self.executed_manifest,
            self.corrected_manifest,
            self.invariance_proof,
            self.locus_sentinel,
            self.executed_code,
        ]


class TestM36CoraBindMaterialization(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.fixture = BinderFixture(Path(self.temporary_directory.name))

    def test_pass_binds_seven_base_hashes_and_two_sensitivity_objects(self) -> None:
        before = {path: digest(path.read_bytes()) for path in self.fixture.input_paths()}

        observed = self.fixture.bind()

        self.assertEqual(observed["status"], "PASS_BOUND_PROVENANCE")
        self.assertEqual(observed["artifact_role"], "NON_CONSUMABLE_PROVENANCE_ADDENDUM")
        self.assertIs(observed["consumable_as_materialization_receipt"], False)
        self.assertEqual(observed["published_binding"]["base_hash_count"], 7)
        self.assertEqual(
            set(observed["published_binding"]["sensitivity_descriptors"]),
            {"targets_zero3", "targets_zero5"},
        )
        self.assertEqual(json.loads(self.fixture.out.read_text(encoding="utf-8")), observed)
        self.assertEqual(before, {path: digest(path.read_bytes()) for path in self.fixture.input_paths()})
        self.assertEqual(self.fixture.receipt["status"], "MATERIALIZED_PASS")
        self.assertEqual(set(self.fixture.receipt["input_descriptors"]), set(BINDER.RECEIPT_ARTIFACTS))
        copy_calls = [command for command in self.fixture.gcloud_calls if command[2] == "cp"]
        self.assertEqual(len(copy_calls), 9)
        self.assertTrue(all(command[3].startswith("gs://") and "#" in command[3]
                            and not str(command[4]).startswith("gs://") for command in copy_calls))

    def test_production_contract_is_anchored_to_the_audited_run(self) -> None:
        self.assertEqual(
            BINDER.EXPECTED_PUBLISHED_PREFIX,
            "gs://teams-usp/frank/lai-exploracion-datos/runs/"
            "m36-cora-chr22-materialize-20260903a/m36_cora_set/",
        )
        self.assertEqual(len(BINDER.EXPECTED_PUBLISHED_METADATA), 9)
        self.assertEqual(
            BINDER.EXPECTED_PUBLISHED_METADATA["materialization_receipt"]["sha256"],
            "e3445e3eda666dc0f717ec0ab290001f22e6c0d57d2013a6c68105d696dc8a5f",
        )
        self.assertEqual(
            BINDER.EXPECTED_PUBLISHED_METADATA["targets_zero5"]["sha256"],
            "8dad1b6e290698d9eef6d7781497b7531c7f6a3e5197d3c49d34749ce0fc3a2f",
        )

    def test_base_hash_mismatch_is_rejected_without_output(self) -> None:
        self.fixture.descriptors["loci"]["sha256"] = "f" * 64
        self.fixture.write_descriptors()

        with self.assertRaisesRegex(BINDER.BindingError, "loci sha256 differs"):
            self.fixture.bind()

        self.assertFalse(self.fixture.out.exists())

    def test_missing_sensitivity_descriptor_is_rejected_without_output(self) -> None:
        del self.fixture.descriptors["targets_zero5"]
        self.fixture.write_descriptors()

        with self.assertRaisesRegex(BINDER.BindingError, "missing required sensitivity"):
            self.fixture.bind()

        self.assertFalse(self.fixture.out.exists())

    def test_existing_output_is_never_overwritten_and_is_checked_first(self) -> None:
        self.fixture.bind()
        original_output = self.fixture.out.read_bytes()
        self.fixture.executed_manifest.unlink()

        with self.assertRaisesRegex(BINDER.BindingError, "refusing to overwrite"):
            self.fixture.bind()

        self.assertEqual(self.fixture.out.read_bytes(), original_output)

    def test_proof_must_authenticate_the_executed_code(self) -> None:
        self.fixture.proof["executed_code_sha256"] = "0" * 64
        self.fixture.write_proof()

        with self.assertRaisesRegex(BINDER.BindingError, "does not authenticate executed_code"):
            self.fixture.bind()

        self.assertFalse(self.fixture.out.exists())

    def test_locus_sentinel_must_be_exactly_header_only(self) -> None:
        self.fixture.locus_sentinel.write_bytes(LOCUS_SENTINEL + b"chr22\t1\tA\tG\n")

        with self.assertRaisesRegex(BINDER.BindingError, "header-only"):
            self.fixture.bind()

        self.assertFalse(self.fixture.out.exists())

    def test_reopened_object_hash_mismatch_is_rejected(self) -> None:
        original = self.fixture.remote_payloads["components"]
        self.fixture.remote_payloads["components"] = original[:-1] + bytes([original[-1] ^ 1])

        with self.assertRaisesRegex(BINDER.BindingError, "reopened SHA-256 differs for components"):
            self.fixture.bind()

        self.assertFalse(self.fixture.out.exists())

    def test_remote_inventory_and_crc_are_not_caller_assertions(self) -> None:
        self.fixture.extra_inventory_uri = BINDER.EXPECTED_PUBLISHED_PREFIX + "unexpected.tsv"
        with self.assertRaisesRegex(BINDER.BindingError, "inventory differs"):
            self.fixture.bind()
        self.assertFalse(self.fixture.out.exists())

        self.fixture.extra_inventory_uri = None
        self.fixture.remote_metadata_overrides["components"] = {"crc32c_hash": "/////w=="}
        with self.assertRaisesRegex(BINDER.BindingError, "remote CRC32C differs for components"):
            self.fixture.bind()
        self.assertFalse(self.fixture.out.exists())

    def test_sensitivity_positive_rows_must_be_identical(self) -> None:
        payloads = {
            "targets": target_payload(1),
            "targets_zero3": target_payload(3).replace(b"\t2\t1\t1.09861228867", b"\t3\t1\t1.38629436112", 1),
            "targets_zero5": target_payload(5),
        }

        with self.assertRaisesRegex(BINDER.BindingError, "positive target rows differ"):
            BINDER.validate_target_sensitivity(self.fixture.receipt, payloads)


if __name__ == "__main__":
    unittest.main()
