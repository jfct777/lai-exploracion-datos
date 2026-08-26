from __future__ import annotations

import contextlib
import copy
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "conf" / "m34_nam_assets.json"
MODULE_PATH = ROOT / "bin" / "m34_validate_nam_assets.py"
SPEC = importlib.util.spec_from_file_location("m34_validate_nam_assets", MODULE_PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def make_rows() -> list[dict[str, str]]:
    aliases = {"AFR": "African", "EUR": "European", "NAM": "Native_American"}
    rows: list[dict[str, str]] = []
    for ancestry in MOD.EXPECTED_ANCESTRIES:
        for role in MOD.EXPECTED_ROLES:
            people = MOD.EXPECTED_PEOPLE[ancestry][role]
            atomic_units = MOD.EXPECTED_ATOMIC_UNITS[ancestry][role]
            populations = MOD.EXPECTED_POPULATIONS[ancestry][role]
            if people == 0:
                assert atomic_units == populations == 0
                continue
            assert people >= populations >= atomic_units > 0
            for index in range(people):
                population_index = index % populations
                unit_index = min(population_index, atomic_units - 1)
                stem = f"{ancestry.lower()}-{role.lower()}"
                rows.append({
                    "sample_id": f"fixture:{stem}:person:{index:04d}",
                    "source": "LOCAL_FIXTURE",
                    "ancestry": aliases[ancestry],
                    "population": f"population-{stem}-{population_index:03d}",
                    "canonical_population": f"canonical-{stem}-{population_index:03d}",
                    "atomic_unit_id": f"atomic-{stem}-{unit_index:03d}",
                    "role": role,
                    "exclusion_reason": "DISCOVERY_CORE" if role == "DISCOVERY" else "",
                })
    return rows


def write_fixture(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOD.EXPECTED_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def find_row(rows: list[dict[str, str]], ancestry: str, role: str) -> dict[str, str]:
    label = {"AFR": "African", "EUR": "European", "NAM": "Native_American"}[ancestry]
    return next(row for row in rows if row["ancestry"] == label and row["role"] == role)


class TestM34ValidateNamAssets(unittest.TestCase):
    def test_canonical_registry_and_local_fixture_pass(self) -> None:
        registry = MOD.validate_registry(REGISTRY_PATH)
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_fixture(Path(directory) / "roles.tsv", make_rows())
            report = MOD.validate_roles_fixture(fixture, registry)
        self.assertEqual(report["target_rows"], 1505)
        self.assertEqual(report["people"], MOD.EXPECTED_PEOPLE)
        self.assertEqual(report["atomic_units"]["NAM"], {
            "REF_TRAIN": 4,
            "SOURCE_VALID": 2,
            "SOURCE_TEST": 2,
            "DISCOVERY": 44,
        })
        self.assertEqual(report["populations"]["NAM"]["DISCOVERY"], 45)

    def test_non_target_excluded_rows_are_accepted_but_not_counted(self) -> None:
        rows = make_rows()
        rows.append({
            "sample_id": "fixture:excluded:0001",
            "source": "LOCAL_FIXTURE",
            "ancestry": "East_Asian",
            "population": "fixture-east-asian",
            "canonical_population": "East_Asian|fixture-east-asian",
            "atomic_unit_id": "excluded-unit-0001",
            "role": "EXCLUDED",
            "exclusion_reason": "OUTSIDE_TARGET_ANCESTRIES",
        })
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_fixture(Path(directory) / "roles-with-excluded.tsv", rows)
            report = MOD.validate_roles_fixture(fixture, MOD.validate_registry(REGISTRY_PATH))
        self.assertEqual(report["rows_read"], 1506)
        self.assertEqual(report["target_rows"], 1505)

    def test_nam_overlap_between_roles_fails(self) -> None:
        for dimension in ("sample_id", "canonical_population", "atomic_unit_id"):
            with self.subTest(dimension=dimension), tempfile.TemporaryDirectory() as directory:
                rows = make_rows()
                ref_row = find_row(rows, "NAM", "REF_TRAIN")
                valid_row = find_row(rows, "NAM", "SOURCE_VALID")
                valid_row[dimension] = ref_row[dimension]
                fixture = write_fixture(Path(directory) / f"nam-overlap-{dimension}.tsv", rows)
                with self.assertRaisesRegex(MOD.AssetContractError, "appears more than once|crosses roles"):
                    MOD.validate_roles_fixture(fixture, MOD.validate_registry(REGISTRY_PATH))

    def test_partial_hash_is_forbidden(self) -> None:
        cases = (
            ("source_assets", "phased_chr22_vcf"),
            ("derived_destinations", "mosaic_truth"),
        )
        for asset_group, logical_id in cases:
            with self.subTest(asset_group=asset_group, logical_id=logical_id):
                registry = load_registry()
                registry[asset_group][logical_id]["sha256"] = "71161973"
                with self.assertRaisesRegex(MOD.AssetContractError, "SHA-256|unobserved hash"):
                    MOD.validate_registry_payload(registry)

    def test_destination_outside_owned_bucket_fails(self) -> None:
        registry = load_registry()
        registry["derived_destinations"]["mosaic_truth"]["gcs_uri"] = (
            "gs://projects-usp/dnaBr-lai/datalake/refined/m34_mosaic_truth.chr22.tsv.gz"
        )
        with self.assertRaisesRegex(MOD.AssetContractError, "outside owned M34 run prefix"):
            MOD.validate_registry_payload(registry)

    def test_relatedness_must_be_pcrelate_plus_ibd_without_king(self) -> None:
        mutations = (
            {"source_methods": ["PC-Relate", "Refined IBD", "KING"]},
            {"king_used": True},
            {"pcrelate_without_king": False},
            {"source_methods": ["PC-Relate"]},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                registry = load_registry()
                registry["relatedness_contract"].update(mutation)
                with self.assertRaisesRegex(MOD.AssetContractError, "relatedness provenance drift"):
                    MOD.validate_registry_payload(registry)

    def test_wrong_person_count_fails(self) -> None:
        rows = make_rows()
        rows.remove(find_row(rows, "AFR", "SOURCE_TEST"))
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_fixture(Path(directory) / "wrong-count.tsv", rows)
            with self.assertRaisesRegex(MOD.AssetContractError, "person counts"):
                MOD.validate_roles_fixture(fixture, MOD.validate_registry(REGISTRY_PATH))

    def test_ancestry_and_role_schema_drift_fails(self) -> None:
        cases = (
            ("ancestries", ["AFR", "EUR", "ASIA"]),
            ("roles", ["REF_TRAIN", "SOURCE_VALID", "SOURCE_TEST"]),
        )
        for key, value in cases:
            with self.subTest(key=key):
                registry = load_registry()
                registry[key] = value
                with self.assertRaises(MOD.AssetContractError):
                    MOD.validate_registry_payload(registry)

    def test_registry_byte_drift_fails_even_if_json_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reformatted = Path(directory) / "registry.json"
            reformatted.write_text(json.dumps(load_registry(), sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(
                MOD.validate_registry_payload(load_registry())["stage"],
                "M34_NAM_ASSET_REGISTRY",
            )
            with self.assertRaisesRegex(MOD.AssetContractError, "byte hash drift"):
                MOD.validate_registry(reformatted)

    def test_cli_reports_offline_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_fixture(Path(directory) / "roles.tsv", make_rows())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                return_code = MOD.main([
                    "--registry", str(REGISTRY_PATH), "--roles-fixture", str(fixture)
                ])
        self.assertEqual(return_code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], MOD.PASS_STATUS)
        self.assertEqual(report["registry_sha256"], MOD.EXPECTED_REGISTRY_SHA256)
        self.assertEqual(report["target_rows"], 1505)
        self.assertFalse(report["gcs_read"])
        self.assertFalse(report["asset_generation"])
        self.assertFalse(report["training"])

    def test_fixture_input_is_not_mutated_by_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_fixture(Path(directory) / "roles.tsv", make_rows())
            before = fixture.read_bytes()
            MOD.validate_roles_fixture(fixture, MOD.validate_registry(REGISTRY_PATH))
            self.assertEqual(fixture.read_bytes(), before)

    def test_registry_payload_input_is_not_mutated(self) -> None:
        registry = load_registry()
        before = copy.deepcopy(registry)
        MOD.validate_registry_payload(registry)
        self.assertEqual(registry, before)


if __name__ == "__main__":
    unittest.main()
