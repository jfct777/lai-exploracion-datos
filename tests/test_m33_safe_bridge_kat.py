#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CORE = load_module("m33_safe_bridge_core", "bin/m33_safe_bridge_core.py")
import sys
sys.modules["m33_safe_bridge_core"] = CORE
RUNNER = load_module("m33_safe_bridge_kat", "bin/m33_safe_bridge_kat.py")
M0 = load_module("m33_m0_contract_for_differential", "bin/m33_m0_contract.py")
CONTRACT = ROOT / "conf/m33_safe_bridge_kat_contract.json"
BASE_CONTRACT = ROOT / "conf/m33_m0_materializer_contract.json"
FIXTURE = ROOT / "tests/fixtures/m33_safe_bridge_minor0_overlap_missing.json"


class CoreTests(unittest.TestCase):
    @staticmethod
    def ref_metadata():
        nodes = list(range(6))
        persons = ["A", "A", "E", "E", "S", "S"]
        ancestries = ["AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA"]
        records = [
            {"node_id": node, "person_id": person, "ancestry": ancestry}
            for node, person, ancestry in zip(nodes, persons, ancestries)
        ]
        return nodes, persons, ancestries, records

    def test_minor_code_zero_is_not_alt_dosage(self) -> None:
        states = np.asarray([[[0, 0], [0, 1], [1, 1]]], dtype=np.int8)
        dosage, observed = CORE.orient_target(states, np.asarray([0, 0, 0], dtype=np.int8))
        np.testing.assert_array_equal(dosage, [[2, 1, 0]])
        np.testing.assert_array_equal(observed, [[1, 1, 1]])

    def test_core_matches_frozen_m0_minor_primitives_exhaustively(self) -> None:
        for minor_code in (0, 1):
            for hap0 in (-1, 0, 1):
                for hap1 in (-1, 0, 1):
                    dosage, observed = CORE.orient_target(
                        np.asarray([[[hap0, hap1]]], dtype=np.int8),
                        np.asarray([minor_code], dtype=np.int8))
                    expected = M0.diploid_minor_dosage(
                        None if hap0 == -1 else hap0, None if hap1 == -1 else hap1, minor_code)
                    self.assertEqual((int(dosage[0, 0]), int(observed[0, 0])), expected)
            states = [-1, minor_code]
            nodes, persons, ancestries, records = self.ref_metadata()
            raw = np.asarray([[state] for state in states * 3], dtype=np.int8)
            summary = CORE.summarize_reference(raw, np.asarray([minor_code], dtype=np.int8),
                                               nodes, persons, ancestries, records)
            expected = M0.reference_minor_summary(
                [None if state == -1 else state for state in states], minor_code)
            self.assertEqual(int(summary["minor_ac"][0, 0]), expected["minor_ac"])
            self.assertEqual(int(summary["callable_an"][0, 0]), expected["callable_an"])
            self.assertAlmostEqual(float(summary["minor_af"][0, 0]), expected["minor_af"])

    def test_missing_target_and_reference_are_explicit(self) -> None:
        dosage, observed = CORE.orient_target(np.asarray([[[-1, 1]]], dtype=np.int8),
                                               np.asarray([1], dtype=np.int8))
        self.assertEqual((int(dosage[0, 0]), int(observed[0, 0])), (0, 0))
        nodes, persons, ancestries, records = self.ref_metadata()
        summary = CORE.summarize_reference(
            np.asarray([[0], [1], [-1], [-1], [1], [1]], dtype=np.int8),
            np.asarray([1], dtype=np.int8), nodes, persons, ancestries, records)
        np.testing.assert_array_equal(summary["callable_an"][:, 0], [2, 0, 2])
        np.testing.assert_array_equal(summary["observed_mask"][:, 0], [1, 0, 1])

    def test_role_firewall_requires_exact_set(self) -> None:
        states = np.zeros((6, 1), dtype=np.int8)
        nodes, persons, ancestries, records = self.ref_metadata()
        with self.assertRaisesRegex(ValueError, "exactly"):
            CORE.summarize_reference(states, np.asarray([0], dtype=np.int8),
                                     [0, 1, 2, 3, 4, 9], persons, ancestries, records)
        with self.assertRaisesRegex(ValueError, "exactly"):
            poisoned_ancestry = list(ancestries)
            poisoned_ancestry[-1] = "AFR"
            CORE.summarize_reference(states, np.asarray([0], dtype=np.int8),
                                     nodes, persons, poisoned_ancestry, records)

    def test_overlap_and_allele_mismatch(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        catalog = [{key: record[key] for key in CORE.LOCUS_FIELDS}
                   for record in fixture["rare_catalog_records"]]
        incremental, overlap = CORE.partition_incremental(
            fixture["selected_loci"], catalog, fixture["flare_loci"])
        self.assertEqual([row[1] for row in incremental], [100, 300])
        self.assertEqual([row[1] for row in overlap], [200])
        mismatch = copy.deepcopy(fixture["flare_loci"])
        mismatch[0]["ref"], mismatch[0]["alt"] = "T", "G"
        with self.assertRaisesRegex(ValueError, "REF/ALT mismatch"):
            CORE.partition_incremental(fixture["selected_loci"], catalog, mismatch)

    def test_f0_rejects_invalid_simplex(self) -> None:
        valid = np.asarray([[[[0.7, 0.2, 0.1]], [[0.1, 0.2, 0.7]]]])
        result = CORE.sanitize_f0(valid)
        self.assertEqual(result.dtype, np.dtype("<f4"))
        invalid = valid.copy()
        invalid[0, 0, 0] = [0.2, 0.2, 0.2]
        with self.assertRaisesRegex(ValueError, "outside tolerance"):
            CORE.sanitize_f0(invalid)

    def test_f0_named_fields_define_haplotype_axis_and_reject_poison(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        flare = CORE.canonical_loci(fixture["flare_loci"])
        parsed = CORE.parse_f0_records(fixture["f0_records"], fixture["target_sample_ids"],
                                       flare, fixture["root_seed"])
        np.testing.assert_allclose(parsed[0, 0, 0], [0.7, 0.2, 0.1])
        np.testing.assert_allclose(parsed[0, 1, 0], [0.1, 0.2, 0.7])
        poisoned = copy.deepcopy(fixture["f0_records"])
        poisoned[0]["GT"] = "0|1"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            CORE.parse_f0_records(poisoned, fixture["target_sample_ids"], flare, fixture["root_seed"])
        wrong_axis = copy.deepcopy(fixture["f0_records"])
        wrong_axis[0]["sample_id"] = "NOT_TARGET"
        with self.assertRaisesRegex(ValueError, "outside authenticated axes"):
            CORE.parse_f0_records(wrong_axis, fixture["target_sample_ids"], flare, fixture["root_seed"])


class RunnerTests(unittest.TestCase):
    def test_kat_outputs_are_deterministic_non_consumable_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            output1, output2 = Path(first) / "out", Path(second) / "out"
            receipt1 = RUNNER.run(FIXTURE, CONTRACT, BASE_CONTRACT, output1)
            receipt2 = RUNNER.run(FIXTURE, CONTRACT, BASE_CONTRACT, output2)
            self.assertEqual(receipt1["status"], RUNNER.STATUS)
            self.assertFalse(receipt1["scientific_evidence"])
            self.assertFalse(receipt1["consumable"])
            self.assertFalse(receipt1["ready_emitted"])
            self.assertEqual(receipt1["selected_overlap_count"], 1)
            self.assertGreater(receipt1["target_missing_cells"], 0)
            self.assertGreater(receipt1["reference_missing_alleles"], 0)
            self.assertEqual(receipt1["artifact_raw_sha256"], receipt2["artifact_raw_sha256"])
            self.assertFalse((output1 / "READY").exists())
            with np.load(output1 / "kat_target_rare_diploid_incremental.npz", allow_pickle=False) as target:
                np.testing.assert_array_equal(target["minor_dosage"], [[2, 2], [1, 0]])
                np.testing.assert_array_equal(target["observed_mask"], [[1, 1], [1, 0]])
            with np.load(output1 / "kat_reference_rare_summary_incremental.npz",
                         allow_pickle=False) as reference:
                np.testing.assert_array_equal(reference["minor_ac"][:, 0], [2, 1, 0])
                np.testing.assert_array_equal(reference["callable_an"][:, 0], [2, 2, 2])
                np.testing.assert_allclose(reference["minor_af"][:, 0], [1, 1 / 2, 0])
                np.testing.assert_array_equal(reference["no_support"][:, 0], [0, 0, 1])
            self.assertEqual(receipt1["expected_ref_node_count"], 6)
            self.assertEqual(receipt1["contributing_ref_node_count"], 6)
            self.assertEqual(receipt1["rejected_non_ref_node_count"], 0)

    def test_poisoned_ref_node_stops_before_output(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        fixture["ref_state_records"][-1]["node_id"] = 999
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poisoned = root / "poison.json"
            poisoned.write_text(json.dumps(fixture))
            output = root / "out"
            with self.assertRaisesRegex(ValueError, "outside authenticated axes"):
                RUNNER.run(poisoned, CONTRACT, BASE_CONTRACT, output)
            self.assertFalse(output.exists())

    def test_numeric_coercions_all_stop_before_output(self) -> None:
        mutations = []
        fixture = json.loads(FIXTURE.read_text())
        changed = copy.deepcopy(fixture)
        changed["rare_catalog_records"][0]["minor_code"] = True
        mutations.append((changed, "minor code"))
        changed = copy.deepcopy(fixture)
        changed["target_haplotype_records"][0]["state"] = 0.9
        mutations.append((changed, "TARGET state"))
        changed = copy.deepcopy(fixture)
        changed["ref_state_records"][0]["state"] = 0.9
        mutations.append((changed, "REF state"))
        changed = copy.deepcopy(fixture)
        changed["expected_ref_records"][0]["node_id"] = 10.9
        mutations.append((changed, "fields or types differ"))
        changed = copy.deepcopy(fixture)
        changed["f0_records"][0]["ANP1"][0] = "0.3"
        mutations.append((changed, "non-numeric"))
        for index, (payload, message) in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                poisoned = root / "poison.json"
                poisoned.write_text(json.dumps(payload))
                output = root / "out"
                with self.assertRaisesRegex(ValueError, message):
                    RUNNER.run(poisoned, CONTRACT, BASE_CONTRACT, output)
                self.assertFalse(output.exists())

    def test_duplicate_locus_minor_binding_and_f0_record_all_stop(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        mutations = []
        changed = copy.deepcopy(fixture)
        changed["selected_loci"].append(copy.deepcopy(changed["selected_loci"][0]))
        mutations.append((changed, "duplicate"))
        changed = copy.deepcopy(fixture)
        changed["rare_catalog_records"].append(copy.deepcopy(changed["rare_catalog_records"][0]))
        mutations.append((changed, "duplicated"))
        changed = copy.deepcopy(fixture)
        changed["f0_records"].append(copy.deepcopy(changed["f0_records"][0]))
        mutations.append((changed, "duplicated"))
        for index, (payload, message) in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                poisoned = root / "poison.json"
                poisoned.write_text(json.dumps(payload))
                output = root / "out"
                with self.assertRaisesRegex(ValueError, message):
                    RUNNER.run(poisoned, CONTRACT, BASE_CONTRACT, output)
                self.assertFalse(output.exists())

    def test_genetic_map_and_nominal_genotype_axes_are_bound(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        mutations = []
        changed = copy.deepcopy(fixture)
        changed["selected_loci"][0]["cM"] = 9.9
        mutations.append((changed, "genetic map"))
        changed = copy.deepcopy(fixture)
        changed["target_haplotype_records"][0]["locus_id"] = 999
        mutations.append((changed, "outside authenticated axes"))
        changed = copy.deepcopy(fixture)
        changed["ref_state_records"][0]["locus_id"] = 999
        mutations.append((changed, "outside authenticated axes"))
        for index, (payload, message) in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                poisoned = root / "poison.json"
                poisoned.write_text(json.dumps(payload))
                output = root / "out"
                with self.assertRaisesRegex(ValueError, message):
                    RUNNER.run(poisoned, CONTRACT, BASE_CONTRACT, output)
                self.assertFalse(output.exists())

    def test_f0_float_identity_and_full_contract_mutation_stop(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        for field in ("chrom", "pos"):
            changed = copy.deepcopy(fixture)
            changed["f0_records"][0][field] = float(changed["f0_records"][0][field])
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                poisoned = root / "poison.json"
                poisoned.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "identity field type"):
                    RUNNER.run(poisoned, CONTRACT, BASE_CONTRACT, root / "out")
        contract = json.loads(CONTRACT.read_text())
        contract["stop_rules"] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poisoned = root / "contract.json"
            poisoned.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "raw hash drifted"):
                RUNNER.run(FIXTURE, poisoned, BASE_CONTRACT, root / "out")

    def test_existing_or_symlink_output_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "must be new"):
                RUNNER.run(FIXTURE, CONTRACT, BASE_CONTRACT, existing)
            link = root / "link"
            link.symlink_to(existing, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must be new"):
                RUNNER.run(FIXTURE, CONTRACT, BASE_CONTRACT, link)

    def test_base_contract_drift_stops_before_output(self) -> None:
        base = json.loads(BASE_CONTRACT.read_text())
        base["execution_authorization"]["safe_bridge"] = True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed = root / "base.json"
            changed.write_text(json.dumps(base))
            output = root / "out"
            with self.assertRaisesRegex(ValueError, "hash drifted"):
                RUNNER.run(FIXTURE, CONTRACT, changed, output)
            self.assertFalse(output.exists())

    def test_root_swap_in_sidecar_stops_before_output(self) -> None:
        contract = json.loads(CONTRACT.read_text())
        contract["technical_roots"]["root17"]["flare_tbi_sha256"] = (
            contract["technical_roots"]["root18"]["flare_tbi_sha256"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed = root / "contract.json"
            changed.write_text(json.dumps(contract))
            output = root / "out"
            with self.assertRaisesRegex(ValueError, "raw hash drifted"):
                RUNNER.run(FIXTURE, changed, BASE_CONTRACT, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
