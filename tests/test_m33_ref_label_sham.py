#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))


def load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


CORE = load("m33_safe_bridge_core", "bin/m33_safe_bridge_core.py")
AUTH = load("m33_ref_label_sham_source_auth", "bin/m33_ref_label_sham_source_auth.py")
KAT = load("m33_ref_label_sham_kat", "bin/m33_ref_label_sham_kat.py")
COMMIT = "a" * 40


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M33ReferenceLabelShamTests(unittest.TestCase):
    def fixture(self):
        return KAT.synthetic_fixture()

    def diploid_fixture(self):
        people = tuple(f"P{index}" for index in range(6))
        labels = ("AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA")
        nodes = {person: (2 * index, 2 * index + 1)
                 for index, person in enumerate(people)}
        dosage = np.asarray([
            [0, 1, 2, 0, 1, 2],
            [2, 2, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 2],
            [1, 0, 2, 1, 2, 0],
        ], dtype="|i1")
        return dosage, people, labels, nodes

    def test_contract_freezes_diagnostic_only_control_and_effective_schema(self):
        contract = KAT.load_contract(ROOT / "conf/m33_ref_label_sham_contract.json")
        self.assertEqual(contract["preregistered_seeds"], list(CORE.REF_LABEL_SHAM_SEEDS))
        self.assertEqual(contract["effective_reference_summary_schema"]["dtypes"]["ancestry"],
                         "|S4")
        self.assertTrue(contract["interpretation"]["diagnostic_only"])
        self.assertFalse(contract["interpretation"]["permutation_p_value"])
        self.assertFalse(contract["scope"]["scientific_evidence"])
        self.assertEqual(contract["scope"]["persisted_outputs"], "receipt_only")
        self.assertEqual(contract["upstream_contracts"], {
            "conf/m33_pre4_preregistration.json": KAT.PRE4_SHA256,
            "conf/m33_m0_materializer_contract.json": KAT.M0_SHA256,
        })

    def test_contract_rejects_any_byte_level_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "contract.json"
            payload = json.loads(
                (ROOT / "conf/m33_ref_label_sham_contract.json").read_text(encoding="utf-8")
            )
            payload["interpretation"]["future_rule"] = "changed"
            mutated.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract bytes differ"):
                KAT.load_contract(mutated)

    def test_source_inventory_is_exact_and_contains_no_document(self):
        self.assertEqual(AUTH.REQUIRED_SOURCES, {
            "bin/m33_safe_bridge_core.py",
            "bin/m33_ref_label_sham_kat.py",
            "bin/m33_ref_label_sham_source_auth.py",
            "conf/m33_ref_label_sham_contract.json",
            "conf/m33_ref_label_sham.config",
            "conf/m33_pre4_preregistration.json",
            "conf/m33_m0_materializer_contract.json",
            "modules/33_REF_LABEL_SHAM_KAT.nf",
            "workflows/m33_ref_label_sham.nf",
            "tests/test_m33_ref_label_sham.py",
        })
        self.assertTrue(all(not path.endswith((".md", ".pdf")) for path in AUTH.REQUIRED_SOURCES))

    def test_permutation_acts_on_complete_people_and_preserves_group_sizes(self):
        fixture = self.fixture()
        original_people = {
            person: fixture["ancestries"][fixture["people"].index(person)]
            for person in sorted(set(fixture["people"]))
        }
        assignments = []
        for seed in CORE.REF_LABEL_SHAM_SEEDS:
            labels, diagnostic = CORE.permute_diploid_reference_labels(
                fixture["node_ids"], fixture["people"], fixture["ancestries"], seed,
            )
            self.assertGreater(diagnostic["moved_person_count"], 0)
            self.assertEqual(Counter(labels), Counter(fixture["ancestries"]))
            for source in CORE.ANCESTRIES:
                transition = diagnostic["ancestry_transition_counts"][source]
                self.assertGreater(
                    sum(transition[target] for target in CORE.ANCESTRIES if target != source), 0)
            for person in sorted(set(fixture["people"])):
                indices = [index for index, value in enumerate(fixture["people"]) if value == person]
                self.assertEqual(len(indices), 2)
                self.assertEqual(labels[indices[0]], labels[indices[1]])
            permuted_people = {
                person: labels[fixture["people"].index(person)] for person in original_people
            }
            self.assertNotEqual(permuted_people, original_people)
            assignments.append(tuple(permuted_people[person] for person in sorted(permuted_people)))
        self.assertEqual(len(set(assignments)), 3)

    def test_rejects_node_split_single_ancestry_and_bad_seed(self):
        fixture = self.fixture()
        broken = fixture["ancestries"].copy()
        broken[1] = "EUR"
        with self.assertRaisesRegex(ValueError, "two same-ancestry nodes"):
            CORE.permute_diploid_reference_labels(
                fixture["node_ids"], fixture["people"], broken, CORE.REF_LABEL_SHAM_SEEDS[0])
        with self.assertRaisesRegex(ValueError, "at least two diploid"):
            CORE.permute_diploid_reference_labels([0, 1], ["P", "P"], ["AFR", "AFR"], 1)
        with self.assertRaisesRegex(ValueError, "seed"):
            CORE.permute_diploid_reference_labels(
                fixture["node_ids"], fixture["people"], fixture["ancestries"], -1)

    def test_rejects_sham_that_cannot_move_every_ancestry(self):
        with self.assertRaisesRegex(ValueError, "without cross-ancestry"):
            CORE.permute_diploid_reference_labels(
                list(range(8)),
                ["A", "A", "B", "B", "C", "C", "D", "D"],
                ["AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA", "ASIA", "ASIA"],
                4,
            )

    def test_assignment_is_invariant_to_node_order(self):
        fixture = self.fixture()
        order = [9, 2, 11, 0, 7, 4, 1, 10, 5, 8, 3, 6]
        for seed in CORE.REF_LABEL_SHAM_SEEDS:
            original, original_diagnostic = CORE.permute_diploid_reference_labels(
                fixture["node_ids"], fixture["people"], fixture["ancestries"], seed,
            )
            reordered, reordered_diagnostic = CORE.permute_diploid_reference_labels(
                [fixture["node_ids"][index] for index in order],
                [fixture["people"][index] for index in order],
                [fixture["ancestries"][index] for index in order],
                seed,
            )
            expected_by_node = dict(zip(fixture["node_ids"], original))
            observed_by_node = {
                fixture["node_ids"][index]: label for index, label in zip(order, reordered)
            }
            self.assertEqual(observed_by_node, expected_by_node)
            self.assertEqual(reordered_diagnostic, original_diagnostic)

    def test_sham_recomputes_exact_summaries_and_preserves_pooled_counts(self):
        fixture = self.fixture()
        original = CORE.summarize_reference(
            fixture["raw_states"], fixture["minor_codes"], fixture["node_ids"],
            fixture["people"], fixture["ancestries"], fixture["expected"],
        )
        shams, diagnostics = CORE.summarize_reference_label_shams(
            fixture["raw_states"], fixture["minor_codes"], fixture["node_ids"],
            fixture["people"], fixture["ancestries"], fixture["expected"],
        )
        self.assertEqual(set(shams), set(CORE.REF_LABEL_SHAM_SEEDS))
        self.assertEqual(len({row["assignment_sha256"] for row in diagnostics}), 3)
        for seed, summary in shams.items():
            labels, _ = CORE.permute_diploid_reference_labels(
                fixture["node_ids"], fixture["people"], fixture["ancestries"], seed,
            )
            oracle = KAT.independent_summary(
                fixture["raw_states"], fixture["minor_codes"], labels)
            for name in oracle:
                np.testing.assert_array_equal(summary[name], oracle[name])
            np.testing.assert_array_equal(summary["minor_ac"].sum(0),
                                          original["minor_ac"].sum(0))
            np.testing.assert_array_equal(summary["callable_an"].sum(0),
                                          original["callable_an"].sum(0))
            self.assertTrue(np.all(summary["minor_ac"] <= summary["callable_an"]))
            expected_af = np.divide(
                summary["minor_ac"], summary["callable_an"],
                out=np.zeros_like(summary["minor_af"]), where=summary["callable_an"] > 0,
            )
            np.testing.assert_array_equal(summary["minor_af"], expected_af)

    def test_diploid_dosage_shams_recompute_exact_complete_person_summaries(self):
        dosage, people, labels, nodes = self.diploid_fixture()
        dosage_before = dosage.copy()
        shams, diagnostics = CORE.summarize_diploid_dosage_reference_label_shams(
            dosage, people, labels, nodes,
            expected_people_by_ancestry={"AFR": 2, "EUR": 2, "ASIA": 2},
        )
        self.assertEqual(set(shams), set(CORE.REF_LABEL_SHAM_SEEDS))
        self.assertEqual(len({row["assignment_sha256"] for row in diagnostics}), 3)
        np.testing.assert_array_equal(dosage, dosage_before)
        diagnostic_text = json.dumps(diagnostics)
        self.assertTrue(all(person not in diagnostic_text for person in people))
        self.assertNotIn("node", diagnostic_text.lower())
        pooled_ac = dosage.sum(axis=1)
        for seed, summary in shams.items():
            flat_nodes = [node for person in people for node in nodes[person]]
            flat_people = [person for person in people for _ in range(2)]
            flat_labels = [label for label in labels for _ in range(2)]
            permuted_nodes, _ = CORE.permute_diploid_reference_labels(
                flat_nodes, flat_people, flat_labels, seed,
            )
            permuted_people = np.asarray(permuted_nodes[::2], dtype=object)
            oracle_ac = np.vstack([
                dosage[:, permuted_people == ancestry].sum(axis=1)
                for ancestry in CORE.ANCESTRIES
            ]).astype("<u2")
            np.testing.assert_array_equal(summary["minor_ac"], oracle_ac)
            np.testing.assert_array_equal(summary["minor_ac"].sum(axis=0), pooled_ac)
            np.testing.assert_array_equal(summary["callable_an"],
                                          np.full((3, 4), 4, dtype="<u2"))
            self.assertEqual(summary["minor_af"].dtype, np.dtype("<f8"))
            self.assertEqual(summary["observed_mask"].dtype, np.dtype("|u1"))

    def test_diploid_dosage_shams_fail_closed_on_axis_node_dosage_and_seed_drift(self):
        dosage, people, labels, nodes = self.diploid_fixture()
        with self.assertRaisesRegex(ValueError, "missing, non-integer or invalid"):
            CORE.summarize_diploid_dosage_reference_label_shams(
                dosage.astype(np.float64), people, labels, nodes)
        invalid = dosage.copy()
        invalid[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "missing, non-integer or invalid"):
            CORE.summarize_diploid_dosage_reference_label_shams(
                invalid, people, labels, nodes)
        incomplete = dict(nodes)
        incomplete.pop(people[-1])
        with self.assertRaisesRegex(ValueError, "mapping differs"):
            CORE.summarize_diploid_dosage_reference_label_shams(
                dosage, people, labels, incomplete)
        duplicate = dict(nodes)
        duplicate[people[-1]] = duplicate[people[0]]
        with self.assertRaisesRegex(ValueError, "duplicated across people"):
            CORE.summarize_diploid_dosage_reference_label_shams(
                dosage, people, labels, duplicate)
        with self.assertRaisesRegex(ValueError, "seeds differ"):
            CORE.summarize_diploid_dosage_reference_label_shams(
                dosage, people, labels, nodes, seeds=(1, 2, 3))
        with self.assertRaisesRegex(ValueError, "required firewall"):
            CORE.summarize_diploid_dosage_reference_label_shams(
                dosage, people, labels, nodes,
                expected_people_by_ancestry={"AFR": 30, "EUR": 30, "ASIA": 30},
            )

        uninformative = np.zeros_like(dosage)
        with self.assertRaisesRegex(ValueError, "identical to the real REF summary"):
            CORE.summarize_diploid_dosage_reference_label_shams(
                uninformative, people, labels, nodes,
                expected_people_by_ancestry={"AFR": 2, "EUR": 2, "ASIA": 2},
            )

    def write_runtime_source_auth(self, base: Path):
        staged = base / "staged"
        for relative in AUTH.REQUIRED_SOURCES:
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        source_auth = base / "source_auth.json"
        source_auth.write_text(json.dumps({
            "stage": "M33_REF_LABEL_SHAM_SOURCE_AUTH",
            "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
            "git_commit": COMMIT,
            "source_sha256": {
                relative: sha256_file(staged / relative)
                for relative in sorted(AUTH.REQUIRED_SOURCES)
            },
        }) + "\n", encoding="utf-8")
        return staged, source_auth

    def test_full_kat_is_receipt_only_deterministic_and_non_consumable(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            staged, source_auth = self.write_runtime_source_auth(base)
            contract = staged / "conf/m33_ref_label_sham_contract.json"
            first = KAT.run(
                contract, staged / "conf/m33_pre4_preregistration.json",
                staged / "conf/m33_m0_materializer_contract.json",
                source_auth, staged, COMMIT, KAT.OCI,
            )
            second = KAT.run(
                contract, staged / "conf/m33_pre4_preregistration.json",
                staged / "conf/m33_m0_materializer_contract.json",
                source_auth, staged, COMMIT, KAT.OCI,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["status"], KAT.STATUS)
            self.assertEqual(len(set(first["sham_summary_semantic_sha256"].values())), 3)
            self.assertEqual(first["input_and_TARGET_semantic_sha256_before"],
                             first["input_and_TARGET_semantic_sha256_after"])
            self.assertFalse(first["consumable"])
            self.assertFalse(first["truth_read"])
            self.assertFalse(first["real_asset_read"])
            self.assertFalse(first["training"])
            self.assertFalse(first["individual_reference_exported"])
            self.assertFalse(first["summary_arrays_persisted"])
            encoded = json.dumps(first).lower()
            for forbidden in ('"p0"', '"p1"', 'node_ids', 'raw_states', 'target_sentinel'):
                self.assertNotIn(forbidden, encoded)

    def test_runtime_source_auth_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            staged, source_auth = self.write_runtime_source_auth(Path(directory))
            AUTH.validate_source_auth(source_auth, COMMIT, staged)
            (staged / "bin/m33_ref_label_sham_kat.py").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                AUTH.validate_source_auth(source_auth, COMMIT, staged)

    def test_nextflow_is_local_offline_sequential_and_receipt_only(self):
        module = (ROOT / "modules/33_REF_LABEL_SHAM_KAT.nf").read_text()
        workflow = (ROOT / "workflows/m33_ref_label_sham.nf").read_text()
        config = (ROOT / "conf/m33_ref_label_sham.config").read_text()
        combined = (module + workflow + config).lower()
        self.assertIn("cache false", module.lower())
        self.assertIn("maxforks 1", module.lower())
        self.assertIn("maxforks = 1", config.lower())
        self.assertIn("--network none", config.lower())
        self.assertIn("--memory 1g", config.lower())
        self.assertIn("overwrite: false", module.lower())
        self.assertIn("receipt.json", module.lower())
        for forbidden in ("gs://projects-usp/", "lai_truth", "mosaic_events", "optimizer.step"):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
