import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify", ROOT / "bin/m33_ref_label_sham_technical_verify.py")
VERIFY = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(VERIFY)


class IndependentShamVerifierTests(unittest.TestCase):
    def fixture(self):
        people = tuple(f"P{i}" for i in range(6))
        labels = ("AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA")
        pairs = {p: (2 * i, 2 * i + 1) for i, p in enumerate(people)}
        dosage = np.asarray([[0, 1, 2, 0, 1, 2], [2, 0, 1, 2, 0, 1],
                             [0, 0, 1, 1, 2, 2]], dtype="|i1")
        return dosage, people, labels, pairs

    def test_oracle_is_deterministic_and_preserves_pooled_ac(self):
        dosage, people, labels, pairs = self.fixture()
        real = VERIFY.aggregate(dosage, labels)
        seen = set()
        for seed in VERIFY.SEEDS:
            permuted = VERIFY.permute_labels(people, labels, pairs, seed)
            sham = VERIFY.aggregate(dosage, permuted)
            np.testing.assert_array_equal(sham["minor_ac"].sum(0), real["minor_ac"].sum(0))
            np.testing.assert_array_equal(sham["callable_an"], np.full((3, 3), 4, dtype="<u2"))
            digest = VERIFY.semantic_sha256(sham)
            self.assertNotIn(digest, seen); seen.add(digest)

        self.assertEqual(
            [VERIFY.permute_labels(people, labels, pairs, seed) for seed in VERIFY.SEEDS],
            [
                ("EUR", "ASIA", "ASIA", "EUR", "AFR", "AFR"),
                ("AFR", "EUR", "EUR", "ASIA", "AFR", "ASIA"),
                ("ASIA", "EUR", "AFR", "AFR", "EUR", "ASIA"),
            ],
        )

    def test_fail_closed_on_seed_and_dosage(self):
        dosage, people, labels, pairs = self.fixture()
        with self.assertRaisesRegex(ValueError, "unregistered"):
            VERIFY.permute_labels(people, labels, pairs, 1)
        invalid = dosage.copy(); invalid[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "invalid diploid"):
            VERIFY.aggregate(invalid, labels)

    def test_source_is_independent_and_truth_free(self):
        source = (ROOT / "bin/m33_ref_label_sham_technical_verify.py").read_text()
        for forbidden in ("import m33_safe_bridge", "import m31", "import m33_a0", "lai_truth"):
            self.assertNotIn(forbidden, source)

    def test_private_people_are_mapped_to_pseudonyms_by_exact_node_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_people = tuple(f"private_{i:03d}" for i in range(90))
            labels = tuple(("AFR", "EUR", "ASIA")[i // 30] for i in range(90))
            private_pairs = {person: (2 * i, 2 * i + 1)
                             for i, person in enumerate(private_people)}
            panel = root / "panel.tsv"
            pairs = root / "pairs.tsv"
            with panel.open("w", encoding="utf-8") as handle:
                for i, label in enumerate(labels):
                    handle.write(f"REF_{label}_{i % 30:03d}\t{label}\n")
            with pairs.open("w", encoding="utf-8") as handle:
                handle.write("sample_id\tancestry\thaplotype_0_node\thaplotype_1_node\n")
                for i, label in enumerate(labels):
                    handle.write(f"REF_{label}_{i % 30:03d}\t{label}\t{2 * i}\t{2 * i + 1}\n")
            mapped, mapped_pairs = VERIFY.map_ref_pseudonyms(
                pairs, panel, private_people, labels, private_pairs,
            )
            self.assertEqual(mapped[0], "REF_AFR_000")
            self.assertEqual(mapped[-1], "REF_ASIA_029")
            self.assertEqual(mapped_pairs["REF_EUR_000"], (60, 61))


if __name__ == "__main__":
    unittest.main()
