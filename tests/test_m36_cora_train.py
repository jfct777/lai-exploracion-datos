from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m36_cora_train", ROOT / "bin/m36_cora_train.py")
TRAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)
MATERIALIZE_SPEC = importlib.util.spec_from_file_location("m36_cora_materialize", ROOT / "bin/m36_cora_materialize.py")
MATERIALIZE = importlib.util.module_from_spec(MATERIALIZE_SPEC)
sys.modules[MATERIALIZE_SPEC.name] = MATERIALIZE
MATERIALIZE_SPEC.loader.exec_module(MATERIALIZE)


class M36CoraTrainTests(unittest.TestCase):
    def test_staged_trainer_resolves_companion_modules_from_task_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wrapper = root / "wrapper"
            work = root / "work"
            wrapper.mkdir()
            work.mkdir()
            shutil.copy2(ROOT / "bin/m36_cora_train.py", wrapper / "m36_cora_train.py")
            shutil.copy2(ROOT / "bin/m36_cora_models.py", work / "m36_cora_models.py")
            shutil.copy2(ROOT / "bin/m36_cora_set.py", work / "m36_cora_set.py")
            result = subprocess.run(
                [sys.executable, str(wrapper / "m36_cora_train.py"), "--help"],
                cwd=work, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_train_seed_axis_is_explicit_and_unique(self) -> None:
        self.assertEqual(TRAIN.parse_train_seeds("1701,2701,3701"), (1701, 2701, 3701))
        with self.assertRaisesRegex(TRAIN.ContractError, "unique"):
            TRAIN.parse_train_seeds("1701,1701")
        with self.assertRaisesRegex(TRAIN.ContractError, "nonnegative"):
            TRAIN.parse_train_seeds("-1")

    @unittest.skipIf(TRAIN.np is None, "local runtime intentionally lacks NumPy/PyTorch")
    def test_cohort_one_hot_vocabulary_is_fit_only_with_unknown_level(self) -> None:
        token = {
            "sample_id": "S1", "event_id": "E1", "mutation_context": "CpG",
            **{field: 0.0 for field in TRAIN.NUMERIC_TOKEN_FIELDS},
        }
        covariates = {
            "S1": {"cohort": "FIT_COHORT", **{field: "0" for field in TRAIN.COVARIATE_FIELDS}},
            "S2": {"cohort": "HELD_OUT_COHORT", **{field: "0" for field in TRAIN.COVARIATE_FIELDS}},
        }
        prep = TRAIN.FitPreprocessor()
        prep.fit([token], covariates, {"S1"})

        class ArrayBackend:
            @staticmethod
            def tensor(value):
                return TRAIN.np.asarray(value)

        _, _, _, encoded = prep.encode_people([token], covariates, ["S1", "S2"], ArrayBackend)
        offset = len(TRAIN.COVARIATE_FIELDS)
        self.assertEqual(prep.cohort_vocab, {"<UNK>": 0, "FIT_COHORT": 1})
        self.assertNotIn("HELD_OUT_COHORT", prep.cohort_vocab)
        self.assertEqual(encoded.shape, (2, offset + 2))
        self.assertEqual(encoded[0, offset + 1], 1.0)
        self.assertEqual(encoded[1, offset], 1.0)

    def test_fold_audit_records_between_component_target_support(self) -> None:
        covariates = {
            "S1": {"cohort": "A"}, "S2": {"cohort": "B"},
            "S3": {"cohort": "A"}, "S4": {"cohort": "B"},
        }
        components = {"S1": "C1", "S2": "C2", "S3": "C3", "S4": "C4"}
        assignment = {"C1": 0, "C2": 1, "C3": 0, "C4": 1}
        targets = [
            {"sample_i": "S1", "sample_j": "S3", "target": "1", "target_stratum": "between_component"},
            {"sample_i": "S2", "sample_j": "S4", "target": "0", "target_stratum": "between_component"},
        ]
        audit = TRAIN.fold_preprocessing_audit(targets, covariates, components, assignment, 2)
        coverage = TRAIN.target_partition_coverage(targets, components, assignment)
        self.assertEqual(audit["0"]["validation_target_counts"]["between_component"]["positive"], 1)
        self.assertEqual(audit["1"]["validation_target_counts"]["between_component"]["zero"], 1)
        self.assertEqual(audit["0"]["validation_unseen_cohorts"], ["A"])
        self.assertEqual(coverage["validation_covered_pairs"], 2)
        self.assertEqual(coverage["cross_fold_not_scored_pairs"], 0)

    @unittest.skipIf(TRAIN.np is None, "local runtime intentionally lacks NumPy/PyTorch")
    def test_carrier_permutation_preserves_mac_and_dosages(self) -> None:
        events, covariates, _, _ = TRAIN.synthetic_inputs("additive")
        transformed = TRAIN.arm_events(events, "carrier_permuted", sorted(row["sample_id"] for row in covariates), 4)
        original_by_event, shuffled_by_event = {}, {}
        for row in events:
            if row["genotype_state"] == "ALT_CARRIER":
                original_by_event.setdefault(row["event_id"], []).append(int(row["genotype"]))
        for row in transformed:
            if row["genotype_state"] == "ALT_CARRIER":
                shuffled_by_event.setdefault(row["event_id"], []).append(int(row["genotype"]))
        self.assertEqual(
            {event: sorted(dosages) for event, dosages in original_by_event.items()},
            {event: sorted(dosages) for event, dosages in shuffled_by_event.items()},
        )

    @unittest.skipIf(TRAIN.np is None, "local runtime intentionally lacks NumPy/PyTorch")
    def test_geometry_only_is_identical_for_each_evaluable_individual(self) -> None:
        events, covariates, _, _ = TRAIN.synthetic_inputs("interaction")
        sample_ids = sorted(row["sample_id"] for row in covariates)
        tokens = TRAIN.build_tokens(events, "geometry_only", sample_ids)
        self.assertTrue(all(token["cm"] >= 0 for token in tokens))
        self.assertTrue(all(token["genotype_dosage"] == 0 for token in tokens))
        self.assertTrue(all(token["is_ac2_het"] == 0 for token in tokens))
        by_sample = {}
        for sample in sample_ids:
            by_sample[sample] = [
                (row["event_id"], row["cm"], row["callability"], row["mutation_context"])
                for row in tokens if row["sample_id"] == sample
            ]
        self.assertEqual(len(tokens), len(sample_ids) * len({row["event_id"] for row in events}))
        self.assertTrue(all(rows == by_sample[sample_ids[0]] for rows in by_sample.values()))

    @unittest.skipIf(TRAIN.np is None, "local runtime intentionally lacks NumPy/PyTorch")
    def test_real_event_lattice_requires_zero_evaluable_and_missing_rows(self) -> None:
        events, covariates, _, _ = TRAIN.synthetic_inputs("additive")
        sample_ids = {row["sample_id"] for row in covariates}
        TRAIN.validate_real_event_lattice(events, sample_ids)
        with self.assertRaisesRegex(TRAIN.ContractError, "event lattice"):
            TRAIN.validate_real_event_lattice(events[:-1], sample_ids)

    @unittest.skipIf(TRAIN.np is None or importlib.util.find_spec("torch") is None, "requires pinned NumPy/PyTorch image")
    def test_both_positive_controls_require_rare_channel_recovery(self) -> None:
        for control in ("additive", "interaction"):
            events, covariate_rows, component_rows, targets = TRAIN.synthetic_inputs(control)
            covariates = {row["sample_id"]: row for row in covariate_rows}
            component_map = {row["sample_id"]: row["pcrelate_component"] for row in component_rows}
            _, _, _, _, metrics, _ = TRAIN.run_successive_halving(
                events, covariates, targets, component_map, 3, ("deep_sets",), (32, 128), 2,
                TRAIN.stable_seed("m36", control), 10,
            )
            gate = TRAIN.positive_gate(metrics, 0.10)
            self.assertTrue(gate["passed"], f"{control}: {gate}")

    def test_real_training_receipt_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "receipt.json"
            path.write_text(json.dumps({"stage": "M36_CORA_MATERIALIZE", "status": "PASS", "synthetic": True}), encoding="utf-8")
            with self.assertRaisesRegex(TRAIN.ContractError, "materialization receipt"):
                TRAIN.validate_materialization_receipt(path)

    def test_real_training_receipt_requires_immutable_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "receipt.json"
            path.write_text(json.dumps({
                "stage": "M36_CORA_MATERIALIZE", "status": "PUBLISHED_PASS", "synthetic": False,
                "feature_schema": "m36_factorized_sparse_v1", "external_target_schema": "m36_external_common_pairs_log1p_v3_pair_total",
                "input_descriptors": {},
            }), encoding="utf-8")
            with self.assertRaisesRegex(TRAIN.ContractError, "bind exactly"):
                TRAIN.validate_materialization_receipt(path)

    def test_train_publication_receipt_rejects_synthetic_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            materialization = root / "materialization.json"
            materialization.write_text(json.dumps({"status": "MATERIALIZED_PASS"}), encoding="utf-8")
            summary = root / "summary.json"
            summary.write_text(json.dumps({"stage": "M36_CORA_SET_TRAIN", "mode": "smoke"}), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(ROOT / "bin/m36_cora_train_receipt.py"),
                "--materialization-receipt", str(materialization), "--train-summary", str(summary),
                "--out", str(root / "receipt.json"),
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)

    def test_train_publication_receipt_rejects_missing_cohort_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            materialization = root / "materialization.json"
            materialization.write_text(json.dumps({"status": "MATERIALIZED_PASS"}), encoding="utf-8")
            summary = root / "summary.json"
            summary.write_text(json.dumps({"stage": "M36_CORA_SET_TRAIN", "mode": "train"}), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(ROOT / "bin/m36_cora_train_receipt.py"),
                "--materialization-receipt", str(materialization), "--train-summary", str(summary),
                "--out", str(root / "receipt.json"),
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FIT-only cohort encoding", result.stderr)

    def test_train_publication_receipt_authenticates_fit_only_cohort_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            materialization = root / "materialization.json"
            materialization.write_text(json.dumps({"status": "MATERIALIZED_PASS"}), encoding="utf-8")
            summary = root / "summary.json"
            preprocessing = [
                "mutation_context_vocabulary", "token_normalization", "numeric_covariate_normalization",
                "cohort_one_hot_vocabulary",
            ]
            categorical = {
                "cohort": {"encoding": "one_hot", "vocabulary_scope": "FIT_only", "unknown_level": "<UNK>"}
            }
            fold_audit = {"0": {"cohort_vocabulary": {"<UNK>": 0, "A": 1}}}
            target_audit = {"total_pairs": 1, "validation_covered_pairs": 1}
            summary.write_text(json.dumps({
                "stage": "M36_CORA_SET_TRAIN", "mode": "train", "fit_only_preprocessing": preprocessing,
                "categorical_covariates": categorical,
                "runs": {"seed_1701": {
                    "fit_only_preprocessing_by_fold": fold_audit,
                    "target_partition_coverage": target_audit,
                }},
            }), encoding="utf-8")
            receipt = root / "receipt.json"
            subprocess.run([
                sys.executable, str(ROOT / "bin/m36_cora_train_receipt.py"),
                "--materialization-receipt", str(materialization), "--train-summary", str(summary),
                "--out", str(receipt),
            ], check=True)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["categorical_covariates"], categorical)
            self.assertEqual(payload["fit_only_preprocessing_by_run"]["seed_1701"], fold_audit)
            self.assertEqual(payload["target_partition_coverage_by_run"]["seed_1701"], target_audit)

    def test_factorized_geometry_control_does_not_expand_sample_by_locus(self) -> None:
        loci = [
            {"event_id": "E1", "chrom": "chr22", "position": "100", "mac": "2", "callability": "1",
             "mutation_context": "CpG", "cm": "0.1", "common_copying_context": "0.2"},
        ]
        carriers = [{"sample_id": "S1", "event_id": "E1", "minor_dosage": "1"}]
        events, missing = TRAIN.factorized_carrier_events(loci, carriers, [], {"S1", "S2"})
        self.assertEqual(len(events), 1)
        self.assertEqual(missing, set())
        self.assertEqual(TRAIN.build_tokens(events, "geometry_only", ["S1", "S2"], factorized=True), [])

    def test_real_loader_consumes_factorized_tables_without_dense_lattice(self) -> None:
        fixture = ROOT / "tests/fixtures"
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            MATERIALIZE.run(type("Args", (), {
                "rare_vcf": fixture / "m36_cora_factorized_rare.vcf",
                "locus_metadata": fixture / "m36_cora_factorized_loci.tab",
                "genetic_map": fixture / "m36_cora_factorized_map.tab",
                "sample_metadata": fixture / "m36_cora_factorized_metadata.tab",
                "pcrelate_components": fixture / "m36_cora_factorized_components.tab",
                "asibd_manifest": fixture / "m36_cora_factorized_asibd_manifest.tab",
                "asibd_segments": [fixture / "m36_cora_factorized_anc1.gapfilled_ibd"],
                "feature_chrom": "chr22", "zero_negative_ratio": 1, "seed": 1701, "outdir": outdir,
            })())
            receipt_path = outdir / "m36_cora_materialization_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["status"] = "PUBLISHED_PASS"
            for descriptor in receipt["input_descriptors"].values():
                descriptor["generation"] = "1"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            args = type("Args", (), {
                "loci": outdir / "m36_cora_loci.tsv", "carriers": outdir / "m36_cora_carriers.tsv",
                "missing": outdir / "m36_cora_missing.tsv", "covariates": outdir / "m36_cora_covariates.tsv",
                "components": outdir / "m36_cora_components.tsv", "targets": outdir / "m36_cora_external_targets.tsv",
                "materialization_receipt": receipt_path, "feature_chrom": "chr22",
            })()
            events, covariates, components, targets, missing = TRAIN.load_real_inputs(args)
            self.assertEqual(len(events), 4)
            self.assertEqual(len(covariates), 4)
            self.assertEqual(len(missing), 1)
            self.assertEqual(len(targets), 6)

    def test_symmetric_pair_feature_definition_is_present(self) -> None:
        source = (ROOT / "bin/m36_cora_models.py").read_text(encoding="utf-8")
        self.assertIn("torch.abs(left - right)", source)
        self.assertIn("left * right", source)
        self.assertIn("baseline + self.residual", source)

    def test_fit_only_context_and_validation_rank_are_explicit(self) -> None:
        source = (ROOT / "bin/m36_cora_train.py").read_text(encoding="utf-8")
        self.assertIn('self.context_vocab = {"<UNK>"', source)
        self.assertIn('scored.sort(key=lambda item: (item[0]', source)
        self.assertIn('left != fold and right != fold', source)
        self.assertIn('left == fold and right == fold', source)
        self.assertIn('synthetic {control} positive control was not recovered', source)
        self.assertIn('"final", final_spec.family', source)
        self.assertNotIn('stable_seed(str(seed), arm, str(fold))', source)
        self.assertIn('"bootstrap_component_MSE_ci95"', source)
        self.assertNotIn("selected_chromosomes", source)


if __name__ == "__main__":
    unittest.main()
