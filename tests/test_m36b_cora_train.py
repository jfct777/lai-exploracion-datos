from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m36b_cora_train", ROOT / "bin/m36b_cora_train.py")
M36B = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M36B
SPEC.loader.exec_module(M36B)
UPSTREAM_SPEC = importlib.util.spec_from_file_location("m36_cora_train", ROOT / "bin/m36_cora_train.py")
UPSTREAM = sys.modules.get("m36_cora_train")
if UPSTREAM is None:
    UPSTREAM = importlib.util.module_from_spec(UPSTREAM_SPEC)
    sys.modules[UPSTREAM_SPEC.name] = UPSTREAM
    UPSTREAM_SPEC.loader.exec_module(UPSTREAM)


def carrier(sample: str, event: str, dosage: int, cm: float = 0.1) -> dict[str, str]:
    return {
        "sample_id": sample, "event_id": event, "chrom": "chr22", "position": "100",
        "mac": str(dosage * 2), "genotype": str(dosage), "callability": "0.99",
        "mutation_context": "CpG", "mutation_context_available": "1", "cm": str(cm),
        "common_copying_context": "0.25", "common_copying_context_available": "1",
        "genotype_state": "ALT_CARRIER", "evaluable_mask": "1",
    }


def covariates(samples: list[str]) -> dict[str, dict[str, str]]:
    result = {}
    for index, sample in enumerate(samples):
        result[sample] = {
            "sample_id": sample, "cohort": "A" if index < len(samples) // 2 else "B",
            "rare_burden": "4", "rare_callability": "0.99",
            "Q_AFR": "0.1", "Q_EUR": "0.7", "Q_NAM": "0.2", "Q_EAS": "0.0",
        }
    return result


@unittest.skipIf(M36B.np is None, "requires pinned NumPy runtime")
class M36BControlTests(unittest.TestCase):
    def test_permutation_preserves_both_margins_and_strata(self) -> None:
        samples = [f"S{i}" for i in range(8)]
        cov = covariates(samples)
        events = []
        for offset, sample in enumerate(samples):
            # Four events per cohort and dosage produce switchable bipartite edges.
            events.append(carrier(sample, f"E{offset % 4}", 1, offset / 10))
            events.append(carrier(sample, f"H{offset % 4}", 2, offset / 10))
        original_event, original_sample = M36B._margin_signature(events)
        permuted, audit = M36B.degree_preserving_permutation(events, cov, set(), 1701, 100.0)
        self.assertEqual(M36B._margin_signature(permuted), (original_event, original_sample))
        self.assertGreater(audit["accepted_swaps"], 0)
        self.assertGreater(audit["moved_carrier_fraction"], 0)
        original_strata = {
            event: sorted(M36B.permutation_stratum(cov[row["sample_id"]]) for row in events if row["event_id"] == event)
            for event in {row["event_id"] for row in events}
        }
        permuted_strata = {
            event: sorted(M36B.permutation_stratum(cov[row["sample_id"]]) for row in permuted if row["event_id"] == event)
            for event in {row["event_id"] for row in permuted}
        }
        self.assertEqual(permuted_strata, original_strata)

    def test_permutation_never_uses_missing_cross_cell(self) -> None:
        samples = ["S0", "S1", "S2", "S3"]
        cov = {sample: dict(covariates(samples)[sample], cohort="A") for sample in samples}
        events = [carrier("S0", "E0", 1), carrier("S1", "E1", 1),
                  carrier("S2", "E2", 1), carrier("S3", "E3", 1)]
        missing = {("S0", "E1"), ("S1", "E0")}
        permuted, _ = M36B.degree_preserving_permutation(events, cov, missing, 2701, 100.0)
        self.assertFalse({(row["sample_id"], row["event_id"]) for row in permuted} & missing)

    def test_permutation_is_split_isolated(self) -> None:
        samples = ["F0", "F1", "V0", "V1"]
        cov = {sample: dict(covariates(samples)[sample], cohort="A") for sample in samples}
        events = [carrier("F0", "EF0", 1), carrier("F1", "EF1", 1),
                  carrier("V0", "EV0", 1), carrier("V1", "EV1", 1)]
        permuted, audit = M36B.partitioned_degree_preserving_permutation(
            events, cov, set(), {"FIT": {"F0", "F1"}, "VALIDATION": {"V0", "V1"}},
            1701, 0.5, 0.01,
        )
        by_sample = {row["sample_id"]: row["event_id"] for row in permuted}
        self.assertTrue(by_sample["F0"].startswith("EF") and by_sample["F1"].startswith("EF"))
        self.assertTrue(by_sample["V0"].startswith("EV") and by_sample["V1"].startswith("EV"))
        self.assertIn("FIT", audit["partitions"])

    def test_permutation_fails_closed_when_too_few_edges_move(self) -> None:
        samples = ["F0", "F1", "V0", "V1"]
        cov = {sample: dict(covariates(samples)[sample], cohort="A") for sample in samples}
        # A single edge in each partition cannot participate in a double-edge
        # swap, so a nominal permutation must not be accepted as a valid null.
        events = [carrier("F0", "EF0", 1), carrier("V0", "EV0", 1)]
        with self.assertRaisesRegex(M36B.ContractError, "did not move enough edges"):
            M36B.partitioned_degree_preserving_permutation(
                events, cov, set(), {"FIT": {"F0", "F1"}, "VALIDATION": {"V0", "V1"}},
                1701, 100.0, 0.50,
            )

    def test_both_known_answer_controls_pass_full_triage_path(self) -> None:
        specs = M36B.available_specs(("deep_sets", "set_transformer"))
        train_kwargs = {
            "pair_batch_size": 1024,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "huber_delta": 1.0,
        }
        for control in M36B.POSITIVE_CONTROLS:
            events, cov, components, targets, missing = M36B.positive_control_inputs(control)
            result = M36B.run_nested_screen(
                events=events, covariates=cov, component_map=components,
                targets=targets, missing_pairs=missing, specs=specs,
                budgets=(16, 64), halving_eta=2, outer_folds=3, inner_folds=2,
                run_seed=M36B.stable_seed("1701", control), bootstrap_reps=100,
                permutation_swap_multiplier=10.0, minimum_moved_carrier_fraction=0.50,
                minimum_relative_mse_reduction=0.10, minimum_positive_folds=2,
                train_kwargs=train_kwargs,
            )
            self.assertTrue(result["promotion_gate"]["passed"], control)
            for fold in result["control_diagnostics"]["carrier_permutation_by_outer_fold"].values():
                for partition in fold["partitions"].values():
                    self.assertTrue(partition["mixing_gate_passed"])
                    self.assertGreaterEqual(partition["moved_carrier_fraction"], 0.50)

    def test_failed_positive_control_aborts_before_real_input_read(self) -> None:
        args = Namespace(
            model_families="deep_sets,set_transformer", halving_budgets="16,64",
            positive_control_budgets="16,64", halving_eta=2, outer_folds=3,
            inner_folds=2, train_seeds="1701", bootstrap_reps=40,
            pair_batch_size=1024, permutation_swap_multiplier=10.0,
            minimum_moved_carrier_fraction=0.50, positive_control_seed=1701,
            learning_rate=0.01, weight_decay=0.0001, huber_delta=1.0,
            minimum_relative_mse_reduction=0.10, minimum_positive_folds=2,
            outdir=Path("unused"), feature_chrom="chr22",
        )
        failed = {
            "selected_architecture_by_outer_fold": {},
            "promotion_gate": {"passed": False},
            "control_diagnostics": {"carrier_permutation_by_outer_fold": {}},
        }
        with mock.patch.object(M36B, "run_nested_screen", return_value=failed), \
                mock.patch.object(M36B, "load_real_inputs") as real_loader:
            with self.assertRaisesRegex(M36B.ContractError, "positive control failed"):
                M36B.run(args)
            real_loader.assert_not_called()

    def test_geometry_keeps_axis_but_erases_identity_and_dosage(self) -> None:
        events = [carrier("S0", "E0", 1, 0.1), carrier("S1", "E1", 2, 0.7)]
        tokens = M36B.geometry_tokens(events, "S0")
        self.assertEqual([row["cm"] for row in tokens], [0.1, 0.7])
        self.assertTrue(all(row["sample_id"] == "S0" for row in tokens))
        self.assertTrue(all(row["genotype_dosage"] == 0 and row["mac_scaled"] == 0 for row in tokens))
        self.assertTrue(all(row["mutation_context"] == "<GEOMETRY>" for row in tokens))

    def test_paired_effects_are_fold_macro_and_component_clustered(self) -> None:
        predictions = {arm: [] for arm in M36B.ARMS}
        for fold in range(3):
            for pair in range(4):
                base = {
                    "pair_id": f"{fold}-{pair}", "outer_fold": fold,
                    "target_stratum": "between_component", "component_i": f"C{fold}-{pair}",
                    "component_j": f"C{fold}-{pair + 1}", "target": 1.0,
                    "prediction": 1.0, "baseline_prediction": 0.0, "absolute_error": 0.0,
                }
                predictions["rare_enabled"].append(dict(base, arm="rare_enabled", squared_error=0.1))
                for arm in M36B.COMPARATORS:
                    predictions[arm].append(dict(base, arm=arm, squared_error=0.4))
        effects = M36B.paired_effects(predictions, 100, 1701)
        macro = [row for row in effects if row["scope"] == "fold_macro" and row["target_stratum"] == "ALL"]
        self.assertEqual(len(macro), 3)
        self.assertTrue(all(abs(row["delta_mse_control_minus_rare"] - 0.3) < 1e-8 for row in macro))
        self.assertTrue(all("component-node" in row["bootstrap_method"] for row in macro))

    def test_small_factorized_end_to_end_keeps_selection_inside_fit(self) -> None:
        events, cov_rows, component_rows, targets = UPSTREAM.synthetic_inputs("interaction")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_table(path: Path, rows: list[dict[str, str]], fields=None) -> None:
                use_fields = fields or list(rows[0])
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = __import__("csv").DictWriter(handle, fieldnames=use_fields, delimiter="\t")
                    writer.writeheader()
                    writer.writerows(rows)

            references = {}
            carriers = []
            for row in events:
                references.setdefault(row["event_id"], {
                    key: row[key] for key in (
                        "event_id", "chrom", "position", "mac", "callability",
                        "mutation_context", "cm", "common_copying_context",
                    )
                })
                if row["genotype_state"] == "ALT_CARRIER":
                    carriers.append({
                        "sample_id": row["sample_id"], "event_id": row["event_id"],
                        "minor_dosage": row["genotype"],
                    })
            paths = {
                "loci": root / "loci.tsv", "carriers": root / "carriers.tsv",
                "missing": root / "missing.tsv", "covariates": root / "covariates.tsv",
                "components": root / "components.tsv", "targets": root / "targets.tsv",
            }
            write_table(paths["loci"], list(references.values()))
            write_table(paths["carriers"], carriers)
            write_table(paths["missing"], [], ["sample_id", "event_id"])
            write_table(paths["covariates"], cov_rows)
            write_table(paths["components"], component_rows)
            write_table(paths["targets"], targets)
            receipt = root / "materialization.json"
            receipt.write_text(json.dumps({
                "stage": "M36_CORA_MATERIALIZE", "status": "MATERIALIZED_PASS",
                "synthetic": False, "feature_schema": "m36_factorized_sparse_v1",
                "external_target_schema": "m36_external_common_pairs_log1p_v3_pair_total",
                "input_descriptors": {
                    name: {"uri": path.name, "generation": "LOCAL", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in paths.items()
                },
            }), encoding="utf-8")
            args = Namespace(
                **paths, materialization_receipt=receipt, feature_chrom="chr22",
                model_families="deep_sets", halving_budgets="1,2", halving_eta=2,
                outer_folds=3, inner_folds=2, train_seeds="1701", bootstrap_reps=40,
                pair_batch_size=16, permutation_swap_multiplier=20.0,
                minimum_moved_carrier_fraction=0.01,
                positive_control_budgets="16,64", positive_control_seed=1701,
                learning_rate=0.01, weight_decay=0.0001, huber_delta=1.0,
                minimum_relative_mse_reduction=0.10, minimum_positive_folds=2,
                outdir=root / "out",
            )
            summary = M36B.run(args)
            self.assertEqual(summary["architecture_selection"], "nested inner component-disjoint CV within each outer FIT partition")
            self.assertEqual(set(summary["runs"]["1701"]["selected_architecture_by_outer_fold"]), {"0", "1", "2"})
            self.assertTrue((args.outdir / "m36b_paired_effects.tsv").is_file())
            prediction_header = (args.outdir / "m36b_predictions.tsv").read_text().splitlines()[0]
            self.assertNotIn("sample_i", prediction_header)
            self.assertNotIn("sample_j", prediction_header)


class M36BProvenanceTests(unittest.TestCase):
    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_receipt_binds_code_config_inputs_outputs_budgets_and_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = {}
            for name in ("loci", "carriers", "missing", "covariates", "components", "targets"):
                path = root / f"{name}.tsv"
                path.write_text(f"{name}\n", encoding="utf-8")
                inputs[name] = path
            materialization = root / "materialization.json"
            materialization.write_text(json.dumps({
                "status": "MATERIALIZED_PASS",
                "input_descriptors": {
                    name: {"uri": path.name, "generation": "LOCAL", "sha256": self._sha(path)}
                    for name, path in inputs.items()
                },
            }), encoding="utf-8")
            summary = root / "m36b_train_summary.json"
            summary.write_text(json.dumps({
                "stage": "M36B_CORA_SET_TRAIN", "status": "TRAINED_EXPLORATORY",
                "effective_parameters": {
                    "train_seeds": [1701], "halving_budgets": [1, 2],
                    "minimum_relative_mse_reduction": 0.10, "minimum_positive_folds": 2,
                    "minimum_moved_carrier_fraction": 0.50,
                    "positive_control_budgets": [16, 64], "positive_control_seed": 1701,
                    "positive_controls": ["additive", "interaction"],
                },
                "architecture_selection": "nested", "uncertainty": "component node",
                "pre_real_positive_controls": {
                    "executed_before_real_data_read": True,
                    "required_controls": ["additive", "interaction"],
                    "all_passed": True,
                    "runs": {
                        "additive": {"promotion_gate": {"passed": True}},
                        "interaction": {"promotion_gate": {"passed": True}},
                    },
                },
            }), encoding="utf-8")
            config = root / "run.config"
            contract = root / "contract.json"
            code = root / "train.py"
            output = root / "effects.tsv"
            config.write_text("params {}\n", encoding="utf-8")
            contract.write_text(json.dumps({
                "stage": "M36B_CORA_SET_EXPLORATORY",
                "promotion": {
                    "minimum_relative_mse_reduction": 0.10,
                    "minimum_positive_outer_folds": 2,
                },
                "controls": {"carrier_permutation_mixing_gate": {
                    "minimum_moved_carrier_fraction_per_outer_partition": 0.50,
                }},
                "pre_real_positive_controls": {
                    "controls": ["additive", "interaction"], "budgets": [16, 64], "seed": 1701,
                },
            }), encoding="utf-8")
            code.write_text("pass\n", encoding="utf-8")
            output.write_text("effect\n", encoding="utf-8")
            receipt = root / "receipt.json"
            command = [
                sys.executable, str(ROOT / "bin/m36b_cora_receipt.py"),
                "--materialization-receipt", str(materialization), "--train-summary", str(summary),
                "--run-config", str(config), "--code", str(code),
                "--design-contract", str(contract),
            ]
            for name, path in inputs.items():
                command.extend(("--input", f"{name}={path}"))
            command.extend(("--output", str(summary), "--output", str(output),
                            "--output-prefix", "gs://example/run", "--out", str(receipt)))
            subprocess.run(command, check=True)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["effective_parameters"]["train_seeds"], [1701])
            self.assertTrue(payload["pre_real_positive_controls"]["all_passed"])
            self.assertIn("run.config", payload["run_config"]["path"])
            self.assertEqual(set(payload["inputs"]), set(inputs))
            output.write_text("tampered\n", encoding="utf-8")
            self.assertNotEqual(payload["outputs"]["effects.tsv"]["sha256"], self._sha(output))

    def test_nextflow_lane_is_isolated_and_receipt_bound(self) -> None:
        workflow = (ROOT / "workflows/m36b_cora_set.nf").read_text(encoding="utf-8")
        module = (ROOT / "modules/36B_CORA_SET.nf").read_text(encoding="utf-8")
        self.assertIn("M36B_CORA_SET_TRAIN", workflow)
        self.assertNotIn("M27F", workflow + module)
        self.assertIn("m36b_provenance_receipt.json", module)
        self.assertIn("--run-config", module)
        self.assertIn("--design-contract", module)
        self.assertIn("'minimum-moved-carrier-fraction'", module)
        self.assertIn("'positive-control-budgets'", module)
        self.assertIn("params.m36b_run_config", workflow)
        self.assertIn("m36b_run_config", (ROOT / "conf/m36b_cora_chr22_screen.config").read_text())
        self.assertIn("m36b_run_config", (ROOT / "conf/m36b_cora_chr22_technical.config").read_text())
        self.assertIn("--input loci=", module)
        self.assertIn("resourceLabels = [team: 'frank'", (ROOT / "conf/m36b_cora_chr22_screen.config").read_text())
        contract = json.loads((ROOT / "conf/m36b_cora_preregistration.json").read_text())
        self.assertIn("chr22 rare-event features predict", contract["scientific_direction"])
        self.assertIn("effect-sign consistency", contract["stop_rule"])

    def test_real_config_binds_only_the_verified_materialization_run(self) -> None:
        config = (ROOT / "conf/m36b_cora_chr22_screen.config").read_text(encoding="utf-8")
        expected = "m36-cora-chr22-materialize-20260902a/m36_cora_set/"
        self.assertEqual(config.count(expected), 7)
        self.assertNotIn("m36-cora-chr22-materialize-20260903a", config)
        self.assertIn(f"{expected}m36_cora_materialization_receipt.json", config)


if __name__ == "__main__":
    unittest.main()
