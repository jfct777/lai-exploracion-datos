import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(
    os.environ.get(
        "M28D_SCORER_PATH", Path(__file__).parents[1] / "bin" / "m28d_b0_scorer.py"
    )
)
SPEC = importlib.util.spec_from_file_location("m28d_b0_scorer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class M28dB0ScorerTest(unittest.TestCase):
    def setUp(self):
        self.genetic_map = MODULE.GeneticMap(
            [MODULE.MapPoint(0, 0.0), MODULE.MapPoint(100, 1.0)]
        )

    @staticmethod
    def probabilities(first, second):
        def one_hot(label):
            return tuple(float(ancestry == label) for ancestry in MODULE.ANCESTRIES)

        return {"T000": (one_hot(first), one_hot(second))}

    def window(self, left, right, first, second, marker_start=0, marker_end=1):
        return MODULE.PredictionWindow(
            left=left,
            right=right,
            marker_start=marker_start,
            marker_end=marker_end,
            n_markers=marker_end - marker_start,
            probabilities=self.probabilities(first, second),
            hard_labels={"T000": (first, second)},
        )

    @staticmethod
    def truth(h0, h1):
        return {"T000": (h0, h1)}

    def test_discrete_voronoi_is_contiguous_and_ties_go_left(self):
        cells = MODULE.discrete_voronoi([10, 20, 31])
        self.assertEqual(cells, [(10, 16), (16, 26), (26, 32)])
        self.assertEqual(sum(right - left for left, right in cells), 22)

    def test_perfect_known_answer(self):
        markers = [10, 20, 30]
        windows = [
            self.window(10, 16, "AFR", "ASIA", 0, 1),
            self.window(16, 26, "EUR", "ASIA", 1, 2),
            self.window(26, 31, "EUR", "ASIA", 2, 3),
        ]
        truth = self.truth(
            [
                MODULE.TruthSegment(10, 16, "AFR"),
                MODULE.TruthSegment(16, 31, "EUR"),
            ],
            [MODULE.TruthSegment(10, 31, "ASIA")],
        )
        result = MODULE.score_objects(markers, windows, truth, self.genetic_map, [0.1, 0.2, 0.5])
        self.assertEqual(result["primary"]["macro"], 0.0)
        self.assertEqual(result["secondary"]["phase_aligned_haplotype_brier"], 0.0)
        self.assertEqual(result["secondary"]["hard_unordered_diploid_state"]["accuracy"], 1.0)
        for summary in result["secondary"]["boundaries"].values():
            self.assertEqual(summary["truth_boundaries"], 1)
            self.assertEqual(summary["predicted_boundaries"], 1)
            self.assertEqual(summary["matched_boundaries"], 1)
            self.assertEqual(summary["matched_distance_median_cm"], 0.0)

    def test_primary_and_core_secondaries_are_global_phase_swap_invariant(self):
        markers = [10, 20, 30]
        direct = [
            self.window(10, 16, "AFR", "ASIA", 0, 1),
            self.window(16, 26, "EUR", "ASIA", 1, 2),
            self.window(26, 31, "EUR", "ASIA", 2, 3),
        ]
        swapped = [
            self.window(10, 16, "ASIA", "AFR", 0, 1),
            self.window(16, 26, "ASIA", "EUR", 1, 2),
            self.window(26, 31, "ASIA", "EUR", 2, 3),
        ]
        truth = self.truth(
            [MODULE.TruthSegment(10, 16, "AFR"), MODULE.TruthSegment(16, 31, "EUR")],
            [MODULE.TruthSegment(10, 31, "ASIA")],
        )
        first = MODULE.score_objects(markers, direct, truth, self.genetic_map, [0.2])
        second = MODULE.score_objects(markers, swapped, truth, self.genetic_map, [0.2])
        self.assertEqual(first["primary"], second["primary"])
        self.assertEqual(
            first["secondary"]["phase_aligned_haplotype_brier"],
            second["secondary"]["phase_aligned_haplotype_brier"],
        )
        self.assertEqual(
            first["secondary"]["hard_unordered_diploid_state"],
            second["secondary"]["hard_unordered_diploid_state"],
        )
        for key in (
            "truth_boundaries",
            "predicted_boundaries",
            "matched_boundaries",
            "missed_truth_boundaries",
            "extra_predicted_boundaries",
            "precision",
            "recall",
            "f1",
            "matched_distance_median_cm",
            "matched_distance_p95_cm",
        ):
            self.assertEqual(
                first["secondary"]["boundaries"]["0.2"][key],
                second["secondary"]["boundaries"]["0.2"][key],
            )

    def test_truth_boundary_inside_prediction_window_uses_exact_overlap(self):
        markers = [10, 20]
        windows = [self.window(10, 21, "AFR", "ASIA", 0, 2)]
        truth = self.truth(
            [MODULE.TruthSegment(10, 18, "AFR"), MODULE.TruthSegment(18, 21, "EUR")],
            [MODULE.TruthSegment(10, 21, "ASIA")],
        )
        result = MODULE.score_objects(markers, windows, truth, self.genetic_map, [0.2])
        self.assertAlmostEqual(result["primary"]["per_ancestry"]["AFR"], 1.5 / 11)
        self.assertAlmostEqual(result["primary"]["per_ancestry"]["EUR"], 1.5 / 11)
        self.assertEqual(result["primary"]["per_ancestry"]["ASIA"], 0.0)
        self.assertAlmostEqual(result["primary"]["macro"], 1.0 / 11)

    def test_missing_boundary_reduces_recall_and_extra_reduces_precision(self):
        truth = [MODULE.Boundary(0.2, "AFR", "EUR")]
        missing = MODULE.ordered_boundary_match(truth, [], 0.2)
        self.assertEqual(missing, [])
        prediction = [
            MODULE.Boundary(0.2, "AFR", "EUR"),
            MODULE.Boundary(0.3, "EUR", "AFR"),
        ]
        matched = MODULE.ordered_boundary_match(truth, prediction, 0.2)
        self.assertEqual(matched, [0.0])
        self.assertEqual(len(prediction) - len(matched), 1)

    def test_boundary_f1_is_zero_when_denominators_exist_without_matches(self):
        markers = [10, 20]
        windows = [
            self.window(10, 16, "AFR", "ASIA", 0, 1),
            self.window(16, 21, "ASIA", "ASIA", 1, 2),
        ]
        truth = self.truth(
            [MODULE.TruthSegment(10, 16, "AFR"), MODULE.TruthSegment(16, 21, "EUR")],
            [MODULE.TruthSegment(10, 21, "ASIA")],
        )
        summary = MODULE.score_objects(
            markers, windows, truth, self.genetic_map, [0.2]
        )["secondary"]["boundaries"]["0.2"]
        self.assertEqual(summary["precision"], 0.0)
        self.assertEqual(summary["recall"], 0.0)
        self.assertEqual(summary["f1"], 0.0)
        self.assertIsNone(summary["per_directed_transition"]["AFR->EUR"]["precision"])
        self.assertEqual(summary["per_directed_transition"]["AFR->EUR"]["recall"], 0.0)

    def test_boundary_matching_is_label_aware_and_one_to_one(self):
        truth = [
            MODULE.Boundary(0.20, "AFR", "EUR"),
            MODULE.Boundary(0.25, "AFR", "EUR"),
        ]
        prediction = [
            MODULE.Boundary(0.21, "EUR", "AFR"),
            MODULE.Boundary(0.24, "AFR", "EUR"),
        ]
        distances = MODULE.ordered_boundary_match(truth, prediction, 0.1)
        self.assertEqual(len(distances), 1)
        self.assertAlmostEqual(distances[0], 0.01)

    def test_boundary_matching_does_not_cross_transition_order(self):
        truth = [
            MODULE.Boundary(0.20, "AFR", "EUR"),
            MODULE.Boundary(0.30, "EUR", "AFR"),
        ]
        prediction = [
            MODULE.Boundary(0.21, "EUR", "AFR"),
            MODULE.Boundary(0.29, "AFR", "EUR"),
        ]
        distances = MODULE.ordered_boundary_match(truth, prediction, 0.2)
        self.assertEqual(len(distances), 1)

    def test_genetic_map_plateau_has_zero_nonnegative_weight(self):
        genetic_map = MODULE.GeneticMap(
            [
                MODULE.MapPoint(0, 0.0),
                MODULE.MapPoint(10, 0.1),
                MODULE.MapPoint(20, 0.1),
                MODULE.MapPoint(30, 0.2),
            ]
        )
        self.assertEqual(genetic_map.cm_at(18) - genetic_map.cm_at(12), 0.0)

    def test_genetic_coordinate_validation_uses_map_and_msp_rounding_policy(self):
        metadata = [
            {"spos": 10, "epos": 20, "sgpos": 0.100005, "egpos": 0.200005},
            {"spos": 30, "epos": 40, "sgpos": 0.300005, "egpos": 0.400005},
        ]
        MODULE.validate_genetic_coordinates(
            [10, 20, 30],
            [0.1, 0.2, 0.3],
            [15, 35],
            [0.15, 0.35],
            metadata,
            self.genetic_map,
            1e-8,
            1e-5,
        )
        bad = [dict(item) for item in metadata]
        bad[1]["egpos"] = 0.42
        with self.assertRaisesRegex(ValueError, "MSP cM endpoints"):
            MODULE.validate_genetic_coordinates(
                [10, 20, 30],
                [0.1, 0.2, 0.3],
                [15, 35],
                [0.15, 0.35],
                bad,
                self.genetic_map,
                1e-8,
                1e-5,
            )

    def test_truth_gap_fails_closed(self):
        markers = [10, 20]
        windows = [self.window(10, 21, "AFR", "ASIA", 0, 2)]
        truth = self.truth(
            [MODULE.TruthSegment(10, 15, "AFR"), MODULE.TruthSegment(16, 21, "EUR")],
            [MODULE.TruthSegment(10, 21, "ASIA")],
        )
        with self.assertRaisesRegex(ValueError, "does not cover"):
            MODULE.score_objects(markers, windows, truth, self.genetic_map, [0.2])

    def test_fb_non_normalized_probability_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "bad.fb"
            path.write_text(
                "chromosome\tphysical position\tgenetic_position\tgenetic_marker_index\t"
                "T000:::hap1:::AFR\tT000:::hap1:::EUR\tT000:::hap1:::ASIA\t"
                "T000:::hap2:::AFR\tT000:::hap2:::EUR\tT000:::hap2:::ASIA\n"
                "22\t10\t0.1\t.\t0.8\t0.3\t0.0\t0.0\t0.0\t1.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "do not sum to one"):
                MODULE.load_fb(path)

    def test_msp_parses_gnomix_code_header_with_first_code_after_colon(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "fixture.msp"
            path.write_text(
                "#Subpopulation order/codes: AFR=0\tEUR=1\tASIA=2\n"
                "#chm\tspos\tepos\tsgpos\tegpos\tn snps\tT000.0\tT000.1\n"
                "22\t10\t20\t0.1\t0.2\t2\t0\t2\n",
                encoding="utf-8",
            )
            metadata, samples, labels = MODULE.load_msp(path)
            self.assertEqual(samples, ["T000"])
            self.assertEqual(metadata[0]["n_snps"], 2)
            self.assertEqual(labels[0]["T000"], ("AFR", "ASIA"))

    def test_hash_mismatch_fails_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "input.txt"
            path.write_text("content\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                MODULE.require_hash(path, "0" * 64, "fixture")

    def test_end_to_end_synthetic_score_and_A_B_compare(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            markers = directory / "b0.tsv"
            markers.write_text(
                "arm_component\tchrom\tposition\tcm\n"
                "B0\tchr22\t10\t0.10\n"
                "B0\tchr22\t20\t0.20\n"
                "B0\tchr22\t30\t0.30\n",
                encoding="utf-8",
            )
            genetic_map = directory / "map.tsv"
            genetic_map.write_text("22\t0\t0.0\n22\t100\t1.0\n", encoding="utf-8")
            truth = directory / "truth.tsv"
            truth.write_text(
                "target_haplotype\tchrom\tstart_bp\tend_bp_exclusive\tancestry\n"
                "T000_h0\tchr22\t10\t16\tAFR\n"
                "T000_h0\tchr22\t16\t31\tEUR\n"
                "T000_h1\tchr22\t10\t31\tASIA\n",
                encoding="utf-8",
            )
            fb = directory / "prediction.fb"
            fb.write_text(
                "chromosome\tphysical position\tgenetic_position\tgenetic_marker_index\t"
                "T000:::hap1:::AFR\tT000:::hap1:::EUR\tT000:::hap1:::ASIA\t"
                "T000:::hap2:::AFR\tT000:::hap2:::EUR\tT000:::hap2:::ASIA\n"
                "22\t10\t0.10\t.\t1\t0\t0\t0\t0\t1\n"
                "22\t20\t0.20\t.\t0\t1\t0\t0\t0\t1\n"
                "22\t30\t0.30\t.\t0\t1\t0\t0\t0\t1\n",
                encoding="utf-8",
            )
            msp = directory / "prediction.msp"
            msp.write_text(
                "#Subpopulation order/codes: AFR=0\tEUR=1\tASIA=2\n"
                "#chm\tspos\tepos\tsgpos\tegpos\tn snps\tT000.0\tT000.1\n"
                "22\t10\t10\t0.10\t0.10\t1\t0\t2\n"
                "22\t20\t20\t0.20\t0.20\t1\t1\t2\n"
                "22\t30\t30\t0.30\t0.30\t1\t1\t2\n",
                encoding="utf-8",
            )
            comparison = directory / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "stage": "M28C_GNOMIX_FULL_B0_RESOURCE_BENCHMARK_COMPARE",
                        "decision": "PASS_FULL_B0_TECHNICAL_BENCHMARK",
                        "scope": "full_B0_training_serialization_reload_and_inference_only_no_target_truth_accuracy_screen_or_effect_estimation",
                        "truth_accessed": False,
                        "target_truth_accuracy_computed": False,
                        "gates": {"T5_INFERENCE": True, "T8_SCOPE": True},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_provenance = directory / "run_provenance.json"
            run_provenance.write_text('{"git_commit":"fixture"}\n', encoding="utf-8")
            target_hash = "f" * 64
            raw_target_hash = "e" * 64
            simulation_outputs = {
                "m28_sources.trees": "a" * 64,
                "m28_mosaic_events.private.tsv.gz": "b" * 64,
                "m28_pools.private.tsv": "c" * 64,
            }
            simulation_manifest = directory / "simulation.manifest.json"
            simulation_manifest.write_text(
                json.dumps(
                    {
                        "stage": "M28_LAI_SIMULATION_PREFLIGHT",
                        "params": {"root_seed": 20260818},
                        "inputs": {"genetic.map.chr22": MODULE.sha256_file(genetic_map)},
                        "sha256": {
                            "m28_lai_truth.tsv.gz": MODULE.sha256_file(truth),
                            **simulation_outputs,
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            b0_preflight_manifest = directory / "b0_preflight.manifest.json"
            b0_preflight_manifest.write_text(
                json.dumps(
                    {
                        "stage": "M28C_B0_INPUT_PREFLIGHT",
                        "params": {"root_seed": 20260818},
                        "inputs": {
                            **simulation_outputs,
                            "m28b_v4_validation_B0.tsv.gz": MODULE.sha256_file(markers),
                        },
                        "sha256": {"m28c_b0_target.vcf.gz": raw_target_hash},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            ingest_report = directory / "ingest.json"
            ingest_report.write_text(
                json.dumps(
                    {
                        "stage": "M28C_B0_GNOMIX_INGEST_AUDIT",
                        "decision": "GO_B0_GNOMIX_TRAINING_PREREGISTRATION",
                        "root_seed": 20260818,
                        "merged_truth_table_accessed": False,
                        "output_sha256": {"m28c_b0_target.vcf.gz": target_hash},
                        "upstream_sha256": {"m28c_b0_target.vcf.gz": raw_target_hash},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            inference_manifests = {}
            for replicate in ("A", "B"):
                path = directory / f"inference_{replicate}.manifest.json"
                path.write_text(
                    json.dumps(
                        {
                            "stage": f"M28C_GNOMIX_FULL_B0_INFER_{replicate}",
                            "params": {
                                "replicate": replicate,
                                "truth_accessed": False,
                                "target_truth_accuracy_computed": False,
                            },
                            "inputs": {"m28c_b0_target.vcf.gz": target_hash},
                            "sha256": {
                                "query_results.fb": MODULE.sha256_file(fb),
                                "query_results.msp": MODULE.sha256_file(msp),
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                inference_manifests[replicate] = path
            known_runner_hash = "1" * 64
            contract = {
                "stage": "M28D_B0_DESCRIPTIVE_SCORING",
                "version": 2,
                "status": "PRE_FROZEN_AMENDED_BEFORE_TRUTH_ACCESS",
                "scope": "descriptive_scoring_and_scorer_validation_only",
                "authenticated_inputs": {
                    "truth_sha256": MODULE.sha256_file(truth),
                    "b0_marker_table_sha256": MODULE.sha256_file(markers),
                    "genetic_map_sha256": MODULE.sha256_file(genetic_map),
                    "replicate_A_fb_sha256": MODULE.sha256_file(fb),
                    "replicate_A_msp_sha256": MODULE.sha256_file(msp),
                    "replicate_B_fb_sha256": MODULE.sha256_file(fb),
                    "replicate_B_msp_sha256": MODULE.sha256_file(msp),
                    "m28c_comparison_sha256": MODULE.sha256_file(comparison),
                    "simulation_manifest_sha256": MODULE.sha256_file(simulation_manifest),
                    "b0_preflight_manifest_sha256": MODULE.sha256_file(
                        b0_preflight_manifest
                    ),
                    "ingest_report_sha256": MODULE.sha256_file(ingest_report),
                    "replicate_A_inference_manifest_sha256": MODULE.sha256_file(
                        inference_manifests["A"]
                    ),
                    "replicate_B_inference_manifest_sha256": MODULE.sha256_file(
                        inference_manifests["B"]
                    ),
                    "target_vcf_sha256": target_hash,
                },
                "authenticated_implementation": {
                    "scorer_sha256": MODULE.sha256_file(MODULE_PATH),
                    "known_answer_runner_sha256": known_runner_hash,
                    "unit_test_sha256": "2" * 64,
                },
                "fixed_domain": {
                    "chromosome_truth": "chr22",
                    "first_b0_marker": 10,
                    "last_b0_marker": 30,
                    "expected_bp_weight": 21,
                    "markers": 3,
                    "windows": 3,
                    "target_samples": 1,
                },
                "genetic_coordinate_validation": {
                    "scoring_cm_source": "authenticated_genetic_map_only",
                    "b0_marker_tolerance_cm": 1e-8,
                    "msp_endpoint_tolerance_cm": 1e-5,
                },
                "secondary_estimands": {
                    "boundary_tolerances_cm": {
                        "primary_descriptive": 0.2,
                        "sensitivities": [0.1, 0.5],
                    }
                },
                "seed_policy": {
                    "seed": 20260818,
                    "role": "protected_validation_seed_descriptive_only",
                },
                "unresolved_before_inference": {
                    "SESOI": "No defensible project value is frozen."
                },
            }
            contract_path = directory / "contract.json"
            contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
            receipt = directory / "known.json"
            receipt.write_text(
                json.dumps(
                    {
                        "stage": "M28D_B0_SCORER_KNOWN_ANSWERS",
                        "decision": "PASS_M28D_SCORER_KNOWN_ANSWERS",
                        "real_truth_accessed": False,
                        "checks": {"synthetic_fixture": True},
                        "unit_suite": {"passed": True, "tests_run": 1},
                        "scorer_sha256": MODULE.sha256_file(MODULE_PATH),
                        "contract_sha256": MODULE.sha256_file(contract_path),
                        "known_answer_runner_sha256": known_runner_hash,
                        "unit_test_file_sha256": "2" * 64,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            pair_auth = directory / "pair_auth.json"
            auth_args = SimpleNamespace(
                    contract=contract_path,
                    truth=truth,
                    b0_markers=markers,
                    genetic_map=genetic_map,
                    fb_a=fb,
                    msp_a=msp,
                    fb_b=fb,
                    msp_b=msp,
                    m28c_comparison=comparison,
                    simulation_manifest=simulation_manifest,
                    b0_preflight_manifest=b0_preflight_manifest,
                    ingest_report=ingest_report,
                    inference_manifest_a=inference_manifests["A"],
                    inference_manifest_b=inference_manifests["B"],
                    known_answer_receipt=receipt,
                    output=pair_auth,
                )
            original_load_truth = MODULE.load_truth
            MODULE.load_truth = lambda *args, **kwargs: self.fail(
                "pair authentication must not parse truth"
            )
            try:
                MODULE.authenticate_pair_command(auth_args)
                bad_inference = directory / "bad_inference_B.json"
                bad_inference.write_text('{"tampered":true}\n', encoding="utf-8")
                auth_args.inference_manifest_b = bad_inference
                with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                    MODULE.authenticate_pair_command(auth_args)
            finally:
                MODULE.load_truth = original_load_truth
                auth_args.inference_manifest_b = inference_manifests["B"]

            scores = {}
            for replicate in ("A", "B"):
                output = directory / f"score_{replicate}.json"
                manifest = directory / f"score_{replicate}.manifest.json"
                MODULE.score_command(
                    SimpleNamespace(
                        contract=contract_path,
                        truth=truth,
                        b0_markers=markers,
                        genetic_map=genetic_map,
                        fb=fb,
                        msp=msp,
                        m28c_comparison=comparison,
                        simulation_manifest=simulation_manifest,
                        b0_preflight_manifest=b0_preflight_manifest,
                        ingest_report=ingest_report,
                        inference_manifest=inference_manifests[replicate],
                        known_answer_receipt=receipt,
                        pair_auth_receipt=pair_auth,
                        run_provenance=run_provenance,
                        replicate=replicate,
                        output=output,
                        manifest=manifest,
                    )
                )
                scores[replicate] = output
                document = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(document["metrics"]["primary"]["macro"], 0.0)
                self.assertFalse(document["BR_BS_authorized"])
                self.assertTrue(manifest.is_file())

            compare_output = directory / "compare.json"
            compare_manifest = directory / "compare.manifest.json"
            MODULE.compare_command(
                SimpleNamespace(
                    score_a=scores["A"],
                    score_b=scores["B"],
                    output=compare_output,
                    manifest=compare_manifest,
                )
            )
            compared = json.loads(compare_output.read_text(encoding="utf-8"))
            self.assertEqual(
                compared["decision"], "PASS_B0_DESCRIPTIVE_SCORER_REPRODUCIBILITY"
            )
            self.assertTrue(compared["scientific_payload_exact"])
            invalid_b = json.loads(scores["B"].read_text(encoding="utf-8"))
            invalid_b["BR_BS_authorized"] = True
            invalid_path = directory / "score_B_invalid.json"
            invalid_path.write_text(json.dumps(invalid_b) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "violates frozen scope"):
                MODULE.compare_command(
                    SimpleNamespace(
                        score_a=scores["A"],
                        score_b=invalid_path,
                        output=directory / "invalid_compare.json",
                        manifest=directory / "invalid_compare.manifest.json",
                    )
                )


if __name__ == "__main__":
    unittest.main()
