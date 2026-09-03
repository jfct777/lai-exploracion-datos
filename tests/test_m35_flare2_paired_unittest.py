import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "m35_flare2_paired.py"
CONTRACT = ROOT / "conf" / "m35_flare2_paired_contract.json"
SPEC = importlib.util.spec_from_file_location("m35_flare2_paired", SCRIPT)
M35 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M35)


def write_vcf(path: Path, samples: list[str], rows: list[list[str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for offset, genotypes in enumerate(rows):
            handle.write(f"22\t{100 + offset * 100}\t.\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(genotypes) + "\n")


class M35PairedTests(unittest.TestCase):
    def fixture(self, root: Path) -> Namespace:
        reference, target = root / "reference.vcf.gz", root / "target.vcf.gz"
        write_vcf(reference, ["AFR1", "EUR1", "NAM1"], [["0|1", "0|0", "1|0"], ["1|1", "0|1", "1|0"]])
        write_vcf(target, ["T1"], [["0|1"], ["1|0"]])
        for value in (root / "reference.vcf.gz.tbi", root / "target.vcf.gz.tbi", root / "flare.jar"):
            value.write_bytes(b"fixture")
        (root / "sample-map.tsv").write_text("AFR1\tAFR\nEUR1\tEUR\nNAM1\tNAM\n", encoding="utf-8")
        (root / "panel-macro.tsv").write_text("AFR\tAFR\nEUR\tEUR\nNAM\tNAM\n", encoding="utf-8")
        (root / "map.tsv").write_text("22\t50\t0\n22\t150\t0.1\n22\t250\t0.2\n", encoding="utf-8")
        (root / "create_model_file.py").write_text("print('fixture')\n", encoding="utf-8")
        return Namespace(contract=CONTRACT, reference_vcf=reference, reference_tbi=root / "reference.vcf.gz.tbi",
                         target_vcf=target, target_tbi=root / "target.vcf.gz.tbi", sample_map=root / "sample-map.tsv",
                         panel_macro_map=root / "panel-macro.tsv",
                         genetic_map=root / "map.tsv", flare_jar=root / "flare.jar",
                         flare2_model_builder=root / "create_model_file.py",
                         flare2_upstream_model_builder=root / "create_model_file.py",
                         outdir=root / "m35", preflight_only=True)

    def test_preflight_freezes_m34_seed_and_canonical_axes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            receipt = M35.run(args)
            delta = json.loads((args.outdir / "m35_paired.delta_manifest.json").read_text())
            self.assertEqual(receipt["status"], "PASS_PREFLIGHT_ONLY")
            self.assertEqual(M35.load_contract(CONTRACT)["methods"]["flare_0_6"]["parameters"]["seed"], 3401103)
            self.assertEqual(delta["shared_axes"]["marker_axis_sha256"],
                             M35.marker_axis_sha256([( "22", 100, "A", "G"), ("22", 200, "A", "G")]))
            self.assertIn("sample_axis_sha256", delta["shared_axes"]["target_axes"])
            self.assertIn("phase_axis_sha256", delta["shared_axes"]["reference_axes"])

    def test_phase_change_does_not_change_marker_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            first = M35.preflight(M35.load_contract(CONTRACT), {name: getattr(args, name) for name in M35.INPUT_NAMES}, args.outdir)
            write_vcf(args.target_vcf, ["T1"], [["1|0"], ["1|0"]])
            args.outdir = root / "m35_changed"
            second = M35.preflight(M35.load_contract(CONTRACT), {name: getattr(args, name) for name in M35.INPUT_NAMES}, args.outdir)
            self.assertEqual(first["delta"]["shared_axes"]["marker_axis_sha256"], second["delta"]["shared_axes"]["marker_axis_sha256"])
            self.assertNotEqual(first["delta"]["shared_axes"]["target_axes"]["phase_axis_sha256"], second["delta"]["shared_axes"]["target_axes"]["phase_axis_sha256"])

    def test_panel_map_is_normalized_to_reference_vcf_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            args.sample_map.write_text("NAM1\tNAM\nAFR1\tAFR\nEUR1\tEUR\n", encoding="utf-8")
            prepared = M35.preflight(
                M35.load_contract(CONTRACT),
                {name: getattr(args, name) for name in M35.INPUT_NAMES},
                args.outdir,
            )
            self.assertEqual(
                (args.outdir / "m35.shared.ref-panel.tsv").read_text(encoding="utf-8"),
                "AFR1\tAFR\nEUR1\tEUR\nNAM1\tNAM\n",
            )
            self.assertEqual(
                (args.outdir / "m35.direct.coarse.ref-panel.tsv").read_text(encoding="utf-8"),
                "AFR1\tAFR\nEUR1\tEUR\nNAM1\tNAM\n",
            )
            self.assertEqual(prepared["delta"]["shared_axes"]["reference_sample_count"], 3)

    def test_direct_uses_macro_panels_while_flare2_uses_fine_panels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            args.sample_map.write_text("AFR1\tA1\nEUR1\tE1\nNAM1\tN1\n", encoding="utf-8")
            args.panel_macro_map.write_text("A1\tAFR\nE1\tEUR\nN1\tNAM\n", encoding="utf-8")
            prepared = M35.preflight(
                M35.load_contract(CONTRACT),
                {name: getattr(args, name) for name in M35.INPUT_NAMES},
                args.outdir,
            )
            delta = prepared["delta"]["method_delta"]
            direct_panel = next(value for value in delta["flare_0_6"]["command_argv"] if value.startswith("ref-panel="))
            flare2_panel = next(value for value in delta["flare2"]["panel_probability_command_argv"] if value.startswith("ref-panel="))
            self.assertTrue(direct_panel.endswith("m35.direct.coarse.ref-panel.tsv"))
            self.assertTrue(flare2_panel.endswith("m35.shared.ref-panel.tsv"))
            self.assertEqual(
                (args.outdir / "m35.direct.coarse.ref-panel.tsv").read_text(encoding="utf-8"),
                "AFR1\tAFR\nEUR1\tEUR\nNAM1\tNAM\n",
            )

    def test_static_isolation_invariants(self):
        config = (ROOT / "conf" / "m35_flare2_paired.config").read_text()
        module = (ROOT / "modules" / "35_FLARE2_PAIRED.nf").read_text()
        workflow = (ROOT / "workflows" / "m35_flare2_paired.nf").read_text()
        dockerfile = (ROOT / "containers" / "m35-flare2" / "Dockerfile").read_text()
        self.assertIn("executor = 'google-batch'", config)
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", config)
        self.assertIn("resourceLabels = [team: 'frank']", config)
        self.assertIn("executor.queueSize = 1", config)
        self.assertIn("COPY flare.jar", dockerfile)
        self.assertIn("COPY create_model_file.py", dockerfile)
        self.assertNotIn("curl", dockerfile.lower())
        builder_sha = hashlib.sha256((ROOT / "containers" / "m35-flare2" / "create_model_file.py").read_bytes()).hexdigest()
        self.assertIn(f"CREATE_MODEL_FILE_SHA256={builder_sha}", dockerfile)
        inference = module.split("process M35_PACK_FLARE_PREDICTION")[0]
        self.assertNotIn("m34Truth", inference)
        self.assertIn("process M35_PACK_FLARE_PREDICTION", module)
        self.assertIn("process M35_SCORE_PAIRED", module)
        self.assertIn("m34_score_predictions.py", workflow)
        self.assertIn("m35_summarize_paired.py", workflow)

    def test_paired_summary_requires_one_truth_and_reports_ancestry_and_boundaries(self):
        summary_spec = importlib.util.spec_from_file_location("m35_summary", ROOT / "bin" / "m35_summarize_paired.py")
        summary = importlib.util.module_from_spec(summary_spec)
        assert summary_spec.loader is not None
        summary_spec.loader.exec_module(summary)
        metric = {
            "status": "PASS_SCORED", "truth_opened_only_by_scorer": True,
            "input_sha256": {"truth": "same-truth"}, "ancestry_names": ["AFR", "EUR", "NAM"],
            "marker_count": 2, "sample_count": 1, "cm_span": 0.1,
            "macro_ancestry_dose_MAE": 0.2, "haplotype_Brier": 0.3, "NAM_truth_present_MAE": 0.4,
            "per_ancestry_MAE": {"AFR": 0.1, "EUR": 0.2, "NAM": 0.3},
            "boundary": {"0.2": {"f1": 0.5, "false_transitions_per_cM": 1.0}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first.json", Path(temporary) / "second.json"
            first.write_text(json.dumps(metric), encoding="utf-8")
            metric["macro_ancestry_dose_MAE"] = 0.1
            metric["per_ancestry_MAE"]["NAM"] = 0.2
            metric["boundary"]["0.2"]["f1"] = 0.7
            second.write_text(json.dumps(metric), encoding="utf-8")
            gate = Path(temporary) / "gate.json"
            gate.write_text(json.dumps({"status": "PASS_NUMERICALLY_EQUIVALENT_TO_M34_CANONICAL_F0",
                                        "direct_metrics_sha256": summary.sha256_file(first)}), encoding="utf-8")
            result = summary.summarize(first, second, gate)
        self.assertEqual(result["shared_truth_sha256"], "same-truth")
        self.assertAlmostEqual(result["delta_flare2_minus_flare060"]["macro_ancestry_dose_MAE"], -0.1)
        self.assertIn("NAM", result["delta_flare2_minus_flare060"]["per_ancestry_MAE"])
        self.assertIn("0.2", result["delta_flare2_minus_flare060"]["boundary"])

    def test_cluster_assignment_recovers_permuted_labels_and_rejects_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.write_text("\n".join([
                "# list of reference panels", "AFR\tEUR\tNAM", "",
                "# p[i][j]: probability that a model state haplotype is in reference panel j",
                "# ignored", "0.05\t0.10\t0.85", "0.80\t0.10\t0.10", "0.10\t0.80\t0.10",
            ]) + "\n", encoding="utf-8")
            mapped = M35.cluster_assignment_from_model(model, ["AFR", "EUR", "NAM"],
                                                        {"AFR": "AFR", "EUR": "EUR", "NAM": "NAM"}, 0.5, 0.25)
            self.assertEqual(mapped["cluster_to_ancestry"], {"0": "NAM", "1": "AFR", "2": "EUR"})
            raw, relabeled = Path(temporary) / "raw.anc.vcf.gz", Path(temporary) / "relabeled.anc.vcf.gz"
            with gzip.open(raw, "wt", encoding="utf-8") as handle:
                handle.write("##ANCESTRY=<anc_0=0,anc_1=1,anc_2=2>\n")
                handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT1\n")
                handle.write("22\t100\t.\tA\tG\t.\tPASS\t.\tAN1:AN2:ANP1:ANP2\t0:2:0.7,0.2,0.1:0.1,0.2,0.7\n")
            M35.relabel_flare2_vcf(raw, relabeled, mapped, ["AFR", "EUR", "NAM"])
            with gzip.open(relabeled, "rt", encoding="utf-8") as handle:
                relabeled_text = handle.read()
            self.assertIn("##ANCESTRY=<AFR=0,EUR=1,NAM=2>", relabeled_text)
            self.assertIn("2:1:0.2,0.1,0.7:0.2,0.7,0.1", relabeled_text)
            model.write_text("\n".join([
                "# list of reference panels", "AFR\tEUR\tNAM", "",
                "# p[i][j]: probability that a model state haplotype is in reference panel j",
                "# ignored", "0.50\t0.50\t0.00", "0.50\t0.50\t0.00", "0.00\t0.00\t1.00",
            ]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(M35.PairedContractError, "ambiguous|insufficient"):
                M35.cluster_assignment_from_model(model, ["AFR", "EUR", "NAM"],
                                                   {"AFR": "AFR", "EUR": "EUR", "NAM": "NAM"}, 0.5, 0.25)

    def test_fine_panels_aggregate_to_macro_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "fine.model"
            model.write_text("\n".join([
                "# list of reference panels", "A1\tA2\tE1\tN1", "",
                "# p[i][j]: probability that a model state haplotype is in reference panel j",
                "# ignored", "0.05\t0.05\t0.10\t0.80", "0.45\t0.45\t0.05\t0.05", "0.10\t0.10\t0.75\t0.05",
            ]) + "\n", encoding="utf-8")
            mapping = M35.cluster_assignment_from_model(
                model, ["AFR", "EUR", "NAM"], {"A1": "AFR", "A2": "AFR", "E1": "EUR", "N1": "NAM"},
                0.5, 0.25,
            )
            self.assertEqual(mapping["cluster_to_ancestry"], {"0": "NAM", "1": "AFR", "2": "EUR"})

    def test_failed_cluster_gate_returns_auditable_no_go_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "no_go.model"
            model.write_text("\n".join([
                "# list of reference panels", "AFR\tEUR\tNAM", "",
                "# p[i][j]: probability that a model state haplotype is in reference panel j",
                "# ignored", "0.96\t0.03\t0.01", "0.01\t0.98\t0.01", "0.10\t0.60\t0.30",
            ]) + "\n", encoding="utf-8")
            evidence = M35.cluster_assignment_evidence_from_model(
                model, ["AFR", "EUR", "NAM"], {"AFR": "AFR", "EUR": "EUR", "NAM": "NAM"}, .5, .25,
            )
            self.assertEqual(evidence["status"], "NO_GO_TRUTH_BLIND_CLUSTER_ASSIGNMENT")
            self.assertEqual(evidence["selected_panel_probability"]["2"], .30)
            self.assertEqual(evidence["failure_reasons"], ["insufficient_panel_support"])
            with self.assertRaisesRegex(M35.PairedContractError, "insufficient"):
                M35.cluster_assignment_from_model(
                    model, ["AFR", "EUR", "NAM"], {"AFR": "AFR", "EUR": "EUR", "NAM": "NAM"}, .5, .25,
                )

    def test_metadata_panel_builder_labels_coarse_and_population_modes(self):
        spec = importlib.util.spec_from_file_location("m35_panels", ROOT / "bin" / "m35_build_ref_panels.py")
        panels = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(panels)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = root / "roles.tsv"
            roles.write_text(
                "sample_id\tancestry\tpopulation\tcanonical_population\trole\n"
                "a\tAfrican\tA1\tAfrican|A1\tREF_TRAIN\n"
                "b\tAfrican\tA2\tAfrican|A2\tREF_TRAIN\n"
                "e\tEuropean\tE1\tEuropean|E1\tREF_TRAIN\n"
                "n\tNative_American\tN1\tNative_American|N1\tREF_TRAIN\n", encoding="utf-8")
            fine = panels.build(roles, root / "fine", "population")
            coarse = panels.build(roles, root / "coarse", "coarse")
            self.assertEqual(fine["panel_count"], 4)
            self.assertEqual(fine["macro_counts"], {"AFR": 2, "EUR": 1, "NAM": 1})
            self.assertEqual(coarse["panel_count"], 3)

    def test_builder_adapter_reseeds_numpy_deterministically(self):
        wrapper = ROOT / "containers" / "m35-flare2" / "m35_create_model_wrapper.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builder = root / "fixture_builder.py"
            builder.write_text("import numpy as np, sys\nopen(sys.argv[3]+'.model','w').write(str(np.random.random()))\n", encoding="utf-8")
            (root / "numpy.py").write_text(
                "state=1\nclass Random:\n def seed(self, value):\n  global state; state=value\n def random(self):\n  global state; state=(1103515245*state+12345)%2147483648; return state/2147483648\nrandom=Random()\n",
                encoding="utf-8",
            )
            panels = root / "panels"
            panels.write_text("fixture\n", encoding="utf-8")
            environment = dict(os.environ, PYTHONPATH=str(root))
            for prefix in (root / "first", root / "second"):
                subprocess.run(["python3", str(wrapper), "--seed", "3401103", "--builder", str(builder),
                                "3", str(panels), str(prefix)], check=True, env=environment)
            self.assertEqual((root / "first.model").read_text(), (root / "second.model").read_text())

    def test_direct_gate_rejects_noncanonical_f0_metric(self):
        verifier_spec = importlib.util.spec_from_file_location("m35_verify", ROOT / "bin" / "m35_verify_direct_f0.py")
        verifier = importlib.util.module_from_spec(verifier_spec)
        assert verifier_spec.loader is not None
        verifier_spec.loader.exec_module(verifier)
        metric = {"status": "PASS_SCORED", "input_sha256": {"truth": "truth"}, "sample_count": 1,
                  "haplotype_count": 2, "marker_count": 2, "ancestry_names": ["AFR", "EUR", "NAM"],
                  "cm_span": 0.1, "boundary": {}, "macro_ancestry_dose_MAE": 0.1,
                  "per_ancestry_MAE": {"AFR": 0.1, "EUR": 0.1, "NAM": 0.1},
                  "NAM_truth_present_MAE": 0.1, "haplotype_Brier": 0.1}
        with tempfile.TemporaryDirectory() as temporary:
            direct, canonical = Path(temporary) / "direct.json", Path(temporary) / "canonical.json"
            direct.write_text(json.dumps(metric), encoding="utf-8")
            canonical.write_text(json.dumps(metric), encoding="utf-8")
            self.assertEqual(verifier.verify(direct, canonical)["status"],
                             "PASS_NUMERICALLY_EQUIVALENT_TO_M34_CANONICAL_F0")
            metric["haplotype_Brier"] = 0.2
            canonical.write_text(json.dumps(metric), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical F0"):
                verifier.verify(direct, canonical)


if __name__ == "__main__":
    unittest.main()
