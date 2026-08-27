#!/usr/bin/env python3

from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOSAICS = ROOT / "modules/34_NAM_MOSAICS.nf"
BRIDGE = ROOT / "modules/34_NAM_PANEL_FACTORS.nf"
TABIX = ROOT / "modules/34_NAM_TABIX.nf"
MODULES = tuple(sorted((ROOT / "modules").glob("34_NAM_*.nf")))
WORKFLOW = ROOT / "workflows/m34_nam_inputs.nf"
CONFIG = ROOT / "conf/m34_nam_inputs.config"
PYTORCH_IMAGE = (
    "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/"
    "m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99"
)
TABIX_IMAGE = (
    "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/"
    "m33-tabix@sha256:e730c35759e3851a92d7f3a6619333105331d97f1ae44a50dfa8d59745c43e54"
)
FLARE_IMAGE = (
    "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/"
    "m30-flare-runtime@sha256:86bf36c5d23407ed187d546f2420a0d2c44fbb6eed12ba81ddfc0f75df6b3a84"
)
EXPERIMENT_SHA256 = "dff5442ff413dd5b2cd901b2407082cf7f0629eb02d927c2942154054993c3ff"


class M34NamInputsNextflowTests(unittest.TestCase):
    def texts(self) -> dict[Path, str]:
        return {
            path: path.read_text(encoding="utf-8")
            for path in (*MODULES, WORKFLOW, CONFIG)
        }

    def test_modules_are_separate_explicit_and_append_safe(self):
        texts = self.texts()
        mosaics, bridge, tabix, workflow = (
            texts[MOSAICS], texts[BRIDGE], texts[TABIX], texts[WORKFLOW]
        )
        for module in (mosaics, bridge, tabix):
            self.assertEqual(module.count("process "), 1)
            self.assertIn("overwrite: false", module)
            self.assertIn("val(split)", module)
            self.assertIn("cpus { params.", module)
            self.assertIn("memory { params.", module)
            self.assertIn("time { params.", module)
        self.assertIn("tuple val(split), val(donorRole)", mosaics)
        self.assertIn("tuple val(split), val(mosaicDonorRole), path(mosaicVcf)", bridge)
        self.assertIn("tuple val(split), val(vcfRole), path(vcf)", tabix)
        self.assertIn("path phasedVcf", mosaics)
        self.assertIn("path panelVcf", bridge)
        self.assertIn("path splitTsv", bridge)
        self.assertIn("path geneticMap", bridge)
        self.assertIn("the run-specific results directory already exists", workflow)

    def test_fit_valid_roles_are_typed_and_propagated(self):
        texts = self.texts()
        mosaics, bridge, workflow = texts[MOSAICS], texts[BRIDGE], texts[WORKFLOW]
        self.assertIn("'FIT', roles.mosaic_fit_donors as String", workflow)
        self.assertIn("'VALID', roles.mosaic_valid_donors as String", workflow)
        self.assertIn("splitCases*.get(0) != ['FIT', 'VALID']", workflow)
        self.assertIn("seeds[rootId + '_FIT'] as Integer", workflow)
        self.assertIn("seeds[rootId + '_VALID'] as Integer", workflow)
        self.assertIn("targetSize.fit as Integer", workflow)
        self.assertIn("targetSize.valid as Integer", workflow)
        self.assertIn("mixtureArgument, generations as Double", workflow)
        self.assertIn("tuple(split, donorRole, mosaicVcf)", workflow)
        self.assertIn("--donor-role ${donorRole}", mosaics)
        self.assertIn("--forbidden-role ${forbiddenRole}", mosaics)
        self.assertIn("--mosaic-donor-role ${mosaicDonorRole}", bridge)

    def test_runtimes_are_exactly_digest_pinned_and_offline(self):
        texts = self.texts()
        mosaics, bridge, tabix, workflow, config = (
            texts[MOSAICS], texts[BRIDGE], texts[TABIX], texts[WORKFLOW], texts[CONFIG]
        )
        combined = "\n".join(texts.values())
        self.assertIn(PYTORCH_IMAGE, config)
        self.assertIn(PYTORCH_IMAGE, workflow)
        self.assertIn(TABIX_IMAGE, config)
        self.assertIn(TABIX_IMAGE, workflow)
        self.assertIn(FLARE_IMAGE, config)
        self.assertIn(FLARE_IMAGE, workflow)
        self.assertIn("container params.m34_inputs_pytorch_image", mosaics)
        self.assertIn("container params.m34_inputs_pytorch_image", bridge)
        self.assertIn("container params.m34_inputs_tabix_image", tabix)
        for module in MODULES:
            self.assertIn("--network none", texts[module])
        self.assertNotIn("gs://", combined)
        self.assertNotIn("google-batch", combined.lower())

    def test_tabix_derives_new_indexes_without_force(self):
        texts = self.texts()
        tabix, workflow = texts[TABIX], texts[WORKFLOW]
        self.assertIn("test ! -e ${vcf}.tbi", tabix)
        self.assertIn("tabix -p vcf ${vcf}", tabix)
        self.assertIn("test -s ${vcf}.tbi", tabix)
        self.assertNotIn("tabix -f", tabix)
        self.assertIn("tuple(split, 'REFERENCE', referenceVcf)", workflow)
        self.assertIn("tuple(split, 'TARGET', targetVcf)", workflow)

    def test_failure_policy_resources_and_scope_are_explicit(self):
        texts = self.texts()
        workflow, config = texts[WORKFLOW], texts[CONFIG]
        combined = "\n".join(texts.values()).lower()
        self.assertIn("errorstrategy = 'terminate'", config.lower())
        self.assertIn("maxretries = 0", config.lower())
        self.assertIn("resourcelabels = [team: 'frank']", config.lower())
        for prefix in (
            "contract", "mosaic", "bridge", "tabix", "flare_contract", "flare",
            "parse", "truth", "manifest", "plan", "train", "score",
        ):
            self.assertIn(f"m34_inputs_{prefix}_cpus", config)
            self.assertIn(f"m34_inputs_{prefix}_memory", config)
            self.assertIn(f"m34_inputs_{prefix}_time", config)
        self.assertIn("m34_nam_train_factorized", combined)
        self.assertIn("m34_nam_train_transformer_factorized", combined)
        self.assertIn("m34_nam_score_valid", combined)
        self.assertIn("m34_nam_pack_baseline", combined)
        self.assertIn("m34_inputs_train_memory = '8 GB'", config)

    def test_root_size_and_replication_plan_are_explicit_and_test_stays_closed(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        config = CONFIG.read_text(encoding="utf-8")
        self.assertIn("m34_inputs_root = 'R0'", config)
        self.assertIn("m34_inputs_target_size = 'small'", config)
        self.assertIn("m34_inputs_task_plan = null", config)
        self.assertIn("runResults.exists() && !workflow.resume", workflow)
        self.assertIn("M34_EXPLORATORY_128_REPLICATION_PLAN", workflow)
        self.assertIn("plan.test_opened != false", workflow)
        self.assertIn("targetSizeId != 'pilot_128'", workflow)
        self.assertIn("rootTasks.size() != 4", workflow)
        self.assertIn("task.maximum_updates == 3200", workflow)
        self.assertIn("m34TaskToken(task, radiusToken)", workflow)
        self.assertNotIn("def runResults = new File(", workflow)

    def test_exact_contract_hash_and_complete_wiring_are_present(self):
        texts = self.texts()
        workflow, config = texts[WORKFLOW], texts[CONFIG]
        self.assertIn(EXPERIMENT_SHA256, config)
        self.assertIn("experimentSha256 != params.m34_inputs_experiment_contract_sha256", workflow)
        self.assertIn("M34_NAM_VALIDATE_EXPERIMENT_CONTRACT", workflow)
        self.assertIn("family == 'transformer_small'", workflow)
        self.assertIn("family != 'transformer_small'", workflow)
        ordered = [
            "M34_NAM_GENERATE_MOSAICS(",
            "M34_NAM_PREPARE_PANEL_FACTORS(",
            "M34_NAM_TABIX_INDEX(",
            "M34_NAM_BUILD_FLARE_CONTRACT(",
            "M34_NAM_RUN_FLARE(",
            "M34_NAM_PARSE_F0(",
            "M34_NAM_ALIGN_TRUTH(",
            "M34_NAM_BUILD_FACTORIZED_MANIFEST(",
            "M34_NAM_TRAIN_FACTORIZED(",
            "M34_NAM_SCORE_VALID(",
        ]
        positions = [workflow.index(value) for value in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("m34_build_factorized_manifest.py", workflow)
        self.assertNotIn("m34_manifest_builder.py", workflow)

    def test_expanded_scope_uses_existing_factorized_training_entrypoints(self):
        texts = self.texts()
        workflow = texts[WORKFLOW]
        train_module = texts[ROOT / "modules/34_NAM_TRAIN_FACTORIZED.nf"]
        manifest_module = texts[ROOT / "modules/34_NAM_FACTORIZED_MANIFEST.nf"]
        self.assertEqual(workflow.count("M34_NAM_TRAIN_FACTORIZED("), 1)
        self.assertIn("m34_train_factorized.py", workflow)
        self.assertIn("m34_train_factorized.py", train_module)
        self.assertIn("path m33ContractPy", train_module)
        self.assertIn("cp ${m33ContractPy} staged/bin/m33_m0_contract.py", train_module)
        self.assertIn("m33_m0_contract.py", workflow)
        self.assertIn("path manifestBuilderPy", manifest_module)
        self.assertIn("python3 ${manifestBuilderPy}", manifest_module)
        self.assertIn("radiusCm, radiusToken, taskToken, taskBase64", workflow)
        self.assertIn("taskBase64, prediction)", workflow)
        self.assertIn("trainedPredictions.mix(baselinePredictions)", workflow)

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow is not installed")
    def test_configuration_parses(self):
        completed = subprocess.run(
            ["nextflow", "-C", str(CONFIG), "config", "-flat"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow is not installed")
    def test_stub_run_executes_fit_and_valid_dag_without_containers(self):
        with tempfile.TemporaryDirectory(prefix="m34-nam-inputs-") as raw:
            root = Path(raw)
            inputs = root / "inputs"
            inputs.mkdir()
            panel = inputs / "panel.chr22.vcf.gz"
            split = inputs / "m27f_split.private.tsv"
            genetic_map = inputs / "genetic.map.chr22"
            flare_jar = inputs / "flare.jar"
            panel.touch()
            split.touch()
            genetic_map.touch()
            flare_jar.touch()
            results = root / "results"
            work = root / "work"
            host_config = root / "host.config"
            host_config.write_text(
                f"includeConfig '{CONFIG}'\n"
                "docker.enabled = false\n",
                encoding="utf-8",
            )

            command = [
                "nextflow", "-C", str(host_config), "run", str(WORKFLOW),
                "-stub-run", "-ansi-log", "false",
                "-work-dir", str(work),
                "--m34_inputs_run_id", "fixture-r0",
                "--m34_inputs_root", "R0",
                "--m34_inputs_target_size", "small",
                "--m34_inputs_results_dir", str(results),
                "--m34_inputs_phased_vcf", str(panel),
                "--m34_inputs_split_tsv", str(split),
                "--m34_inputs_genetic_map", str(genetic_map),
                "--m34_inputs_flare_jar", str(flare_jar),
                "--m34_inputs_experiment_contract",
                str(ROOT / "conf/m34_nam_experiment_contract.json"),
                "--m34_inputs_adaptive_contract",
                str(ROOT / "conf/m34_adaptive_sweep_contract.json"),
            ]
            environment = dict(os.environ)
            environment["NXF_OFFLINE"] = "true"
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)

            run_results = results / "fixture-r0"
            for split_name in ("fit", "valid"):
                self.assertEqual(
                    len([
                        path for path in (run_results / split_name / "mosaics").rglob("*")
                        if path.is_file()
                    ]),
                    4,
                )
                self.assertEqual(
                    len([
                        path for path in (run_results / split_name / "bridge").rglob("*")
                        if path.is_file()
                    ]),
                    7,
                )
                for role in ("reference", "target"):
                    published = run_results / split_name / "indexed" / role
                    files = [path for path in published.rglob("*") if path.is_file()]
                    self.assertEqual(len(files), 2)
                    self.assertEqual(len([path for path in files if path.suffix == ".tbi"]), 1)
                self.assertEqual(
                    len([path for path in (run_results / split_name / "flare_contract").rglob("*")
                         if path.is_file()]),
                    1,
                )
                self.assertGreaterEqual(
                    len([path for path in (run_results / split_name / "flare").rglob("*")
                         if path.is_file()]),
                    2,
                )
                self.assertEqual(
                    len([path for path in (run_results / split_name / "f0").rglob("*")
                         if path.is_file()]),
                    3,
                )
                self.assertEqual(
                    len([path for path in (run_results / split_name / "truth").rglob("*")
                         if path.is_file()]),
                    2,
                )
            self.assertEqual(
                len([path for path in (run_results / "manifest").rglob("*")
                     if path.is_file()]),
                2,
            )
            self.assertEqual(
                len([path for path in (run_results / "models").rglob("*")
                     if path.is_file()]),
                6,
            )
            self.assertEqual(
                len([path for path in (run_results / "metrics").rglob("*.json")
                     if path.is_file()]),
                3,
            )

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow is not installed")
    def test_128_replication_plan_reaches_selected_root_dag(self):
        with tempfile.TemporaryDirectory(prefix="m34-nam-128-") as raw:
            root = Path(raw)
            inputs = root / "inputs"
            inputs.mkdir()
            files = {
                name: inputs / name for name in
                ("panel.vcf.gz", "split.tsv", "map.txt", "flare.jar")
            }
            for path in files.values():
                path.touch()
            plan = ROOT / "tests/fixtures/m34_128_replication.plan.json"
            plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
            host_config = root / "host.config"
            host_config.write_text(
                f"includeConfig '{CONFIG}'\n"
                "docker.enabled = false\n",
                encoding="utf-8",
            )
            command = [
                "nextflow", "-C", str(host_config), "run", str(WORKFLOW),
                "-stub-run", "-ansi-log", "false", "-work-dir", str(root / "work"),
                "--m34_inputs_run_id", "fixture-r2-128",
                "--m34_inputs_results_dir", str(root / "results"),
                "--m34_inputs_root", "R2",
                "--m34_inputs_target_size", "pilot_128",
                "--m34_inputs_fit_people", "96",
                "--m34_inputs_valid_people", "32",
                "--m34_inputs_task_plan", str(plan),
                "--m34_inputs_task_plan_sha256", plan_sha256,
                "--m34_inputs_phased_vcf", str(files["panel.vcf.gz"]),
                "--m34_inputs_split_tsv", str(files["split.tsv"]),
                "--m34_inputs_genetic_map", str(files["map.txt"]),
                "--m34_inputs_flare_jar", str(files["flare.jar"]),
                "--m34_inputs_experiment_contract",
                str(ROOT / "conf/m34_nam_experiment_contract.json"),
                "--m34_inputs_adaptive_contract",
                str(ROOT / "conf/m34_adaptive_sweep_contract.json"),
            ]
            environment = dict(os.environ)
            environment["NXF_OFFLINE"] = "true"
            completed = subprocess.run(
                command, cwd=root, env=environment, capture_output=True, text=True,
                check=False, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            metrics = list((root / "results/fixture-r2-128/metrics").rglob("*.json"))
            self.assertEqual(len(metrics), 5)


if __name__ == "__main__":
    unittest.main()
