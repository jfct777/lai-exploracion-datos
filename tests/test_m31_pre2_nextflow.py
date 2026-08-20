import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))

import m31_pre2_pipeline as PRE2


class Pre2NextflowBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = (REPO / "modules" / "31_ORDERED_LINEAR_PRE2.nf").read_text()
        cls.workflow = (REPO / "workflows" / "m31_ordered_linear_pre2.nf").read_text()
        cls.config = (REPO / "conf" / "m31_ordered_linear_pre2.config").read_text()

    def test_only_scorer_process_mentions_root18_truth(self):
        before_scorer, scorer = self.module.split("process M31_PRE2_SCORE_ROOT18", 1)
        self.assertNotIn("root18_truth", before_scorer.lower())
        self.assertIn("val root18_truth_source", scorer)
        self.assertNotIn("path root18_truth", scorer.lower())

    def test_stop_channel_is_the_only_scorer_trigger(self):
        self.assertIn("M31_PRE2_ROOT17_GATE.out.open_token", self.workflow)
        self.assertIn("optional: true, emit: open_token", self.module)
        self.assertIn("cache false", self.module)
        self.assertIn("maxRetries 0", self.module)

    def test_worker_screen_is_real_and_exact(self):
        self.assertIn("channel.of(1, 4, 8)", self.workflow)
        self.assertIn("--worker-dir worker-1 --worker-dir worker-4 --worker-dir worker-8", self.module)
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
        ):
            self.assertIn(f"export {name}=1", self.module)

    def test_config_is_local_and_has_no_cloud_executor(self):
        self.assertIn("executor = 'local'", self.config)
        self.assertNotIn("google-batch", self.config.lower())
        self.assertIn("2c30d018028636ac1b7a4890641e04b3e15be8c79d991dfade35b90db0e17bd1", self.config)
        self.assertIn("m31_pre2_pregate_container_options = ''", self.config)
        self.assertNotIn("DNABR_CONTAINER_OPTIONS", self.config)
        self.assertNotIn("DNABR_M31_PRE2_SCORER_CONTAINER_OPTIONS", self.config)
        self.assertIn("/usr/lib/google-cloud-sdk:/usr/lib/google-cloud-sdk:ro", self.config)
        self.assertIn("gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/runs/", self.config)
        self.assertIn("m31_pre2_min_local_disk_gib = 20", self.config)
        self.assertIn("workDir = params.m31_pre2_work_dir", self.config)

    def test_real_run_requires_external_authorization(self):
        self.assertIn("M31_PRE2_VERIFY_AUTHORIZATION", self.workflow)
        self.assertIn("m31_pre2_execution_authorization", self.workflow)
        self.assertIn("--execution-authorization", self.module)

    def test_every_pipeline_process_stages_import_dependencies(self):
        self.assertEqual(self.module.count("python3 ${pipeline_py}"), 7)
        for dependency in ("contract_validator", "runner_py", "core_py", "receipt_py"):
            self.assertEqual(self.module.count(f"path {dependency}"), 6)


class Pre2OpeningLedgerTest(unittest.TestCase):
    def test_claim_is_permanent_and_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger"
            claim = PRE2._claim_once(ledger, "run-a", "a" * 64)
            payload = json.loads(claim.read_text())
            self.assertEqual(payload["status"], "CLAIMED_ROOT18_CONSUMED")
            with self.assertRaisesRegex(PRE2.Pre2Error, "already been claimed"):
                PRE2._claim_once(ledger, "run-a", "a" * 64)
            with self.assertRaisesRegex(PRE2.Pre2Error, "already been claimed"):
                PRE2._claim_once(ledger, "different-run-id", "b" * 64)

    def test_worker_screen_rejects_any_semantic_difference(self):
        base = {
            "workers": 1,
            "scientific_fingerprint": {"fit": "same"},
            "scientific_fingerprint_sha256": PRE2.sha256_payload({"fit": "same"}),
        }
        manifests = [dict(base, workers=value) for value in (1, 4, 8)]
        manifests[2] = {
            **manifests[2], "scientific_fingerprint": {"fit": "different"},
            "scientific_fingerprint_sha256": PRE2.sha256_payload({"fit": "different"}),
        }
        with tempfile.TemporaryDirectory() as temporary:
            args = type("Args", (), {
                "worker_dir": [Path(temporary) / str(value) for value in (1, 4, 8)],
                "output": Path(temporary) / "screen.json",
            })()
            with mock.patch.object(PRE2, "_load_worker", side_effect=manifests):
                with self.assertRaisesRegex(PRE2.Pre2Error, "differ"):
                    PRE2.verify_workers(args)


class Pre2PairedBootstrapTest(unittest.TestCase):
    @staticmethod
    def _summary(counts):
        total = sum(item.value for item in counts)
        return {"supported_metric": None if total == 0 else total / len(counts)}

    def test_bootstrap_is_paired_and_reports_undefined_replicates(self):
        d_counts = [SimpleNamespace(sample_id="A", value=0), SimpleNamespace(sample_id="B", value=1)]
        f0_counts = [SimpleNamespace(sample_id="A", value=0), SimpleNamespace(sample_id="B", value=2)]
        with (
            mock.patch.object(PRE2.core, "BOOTSTRAP_REPLICATES", 16),
            mock.patch.object(PRE2.core, "BOOTSTRAP_SEED", 7),
            mock.patch.object(PRE2.runner, "summarize_counts", side_effect=self._summary),
        ):
            report = PRE2._paired_bootstrap_counts({"F0": f0_counts, "D": d_counts})
        marginal = report["marginal"]["D"]["supported_metric"]
        delta = report["paired_deltas"]["D_minus_F0"]["supported_metric"]
        self.assertEqual(marginal["n_valid_replicates"] + marginal["n_undefined_replicates"], 16)
        self.assertGreater(marginal["n_undefined_replicates"], 0)
        self.assertEqual(delta["n_valid_replicates"], marginal["n_valid_replicates"])
        self.assertEqual(report["pairing"], "same_sample_order_and_same_resampled_indexes_for_all_arms")

    def test_bootstrap_rejects_different_sample_order(self):
        d_counts = [SimpleNamespace(sample_id="A", value=1), SimpleNamespace(sample_id="B", value=2)]
        f0_counts = [SimpleNamespace(sample_id="B", value=2), SimpleNamespace(sample_id="A", value=1)]
        with self.assertRaisesRegex(PRE2.Pre2Error, "sample IDs/order differ"):
            PRE2._paired_bootstrap_counts({"F0": f0_counts, "D": d_counts})


class Pre2AuthorizationTest(unittest.TestCase):
    def test_authorization_is_run_commit_container_and_cost_bound(self):
        contract = REPO / "conf" / "m31_ordered_linear_pre2_preregistration.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization = root / "authorization.json"
            payload = {
                "schema_version": "1.0.0",
                "experiment_id": "M31_ORDERED_LINEAR_DEV_PRE2",
                "status": "AUTHORIZED_REAL_RUN",
                "scope": "ROOT17_FIT_GATE_AND_CONDITIONAL_SINGLE_ROOT18_SCORE",
                "run_id": "run-a",
                "contract_sha256": PRE2.core.sha256_file(contract),
                "git_commit": "a" * 40,
                "container_digest": "sha256:" + "b" * 64,
                "execution_source_sha256": {"pipeline.py": "c" * 64},
                "max_cost_usd": 5.0,
                "authorized_by": "jfct777",
                "authorized_utc": "2026-08-20T00:00:00Z",
                "explicit_user_authorization": True,
            }
            authorization.write_text(json.dumps(payload), encoding="utf-8")
            args = type("Args", (), {
                "authorization": authorization, "contract": contract,
                "run_id": "run-a", "expected_git_commit": "a" * 40,
                "container_digest": "sha256:" + "b" * 64,
                "expected_execution_source_sha256_json": json.dumps(
                    {"pipeline.py": "c" * 64}
                ),
                "max_cost_usd": 5.0, "output": root / "report.json",
            })()
            report = PRE2.verify_authorization(args)
            self.assertEqual(report["status"], "PASS_EXECUTION_AUTHORIZATION")

    def test_authorization_cannot_exceed_launch_cap(self):
        contract = REPO / "conf" / "m31_ordered_linear_pre2_preregistration.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization = root / "authorization.json"
            payload = {
                "schema_version": "1.0.0", "experiment_id": "M31_ORDERED_LINEAR_DEV_PRE2",
                "status": "AUTHORIZED_REAL_RUN",
                "scope": "ROOT17_FIT_GATE_AND_CONDITIONAL_SINGLE_ROOT18_SCORE",
                "run_id": "run-a", "contract_sha256": PRE2.core.sha256_file(contract),
                "git_commit": "a" * 40, "container_digest": "sha256:" + "b" * 64,
                "execution_source_sha256": {"pipeline.py": "c" * 64},
                "max_cost_usd": 6.0, "authorized_by": "jfct777",
                "authorized_utc": "2026-08-20T00:00:00Z",
                "explicit_user_authorization": True,
            }
            authorization.write_text(json.dumps(payload), encoding="utf-8")
            args = type("Args", (), {
                "authorization": authorization, "contract": contract,
                "run_id": "run-a", "expected_git_commit": "a" * 40,
                "container_digest": "sha256:" + "b" * 64,
                "expected_execution_source_sha256_json": json.dumps(
                    {"pipeline.py": "c" * 64}
                ),
                "max_cost_usd": 5.0, "output": root / "report.json",
            })()
            with self.assertRaisesRegex(PRE2.Pre2Error, "cost cap"):
                PRE2.verify_authorization(args)

    def test_authorization_rejects_execution_source_drift(self):
        contract = REPO / "conf" / "m31_ordered_linear_pre2_preregistration.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization = root / "authorization.json"
            payload = {
                "schema_version": "1.0.0", "experiment_id": "M31_ORDERED_LINEAR_DEV_PRE2",
                "status": "AUTHORIZED_REAL_RUN",
                "scope": "ROOT17_FIT_GATE_AND_CONDITIONAL_SINGLE_ROOT18_SCORE",
                "run_id": "run-a", "contract_sha256": PRE2.core.sha256_file(contract),
                "git_commit": "a" * 40, "container_digest": "sha256:" + "b" * 64,
                "execution_source_sha256": {"pipeline.py": "c" * 64},
                "max_cost_usd": 5.0, "authorized_by": "jfct777",
                "authorized_utc": "2026-08-20T00:00:00Z",
                "explicit_user_authorization": True,
            }
            authorization.write_text(json.dumps(payload), encoding="utf-8")
            args = type("Args", (), {
                "authorization": authorization, "contract": contract,
                "run_id": "run-a", "expected_git_commit": "a" * 40,
                "container_digest": "sha256:" + "b" * 64,
                "expected_execution_source_sha256_json": json.dumps(
                    {"pipeline.py": "d" * 64}
                ),
                "max_cost_usd": 5.0, "output": root / "report.json",
            })()
            with self.assertRaisesRegex(PRE2.Pre2Error, "source hashes differ"):
                PRE2.verify_authorization(args)

    def test_fit_boundary_rejects_forged_or_wrong_run_report(self):
        expected_sources = {"pipeline.py": "c" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            report_path.write_text(
                json.dumps({"status": "PASS_EXECUTION_AUTHORIZATION"}), encoding="utf-8",
            )
            with self.assertRaisesRegex(PRE2.Pre2Error, "report fields differ"):
                PRE2._validate_authorization_report(
                    report_path, run_id="run-a", contract_sha256="d" * 64,
                    git_commit="a" * 40, container_digest="sha256:" + "b" * 64,
                    execution_source_sha256=expected_sources,
                )
            valid = {
                "schema_version": "1.0.0", "status": "PASS_EXECUTION_AUTHORIZATION",
                "run_id": "run-b", "git_commit": "a" * 40,
                "container_digest": "sha256:" + "b" * 64,
                "contract_sha256": "d" * 64,
                "execution_source_sha256": expected_sources,
                "max_cost_usd": 5.0, "authorization_artifact_sha256": "e" * 64,
                "root18_truth_accessed": False,
            }
            report_path.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(PRE2.Pre2Error, "run ID differs"):
                PRE2._validate_authorization_report(
                    report_path, run_id="run-a", contract_sha256="d" * 64,
                    git_commit="a" * 40, container_digest="sha256:" + "b" * 64,
                    execution_source_sha256=expected_sources,
                )


if __name__ == "__main__":
    unittest.main()
