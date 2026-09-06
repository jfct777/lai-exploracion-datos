"""Local synthetic zero-origin regressions; no project results or genotype I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import m39_zero_origin_audit as audit


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_fixture(root: Path) -> dict[str, Path]:
    """Six artificial people; exact, cM-tied and interpolated synthetic loci.

    Historical production functions construct expectations. The auditor must
    independently recover them, including float32 storage and exact-key priority.
    """
    from m37_trace_core import baseline_to_states
    from m39_bind_anchor_truth import project_probabilities

    root.mkdir(parents=True, exist_ok=True)
    paths = {role: root / (role + (".json" if role == "binding_receipt" else ".npz"))
             for role in audit.INPUT_ROLES}
    paths.update(manifest=root / "manifest.json", output=root / "audit.json")
    keys = np.asarray([hashlib.sha256(f"fixture-only-{i}".encode()).hexdigest() for i in range(6)], dtype="S64")
    common = {"chrom": np.full(3, 22, dtype=np.uint8), "pos": np.array([15, 25, 35], dtype=np.int64),
              "ref": np.array([b"A"] * 3, dtype="S1"), "alt": np.array([b"C"] * 3, dtype="S1"),
              "coords": np.array([.5, 1., 1.5]), "locus_id": np.arange(3, dtype=np.uint64),
              "anchor_indices": np.arange(3, dtype=np.int64), "state_names": np.array(audit.STATE_NAMES, dtype="S2")}
    projections, projection_audit = {}, {}
    rng = np.random.default_rng(41)
    for source, positions, cm in (("fminus", [10, 20, 30, 40, 50], [0., 1., 1., 2., 3.]),
                                  ("full", [10, 15, 20, 25, 30, 35, 40, 50], [0., .5, 1., 1., 1., 1.5, 2., 3.])):
        p = rng.random((6, 2, len(positions), 3)).astype(np.float32)
        p /= p.sum(-1, keepdims=True)
        p[:2, :, :, :] = [1., 0., 0.]
        p[4, :, :, :] = [0., 1., 0.]
        p[5, :, :, :] = [1., 1e-30, 0.]
        source_data = {"sample_key_sha256": keys, "marker_chrom": np.full(len(positions), 22, dtype=np.uint8),
                       "marker_pos": np.array(positions, dtype=np.int64), "marker_ref": np.array([b"A"] * len(positions)),
                       "marker_alt": np.array([b"C"] * len(positions)), "F0": p}
        np.savez_compressed(paths[source], **source_data)
        np.savez_compressed(paths[source + "_cm"], marker_cM=np.asarray(cm))
        exact = np.array([positions.index(x) if x in positions else -1 for x in common["pos"]])
        projections[source], projection_audit[source] = project_probabilities(baseline_to_states(p), np.array(cm), common["coords"], exact)
    for part, rows in (("development", np.arange(4)), ("score", np.arange(4, 6))):
        package = dict(common, sample_key_sha256=keys[rows], source_indices=rows,
                       baseline=projections["fminus"][rows], full_baseline=projections["full"][rows],
                       truth_state=np.full((len(rows), 3), 3, dtype=np.int8))
        if part == "development":
            package.update(train_indices=np.arange(2), select_indices=np.arange(2, 4))
        else:
            package.update(score_indices=np.arange(2))
        np.savez_compressed(paths[part], **package)
    binding = {"schema_version": "m39-anchor-truth-v1", "scope": "exploratory_chr22_R0_FIT_no_training",
               "decision": "PASS_EXPLORATORY_EXACT_ANCHOR_BINDING", "people": 6, "anchors": 3,
               "boundaries": {"VALID_TEST_opened": False, "source_TEST_opened": False},
               "inputs": {role: {"sha256": audit.sha256(paths[role])} for role in ("fminus", "fminus_cm", "full", "full_cm")},
               "outputs": {part + ".npz": {"sha256": audit.sha256(paths[part]), "people": 4 if part == "development" else 2,
                            "roles": {"TRAIN": 2 if part == "development" else 0, "SELECT": 2 if part == "development" else 0,
                                      "SCORE": 0 if part == "development" else 2}} for part in ("development", "score")},
               "projection_audit": projection_audit, "code_sha256": {}}
    write_json(paths["binding_receipt"], binding)
    manifest = {"schema_version": audit.SCHEMA, "scope": audit.SCOPE,
                "inputs": {role: {"path": paths[role].name, "sha256": audit.sha256(paths[role])} for role in audit.INPUT_ROLES}}
    write_json(paths["manifest"], manifest)
    return paths


def change_npz(path: Path, **updates) -> None:
    data = audit.load_npz(path)
    data.update(updates)
    np.savez_compressed(path, **data)


def resign(paths: dict, role: str) -> None:
    binding = json.loads(paths["binding_receipt"].read_text())
    if role in ("development", "score"):
        binding["outputs"][role + ".npz"]["sha256"] = audit.sha256(paths[role])
    elif role != "binding_receipt":
        binding["inputs"][role]["sha256"] = audit.sha256(paths[role])
    write_json(paths["binding_receipt"], binding)
    manifest = json.loads(paths["manifest"].read_text())
    for name in (role, "binding_receipt"):
        manifest["inputs"][name]["sha256"] = audit.sha256(paths[name])
    write_json(paths["manifest"], manifest)


class ProbabilityStages(unittest.TestCase):
    def test_symmetric_six_states(self):
        p = np.array([[[[.2, .3, .5]], [[.6, .1, .3]]]], dtype=np.float32)
        high, low = audit.six_states(p)
        np.testing.assert_allclose(high, [[[.12, .20, .36, .03, .14, .15]]], rtol=1e-6)
        np.testing.assert_array_equal(low, audit.six_states(p[:, ::-1])[1])

    def test_product_float32_underflow_is_not_source_absence(self):
        p = np.array([[[[1., 1e-30, 0.]], [[1., 1e-30, 0.]]]], dtype=np.float32)
        high, low = audit.six_states(p)
        self.assertGreater(high[0, 0, 3], 0)
        self.assertEqual(low[0, 0, 3], 0)
        plan = audit.projection_plan(np.array([0.]), np.array([0.]), np.array([0]))
        _, labels = audit.classify_zeros(high, low, plan)
        self.assertEqual(labels[0, 0, 3], 1)
        self.assertEqual(labels[0, 0, 5], 0)

    def test_final_projection_cast_underflow(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        states = np.array([[[1., tiny, 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]]], dtype=np.float32)
        plan = audit.projection_plan(np.array([0., 1.]), np.array([.9]), np.array([-1]))
        stored, labels = audit.classify_zeros(states.astype(float), states, plan)
        self.assertEqual(stored[0, 0, 1], 0)
        self.assertEqual(labels[0, 0, 1], 3)

    def test_exact_key_precedes_cm_ties(self):
        states = np.eye(6, dtype=np.float32)[[0, 1]][None]
        plan = audit.projection_plan(np.array([1., 1.]), np.array([1., 1.]), np.array([0, -1]))
        out, _ = audit.project(states, plan)
        np.testing.assert_array_equal(out[0, 0], states[0, 0])
        np.testing.assert_array_equal(out[0, 1], [.5, .5, 0, 0, 0, 0])

    def test_positive_contributor_rescues_zero(self):
        states = np.eye(6, dtype=np.float32)[[0, 1]][None]
        plan = audit.projection_plan(np.array([0., 1.]), np.array([.5]), np.array([-1]))
        out, labels = audit.classify_zeros(states.astype(float), states, plan)
        np.testing.assert_array_equal(out[0, 0, :2], [.5, .5])
        np.testing.assert_array_equal(labels[0, 0, :2], [-1, -1])
        classes = audit.support_categories(states.astype(float), plan, labels)
        np.testing.assert_array_equal(classes[0, 0, :2], [2, 2])
        np.testing.assert_array_equal(classes[0, 0, 2:], [0, 0, 0, 0])

    def test_all_positive_support_class(self):
        states = np.full((1, 2, 6), 1 / 6, dtype=np.float32)
        plan = audit.projection_plan(np.array([0., 1.]), np.array([.5]), np.array([-1]))
        _, labels = audit.classify_zeros(states.astype(float), states, plan)
        np.testing.assert_array_equal(audit.support_categories(states.astype(float), plan, labels), np.full((1, 1, 6), 3))

    def test_projection_float64_underflow_class(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        states = np.array([[[1., 0., 0., 0., 0., 0.], [1., tiny, 0., 0., 0., 0.]]], dtype=np.float32)
        plan = audit.projection_plan(np.array([0., 1.]), np.array([1e-300]), np.array([-1]))
        _, labels = audit.classify_zeros(states.astype(float), states, plan)
        self.assertEqual(labels[0, 0, 1], 2)

    def test_clamps_and_zero_weight_support(self):
        states = np.eye(6, dtype=np.float32)[[0, 1]][None]
        plan = audit.projection_plan(np.array([0., 1.]), np.array([-1., 2.]), np.array([-1, -1]))
        out, _ = audit.project(states, plan)
        np.testing.assert_array_equal(out[0], states[0])
        plan = [{"groups": [np.array([0]), np.array([1])], "weights": [1., 0.]}]
        _, labels = audit.classify_zeros(states.astype(float), states, plan)
        self.assertEqual(labels[0, 0, 1], 0)

    def test_nan_and_negative_probabilities_rejected(self):
        for bad in (np.nan, -.1):
            p = np.ones((1, 2, 1, 3), dtype=np.float32) / 3
            p[0, 0, 0, 0] = bad
            with self.assertRaises(audit.ZeroOriginAuditError):
                audit.six_states(p)

    def test_coordinate_drift_rejected(self):
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "exact-key cM"):
            audit.projection_plan(np.array([0., 1.]), np.array([.5]), np.array([0]))


class AuthenticatedAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_audit(self, **kwargs):
        return audit.audit(self.paths["manifest"], self.root, self.paths["output"], **kwargs)

    def test_complete_replay_aggregate_and_input_preservation(self):
        hashes = {role: audit.sha256(self.paths[role]) for role in audit.INPUT_ROLES}
        result = self.run_audit(chunk_people=1)
        self.assertEqual(result["decision"], "PASS_EXACT_STORED_BASELINE_REPLAY")
        self.assertFalse(result["parser_zero_creation_assessed"])
        self.assertEqual(result["raw_internal_probability"], "unavailable")
        for source in ("fminus", "full"):
            record = result["results"][source]
            self.assertEqual(record["max_absolute_replay_error"], 0)
            self.assertEqual(record["roles"]["SCORE"]["true_state_zeros"], 3)
            self.assertEqual(record["roles"]["SCORE"]["true_state_zero_origins"][audit.ORIGINS[1]], 3)
        self.assertEqual(hashes, {role: audit.sha256(self.paths[role]) for role in audit.INPUT_ROLES})
        output_text = self.paths["output"].read_text()
        for key in audit.load_npz(self.paths["fminus"])["sample_key_sha256"]:
            self.assertNotIn(key.decode(), output_text)
        self.assertNotIn(str(self.root), output_text)
        self.assertEqual(self.paths["output"].stat().st_mode & 0o777, 0o400)

    def test_chunk_invariance(self):
        first = self.run_audit(chunk_people=1)
        other = self.root / "other.json"
        second = audit.audit(self.paths["manifest"], self.root, other, chunk_people=6)
        self.assertEqual(first, second)

    def test_refuse_overwrite(self):
        self.paths["output"].write_text("preserve")
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "overwrite"):
            self.run_audit()
        self.assertEqual(self.paths["output"].read_text(), "preserve")

    def test_hash_mismatch_before_npz_load(self):
        self.paths["score"].write_bytes(b"tamper")
        with mock.patch.object(audit, "load_npz", side_effect=AssertionError("must authenticate first")):
            with self.assertRaisesRegex(audit.ZeroOriginAuditError, "SHA-256 mismatch"):
                self.run_audit()
        self.assertFalse(self.paths["output"].exists())

    def test_binder_chain_mismatch_rejected(self):
        manifest = json.loads(self.paths["manifest"].read_text())
        self.paths["score"].write_bytes(b"changed-but-resigned-manifest-only")
        manifest["inputs"]["score"]["sha256"] = audit.sha256(self.paths["score"])
        write_json(self.paths["manifest"], manifest)
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "authenticated by binder"):
            self.run_audit()

    def test_input_inventory_forbids_genotypes(self):
        manifest = json.loads(self.paths["manifest"].read_text())
        manifest["inputs"]["genotypes"] = manifest["inputs"]["score"]
        write_json(self.paths["manifest"], manifest)
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "inventory"):
            self.run_audit()

    def test_changed_probability_fails_exact_replay(self):
        p = audit.load_npz(self.paths["score"])["baseline"]
        p[0, 0] = [1., 0., 0., 0., 0., 0.]
        change_npz(self.paths["score"], baseline=p)
        resign(self.paths, "score")
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "replay is not exact"):
            self.run_audit()

    def test_fractional_truth_rejected_before_cast(self):
        truth = audit.load_npz(self.paths["score"])["truth_state"].astype(float) + .1
        change_npz(self.paths["score"], truth_state=truth)
        resign(self.paths, "score")
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "truth labels"):
            self.run_audit()

    def test_sample_overlap_rejected(self):
        keys = audit.load_npz(self.paths["score"])["sample_key_sha256"]
        keys[0] = audit.load_npz(self.paths["development"])["sample_key_sha256"][0]
        change_npz(self.paths["score"], sample_key_sha256=keys)
        resign(self.paths, "score")
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "duplicated"):
            self.run_audit()

    def test_role_overlap_rejected(self):
        change_npz(self.paths["development"], select_indices=np.array([1, 2]))
        resign(self.paths, "development")
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "role indices overlap"):
            self.run_audit()

    def test_anchor_allele_drift_rejected(self):
        change_npz(self.paths["score"], alt=np.array([b"G"] * 3))
        resign(self.paths, "score")
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "anchor axes differ"):
            self.run_audit()

    def test_scope_cannot_reopen_valid_test(self):
        binding = json.loads(self.paths["binding_receipt"].read_text())
        binding["boundaries"]["VALID_TEST_opened"] = True
        write_json(self.paths["binding_receipt"], binding)
        resign(self.paths, "binding_receipt")
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "scope differs"):
            self.run_audit()

    def test_path_escape_rejected(self):
        manifest = json.loads(self.paths["manifest"].read_text())
        manifest["inputs"]["score"]["path"] = "../score.npz"
        write_json(self.paths["manifest"], manifest)
        with self.assertRaisesRegex(audit.ZeroOriginAuditError, "escapes root"):
            self.run_audit()

    def test_cli_fixture(self):
        completed = subprocess.run([sys.executable, str(Path(audit.__file__)), "--manifest", str(self.paths["manifest"]),
                                    "--input-root", str(self.root), "--output", str(self.paths["output"])],
                                   check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(completed.stdout)["decision"], "PASS_EXACT_STORED_BASELINE_REPLAY")


if __name__ == "__main__":
    unittest.main()
