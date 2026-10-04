"""Synthetic, private fixtures only; no real cohort, graph fitting or cloud jobs."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import r02_community_evidence as community


def write_tsv(path, rows, fields=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def reference(path):
    return dict(path=str(Path(path).absolute()), sha256=community.sha256(path))


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class SyntheticFixture:
    """Factory usable by the separate Nextflow smoke test; writes only supplied root.

    `fixture = SyntheticFixture(root)` exposes `contract_path`, `contract_hash`,
    and six completely invented sample IDs. No files from a real run are read.
    """
    def __init__(self, root, units="fraction", all_unassigned=False):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples = ["fiction_B", "fiction_A", "fiction_D", "fiction_C", "fiction_E", "fiction_F"]
        self.sample_path = self.root / "samples.txt"
        self.sample_path.write_text("\n".join(self.samples) + "\n")
        self.kinship_sha = "c" * 64
        self.phis = [.0221, .0442]
        self.metadata_path = self.root / "metadata.tsv"
        self.metadata = []
        vectors = [(.2, .6, .1, .1), (.3, .5, .1, .1), (.4, .4, .1, .1),
                   (.5, .3, .1, .1), (.4, None, .1, .1), (None, None, None, None)]
        for i, sid in enumerate(self.samples):
            row = dict(ID=sid, finestructure_clusters="reference_" + str(i // 2),
                       Region="N/A" if i == 5 else "Region_" + str(i % 2), State="Unknown" if i == 4 else "State_A",
                       Cohort="Collection_" + str(i % 2), Source="Study_A", Origin="Recruitment_A",
                       City="PRIVATE_CITY_SENTINEL", clinical="PRIVATE_CLINICAL_SENTINEL")
            row.update({field: "NA" if value is None else value * (100 if units == "percent" else 1)
                        for field, value in zip(community.ANCESTRY, vectors[i])})
            self.metadata.append(row)
        self.rewrite_metadata()
        self.study_dir = self.root / "study"
        self.study_dir.mkdir()
        self.people = []
        for i, row in enumerate(self.metadata):
            self.people.append(dict(sample_id=self.samples[i], role="UNASSIGNED",
                **{field: row[field] for field in ("Cohort", "Region", "State", "Source", "Origin")},
                **{"component_phi_0.0221": ["dep_A", "dep_A", "NA", "NA", "dep_E", "dep_F"][i],
                   "component_phi_0.0442": ["dep_A", "dep_B", "NA", "NA", "dep_E", "dep_F"][i]}))
        self.study = dict(schema="r02_study_design_v1", status="COMPLETE_DESCRIPTIVE_AUDIT",
            n_samples=6, sample_ids_sha256=community.evidence.sample_hash(self.samples),
            thresholds=self.phis, kinship_audit=dict(sha256=self.kinship_sha), roles_supplied=False,
            nominal_columns=["Cohort", "Region", "State", "Source", "Origin"])
        self.rewrite_study()
        self.graphs = []
        self.graph_data = {}
        for family in ("H", "R", "C", "R_plus_C"):
            labels = [-1] * 6 if all_unassigned else [0, 0, 1, 1, -1, -1]
            if family == "C" and not all_unassigned:
                labels = [9, 8, 9, 8, -1, -1]
            self.add_graph(family, labels, all_unassigned)
        self.contract_path = self.root / "inputs.json"
        self.contract = dict(schema=community.SCHEMA, expected_samples=6,
            samples=reference(self.sample_path), study_contract=reference(self.study_dir / "study_contract.json"),
            metadata=dict(file=reference(self.metadata_path), sample_column="ID", ancestry_units=units,
                          ancestry_sum_tolerance=.001), graphs=self.graphs,
            comparisons=[dict(left="primary/R", right="primary/C", left_resolution=.5, right_resolution=.5),
                         dict(left="primary/C", right="primary/R_plus_C", left_resolution=.5, right_resolution=.5)])
        self.save_contract()

    @property
    def contract_hash(self):
        return community.sha256(self.contract_path)

    def save_contract(self):
        write_json(self.contract_path, self.contract)

    def rewrite_metadata(self):
        write_tsv(self.metadata_path, self.metadata)
        if hasattr(self, "contract"):
            self.contract["metadata"]["file"] = reference(self.metadata_path)
            self.save_contract()

    def rewrite_study(self):
        write_tsv(self.study_dir / "persons.private.tsv", self.people)
        self.study["outputs_sha256"] = {"persons.private.tsv": community.sha256(self.study_dir / "persons.private.tsv")}
        write_json(self.study_dir / "study_contract.json", self.study)
        if hasattr(self, "contract"):
            self.contract["study_contract"] = reference(self.study_dir / "study_contract.json")
            self.save_contract()

    def add_graph(self, family, labels, inactive):
        key = "L250000_G50000_N20_T1000000_U0" if family == "H" else family
        directory = self.root / ("H" if family == "H" else "primary") / key
        directory.mkdir(parents=True)
        gammas = [.5, 1.] if family == "H" else [.5]
        active = [False] * 6 if inactive else [True] * 5 + [False]
        nodes = []
        for i, sid in enumerate(self.samples):
            node = dict(sample_id=sid, degree=2 if active[i] else 0, weighted_degree=123. if active[i] else 0.)
            node.update(node_id=i) if family == "H" else node.update(active=active[i])
            nodes.append(node)
        node_name = "graph_nodes.tsv" if family == "H" else "nodes.private.tsv"
        write_tsv(directory / node_name, nodes)
        assignments = [dict(sample_id=sid, **{f"community_res_{g:g}": labels[i] for g in gammas})
                       for i, sid in enumerate(self.samples)]
        write_tsv(directory / "leiden_assignments.tsv", assignments)
        settings = dict(expected_samples=6, resolutions=gammas, min_community_size=2)
        manifest = dict(status="COMPLETE_NO_POSITIVE_EDGES" if inactive else "COMPLETE_DESCRIPTIVE",
                        chromosomes=community.AUTOSOMES, n_cohort=6, n_active=sum(active), n_edges=sum(active),
                        outputs_sha256={name: community.sha256(directory / name)
                                        for name in (node_name, "leiden_assignments.tsv")})
        if family == "H":
            manifest.update(configuration=dict(config_id=key), no_founder_classification=True, parameters=settings,
                            source_sha256=dict(sample_ids=community.evidence.sample_hash(self.samples)))
        else:
            manifest.update(graph=family, weight_units="dimensionless", settings=settings)
        write_json(directory / "manifest.json", manifest)
        specs = []
        for i, phis in enumerate(([phi] for phi in self.phis) if family == "H" else [self.phis]):
            kd = self.root / "diagnostics" / key / str(i)
            kd.mkdir(parents=True)
            base = dict(n_cohort=6, n_active=sum(active), **{"config_id" if family == "H" else "graph": key})
            graph_rows = [dict(**base, kinship_threshold=phi, native_metric="not_homogenized") for phi in phis]
            partition_rows = [dict(**base, kinship_threshold=phi, resolution=g,
                n_assigned=sum(label >= 0 for label in labels), n_communities=len(set(labels) - {-1}))
                for phi in phis for g in gammas]
            write_tsv(kd / "graph_kinship_summary.tsv", graph_rows)
            write_tsv(kd / "partition_kinship_summary.tsv", partition_rows)
            km = dict(status="COMPLETE_DESCRIPTIVE_EDGE_KINSHIP" if family == "H"
                           else "COMPLETE_DESCRIPTIVE_WEIGHTED_EDGE_KINSHIP",
                pcrelate_audit=dict(sha256=self.kinship_sha), no_reclustering=True, no_new_kinship=True, no_pvalues=True,
                input_sha256={f"{key}/{name}": community.sha256(directory / name)
                              for name in ("manifest.json", "leiden_assignments.tsv")},
                outputs_sha256={name: community.sha256(kd / name)
                               for name in ("graph_kinship_summary.tsv", "partition_kinship_summary.tsv")})
            write_json(kd / "manifest.json", km)
            specs.append(reference(kd / "manifest.json"))
        spec = dict(id=("H/" if family == "H" else "primary/") + key, family=family,
                    manifest=reference(directory / "manifest.json"), kinship_manifests=specs)
        self.graphs.append(spec)
        self.graph_data[family] = dict(directory=directory, spec=spec, manifest=manifest, labels=labels,
                                       assignments=assignments, nodes=nodes)

    def run(self, name="output"):
        return community.run(self.contract_path, self.contract_hash, self.root / name, 512)


class CommunityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fx = SyntheticFixture(self.temp.name)

    def test_complete_manifest_preserves_scales_and_privacy(self):
        result = self.fx.run()
        self.assertEqual(result["status"], community.STATUS)
        self.assertEqual(result["incremental_rare_utility"], "NOT_ESTIMATED")
        self.assertEqual((result["n_graphs"], result["n_partitions"]), (4, 5))
        self.assertEqual(result["graph_ledger"][0]["resolutions"], [.5, 1.])
        self.assertEqual(result["graph_ledger"][1]["resolutions"], [.5])
        self.assertFalse(result["public_distribution_allowed"])
        output = self.fx.root / "output"
        for path in output.iterdir():
            text = path.read_text()
            self.assertNotIn("PRIVATE_CITY_SENTINEL", text)
            self.assertNotIn("PRIVATE_CLINICAL_SENTINEL", text)
            for sid in self.fx.samples:
                self.assertNotIn(sid, text)
        for name, digest in result["outputs_sha256"].items():
            self.assertEqual(community.sha256(output / name), digest)
        for name, record in result["input_files"].items():
            self.assertEqual(community.sha256(name), record["sha256"])
        reused = json.loads((output / "kinship_reused.json").read_text())
        self.assertTrue(any(row["native_fields"].get("native_metric") == "not_homogenized" for row in reused))

    def test_denominators_missingness_and_observation_scopes(self):
        result = self.fx.run()
        output = self.fx.root / "output"
        scopes = [r for r in read_rows(output / "scopes.tsv") if r["graph_id"] == "primary/R"]
        self.assertEqual({r["scope"]: int(r["n_samples"]) for r in scopes if r["scope"] != "community"},
                         dict(cohort=6, assigned=4, unassigned=2, active_unassigned=1, isolated=1))
        for r in scopes:
            self.assertAlmostEqual(float(r["fraction_cohort"]), int(r["n_samples"]) / 6)
        coverage = {r["field"]: r for r in result["metadata_audit"]["coverage"]}
        self.assertEqual(coverage["Region"]["n_measured"], 5)  # N/A is normalized consistently.
        categorical = [r for r in read_rows(output / "categorical.tsv")
                       if r["graph_id"] == "primary/R" and r["scope"] == "cohort" and r["field"] == "Region"]
        self.assertEqual(sum(int(r["n"]) for r in categorical), 6)
        self.assertEqual([int(r["n"]) for r in categorical if r["is_missing"] == "True"], [1])
        ancestry = [r for r in read_rows(output / "ancestry.tsv")
                    if r["graph_id"] == "primary/R" and r["scope"] == "unassigned"
                    and r["field"] == community.ANCESTRY[0]]
        per_field, complete = {r["observation_scope"]: r for r in ancestry}.values()
        self.assertEqual((per_field["n_observed"], per_field["field_n_missing"], per_field["mean_fraction"]), ("1", "1", "0.4"))
        self.assertEqual((complete["n_observed"], complete["field_n_missing"], complete["n_unavailable_for_scope"],
                          complete["n_excluded_incomplete_vector"], complete["mean_fraction"]), ("0", "1", "2", "1", "NA"))
        observed = [r for r in read_rows(output / "ancestry.tsv")
                    if r["graph_id"] == "primary/R" and r["scope"] == "community" and r["community_local"] == "0"
                    and r["field"] == community.ANCESTRY[0] and r["observation_scope"] == "field_observed"][0]
        self.assertAlmostEqual(float(observed["mean_fraction"]), .25)
        self.assertAlmostEqual(float(observed["std_fraction"]), .05)
        self.assertEqual(observed["std_ddof"], "0")

    def test_component_unknown_is_not_zero_and_both_phi(self):
        self.fx.run()
        rows = [r for r in read_rows(self.fx.root / "output/components.tsv")
                if r["graph_id"] == "primary/R" and r["scope"] == "community"]
        unknown = [r for r in rows if r["community_local"] == "1"]
        self.assertEqual({float(r["kinship_threshold"]) for r in unknown}, set(self.fx.phis))
        for row in unknown:
            self.assertEqual((row["n_component_known"], row["n_component_missing"]), ("0", "2"))
            self.assertEqual((row["max_component_n"], row["max_observed_component_fraction_all"], row["component_status"]),
                             ("NA", "NA", "NO_EVALUABLE"))
        known = [r for r in rows if r["community_local"] == "0"]
        self.assertEqual({r["kinship_threshold"]: r["n_components"] for r in known}, {"0.0221": "1", "0.0442": "2"})
        cohort = [r for r in read_rows(self.fx.root / "output/components.tsv")
                  if r["graph_id"] == "primary/R" and r["scope"] == "cohort" and r["kinship_threshold"] == "0.0221"][0]
        self.assertEqual((cohort["n_component_known"], cohort["n_component_missing"], cohort["max_component_n"]), ("4", "2", "2"))
        self.assertAlmostEqual(float(cohort["max_observed_component_fraction_all"]), 2 / 6)
        self.assertAlmostEqual(float(cohort["max_component_fraction_known"]), 2 / 4)
        self.assertEqual(cohort["component_status"], "DESCRIPTIVE_PARTIAL")

    def test_percent_and_fraction_are_equivalent(self):
        a = self.fx.run()
        percent = SyntheticFixture(self.fx.root / "percent", units="percent")
        b = percent.run()
        self.assertEqual(a["outputs_sha256"]["ancestry.tsv"], b["outputs_sha256"]["ancestry.tsv"])
        self.assertEqual(b["metadata_audit"]["scale_divisor"], 100.)

    def test_common_assigned_ari_and_crosstab(self):
        self.fx.run()
        comparisons = read_rows(self.fx.root / "output/comparisons.tsv")
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(comparisons[0]["n_both_assigned"], "4")
        self.assertAlmostEqual(float(comparisons[0]["ari"]), -.5)
        cross = [r for r in read_rows(self.fx.root / "output/crosstab.tsv") if r["left_graph"] == "primary/R"]
        self.assertEqual(sum(int(r["n"]) for r in cross), 6)
        self.assertEqual([int(r["n"]) for r in cross if r["either_unassigned"] == "True"], [2])

    def test_different_support_and_degenerate_ari(self):
        graphs = [dict(id="A", labels={.5: [0, 0, 1, 1, -1, -1]}),
                  dict(id="B", labels={.5: [-1, 2, 3, 3, 2, -1]})]
        rows = list(community.compare_partitions([dict(left="A", right="B", left_resolution=.5, right_resolution=.5)], graphs))
        comparison = rows[0][1]
        self.assertEqual((comparison["n_both_assigned"], comparison["n_left_only_assigned"],
                          comparison["n_right_only_assigned"], comparison["n_neither_assigned"]), (3, 1, 1, 1))
        self.assertEqual(comparison["ari"], 1.)  # label permutation, common support only.
        graphs[1]["labels"][.5] = [2, 2, 2, 2, -1, -1]
        row = next(community.compare_partitions([dict(left="A", right="B", left_resolution=.5, right_resolution=.5)], graphs))[1]
        self.assertIsNone(row["ari"])
        self.assertIn("NO_EVALUABLE", row["status"])

    def test_all_isolated_no_assignments(self):
        empty = SyntheticFixture(self.fx.root / "empty", all_unassigned=True)
        empty.run()
        scopes = read_rows(empty.root / "output/scopes.tsv")
        self.assertFalse(any(r["scope"] == "community" for r in scopes))
        self.assertTrue(all(r["n_assigned"] == "0" for r in scopes))
        self.assertTrue(all(r["ari"] == "NA" for r in read_rows(empty.root / "output/comparisons.tsv")))

    def test_no_overwrite(self):
        self.fx.run()
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.fx.run()

    def test_contract_and_artifact_tampering(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            community.run(self.fx.contract_path, "0" * 64, self.fx.root / "output", 512)
        self.fx.metadata_path.write_text(self.fx.metadata_path.read_text() + "tamper\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.fx.run()
        self.assertFalse((self.fx.root / "output").exists())

    def test_incomplete_kinship_and_wrong_binding_rejected(self):
        self.fx.graphs[0]["kinship_manifests"] = self.fx.graphs[0]["kinship_manifests"][:1]
        self.fx.save_contract()
        with self.assertRaisesRegex(ValueError, "Incomplete reused kinship grid"):
            self.fx.run()
        spec = self.fx.graphs[0]["kinship_manifests"][0]
        path = Path(spec["path"])
        manifest = json.loads(path.read_text())
        first = next(iter(manifest["input_sha256"]))
        manifest["input_sha256"][first] = "d" * 64
        write_json(path, manifest)
        spec["sha256"] = community.sha256(path)
        self.fx.save_contract()
        with self.assertRaisesRegex(ValueError, "not bound uniquely"):
            self.fx.run()

    def test_units_and_partial_sum_rejected_without_guessing(self):
        self.fx.contract["metadata"]["ancestry_units"] = "percent"
        self.fx.save_contract()
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.fx.run()
        self.fx.contract["metadata"]["ancestry_units"] = "fraction"
        self.fx.metadata[4][community.ANCESTRY[0]] = 1.
        self.fx.rewrite_metadata()
        with self.assertRaisesRegex(ValueError, "exceed one"):
            self.fx.run()

    def test_metadata_conflicts_and_study_disagreement(self):
        self.fx.metadata[0]["Cohort"] = "different_collection"
        self.fx.rewrite_metadata()
        with self.assertRaisesRegex(ValueError, "differs from authenticated study"):
            self.fx.run()
        self.fx.metadata.append({**self.fx.metadata[0], "clinical": "conflicting_private_value"})
        self.fx.rewrite_metadata()
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            self.fx.run()

    def test_missing_metadata_column_rejected(self):
        del self.fx.metadata[0][community.ANCESTRY[0]]
        for row in self.fx.metadata[1:]:
            del row[community.ANCESTRY[0]]
        self.fx.rewrite_metadata()
        with self.assertRaisesRegex(ValueError, "Missing authorized metadata column"):
            self.fx.run()

    def test_infinite_ancestry_rejected_and_nan_is_declared_missing(self):
        self.fx.metadata[0][community.ANCESTRY[0]] = "inf"
        self.fx.rewrite_metadata()
        with self.assertRaisesRegex(ValueError, "outside declared unit range"):
            self.fx.run()
        self.fx.metadata[0][community.ANCESTRY[0]] = "NaN"
        self.fx.rewrite_metadata()
        result = self.fx.run()
        coverage = {row["field"]: row["n_measured"] for row in result["metadata_audit"]["coverage"]}
        self.assertEqual(coverage[community.ANCESTRY[0]], 4)

    def test_reordered_cohort_and_declared_units_rejected(self):
        self.fx.people[0], self.fx.people[1] = self.fx.people[1], self.fx.people[0]
        self.fx.rewrite_study()
        with self.assertRaisesRegex(ValueError, "order/coverage differs"):
            self.fx.run()
        self.fx.people[0], self.fx.people[1] = self.fx.people[1], self.fx.people[0]
        self.fx.rewrite_study()
        self.fx.contract["metadata"]["ancestry_units"] = "infer"
        self.fx.save_contract()
        with self.assertRaisesRegex(ValueError, "Explicit fraction or percent"):
            self.fx.run()

    def test_exact_duplicate_metadata_collapses_without_new_person(self):
        self.fx.metadata.append(dict(self.fx.metadata[0]))
        self.fx.rewrite_metadata()
        result = self.fx.run()
        self.assertEqual(result["n_cohort"], 6)
        self.assertEqual(result["metadata_audit"]["exact_duplicate_rows"], 1)

    def test_duplicate_graph_and_comparison_rejected(self):
        self.fx.contract["graphs"].append(self.fx.contract["graphs"][0])
        self.fx.save_contract()
        with self.assertRaisesRegex(ValueError, "Duplicate graph ID"):
            self.fx.run()
        self.fx.contract["graphs"].pop()
        self.fx.contract["comparisons"].append(self.fx.contract["comparisons"][0])
        self.fx.save_contract()
        with self.assertRaisesRegex(ValueError, "duplicate partition comparison"):
            self.fx.run()

    def test_json_duplicate_key_and_memory_guard(self):
        path = self.fx.root / "duplicate.json"
        path.write_text('{"a": 1, "a": 2}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            community.strict_json(path)
        path.write_text('{"a": NaN}')
        with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
            community.strict_json(path)
        with self.assertRaisesRegex(ValueError, "RSS exceeds"):
            community.run(self.fx.contract_path, self.fx.contract_hash, self.fx.root / "output", 1)

    def test_cli(self):
        result = subprocess.run([sys.executable, str(ROOT / "bin/r02_community_evidence.py"),
            "--inputs", str(self.fx.contract_path), "--expected-inputs-sha256", self.fx.contract_hash,
            "--output-dir", str(self.fx.root / "cli"), "--max-memory-mb", "512"],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], community.STATUS)


if __name__ == "__main__":
    unittest.main()
