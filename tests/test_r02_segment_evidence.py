"""Small synthetic M14.2 contracts, indexed readers, masks and CLI checks."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pysam

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import r02_segment_evidence as local
import test_r02_genomic_pair_evidence as pair_fixture


class SegmentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = ["a", "b", "c"]
        self.sample_file = self.root / "samples.ids"
        self.sample_file.write_text("\n".join(self.samples) + "\n")
        # Reuse the tested source-minor fixture, not a second allele convention.
        self.vcf = self.index_vcf(pair_fixture.PairEvidenceTests.write_vcf(self))
        self.chains = self.root / "candidate_chains.tsv.gz"
        self.configurations = self.root / "configuration_summary.tsv"
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [
            ["chr22", "a", "b", 100, 100, 100, 1, 1],
            ["22", "b", "a", 100, 100, 100, 1, 1],  # exact duplicate/reversed pair
            ["22", "a", "b", 200, 100, 100, 1, 1],
        ])
        self.write_table(self.configurations, local.CONFIG_COLUMNS, [
            ["L1_G100_N1", 100, 1, 1, "1", 1], ["L1_G100_N2", 100, 1, 2, "2", 1],
            ["L1_G200_N1", 200, 1, 1, "1", 1],
        ])

    def index_vcf(self, path):
        compressed = Path(str(path) + ".gz")
        pysam.tabix_compress(str(path), str(compressed), force=True)
        pysam.tabix_index(str(compressed), preset="vcf", force=True)
        return compressed

    @staticmethod
    def write_table(path, columns, rows):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)

    def args(self, output="out", extra=()):
        return local.argument_parser().parse_args([
            "--chains", str(self.chains), "--configurations", str(self.configurations),
            "--rare-vcf", str(self.vcf), "--samples", str(self.sample_file),
            "--chromosome", "22", "--genome-build", "GRCh38",
            "--expected-source-samples", "101", "--output-dir", str(self.root / output),
            *extra])

    def rows(self, output="out"):
        return list(local.table_rows(self.root / output / "segment_evidence.tsv.gz", local.EVIDENCE_COLUMNS))

    def contract(self, name, schema, **kwargs):
        path = self.root / name
        path.write_text(json.dumps(dict(schema=schema, genome_build="GRCh38",
                                      coordinate_system=local.COORDINATES, **kwargs)))
        return path

    def spanning_chain(self):
        # Construct real M14 chains with two shared sites and singleton/missing
        # sites between them. Only the endpoints' source GT/RD are changed.
        raw = self.root / "spanning.vcf"
        with pysam.VariantFile(str(self.vcf)) as source, pysam.VariantFile(str(raw), "w", header=source.header) as sink:
            for record in source:
                if record.pos == 500:
                    record.samples["a"]["GT"], record.samples["a"]["RD"] = (0, 1), 1
                    record.samples["b"]["GT"], record.samples["b"]["RD"] = (0, 1), 1
                    for sample in ("other_0", "other_1"):
                        record.samples[sample]["GT"], record.samples[sample]["RD"] = (0, 0), 0
                sink.write(record)
        self.vcf = self.index_vcf(raw)
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [["22", "a", "b", 500, 100, 500, 401, 2]])
        self.write_table(self.configurations, local.CONFIG_COLUMNS, [
            ["L401_G500_N2", 500, 401, 2, "2", 2], ["L1000_G500_N3", 500, 1000, 3, "3", 3]])

    def test_native_tables_deduplicate_and_link_without_zero_configs_lost(self):
        result = local.run(self.args())
        self.assertEqual(result["checks"]["counts"]["duplicate_chain_rows"], 1)
        self.assertEqual(result["checks"]["counts"]["unique_chains"], 2)
        self.assertEqual(result["checks"]["counts"]["configuration_links"], 2)
        self.assertEqual(len(list(local.table_rows(self.root / "out/configurations.tsv", local.CONFIG_COLUMNS))), 3)
        rows = self.rows()
        self.assertEqual([(r["I"], r["U"], r["Q"], r["J"]) for r in rows], [("1", "1", "1", "1.0")] * 2)
        self.assertTrue(all(r["O"] == r["Q_C"] == r["callable_bp"] == r["ibd_union_bp"] == "NA" for r in rows))
        self.assertTrue(all(r["ibd_status"].startswith("NO_EVALUABLE") for r in rows))
        for name, digest in result["outputs_sha256"].items():
            self.assertEqual(local.sha256(self.root / "out" / name), digest)
        self.assertEqual(result["sample_ids_sha256"], local.sha256(self.sample_file))
        self.assertEqual(result["source_rare_contract"]["dnabr_rare_cohort_n_samples"], 101)

    def test_missing_and_single_carrier_are_retained_across_chunks_blocks(self):
        self.spanning_chain()
        local.run(self.args(extra=["--block-bp", "150", "--chunk-sites", "1"]))
        row = self.rows()[0]
        self.assertEqual([int(row[k]) for k in ("I", "U", "Q", "rare_catalog_sites", "rare_missing_gt_count")], [2, 3, 4, 5, 1])
        self.assertEqual(row["max_shared_gap_bp"], "400")
        self.assertAlmostEqual(float(row["shared_site_density_per_bp"]), 2 / 401)
        self.assertAlmostEqual(float(row["J"]), 2 / 3)
        local.run(self.args("other", ["--block-bp", "1000", "--chunk-sites", "100"]))
        self.assertEqual(self.rows(), self.rows("other"))

    def test_fixed_source_ref_not_reoriented_in_subset(self):
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [["22", "a", "c", 500, 300, 300, 1, 1]])
        self.write_table(self.configurations, local.CONFIG_COLUMNS, [["L1_G500_N1", 500, 1, 1, "1", 1]])
        # a has 1/1 at the REF-minor site, so this false ALT-sharing chain fails.
        with self.assertRaisesRegex(ValueError, "source rare genotypes disagree"):
            local.run(self.args())
        self.assertFalse((self.root / "out/manifest.json").exists())

    def test_prefix_counts_match_bruteforce_and_zero_union(self):
        rng = np.random.default_rng(23)
        g = rng.integers(-1, 3, size=(37, 4), dtype=np.int8)
        positions = list(range(100, 137))
        rows = []
        for a, b in (("a", "b"), ("a", "c"), ("c", "d")):
            for lo, hi in ((100, 136), (105, 109), (110, 130)):
                rows.append(dict(sample_a=a, sample_b=b, start_pos=lo, end_pos=hi,
                    I=0, U=0, Q=0, rare_catalog_sites=0, rare_missing_gt_count=0,
                    first_shared=None, last_shared=None, max_shared_gap=0))
        order = dict(zip("abcd", range(4)))
        local.count_chunk(rows, positions, g, order, "rare")
        for row in rows:
            a, b = g[row["start_pos"]-100:row["end_pos"]-99, order[row["sample_a"]]], g[row["start_pos"]-100:row["end_pos"]-99, order[row["sample_b"]]]
            valid = (a >= 0) & (b >= 0)
            self.assertEqual(row["I"], int((valid & (a > 0) & (b > 0)).sum()))
            self.assertEqual(row["U"], int((valid & ((a > 0) | (b > 0))).sum()))
            self.assertEqual(row["Q"], int(valid.sum()))
            self.assertEqual(row["rare_missing_gt_count"], int((a < 0).sum() + (b < 0).sum()))
        row = dict(sample_a="a", sample_b="b", start_pos=1, end_pos=2, I=0, U=0, Q=0,
                   rare_catalog_sites=0, rare_missing_gt_count=0, first_shared=None, last_shared=None, max_shared_gap=0)
        local.count_chunk([row], [1, 2], [[0, 0], [-1, 0]], {"a": 0, "b": 1}, "rare")
        self.assertEqual((row["I"], row["U"], row["Q"]), (0, 0, 1))

    def test_rare_source_contract_and_invalid_rd_fail(self):
        self.vcf = self.index_vcf(pair_fixture.PairEvidenceTests.write_vcf(self, bad_rd=True))
        with self.assertRaisesRegex(ValueError, "RD disagrees"):
            local.run(self.args())
        self.assertFalse((self.root / "out/manifest.json").exists())

    def test_original_multiallelic_rare_record_fail(self):
        self.vcf = self.index_vcf(pair_fixture.PairEvidenceTests.write_vcf(self, orig=3))
        with self.assertRaisesRegex(ValueError, "Originally multiallelic"):
            local.run(self.args())

    def test_reversed_analytical_order_changes_pair_order_not_counts(self):
        self.spanning_chain()
        local.run(self.args())
        original = self.rows()[0]
        self.sample_file.write_text("c\nb\na\n")
        local.run(self.args("other"))
        reversed_row = self.rows("other")[0]
        self.assertEqual((reversed_row["sample_a"], reversed_row["sample_b"]), ("b", "a"))
        for key in ("I", "U", "Q", "J", "rare_catalog_sites", "rare_missing_gt_count"):
            self.assertEqual(reversed_row[key], original[key])

    def test_wrong_source_size_and_unknown_sample_fail(self):
        args = self.args()
        args.expected_source_samples = 100
        with self.assertRaisesRegex(ValueError, "source sample count"):
            local.run(args)
        self.sample_file.write_text("a\nb\nunknown\n")
        with self.assertRaisesRegex(ValueError, "Analytical sample absent"):
            local.run(self.args("unknown"))

    def test_conflicting_duplicate_chain_rejected(self):
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [
            ["22", "a", "b", 500, 100, 500, 401, 2], ["22", "b", "a", 500, 100, 500, 401, 3]])
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            local.run(self.args())

    def test_overlapping_and_joinable_chains_fail(self):
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [
            ["22", "a", "b", 500, 100, 500, 401, 2], ["22", "a", "b", 500, 400, 600, 201, 2]])
        with self.assertRaisesRegex(ValueError, "Overlapping or nonmaximal"):
            local.run(self.args())

    def test_native_configuration_summary_mismatch_fail(self):
        self.write_table(self.configurations, local.CONFIG_COLUMNS + ("n_segments",), [["L1_G100_N1", 100, 1, 1, "1", 1, 8]])
        with self.assertRaisesRegex(ValueError, "Configuration summary mismatch"):
            local.run(self.args())

    def test_resource_guard_never_truncates(self):
        with self.assertRaisesRegex(ValueError, "Active-chain resource limit"):
            local.run(self.args(extra=["--max-active-intervals", "1"]))
        self.assertFalse((self.root / "out/manifest.json").exists())

    def test_configuration_lookup_uses_covering_index(self):
        db = local.create_database(self.root / "index.sqlite")
        self.addCleanup(db.close)
        plan = db.execute("EXPLAIN QUERY PLAN SELECT chain_id FROM links WHERE config_id=?", ("x",)).fetchall()
        self.assertIn("links_config_chain", " ".join(str(tuple(row)) for row in plan))

    def test_resource_limits_are_validated(self):
        for flag, value in (("--max-db-mb", "0"), ("--max-db-mb", "nan"),
                            ("--min-free-disk-mb", "-1"), ("--min-free-disk-mb", "inf"),
                            ("--resource-check-rows", "0"), ("--max-preflight-memory-mb", "-1"),
                            ("--max-preflight-blocks", "0")):
            with self.subTest(flag=flag, value=value), self.assertRaisesRegex(ValueError, "must be"):
                local.run(self.args(extra=[flag, value]))
        self.assertFalse((self.root / "out").exists())

    def test_disk_guard_records_failure_without_completion(self):
        with mock.patch.object(local.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            with self.assertRaisesRegex(ValueError, "Free-disk"):
                local.run(self.args())
        self.assertEqual(json.loads((self.root / "out/progress.json").read_text())["status"], "FAILED")
        self.assertFalse((self.root / "out/manifest.json").exists())

    def test_sqlite_hard_limit_and_progress_checkpoints(self):
        with self.assertRaises(local.sqlite3.DatabaseError):
            local.run(self.args(extra=["--max-db-mb", "0.004"]))
        self.assertFalse((self.root / "out/manifest.json").exists())
        result = local.run(self.args("ok", ["--resource-check-rows", "1", "--chunk-sites", "1"]))
        for name, checksum in result["operational_outputs_sha256"].items():
            self.assertEqual(local.sha256(self.root / "ok" / name), checksum)
        progress = json.loads((self.root / "ok/progress.json").read_text())
        self.assertEqual(progress["status"], "COMPLETE_DESCRIPTIVE_NOT_VALIDATED")
        self.assertGreater(progress["peak"]["sqlite_bytes"], 0)
        self.assertGreater(progress["peak"]["rss_bytes"], 0)
        events = [json.loads(line) for line in (self.root / "ok/progress.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["phase"] == "load_chains" and row["counts"].get("input_chain_rows") == 2 for row in events))
        self.assertTrue(any(row["phase"] == "rare_block_complete" and row["block_seconds"] >= 0 for row in events))

    def test_geometry_preflight_never_reads_genotypes_or_sqlite(self):
        with mock.patch.object(local, "annotate_vcf_blocks", side_effect=AssertionError("genotyping forbidden")), \
                mock.patch.object(local, "rare_dosage", side_effect=AssertionError("GT forbidden")), \
                mock.patch.object(local, "create_database", side_effect=AssertionError("SQLite forbidden")):
            result = local.run(self.args(extra=["--preflight-only", "--resource-check-rows", "1"]))
        self.assertEqual(result["status"], "PREFLIGHT_GEOMETRY_ONLY_NOT_EVIDENCE")
        self.assertEqual(result["checks"]["genotype_records_read"], 0)
        self.assertFalse(result["checks"]["configuration_summary_totals_validated"])
        counts = result["checks"]["counts"]
        self.assertEqual(counts["input_chain_rows"], 3)
        self.assertEqual(counts["peak_active_chain_rows_upper_bound"], 3)
        self.assertEqual(counts["distinct_pair_blocks"], 1)
        self.assertEqual(result["n_configurations"], 3)
        self.assertFalse((self.root / "out/manifest.json").exists())
        self.assertFalse((self.root / "out/segment_evidence.tsv.gz").exists())
        self.assertEqual(json.loads((self.root / "out/progress.json").read_text())["status"], result["status"])
        for name, checksum in result["operational_outputs_sha256"].items():
            self.assertEqual(local.sha256(self.root / "out" / name), checksum)

    def test_geometry_bitsets_cross_word_boundaries_and_match_bruteforce(self):
        rows = [["22", "a", "b", 1000, 1, 65, 65, 2],
                ["22", "b", "a", 1000, 64, 128, 65, 2],
                ["22", "a", "c", 1000, 65, 129, 65, 2],
                ["22", "b", "c", 1000, 63, 64, 2, 2]]
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], rows)
        result = local.run(self.args(extra=["--preflight-only", "--block-bp", "1", "--max-active-intervals", "1"]))
        observed = list(local.table_rows(self.root / "out/preflight_blocks.tsv", ("block_start", "distinct_pairs")))
        for block in observed:
            position = int(block["block_start"])
            active = [r for r in rows if r[4] <= position <= r[5]]
            pairs = {tuple(sorted(r[1:3])) for r in active}
            self.assertEqual(int(block["active_chain_rows_upper_bound"]), len(active))
            self.assertEqual(int(block["distinct_pairs"]), len(pairs))
        self.assertEqual(result["checks"]["counts"]["distinct_pair_blocks"], 128 + 65 + 2)
        self.assertGreater(result["checks"]["counts"]["blocks_exceeding_active_row_bound"], 0)

    def test_preflight_bounds_fail_before_numpy_allocation(self):
        for output, extra, message in (("memory", ["--max-preflight-memory-mb", "0.000001"], "NumPy-buffer"),
                                        ("blocks", ["--block-bp", "1", "--max-preflight-blocks", "1"], "block resource")):
            with mock.patch.object(local.np, "zeros", side_effect=AssertionError("must not allocate")):
                with self.assertRaisesRegex(ValueError, message):
                    local.run(self.args(output, ["--preflight-only", *extra]))
            self.assertFalse((self.root / output / "preflight.json").exists())

    def test_preflight_validates_all_rows_and_header_but_not_gt_agreement(self):
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [["22", "a", "c", 100, 100, 100, 1, 1]])
        local.run(self.args(extra=["--preflight-only"]))
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [["22", "a", "c", 100, 100, 100, 1, 1], ["22", "a", "b", 100, 0, 1, 2, 1]])
        with self.assertRaisesRegex(ValueError, "Invalid M14 coordinates"):
            local.run(self.args("invalid", ["--preflight-only"]))
        args = self.args("header", ["--preflight-only"])
        args.expected_source_samples = 100
        with self.assertRaises(ValueError):
            local.run(args)

    def test_empty_preflight_does_not_invent_blocks(self):
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [])
        result = local.run(self.args(extra=["--preflight-only"]))
        self.assertEqual(result["checks"]["counts"].get("occupied_blocks", 0), 0)
        self.assertEqual(list(local.table_rows(self.root / "out/preflight_blocks.tsv", ("block_start",))), [])

    def test_cli_preflight_has_distinct_non_scientific_completion(self):
        args = self.args(extra=["--preflight-only"])
        command = [sys.executable, str(Path(local.__file__))]
        for key, value in vars(args).items():
            if value is True:
                command.append("--" + key.replace("_", "-"))
            elif value is not None and value is not False:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "PREFLIGHT_GEOMETRY_ONLY_NOT_EVIDENCE")
        self.assertFalse((self.root / "out/manifest.json").exists())

    def test_guards_detect_growth_during_chain_load_and_blocks(self):
        original = local.shutil.disk_usage
        calls = 0

        def simulated_disk(path):
            nonlocal calls
            calls += 1
            return original(path) if calls < 4 else mock.Mock(free=0)

        with mock.patch.object(local.shutil, "disk_usage", side_effect=simulated_disk):
            with self.assertRaisesRegex(ValueError, "Free-disk"):
                local.run(self.args(extra=["--resource-check-rows", "1"]))
        progress = json.loads((self.root / "out/progress.json").read_text())
        self.assertEqual(progress["status"], "FAILED")
        self.assertEqual(progress["phase"], "load_chains")
        self.assertGreater(progress["counts"]["input_chain_rows"], 0)
        check = local.ResourceProgress.check

        def block_failure(monitor, phase, counts, **details):
            if phase == "rare_block_complete":
                with mock.patch.object(local.shutil, "disk_usage", return_value=mock.Mock(free=0)):
                    return check(monitor, phase, counts, **details)
            return check(monitor, phase, counts, **details)

        with mock.patch.object(local.ResourceProgress, "check", block_failure):
            with self.assertRaisesRegex(ValueError, "Free-disk"):
                local.run(self.args("block"))
        progress = json.loads((self.root / "block/progress.json").read_text())
        self.assertEqual(progress["status"], "FAILED")
        self.assertEqual(progress["phase"], "rare_block_complete")
        self.assertFalse((self.root / "block/manifest.json").exists())

    def test_empty_chains_keep_all_configurations_and_no_fake_genotype_result(self):
        self.write_table(self.chains, local.CHAIN_COLUMNS[1:], [])
        result = local.run(self.args())
        self.assertEqual(self.rows(), [])
        self.assertEqual(result["checks"]["counts"].get("rare_records_visited", 0), 0)
        self.assertEqual(len(list(local.table_rows(self.root / "out/configurations.tsv", local.CONFIG_COLUMNS))), 3)

    def test_unindexed_vcf_and_overwrite_fail(self):
        args = self.args()
        args.rare_vcf = str(self.root / "rare.vcf")
        with self.assertRaisesRegex(ValueError, "Indexed VCF"):
            local.run(args)
        local.run(self.args())
        with self.assertRaisesRegex(ValueError, "Output exists"):
            local.run(self.args())

    def map_fixture(self, rows=None):
        path = self.root / "map.tsv"
        self.write_table(path, ("chrom", "pos", "cm"), rows or [[22, 100, 1.0], [22, 300, 2.0], [22, 500, 2.0]])
        contract = self.contract("map.json", "r02_genetic_map_v1", sha256=local.sha256(path), units="cM", format="tsv_chrom_pos_cm")
        return ["--genetic-map", str(path), "--map-contract", str(contract)]

    def test_map_interpolation_plateau_and_no_extrapolation(self):
        self.map_fixture()
        map_data = local.GeneticMap(self.root / "map.tsv", "22", 10)
        self.assertEqual(map_data.length(100, 500), 1)
        self.assertEqual(map_data.at(200), 1.5)
        self.assertEqual(map_data.length(300, 500), 0)
        self.assertIsNone(map_data.at(99))
        self.assertIsNone(map_data.at(501))
        self.spanning_chain()
        local.run(self.args(extra=self.map_fixture([[22, 101, 0], [22, 500, 1]])))
        self.assertEqual(self.rows()[0]["map_status"], "NO_EVALUABLE:outside_map_support")
        self.assertEqual(self.rows()[0]["length_cm"], "NA")

    def test_bad_map_monotonicity_hash_build_fail(self):
        extra = self.map_fixture([[22, 100, 1], [22, 200, .5]])
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            local.run(self.args(extra=extra))
        extra = self.map_fixture()
        payload = json.loads((self.root / "map.json").read_text())
        payload["genome_build"] = "GRCh37"
        (self.root / "map.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "genome build"):
            local.run(self.args(extra=extra))
        extra = self.map_fixture()
        with (self.root / "map.tsv").open("a") as handle:
            handle.write("22\t600\t2.1\n")
        with self.assertRaisesRegex(ValueError, "Map hash"):
            local.run(self.args(extra=extra))

    def common_fixture(self):
        path = self.root / "common.vcf"
        header = pysam.VariantHeader()
        header.contigs.add("chr22", length=10000)
        header.info.add("ORIG_NALLELES", 1, "Integer", "Original count")
        header.formats.add("GT", 1, "String", "Genotype")
        header.add_line("##dnabr_original_alleles=v1")
        for sample in self.samples:
            header.add_sample(sample)
        with pysam.VariantFile(str(path), "w", header=header) as sink:
            for pos, calls in [(100, [(0, 0), (1, 1), (0, 1)]),
                               (200, [(0, 0), (0, 1), (0, 0)]),
                               (300, [(0, 0), (None, 1), (1, 1)]),
                               (500, [(1, 1), (0, 0), (0, 1)])]:
                row = sink.new_record(contig="chr22", start=pos-1, alleles=("A", "C"))
                row.info["ORIG_NALLELES"] = 2
                row.filter.add("PASS")
                for sample, gt in zip(self.samples, calls):
                    row.samples[sample]["GT"] = gt
                sink.write(row)
        path = self.index_vcf(path)
        contract = self.contract("common.json", "r02_segment_common_v1", vcf_sha256=local.sha256(path), analytical_samples_sha256=local.sample_hash(self.samples), min_maf=.1, max_missing=.5,
            frequency_scope="analytical_complete_diploid", site_filter="PASS", original_biallelic=True, mask="complete_diploid_GT")
        return ["--common-vcf", str(path), "--common-contract", str(contract)]

    def test_common_opposite_homozygotes_with_joint_missingness(self):
        self.spanning_chain()
        result = local.run(self.args(extra=self.common_fixture()))
        row = self.rows()[0]
        self.assertEqual((row["O"], row["Q_C"], row["common_status"]), ("2", "3", "OK"))
        self.assertIn("not certified IBD", result["semantics"]["common"])

    def test_common_requires_explicit_matched_contract(self):
        with self.assertRaisesRegex(ValueError, "must be paired"):
            local.run(self.args(extra=["--common-vcf", str(self.vcf)]))
        extra = self.common_fixture()
        payload = json.loads((self.root / "common.json").read_text())
        payload["analytical_samples_sha256"] = "bad"
        (self.root / "common.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "cohort/order mismatch"):
            local.run(self.args(extra=extra))

    def test_common_eligibility_filters_do_not_count_split_multiallelic_duplicates(self):
        self.spanning_chain()
        extra = self.common_fixture()
        raw = self.root / "common_extra.vcf"
        with pysam.VariantFile(extra[1]) as source, pysam.VariantFile(str(raw), "w", header=source.header) as sink:
            for record in source:
                sink.write(record)
                if record.pos == 100:
                    record.info["ORIG_NALLELES"] = 3
                    sink.write(record)
        path = self.index_vcf(raw)
        extra[1] = str(path)
        contract_path = Path(extra[3])
        payload = json.loads(contract_path.read_text())
        payload["vcf_sha256"] = local.sha256(path)
        contract_path.write_text(json.dumps(payload))
        local.run(self.args(extra=extra))
        self.assertEqual((self.rows()[0]["O"], self.rows()[0]["Q_C"]), ("2", "3"))

    def ibd_fixture(self, empty=False, callable_pair=("a", "b")):
        intervals, callability = self.root / "ibd.tsv", self.root / "callable.tsv"
        columns = ("chrom", "sample_a", "sample_b", "start_pos", "end_pos")
        self.write_table(intervals, columns + ("ibd_state",), [] if empty else [
            [22, "a", "b", 50, 220, "IBD1"], [22, "b", "a", 180, 350, "IBD2"], [22, "a", "b", 340, 450, "IBD1"]])
        self.write_table(callability, columns, [[22, *callable_pair, 100, 200], [22, *callable_pair, 250, 400]])
        contract = self.contract("ibd.json", "r02_segment_ibd_v1", semantics="any_copy_ibd_territory", caller=dict(name="synthetic", version="1"), absence_means_no_called_ibd_within_callable=True,
            intervals=dict(path=intervals.name, sha256=local.sha256(intervals)), callability=dict(path=callability.name, sha256=local.sha256(callability)))
        return ["--ibd-contract", str(contract)]

    def test_ibd_union_clips_pair_callability_and_never_sums_copy_lengths(self):
        self.spanning_chain()
        local.run(self.args(extra=self.ibd_fixture()))
        row = self.rows()[0]
        self.assertEqual((row["callable_bp"], row["ibd_union_bp"], row["ibd_fraction"], row["ibd_status"]), ("252", "252", "1.0", "OK"))
        self.assertEqual((row["ibd_overlap_pieces"], row["ibd_first_overlap_pos"], row["ibd_last_overlap_pos"], row["ibd_left_uncovered_bp"], row["ibd_right_uncovered_bp"]), ("2", "100", "400", "0", "100"))

    def test_ibd_pair_territory_includes_outside_chains_and_pairs_without_m14(self):
        self.spanning_chain()
        extra = self.ibd_fixture()
        with (self.root / "ibd.tsv").open("a") as handle:
            handle.write("22\ta\tb\t600\t700\tIBD1\n22\ta\tc\t700\t800\tIBD2\n")
        with (self.root / "callable.tsv").open("a") as handle:
            handle.write("22\ta\tb\t600\t700\n22\ta\tc\t700\t800\n")
        path = self.root / "ibd.json"
        contract = json.loads(path.read_text())
        contract["intervals"]["sha256"] = local.sha256(self.root / "ibd.tsv")
        contract["callability"]["sha256"] = local.sha256(self.root / "callable.tsv")
        path.write_text(json.dumps(contract))
        result = local.run(self.args(extra=extra))
        rows = list(local.table_rows(self.root / "out/ibd_pair_territory.tsv.gz", local.IBD_PAIR_COLUMNS))
        self.assertEqual([(r["sample_a"], r["sample_b"], r["callable_bp"], r["ibd_union_bp"], r["ibd_pieces"]) for r in rows], [("a", "b", "353", "353", "3"), ("a", "c", "101", "101", "1")])
        self.assertEqual(self.rows()[0]["ibd_union_bp"], "252")
        self.assertIn("ibd_pair_territory.tsv.gz", result["outputs_sha256"])

    def test_empty_called_ibd_is_zero_only_with_explicit_pair_territory(self):
        self.spanning_chain()
        local.run(self.args(extra=self.ibd_fixture(empty=True)))
        self.assertEqual(self.rows()[0]["ibd_union_bp"], "0")
        self.assertEqual(self.rows()[0]["ibd_overlap_pieces"], "0")
        self.assertEqual(self.rows()[0]["ibd_first_overlap_pos"], "NA")
        local.run(self.args("other", self.ibd_fixture(callable_pair=("a", "c"))))
        self.assertEqual(self.rows("other")[0]["callable_bp"], "NA")
        self.assertEqual(self.rows("other")[0]["ibd_union_bp"], "NA")
        territories = list(local.table_rows(self.root / "other/ibd_pair_territory.tsv.gz", local.IBD_PAIR_COLUMNS))
        ab = next(row for row in territories if row["sample_b"] == "b")
        self.assertEqual((ab["callable_bp"], ab["ibd_union_bp"]), ("NA", "NA"))

    def test_ibd_missing_callability_or_copy_semantics_fail(self):
        extra = self.ibd_fixture()
        path = self.root / "ibd.json"
        payload = json.loads(path.read_text())
        payload["semantics"] = "haplotype_length_sum"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "copy/haplotype semantics"):
            local.run(self.args(extra=extra))
        extra = self.ibd_fixture()
        payload = json.loads(path.read_text())
        del payload["callability"]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "pair callability"):
            local.run(self.args("other", extra))

    def test_interval_algebra(self):
        self.assertEqual(local.merge_intervals([(10, 20), (15, 25), (26, 30), (1, 2)]), [(1, 2), (10, 30)])
        self.assertEqual(local.intersect_intervals([(10, 30)], [(1, 15), (20, 25)]), [(10, 15), (20, 25)])
        self.assertEqual(local.interval_bp([(1, 10), (5, 15)]), 15)
        with self.assertRaisesRegex(ValueError, "Invalid interval"):
            local.merge_intervals([(0, 10)])

    def test_cli_small_indexed_vcf(self):
        args = self.args()
        command = [sys.executable, str(Path(local.__file__))]
        for key, value in vars(args).items():
            if value is True:
                command.append("--" + key.replace("_", "-"))
            elif value is not None and value is not False:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "COMPLETE_DESCRIPTIVE_NOT_VALIDATED")
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("never overwrite", repeated.stderr)

    def test_cli_all_optional_inputs_and_native_producer_consumer(self):
        import r02_biological_evaluation as evaluation
        self.spanning_chain()
        args = self.args(extra=self.common_fixture() + self.map_fixture() + self.ibd_fixture())
        command = [sys.executable, str(Path(local.__file__))]
        for key, value in vars(args).items():
            if value is True:
                command.append("--" + key.replace("_", "-"))
            elif value is not None and value is not False:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        row = self.rows()[0]
        self.assertEqual((row["O"], row["Q_C"], row["length_cm"], row["ibd_union_bp"]), ("2", "3", "1.0", "252"))
        kinship = self.root / "kinship.tsv"
        kinship.write_text("ID1\tID2\tkin\na\tb\t0.01\na\tc\t0\nb\tc\t0\n")
        evaluation.run(segment_manifests=[self.root / "out/manifest.json"], sample_ids=self.sample_file,
            expected_samples=3, pcrelate_file=kinship, expected_pcrelate_sha256=local.sha256(kinship),
            expected_configurations=2, output_dir=self.root / "evaluated", min_free_disk_mb=0, scratch_dir=self.root)
        rows = list(local.table_rows(self.root / "evaluated/configuration_metrics.tsv", ("config_id", "metric", "value")))
        j = next(row for row in rows if row["config_id"] == "L401_G500_N2" and row["metric"] == "rare_local_J")
        self.assertAlmostEqual(float(j["value"]), 2/3)
        empty = next(row for row in rows if row["config_id"] == "L1000_G500_N3" and row["metric"] == "rare_local_J")
        self.assertEqual(empty["value"], "NA")


if __name__ == "__main__":
    unittest.main()
