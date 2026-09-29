"""Synthetic tests only: no human data, cloud, pipelines or model execution."""
import csv
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest

import pandas as pd


SPEC = importlib.util.spec_from_file_location("rare_segment_plots", Path(__file__).resolve().parents[1] / "bin/rare_segment_plots.py")
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)


def fixture():
    return pd.DataFrame([
        dict(chrom="22", sample_a="person_b", sample_b="person_a", segment_id=2,
             start_pos=100, end_pos=199, length_bp=100, n_shared_variants=3, jaccard=1),
        dict(chrom="chr22", sample_a="person_a", sample_b="person_b", segment_id=1,
             start_pos=300, end_pos=399, length_bp=100, n_shared_variants=4, jaccard=1),
        dict(chrom="22", sample_a="person_c", sample_b="person_a", segment_id=3,
             start_pos=150, end_pos=250, length_bp=101, n_shared_variants=5, jaccard=1),
    ])


class SegmentViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def render(self, frame=None, **kwargs):
        options = dict(dpi=50, formats=("png", "pdf"), chromosome_length=1000)
        options.update(kwargs)
        return PLOTS.render_segment_views(fixture() if frame is None else frame, self.root, "22", **options)

    def read_ledger(self, manifest, kind):
        with (self.root / manifest["views"][kind]["row_ledger"]).open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def test_counts_endpoints_partners_and_private_map(self):
        frame = fixture()
        saved = frame.copy(deep=True)
        result = self.render(frame, all_samples=["person_d", "person_c", "person_b", "person_a"])
        pd.testing.assert_frame_equal(frame, saved)
        self.assertEqual((result["n_segments"], result["n_pairs"], result["n_active_individuals"]), (3, 2, 3))
        self.assertEqual(result["n_samples_without_input_segments"], 1)
        pair = self.read_ledger(result, "pairwise_segments")
        individual = self.read_ledger(result, "individual_segments")
        self.assertEqual(len(pair), 3)
        self.assertEqual(len(individual), 6)
        self.assertEqual(result["views"]["pairwise_segments"]["n_rows"], 2)
        self.assertEqual(result["views"]["individual_segments"]["n_rows"], 4)
        for row in pair:
            matches = [r for r in individual if r["record_id"] == row["record_id"]]
            self.assertEqual(len(matches), 2)
            self.assertEqual({(r["sample"], r["partner"]) for r in matches},
                             {(row["sample"], row["partner"]), (row["partner"], row["sample"])})
            for item in matches:
                self.assertEqual([item[k] for k in ("start_pos", "end_pos", "length_bp", "n_shared_variants")],
                                 [row[k] for k in ("start_pos", "end_pos", "length_bp", "n_shared_variants")])
        self.assertEqual(sorted((int(r["start_pos"]), int(r["end_pos"])) for r in pair), [(100, 199), (150, 250), (300, 399)])
        private = self.root / result["private_sample_map"]
        self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
        self.assertIn("person_a", private.read_text())
        self.assertNotIn("person_a", json.dumps(result))
        self.assertNotIn("person_a", (self.root / result["views"]["pairwise_segments"]["row_ledger"]).read_text())
        for kind in result["views"].values():
            self.assertTrue((self.root / kind["figures"]["png"]).read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((self.root / kind["figures"]["pdf"]).read_bytes().startswith(b"%PDF"))

    def test_deterministic_under_input_and_universe_permutation(self):
        one = self.render(all_samples=["person_c", "person_a", "person_b"])
        directory = self.root / "second"
        two = PLOTS.render_segment_views(fixture().iloc[::-1], directory, "chr22",
                all_samples=["person_b", "person_c", "person_a"], dpi=50,
                formats=("png", "pdf"), chromosome_length=1000)
        self.assertEqual(one, two)
        for path in self.root.iterdir():
            if path.is_file():
                self.assertEqual(path.read_bytes(), (directory / path.name).read_bytes(), path.name)

    def test_no_overwrite_or_partial_write_on_collision(self):
        collision = self.root / "chr22.individual_segments.pdf"
        collision.write_bytes(b"user-owned")
        with self.assertRaises(FileExistsError):
            self.render()
        self.assertEqual(collision.read_bytes(), b"user-owned")
        self.assertEqual(list(self.root.iterdir()), [collision])

    def test_broken_symlink_not_overwritten(self):
        target = self.root / "chr22.sample_labels.private.tsv"
        target.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            self.render()
        self.assertTrue(target.is_symlink())

    def test_overlapping_intervals_never_merge_and_use_distinct_lanes(self):
        frame = fixture().iloc[:2].copy()
        frame.loc[1, ["start_pos", "end_pos", "length_bp"]] = [150, 250, 101]
        result = self.render(frame)
        rows = self.read_ledger(result, "pairwise_segments")
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["lane_index"] for r in rows}, {"0", "1"})
        self.assertEqual({r["lane_count"] for r in rows}, {"2"})

    def test_base_centred_rectangle_matches_inclusive_coordinates(self):
        records = PLOTS._load_segments(fixture(), "22")
        _, views = PLOTS._prepare_views(records, ["person_a", "person_b", "person_c"])
        rows, bars = views["pairwise_segments"]
        figure = PLOTS._make_figure("pairwise_segments", rows, bars, "22", 1000, 399, 3, 3)
        patches = figure.axes[0].patches
        self.assertEqual(len(patches), 3)
        for patch, bar in zip(patches, bars):
            self.assertEqual(patch.get_x(), bar["start_pos"] - 0.5)
            self.assertEqual(patch.get_x() + patch.get_width(), bar["end_pos"] + 0.5)
        figure.clear()

    def test_synthetic_126_intervals_123_pairs_116_people(self):
        people = [f"fictional_{i:03d}" for i in range(116)]
        pairs = [(people[i], people[i + 1]) for i in range(115)]
        pairs += [(people[0], people[i]) for i in range(2, 10)]
        records = [dict(chrom="22", sample_a=a, sample_b=b, start_pos=100,
                        end_pos=199, length_bp=100, n_shared_variants=5) for a, b in pairs]
        records += [dict(row, start_pos=300, end_pos=399) for row in records[:3]]
        checked = PLOTS._load_segments(pd.DataFrame(records), "22")
        _, views = PLOTS._prepare_views(checked, people)
        self.assertEqual(len(views["pairwise_segments"][0]), 123)
        self.assertEqual(len(views["pairwise_segments"][1]), 126)
        self.assertEqual(len(views["individual_segments"][0]), 246)
        self.assertEqual(len(views["individual_segments"][1]), 252)
        self.assertEqual(len({r["sample"] for r in views["individual_segments"][0]}), 116)

    def test_single_base_interval_and_empty_input(self):
        one = fixture().iloc[:1].copy()
        one.loc[0, ["start_pos", "end_pos", "length_bp", "n_shared_variants"]] = [1, 1, 1, 1]
        result = self.render(one)
        self.assertEqual(self.read_ledger(result, "pairwise_segments")[0]["length_bp"], "1")
        empty = PLOTS.render_segment_views(fixture().iloc[:0], self.root / "empty", "22", dpi=50, formats=("png",))
        self.assertEqual(empty["n_segments"], 0)
        self.assertEqual(empty["views"]["individual_segments"]["n_bars"], 0)

    def test_tsv_gzip_input(self):
        path = self.root / "synthetic.tsv.gz"
        fixture().to_csv(path, sep="\t", index=False)
        self.assertEqual(self.render(path)["n_segments"], 3)

    def test_nextflow_cli(self):
        path = self.root / "synthetic.tsv.gz"
        fixture().to_csv(path, sep="\t", index=False)
        sample_path = self.root / "synthetic.samples.txt"
        sample_path.write_text("person_d\nperson_c\nperson_a\nperson_b\n")
        result = PLOTS.main(["--segments", str(path), "--sample-ids-file", str(sample_path),
                            "--chr", "22", "--output-dir", str(self.root / "cli"),
                            "--dpi", "40", "--chromosome-length", "1000", "--formats", "png"])
        self.assertEqual(result, 0)
        manifest = json.loads((self.root / "cli/chr22.segment_views.manifest.json").read_text())
        self.assertEqual(manifest["n_samples_in_label_universe"], 4)

    def test_invalid_rows_fail_before_writing(self):
        cases = [("chrom", "21"), ("start_pos", 0), ("end_pos", 99), ("length_bp", 0),
                 ("length_bp", 99), ("start_pos", 100.5), ("n_shared_variants", -1),
                 ("n_shared_variants", float("nan")), ("sample_a", ""), ("sample_a", "person_a")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                frame = fixture().astype(object)
                frame.loc[0, field] = value
                with self.assertRaises(ValueError):
                    self.render(frame)
                self.assertFalse(list(self.root.iterdir()))

    def test_duplicate_unordered_interval_rejected_even_with_different_counts(self):
        row = fixture().iloc[:1].copy()
        duplicate = row.copy()
        duplicate.loc[0, ["sample_a", "sample_b", "n_shared_variants"]] = ["person_a", "person_b", 20]
        with self.assertRaisesRegex(ValueError, "Duplicate unordered"):
            self.render(pd.concat([row, duplicate], ignore_index=True))

    def test_invalid_configuration_or_missing_columns(self):
        for options in [dict(all_samples=["person_a"]), dict(all_samples=["person_a"] * 2),
                        dict(all_samples="person_a"), dict(dpi=0), dict(formats=("svg",)),
                        dict(formats=("png", "png")), dict(prefix="../unsafe"),
                        dict(chromosome_length=200)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.render(**options)
        with self.assertRaises(ValueError):
            self.render(fixture().drop(columns=["n_shared_variants"]))
        self.assertFalse(list(self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
