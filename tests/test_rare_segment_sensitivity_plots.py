"""Aggregate-only synthetic plot tests; no real participants or detectors."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import rare_segment_sensitivity_plots as PLOTS


def fixture():
    summary = pd.DataFrame([dict(universe=u, n_units=3, n_positions=6, n_distances=3,
              median_distance_bp=10, p90_distance_bp=26, p95_distance_bp=28) for u in PLOTS.UNIVERSES])
    hist = pd.DataFrame([dict(universe=u, bin_start_bp=a, bin_end_bp=b, count=c)
                        for u in PLOTS.UNIVERSES for a, b, c in [(1, 10, 1), (10, 100, 2), (100, "inf", 0)]])
    grid = pd.DataFrame([
        dict(max_gap_bp=25000, min_length_bp=100000, min_shared_effective=5, n_segments=2, n_pairs=2, total_shared_bp=220000),
        dict(max_gap_bp=25000, min_length_bp=250000, min_shared_effective=11, n_segments=0, n_pairs=0, total_shared_bp=0),
        dict(max_gap_bp=50000, min_length_bp=100000, min_shared_effective=5, n_segments=3, n_pairs=2, total_shared_bp=400000),
    ])
    return summary, hist, grid


def chain_fixture():
    return pd.DataFrame([dict(max_gap_bp=gap, metric="length_bp", bin_start=a, bin_end=b, count=c)
                        for gap in (25000, 50000, 100000)
                        for a, b, c in ((1, 2, 10), (2, 1000000, 3), (1000000, "inf", 2))])


class SensitivityPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def render(self, values=None, **kwargs):
        options = dict(dpi=40, formats=("png", "pdf"))
        options.update(kwargs)
        return PLOTS.render_sensitivity_views(*(fixture() if values is None else values), self.root, "22", **options)

    def test_four_figures_and_separate_universes(self):
        result = self.render()
        self.assertEqual(result["n_effective_configurations"], 3)
        self.assertEqual(len(result["figures"]), 4)
        self.assertEqual(set(result["distance_quantiles_and_counts"]), set(PLOTS.UNIVERSES))
        for universe in result["distance_quantiles_and_counts"].values():
            self.assertEqual(universe["n_distances"], 3)
            self.assertEqual(universe["quantiles_bp"]["median_distance_bp"], 10)
        for item in result["figures"].values():
            self.assertTrue((self.root / item["png"]).read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((self.root / item["pdf"]).read_bytes().startswith(b"%PDF"))

    def test_missing_cells_masked_zero_cell_preserved_common_scales(self):
        rows = PLOTS._grid_data(fixture()[2])
        figure = PLOTS._grid_figure(rows, "n_segments", "22")
        images = [ax.images[0] for ax in figure.axes if ax.images]
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].get_clim(), images[1].get_clim())
        first = images[0].get_array()
        self.assertEqual(first[1, 1], 0)
        self.assertFalse(first.mask[1, 1])
        self.assertTrue(first.mask[0, 1])
        self.assertEqual(first[0, 0], 2)
        figure.clear()

    def test_histogram_counts_exact_overflow_and_log_axes(self):
        summary, hist, _ = fixture()
        summary.loc[0, "n_distances"] = 4
        hist.loc[2, "count"] = 1
        data = PLOTS._distance_data(summary, hist)
        figure = PLOTS._distance_figure(data, "22")
        for ax in figure.axes:
            self.assertEqual(ax.get_xscale(), "log")
            self.assertEqual(sum(p.get_height() for p in ax.patches), 3)
        self.assertTrue(any("Overflow" in text.get_text() for text in figure.axes[0].texts))
        figure.clear()

    def test_empty_universe(self):
        summary, hist, grid = fixture()
        summary.loc[0, ["n_units", "n_positions", "n_distances"]] = 0
        summary.loc[0, ["median_distance_bp", "p90_distance_bp", "p95_distance_bp"]] = float("nan")
        hist.loc[hist["universe"] == summary.loc[0, "universe"], "count"] = 0
        result = self.render((summary, hist, grid), formats=("png",))
        self.assertEqual(result["distance_quantiles_and_counts"][summary.loc[0, "universe"]]["quantiles_bp"], {})

    def test_chain_histogram_before_filter_and_cdf_denominator(self):
        data = PLOTS._chain_data(chain_fixture())
        figure = PLOTS._chain_figure(data, "22")
        cdf = figure.axes[1].lines[0]
        self.assertEqual(list(cdf.get_xdata()), [1, 2, 1000000])
        self.assertEqual(list(cdf.get_ydata()), [0, 10/15, 13/15])
        self.assertEqual(len(figure.axes), 6)
        figure.clear()
        result = self.render(chain_histogram=chain_fixture(), formats=("png",))
        self.assertEqual(len(result["figures"]), 5)
        self.assertEqual(result["chain_histogram_counts_by_gap"], {"25000": 15, "50000": 15, "100000": 15})
        self.assertEqual(result["chains_below_1mb_exact_if_bin_boundary"], {"25000": 13, "50000": 13, "100000": 13})

    def test_validation_reconciliation_and_duplicate_grid(self):
        summary, hist, grid = fixture()
        bad_hist = hist.copy()
        bad_hist.loc[0, "count"] = 999
        with self.assertRaises(ValueError):
            self.render((summary, bad_hist, grid))
        with self.assertRaises(ValueError):
            self.render((summary, hist, pd.concat([grid, grid.iloc[:1]])))
        bad_grid = grid.copy()
        bad_grid.loc[0, "total_shared_bp"] = 1
        with self.assertRaises(ValueError):
            self.render((summary, hist, bad_grid))
        with self.assertRaises(ValueError):
            self.render((summary.iloc[:2], hist, grid))
        self.assertFalse(list(self.root.iterdir()))

    def test_no_overwrite(self):
        path = self.root / "chr22.sensitivity_n_pairs.pdf"
        path.write_bytes(b"owned")
        with self.assertRaises(FileExistsError):
            self.render()
        self.assertEqual(path.read_bytes(), b"owned")
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_tsv_cli(self):
        summary, hist, grid = fixture()
        paths = [self.root / name for name in ("summary.tsv", "hist.tsv", "grid.tsv")]
        for frame, path in zip((summary, hist, grid), paths):
            frame.to_csv(path, index=False, sep="\t")
        result = PLOTS.main(["--distance-summary", str(paths[0]), "--distance-histogram", str(paths[1]),
                    "--configuration-summary", str(paths[2]), "--output-dir", str(self.root / "cli"),
                    "--chr", "22", "--dpi", "40", "--formats", "png"])
        self.assertEqual(result, 0)
        manifest = json.loads((self.root / "cli/chr22.sensitivity_plots.manifest.json").read_text())
        self.assertEqual(manifest["n_effective_configurations"], 3)


if __name__ == "__main__":
    unittest.main()
