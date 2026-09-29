import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location('presentation', Path(__file__).resolve().parents[1] / 'bin/rare_segment_presentation.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def check_effective_count():
    assert m.geometry_minimum(1000000, 50000) == 21
    assert m.geometry_minimum(250000, 50000) == 6
    frame = pd.DataFrame([dict(min_length_bp=1000000, max_gap_bp=50000, min_shared_effective=21)])
    assert m.configuration(frame, 1000000, 50000, 20).min_shared_effective == 21


def check_selection_and_pseudonyms(tmp_path):
    path = tmp_path / 'chains.tsv.gz'
    with gzip.open(path, 'wt') as h:
        h.write('chrom\tsample_a\tsample_b\tmax_gap_bp\tstart_pos\tend_pos\tlength_bp\tn_shared_variants\n'
                '22\tone\ttwo\t50\t100\t199\t100\t20\n'
                '22\tone\ttwo\t100\t100\t399\t300\t20\n'
                '22\tone\tthree\t50\t200\t210\t11\t20\n')
    target = pd.Series(dict(max_gap_bp=50, min_length_bp=100, min_shared_effective=20,
                            n_segments=1, n_pairs=1, total_shared_bp=100, n_shared_variants_total=20))
    rows, stats = m.select_chains(path, target, '22', ['one', 'two', 'three'])
    assert stats['n_segments'] == 1
    assert rows[0]['sample_a'] == 'S0001'
    assert rows[0]['sample_b'] == 'S0003'
    assert rows[0]['length_bp'] == 100
    target['n_segments'] = 2
    with unittest.TestCase().assertRaisesRegex(ValueError, 'reproduce'):
        m.select_chains(path, target, '22', ['one', 'two', 'three'])


def check_detail_does_not_duplicate():
    assert m.detail_ranks(3, 40).tolist() == [1, 2, 3]
    ranks = m.detail_ranks(45141, 40)
    assert len(ranks) == 40 and ranks[0] == 1 and ranks[-1] == 45141


class PresentationTests(unittest.TestCase):
    def test_effective_count(self):
        check_effective_count()

    def test_selection_and_pseudonyms(self):
        with tempfile.TemporaryDirectory() as tmp:
            check_selection_and_pseudonyms(Path(tmp))

    def test_detail_does_not_duplicate(self):
        check_detail_does_not_duplicate()
