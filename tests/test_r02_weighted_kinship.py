"""Known denominators, missingness, duplicate protection and real graph interface."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
import r02_weighted_kinship as kin
import r02_weighted_communities as weighted
from tests.test_r02_weighted_communities import WeightedCommunitiesTests


class KinshipTests(unittest.TestCase):
    def test_missing_does_not_mean_unrelated_and_weights_not_counts(self):
        result=kin.metrics(np.array([1.,2.,3.,4.]),np.array([.1,.01,np.nan,np.nan]),
                           np.array([kin.FINITE,kin.FINITE,kin.ABSENT,kin.NONFINITE]),.0221)
        self.assertEqual(result['n_edges'],4)
        self.assertEqual(result['n_missing_absent'],1)
        self.assertEqual(result['n_missing_nonfinite'],1)
        self.assertEqual(result['fraction_n_kin_ge_all_edges'],.25)
        self.assertEqual(result['fraction_weight_kin_ge_all_edges'],.1)
        self.assertEqual(result['fraction_n_kin_ge_observed_edges'],.5)
        self.assertAlmostEqual(result['fraction_weight_kin_ge_observed_edges'],1/3)
        empty=kin.metrics(np.array([]),np.array([]),np.array([],dtype='uint8'),.0221)
        self.assertIsNone(empty['fraction_n_kin_ge_all_edges'])

    def test_reversed_duplicates_and_hash_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'kin.tsv'
            target=np.array([[False,True],[True,False]])
            path.write_text('ID1\tID2\tkin\na\tb\tNA\nb\ta\t0.1\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                kin.stream_kinship_arrays(path,{'a':0,'b':1},target,kin.evidence.sha256(path))
            path.write_text('ID1\tID2\tkin\na\tb\t0.1\n')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                kin.stream_kinship_arrays(path,{'a':0,'b':1},target,'0'*64)

    def test_actual_weighted_graph_interface_and_disjoint_classes(self):
        fixture=WeightedCommunitiesTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root=fixture.root
        graphs=root/'graphs'
        with patch.object(weighted,'draw_graph',return_value={'test_only':'no plotting'}):
            weighted.run(fixture.bundle(),fixture.samples_path,fixture.settings_path,graphs)
        source=root/'kin.tsv'
        source.write_text('ID1\tID2\tkin\na\tb\t0.1\na\tc\tNA\nc\td\t0.01\n')
        out=root/'diagnostics'
        result=kin.run(graphs,fixture.samples_path,6,source,kin.evidence.sha256(source),out)
        self.assertEqual(result['n_graphs'],3)
        self.assertEqual(result['n_partition_rows'],18)
        self.assertTrue(result['no_new_kinship'])
        self.assertTrue(result['no_pvalues'])
        self.assertTrue((out/'partition_kinship_summary.tsv').is_file())
        with self.assertRaisesRegex(ValueError,'exists'):
            kin.run(graphs,fixture.samples_path,6,source,kin.evidence.sha256(source),out)


if __name__=='__main__':unittest.main()
