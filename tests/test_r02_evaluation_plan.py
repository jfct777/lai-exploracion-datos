"""Small execution-contract tests; no cloud, cohort inference or biological claims."""
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r02_plan_controller', ROOT/'bin/r02_autosome_pipeline.py')
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
GRID = dict(lengths=[100000,250000,500000,1000000,2000000],
            gaps=[25000,50000,100000],counts=[5,10,20,50,100,200])


class EvaluationPlanTests(unittest.TestCase):
    def test_all_effective_detectors_are_included_without_clustering_expansion(self):
        settings = pipeline.all_m14_diagnostic_settings(GRID, 7)
        configs = settings['configurations']
        self.assertEqual(len(configs), 74)
        self.assertEqual(settings['expected_samples'], 7)
        identities = {(c['length_bp'],c['gap_bp'],c['min_shared']) for c in configs}
        for length in GRID['lengths']:
            for gap in GRID['gaps']:
                for count in GRID['counts']:
                    effective = max(count, (length-2)//gap+2)
                    self.assertIn((length,gap,effective), identities)
        self.assertEqual({c['min_edge_bp'] for c in configs}, {1})
        self.assertEqual(len(pipeline.H_SETTINGS['configurations']), 6)

    def test_bad_geometry_is_rejected(self):
        for key in ('lengths','gaps','counts'):
            bad = dict(GRID, **{key:[0]})
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                pipeline.all_m14_diagnostic_settings(bad, 7)
        with self.assertRaises(ValueError):
            pipeline.all_m14_diagnostic_settings(dict(GRID, lengths=[]), 7)

    def test_frozen_settings_reach_clustering_and_diagnostics_precede_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            r=object.__new__(pipeline.Runner)
            r.run=root; r.bin=ROOT/'bin'; r.samples=root/'samples.txt'
            r.c=dict(reuse_chr22={'sensitivity':str(root/'old22')},
                     evaluate_all_m14_configurations=True,m14=GRID,analytical_samples=7,
                     m14_diagnostic_edge_thresholds_bp=[0,250000,500000,750000,1000000],
                     m14_diagnostic_kinship_thresholds=[.0221,.0442],
                     pcrelate=str(root/'kin.tsv'),pcrelate_sha256='a'*64,
                     metadata=str(root/'metadata.tsv'),metadata_color_column='label')
            hs=json.loads(json.dumps(pipeline.H_SETTINGS))
            hs.update(expected_samples=7,resolutions=[.7,1.])
            (root/'h_settings.json').write_text(json.dumps(hs))
            ds=pipeline.all_m14_diagnostic_settings(GRID,7)
            (root/'m14_diagnostic_settings.json').write_text(json.dumps(ds))
            calls=[]
            r.docker=lambda stage,args,**kw: calls.append((stage,[str(x) for x in args]))
            r.verify_inputs=lambda paths: None
            r.publish=lambda folder,relative: None
            with patch.object(pipeline,'sha',return_value='a'*64):
                r.aggregate()
            by_stage=dict(calls)
            prep=by_stage['all22_H_prepare']
            self.assertEqual(json.loads(base64.b64decode(prep[prep.index('--settings-base64')+1])),hs)
            agg=by_stage['all22_H_aggregate']
            self.assertEqual(agg[agg.index('--settings')+1],str(root/'m14_diagnostic_settings.json'))
            diag=by_stage['all22_M14_configuration_diagnostics']
            self.assertEqual(diag[diag.index('--expected-configurations')+1],'74')
            stages=[name for name,_ in calls]
            self.assertLess(stages.index('all22_M14_configuration_diagnostics'),stages.index('all22_H_plots'))
            self.assertEqual(diag[diag.index('--edge-thresholds-bp')+1],'0,250000,500000,750000,1000000')


if __name__=='__main__':
    unittest.main()
