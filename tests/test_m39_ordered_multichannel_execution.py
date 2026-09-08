"""The additional channels use the same sealed and bounded Batch transport."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import m39_ordered_gpu_manifest as manifest
from test_m39_ordered_gpu_execution import make_plan


class MultichannelExecutionTests(unittest.TestCase):
    def make_plan(self, root, stage):
        path = make_plan(root, groups=1)
        plan = json.loads(path.read_text())
        template = json.loads((root / plan['groups'][0]['configs'][0]['file']).read_text())
        plan['stage'] = stage
        plan['groups'][0]['configs'] = []
        for arm in manifest.STAGE_ARMS[stage]:
            cfg = dict(template, schema_version=manifest.MULTICHANNEL_CONFIG_SCHEMA,
                       arm=arm, case_id='candidate-0-' + arm,
                       multichannel={'hidden_width': 32, 'initial_gate': .01},
                       steps=64, evaluate_initial=True, evaluate_every_steps=64)
            target = root / (cfg['case_id'] + '.json')
            target.write_text(json.dumps(cfg))
            plan['groups'][0]['configs'].append({'file': target.name, 'sha256': manifest.sha256(target)})
        path.write_text(json.dumps(plan))
        return path

    def test_stages_bind_complete_arms_and_correct_trainer(self):
        for stage, expected in (('multichannel_screen', 2), ('multichannel_followup', 5),
                                ('multichannel_technical', 5)):
            with tempfile.TemporaryDirectory() as root:
                path = self.make_plan(Path(root), stage)
                self.assertEqual(len(manifest.load_plan(path)['groups'][0]['configs']), expected)
                self.assertEqual(manifest.training_entrypoint(stage), 'm39_ordered_multichannel_training.py')
        self.assertEqual(manifest.training_entrypoint('controlled_followup'), 'm39_ordered_training.py')
        with self.assertRaises(ValueError):
            manifest.training_entrypoint('unregistered')

    def test_missing_arm_and_historical_schema_cannot_masquerade_as_multichannel(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_plan(Path(root), 'multichannel_followup')
            plan = json.loads(path.read_text())
            plan['groups'][0]['configs'].pop()
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'inventory'):
                manifest.load_plan(path)
        with tempfile.TemporaryDirectory() as root:
            path = self.make_plan(Path(root), 'multichannel_screen')
            plan = json.loads(path.read_text())
            spec = plan['groups'][0]['configs'][0]
            target = Path(root) / spec['file']
            cfg = json.loads(target.read_text())
            cfg['schema_version'] = 'm39-ordered-anchor-training-v1'
            target.write_text(json.dumps(cfg))
            spec['sha256'] = manifest.sha256(target)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'scope/device'):
                manifest.load_plan(path)

    def test_profile_is_bounded_and_never_a_scientific_screen(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_plan(Path(root), 'multichannel_technical')
            plan = json.loads(path.read_text())
            for spec in plan['groups'][0]['configs']:
                target = Path(root) / spec['file']
                cfg = json.loads(target.read_text())
                cfg['steps'] = cfg['evaluate_every_steps'] = 129
                target.write_text(json.dumps(cfg))
                spec['sha256'] = manifest.sha256(target)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'tiny complete'):
                manifest.load_plan(path)


if __name__ == '__main__':
    unittest.main()
