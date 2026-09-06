"""Freeze, naming and detached-command checks without accessing real inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'cal_launcher', Path(__file__).resolve().parents[1] / 'bin/m39_launch_calibration.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        for name in MODULE.SOURCES:
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)
        original = self.repo / 'private'
        original.mkdir()
        self.original = original
        for name in ('development.npz', 'score.npz', 'selection.json', 'receipt.json',
                     'common.npz', 'pooled.npz', 'carrier.npz'):
            (original / name).write_bytes(name.encode())
        entry = lambda name: {'path': name, 'sha256': MODULE.sha256(original / name)}
        manifest = {'selection': entry('selection.json'), 'score_receipt': entry('receipt.json'),
                    'comparators': [entry(f'{arm}.npz') for arm in ('common', 'pooled', 'carrier')]}
        MODULE.write_json(original / 'comparators.json', manifest)
        plan = {f'{role}_sha256': MODULE.sha256(original / f'{role}.npz')
                for role in ('development', 'score')}
        plan['comparators_manifest_sha256'] = MODULE.sha256(original / 'comparators.json')
        MODULE.write_json(original / 'plan.json', plan)
        self.args = argparse.Namespace(run_dir=self.repo / '.claude/runs/fixture-run',
            development=original / 'development.npz', score=original / 'score.npz',
            plan=original / 'plan.json', comparators=original / 'comparators.json',
            max_minutes=15.)
        def output(command, **kwargs):
            return '' if command[1] == 'status' else 'f' * 40
        self.addCleanup(patch.stopall)
        patch.object(MODULE, '__file__', str(self.repo / 'bin/m39_launch_calibration.py')).start()
        patch.object(MODULE.subprocess, 'check_output', side_effect=output).start()
        patch.object(MODULE.subprocess, 'run').start()

    def test_freeze_authenticates_all_inputs_and_uses_flat_manifest(self):
        original_hash = MODULE.sha256(self.args.comparators)
        run = MODULE.prepare(self.args)
        request = json.loads((run / 'launch.json').read_text())
        for name, digest in request['frozen_sha256'].items():
            self.assertEqual(MODULE.sha256(run / name), digest)
        manifest = json.loads((run / 'comparators/manifest.json').read_text())
        for entry in [manifest['selection'], manifest['score_receipt'], *manifest['comparators']]:
            self.assertEqual(Path(entry['path']).name, entry['path'])
        self.assertEqual(original_hash, MODULE.sha256(self.args.comparators))
        self.assertEqual(request['max_minutes'], 15.)
        self.assertEqual(request['new_instances'], 0)

    def test_refuses_existing_output(self):
        MODULE.prepare(self.args)
        with self.assertRaises(ValueError):
            MODULE.prepare(self.args)

    def test_rejects_changed_input(self):
        self.args.score.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            MODULE.prepare(self.args)

    def test_rejects_changed_manifest(self):
        self.args.comparators.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            MODULE.prepare(self.args)

    def test_rejects_unbounded_runtime(self):
        for limit in (0, -1, 16, float('inf'), float('nan')):
            self.args.max_minutes = limit
            with self.assertRaises(ValueError):
                MODULE.prepare(self.args)

    def test_rejects_output_outside_private_runs(self):
        self.args.run_dir = self.repo / 'outside'
        with self.assertRaises(ValueError):
            MODULE.prepare(self.args)


if __name__ == '__main__':
    unittest.main()
