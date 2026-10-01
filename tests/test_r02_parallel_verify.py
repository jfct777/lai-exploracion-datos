"""Bounded synthetic checks: no cloud calls, SSH or cohort calculations."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('verify', ROOT/'bin/r02_parallel_verify.py')
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class VerifyTests(unittest.TestCase):
    def test_indices(self):
        self.assertEqual(verify.parse_workers('1,12'), [1, 12])
        for bad in ('', '0', '13', '1,1', 'hello'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify.parse_workers(bad)

    def test_remote_payload_syntax(self):
        ast.parse(verify.REMOTE_SCRIPT)
        self.assertEqual(len(verify.CORE_FILES), 13)
        self.assertNotIn('write_text', verify.REMOTE_SCRIPT)
        self.assertNotIn('unlink', verify.REMOTE_SCRIPT)

    def test_strict_record_marker(self):
        self.assertEqual(verify.parse_remote('Login banner\n'+verify.MARKER+'{"ok":true}\n'), {'ok': True})
        for text in ('{}', verify.MARKER+'{}\n'+verify.MARKER+'{}'):
            with self.assertRaises(ValueError):
                verify.parse_remote(text)

    def test_remote_argument_is_quoted(self):
        import shlex
        path = '/tmp/example with spaces; touch forbidden'
        tokens = shlex.split(verify.remote_command(path))
        self.assertEqual(tokens[-2], path)
        self.assertEqual(json.loads(tokens[-1]), list(verify.CORE_FILES))

    def test_wrong_live_identity_never_ssh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'worker01.creation.json').write_text(json.dumps({'returncode': 0,
                'response': [{'name': 'authorized-vm', 'id': '123'}]}))
            worker = {'worker': 'worker01', 'name': 'authorized-vm'}
            completed = type('Completed', (), {'returncode': 0, 'stdout': json.dumps(
                {'name': 'authorized-vm', 'id': '456', 'status': 'RUNNING'})})()
            with patch.object(verify.subprocess, 'run', return_value=completed) as mocked:
                result = verify.verify_worker({}, worker, root, root/'key', ROOT)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(result['verification'], 'UNVERIFIED')
            self.assertIn('identity differs', result['error'])

    def test_stopped_instance_never_ssh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = {'name': 'authorized-vm', 'id': '123', 'status': 'TERMINATED'}
            (root/'worker01.creation.json').write_text(json.dumps({'returncode': 0, 'response': [created]}))
            completed = type('Completed', (), {'returncode': 0, 'stdout': json.dumps(created)})()
            with patch.object(verify.subprocess, 'run', return_value=completed) as mocked:
                result = verify.verify_worker({}, {'worker': 'worker01', 'name': 'authorized-vm'}, root, root/'key', ROOT)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(result['verification'], 'INSTANCE_NOT_RUNNING')


if __name__ == '__main__':
    unittest.main()
