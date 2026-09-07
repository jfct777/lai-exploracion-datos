"""Sealed local orchestration checks; no Docker, Nextflow tasks or real arrays."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ordered_training_launcher', ROOT / 'bin/m39_launch_ordered_training.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='m39-launch-test-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)
        self.commit = 'a' * 40
        for name in MODULE.SOURCES:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(name)
        self.profile = {'cases': [{'id': 'cnn-small-b1-median-common', 'family': 'cnn',
            'size': 'small', 'batch_size': 1, 'window': 'median', 'arm': 'common', 'core_sites': 256}]}
        self.profile_path = self.repo / MODULE.PROFILE
        self.profile_path.write_text(json.dumps(self.profile))
        self.private = self.repo / '.claude/runs'
        source = self.private / 'source-run/ordered-output/ordered-profile'
        self.store = source / 'people_48/radius_1cm'
        self.store.mkdir(parents=True)
        (self.store / 'manifest.json').write_text('{}')
        self.parent = source / 'receipt.json'
        self.parent.write_text('{}')
        self.folds = self.private / 'folds.npz'
        self.folds.write_bytes(b'synthetic-fold-placeholder')
        self.run = self.private / 'new-run'
        self.run.mkdir()
        self.args = argparse.Namespace(run_dir=self.run, store_dir=self.store,
            parent_receipt=self.parent, folds=self.folds, profile_config=self.profile_path)
        for name, path in [('parent_receipt', self.parent), ('store_manifest', self.store / 'manifest.json'),
                           ('folds', self.folds)]:
            self.profile[f'{name}_sha256'] = MODULE.sha256(path)
        self.profile_path.write_text(json.dumps(self.profile))
        self.blobs = {name: (self.repo / name).read_bytes() for name in MODULE.SOURCES}
        self.dirty = ''
        def git_output(command, **kwargs):
            if command[1] == 'status':
                return self.dirty
            if command[1] == 'rev-parse':
                return self.commit
            if command[1] == 'show':
                name = command[2].split(':', 1)[1]
                if name not in self.blobs:
                    raise subprocess.CalledProcessError(128, command)
                return self.blobs[name]
            raise AssertionError(command)
        mock = patch.object(MODULE.subprocess, 'check_output', side_effect=git_output)
        mock.start()
        self.addCleanup(mock.stop)

    def prepare(self):
        return MODULE.prepare(self.args, repo=self.repo)

    def update_profile(self):
        self.profile_path.write_text(json.dumps(self.profile))
        self.blobs[MODULE.PROFILE] = self.profile_path.read_bytes()

    def test_freezes_every_source_without_copying_store(self):
        command, receipt_path, session = self.prepare()
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt['source_commit'], self.commit)
        self.assertEqual(receipt['argv'], command)
        self.assertIn('NXF_OFFLINE=true', command)
        self.assertIn('--m39_source_commit', command)
        self.assertNotIn('-resume', command)
        self.assertEqual(receipt['resources']['max_forks'], 2)
        self.assertEqual(receipt['resources']['memory_gib_total'], 16)
        self.assertFalse(receipt['gpu'])
        self.assertFalse(receipt['full_epoch'])
        self.assertEqual(receipt['inputs']['store_manifest']['sha256'], MODULE.sha256(self.store / 'manifest.json'))
        for relative in MODULE.SOURCES:
            copy = self.run / 'frozen-source' / relative
            self.assertEqual(copy.read_bytes(), self.blobs[relative])
            self.assertEqual(MODULE.sha256(copy), receipt['source_sha256'][relative])
        self.assertFalse((self.run / 'frozen-source/ordered-store').exists())
        self.assertFalse((self.run / 'ordered-training-output').exists())
        self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
        self.assertRegex(session, r'^m39-training-[0-9a-f]{16}$')

    def test_snapshot_unaffected_by_later_worktree_edits(self):
        self.prepare()
        (self.repo / 'bin/m39_ordered_models.py').write_text('changed after seal')
        self.assertEqual((self.run / 'frozen-source/bin/m39_ordered_models.py').read_bytes(),
                         self.blobs['bin/m39_ordered_models.py'])

    def test_rejects_reusing_launch_receipt(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'Immutable'):
            self.prepare()

    def test_rejects_existing_output_including_dangling_symlink(self):
        (self.run / 'ordered-training-output').symlink_to(self.run / 'absent')
        with self.assertRaisesRegex(ValueError, 'Immutable'):
            self.prepare()

    def test_rejects_dirty_tracked_code(self):
        self.dirty = ' M bin/m39_ordered_models.py'
        with self.assertRaisesRegex(ValueError, 'Commit tracked'):
            self.prepare()
        self.assertFalse((self.run / 'frozen-source').exists())

    def test_rejects_uncommitted_required_source(self):
        del self.blobs['bin/m39_ordered_models.py']
        with self.assertRaisesRegex(ValueError, 'Commit required'):
            self.prepare()

    def test_rejects_tree_commit_mismatch_even_if_status_clean(self):
        (self.repo / 'bin/m39_ordered_models.py').write_text('different')
        with self.assertRaisesRegex(ValueError, 'differs from source commit'):
            self.prepare()

    def test_rejects_unversioned_alternate_profile(self):
        alternate = self.run / 'alternate.json'
        alternate.write_text(json.dumps(self.profile))
        self.args.profile_config = alternate
        with self.assertRaisesRegex(ValueError, 'versioned conf'):
            self.prepare()

    def test_rejects_run_outside_private_tree(self):
        self.args.run_dir = self.repo
        with self.assertRaisesRegex(ValueError, 'private project runs'):
            self.prepare()

    def test_rejects_private_symlink_escape(self):
        escape = self.private / 'escape.npz'
        escape.symlink_to(self.profile_path)
        self.args.folds = escape
        with self.assertRaisesRegex(ValueError, 'private project runs'):
            self.prepare()

    def test_rejects_unrelated_parent_receipt(self):
        other = self.private / 'receipt.json'
        other.write_text('{}')
        self.args.parent_receipt = other
        with self.assertRaisesRegex(ValueError, 'belong to'):
            self.prepare()

    def test_rejects_changed_input_seal_before_creating_outputs(self):
        self.folds.write_bytes(b'changed-fold-placeholder')
        with self.assertRaisesRegex(ValueError, 'folds differs'):
            self.prepare()
        self.assertFalse((self.run / 'frozen-source').exists())

    def test_rejects_unsafe_container_user(self):
        with patch.dict(os.environ, {'M39_CONTAINER_USER': '1:2 --privileged'}):
            with self.assertRaisesRegex(ValueError, 'numeric uid:gid'):
                self.prepare()

    def test_main_detaches_exact_sealed_argv_without_running_tasks(self):
        argv = []
        for name, value in vars(self.args).items():
            argv.extend((f'--{name.replace("_", "-")}', str(value)))
        with patch.object(MODULE, '__file__', str(self.repo / 'bin/m39_launch_ordered_training.py')), \
                patch.object(MODULE.subprocess, 'run') as launch, patch('builtins.print'):
            previous_umask = os.umask(0o077)
            try:
                MODULE.main(argv)
            finally:
                os.umask(previous_umask)
        receipt = json.loads((self.run / 'ordered-training.launch.json').read_text())
        actual = launch.call_args.args[0]
        self.assertEqual(actual[:3], ['tmux', 'new-session', '-d'])
        self.assertEqual(shlex.split(actual[-1]), receipt['argv'])
        self.assertTrue(launch.call_args.kwargs['check'])

    def test_rejects_unsafe_or_duplicate_case_id(self):
        self.profile['cases'][0]['id'] = '../escape'
        self.update_profile()
        with self.assertRaisesRegex(ValueError, 'Unsafe case'):
            self.prepare()
        self.profile['cases'][0]['id'] = 'safe'
        self.profile['cases'].append(dict(self.profile['cases'][0]))
        self.update_profile()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.prepare()

    def test_rejects_resource_scope_expansion(self):
        for key, value in [('arm', 'pooled'), ('core_sites', 512), ('batch_size', 4),
                           ('batch_size', True), ('family', 'global_attention')]:
            with self.subTest(key=key, value=value):
                case = dict(self.profile['cases'][0], **{key: value})
                with self.assertRaises(ValueError):
                    MODULE._case_ids({'cases': [case]})

    def test_rejects_empty_or_unbounded_cases(self):
        for cases in ([], [self.profile['cases'][0]] * 33):
            with self.assertRaisesRegex(ValueError, 'bounded'):
                MODULE._case_ids({'cases': cases})


class OrchestrationTests(unittest.TestCase):
    def test_torch_cache_is_explicit_before_python_without_identity_override(self):
        process = (ROOT / 'modules/39_ORDERED_TRAINING_PROFILE.nf').read_text()
        cache = 'TORCHINDUCTOR_CACHE_DIR=/tmp/m39-torch-cache'
        self.assertIn(cache, process)
        self.assertLess(process.index(cache), process.index('python3 m39_profile_ordered_training.py'))
        # Each Docker task has its own ephemeral /tmp; no passwd-backed user lookup.
        self.assertNotRegex(process, r'\b(?:HOME|USER|LOGNAME)\s*=')
        self.assertNotIn('$HOME', process)
        self.assertIn("'modules/39_ORDERED_TRAINING_PROFILE.nf'", (ROOT / 'bin/m39_launch_ordered_training.py').read_text())

    def test_secondary_local_imports_are_staged_and_sealed(self):
        workflow = (ROOT / MODULE.WORKFLOW).read_text()
        for source in MODULE.SOURCES:
            if not source.startswith('bin/'):
                continue
            tree = ast.parse((ROOT / source).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    local = f'bin/{node.module}.py'
                    if (ROOT / local).is_file():
                        with self.subTest(source=source, imported=local):
                            self.assertIn(local, MODULE.SOURCES)
                            self.assertIn(f"'{node.module}.py'", workflow)

    def test_case_receipt_inventory_matches_all_staged_python(self):
        runner = ast.parse((ROOT / 'bin/m39_profile_ordered_training.py').read_text())
        assignment = next(node for node in runner.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'SOURCE_FILES'
                                  for target in node.targets))
        recorded = set(ast.literal_eval(assignment.value))
        staged = {Path(source).name for source in MODULE.SOURCES if source.startswith('bin/')
                  and source != 'bin/m39_launch_ordered_training.py'}
        self.assertEqual(recorded, staged)

    def test_hard_local_limits_and_readonly_symlinks(self):
        config = (ROOT / MODULE.CONFIG).read_text()
        process = (ROOT / 'modules/39_ORDERED_TRAINING_PROFILE.nf').read_text()
        for literal in ('cpus 2', "memory '8 GB'", "time '20m'", 'maxForks 2'):
            self.assertIn(literal, process)
        for literal in ("stageInMode = 'symlink'", 'docker.writableInputMounts = false',
                        'executor.cpus = 4', "executor.memory = '16 GB'", '--network none',
                        '--pull never', '--memory 8g --memory-swap 8g'):
            self.assertIn(literal, config)
        self.assertIn('path("${case_id}")', process)
        self.assertIn("overwrite: false", process)
        self.assertNotIn('--score', process.lower())

    def test_all_staged_code_is_in_the_sealed_inventory(self):
        workflow = (ROOT / MODULE.WORKFLOW).read_text()
        for name in ('m39_profile_ordered_training.py', 'm39_ordered_models.py',
                     'm39_ordered_batches.py', 'm39_ordered_context.py'):
            self.assertIn(name, workflow)
            self.assertIn(f'bin/{name}', MODULE.SOURCES)
        self.assertIn('channel.fromList(ids)', workflow)
        self.assertIn('Profile differs from launch seal', workflow)


if __name__ == '__main__':
    unittest.main()
