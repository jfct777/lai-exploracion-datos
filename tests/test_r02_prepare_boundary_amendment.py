"""Preparation is local and compatible with adapter; all PIDs/data are synthetic."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_r02_apply_amendment as fixtures

ROOT, adapter, write = fixtures.ROOT, fixtures.adapter, fixtures.write


SPEC = importlib.util.spec_from_file_location('r02_prepare_amendment', ROOT/'bin/r02_prepare_boundary_amendment.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AmendmentTests('test_dry_validation_is_read_only_and_keeps_original_inputs')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.run
        self.repo = self.fixture.base/'checkout'
        (self.repo/'.git').mkdir(parents=True)
        for path in self.fixture.new.rglob('*'):
            if path.is_file():
                write(self.repo/path.relative_to(self.fixture.new), path.read_text())
        for name in ('r02_exec_task.py', 'r02_prepare_boundary_amendment.py', 'r02_stage_boundary_handoff.py'):
            write(self.repo/'bin'/name, (ROOT/'bin'/name).read_text())
        # All existing scientific dependencies already belonged to the original run.
        for path in self.repo.rglob('*'):
            if path.is_file() and str(path.relative_to(self.repo)) not in prepare.ADDED_SOURCE_ALLOWLIST:
                write(self.fixture.old/path.relative_to(self.repo), path.read_text())
        self.fixture.config['free_disk_min_gib'] = 12
        write(self.run/'run.json', self.fixture.config)
        self.freeze_original()
        self.completed = self.fixture.cp.read_text()
        self.fixture.cp.unlink()
        write(self.run/'status.json', dict(stage='chr21_M01_M02_M021', state='RUNNING', pid=222))
        self.amend = self.run/'repairs/reviewed'
        self.root_patch = patch.object(prepare, 'ROOT', self.repo)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.pid_patch = patch.object(prepare, 'identity', side_effect=self.identity)
        self.pid_mock = self.pid_patch.start()
        self.addCleanup(self.pid_patch.stop)
        self.limit_patch = patch.object(prepare.resource, 'prlimit', return_value=(4000, 4000))
        self.limit_mock = self.limit_patch.start()
        self.addCleanup(self.limit_patch.stop)

    def freeze_original(self):
        self.fixture.hash_source(self.fixture.old, self.run/'source.sha256.json')
        write(self.run/'frozen.sha256.json', {name: adapter.sha(self.run/name) for name in
            ('run.json', 'input_objects.json', 'samples.txt', 'h_settings.json', 'source.sha256.json')})

    def identity(self, pid):
        if pid == 111:
            return dict(pid=111, start_ticks=1000, cmdline_sha256='a'*64), 1, \
                ('python3 r02_autosome_pipeline.py --run-dir ' + str(self.run)).encode()
        if pid == 222:
            return dict(pid=222, start_ticks=1001, cmdline_sha256='b'*64), 111, \
                ('nextflow run ' + str(self.run/'chr21')).encode()
        raise AssertionError('Real PID lookup forbidden')

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            prepare.prepare(self.run, self.amend, 111, 222)
        return json.loads(output.getvalue())

    def test_snapshot_complete_request_only_and_original_unchanged(self):
        before = {str(p): p.read_bytes() for p in self.fixture.old.rglob('*') if p.is_file()}
        result = self.execute()
        after = {str(p): p.read_bytes() for p in self.fixture.old.rglob('*') if p.is_file()}
        self.assertEqual(before, after)  # Includes the absence of new __pycache__.
        self.assertEqual(result['status'], 'PREPARED_NOT_ARMED')
        request = json.loads((self.amend/'request.json').read_text())
        self.assertEqual(result['request_sha256'], adapter.sha(self.amend/'request.json'))
        self.assertEqual(request['amendment_source_manifest_sha256'], adapter.sha(self.amend/'source.sha256.json'))
        self.assertEqual(request['amendment_template']['overrides'], adapter.PROTOCOL_DELTA)
        self.assertNotIn('checkpoint_sha256', request['amendment_template']['boundary'])
        self.assertFalse((self.amend/'amendment.json').exists())
        self.assertFalse((self.amend/'frozen.sha256.json').exists())
        self.assertFalse(self.fixture.cp.exists())
        self.assertFalse((self.run/'m14_diagnostic_settings.json').exists())
        hashes = json.loads((self.amend/'source.sha256.json').read_text())
        self.assertTrue(all('bin/' + name in hashes for name in adapter.ANALYSIS_TOOLS))
        self.limit_mock.assert_called_once_with(111, prepare.resource.RLIMIT_NPROC)  # Getter only.

    def test_final_watcher_style_files_are_accepted_by_adapter(self):
        self.execute()
        request = json.loads((self.amend/'request.json').read_text())
        write(self.fixture.cp, self.completed)
        amendment = request['amendment_template']
        amendment['boundary']['checkpoint_sha256'] = adapter.sha(self.fixture.cp)
        write(self.amend/'amendment.json', amendment)
        write(self.amend/'frozen.sha256.json', {name: adapter.sha(self.amend/name) for name in
            ('request.json', 'source.sha256.json', 'amendment.json')})
        runner, _, evidence = adapter.build_runner(self.run, self.amend)
        self.assertEqual(runner.bin, self.amend/'source/bin')
        self.assertEqual(runner.source, self.fixture.old)
        self.assertEqual(evidence['boundary_sha256'], adapter.sha(self.fixture.cp))

    def test_changed_protected_or_other_workflow_rejected_before_mkdir(self):
        write(self.repo/'workflows/r02_analysis_task.nf', '# changed')
        with self.assertRaisesRegex(ValueError, 'allowlist|preserve'):
            self.execute()
        self.assertFalse(self.amend.exists())

    def test_changed_runner_preprocess_rejected_before_mkdir(self):
        path = self.repo/'bin/r02_autosome_pipeline.py'
        write(path, path.read_text().replace("outdir=str(folder/'preprocess'), cpus=6", "outdir=str(folder/'preprocess'), cpus=9"))
        with self.assertRaisesRegex(ValueError, 'preprocessing implementation'):
            self.execute()
        self.assertFalse(self.amend.exists())

    def test_current_preprocessing_parameters_rejected_before_mkdir(self):
        path = self.run/'chr21/parameters.json'
        value = json.loads(path.read_text())
        value['lai_rare_max_maf'] = .02
        write(path, value)
        with self.assertRaisesRegex(ValueError, 'preprocessing parameters differ'):
            self.execute()
        self.assertFalse(self.amend.exists())

    def test_executor_missing_diagnostics_allowlist_rejected(self):
        path = self.repo/'bin/r02_exec_task.py'
        write(path, path.read_text().replace("    'r02_m14_configuration_diagnostics.py',\n", ''))
        with self.assertRaisesRegex(ValueError, 'executor does not allow'):
            self.execute()
        self.assertFalse(self.amend.exists())

    def test_unrelated_new_code_and_missing_dependency_rejected(self):
        path = self.repo/'bin/unrelated.py'
        write(path, '# unrelated code')
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            self.execute()
        self.assertFalse(self.amend.exists())
        path.unlink()
        (self.repo/'bin/m165_graph_kinship.py').unlink()
        with self.assertRaisesRegex(ValueError, 'allowlist|Missing'):
            self.execute()
        self.assertFalse(self.amend.exists())

    def test_wrong_status_or_parentage_rejected_before_mkdir(self):
        write(self.run/'status.json', dict(stage='chr20_M01_M02_M021', state='RUNNING', pid=222))
        with self.assertRaisesRegex(ValueError, 'currently authenticated'):
            self.execute()
        self.assertFalse(self.amend.exists())
        write(self.run/'status.json', dict(stage='chr21_M01_M02_M021', state='RUNNING', pid=222))
        def wrong(pid):
            ident, parent, command = self.identity(pid)
            return ident, (999 if pid == 222 else parent), command
        self.pid_mock.side_effect = wrong
        with self.assertRaisesRegex(ValueError, 'parentage'):
            self.execute()
        self.assertFalse(self.amend.exists())

    def test_source_mutation_during_copy_produces_no_request(self):
        original = prepare.shutil.copy2
        changed = False
        def mutate(source, target):
            nonlocal changed
            if Path(source).name == 'rare_allele_sharing_painter.py' and not changed:
                changed = True
                write(Path(source), '# concurrent code change')
            return original(source, target)
        with patch.object(prepare.shutil, 'copy2', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'Source changed while freezing'):
                self.execute()
        self.assertFalse((self.amend/'request.json').exists())

    def test_changed_process_after_copy_produces_no_request(self):
        count = 0
        def changed(pid):
            nonlocal count
            count += 1
            ident, parent, command = self.identity(pid)
            if count > 2:
                ident = dict(ident, start_ticks=9999)
            return ident, parent, command
        self.pid_mock.side_effect = changed
        with self.assertRaisesRegex(ValueError, 'Process identity changed'):
            self.execute()
        self.assertFalse((self.amend/'request.json').exists())

    def test_existing_amendment_not_overwritten(self):
        self.execute()
        before = (self.amend/'request.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'new direct subdirectory'):
            self.execute()
        self.assertEqual((self.amend/'request.json').read_bytes(), before)

    def test_original_tampering_rejected_before_module_import(self):
        path = self.fixture.old/'bin/r02_autosome_pipeline.py'
        write(path, "raise AssertionError('Do not execute unauthenticated original code')\n")
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            self.execute()
        self.assertFalse(self.amend.exists())


if __name__ == '__main__':
    unittest.main()
