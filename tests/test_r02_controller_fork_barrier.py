import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('controller_barrier',Path(__file__).parents[1]/'bin/r02_controller_fork_barrier.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ControllerBarrierTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        self.folder=self.run/'repairs/preprocess-v1/controller-boundary';self.folder.mkdir(parents=True)
        self.path=self.folder/'manifest.json';self.path.write_text('{}')
        (self.run/'run.json').write_text('{"run_id":"test-r02"}')
        (self.run/'status.json').write_text('{"state":"COMPLETE","stage":"COMPLETE_PUBLISHED"}')
        coordinator=self.run/'repairs/preprocess-v1/coordinator';coordinator.mkdir()
        (coordinator/'resume_status.json').write_text('{"state":"PARALLEL_AGGREGATION_COMPLETE"}')
        self.successor=self.folder/'resume.py';self.successor.write_text('# test')
        self.manifest=self.folder/'resume.json';self.manifest.write_text('{}')
        done=self.run/'repairs/local-preprocess-v1';done.mkdir(parents=True);(done/'completed.json').write_text('{}')
        self.config=dict(run_dir=str(self.run),supervisor={'pid':123},child={'pid':124},
            original_limits=[500,500],deadline_utc='2099-01-01T00:00:00+00:00',user='jose.tantalean',
            successor=dict(script=str(self.successor),script_sha256=m.sha(self.successor),
                           manifest=str(self.manifest),manifest_sha256=m.sha(self.manifest)))
        self.module=mock.Mock()
        self.module.authenticated.side_effect=[{'pid':124},None,None]
        self.module.containers.return_value=[]
        self.module.write_json.side_effect=lambda p,value,**kwargs:Path(p).write_text(json.dumps(value))

    def context(self):
        stack=__import__('contextlib').ExitStack()
        stack.enter_context(mock.patch.object(m.os,'geteuid',return_value=0))
        stack.enter_context(mock.patch.dict(m.os.environ,{'CLOUDSDK_CONFIG':'/tmp/test-cloud-config'}))
        stack.enter_context(mock.patch.object(m.resource,'prlimit',return_value=(500,500)))
        stack.enter_context(mock.patch.object(m.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)))
        return stack

    def test_success_preserves_child_and_invokes_bound_successor(self):
        with self.context():m.apply(self.config,self.path,self.module)
        self.module.send.assert_has_calls([
            mock.call(self.config['supervisor'],m.signal.SIGSTOP),
            mock.call(self.config['supervisor'],m.signal.SIGCONT)])
        self.module.stop_containers.assert_not_called()
        self.assertEqual(json.loads((self.folder/'status.json').read_text())['state'],'SUCCESSOR_COMPLETED')

    def test_child_exit_before_restricting_resumes_supervisor_without_stopping_child(self):
        self.module.authenticated.side_effect=[None]
        with self.context(),self.assertRaisesRegex(ValueError,'before arming'):m.apply(self.config,self.path,self.module)
        self.module.stop_containers.assert_not_called()
        self.module.send.assert_any_call(self.config['supervisor'],m.signal.SIGCONT)

    def test_missing_recovery_never_starts_successor(self):
        (self.run/'repairs/local-preprocess-v1/completed.json').unlink()
        self.module.authenticated.side_effect=[{'pid':124},None,None,None,None]
        with self.context(),mock.patch.object(m.subprocess,'run') as launch,self.assertRaisesRegex(ValueError,'genuine recovery'):
            m.apply(self.config,self.path,self.module)
        launch.assert_not_called()

    def test_changed_successor_is_rejected(self):
        self.successor.write_text('# changed')
        self.module.authenticated.side_effect=[{'pid':124},None,None,None,None]
        with self.context(),mock.patch.object(m.subprocess,'run') as launch,self.assertRaisesRegex(ValueError,'changed before execution'):
            m.apply(self.config,self.path,self.module)
        launch.assert_not_called()

    def test_no_second_arm(self):
        (self.folder/'armed.json').write_text('{}')
        with self.context(),self.assertRaisesRegex(ValueError,'already armed'):m.apply(self.config,self.path,self.module)
        self.module.send.assert_not_called()

    def test_nonroot_cannot_apply(self):
        with mock.patch.object(m.os,'geteuid',return_value=1000),self.assertRaisesRegex(ValueError,'root service'):
            m.apply(self.config,self.path,self.module)
        self.module.send.assert_not_called()

if __name__=='__main__':unittest.main()
