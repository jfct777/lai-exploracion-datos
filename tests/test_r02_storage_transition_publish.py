"""Synthetic publication tests; no real helper invocation or cloud writes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('storage_publish',ROOT/'bin/r02_storage_transition_publish.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class FakeCloud:
    def __init__(self):self.objects={};self.uploaded=[]
    def upload(self,path,uri):
        content=Path(path).read_bytes()
        if uri in self.objects and self.objects[uri]!=content:raise ValueError('overwrite prohibited')
        self.objects[uri]=content;self.uploaded.append(Path(path).name)
        return dict(name=Path(path).name,uri=uri,generation='1',**m.digests(path))


class PublishTests(unittest.TestCase):
    def fixture(self,root):
        run=root/'parent/parallel/worker01';run.mkdir(parents=True)
        mount=root/'gcs'
        destination=mount/'refined/DNABR_QC/presentacion/biologico/R02_20260930/parent/parallel/worker01'
        m.write_fixed(run/'run.json',dict(destination=str(destination),
            parallel_worker=dict(parent_run_id='parent',worker_id='worker01')))
        m.write_fixed(run/'frozen.sha256.json',{'run.json':m.sha(run/'run.json')})
        spec=root/'spec.json'
        m.write_fixed(spec,dict(schema='r02_local_storage_boundary_v1',run_dir=str(run),
            helper_sha256=m.HELPER_SHA256,worker_run_sha256=m.sha(run/'run.json'),
            worker_frozen_sha256=m.sha(run/'frozen.sha256.json')))
        with patch.object(m,'MOUNT',mount):
            context=m.validate(ROOT/'bin/r02_local_storage_boundary.py',spec,m.sha(spec))
        return context

    def execution(self,context,code):
        def fake(command,**kwargs):
            (context['directory']/'events.jsonl').write_text('{"state":"synthetic"}\n')
            return subprocess.CompletedProcess(command,code)
        return fake

    def test_preflight_does_not_execute_or_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=self.fixture(Path(tmp))
            self.assertFalse(context['directory'].exists())
            self.assertIn('/parallel/worker01/00_worker_provenance/storage_transition/',context['prefix'])

    def test_success_publishes_only_allowlist_and_does_not_change_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=self.fixture(Path(tmp));cloud=FakeCloud();before=m.sha(context['run']/'frozen.sha256.json')
            with patch.object(m.subprocess,'run',side_effect=self.execution(context,0)):
                self.assertEqual(m.run(context,cloud),0)
            self.assertEqual(set(cloud.uploaded),{'helper.py','spec.json','events.jsonl','wrapper_execution.json','publication.json'})
            self.assertEqual(m.sha(context['run']/'frozen.sha256.json'),before)
            self.assertFalse(json.loads(cloud.objects[context['prefix']+'/publication.json'])['genotype_files_uploaded'])

    def test_helper_failure_still_publishes_and_returns_original_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=self.fixture(Path(tmp));cloud=FakeCloud()
            with patch.object(m.subprocess,'run',side_effect=self.execution(context,17)):
                self.assertEqual(m.run(context,cloud),17)
            receipt=m.read(context['directory']/'wrapper_execution.json')
            self.assertEqual(receipt['helper_returncode'],17)
            self.assertIn('publication.json',cloud.uploaded)

    def test_retry_republishes_without_repeating_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=self.fixture(Path(tmp));cloud=FakeCloud()
            with patch.object(m.subprocess,'run',side_effect=self.execution(context,0)) as execute:
                self.assertEqual(m.run(context,cloud),0)
                self.assertEqual(m.run(context,cloud),0)
                self.assertEqual(execute.call_count,1)

    def test_publication_error_preserves_failed_helper_code_and_flags_successful_helper(self):
        for code,expected in ((17,17),(0,74)):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as tmp:
                context=self.fixture(Path(tmp))
                with patch.object(m.subprocess,'run',side_effect=self.execution(context,code)), \
                     patch.object(m,'publish',side_effect=RuntimeError('synthetic')):
                    self.assertEqual(m.run(context,FakeCloud()),expected)
                self.assertEqual(m.read(context['directory']/'wrapper_execution.json')['helper_returncode'],code)

    def test_unfinished_wrapper_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=self.fixture(Path(tmp));context['directory'].mkdir(parents=True)
            m.write_fixed(context['directory']/'wrapper_started.json',{'started':'synthetic'})
            with patch.object(m.subprocess,'run',side_effect=AssertionError('do not repeat')):
                with self.assertRaisesRegex(ValueError,'Interrupted'):m.run(context,FakeCloud())

    def test_arbitrary_fields_and_oversized_files_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context=self.fixture(root);settings=m.read(context['spec']);settings['token']='not-a-secret-fixture'
            context['spec'].write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError,'Unexpected storage specification'):
                m.validate(context['helper'],context['spec'],m.sha(context['spec']))
            p=root/'large';p.write_bytes(b'12345')
            with patch.object(m,'MAX_SMALL_BYTES',4),self.assertRaises(ValueError):m.small_file(p)

    def test_generation_zero_and_md5_checked_even_for_existing_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'events.jsonl';source.write_text('synthetic');digest=m.digests(source)
            responses=[subprocess.CompletedProcess([],1,'','HTTPError 412: exists'),
                       subprocess.CompletedProcess([],0,json.dumps(dict(generation='123',size=digest['bytes'],md5_hash=digest['md5_base64'])),'')]
            with patch.object(m.subprocess,'run',side_effect=responses) as commands:
                record=m.GCS().upload(source,'gs://private/fixture')
            self.assertEqual(record['generation'],'123')
            self.assertIn('--if-generation-match=0',commands.call_args_list[0].args[0])
            self.assertIn('--content-md5='+digest['md5_base64'],commands.call_args_list[0].args[0])

    def test_conflicting_remote_bytes_or_auth_error_are_not_suppressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'events.jsonl';source.write_text('synthetic')
            for responses in ([subprocess.CompletedProcess([],1,'','auth failed')],
                              [subprocess.CompletedProcess([],0,'',''),subprocess.CompletedProcess([],0,json.dumps(dict(generation='1',size=1,md5_hash='wrong')),'')]):
                with patch.object(m.subprocess,'run',side_effect=responses),self.assertRaises(ValueError):
                    m.GCS().upload(source,'gs://private/fixture')


if __name__=='__main__':unittest.main()
