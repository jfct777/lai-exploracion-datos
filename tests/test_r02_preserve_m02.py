import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location('preserve', Path(__file__).parents[1]/'bin/r02_preserve_m02.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class PreservationTests(unittest.TestCase):
    def test_cleanup_cannot_be_reenabled_by_publication(self):
        with self.assertRaises(PermissionError):
            p.cleanup_guard('/nonexistent/spec.json', '0'*64)

    def test_trace_success_and_noncompletion(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'trace.tsv'
            for state, valid in [('COMPLETED', True), ('CACHED', True), ('RUNNING', False), ('FAILED', False)]:
                path.write_text('name\tstatus\texit\nPREPROCESS_FILTER_SNV_BIALLELIC_PASS (chr13)\t'+state+'\t0\n')
                if valid:
                    self.assertEqual(p.validate_m02_trace(path, 13)['status'], state)
                else:
                    with self.assertRaises(ValueError):
                        p.validate_m02_trace(path, 13)
            with self.assertRaises(ValueError):
                p.validate_m02_trace(path, 14)

    def test_metadata_requires_all_fields_and_generation(self):
        expected = dict(bytes=8, md5_base64='ABC', generation='123')
        p.verify_metadata(dict(size=8, md5_hash='ABC', generation=123), expected)
        for actual in [dict(size=9, md5_hash='ABC', generation=123),
                       dict(size=8, md5_hash='DEF', generation=123),
                       dict(size=8, md5_hash='ABC', generation=124)]:
            with self.assertRaises(ValueError):
                p.verify_metadata(actual, expected)

    def test_fixed_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'record.json'
            p.fixed(path, {'a': 1})
            p.fixed(path, {'a': 1})
            with self.assertRaises(ValueError):
                p.fixed(path, {'a': 2})
            self.assertEqual(json.loads(path.read_text()), {'a': 1})

    def test_hashes_and_hardlink_survive_unlink(self):
        with tempfile.TemporaryDirectory() as d:
            source, held = Path(d)/'source', Path(d)/'held'
            source.write_bytes(b'genotypes-preserved')
            p.os.link(source, held)
            first = p.digests(source)
            source.unlink()
            self.assertEqual(p.digests(held), first)

    def test_guard_rejects_wrong_identity_incomplete_and_changed_cloud(self):
        target = dict(destination='gs://private/test/', worker='worker05', chromosome=20,
                      spec_sha256='x'*64, files=[dict(name='a', bytes=1), dict(name='b', bytes=2)])
        receipt = dict(schema='r02_m02_preserved_v1', state='PUBLISHED_VERIFIED',
                       worker='worker05', chromosome=20, spec_sha256='x'*64,
                       files=[dict(name=n, bytes=size, uri='gs://private/test/'+n,
                                   sha256='a'*64, generation='1', md5_base64='MD5')
                              for n, size in [('a', 1), ('b', 2)]])
        class Cloud:
            def metadata(self, uri):
                return dict(generation='1', size=1 if uri.endswith('/a') else 2, md5_hash='MD5')
            def read(self, uri, generation):
                return receipt
        cloud = Cloud()
        self.assertEqual(p.verify_preservation(target, cloud)['state'], 'PUBLISHED_VERIFIED')
        receipt['chromosome'] = 19
        with self.assertRaises(ValueError):
            p.verify_preservation(target, cloud)
        receipt['chromosome'] = 20
        receipt['files'].pop()
        with self.assertRaises(ValueError):
            p.verify_preservation(target, cloud)

    def test_destination_rejects_other_bucket(self):
        with self.assertRaises(ValueError):
            p.destination(dict(worker='worker05', chromosome=20, destination='gs://other/'))


if __name__ == '__main__':
    unittest.main()
