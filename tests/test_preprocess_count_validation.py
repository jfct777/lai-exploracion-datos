"""Synthetic TBI metadata and task receipts; no cloud jobs or human genotypes."""
import gzip
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('count_validation', ROOT/'bin/preprocess_count_validation.py')
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def tbi_bytes(*, counts=None, chrom=21, duplicate=False, chunks=1, unplaced=None):
    """Minimal TBI with one data bin, optionally the htslib metadata bin."""
    name = f'chr{chrom}\0'.encode()
    data = b'TBI\x01' + struct.pack('<8i', 1, 2, 1, 2, 0, 35, 0, len(name)) + name
    data += struct.pack('<i', 1 + (counts is not None) + duplicate)
    data += struct.pack('<Ii', 4681, chunks) + struct.pack('<QQ', 100, 200) * chunks
    if counts is not None:
        data += struct.pack('<IiQQQQ', 37450, 2, 100, 200, counts, 0)
    if duplicate:
        data += struct.pack('<IiQQ', 4681, 1, 100, 200)
    data += struct.pack('<iQ', 1, 100)
    if unplaced is not None:
        data += struct.pack('<Q', unplaced)
    return data


class CountValidationTests(unittest.TestCase):
    def fixture(self, root, *, raw=0, counts=None, split=False):
        counts_path = root/'preprocess/02_filter/dnabr.hg38.2723.chr21.counts.tsv'
        counts_path.parent.mkdir(parents=True)
        counts_path.write_text(f'chr\tstep\tn_variants\n21\traw\t{raw}\n21\tnorm\t10\n')
        annotation = root/'work/aa/bbbbbb123456'
        annotation.mkdir(parents=True)
        (annotation/'.exitcode').write_text('0\n')
        (annotation/'.command.out').write_text('Annotated 9 records at 8 supplied-input sites\n')
        index = root/'source.vcf.gz.tbi'
        index.write_bytes(gzip.compress(tbi_bytes(counts=counts)))
        (annotation/'dnabr.hg38.2723.chr21.vcf.gz.tbi').symlink_to(index)
        (root/'parameters.json').write_text(json.dumps({'r02_raw_vcf':str(root/'source.vcf.gz')}))
        trace_header = 'hash\tname\tstatus\texit\n'
        if split:
            normalization = root/'work/cc/dddddd123456'
            normalization.mkdir(parents=True)
            (normalization/'.exitcode').write_text('0')
            trace = ('aa/bbbbbb\tPREPROCESS_NORM_LEFTALIGN_CHECKPOINTED:ANNOTATE_ORIGINAL_ALLELES (chr21)\tCACHED\t0\n'
                     'cc/dddddd\tPREPROCESS_NORM_LEFTALIGN_CHECKPOINTED:NORMALIZE_ANNOTATED_ALLELES (chr21)\tCOMPLETED\t0\n')
        else:
            normalization = annotation
            trace = 'aa/bbbbbb\tPREPROCESS_NORM_LEFTALIGN (chr21)\tCOMPLETED\t0\n'
        norm_log = normalization/'dnabr.hg38.2723.chr21.norm.log'
        norm_log.write_text('Lines   total/split/realigned/skipped:\t9/1/2/0\n')
        (root/'trace.tsv').write_text(trace_header + trace)
        return counts_path, annotation, index, norm_log

    def test_missing_statistics_are_unknown_not_zero_and_no_files_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts, task, index, norm_log = self.fixture(root)
            before = {str(p): validation.sha(p) for p in root.rglob('*') if p.is_file()}
            result = validation.validate_raw_record_count(root,21)
            self.assertEqual(result['schema'], validation.SCHEMA)
            self.assertEqual(result['raw_index_records'],0)
            self.assertEqual(result['raw_records'],9)
            self.assertFalse(result['source_index']['record_statistics_present'])
            self.assertIsNone(result['source_index']['mapped_records'])
            self.assertEqual(result['raw_record_count_source'],'sequential_m01_with_missing_tbi_statistics')
            self.assertEqual(before, {str(p): validation.sha(p) for p in root.rglob('*') if p.is_file()})

    def test_split_checkpointed_m01_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root,split=True)
            self.assertEqual(validation.validate_raw_record_count(root,21)['raw_records'],9)

    def test_real_statistics_mismatch_remains_fatal(self):
        for indexed_count in (0,8,9):
            with self.subTest(count=indexed_count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.fixture(root,counts=indexed_count)
                with self.assertRaisesRegex(ValueError,'not explained'):
                    validation.validate_raw_record_count(root,21)

    def test_nonzero_disagreement_is_not_rescued_by_missing_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root,raw=8)
            with self.assertRaisesRegex(ValueError,'does not match'):
                validation.validate_raw_record_count(root,21)

    def test_matching_positive_counts_retain_independent_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root,raw=9,counts=9)
            result=validation.validate_raw_record_count(root,21)
            self.assertEqual(result['raw_record_count_source'],'source_index_and_sequential_m01')

    def test_failed_or_ambiguous_m01_does_not_supply_count(self):
        for change in ('exit','trace','duplicate','missing','wronghash'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _,task,_,_ = self.fixture(root)
                trace = root/'trace.tsv'
                if change == 'exit': (task/'.exitcode').write_text('1')
                elif change == 'trace': trace.write_text(trace.read_text().replace('COMPLETED','FAILED'))
                elif change == 'duplicate': trace.write_text(trace.read_text()+trace.read_text().splitlines()[1]+'\n')
                elif change == 'missing': trace.write_text(trace.read_text().splitlines()[0]+'\n')
                else: trace.write_text(trace.read_text().replace('aa/bbbbbb','../escape'))
                with self.assertRaises(ValueError):validation.validate_raw_record_count(root,21)

    def test_superseded_failed_logs_do_not_contaminate_current_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            stale = root/'work/ee/ffffff/.command.out'; stale.parent.mkdir(parents=True)
            stale.write_text('Annotated 999 records at 888 supplied-input sites\n')
            self.assertEqual(validation.validate_raw_record_count(root,21)['raw_records'],9)

    def test_m01_duplicate_or_impossible_counts_rejected(self):
        for content in ('', 'Annotated 8 records at 9 supplied-input sites\n',
                        'Annotated 9 records at 8 supplied-input sites\n'*2):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _,task,_,_ = self.fixture(root)
                (task/'.command.out').write_text(content)
                with self.assertRaises(ValueError):validation.validate_raw_record_count(root,21)

    def test_normalization_must_corroborate_and_skip_nothing(self):
        for totals in ('8/1/2/0','9/1/2/1',''):
            with self.subTest(totals=totals), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _,_,_,norm_log = self.fixture(root)
                norm_log.write_text('Lines total/split/realigned/skipped:\t'+totals+'\n')
                with self.assertRaisesRegex(ValueError,'does not corroborate'):
                    validation.validate_raw_record_count(root,21)

    def test_wrong_index_path_chromosome_or_empty_index_rejected(self):
        for kind in ('path','chromosome','empty','unplaced'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _,_,index,_ = self.fixture(root)
                if kind == 'path':
                    other = root/'other.tbi';other.write_bytes(index.read_bytes());index=other
                elif kind == 'chromosome':index.write_bytes(gzip.compress(tbi_bytes(chrom=20)))
                elif kind == 'empty':index.write_bytes(gzip.compress(tbi_bytes(chunks=0)))
                else:index.write_bytes(gzip.compress(tbi_bytes(unplaced=1)))
                with self.assertRaises(ValueError):
                    validation.validate_raw_record_count(root,21,source_index=index)

    def test_parser_rejects_corrupt_trailing_truncated_or_duplicate_metadata(self):
        for data in (b'not TBI', tbi_bytes()[:-1], tbi_bytes()+b'x', tbi_bytes(duplicate=True)):
            with self.subTest(data=data[:20]), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'bad.tbi';path.write_bytes(gzip.compress(data))
                with self.assertRaises(ValueError):validation.tabix_record_statistics(path,21)

    def test_parser_distinguishes_absent_zero_and_positive_statistics(self):
        for count in (None,0,9):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'index.tbi';path.write_bytes(gzip.compress(tbi_bytes(counts=count)))
                result=validation.tabix_record_statistics(path,21)
                self.assertEqual(result['mapped_records'],count)
                self.assertEqual(result['record_statistics_present'],count is not None)


if __name__ == '__main__':
    unittest.main()
