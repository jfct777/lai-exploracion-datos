import gzip
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from validate_rare_contract import validate


class ConsumerContractTests(unittest.TestCase):
    def test_guard_uses_header_even_if_renamed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'historical.rare.vcf.gz'
            with gzip.open(path, 'wt') as handle:
                handle.write('##dnabr_rare_contract=minor_v1\n#CHROM\tPOS\n')
            validate(path, [], 'source_minor')
            for labels, mode in [(['M13'], ''), ([], 'historical_alt'), (['M24'], 'source_minor')]:
                with self.subTest(labels=labels, mode=mode), self.assertRaises(ValueError):
                    validate(path, labels, mode)

    def test_legacy_keeps_explicit_legacy_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'file.vcf'
            path.write_text('##fileformat=VCFv4.2\n#CHROM\tPOS\n')
            validate(path, ['M13'], 'historical_alt')
            with self.assertRaises(ValueError):
                validate(path, [], 'source_minor')

    def test_duplicate_unknown_and_missing_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'file.vcf'
            for text in ['##dnabr_rare_contract=minor_v2\n#CHROM\tPOS\n',
                         '##dnabr_rare_contract=minor_v1\n' * 2 + '#CHROM\tPOS\n', '']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    validate(path, [])
