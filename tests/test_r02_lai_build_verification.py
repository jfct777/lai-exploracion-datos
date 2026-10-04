"""Real bcftools REF audits of tiny synthetic FASTA/VCF fixtures only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import r02_lai_allele_support as base
import r02_lai_build_verification as audit
from test_r02_lai_genotype_support import GenotypeFixture, write_json


class BuildAuditFixture:
    def __init__(self, root):
        import pysam
        self.fx = GenotypeFixture(root)
        self.fx.panel.write_text(self.fx.panel.read_text().replace('length=50818468', 'length=100'))
        self.fasta = self.fx.root / 'reference.fa'
        sequence = list('A' * 100)
        sequence[19] = 'G'
        self.fasta.write_text('>chr22\n' + ''.join(sequence) + '\n')
        pysam.faidx(str(self.fasta))
        self.identity = self.fx.root / 'reference_identity.json'
        write_json(self.identity, dict(schema='r02_lai_reference_identity_v1', assembly='GRCh38',
                                       assembly_accession='SYNTHETIC_NOT_REAL_ASSEMBLY', source_uri='synthetic://fixture',
                                       identity_evidence='Tiny synthetic test, no biological claim',
                                       fasta_sha256=base.sha256(self.fasta), fai_sha256=base.sha256(str(self.fasta)+'.fai'),
                                       contigs={'chr22': 100}))

    def args(self, output='receipt.json'):
        return audit.parser().parse_args([
            '--vcf', str(self.fx.panel), '--expected-vcf-sha256', base.sha256(self.fx.panel),
            '--fasta', str(self.fasta), '--reference-contract', str(self.identity),
            '--expected-reference-contract-sha256', base.sha256(self.identity),
            '--contig-map', str(self.fx.catalogue.contigs), '--expected-contig-map-sha256', base.sha256(self.fx.catalogue.contigs),
            '--chromosome', '22', '--reference-contig', 'chr22', '--max-records', '10', '--max-line-bytes', '8388608',
            '--timeout-seconds', '30', '--min-free-gib', '0', '--out', str(self.fx.root / output)])


class BuildAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.fixture = BuildAuditFixture(temp.name)
        self.fx, self.fasta, self.identity = self.fixture.fx, self.fixture.fasta, self.fixture.identity

    def args(self, output='receipt.json'):
        return self.fixture.args(output)

    def test_actual_bcftools_receipt_full_input_and_no_repair(self):
        before = self.fx.panel.read_bytes()
        result = audit.run(self.args())
        self.assertEqual(result['method'], audit.METHOD)
        self.assertEqual(result['status'], 'VERIFIED')
        self.assertEqual(result['reference_audit']['records_checked'], 3)
        self.assertEqual(result['reference_audit']['mismatches'], 0)
        self.assertEqual(result['vcf_sha256'], base.sha256(self.fx.panel))
        self.assertEqual(result['fasta_sha256'], base.sha256(self.fasta))
        self.assertEqual(self.fx.panel.read_bytes(), before)
        self.assertNotIn('fictional_ref_', json.dumps(result))
        self.assertTrue(result['package_versions']['bcftools'].startswith('bcftools '))
        with self.assertRaisesRegex(ValueError, 'Receipt exists'):
            audit.run(self.args())

    def test_explicit_contig_rename_does_not_change_source(self):
        self.fx.panel.write_text(self.fx.panel.read_text().replace('chr22', '22'))
        before = self.fx.panel.read_bytes()
        result = audit.run(self.args())
        self.assertEqual(result['reference_audit']['contig_rename'], {'22': 'chr22'})
        self.assertEqual(self.fx.panel.read_bytes(), before)

    def test_last_row_ref_mismatch_fails_no_receipt(self):
        self.fx.panel.write_text(self.fx.panel.read_text().replace('\t30\t.\tA\tT', '\t30\t.\tC\tT'))
        with self.assertRaisesRegex(ValueError, 'REF/FASTA audit failed'):
            audit.run(self.args())
        self.assertFalse((self.fx.root / 'receipt.json').exists())

    def test_reference_hash_and_index_must_be_authenticated(self):
        self.fasta.write_text(self.fasta.read_text().replace('AAAA', 'CAAA', 1))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            audit.run(self.args())
        Path(str(self.fasta)+'.fai').rename(self.fx.root / 'index_saved.fai')
        with self.assertRaisesRegex(ValueError, 'Existing FASTA and adjacent .fai'):
            audit.run(self.args())
        self.assertFalse(Path(str(self.fasta)+'.fai').exists())

    def test_header_contig_length_and_record_bounds_checked(self):
        self.fx.panel.write_text(self.fx.panel.read_text().replace('length=100', 'length=101'))
        with self.assertRaisesRegex(ValueError, 'header contig length'):
            audit.run(self.args())
        self.fx.panel.write_text(self.fx.panel.read_text().replace('length=101', 'length=100').replace('\t30\t', '\t101\t'))
        with self.assertRaisesRegex(ValueError, 'outside the authenticated contig'):
            audit.run(self.args())

    def test_unknown_contig_and_resource_limits_fail_before_audit(self):
        self.fx.panel.write_text(self.fx.panel.read_text().replace('chr22', 'chr23'))
        with self.assertRaisesRegex(ValueError, 'explicit mapping'):
            audit.run(self.args())
        self.fx.panel.write_text(self.fx.panel.read_text().replace('chr23', 'chr22'))
        args = self.args()
        args.max_records = 1
        with self.assertRaisesRegex(ValueError, 'max-records'):
            audit.run(args)


if __name__ == '__main__':
    unittest.main()
