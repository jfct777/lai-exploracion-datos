import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bin'))
import r02_lai_role_subset as subset


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.vcf'
        self.source.write_text('##fileformat=VCFv4.2\n##contig=<ID=chr22,length=50818468>\n'
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
            '##INFO=<ID=AC,Number=A,Type=Integer,Description="Full cohort">\n'
            '##INFO=<ID=AN,Number=1,Type=Integer,Description="Full cohort">\n'
            '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tR\tV\tT\n'
            'chr22\t10\t.\tA\tC,G\t.\t.\tAC=1,2;AN=6\tGT\t0|2\t1/2\t0/2\n'
            'chr22\t20\t.\tG\tT\t.\t.\tAC=0;AN=4\tGT\t0/0\t0/.\t./.\n')
        self.roles = self.root/'roles.tsv'
        self.roles.write_text('sample_id\trole\tancestry\nR\tREF_TRAIN\tEUR\nV\tSOURCE_VALID\tNAM\nT\tSOURCE_TEST\tNAM\n')

    def test_allowlist_preserves_order(self):
        self.assertEqual(subset.allowlist(self.roles,['R','V','T'],1,{'NAM':1}),['V'])

    def test_bad_roles_and_order_fail(self):
        with self.assertRaises(ValueError): subset.allowlist(self.roles,['V','R','T'],1,{'NAM':1})
        with self.assertRaises(ValueError): subset.allowlist(self.roles,['R','V','T'],2,{'NAM':1})
        with self.assertRaises(ValueError): subset.allowlist(self.roles,['R','V','V'],1,{'NAM':1})

    @unittest.skipUnless(shutil.which('bcftools'), 'Pinned container required')
    def test_full_projection_and_negative_mutations(self):
        ref = self.root/'ref.json'
        ref.write_text(json.dumps(dict(schema='r02_lai_genotype_support_v1',
            status='COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION',
            role_validation=dict(role='REF_TRAIN',n_role_members=1),input_files=dict(roles=dict(sha256=subset.sha(self.roles))))))
        contract=self.root/'contract.json'
        data=dict(schema='r02_lai_role_subset_contract_v1',role='SOURCE_VALID',expected_ref_samples=1,
            expected_source_samples=3,expected_selected_samples=1,expected_ancestries={'NAM':1},
            expected_records=2,bcftools_version='bcftools 1.16',input_sha256={
                'source_vcf':subset.sha(self.source),'roles':subset.sha(self.roles),'ref_receipt':subset.sha(ref)})
        contract.write_text(json.dumps(data))
        args=SimpleNamespace(contract=contract,source_vcf=self.source,roles=self.roles,ref_receipt=ref,
            outdir=self.root/'out',min_free_gib=0,max_line_bytes=8388608,timeout_seconds=30,bcftools='bcftools')
        result=subset.run(args)
        self.assertEqual(result['records'],2)
        self.assertFalse(result['info_ac_an_updated'])
        self.assertFalse(result['source_test_projected'])
        vcf=Path(args.outdir)/'source_valid.chr22.vcf.gz'
        with gzip.open(vcf,'rt') as f: text=f.read()
        self.assertIn('AC=1,2;AN=6',text)
        self.assertIn('\t1/2\n',text)
        self.assertIn('\t0/.\n',text)
        changed=self.root/'changed.vcf'
        changed.write_text(text.replace('\t1/2\n','\t0/2\n'))
        with self.assertRaises(ValueError): subset.verify_projection(self.source,changed,['V'],2,8388608)
        changed.write_text(text.replace('\tA\tC,G\t','\tA\tC\t'))
        with self.assertRaises(ValueError): subset.verify_projection(self.source,changed,['V'],2,8388608)
        changed.write_text('\n'.join(text.splitlines()[:-1])+'\n')
        with self.assertRaises(ValueError): subset.verify_projection(self.source,changed,['V'],2,8388608)
        with self.assertRaises(ValueError): subset.run(args)

    @unittest.skipUnless(os.environ.get('RUN_R02_NEXTFLOW_TEST') == '1', 'Explicit synthetic Nextflow opt-in')
    def test_nextflow_fresh_resume(self):
        project = Path(__file__).resolve().parents[1]
        ref = self.root/'ref.json'
        ref.write_text(json.dumps(dict(schema='r02_lai_genotype_support_v1',
            status='COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION',
            role_validation=dict(role='REF_TRAIN',n_role_members=1),
            input_files=dict(roles=dict(sha256=subset.sha(self.roles))))))
        contract = self.root/'contract.json'
        contract.write_text(json.dumps(dict(schema='r02_lai_role_subset_contract_v1',role='SOURCE_VALID',
            expected_ref_samples=1,expected_source_samples=3,expected_selected_samples=1,
            expected_ancestries={'NAM':1},expected_records=2,bcftools_version='bcftools 1.16',
            input_sha256=dict(source_vcf=subset.sha(self.source),roles=subset.sha(self.roles),ref_receipt=subset.sha(ref)))))
        script = self.root/'r02_lai_role_subset.py'
        shutil.copy2(project/'bin/r02_lai_role_subset.py',script)
        params = self.root/'params.json'
        params.write_text(json.dumps(dict(subset_script=str(script),source_vcf=str(self.source),roles=str(self.roles),
            ref_receipt=str(ref),contract=str(contract),min_free_gib=0,max_line_bytes=8388608,timeout_seconds=30)))
        config = self.root/'resources.config'
        config.write_text("process.executor='local'\nprocess.memory='2 GB'\nprocess.time='3m'\n"
            "process.container='sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9'\n"
            "docker.enabled=true\ndocker.runOptions='--network none --user 1017:1020 --cpus 1 --memory 2g'\n")
        env={**os.environ,'NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true','NXF_SYNTAX_PARSER':'v1',
             'NXF_OPTS':'-Xms64m -Xmx512m'}
        command=['nextflow','-C',str(config),'run',str(project/'workflows/r02_lai_role_subset.nf'),
                 '-params-file',str(params),'-work-dir',str(self.root/'work'),'-ansi-log','false']
        for i in range(3):
            if i == 2:
                old=script.stat()
                script.write_text(script.read_text().replace('Mechanical SOURCE_VALID projection','Mechanical SOURCE_VALID Projection'))
                os.utime(script,ns=(old.st_atime_ns,old.st_mtime_ns))
                self.assertEqual(script.stat().st_size,old.st_size)
            trace=self.root/f'trace{i}.tsv'
            result=subprocess.run(command+['-with-trace',str(trace)]+(['-resume'] if i else []),
                cwd=self.root,env=env,text=True,capture_output=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            with trace.open() as f: rows=list(__import__('csv').DictReader(f,delimiter='\t'))
            self.assertEqual([r['status'] for r in rows],['CACHED' if i==1 else 'COMPLETED'])


if __name__ == '__main__': unittest.main()
