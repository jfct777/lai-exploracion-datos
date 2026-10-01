"""Operational contracts; no human genotypes or cloud jobs."""
import importlib.util
import json
import gzip
import os
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('r02_controller',ROOT/'bin/r02_autosome_pipeline.py')
pipeline=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class ControllerTests(unittest.TestCase):
    def test_live_controller_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pipeline.execution_lock(tmp):
                with self.assertRaisesRegex(RuntimeError,'already owns'):
                    with pipeline.execution_lock(tmp): pass
            with pipeline.execution_lock(tmp): pass

    def test_generation_and_size_are_rechecked(self):
        r=object.__new__(pipeline.Runner)
        r.objects={'/source':{'generation':'10','size':42,'crc32c_hash':'abc'}}
        with patch.object(pipeline,'object_metadata',return_value={'generation':10,'size':42,'crc32c_hash':'abc'}):
            r.verify_inputs(['/source'])
            with self.assertRaisesRegex(ValueError,'not authenticated'): r.verify_inputs(['/other'])
        for bad in ({'generation':'11','size':42},{'generation':'10','size':43}):
            with patch.object(pipeline,'object_metadata',return_value=bad):
                with self.assertRaisesRegex(ValueError,'changed'): r.verify_inputs(['/source'])

    def test_publication_excludes_work_and_large_temporary_genotypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            keep=['rare_evidence.npz','rare_evidence.manifest.json','common/primary/grm.grm.bin',
                  'preprocess/lai_rare/new.rare.minor.vcf.gz','preprocessing_counts.tsv']
            skip=['work/aa/bb/large.vcf.gz','preprocess/02_filter/large.vcf.gz',
                  '.nextflow/cache/data','common/common.pgen','common/common.filtered.bcf',
                  'common/common.pvar','pairs.sqlite','.nextflow.log']
            for name in keep+skip:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
            self.assertEqual({str(p.relative_to(root)) for p in pipeline.publication_files(root)},set(keep))

    def test_sequential_counts_must_match_old_index_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            counts=root/'preprocess/02_filter/dnabr.hg38.2723.chr21.counts.tsv'
            counts.parent.mkdir(parents=True)
            counts.write_text('chr\tstep\tn_variants\n21\traw\t9\n21\tnorm\t10\n')
            log=root/'work/aa/bb/.command.out';log.parent.mkdir(parents=True)
            log.write_text('Annotated 9 records at 8 supplied-input sites\n')
            self.assertEqual(pipeline.validate_raw_record_count(root,21)['m01_original_sites'],8)
            log.write_text('Annotated 8 records at 8 supplied-input sites\n')
            with self.assertRaisesRegex(ValueError,'does not match'): pipeline.validate_raw_record_count(root,21)

    def test_checkpoint_rejects_changed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result';p.write_text('correct')
            records=pipeline.files_manifest([p]);pipeline.verify_files(records)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed or is missing'): pipeline.verify_files(records)

    def publication_fixture(self, root):
        r=object.__new__(pipeline.Runner)
        r.run=root/'run';r.run.mkdir();(r.run/'checkpoints').mkdir()
        r.dest=root/'unmounted';r.current_stage='old_calculation'
        folder=root/'results';folder.mkdir();source=folder/'result.txt';source.write_text('verified bytes')
        return r,folder,source

    def test_publication_does_not_require_fuse_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,folder,source=self.publication_fixture(root)
            digest=pipeline.publication_digests(source)
            metadata={'generation':'123','size':digest['bytes'],'md5_hash':digest['md5_base64']}
            with patch.object(pipeline,'MOUNT',root), patch.object(pipeline,'object_metadata',return_value=metadata), \
                 patch.object(pipeline.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')) as upload:
                r.publish(folder,'chr22')
                self.assertFalse(r.dest.exists())  # FUSE never sees the new object.
                args=upload.call_args.args[0]
                self.assertNotIn('--no-clobber',args)
                self.assertIn('--if-generation-match=0',args)
                self.assertIn('--content-md5='+digest['md5_base64'],args)
                receipt=r.run/'checkpoints/publish_chr22.json'
                self.assertEqual(json.loads(receipt.read_text())['files'][0]['sha256'],digest['sha256'])
                r.publish(folder,'chr22')
                self.assertEqual(upload.call_count,1)  # Verified receipt needs no upload.

    def test_publication_rejects_conflicting_remote_or_missing_checksum(self):
        for bad in ({'generation':'123','size':14,'md5_hash':'bad'},
                    {'generation':'123','size':14}, {'size':14,'md5_hash':'bad'}):
            with self.subTest(metadata=bad), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);r,folder,source=self.publication_fixture(root)
                with patch.object(pipeline,'MOUNT',root), patch.object(pipeline,'object_metadata',return_value=bad), \
                     patch.object(pipeline.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')):
                    with self.assertRaisesRegex(ValueError,'differ from source or lack checksum'):
                        r.publish(folder,'chr22')
                self.assertFalse((r.run/'checkpoints/publish_chr22.json').exists())

    def test_publication_retry_after_upload_does_not_recompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,folder,source=self.publication_fixture(root)
            digest=pipeline.publication_digests(source)
            metadata={'generation':'123','size':digest['bytes'],'md5_hash':digest['md5_base64']}
            with patch.object(pipeline,'MOUNT',root), \
                 patch.object(pipeline.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','ERROR: HTTPError 412: exists')), \
                 patch.object(pipeline,'object_metadata',side_effect=[RuntimeError('API unavailable'),metadata]):
                with self.assertRaisesRegex(RuntimeError,'API unavailable'):r.publish(folder,'chr22')
                self.assertEqual(r.current_stage,'publish_chr22')
                r.publish(folder,'chr22')
                self.assertTrue((r.run/'checkpoints/publish_chr22.json').exists())

    def test_publication_rejects_source_mutation_during_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,folder,source=self.publication_fixture(root)
            digest=pipeline.publication_digests(source)
            metadata={'generation':'123','size':digest['bytes'],'md5_hash':digest['md5_base64']}
            def mutate(*args,**kwargs):
                source.write_text('changed')
                return subprocess.CompletedProcess([],0,'','')
            with patch.object(pipeline,'MOUNT',root), patch.object(pipeline,'object_metadata',return_value=metadata), \
                 patch.object(pipeline.subprocess,'run',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'changed during publication'):r.publish(folder,'chr22')
            self.assertFalse((r.run/'checkpoints/publish_chr22.json').exists())

    def test_publication_does_not_suppress_auth_or_cli_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,folder,source=self.publication_fixture(root)
            with patch.object(pipeline,'MOUNT',root), patch.object(pipeline,'object_metadata') as describe, \
                 patch.object(pipeline.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','auth failed')):
                with self.assertRaises(subprocess.CalledProcessError):r.publish(folder,'chr22')
                describe.assert_not_called()

    def test_preparation_cannot_run_from_frozen_source(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline,'ROOT',Path(tmp)):
            with self.assertRaisesRegex(ValueError,'project checkout'):
                pipeline.prepare(Path(tmp)/'newrun',pipeline.PREP_IMAGE)

    def cleanup_fixture(self,root):
        r=object.__new__(pipeline.Runner)
        r.run=root/'run';r.run.mkdir()
        (r.run/'checkpoints').mkdir()
        bulk=root/'bulk';bulk.mkdir();r.c={'bulk':str(bulk)}
        folder=r.run/'chr21';folder.mkdir()
        marker=folder/'work/aa/bb/test.large_temp_dir.txt';marker.parent.mkdir(parents=True)
        target=bulk/'m01_chr21.ABC12345';target.mkdir()
        (target/'dnabr.hg38.2723.chr21.original.bcf').write_text('new temporary')
        marker.write_text(str(target)+'\n')
        return r,folder,marker,target

    def test_cleanup_requires_publication_and_never_removes_old_chr22(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,folder,marker,target=self.cleanup_fixture(root)
            with self.assertRaisesRegex(ValueError,'before successful publication'): r.cleanup_bulk(folder,21)
            r.cleanup_bulk(folder,22)
            self.assertTrue(target.exists())

    def test_cleanup_is_exact_and_idempotent_after_partial_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,folder,marker,target=self.cleanup_fixture(root)
            receipt=r.run/'checkpoints/publish_01_estructura_desarrollo_por_cromosoma_chr21.json'
            receipt.write_text('{}')
            with patch.object(pipeline,'MOUNT',root):
                r.cleanup_bulk(folder,21)
                self.assertFalse(target.exists())
                # Simulate interruption after unlink/rmdir, before final receipt.
                (folder/'bulk_cleanup.json').unlink()
                r.cleanup_bulk(folder,21)
                self.assertTrue((folder/'bulk_cleanup.json').exists())

    def test_cleanup_rejects_foreign_or_unexpected_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,folder,marker,target=self.cleanup_fixture(root)
            (r.run/'checkpoints/publish_01_estructura_desarrollo_por_cromosoma_chr21.json').write_text('{}')
            bad=target/'unrelated.txt';bad.write_text('do not touch')
            with patch.object(pipeline,'MOUNT',root):
                with self.assertRaisesRegex(ValueError,'Unexpected generated'):r.cleanup_bulk(folder,21)
            self.assertTrue(bad.exists())

    def test_completed_chromosome_does_not_reopen_deleted_intermediates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir()
            result=root/'rare_evidence.npz';result.write_bytes(b'synthetic')
            pipeline.write_new(root/'checkpoints/chr21_complete.json',{'outputs':pipeline.files_manifest([result])})
            r=object.__new__(pipeline.Runner);r.run=root
            with patch.object(r,'preprocess',side_effect=AssertionError('must not reopen')):
                r.analyze_chromosome(21)

    def test_science_checkpoint_resumes_publication_without_recalculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir()
            result=root/'rare_evidence.npz';result.write_bytes(b'synthetic')
            pipeline.write_new(root/'checkpoints/chr22_science_complete.json',
                               {'outputs':pipeline.files_manifest([result])})
            r=object.__new__(pipeline.Runner);r.run=root
            with patch.object(r,'preprocess',side_effect=AssertionError('must not recalculate')), \
                 patch.object(r,'finalize_chromosome') as finalize:
                r.analyze_chromosome(22)
                finalize.assert_called_once_with(root/'chr22',22)


@unittest.skipUnless(os.environ.get('RUN_R02_PREPROCESS_INTEGRATION')=='1',
                     'opt-in synthetic Nextflow/Docker workflow')
class ActualWorkflowTests(unittest.TestCase):
    def test_new_workflow_runs_M01_M02_M021_with_no_sample_subset(self):
        with tempfile.TemporaryDirectory(prefix='r02-workflow-test-',dir=ROOT/'.claude/runs') as tmp:
            base=Path(tmp);bulk=base/'bulk';bulk.mkdir()
            ref=base/'reference.fa';ref.write_text('>chr21\n'+'A'*1000+'\n')
            Path(str(ref)+'.fai').write_text('chr21\t1000\t7\t1000\t1001\n')
            samples=[f's{i}' for i in range(50)]
            vcf=base/'input.vcf'
            header=('##fileformat=VCFv4.2\n##contig=<ID=chr21,length=1000>\n'
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
                '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'+'\t'.join(samples)+'\n')
            rows=[]
            for pos,alt,gts in [(100,'C',['0/1']*2+['0/0']*48),
                                (200,'C',['0/1']*2+['1/1']*48),
                                (300,'C,G',['0/1']*2+['0/0']*48)]:
                rows.append(f'chr21\t{pos}\t.\tA\t{alt}\t.\tPASS\t.\tGT\t'+'\t'.join(gts))
            vcf.write_text(header+'\n'.join(rows)+'\n')
            subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
                '-v',f'{base}:{base}','-w',str(base),pipeline.PREP_IMAGE,'bash','-c',
                'bcftools view -Oz -o raw.vcf.gz input.vcf && bcftools index -t raw.vcf.gz'],check=True,timeout=30)
            params=dict(outdir=str(base/'results'),r02_chrom=21,r02_raw_vcf=str(base/'raw.vcf.gz'),
                r02_bin_dir=str(ROOT/'bin'),ref_fasta=str(ref),preprocess_large_temp_dir=str(bulk),
                cpus=1,memory='1 GB',time='5m',resources={},bcftools_min_alleles=2,
                plink_max_alleles=2,plink_snps_only=True,max_maf=None,keep_pass=True,
                lai_rare_max_maf=.03,lai_rare_min_mac=2,lai_rare_keep_format='GT',lai_rare_remove_info=False)
            pipeline.write_new(base/'parameters.json',params)
            config=base/'runtime.config'
            config.write_text(f"""process.executor='local'
process.container='{pipeline.PREP_IMAGE}'
process.stageInMode='symlink'
docker.enabled=true
docker.runOptions='--network none --user {os.getuid()}:{os.getgid()}'
executor.queueSize=1
process {{
withName:PREPROCESS_NORM_LEFTALIGN {{ publishDir=[enabled:false] }}
withName:PREPROCESS_FILTER_SNV_BIALLELIC_PASS {{ publishDir=[path:'{base}/results/02_filter',mode:'symlink',overwrite:false] }}
withName:LAI_RARE_BIALELIC_ONLY {{ publishDir=[path:'{base}/results/lai_rare',mode:'link',overwrite:false] }}
}}
""")
            result=subprocess.run(['nextflow','-log',str(base/'nextflow.log'),'-C',str(config),'run',
                str(ROOT/'workflows/r02_preprocess_autosome.nf'),'-params-file',str(base/'parameters.json'),
                '-work-dir',str(base/'work'),'-ansi-log','false'],cwd=base,capture_output=True,text=True,
                timeout=180,env={**os.environ,'NXF_SYNTAX_PARSER':'v1','NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true'})
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            out=base/'results/lai_rare/dnabr.hg38.2723.chr21.rare.minor.vcf.gz'
            with gzip.open(out,'rt') as handle:data=[line.rstrip().split('\t') for line in handle if not line.startswith('#')]
            self.assertEqual([int(row[1]) for row in data],[100,200])
            self.assertIn('RARE_ALLELE=0',data[1][7])
            contract=json.loads(out.with_name('dnabr.hg38.2723.chr21.rare.minor.contract.json').read_text())
            self.assertEqual(contract['cohort_samples_after'],50)
            self.assertEqual(contract['counts']['excluded_original_multiallelic'],2)


if __name__=='__main__':unittest.main()
