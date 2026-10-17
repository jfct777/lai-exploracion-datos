#!/usr/bin/env python3
"""Authenticated host transport for the bounded diploid concordance reader.

Only the task's downloaded, reproducible archive is removed after success.
Each chromosome publishes independently, so interrupted campaigns retain work.
"""
import argparse,json,os,shutil,signal,subprocess,time,uuid
from pathlib import Path
import r02_publish_evidence as pub
import r02_lai_case_ownership as transfer
import r02_rephase_marker_support_task as transport

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--sha256',required=True)
    args=ap.parse_args();job_path=Path(args.job).resolve();pub.require(pub.pipeline.sha(job_path)==args.sha256,'Job changed')
    j=pub.read_unique_json(job_path);root=Path(j['root']);cwd=Path.cwd().resolve()
    pub.require(cwd.is_relative_to(root) and root.resolve()==root,'Owned work directory required')
    pub.require(subprocess.check_output(['findmnt','-n','-o','FSTYPE','--target',str(root)],text=True).strip()=='ext4','Local block filesystem required')
    for p,h in j['code_sha256'].items():pub.require(pub.pipeline.sha(root/p)==h,'Frozen code changed: '+p)
    deadline=min(j['deadline_epoch'],time.time()+j['task_seconds'])
    def expire(*_):raise TimeoutError('Genotype audit task deadline')
    signal.signal(signal.SIGTERM,expire);signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.time()))
    token=uuid.uuid4().hex;name='r02-gt-'+j['chromosome']+'-'+token[:8];source=reader=None
    archive=cwd/'downloaded-events.tar.gz'
    try:
        pub.require(shutil.disk_usage(cwd).free>=j['disk_reserve_bytes']+j['archive']['bytes']+j['max_output_bytes'],'Insufficient staging space and reserve')
        transfer.download_pinned(j['archive'],archive,deadline_epoch=deadline)
        c=j['reader_spec'];c['events_archive']['path']=str(archive)
        pub.write_receipt_new(cwd/'reader-spec.json',c)
        before=transport.metadata(j['source'],deadline);pub.write_receipt_new(cwd/'source-before.json',before)
        command=['docker','run','--rm','--name',name,'--label','r02-owner-token='+token,'--label','r02-run=k04-genotype-1007n','--network=none','--cpus=1','--memory=2g','--memory-swap=2g','--user',str(os.getuid())+':'+str(os.getgid()),'-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-v',str(root)+':'+str(root)+':ro','-v',str(cwd)+':'+str(cwd)+':rw','-w',str(cwd),'-i','--entrypoint','python3',j['image'],'-B',str(root/'frozen/bin/r02_rephase_genotype_concordance.py'),'--spec',str(cwd/'reader-spec.json'),'--spec-sha256',pub.pipeline.sha(cwd/'reader-spec.json'),'--output-dir',str(cwd/'result')]
        pub.write_receipt_new(cwd/'started.json',dict(chromosome=j['chromosome'],utc=pub.pipeline.now(),deadline_epoch=deadline,job_sha256=args.sha256,container_name=name))
        with (cwd/'reader.stdout').open('xb') as out,(cwd/'reader.stderr').open('xb') as err,(cwd/'source.stderr').open('xb') as srcerr:
            source=subprocess.Popen(['gcloud','storage','cat',j['source']['uri']],stdout=subprocess.PIPE,stderr=srcerr,start_new_session=True)
            reader=subprocess.Popen(command,stdin=source.stdout,stdout=out,stderr=err,start_new_session=True);source.stdout.close()
            pub.require(reader.wait(timeout=max(.01,deadline-time.time()))==0,'Reader failed; inspect reader.stderr')
            pub.require(source.wait(timeout=max(.01,deadline-time.time()))==0,'GCS stream failed')
        after=transport.metadata(j['source'],deadline);pub.write_receipt_new(cwd/'source-after.json',after)
        pub.require(all(before[k]==after[k] for k in ['generation','size','md5_hash','crc32c_hash']),'Source changed during audit')
        s=pub.read_unique_json(cwd/'result/summary.json');table=cwd/'result'/s['locus_role_table']['path']
        pub.require(pub.pipeline.sha(table)==s['locus_role_table']['sha256'] and table.stat().st_size==s['locus_role_table']['bytes'],'Result table changed')
        pub.require(s['chromosome']==j['chromosome'] and s['status']=='COMPLETE_DIPLOID_DOSAGE_CONCORDANCE_QC_ONLY','Wrong result identity')
        publications=[]
        for p in [table,cwd/'reader-spec.json',cwd/'source-before.json',cwd/'source-after.json',cwd/'result/summary.json']:
            record=dict(name=p.name,path=str(p),uri=j['publication_prefix']+'/'+p.name,**pub.pipeline.publication_digests(p))
            pub.upload_one(record);pub.verify_remote(record,pinned=True);publications.append(record)
        pub.write_receipt_new(cwd/'complete.json',dict(status='CHROMOSOME_COMPLETE_AND_PUBLISHED_DIPLOID_QC_ONLY',utc=pub.pipeline.now(),chromosome=j['chromosome'],job_sha256=args.sha256,files=publications))
        r=dict(name='complete.json',path=str(cwd/'complete.json'),uri=j['publication_prefix']+'/complete.json',**pub.pipeline.publication_digests(cwd/'complete.json'))
        pub.upload_one(r);pub.verify_remote(r,pinned=True)
        pub.write_receipt_new(cwd/'publication.json',r)
        # This is the only task-created, authenticated copy of a pinned GCS input.
        # It is reproducible and not needed to resume a finished Nextflow task.
        pub.require(pub.pipeline.sha(archive)==j['archive']['sha256'],'Downloaded archive identity changed')
        archive.unlink()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);transport.stop_child(reader);transport.stop_child(source);transport.cleanup_container(name,token)

if __name__=='__main__':main()
