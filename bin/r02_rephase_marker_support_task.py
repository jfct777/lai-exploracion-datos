#!/usr/bin/env python3
"""Bounded host I/O plus an isolated metadata-only Docker reader."""
import argparse,hashlib,json,os,re,signal,subprocess,sys,time,uuid
from pathlib import Path
import r02_publish_evidence as pub
import r02_lai_case_ownership as transfer
require=pub.require;read=pub.read_unique_json;save=pub.write_receipt_new
IMAGE='sha256:76a0fddd300d3634363b5a13a1d28983122644a42e7d6c18339f9c8ac08548a7'

def metadata(record,deadline):
 uri,separator,generation=record['uri'].partition('#')
 require(separator=='#'and generation==str(record['generation']),'Explicit pinned stream generation required')
 require(re.fullmatch(r'gs://projects-usp/dna-do-brasil/dnabr-lai-gnomix/persnplai_default_rephase_results/chr([1-9]|1[0-9]|2[0-2])/dnabr\.chr\1\.rephased\.vcf',uri),'Only exact legacy rephased VCF metadata is readable')
 value=json.loads(subprocess.check_output(['gcloud','storage','objects','describe',uri,'--format=json'],text=True,timeout=min(120,max(.01,deadline-time.time()))))
 require(str(value['generation'])==str(record['generation'])and int(value['size'])==record['bytes']and value['md5_hash']==record['md5_base64'],'Pinned remote source changed')
 return value

def stop_child(child):
 if child is None or child.poll()is not None:return
 os.killpg(child.pid,signal.SIGTERM)
 try:child.wait(timeout=5)
 except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

def cleanup_container(name,token):
 q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
 if q.returncode:
  require('No such object'in q.stderr or 'No such container'in q.stderr,'Cannot establish container absence');return
 value=json.loads(q.stdout);require(len(value)==1 and value[0]['Config']['Labels'].get('r02-owner-token')==token,'Container ownership differs')
 subprocess.run(['docker','rm','--force',name],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=15)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['axis','stream']);ap.add_argument('--spec',required=True,type=Path);ap.add_argument('--axis',type=Path);ap.add_argument('--deadline-epoch',required=True,type=float);args=ap.parse_args()
 started=time.monotonic();spec=read(args.spec);require(spec['task_script_sha256']==pub.pipeline.sha(__file__),'Host task version differs')
 root=Path(spec['runtime_root']);cwd=Path.cwd();require(root.is_absolute()and root.resolve()==root and cwd.is_relative_to(root),'Owned canonical local work directory required')
 require(subprocess.check_output(['findmnt','-n','-o','FSTYPE','--target',str(root)],text=True).strip()=='ext4','Local ext4 required')
 deadline=min(args.deadline_epoch,time.time()+spec['task_seconds_limit']);require(deadline-time.time()>5,'No admitted task time remains')
 def expired(sig,frame):raise TimeoutError('Metadata-only task absolute time limit')
 signal.signal(signal.SIGTERM,expired);signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
 output_name='axis-metadata'if args.stage=='axis'else'result'
 for existing in [output_name,'complete.json']:
  path=cwd/existing
  if path.exists():
   prior=cwd/'previous-attempts'/str(time.time_ns());prior.mkdir(parents=True);path.rename(prior/existing)
 token=uuid.uuid4().hex;name='r02-phase-'+spec['chromosome']+'-'+args.stage+'-'+token[:8];docker=None;source=None
 save(cwd/('attempt-'+token+'.json'),dict(stage=args.stage,chromosome=spec['chromosome'],spec_sha256=pub.pipeline.sha(args.spec),deadline_epoch=deadline,container_name=name,owner_token=token,reader_fields=['CHROM','POS','REF','ALT'],genotype_arrays_opened=False))
 try:
  before=None
  if args.stage=='axis':
   record=spec['events_archive'];archive=Path(record['local_path']);archive.parent.mkdir(parents=True,exist_ok=True)
   import shutil
   require(shutil.disk_usage(archive.parent).free>=spec['disk_reserve_bytes']+(0 if archive.exists()else record['bytes']),'Archive staging would consume disk reserve')
   transfer.download_pinned(record,archive,deadline_epoch=deadline)
  else:
   require(args.axis is not None,'Prepared axis missing');before=metadata(spec['rephased_vcf'],deadline);save(cwd/('source-before-'+token+'.json'),before)
  command=['docker','run','--rm','--name',name,'--label','r02-owner-token='+token,'--label','r02-task=k04-phase-metadata','--cpus=1','--memory=1g','--memory-swap=1g','--network=none','--user',str(os.getuid())+':'+str(os.getgid()),'-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-v',str(root)+':'+str(root)+':ro','-v',str(cwd)+':'+str(cwd)+':rw','-w',str(cwd),'--entrypoint','python3']
  if args.stage=='stream':command+=['-i']
  command+=[IMAGE,'-B',str(root/'frozen/bin/r02_rephase_marker_support.py'),args.stage,'--spec',str(args.spec),'--output',output_name]
  if args.axis:command+=['--axis',str(args.axis.resolve())]
  with (cwd/('reader-'+token+'.stdout')).open('xb')as out,(cwd/('reader-'+token+'.stderr')).open('xb')as err,(cwd/('source-'+token+'.stderr')).open('xb')as source_err:
   if args.stage=='stream':source=subprocess.Popen(['gcloud','storage','cat',spec['rephased_vcf']['uri']],stdout=subprocess.PIPE,stderr=source_err,start_new_session=True)
   docker=subprocess.Popen(command,stdin=source.stdout if source else subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
   if source:source.stdout.close()
   require(docker.wait(timeout=max(.01,deadline-time.time()))==0,'Metadata reader failed; partial attempt preserved')
   if source:require(source.wait(timeout=max(.01,deadline-time.time()))==0,'Pinned VCF transport failed')
  result=read(cwd/output_name/'manifest.json');require(result['spec_sha256']==pub.pipeline.sha(args.spec),'Result input binding differs')
  if before is not None:
   after=metadata(spec['rephased_vcf'],deadline);save(cwd/('source-after-'+token+'.json'),after)
   require(all(before[k]==after[k]for k in ['generation','size','md5_hash','crc32c_hash']),'Source metadata changed during whole-stream authentication')
  save(cwd/'complete.json',dict(schema='r02_rephase_metadata_task_complete_v1',stage=args.stage,chromosome=spec['chromosome'],status='AUTHENTICATED_METADATA_ONLY_NO_GT_ANALYSIS',result_manifest_sha256=pub.pipeline.sha(cwd/output_name/'manifest.json'),spec_sha256=pub.pipeline.sha(args.spec),host_code_sha256=pub.pipeline.sha(__file__),source_generation=spec['rephased_vcf']['generation']if args.stage=='stream'else spec['events_archive']['generation'],source_verified_before_and_after=(args.stage=='stream'),archive_identity_checked_before_reuse=(args.stage=='axis'),seconds_total=time.monotonic()-started))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);stop_child(docker);stop_child(source);cleanup_container(name,token)
if __name__=='__main__':main()
