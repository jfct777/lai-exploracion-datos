#!/usr/bin/env python3
"""Stage only declared common arrays from authenticated R02 event archives.

No genotype interpretation, publication, deletion, or scientific admission.
The original compressed archives remain available for later rare consumers.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,os,re,shutil,signal,subprocess,tarfile,time
from pathlib import Path,PurePosixPath
import r02_publish_evidence as pub
import r02_autosome_pipeline as pipeline

require=pub.require
SCHEMA='r02_exact_common_event_staging_v1'

def remaining(deadline_epoch,ceiling):
 require(isinstance(deadline_epoch,(int,float))and not isinstance(deadline_epoch,bool),'Absolute deadline required')
 seconds=deadline_epoch-time.time();require(seconds>0,'Staging absolute deadline reached')
 return min(seconds,ceiling)

def transfer(command,timeout):
 child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 try:
  out,err=child.communicate(timeout=timeout)
  if child.returncode:raise subprocess.CalledProcessError(child.returncode,command,output=out,stderr=err)
 except BaseException:
  if child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.communicate(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.communicate()
  raise

def within(path,root):
 path=Path(path);root=Path(root)
 require(path.is_absolute()and root.is_absolute()and '..'not in path.parts and path.resolve()==path and root.resolve()==root,'Canonical absolute path required')
 require(path.is_relative_to(root)and path!=root,'Staging path escaped runtime')
 require(not any(p.is_symlink()for p in (path,*path.parents)),'Symlink in staging path')
 return path

def matches(path,record,md5=False):
 require(Path(path).is_file()and not Path(path).is_symlink(),'Missing or linked source')
 actual=pipeline.publication_digests(path)
 require(actual['bytes']==record['bytes']and actual['sha256']==record['sha256'],'Existing or transferred bytes differ: '+str(path))
 if md5:require(actual['md5_base64']==record['md5_base64'],'MD5 differs: '+str(path))

def material(row,metadata_root):
 chromosome=row['chromosome'];require(type(chromosome)is str and chromosome in map(str,range(1,23)),'Invalid autosome')
 inputs={}
 for kind in ('inventory','manifest'):
  path=within(Path(metadata_root)/pub.safe_relative(row['metadata_'+kind]),Path(metadata_root));matches(path,row[kind],True);inputs[kind]=pub.read_unique_json(path)
 inventory,manifest=inputs['inventory'],inputs['manifest'];require(str(inventory['chromosome'])==str(manifest['chromosome'])==chromosome,'Chromosome differs')
 require(inventory['archive_sha256']==row['archive']['sha256'],'Inventory does not bind archive')
 entries=inventory['entries'];lookup={e['path']:e for e in entries};require(len(lookup)==len(entries),'Duplicate inventory member')
 requested=row['extracted_files'];require(requested and all(re.fullmatch(r'common\.[0-9]{6}\.npz',e['path'])for e in requested),'Only common arrays are eligible')
 selected={e['path']:e for e in requested};require(len(selected)==len(requested),'Duplicate common request')
 declared={e['path']:{k:e[k]for k in ('path','bytes','sha256')}for e in manifest['channels']['common']}
 require(selected==declared,'Common requests differ from exact source manifest')
 selected['manifest.json']={k:row['manifest'][k]for k in ('bytes','sha256')}
 prefix='published/chr'+chromosome+'/'
 for name,entry in selected.items():
  inv=lookup.get(prefix+name);require(inv and inv['kind']=='FILE'and all(inv[k]==entry[k]for k in ('bytes','sha256')),'Selected member not authenticated by inventory')
 return selected,lookup,prefix

def stage_archive(record,destination,*,runtime_root,disk_reserve_bytes,download_timeout_seconds,deadline_epoch,reuse_archive=None):
 destination=within(destination,runtime_root);destination.parent.mkdir(parents=True,exist_ok=True)
 pub.validate_sha(record['sha256']);require(type(record['bytes'])is int and record['bytes']>0 and re.fullmatch(r'[0-9]+',record['generation']),'Invalid archive identity')
 pub.verify_remote(record,pinned=True,timeout_seconds=remaining(deadline_epoch,120))
 if destination.exists():matches(destination,record,True);return destination,'REUSED_AUTHENTICATED_DESTINATION'
 if reuse_archive is not None:
  source=Path(reuse_archive);pub.safe_local(source);matches(source,record,True)
  require(shutil.disk_usage(destination.parent).free>=disk_reserve_bytes+record['bytes'],'Insufficient staging reserve')
  partial=destination.with_name(destination.name+'.partial')
  require(not os.path.lexists(partial),'Previous partial requires explicit recovery')
  with source.open('rb')as inp,partial.open('xb')as out:shutil.copyfileobj(inp,out,1048576)
  operation='COPIED_AUTHENTICATED_LOCAL_EQUIVALENT'
 else:
  require(shutil.disk_usage(destination.parent).free>=disk_reserve_bytes+record['bytes'],'Insufficient staging reserve')
  partial=destination.with_name(destination.name+'.partial');require(not os.path.lexists(partial),'Previous partial requires explicit recovery')
  transfer(['gcloud','storage','cp',record['uri']+'#'+record['generation'],str(partial)],remaining(deadline_epoch,download_timeout_seconds))
  operation='DOWNLOADED_PINNED_GENERATION'
 matches(partial,record,True);pub.verify_remote(record,pinned=True,timeout_seconds=remaining(deadline_epoch,120));os.link(partial,destination);partial.unlink()
 return destination,operation

def extract_common(archive,destination,selected,lookup,prefix,*,runtime_root,disk_reserve_bytes,deadline_epoch):
 destination=within(destination,runtime_root);destination.mkdir(parents=True,exist_ok=True)
 unknown=[p.name for p in destination.iterdir()if p.name not in selected]
 require(not unknown,'Unclassified existing files in common destination')
 for name,r in selected.items():
  path=within(destination/name,runtime_root)
  if os.path.lexists(path):matches(path,r)
 missing=sum(r['bytes']for name,r in selected.items()if not (destination/name).exists())
 require(shutil.disk_usage(destination).free>=disk_reserve_bytes+missing,'Insufficient common extraction reserve')
 seen=set();written=[];expanded=0;ceiling=sum(e.get('bytes',0)for e in lookup.values()if e['kind']=='FILE')
 with tarfile.open(archive,'r|gz')as stream:
  for member in stream:
   remaining(deadline_epoch,120)
   name=PurePosixPath(member.name)
   require(not name.is_absolute()and '..'not in name.parts and '\\'not in member.name and '\x00'not in member.name,'Unsafe archive member')
   if member.isfile():expanded+=member.size;require(expanded<=ceiling,'Archive expands beyond authenticated inventory')
   if not member.name.startswith(prefix):continue
   relative=member.name[len(prefix):]
   if relative not in selected:continue
   require(relative not in seen and member.isfile()and not member.issym()and not member.islnk(),'Duplicate or linked selected member')
   record=selected[relative];require(member.size==record['bytes'],'Selected member size differs');seen.add(relative);target=destination/relative
   if target.exists():continue
   require(shutil.disk_usage(destination).free>=disk_reserve_bytes+member.size,'Disk reserve breached during extraction')
   partial=target.with_name(target.name+'.partial');require(not os.path.lexists(partial),'Previous extraction partial requires explicit recovery')
   with stream.extractfile(member)as inp,partial.open('xb')as out:shutil.copyfileobj(inp,out,1048576)
   matches(partial,record);os.link(partial,target);partial.unlink();written.append(relative)
 require(seen==set(selected),'Missing selected archive members')
 for name,record in selected.items():matches(destination/name,record)
 return dict(files=len(selected),new_files=len(written),extracted_bytes=sum(r['bytes']for r in selected.values()),rare_arrays_extracted=0)

def stage_one(row,*,runtime_root,metadata_root,disk_reserve_bytes,deadline_epoch,download_timeout_seconds=900,reuse_archive=None):
 started=time.monotonic();require(type(disk_reserve_bytes)is int and disk_reserve_bytes>=0 and type(download_timeout_seconds)is int and download_timeout_seconds>0,'Explicit nonnegative disk reserve and positive transfer timeout required')
 remaining(deadline_epoch,120);selected,lookup,prefix=material(row,metadata_root)
 archive,operation=stage_archive(row['archive'],row['archive_destination'],runtime_root=runtime_root,disk_reserve_bytes=disk_reserve_bytes,download_timeout_seconds=download_timeout_seconds,deadline_epoch=deadline_epoch,reuse_archive=reuse_archive)
 result=extract_common(archive,row['destination'],selected,lookup,prefix,runtime_root=runtime_root,disk_reserve_bytes=disk_reserve_bytes,deadline_epoch=deadline_epoch)
 return dict(schema='r02_exact_common_staged_v1',status='COMMON_AND_MANIFEST_VERIFIED_NO_GENOTYPE_ANALYSIS',chromosome=row['chromosome'],archive_operation=operation,archive=dict(row['archive']),archive_path=str(archive),destination=row['destination'],seconds=time.monotonic()-started,code_sha256=pipeline.sha(__file__),**result)

def stage_all(plan_path,expected_sha,*,runtime_root,metadata_root,disk_reserve_bytes,deadline_epoch,download_timeout_seconds=900):
 require(pipeline.sha(plan_path)==expected_sha,'Staging plan changed');plan=pub.read_unique_json(plan_path)
 require(plan['schema']==SCHEMA and plan['extract_rare_arrays']is False and plan['preserve_archives_for_rare_consumer']is True,'Exact common-only scope required')
 rows=plan['chromosomes'];require(len(rows)==22 and {r['chromosome']for r in rows}==set(map(str,range(1,23))),'Exact22 autosomes required')
 require(sum(r['archive']['bytes']for r in rows)==plan['total_compressed_bytes']and sum(e['bytes']for r in rows for e in r['extracted_files'])==plan['total_extracted_common_bytes'],'Staging byte budget differs')
 for row in rows:yield stage_one(row,runtime_root=runtime_root,metadata_root=metadata_root,disk_reserve_bytes=disk_reserve_bytes,deadline_epoch=deadline_epoch,download_timeout_seconds=download_timeout_seconds)
