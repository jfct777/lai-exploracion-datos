#!/usr/bin/env python3
"""Stage generation-pinned common and prepared sources for the 88-view campaign."""
import argparse,hashlib,json,os,shutil,subprocess,tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path,PurePosixPath

def require(v,m):
    if not v:raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def pin(p,h):require(Path(p).is_file()and sha(p)==h,'Pinned staging input differs');return load(p)
def checked(path,row):return path.is_file()and path.stat().st_size==row['bytes']and sha(path)==row['sha256']
def fetch(row,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if checked(path,row):return 'reused'
    require(not path.exists(),'Existing staged object differs: '+str(path));partial=path.with_name(path.name+'.partial')
    if partial.exists():
        if checked(partial,row):os.replace(partial,path);return 'reused'
        require(partial.is_file()and not partial.is_symlink(),'Unsafe partial transfer path')
        partial.unlink()
    subprocess.run(['gcloud','storage','cp',row['generation_uri'],str(partial)],check=True)
    require(checked(partial,row),'Transferred object differs: '+row['generation_uri']);os.replace(partial,path);return 'downloaded'
def safe(name):
    p=PurePosixPath(name);require(not p.is_absolute()and '..'not in p.parts and '\\'not in name,'Unsafe tar member');return p
def stage(assembly_path,assembly_sha,archive_path,archive_sha,inventory_path,inventory_sha,root,workers,min_free):
    assembly=pin(assembly_path,assembly_sha);archive_record=pin(archive_path,archive_sha);inventory=pin(inventory_path,inventory_sha)
    require(assembly['view_count']==88 and archive_record['sha256']==inventory['archive_sha256'],'88-view source identity differs')
    root=Path(root).resolve();intent_value={'schema':'r02_common_88_stage_intent_v1',
      'assembly_sha256':assembly_sha,'archive_sha256':archive_sha,
      'inventory_sha256':inventory_sha,'minimum_free_bytes':min_free}
    intent=root/'stage-intent.json'
    if root.exists():
        require(root.is_dir()and intent.is_file()and load(intent)==intent_value,
                'Existing staging root is not the exact resumable campaign')
    else:
        require(shutil.disk_usage(root.parent).free>=min_free,
                'Campaign requires measured minimum free bytes before first stage')
        root.mkdir(mode=0o700);intent.write_text(json.dumps(intent_value,indent=2,sort_keys=True)+'\n')
    prefixes={s:next(r['prepared']['directory_uri'] for r in assembly['views']if r['exclusion_scope']==s)
              for s in ('BLOCK_AND_LD_MARGIN','WHOLE_CHROMOSOME')}
    jobs=[]
    for row in assembly['staging']['objects']:
        matches=[(s,u)for s,u in prefixes.items()if row['uri'].startswith(u+'/')];require(len(matches)==1,'Prepared URI outside exact scopes')
        scope,prefix=matches[0];jobs.append((row,root/'scopes'/scope/row['uri'][len(prefix)+1:]))
    archive_local=root/'source-archive/useful-products.tar';jobs.append((archive_record,archive_local))
    with ThreadPoolExecutor(max_workers=workers)as pool:states=list(pool.map(lambda x:fetch(*x),jobs))
    selected={}
    for row in inventory['entries']:
        if row.get('disposition')=='PRESERVE'and row.get('path','').startswith('published/targets/target_'):
            require(row['path'].endswith(('/manifest.json','/continuous.private.npz')),'Unexpected target member')
            selected[row['path']]=row
    require(len(selected)==176 and len({p.split('/')[2]for p in selected})==88,'Exact 88 common target pairs required')
    members={r['archive_member'] for r in selected.values()};extracted={}
    with tarfile.open(archive_local,'r:*')as tar:
        index={m.name:m for m in tar.getmembers()};require(len(index)==len(tar.getmembers()),'Duplicate tar member')
        for name in members:
            safe(name);m=index.get(name);require(m and m.isfile()and not m.issym()and not m.islnk(),'Selected tar member differs')
            row=next(r for r in selected.values()if r['archive_member']==name);target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            if checked(target,row):extracted[name]=target;continue
            require(not target.exists(),'Existing extracted common member differs')
            with tar.extractfile(m)as src,target.open('xb')as dst:shutil.copyfileobj(src,dst,1<<20)
            require(checked(target,row),'Extracted common member differs');extracted[name]=target
    for name,row in selected.items():
        target=root/name
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);os.link(extracted[row['archive_member']],target)
        require(checked(target,row),'Restored common target differs')
    receipt={'schema':'r02_common_88_staging_receipt_v1','status':'EXACT_88_SOURCES_STAGED_VERIFIED',
      'prepared_objects':assembly['staging']['unique_objects'],'prepared_bytes':assembly['staging']['total_bytes'],
      'archive_bytes':archive_record['bytes'],'common_target_files':176,'downloaded':states.count('downloaded'),
      'reused':states.count('reused'),'evaluate_targets_computed':False}
    (root/'stage-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('assembly','assembly-sha256','archive','archive-sha256','inventory','inventory-sha256','root','workers','min-free-bytes'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(stage(a.assembly,a.assembly_sha256,a.archive,a.archive_sha256,a.inventory,a.inventory_sha256,a.root,int(a.workers),int(a.min_free_bytes)),sort_keys=True))
