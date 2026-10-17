#!/usr/bin/env python3
"""Atomically retain and verify complete K04 BUILD products; never fit models."""
import argparse, fcntl, hashlib, json, os, re, shutil
from pathlib import Path

SCHEMA = "r02_k04_build_publication_v1"
COMPLETE = "r02_k04_build_66_complete_v1"
TASK = re.compile(r"chr(0[1-9]|1[0-9]|2[0-2])_(AFR|EUR|NAM)\Z")

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def load(path):
    with Path(path).open() as stream: return json.load(stream)

def atomic_json(path, value):
    path=Path(path); tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,path)

def validate_product(folder, task_id, request_sha):
    folder=Path(folder); manifest=load(folder/'manifest.json')
    require(manifest.get('schema')=='r02_k04_ccqcqr_inputs_v1' and
            manifest.get('status')=='COMPLETE_REVIEWED_MODEL_INPUTS_NOT_MODEL_SELECTION_OR_BIOLOGICAL_VALIDATION',
            'Incomplete or wrong BUILD manifest')
    require(manifest.get('request_sha256')==request_sha and
            task_id==f"chr{int(manifest['target_chromosome']):02d}_{manifest['ancestry']}",
            'BUILD task/request identity differs')
    outputs=manifest.get('output_sha256',{})
    require(outputs and set(outputs)=={str(p.relative_to(folder)) for p in folder.rglob('*')
                                      if p.is_file() and p.name!='manifest.json'},
            'BUILD output inventory differs')
    for name,digest in outputs.items():
        path=folder/name
        require(path.is_file() and not path.is_symlink() and sha(path)==digest,
                'BUILD output hash differs: '+name)
    require(sum((folder/name).stat().st_size for name in outputs)==manifest['output_bytes'] and
            manifest.get('evaluate_target_read') is False and manifest.get('model_fit') is False,
            'BUILD bytes/scope differ')
    return manifest,sha(folder/'manifest.json')

def publish(args):
    require(TASK.fullmatch(args.task_id) is not None,'Invalid BUILD task id')
    root=Path(args.output_root).resolve(); require(root.is_dir() and not root.is_symlink(),'Real output root required')
    (root/'receipts').mkdir(mode=0o700,exist_ok=True); (root/'.locks').mkdir(mode=0o700,exist_ok=True)
    source=Path(args.product).resolve(); manifest,manifest_sha=validate_product(source,args.task_id,args.request_sha256)
    destination=root/source.name; receipt_path=root/'receipts'/f'{args.task_id}.json'
    with (root/'.locks'/f'{args.task_id}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if destination.exists() or receipt_path.exists():
            require(destination.is_dir() and receipt_path.is_file(),'Partial prior publication differs')
            prior=load(receipt_path); _,dest_sha=validate_product(destination,args.task_id,args.request_sha256)
            require(prior.get('schema')==SCHEMA and prior.get('manifest_sha256')==dest_sha==manifest_sha,
                    'Prior publication identity differs')
        else:
            reserve=args.reserve_bytes
            require(shutil.disk_usage(root).free >= manifest['output_bytes']+reserve,
                    'Publication disk reserve exhausted')
            partial=root/f'.{source.name}.partial'
            require(not partial.exists(),'Unresolved prior partial publication')
            shutil.copytree(source,partial,symlinks=False)
            _,copied_sha=validate_product(partial,args.task_id,args.request_sha256)
            require(copied_sha==manifest_sha,'Copied BUILD manifest differs')
            os.replace(partial,destination)
            receipt=dict(schema=SCHEMA,status='PUBLISHED_LOCAL_VERIFIED',task_id=args.task_id,
                         request_sha256=args.request_sha256,manifest_sha256=manifest_sha,
                         output_bytes=manifest['output_bytes'],output_files=len(manifest['output_sha256']),
                         destination=str(destination),evaluate_opened=False,models_fitted=0)
            atomic_json(receipt_path,receipt)
        shutil.copy2(receipt_path,args.receipt)

def verify(args):
    require(TASK.fullmatch(args.task_id) is not None,'Invalid BUILD task id')
    root=Path(args.output_root).resolve(); receipt_path=root/'receipts'/f'{args.task_id}.json'
    product=root/f'model_inputs_{args.task_id}'
    require(receipt_path.is_file() and product.is_dir(),'Complete prior publication pair required')
    row=load(receipt_path); _,manifest_sha=validate_product(product,args.task_id,args.request_sha256)
    require(row.get('schema')==SCHEMA and row.get('status')=='PUBLISHED_LOCAL_VERIFIED' and
            row.get('request_sha256')==args.request_sha256 and row.get('manifest_sha256')==manifest_sha and
            row.get('destination')==str(product) and row.get('evaluate_opened') is False and
            row.get('models_fitted')==0,'Prior publication receipt differs')
    shutil.copy2(receipt_path,args.receipt)

def gate(args):
    root=Path(args.output_root).resolve(); receipts=sorted(Path(args.receipts).glob('*.json'))
    expected={f'chr{c:02d}_{a}' for c in range(1,23) for a in ('AFR','EUR','NAM')}
    rows=[load(p) for p in receipts]
    require(len(rows)==66 and {r.get('task_id') for r in rows}==expected,'Exactly 66 publication receipts required')
    for row in rows:
        require(row.get('schema')==SCHEMA and row.get('status')=='PUBLISHED_LOCAL_VERIFIED' and
                row.get('evaluate_opened') is False and row.get('models_fitted')==0,'Publication receipt differs')
        folder=Path(row['destination'])
        require(folder.is_absolute() and folder.parent==root and
                folder.name==f"model_inputs_{row['task_id']}",'Published destination differs')
        _,digest=validate_product(folder,row['task_id'],row['request_sha256'])
        require(digest==row['manifest_sha256'],'Published product/receipt differs')
    value=dict(schema=COMPLETE,status='COMPLETE_66_BUILD_PRODUCTS_LOCAL_VERIFIED',
               batch_sha256=args.batch_sha256,tasks=66,total_output_bytes=sum(r['output_bytes'] for r in rows),
               receipts_sha256={r['task_id']:sha(root/'receipts'/f"{r['task_id']}.json") for r in rows},
               models_fitted=0,evaluate_opened=False)
    atomic_json(root/'COMPLETE.json',value); shutil.copy2(root/'COMPLETE.json',args.complete)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('publish'); q.add_argument('--task-id',required=True); q.add_argument('--request-sha256',required=True)
    q.add_argument('--product',required=True); q.add_argument('--output-root',required=True); q.add_argument('--receipt',required=True)
    q.add_argument('--reserve-bytes',required=True,type=int); q.set_defaults(function=publish)
    q=sub.add_parser('verify'); q.add_argument('--task-id',required=True); q.add_argument('--request-sha256',required=True)
    q.add_argument('--output-root',required=True); q.add_argument('--receipt',required=True); q.set_defaults(function=verify)
    q=sub.add_parser('gate'); q.add_argument('--receipts',required=True); q.add_argument('--output-root',required=True)
    q.add_argument('--batch-sha256',required=True); q.add_argument('--complete',required=True); q.set_defaults(function=gate)
    args=p.parse_args(); os.umask(0o077); args.function(args)
if __name__=='__main__': main()
