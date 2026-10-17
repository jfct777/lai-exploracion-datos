#!/usr/bin/env python3
"""Stage one of the 88 authenticated FIT/SELECT target bundles for the trainer."""
from __future__ import annotations
import argparse, base64, hashlib, json, re, shutil, subprocess, tarfile
from pathlib import Path, PurePosixPath


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def download(descriptor, target):
    require(str(descriptor['uri']).startswith('gs://projects-usp/dnaBr-lai/datalake/') and
            re.fullmatch(r'[1-9][0-9]*',str(descriptor['generation'])), 'unpinned target object')
    subprocess.run(['gcloud','storage','cp',descriptor['uri']+'#'+str(descriptor['generation']),str(target)],check=True)
    require(target.stat().st_size==descriptor['bytes'] and sha(target)==descriptor['sha256'],
            'downloaded target object differs')


def validate_bundle(folder, view_id):
    manifest_path=folder/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    require(manifest.get('schema')=='r02_k03_neural_target_adapter_result_v1' and
            manifest.get('status')=='FIT_SELECT_TARGETS_COMPLETE_NOT_TRAINING_ADMISSION' and
            manifest.get('evaluate_targets_computed') is False and set(manifest.get('targets',{}))=={'FIT','SELECT'},
            'target manifest differs')
    for role in ('FIT','SELECT'):
        item=manifest['targets'][role]
        target=folder/item['path']; receipt=folder/item['receipt_path']
        require(target.is_file() and receipt.is_file() and sha(target)==item['sha256'] and
                sha(receipt)==item['receipt_sha256'], role+' target authentication differs')
        value=json.loads(receipt.read_text())
        require(value.get('stage')==role or value.get('role')==role or role in value.get('roles',[role]),
                role+' target receipt role differs')
    handoff={'schema':'r02_structure_trainer_target_handoff_v1','view_id':view_id,
             'target_manifest_path':str(manifest_path.resolve()),'target_manifest_sha256':sha(manifest_path),
             'roles':['FIT','SELECT'],'evaluate':False,'training_allowed':False,
             'consumer':'bin/r02_context_train.py'}
    (folder/'trainer-target.json').write_text(json.dumps(handoff,indent=2,sort_keys=True)+'\n')
    return handoff


def run(catalogue_path,catalogue_sha,view_id,out):
    require(sha(catalogue_path)==catalogue_sha,'target catalogue changed')
    catalogue=json.loads(Path(catalogue_path).read_text())
    matches=[x for x in catalogue['targets'] if x['view_id']==view_id]
    require(len(matches)==1,'unknown/duplicate target view')
    out=Path(out);out.mkdir(parents=True,exist_ok=False); target=matches[0]
    if target['source']=='common88_direct_objects':
        for name,descriptor in target['objects'].items():
            require(PurePosixPath(name).name==name,'unsafe target file name')
            download(descriptor,out/name)
    else:
        archive=out/'bundle.tar.gz';download(target['archive'],archive)
        expected={PurePosixPath('published/neural-targets')/name for name in
                  ('manifest.json','FIT.target.private.npz','FIT.target.receipt.json',
                   'SELECT.target.private.npz','SELECT.target.receipt.json')}
        with tarfile.open(archive,'r:gz') as source:
            members={PurePosixPath(m.name):m for m in source.getmembers()}
            require(all(name in members and members[name].isfile() and not members[name].issym()
                        and not members[name].islnk() for name in expected),'archive target inventory differs')
            for name in expected:
                stream=source.extractfile(members[name]); require(stream is not None,'cannot extract target')
                with (out/name.name).open('xb') as sink: shutil.copyfileobj(stream,sink,8*1024*1024)
        archive.unlink()
        local_manifest=Path(target['manifest']['path'])
        require(sha(out/'manifest.json')==target['manifest']['sha256']==sha(local_manifest),
                'archived target manifest differs')
    return validate_bundle(out,view_id)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue',required=True,type=Path);p.add_argument('--catalogue-sha256',required=True)
    p.add_argument('--view-id',required=True);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.catalogue,a.catalogue_sha256,a.view_id,a.out)
