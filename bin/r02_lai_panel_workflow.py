#!/usr/bin/env python3
"""Authenticate autosomal preparations, merge with PLINK2, and summarize sensitivity.

No thresholds select donors, no role is changed, and no simulation is launched.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            require(k not in d, 'Duplicate JSON key')
            d[k] = v
        return d
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def auth(d):
    p = Path(d['path'])
    require(p.is_file() and sha(p) == d['sha256'], 'Input hash mismatch: ' + p.name)
    return p


def bundle(path):
    p = Path(path).resolve()
    m = read(p/'manifest.json')
    for name, digest in m['outputs_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe manifest name')
        require(sha(p/name) == digest, 'Product changed: ' + name)
    return p, m


def finish(out, schema, status, **kw):
    save(out/'manifest.json', dict(schema=schema, status=status, **kw,
        outputs_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))


def merge(contract_path, digest, preparations, out):
    require(sha(contract_path) == digest, 'Campaign changed')
    c = read(contract_path)
    require(c['schema'] == 'r02_lai_panel_campaign_v1', 'Wrong campaign')
    all_preps = [bundle(p) for p in preparations]
    by_chr = {str(m['chromosome']):(p,m) for p,m in all_preps}
    require(len(by_chr) == len(all_preps) and set(by_chr) == set(c['chromosomes']), 'Incomplete/duplicate chromosomes')
    roles = auth(c['kinship_roles'])
    with roles.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    ids = {r['sample_id'] for r in rows}
    require(len(ids) == len(rows) and all(r['role'] in ('REF_TRAIN','SOURCE_VALID') for r in rows), 'Invalid open roles')
    out.mkdir(); out=out.resolve()
    prefixes=[]; nmarkers=0
    for chrom in sorted(by_chr, key=int):
        p,m=by_chr[chrom]
        require(m['schema']=='r02_lai_panel_prepare_v1', 'Wrong preparation')
        require(m['status']=='COMPLETE_PANEL_PREPARATION_NOT_KINSHIP_PHASE_OR_BIOLOGICAL_VALIDATION', 'Preparation incomplete')
        expected = next(j['contract']['sha256'] for j in c['jobs'] if j['chromosome']==chrom)
        require(m['inputs']['contract']['sha256']==expected, 'Preparation uses another contract')
        fam=[line.split() for line in (p/'kinship.fam').read_text().splitlines()]
        require(len(fam)==len(ids) and {r[1] for r in fam}==ids, 'Per-chromosome FAM differs')
        bim=[line.split() for line in (p/'kinship.bim').read_text().splitlines()]
        require(bim and all(r[0].removeprefix('chr')==chrom for r in bim), 'Wrong chromosome BED')
        nmarkers+=len(bim); prefixes.append(str(p/'kinship'))
    (out/'merge-list.txt').write_text('\n'.join(prefixes)+'\n')
    plink=auth(c['plink2'])
    command=[str(plink),'--pmerge-list',str(out/'merge-list.txt'),'bfile','--make-bed','--threads',str(c['merge_threads']),
      '--memory',str(c['merge_memory_mib']),'--out',str(out/'panel')]
    save(out/'command.json',command)
    with (out/'execution.log').open('x') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=c['merge_timeout_seconds'])
    merged_fam=[r.split() for r in (out/'panel.fam').read_text().splitlines()]
    require(len(merged_fam)==len(ids) and {r[1] for r in merged_fam}==ids, 'Merge changed samples')
    with (out/'panel.bim').open() as f:
        require(sum(1 for _ in f)==nmarkers,'Merge changed marker count')
    shutil.copyfile(roles,out/'roles.tsv')
    desc=lambda name:dict(path=str(out/name),sha256=sha(out/name))
    save(out/'gds_contract.json',dict(schema='r02_lai_panel_gds_input_v1',
      inputs={k:desc(v) for k,v in dict(bed='panel.bed',bim='panel.bim',fam='panel.fam',roles='roles.tsv').items()},
      ld=c['ld'],kinship=c['kinship']))
    finish(out,'r02_lai_panel_merge_v1','COMPLETE_PRIVATE_AUTOSOMAL_COMMONS_NOT_KINSHIP',
      campaign_sha256=digest,chromosomes=c['chromosomes'],samples=len(ids),markers=nmarkers,
      preparation_manifests={k:sha(p/'manifest.json') for k,(p,m) in by_chr.items()})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True)
    p.add_argument('--preparation',action='append',required=True);p.add_argument('--outdir',type=Path,required=True)
    a=p.parse_args(argv)
    merge(a.contract,a.contract_sha256,a.preparation,a.outdir)


if __name__=='__main__': main()
