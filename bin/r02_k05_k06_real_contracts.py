#!/usr/bin/env python3
"""Build exact local K05 readiness and K06 real-input validation contracts."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def d(p):
    p=Path(p).resolve();
    if not p.is_file(): raise FileNotFoundError(p)
    return {'path':str(p),'sha256':sha(p)}

def build(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    k05_summary=ROOT/'.claude/runs/r02-corrections-20261006j/k05-comparable-20261007w/outputs/summary.json'
    k05={'schema':'r02_k05_paired_comparison_v1','mode':'READINESS',
         'historical_support_audit':d(k05_summary),
         'missing_dependencies':['TWO_PARTITIONS_LEARNED_ONLY_IN_FIT',
             'FIXED_PROJECTIONS_FOR_SAME_PEOPLE_IN_SELECT_AND_EVALUATE',
             'COMMON_DYAD_OUTCOME_TABLE_WITH_PERSON_AND_PAIR_DEPENDENCY_COMPONENTS',
             'ADMITTED_PERSON_OR_DEPENDENCY_COMPONENT_UNCERTAINTY_POLICY']}
    all22=json.loads((ROOT/'.claude/runs/r02-corrections-20261006j/parallel-completion-20261007v/K06_ALL22.json').read_text())
    sources={str(x['chromosome']):Path(x['path']).resolve().parent/'provenance.tsv' for x in all22['sources']}
    people=ROOT/'.claude/runs/r02-community-ibd-20261005m/bundle/community/individuals.private.tsv'
    chromosomes={}
    for chrom in map(str,range(1,23)):
        base=ROOT/f'.claude/runs/r02-community-ibd-20261005m/bundle/ibd/chr{chrom}'
        chromosomes[chrom]={
            'genetic_map':d(ROOT/f'.claude/runs/r02-lai-external-ibd-pairs-20261005f/inputs/genetic.map.chr{chrom}.gz'),
            'segments_contract':d(base/'ibd_contract.json'),'segments':d(base/'intervals.tsv.gz'),
            'reference_provenance':d(sources[chrom])}
    first=json.loads((ROOT/'.claude/runs/r02-community-ibd-20261005m/bundle/ibd/chr1/ibd_contract.json').read_text())
    k06={'schema':'r02_k06_observability_v1','genome_build':'GRCh38','coordinate_system':'1-based-inclusive',
         'analytical_samples_sha256':first['analytical_samples_sha256'],'people':d(people),
         'chromosomes':chromosomes,'calibration':None}
    for name,value in [('k05-readiness.json',k05),('k06-real-inputs.json',k06)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    return {'k05':d(out/'k05-readiness.json'),'k06':d(out/'k06-real-inputs.json')}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(build(p.parse_args().output_dir),indent=2))

