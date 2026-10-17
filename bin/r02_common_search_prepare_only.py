#!/usr/bin/env python3
"""Compute authenticated FIT statistics and finite grids without fit admissions."""
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import r02_structure_k03_common_search as frozen

core=frozen.core; family=frozen.family; operator=frozen.operator; fit=frozen.fit
def require(v,m):
    if not v:raise ValueError(m)

def validate(request,request_sha,admission):
    required={'schema','evidence_scope','scientific_spec_sha256','source_scientific_spec_sha256','source',
      'implementation_sha256','fit_implementation_sha256','policy','parameters','resources'}
    require(set(request)==required and request['schema']=='r02_k03_common_finite_search_request_v1'
      and request['implementation_sha256']==frozen.implementation()
      and request['fit_implementation_sha256']==fit.implementation() and request['evidence_scope']=='DNABR_R02',
      'exact frozen search request/code required')
    require(admission=={'schema':'r02_k03_common_prepare_only_admission_v1','status':'FIT_STATISTICS_AND_GRID_ONLY_ADMITTED',
      'evidence_scope':'DNABR_R02','scientific_spec_sha256':request['scientific_spec_sha256'],
      'request_sha256':request_sha,'fit_allowed':False,'select_allowed':False,'evaluate_allowed':False},
      'exact per-request prepare-only admission required')
    policy=request['policy'];require(policy=={
      'epsilon':0.01,'subdivisions_per_decade':2,'guard_decades':1,'max_penalties':64,
      'plausibility_rule':'LOWER_BOUND_LE_MINIMUM_UPPER_BOUND','selection_rule':'RETAIN_NUMERICALLY_PLAUSIBLE_SET',
      'boundary_action':'REQUIRE_EXTENSION_NO_SELECTION'},'frozen finite-family policy differs')

def prepare(request_path,request_sha,admission_path,admission_sha,output,source_directory=None):
    os.umask(0o077);pins={};request=core.load_json(frozen._pin(request_path,request_sha,2**20,pins))
    admission=core.load_json(frozen._pin(admission_path,admission_sha,2**20,pins));validate(request,request_sha,admission)
    request_for_load=request
    if source_directory is not None:
      request_for_load=dict(request);request_for_load['source']=dict(request['source']);request_for_load['source']['directory']=str(Path(source_directory).resolve())
    _,source_manifest,persons,points,records=frozen._load_source(request_for_load,pins);roles=core.validate_roles(persons)
    i,j,_,_,weights,extra=fit.role_pairs(persons,'FIT',records['FIT']);indices=np.asarray(roles['FIT'],dtype=np.int64)
    design=operator.PairDesign(np.asarray(points)[indices],np.searchsorted(indices,i),np.searchsorted(indices,j),extra=extra,
      **{k:request['parameters'][k]for k in('rows_per_chunk','max_working_mib','backend','preprocessing_backend','workers')})
    identity=family.content_sha256({'source_manifest_sha256':request['source']['manifest_sha256'],
      'source_outputs_sha256':source_manifest['outputs_sha256'],'role':'FIT'})
    stats=family.fit_gram_statistics(design,weights,role='FIT',input_identity=identity,
      max_seconds=request['resources']['statistics_max_seconds']);stats_hash=family.content_sha256(stats);p=request['policy']
    grid=family.finite_family(stats,expected_statistics_sha256=stats_hash,epsilon=p['epsilon'],
      subdivisions_per_decade=p['subdivisions_per_decade'],guard_decades=p['guard_decades'],max_penalties=p['max_penalties'])
    require(grid['status']=='PLANNED_NOT_FITTED'and grid['penalties'],'finite grid unavailable under frozen cap')
    out=Path(output).resolve();require(not out.exists()and out.parent.is_dir()and shutil.disk_usage(out.parent).free>=request['resources']['min_free_bytes'],'new bounded output required')
    out.mkdir();core.write_json(out/'statistics.json',stats);core.write_json(out/'family.json',grid)
    summary={'schema':'r02_k03_common_prepare_only_result_v1','status':'FIT_STATISTICS_AND_GRID_COMPLETE_NO_FITS',
      'request_sha256':request_sha,'admission_sha256':admission_sha,'source_manifest_sha256':request['source']['manifest_sha256'],
      'statistics_sha256':core.sha(out/'statistics.json'),'family_sha256':core.sha(out/'family.json'),
      'n_penalties':len(grid['penalties']),'fit_allowed':False,'fit_requests_generated':False,'fit_admissions_generated':False,
      'select_allowed':False,'evaluate_allowed':False,'evaluate_targets_computed':False,'biological_validation':False}
    core.write_json(out/'manifest.json',summary)
    require(all(core.sha(path)==digest for path,digest in pins.items())and frozen.implementation()==request['implementation_sha256'],'inputs/code changed during prepare')
    return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();[p.add_argument('--'+n,required=True)for n in('request','request-sha256','admission','admission-sha256','output','source-directory')];a=p.parse_args();print(json.dumps(prepare(a.request,a.request_sha256,a.admission,a.admission_sha256,a.output,a.source_directory),sort_keys=True))
