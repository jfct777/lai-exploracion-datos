#!/usr/bin/env python3
"""Compare frozen C, C+Q and C+Q+R predictions against common target intervals.

No model fitting, endpoint refitting, target midpoint, or sampling inference.
The caller must authenticate a completed model/selection receipt. Recording its
hash binds the claim; it does not itself prove the receipt's scientific validity.
"""
import argparse
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
import r02_k04_identified_comparison as maths
import r02_k04_common_target as target

require=target.require


def evaluate(lower, upper, opportunities, predictions, indices):
    """Equal eligible unordered pairs; same exact support for all three arms."""
    ix=np.asarray(indices,dtype=int)
    take=np.ix_(ix,ix)
    den=opportunities[take]
    keep=np.triu(den>0,1)
    if not np.any(keep):
        return dict(status='NOT_ESTIMABLE',n_pairs=0)
    lo=lower[take][keep]/den[keep];hi=upper[take][keep]/den[keep]
    require(np.all((0<=lo)&(lo<=hi)&(hi<=1)), 'Outcome interval outside domain')
    arms={k:v[take][keep] for k,v in predictions.items()}
    require(set(arms)=={'C','CQ','CQR'} and
            all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in arms.values()),
            'Finite frozen predictions on identical supported pairs required')
    result={}
    for baseline,candidate in (('C','CQ'),('CQ','CQR'),('C','CQR')):
        bounds={}
        for weight,weights in (('equal_pairs',np.ones(len(lo))),('opportunity_weighted_sensitivity',den[keep])):
            l,u=maths.fixed_squared_error_improvement(lo,hi,arms[baseline],arms[candidate],weights)
            bounds[weight]=dict(lower=l,upper=u,interpretation=(
                'UNIFORM_CONDITIONAL_IMPROVEMENT' if l>0 else
                'UNIFORM_CONDITIONAL_DETERIORATION' if u<0 else 'SIGN_UNRESOLVED'))
        result[baseline+'_minus_'+candidate]=bounds
    return dict(status='FIXED_PREDICTION_OUTER_COMPARISON',n_pairs=len(lo),contrasts=result)


def run(request,digest,output):
    require(maths.sha256(request)==digest,'Evaluation request hash differs')
    r=json.loads(Path(request).read_text())
    require(r['schema']=='r02_k04_fixed_evaluate_input_v1' and r['role']=='EVALUATE',
            'Explicit frozen EVALUATE request required')
    freeze=Path(r['model_freeze']['path'])
    require(maths.sha256(freeze)==r['model_freeze']['sha256'],'Model freeze receipt differs')
    model=json.loads(freeze.read_text())
    require(model['status']=='COMPLETE_FIT_SELECT_FROZEN' and model['evaluate_outcomes_used'] is False
            and model['arms']==['C','CQ','CQR'] and model['target_chromosome_excluded_from_all_features'] is True,
            'Fit/selection freeze or common controls absent')
    result=[]
    require(len(r['chromosomes'])==22 and {str(x['chromosome']) for x in r['chromosomes']}=={str(i) for i in range(1,23)},
            'Complete unique 22-autosome evaluation required')
    for row in r['chromosomes']:
        d=Path(row['target_directory']);m=json.loads((d/'manifest.json').read_text())
        require(maths.sha256(d/'manifest.json')==row['target_manifest_sha256'] and m['schema']==target.OUTPUT
                and str(m['chromosome'])==str(row['chromosome']),'Target manifest mismatch')
        for name in ('counts.npz','persons.tsv'):
            require(maths.sha256(d/name)==m['outputs_sha256'][name], 'Target bytes differ')
        with (d/'persons.tsv').open() as stream: persons=list(csv.DictReader(stream,delimiter='\t'))
        require(dict(__import__('collections').Counter(p['role'] for p in persons))=={'FIT':1460,'SELECT':476,'EVALUATE':683},
                'Human role counts differ')
        indices=[i for i,p in enumerate(persons) if p['role']=='EVALUATE']
        prediction=Path(row['predictions']['path'])
        require(maths.sha256(prediction)==row['predictions']['sha256'] and
                model['prediction_sha256'][str(row['chromosome'])]==row['predictions']['sha256'] and
                model['persons_sha256']==m['outputs_sha256']['persons.tsv'], 'Prediction/person freeze binding differs')
        for path in (prediction,d/'counts.npz'):
            with zipfile.ZipFile(path) as z:
                require(sum(v.file_size for v in z.infolist()) <= r['max_array_bytes'], 'Array admission exceeded')
        with np.load(d/'counts.npz',allow_pickle=False) as data,np.load(prediction,allow_pickle=False) as pred:
            per={}
            for a in maths.ANCESTRIES:
                arrays={arm:pred[arm+'_'+a] for arm in ('C','CQ','CQR')}
                require(all(v.shape==(len(persons),len(persons)) and np.array_equal(v,v.T,equal_nan=True)
                            for v in arrays.values()),'Prediction shape/symmetry differs')
                per[a]=evaluate(data['lower_count_'+a],data['upper_count_'+a],data['opportunities_'+a],arrays,indices)
            result.append(dict(chromosome=str(row['chromosome']),by_ancestry=per))
    # Equal supported chromosomes, never pooled pairs mistaken for replicates.
    aggregate={}
    for a in maths.ANCESTRIES:
        supported=[c['by_ancestry'][a] for c in result if c['by_ancestry'][a]['n_pairs']]
        aggregate[a]=dict(n_supported_chromosomes=len(supported),contrasts={})
        if supported:
            for contrast in supported[0]['contrasts']:
                aggregate[a]['contrasts'][contrast]={k:float(np.mean([v['contrasts'][contrast]['equal_pairs'][k] for v in supported]))
                                                     for k in ('lower','upper')}
    target.common.material.write_json(output,dict(schema='r02_k04_fixed_evaluation_v1',by_chromosome=result,
        aggregate=aggregate,request_sha256=digest,model_freeze_sha256=r['model_freeze']['sha256'],
        sampling_uncertainty=False,phase_imputed=False,ancestry_truth_calibrated=False,
        interpretation='FINITE_COHORT_OUTER_BOUNDS_NOT_CONFIDENCE_INTERVAL'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.request,a.request_sha256,a.output)
