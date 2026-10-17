#!/usr/bin/env python3
"""Select the best authenticated compatible K04 checkpoint, never by recency."""
from __future__ import annotations
import argparse, ast, hashlib, json, math, re, struct, zipfile
from pathlib import Path

SCHEMA="r02_k04_incumbent_catalog_v1"
HEX=re.compile(r"[0-9a-f]{64}")

def require(value,message):
    if not value: raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def strict_json(path):
    def unique(pairs):
        out={}
        for k,v in pairs: require(k not in out,'duplicate JSON key: '+k); out[k]=v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def descriptor(item):
    require(set(item)=={'path','sha256','bytes'} and HEX.fullmatch(item['sha256'] or ''),'invalid descriptor')
    p=Path(item['path']); require(p.is_file() and p.stat().st_size==item['bytes'],'missing/size mismatch: '+str(p))
    require(sha(p)==item['sha256'],'SHA256 mismatch: '+str(p)); return p
def npz_member_sha(path,name):
    with zipfile.ZipFile(path) as z:
        require(name in z.namelist(),'NPZ member absent: '+name)
        return hashlib.sha256(z.read(name)).hexdigest()
def npz_metadata(path):
    with zipfile.ZipFile(path) as z: raw=z.read('metadata_utf8.npy')
    require(raw[:6]==b'\x93NUMPY','invalid metadata NPY')
    major,minor=raw[6],raw[7]
    if major==1: length=struct.unpack('<H',raw[8:10])[0]; start=10
    elif major in (2,3): length=struct.unpack('<I',raw[8:12])[0]; start=12
    else: raise ValueError('unsupported NPY version')
    header=ast.literal_eval(raw[start:start+length].decode('latin1').strip())
    require(header['descr']=='|u1' and header['fortran_order'] is False and len(header['shape'])==1,'metadata NPY type differs')
    payload=raw[start+length:]; require(len(payload)==header['shape'][0],'metadata NPY length differs')
    return json.loads(payload.decode())
def diagnostic_ok(d):
    return (isinstance(d,dict) and d.get('dual_certificate_eligible') is True and
        d.get('feasible_dual_sum_residual')==0.0 and d.get('primal_dual_gap',-1)>=0 and
        d.get('scope')=='FP64_CONDITIONAL_PROPAGATED_NOT_EXACT' and
        d.get('exact_feasibility_repair',{}).get('exact_real_sum_zero') is True and
        math.isfinite(d.get('feasible_dual_objective',float('nan'))) and
        math.isfinite(d.get('primal_objective',float('nan'))))

def authenticate_entry(entry,requirements):
    reasons=[]
    if entry.get('compatibility')!=requirements['compatibility']: reasons.append('INCOMPATIBLE_OBJECTIVE_DATA_OR_PARAMETERS')
    try:
        authenticated={name:descriptor(value) for name,value in entry['evidence'].items()}
        receipt_path=authenticated['receipt']; receipt=strict_json(receipt_path)
    except Exception as error: return None,['EVIDENCE_RECEIPT_'+str(error)]
    if entry.get('disposition')!='CERTIFIED_CHECKPOINT': return None,[entry.get('catalog_rejection','NOT_A_CERTIFIED_CHECKPOINT')]
    try:
        checkpoint=descriptor(entry['checkpoint']); metadata=npz_metadata(checkpoint)
        point=entry['bindings']['certified_point']; require(npz_member_sha(checkpoint,point['member'])==point['sha256'],'certified point hash mismatch')
        primal=entry['bindings']['primal']; primal_path=descriptor(primal['checkpoint'])
        observed={name:npz_member_sha(primal_path,member) for name,member in primal['members'].items()}
        require(observed==primal['member_sha256'],'primal member hash mismatch')
        require(canonical(observed)==requirements['primal_binding_sha256'],'primal binding incompatible')
    except Exception as error: return None,reasons+['CHECKPOINT_OR_BINDING_'+str(error)]
    rd=receipt.get('diagnostic',{}); md=metadata.get('diagnostic',{})
    if not diagnostic_ok(rd) or not diagnostic_ok(md): reasons.append('NUMERICAL_CERTIFICATE_INVALID')
    for key in ('feasible_dual_objective','primal_objective','primal_dual_gap'):
        if rd.get(key)!=md.get(key) or rd.get(key)!=entry['certificate'].get(key): reasons.append('CERTIFICATE_VALUE_MISMATCH_'+key)
    if receipt.get('candidate_promoted_as_incumbent') is False: reasons.append('RECEIPT_REJECTED_CANDIDATE')
    if receipt.get('scientific_model_admitted') is not False: reasons.append('SCIENTIFIC_SCOPE_CHANGED')
    gradient=entry['bindings'].get('gradient')
    gradient_reuse=False
    if gradient:
        try:
            gradient_reuse=(npz_member_sha(checkpoint,gradient['member'])==gradient['sha256'] and
                gradient['point_sha256']==point['sha256'])
        except Exception: gradient_reuse=False
    if reasons: return None,reasons
    return {'id':entry['id'],'checkpoint':entry['checkpoint'],'checkpoint_metadata_schema':metadata.get('schema'),
        'feasible_dual_objective':rd['feasible_dual_objective'],'primal_objective':rd['primal_objective'],
        'primal_dual_gap':rd['primal_dual_gap'],'certified_point_sha256':point['sha256'],
        'primal_binding_sha256':canonical(observed),'gradient_reuse_allowed':gradient_reuse,
        'gradient_reuse_reason':('EXACT_POINT_BINDING' if gradient_reuse else 'ABSENT_OR_BOUND_TO_DIFFERENT_POINT')},[]

def recurrence_decision(selected,recurrence,requested_rhs_sha256):
    if not recurrence: return {'allowed':False,'reasons':['NO_RECURRENCE_SUPPLIED']}
    reasons=[]
    if recurrence.get('point_sha256')!=selected['certified_point_sha256']: reasons.append('POINT_CHANGED')
    if recurrence.get('rhs_sha256')!=requested_rhs_sha256: reasons.append('RHS_CHANGED_OR_UNBOUND')
    if recurrence.get('gradient_point_sha256')!=selected['certified_point_sha256']: reasons.append('GRADIENT_POINT_CHANGED')
    return {'allowed':not reasons,'reasons':reasons or ['EXACT_POINT_RHS_AND_GRADIENT_BINDING']}

def select(catalog,requirements):
    require(catalog.get('schema')==SCHEMA and isinstance(catalog.get('entries'),list),'catalog schema')
    require(set(requirements)=={'compatibility','primal_binding_sha256','requested_rhs_sha256'},'requirements fields')
    accepted=[]; rejected=[]
    for entry in catalog['entries']:
        value,reasons=authenticate_entry(entry,requirements)
        (accepted if value else rejected).append(value if value else {'id':entry.get('id'),'reasons':reasons,'preserved':True})
    require(accepted,'no authenticated compatible certified checkpoint')
    accepted.sort(key=lambda x:(x['feasible_dual_objective'],x['id']),reverse=True)
    selected=accepted[0]
    recurrence=[x for x in catalog['entries'] if x.get('recurrence')]
    selected['recurrence_reuse']=recurrence_decision(selected,recurrence[0]['recurrence'] if recurrence else None,requirements['requested_rhs_sha256'])
    return {'schema':'r02_k04_incumbent_selection_v1','status':'PASS_BEST_COMPATIBLE_CERTIFIED_CHECKPOINT',
        'selected':selected,'accepted_ranked':accepted,'rejected':rejected,
        'selection_rule':'maximum feasible_dual_objective after exact compatibility, evidence, feasibility and primal-binding checks; never time or filename',
        'next_solver_integrated':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--catalog-sha256',required=True)
    p.add_argument('--requirements',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(HEX.fullmatch(a.catalog_sha256 or '') and sha(a.catalog)==a.catalog_sha256,'catalog hash mismatch')
    result=select(strict_json(a.catalog),strict_json(a.requirements))
    require(not a.output.exists(),'refuse overwrite');a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'selected':result['selected']['id'],'dual_objective':result['selected']['feasible_dual_objective']},sort_keys=True))
if __name__=='__main__': main()
