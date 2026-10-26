#!/usr/bin/env python3
"""Check existing LAI event counts and original donor units against references."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from r02_lai_role_units import role_units, require
from r02_lai_event_annotations import ANCESTRIES, counted_is_alt, count_pair

SCHEMA = 'r02_lai_reference_linkage_request_v1'
INVENTORY_SCHEMA = 'r02_lai_reference_linkage_inventory_v1'
ANCESTRY_LABELS = dict(zip(('AFR', 'EUR', 'NAM'), ANCESTRIES)) | {a:a for a in ANCESTRIES}

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''): digest.update(chunk)
    return digest.hexdigest()

def open_text(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()

def header(path):
    with open_text(path) as stream:
        for line in stream:
            if line.startswith('#CHROM\t'):
                samples=line.rstrip('\n').split('\t')[9:]
                require(samples and len(set(samples))==len(samples),'Missing or duplicate VCF samples')
                return samples
    raise ValueError('VCF sample header absent')

def event_rows(path, chromosome):
    with open_text(path) as stream: rows=list(csv.DictReader(stream,delimiter='\t'))
    keys=[]
    for row in rows:
        require(row['chromosome']==str(chromosome),'Catalogue chromosome differs')
        key=(row['chromosome'],int(row['position_bp']),row['ref'],row['alt'])
        counted_is_alt(row['ref'],row['alt'],row['counted_allele'])
        for ancestry in ANCESTRIES:
            count_pair(row[f'reference_{ancestry}_ac'],row[f'reference_{ancestry}_an'],ancestry)
        keys.append(key)
    require(rows and len(set(keys))==len(keys),'Empty or duplicate event catalogue')
    return dict(zip(keys,rows))

def reference_counts(path, wanted, ancestry_by_person):
    samples=header(path)
    require(set(samples)==set(ancestry_by_person),'Reference ancestry/person axes differ')
    ancestry_index=[ANCESTRIES.index(ancestry_by_person[s]) for s in samples]
    found={}
    with open_text(path) as stream:
        for line in stream:
            if line.startswith('#'):continue
            parts=line.rstrip('\n').split('\t')
            key=(parts[0],int(parts[1]),parts[3],parts[4])
            if key not in wanted:continue
            require(key not in found,'Duplicate reference locus')
            require(len(parts)==9+len(samples),'Reference genotype axis differs')
            fmt=parts[8].split(':');require('GT' in fmt,'GT absent at event locus')
            gt_index=fmt.index('GT');counts=[[0,0] for _ in ANCESTRIES]
            for cell, ancestry in zip(parts[9:],ancestry_index):
                values=cell.split(':');require(gt_index<len(values),'GT field truncated')
                gt=values[gt_index].replace('|','/').split('/')
                # Partial or wholly missing diploid GT contributes neither copy.
                if '.' in gt:continue
                require(len(gt)==2 and set(gt)<={'0','1'},'Non-diploid/biallelic reference GT')
                counts[ancestry][0]+=sum(int(a) for a in gt);counts[ancestry][1]+=2
            found[key]=counts
    return found

def compare(rows, found):
    problems=[];known=unknown=0
    for key,row in rows.items():
        if key not in found:
            problems.append({'locus':list(key),'reason':'LOCUS_ABSENT_FROM_REFERENCE'});continue
        is_alt=counted_is_alt(row['ref'],row['alt'],row['counted_allele'])
        for i,ancestry in enumerate(ANCESTRIES):
            ac,an,present=count_pair(row[f'reference_{ancestry}_ac'],row[f'reference_{ancestry}_an'],ancestry)
            if not present:unknown+=1;continue
            known+=1;alt_ac,called_an=found[key][i];observed_ac=alt_ac if is_alt else called_an-alt_ac
            if (ac,an)!=(observed_ac,called_an):
                problems.append({'locus':list(key),'ancestry':ancestry,'reason':'COUNT_MISMATCH',
                    'catalogue_ac_an':[ac,an],'reference_ac_an':[observed_ac,called_an]})
    return dict(known_ancestry_counts=known,unknown_ancestry_counts=unknown,problems=problems)

def admitted_cases(path):
    inventory=json.loads(Path(path).read_text())
    require(set(inventory)=={'schema','cases'} and inventory.get('schema')==INVENTORY_SCHEMA,
            'Admitted inventory schema differs')
    require(isinstance(inventory['cases'],list),'Admitted inventory cases absent')
    cases=[];ids=set()
    for case in inventory['cases']:
        require(isinstance(case,dict) and set(case)=={'task_id','chromosome','role'},
                'Admitted case fields differ')
        task_id=case['task_id'];chromosome=case['chromosome'];role=case['role']
        require(isinstance(task_id,str) and isinstance(chromosome,str) and
                task_id and task_id not in ids,
                'Duplicate admitted task or non-string chromosome')
        require(chromosome in {'1','22'} and role in {'FIT','SELECT'},'Admitted case axis differs')
        ids.add(task_id);cases.append((task_id,chromosome,role))
    require(len(cases)==12 and Counter(chrom for _,chrom,_ in cases)=={'1':6,'22':6},
            'Admitted inventory must contain twelve cases across chromosomes 1 and 22')
    for chromosome in ('1','22'):
        require(Counter(role for _,chrom,role in cases if chrom==chromosome)=={'FIT':3,'SELECT':3},
                'Admitted chromosome must contain three FIT and three SELECT cases')
    return set(cases)

def run(request_path, expected_sha, output):
    start=time.monotonic();require(sha(request_path)==expected_sha,'Request changed')
    request=json.loads(Path(request_path).read_text())
    require(request.get('schema')==SCHEMA and request.get('operation')=='CHECK_FIXED_CATALOGUE_REFERENCE_AND_UNITS', 'Wrong operation')
    require(request.get('direction')=='FOLD_A_TO_FOLD_B' and request.get('training_authorized') is False,'Development preparation only')
    verified={};stats={}
    def source(desc):
        require(set(desc)=={'path','sha256'},'Exact source descriptor required')
        p=Path(desc['path']);pin=desc['sha256']
        if str(p) not in verified:
            require(sha(p)==pin,'Source changed: '+str(p));verified[str(p)]=pin
            st=p.stat();stats[str(p)]=(st.st_size,st.st_mtime_ns,st.st_ino)
        require(verified[str(p)]==pin,'Conflicting source identities');return p
    chromosome=str(request.get('chromosome'))
    require(chromosome in {'1','22'},'Request chromosome differs')
    inventory=admitted_cases(source(request['admitted_inventory']))
    requested_cases=request.get('cases')
    require(isinstance(requested_cases,list),'Request cases absent')
    requested={(case.get('task_id'),chromosome,case.get('role')) for case in requested_cases
               if isinstance(case,dict)}
    expected={case for case in inventory if case[1]==chromosome}
    require(len(requested_cases)==6 and len(requested)==6 and requested==expected,
            'Request cases differ from admitted chromosome inventory')
    ref=source(request['reference_vcf']);reference_people=header(ref)
    union={};cases=[];ancestry=None;ids=set()
    for case in requested_cases:
        require(case['task_id'] not in ids,'Duplicate case');ids.add(case['task_id'])
        require(case['role'] in ('FIT','SELECT'),'EVALUATE closed')
        upstream=json.loads(source(case['case_receipt']).read_text())
        require(upstream.get('schema')=='r02_lai_expanded_alt_ref_case_v1' and
                upstream['task_id']==case['task_id'] and upstream['role']==case['role'] and
                str(upstream['chromosome'])==chromosome, 'Case identity differs')
        require(upstream.get('status')=='PREPARED_NO_INFERENCE' and
                upstream.get('truth_read') is False and upstream.get('evaluate_opened') is False and
                upstream.get('inference_executed') is False and upstream.get('training_executed') is False,
                'DA case is not preparation-only')
        require(upstream['input_descriptors']['reference_vcf']==request['reference_vcf'] and
                upstream['input_descriptors']['target_vcf']==case['target_vcf'], 'Genotype lineage differs')
        require(upstream['outputs_sha256']['event.alt_ref.tsv.gz']==case['event_catalogue']['sha256'] and
                upstream['outputs_sha256']['BASE/roles.tsv']==case['roles']['sha256'], 'Catalogue/role lineage differs')
        provenance=upstream['catalogue_sources']['metadata']
        source(provenance['ref_train_receipt']);source(provenance['ref_train_support'])
        catalog=source(case['event_catalogue']);rows=event_rows(catalog,chromosome)
        target_people=header(source(case['target_vcf']))
        require(set(reference_people).isdisjoint(target_people),'Reference and target sample overlap')
        by_person,units=role_units(source(case['roles']),set(reference_people),set(target_people),set(ANCESTRY_LABELS),False)
        by_person={person:ANCESTRY_LABELS[label] for person,label in by_person.items()}
        require(ancestry is None or ancestry==by_person,'Reference ancestries differ between cases')
        ancestry=by_person
        for key,row in rows.items():
            if key in union:require(counted_is_alt(key[2],key[3],row['counted_allele'])==counted_is_alt(key[2],key[3],union[key]['counted_allele']),'Conflicting counted allele')
            union[key]=row
        cases.append((case,rows,units,len(target_people),provenance))
    require(cases,'Empty case set')
    found=reference_counts(ref,union,ancestry);results=[]
    for case,rows,units,n_target,provenance in cases:
        comparison=compare(rows,found)
        results.append(dict(task_id=case['task_id'],role=case['role'],event_loci=len(rows),reference_people=len(reference_people),target_people=n_target,
            original_units=units,reference_target_person_overlap=0,reference_target_original_unit_overlap=0,
            reference_join_key=['chromosome','position_bp','REF','ALT'],reference_duplicate_keys=0,
            event_catalogue_sha256=case['event_catalogue']['sha256'],reference_vcf_sha256=request['reference_vcf']['sha256'],
            ref_train_receipt_sha256=provenance['ref_train_receipt']['sha256'],
            ref_support_by_provenance_sha256=provenance['ref_train_support']['sha256'],
            reference_roles_sha256=case['roles']['sha256'],**comparison))
    for filename,old in stats.items():
        st=Path(filename).stat();require(old==(st.st_size,st.st_mtime_ns,st.st_ino),'Input mutated during execution')
    result=dict(schema='r02_lai_reference_linkage_receipt_v1',request_sha256=expected_sha,
        status='COUNTS_AND_ORIGINAL_UNITS_MATCH' if all(not r['problems'] for r in results) else 'BLOCKED_COUNTS_OR_LOCI_DIFFER',
        cases=results,inputs_sha256=verified,elapsed_seconds=time.monotonic()-start,
        training_authorized=False,controls_materialized=False,truth_loaded=False,evaluate_opened=False,
        limitation='Identity/unit separation does not prove unrelatedness or independent final donors; unknown counts remain unknown.')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);final=out/'receipt.json'
    require(not final.exists(),'Output already exists; resume completed Nextflow task')
    temporary=out/'receipt.tmp'
    with temporary.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(temporary,final);temporary.unlink()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps({'status':run(args.request,args.request_sha256,args.output)['status']}))
