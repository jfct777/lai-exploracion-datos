#!/usr/bin/env python3
"""Prepare a matched FLARE marker intervention from fixed parental candidates.

No truth, ancestry result, genotype imputation or candidate selection is used.
The baseline retains every other parental marker, and is not called common-only.
"""
from __future__ import annotations
import argparse,csv,gzip,json,os,resource,time
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
import pysam
import r02_lai_simulate as sim
import r02_lai_flare_plan as plan
from r02_lai_support_review import require,read,write,sha

SCHEMA='r02_lai_flare_material_input_v1'
ANCESTRY={'African':'AFR','European':'EUR','Native_American':'NAM'}
BASE='BASELINE_EXCLUDING_EVALUATED_DNABR_RARE'
FULL='BASELINE_PLUS_EVALUATED_DNABR_RARE'
INPUTS={'reference_vcf','reference_index','target_vcf','target_index','original_roles','fold_roles',
        'candidate_loci','candidate_manifest','genetic_map','flare_jar','simulation_manifest','simulation_preflight','parental_manifest'}
CODE_FILES=('r02_lai_flare_material.py',*plan.CODE_FILES)


def descriptor(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def candidates(files,chromosome):
    m=read(files['candidate_manifest'])
    require(m['schema']=='r02_lai_donor_haplotypes_v1' and m['chromosome']==chromosome and
            m['outputs_sha256']['loci.tsv.gz']==sha(files['candidate_loci']) and m['source_test_opened'] is False,
            'Authenticated original L1 candidate universe required')
    result={}
    with gzip.open(files['candidate_loci'],'rt')as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            if row['ancestry']!='ALL':continue
            key=(row['chromosome'],int(row['position_bp']),row['REF'],row['ALT'])
            require(key[0]==chromosome and key not in result and row['counted_allele']in key[2:] and
                    row['map_range_status']=='INSIDE','Invalid candidate axis or counted allele')
            result[key]=row
    require(len(result)==m['summary']['candidate_loci']>0,'Candidate denominator differs')
    return result


def role_axis(files,reference_samples,target_samples,fold,simulation):
    with files['original_roles'].open()as f:original_rows=list(csv.DictReader(f,delimiter='\t'))
    original={r['sample_id']:r for r in original_rows}
    require(len(original)==len(original_rows) and all(original),'Duplicate or empty original role identity')
    with files['fold_roles'].open()as f:folds=list(csv.DictReader(f,delimiter='\t'))
    chosen=[r for r in folds if r['role']=='SOURCE_VALID' and r['split']==fold]
    require(chosen and len({r['sample_id']for r in chosen})==len(chosen),'Empty/duplicate donor fold')
    for r in chosen:
        old=original[r['sample_id']]
        require(old['role']==r['role'] and old['ancestry']==r['ancestry'] and old['atomic_unit_id']==r['family_component'],
                'Donor role, ancestry or original atomic unit changed')
    references=[original[s]for s in reference_samples]
    require(all(r['role']=='REF_TRAIN' and r['ancestry']in ANCESTRY for r in references),'Reference role changed')
    require(set(reference_samples)=={r['sample_id']for r in original.values()if r['role']=='REF_TRAIN'},'Reference roster differs')
    require(set(reference_samples).isdisjoint(target_samples),'Reference/target identities overlap')
    ru={r['atomic_unit_id']for r in references};du={(r['family_component'],ANCESTRY[r['ancestry']])for r in chosen}
    require(ru.isdisjoint({x[0]for x in du}),'Reference and donor original units overlap')
    require(simulation['donor_split']==fold and simulation['distinct_donor_count']==len(chosen), 'Simulation donor fold differs')
    order=sorted(range(len(references)),key=lambda i:(list(ANCESTRY).index(references[i]['ancestry']),i))
    return references,chosen,sorted(du),order


def prepare(config_path,digest,out):
    started=time.monotonic();c=read(sim.descriptor(dict(path=str(Path(config_path).resolve()),sha256=digest)))
    require(c.get('schema')==SCHEMA and c.get('scope')in {'SYNTHETIC_TEST','CONDITIONAL_DEVELOPMENT'},'Explicit preparation scope required')
    require(c.get('marker_conditions')==[BASE,FULL] and c.get('candidate_policy')=='ALL_PREDECLARED_L1_CANDIDATES_NO_FOLD_RESELECTION','Preserve both exact marker conditions')
    require(c.get('site_filter')=='PASS_OR_UNFILTERED' and c.get('genome_build')in {'SYNTHETIC','GRCh38'},'Explicit site/build policy required')
    require(set(c['inputs'])==INPUTS | plan.export_inputs(c),'Exact predictor preparation inputs required; truth forbidden')
    files={k:sim.descriptor(v)for k,v in c['inputs'].items()};chrom=str(c['chromosome']);limits=c['resources']
    export=plan.validate_export(c,files)
    require(sum(p.stat().st_size for p in files.values())<=limits['max_input_bytes'],'Input byte bound exceeded')
    frequency_policy=c.get('reference_frequency_policy','RETAIN_ALL_REFERENCE_FREQUENCY_STRATA')
    expected_frequency={'RETAIN_ALL_REFERENCE_FREQUENCY_STRATA':(0,0),'FLARE_DOCUMENTED_SEQUENCE_DEFAULTS':(.005,50)}
    require(frequency_policy in expected_frequency and
            (c['parameters']['min-maf'],c['parameters']['min-mac'])==expected_frequency[frequency_policy] and c['parameters']['array'] is False,
            'Explicit matched reference frequency policy required; no silent frequency filter')
    source=read(files['simulation_manifest']);parental=read(files['parental_manifest'])
    expected='COMPLETE_SYNTHETIC_SOFTWARE_TEST'if c['scope']=='SYNTHETIC_TEST'else'COMPLETE_SIMULATED_SCENARIO_NOT_LAI_VALIDATION'
    require(source['schema']=='r02_lai_simulate_output_v1' and source['status']==expected and source['chromosomes']==[chrom] and
            source['genome_build']==c['genome_build'] and source['outputs_sha256']['genotypes.vcf.gz']==sha(files['target_vcf']),
            'Target is not the original completed simulation predictor')
    require(source['outputs_sha256'].get('prepared/admission.json')==sha(files['simulation_preflight']),
            'Simulation donor preflight is not bound to the completed simulation')
    checked=read(files['simulation_preflight'])['input_contract']
    require(checked['inputs']['roles']['sha256']==sha(files['fold_roles']) and
            checked['preparation_provenance']['original_roles']['sha256']==sha(files['original_roles']) and
            checked['donor_split']==c['donor_split'], 'Simulation original roles or donor fold changed')
    require(parental['schema']=='r02_lai_real_material_v1' and parental['chromosome']==chrom and parental['genome_build']==c['genome_build'] and
            parental['outputs_sha256']['ref_train.chr'+chrom+'.vcf.gz']==sha(files['reference_vcf']),'Parent reference provenance differs')
    rare=candidates(files,chrom)
    ref_it,tar_it=sim.vcf_rows(files['reference_vcf']),sim.vcf_rows(files['target_vcf'])
    references,sentinel=next(ref_it);targets,sentinel2=next(tar_it)
    require(sentinel is sentinel2 is None and references and targets and len(set(references))==len(references) and len(set(targets))==len(targets),'Invalid sample axes')
    refs,donors,units,order=role_axis(files,references,targets,c['donor_split'],source)
    out=Path(out).resolve();require(not out.exists() and out.parent.is_dir(),'Fresh output required')
    sim.local_filesystem(out.parent);require(sim.shutil.disk_usage(out.parent).free>=limits['disk_reserve_gib']*2**30,'Disk reserve violated')
    out.mkdir(mode=0o700)
    counts={BASE:0,FULL:0};matched=set();last=0;refzero=0;retained_candidates=0;filtered=0;source_loci=0
    threshold=plan.effective_min_mac(c['parameters'],len(references))
    with ExitStack()as stack:
        writers={};inventories={}
        for arm in (BASE,FULL):
            directory=out/arm;directory.mkdir()
            writers[arm]={}
            for label,ids in [('reference',[references[i]for i in order]),('target',targets)]:
                writer=stack.enter_context(pysam.BGZFile(str(directory/(label+'.vcf.gz')),'wb'));writers[arm][label]=writer
                header=f'##fileformat=VCFv4.2\n##contig=<ID={chrom}>\n##FORMAT=<ID=GT,Number=1,Type=String,Description="Original phased genotype">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'+'\t'.join(ids)+'\n';writer.write(header.encode())
            stream=stack.enter_context((directory/'marker_inventory.tsv').open('x'));inventories[arm]=csv.writer(stream,delimiter='\t',lineterminator='\n')
            inventories[arm].writerow(('chromosome','position','ref','alt','channel','frequency_definition'))
        ledger=csv.writer(stack.enter_context((out/'candidate_inventory.tsv').open('x')),delimiter='\t',lineterminator='\n')
        ledger.writerow(('chromosome','position','ref','alt','counted_allele','reference_counted_copies','source_candidate_fixed_before_simulation','reference_minor_count','retained_after_reference_filter'))
        for a,b in zip_longest(ref_it,tar_it):
            require(a is not None and b is not None,'Reference/target marker count differs')
            rs,row=a;ts,other=b;require(row is not None and other is not None and rs==references and ts==targets,'Unexpected source header')
            key=(row[0],int(row[1]),row[3],row[4]);require(key==(other[0],int(other[1]),other[3],other[4]) and key[0]==chrom and key[1]>last,'Marker/allele/order differs')
            require(row[3]in 'ACGT' and row[4]in 'ACGT' and len(row[3])==len(row[4])==1 and row[3]!=row[4],'Biallelic SNV required')
            require(row[6]in {'.','PASS'} and other[6]in {'.','PASS'},'Recorded site filter failure')
            calls=sim.genotypes(row);sim.genotypes(other)
            source_loci+=1;alt_count=sum(sum(call)for call in calls);mac=min(alt_count,2*len(references)-alt_count);retained=mac>=threshold
            require(other[7]=='.' and other[8]=='GT','Target has truth or other predictor annotations')
            israre=key in rare
            if israre:
                candidate=rare[key];selected=key[2:].index(candidate['counted_allele']);ac=sum(x==selected for call in calls for x in call);refzero+=ac==0
                ledger.writerow((*key,candidate['counted_allele'],ac,'true',mac,str(retained).lower()));matched.add(key);retained_candidates+=retained
            last=key[1]
            require(source_loci<=limits['max_markers'],'Source marker bound exceeded')
            if source_loci%1024==0:sim.resource_guard(out,dict(timeout_seconds=limits['timeout_seconds'],disk_reserve_gib=limits['disk_reserve_gib'],max_output_gib=limits['max_output_bytes']/2**30),started)
            if not retained:filtered+=1;continue
            refrow=[*row[:7],'.','GT',*(row[9:][i]for i in order)]
            require(row[8]=='GT','Reference FORMAT must be GT-only upstream material')
            for arm in ((FULL,)if israre else(BASE,FULL)):
                writers[arm]['reference'].write(('\t'.join(refrow)+'\n').encode());writers[arm]['target'].write(('\t'.join(other)+'\n').encode())
                inventories[arm].writerow((*key,'EVALUATED_DNABR_RARE'if israre else'BASELINE_OTHER',
                    'Fixed L1 DNABR source2723 minor catalogue and observed SOURCE_VALID candidate; no fold reselection'if israre else'All other eligible parental markers; frequency not asserted common'))
                counts[arm]+=1
            last=key[1]
            require(counts[FULL]<=limits['max_markers'],'Marker bound exceeded')
            if counts[FULL]%1024==0:sim.resource_guard(out,dict(timeout_seconds=limits['timeout_seconds'],disk_reserve_gib=limits['disk_reserve_gib'],max_output_gib=limits['max_output_bytes']/2**30),started)
    require(matched==set(rare) and counts[BASE]>=2 and counts[FULL]-counts[BASE]==retained_candidates,'Incomplete exact candidate intervention or fewer than two baseline positions')
    generations=c.get('initial_generations',[c['parameters']['gen']])
    require(generations and len(set(generations))==len(generations) and c['parameters']['gen']in generations and
            all(type(g)in(int,float)and g>=1 for g in generations),'Invalid predeclared generation initializations')
    if len(generations)>1:require(c['parameters']['em'] and generations==[5,10,16],'This comparison predeclares only EM initialization5/10/16')
    plans=[];all_plans=[];base_plans=[]
    for arm in(BASE,FULL):
        directory=out/arm
        for label in('reference','target'):pysam.tabix_index(str(directory/(label+'.vcf.gz')),preset='vcf',csi=True)
        with(directory/'sample_map.tsv').open('x')as f:
            for i in order:f.write(references[i]+'\t'+ANCESTRY[refs[i]['ancestry']]+'\n')
        with(directory/'roles.tsv').open('x')as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(('sample_id','role','atomic_unit_id','ancestry'))
            for i in order:w.writerow((references[i],'REF_TRAIN',refs[i]['atomic_unit_id'],ANCESTRY[refs[i]['ancestry']]))
            for sample in targets:
                for unit,anc in units:w.writerow((sample,'SIMULATED_TARGET',unit,anc))
        inp={k:descriptor(directory/n)for k,n in [('reference_vcf','reference.vcf.gz'),('reference_index','reference.vcf.gz.csi'),('target_vcf','target.vcf.gz'),('target_index','target.vcf.gz.csi'),('sample_map','sample_map.tsv'),('roles','roles.tsv'),('marker_inventory','marker_inventory.tsv')]}
        inp.update({k:descriptor(files[k])for k in('genetic_map','flare_jar','simulation_manifest')})
        inp.update({k:descriptor(files[k])for k in plan.export_inputs(c)})
        provenance=dict(schema='r02_lai_flare_material_v1',genome_build=c['genome_build'],inputs_sha256={k:v['sha256']for k,v in inp.items()},source_test_genotypes_used=False,original_donor_units_preserved=True,simulation_manifest_sha256=inp['simulation_manifest']['sha256'],source_predictors_sha256=source['outputs_sha256']['genotypes.vcf.gz'],parental_manifest_sha256=sha(files['parental_manifest']),candidate_manifest_sha256=sha(files['candidate_manifest']),candidate_loci_sha256=sha(files['candidate_loci']),original_roles_sha256=sha(files['original_roles']),fold_roles_sha256=sha(files['fold_roles']))
        write(directory/'material_provenance.json',provenance);inp['material_provenance']=descriptor(directory/'material_provenance.json')
        pc=dict(schema=plan.SCHEMA,scope=c['scope'],method='DIRECT_MACROANCESTRY',genome_build=c['genome_build'],chromosome=chrom,ancestry_names=['AFR','EUR','NAM'],marker_condition=arm,site_filter=c['site_filter'],target_population_assumption=c['target_population_assumption'],parameters=c['parameters'],generation_policy=c['generation_policy'],inputs=inp,flare_version=c['flare_version'],resources=c['inference_resources'])
        pc['probability_export']=export['label']
        pc['evaluated_candidates_retained']=retained_candidates if arm==FULL else 0
        for gen in generations:
            current={**pc,'parameters':{**pc['parameters'],'gen':gen}}
            if len(generations)>1:current['initialization_label']='gen'+str(gen)
            name='plan-input.json'if gen==c['parameters']['gen']else'plan-input.gen'+str(gen)+'.json'
            plan.validate(current);write(directory/name,current);record=descriptor(directory/name);all_plans.append(record)
            if arm==BASE:base_plans.append(record)
            if gen==c['parameters']['gen']:plans.append(record)
    for value in c['inputs'].values():sim.descriptor(value)
    result=dict(schema='r02_lai_flare_material_pair_v1',status='PAIRED_PREDICTOR_INPUTS_PREPARED_NO_INFERENCE',chromosome=chrom,scope=c['scope'],marker_counts=counts,candidate_loci=len(rare),candidate_reference_zero=refzero,reference_people=len(references),simulated_people=len(targets),original_donor_people=len(donors),truth_accessed=False,source_test_opened=False,reference_columns_reordered=True,genotypes_modified=False,alleles_modified=False,candidate_reselection=False,common_only_claim=False,plans=plans,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,inputs_sha256={k:v['sha256']for k,v in c['inputs'].items()},code_sha256={n:sha(Path(__file__).with_name(n))for n in CODE_FILES},outputs_sha256={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})
    result.update(reference_frequency_policy=frequency_policy,effective_min_mac_reference_only=threshold,source_loci=source_loci,filtered_loci=filtered,retained_candidates=retained_candidates)
    identical=all(sha(out/BASE/n)==sha(out/FULL/n)for n in ('reference.vcf.gz','target.vcf.gz','marker_inventory.tsv','roles.tsv','sample_map.tsv'))
    require(identical==(retained_candidates==0),'Zero-candidate comparator identity disagrees with byte equivalence')
    result['inference_plans']=base_plans if identical else all_plans
    result['initial_generations']=generations
    result['plan_aliases']={FULL:BASE}if identical else {}
    result['identical_predictors_adjusted_once']=identical
    write(out/'manifest.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();os.umask(0o077);prepare(a.contract,a.contract_sha256,a.output_dir)
if __name__=='__main__':main()
