#!/usr/bin/env python3
"""Describe ancestry composition of reported IBD haplotype-edge territory.

Existing IBD × Gnomix annotations are reused, not inferred again. Physical
unions preserve both people and homologues. The denominator is reported
haplotype-edge territory, never ancestral opportunity across the genome,
diploid IBD1/2, or phased rare-allele frequency. All detailed outputs are private.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from r02_community_levels import PHIS, require, sha256
from r02_community_ibd import Guard, Ledger, load_people, save_json, strict_json, table, write_table

SCHEMA = "r02_community_ibd_ancestry_inputs_v1"
PART_SCHEMA = "r02_community_ibd_ancestry_chromosome_v1"
STATUS = "COMPLETE_REPORTED_IBD_ANCESTRY_COMPOSITION_NOT_RARE_PHASE_VALIDATION"
STATES = ("CONCORDANT_AFR", "CONCORDANT_EUR", "CONCORDANT_NAM", "DISCORDANT", "UNKNOWN")
FIELDS = tuple(s.lower()+"_bp" for s in STATES) + ("both_posteriors_available_bp", "hap_pair_union_bp_sum")
LABELS = {"African":0, "European":1, "Native-American":2, "NA":-1}
POSTERIOR_STATES = {"MISSING_FB":0, "BOTH_AVAILABLE":1, "A_ONLY":2, "B_ONLY":3}
NA = "NA"
LIMITATIONS = [
    "Ancestry of existing reported IBD, not phased ancestry of DNABR rare alleles.",
    "Denominator is union bp summed over distinct haplotype-pair edges, not genome-wide ancestral opportunity.",
    "Distinct haplotype edges may cover the same physical coordinate; this sum is not diploid IBD1/IBD2.",
    "Common IBD/Gnomix phased lineage is declared; historical byte identity and phase accuracy are not certified.",
    "Hard local-ancestry labels are estimates, not true ancestry; discordance and UNKNOWN remain explicit.",
    "Both posterior vectors available does not mean high confidence; no threshold or probability product is used.",
    "No-reported IBD is not biological absence, and genotype/caller detectability is not calibrated here.",
    "Global ancestry, collection and technical differences are not adjusted by this composition summary.",
    "Fixed communities and PC-Relate component strata are descriptive, not independent populations or pedigrees.",
    "Pair-chromosome observations are not unique autosomal pairs or independent biological replicates.",
    "No p-values, confidence intervals, winner, rare increment, founder event or new ancestry inference.",
    "Original separate posterior values remain in authenticated child tables; private data are not made public.",
]


def load_inputs(path, expected_sha):
    path = Path(path).resolve(strict=True)
    ledger = Ledger(path.parent)
    ledger.add(dict(path=str(path),sha256=expected_sha))
    value = strict_json(path)
    require(set(value) == {"schema","expected_samples","expected_partitions","chromosomes","community_evidence","annotations"}
            and value["schema"] == SCHEMA, "Unexpected ancestry-composition contract")
    chromosomes = value["chromosomes"]
    require(chromosomes and len(set(chromosomes)) == len(chromosomes) and all(c in map(str,range(1,23)) for c in chromosomes),
            "Explicit unique autosomes required")
    require([r["chromosome"] for r in value["annotations"]] == chromosomes, "Annotation inventory differs")
    people = load_people(value, ledger)
    require(people["phis"] == list(PHIS), "PC-Relate threshold contract differs")
    return value, ledger, people


def canonical_edge(a,b,ha,hb,n):
    require(0 <= a < n and 0 <= b < n and a != b and ha in (1,2) and hb in (1,2), "Invalid distinct people/homologues")
    if a > b:
        a,b,ha,hb = b,a,hb,ha
    return (a*n+b)*4 + (ha-1)*2 + hb-1


def compact_rows(parent_path, child_path, chrom, people, output, guard, max_temp_mb):
    """Join ordered parents/children once; verify each original parent partition."""
    groups = itertools.groupby(table(child_path), key=lambda r:r["record_id"])
    sentinel = object(); seen=set(); counter=Counter()
    with Path(output).open("x") as out:
        for parent, group in itertools.zip_longest(table(parent_path), groups, fillvalue=sentinel):
            require(parent is not sentinel and group is not sentinel, "Parents and children are incomplete")
            record, rows=group
            require(record == parent["record_id"] and record not in seen, "Parent order/identity mismatch")
            seen.add(record)
            a,b=people["order"].get(parent["sample_a"]),people["order"].get(parent["sample_b"])
            require(a is not None and b is not None, "Parent person outside analytical cohort")
            edge=canonical_edge(a,b,int(parent["hap_a"]),int(parent["hap_b"]),len(people["samples"]))
            lo,hi=int(parent["start_pos"]),int(parent["end_pos"])
            require(parent["chrom"]==chrom and 1 <= lo <= hi and int(parent["parent_bp"])==hi-lo+1,
                    "Parent chromosome/units differ")
            cursor=lo; counts=Counter(); n_children=0
            for child in rows:
                start,end=int(child["start_pos"]),int(child["end_pos"])
                require(child["chrom"]==chrom and start==cursor and start <= end <= hi and
                        int(child["length_bp"])==end-start+1, "Children do not partition their parent")
                require(child["state"] in STATES and child["hardcall_a"] in LABELS and child["hardcall_b"] in LABELS,
                        "Unknown ancestry state/label")
                la,lb=LABELS[child["hardcall_a"]],LABELS[child["hardcall_b"]]
                state=4 if min(la,lb)<0 else la if la==lb else 3
                require(STATES[state]==child["state"], "Label/state contradiction")
                require(child["posterior_status"] in POSTERIOR_STATES,
                        "Unexpected posterior availability")
                posterior=POSTERIOR_STATES[child["posterior_status"]]
                # A/B refer to source-record order; reverse these when canonicalizing people.
                if a>b and posterior in (2,3):posterior=5-posterior
                available=int(posterior==1)
                window=-1 if child["msp_window_index"]==NA else int(child["msp_window_index"])
                require(window >= -1 and (window != -1 or state==4), "Invalid MSP window/gap state")
                # No posterior arithmetic or new confidence threshold is introduced.
                out.write(f"{edge}\t{start}\t{end}\t{state}\t{posterior}\t{window}\n")
                cursor=end+1; n_children+=1; counter["children"]+=1
                counts[child["state"].lower()+"_bp"] += end-start+1
                counts["both_posteriors_bp"] += available*(end-start+1)
                if counter["children"]%50000==0:
                    out.flush();guard.check()
                    require(out.tell() <= max_temp_mb*2**20, "Compact chromosome temporary limit exceeded")
            require(cursor==hi+1 and n_children==int(parent["children_count"]), "Incomplete parent child coverage")
            require(all(counts[s.lower()+"_bp"]==int(parent[s.lower()+"_bp"]) for s in STATES) and
                    counts["both_posteriors_bp"]==int(parent["both_posteriors_bp"]), "Parent ancestry sums differ")
            counter["parents"]+=1
        out.flush();guard.check()
        require(out.tell() <= max_temp_mb*2**20, "Compact chromosome temporary limit exceeded")
    return dict(counter)


def edge_union(rows):
    """Union ordered intervals from one edge without silently resolving conflicts.

    All rows come from the same frozen chromosome/Gnomix annotation source.
    Overlap with different window IDs is inconsistent source geometry. Within
    that window, contradictory states or availability are integrity errors.
    """
    totals=np.zeros(len(FIELDS),dtype=np.int64)
    current=None
    def finish(value):
        start,end,state,available,window=value
        length=end-start+1
        totals[state]+=length;totals[5]+=length*int(available==1);totals[6]+=length
    for value in rows:
        start,end,state,available,window=map(int,value)
        require(start>=1 and end>=start and 0<=state<5 and available in (0,1,2,3) and window>=-1,
                "Invalid compact interval")
        if current is None:
            current=[start,end,state,available,window];continue
        require(start>=current[0], "Intervals not ordered")
        if start<=current[1]:
            require(state==current[2], "Overlapping ancestry states contradict")
            require(window==current[4], "Overlapping MSP windows contradict under one frozen source")
            require(available==current[3], "Posterior availability contradicts within the same MSP window")
            current[1]=max(end,current[1])
        elif start==current[1]+1 and [state,available,window]==current[2:]:
            current[1]=end
        else:
            finish(current);current=[start,end,state,available,window]
    if current is not None:finish(current)
    require(sum(totals[:5])==totals[6] and 0<=totals[5]<=totals[6], "State union does not partition total")
    return totals


def sorted_pair_rows(sorted_path, people):
    n=len(people["samples"])
    with Path(sorted_path).open() as handle:
        parsed=(line.rstrip().split("\t") for line in handle)
        previous=-1; pair=None; sums=np.zeros(len(FIELDS),dtype=np.int64)
        for edge, rows in itertools.groupby(parsed,key=lambda r:int(r[0])):
            require(edge>previous,"Duplicate or unordered edge group");previous=edge
            value=edge//4
            if pair is not None and value!=pair:
                a,b=divmod(pair,n)
                yield dict(sample_a=people["samples"][a],sample_b=people["samples"][b],**dict(zip(FIELDS,map(int,sums))))
                sums=np.zeros(len(FIELDS),dtype=np.int64)
            pair=value;sums+=edge_union(row[1:] for row in rows)
        if pair is not None:
            a,b=divmod(pair,n)
            yield dict(sample_a=people["samples"][a],sample_b=people["samples"][b],**dict(zip(FIELDS,map(int,sums))))


def summarize_pairs(pair_path, people, guard, block_size):
    """Vectorized descriptive totals; no permutation or independent-pair test."""
    common=np.asarray([people["ancestry"][i] is not None and
        all(v[i]>=0 for v in people["labels"].values()) for i in range(len(people["samples"]))])
    totals=empty_totals(people);iterator=iter(table(pair_path))
    labels_by_partition={p['partition_id']:np.asarray(people['labels'][p['partition_id']]) for p in people['partitions']}
    while True:
        rows=list(itertools.islice(iterator,block_size))
        if not rows:break
        a=np.asarray([people["order"][r["sample_a"]] for r in rows]);b=np.asarray([people["order"][r["sample_b"]] for r in rows])
        values=np.asarray([[int(r[f]) for f in FIELDS] for r in rows],dtype=np.int64)
        require(np.all(a<b) and np.all(values>=0) and np.all(values[:,:5].sum(axis=1)==values[:,6]) and
                np.all(values[:,5]<=values[:,6]),"Invalid pair ancestry union")
        require(int(values.sum())<2**53,"Block sums exceed exact bincount integer range")
        relation_masks={None:[("ALL",np.ones(len(rows),dtype=bool))]}
        for phi in PHIS:
            comp=people['components'][phi]
            ca=np.asarray([comp[i] for i in a],dtype=object);cb=np.asarray([comp[i] for i in b],dtype=object)
            known=np.asarray([(x is not None and y is not None) for x,y in zip(ca,cb)])
            same=ca==cb
            relation_masks[phi]=[("SAME_COMPONENT",known&same),("DIFFERENT_COMPONENT",known&~same),
                                 ("UNKNOWN_COMPONENT",~known)]
        for p in people["partitions"]:
            labels=labels_by_partition[p['partition_id']];la,lb=labels[a],labels[b]
            for support, eligible in (("assigned_partition",(la>=0)&(lb>=0)),("common_all_partitions",common[a]&common[b])):
                for phi in (None,*PHIS):
                    for relation,relation_mask in relation_masks[phi]:
                        selected=eligible&relation_mask
                        if not selected.any():continue
                        same=la==lb
                        # Group0 = within-all, group1 = between-all, then individual communities.
                        for mode,mask,g in (("all",selected,np.where(same,0,1)),
                                            ("community",selected&same,la+2)):
                            if not mask.any():continue
                            group=g[mask];count=np.bincount(group)
                            columns=np.column_stack([np.rint(np.bincount(group,weights=values[mask,j])).astype(np.int64)
                                                    for j in range(len(FIELDS))])
                            for index in np.flatnonzero(count):
                                scope="WITHIN_ALL" if index==0 else "BETWEEN_ALL" if index==1 else f"COMMUNITY_{index-2}"
                                key=p["partition_id"],support,phi,relation,scope
                                if key not in totals:totals[key]=np.zeros(len(FIELDS)+1,dtype=np.int64)
                                totals[key][:-1]+=columns[index];totals[key][-1]+=count[index]
        guard.check()
    return totals


def empty_totals(people):
    """Keep communities without reported territory visible, with a zero denominator."""
    common=[i for i in range(len(people["samples"])) if people["ancestry"][i] is not None and
            all(v[i]>=0 for v in people["labels"].values())]
    result={}
    for p in people["partitions"]:
        labels=people["labels"][p["partition_id"]]
        for support,indices in (("assigned_partition",[i for i,v in enumerate(labels) if v>=0]),
                                ("common_all_partitions",common)):
            scopes=["WITHIN_ALL","BETWEEN_ALL"]+[f"COMMUNITY_{c}" for c in sorted({labels[i] for i in indices})]
            for phi in (None,*PHIS):
                relations=("ALL",) if phi is None else ("SAME_COMPONENT","DIFFERENT_COMPONENT","UNKNOWN_COMPONENT")
                for relation in relations:
                    for scope in scopes:
                        result[p["partition_id"],support,phi,relation,scope]=np.zeros(len(FIELDS)+1,dtype=np.int64)
    return result


def result_rows(totals,people):
    lookup={p["partition_id"]:p for p in people["partitions"]}
    for key,values in sorted(totals.items(),key=lambda x:str(x[0])):
        pid,support,phi,relation,scope=key;p=lookup[pid];den=int(values[6])
        require(sum(values[:5])==den and 0<=values[5]<=den,"Summary states do not partition denominator")
        yield dict(partition_id=pid,graph_id=p["graph_id"],family=p["family"],resolution=p["resolution"],
            support=support,phi=phi,relation=relation,scope=scope,
            **dict(zip(FIELDS,map(int,values[:-1]))),pair_chromosome_observations=int(values[-1]),
            **{s.lower()+"_fraction":float(values[j]/den) if den else None for j,s in enumerate(STATES)},
            both_posteriors_available_fraction=float(values[5]/den) if den else None,
            denominator_status="REPORTED_HAPLOTYPE_EDGE_TERRITORY" if den else "NO_REPORTED_TERRITORY")


def finish(output,ledger,schema,status,**extra):
    ledger.recheck()
    value=dict(schema=schema,status=status,**extra,inputs=ledger.records,
        code_sha256={name:sha256(Path(__file__).parent/name) for name in
            ("r02_community_ibd_ancestry.py","r02_community_ibd.py","r02_community_levels.py")},
        contains_individual_identifiers=True,public_distribution_allowed=False,
        outputs_sha256={p.name:sha256(p) for p in Path(output).iterdir() if p.is_file()})
    save_json(Path(output)/"manifest.json",value)
    return value


def chromosome_stage(path,expected_sha,chrom,output,guard,sort_memory_mb,max_temp_mb,block_size,sort_timeout):
    started=time.monotonic();contract,ledger,people=load_inputs(path,expected_sha)
    require(chrom in contract["chromosomes"],"Unexpected chromosome")
    spec=next(r for r in contract["annotations"] if r["chromosome"]==chrom)
    mp=ledger.add(spec["manifest"]);m=strict_json(mp);cp=ledger.add(spec["contract"]);c=strict_json(cp)
    require(m["schema"]=="r02_ibd_local_ancestry_v1" and
            m['status']=='ANNOTATED_REPORTED_IBD_WITH_EXISTING_GNOMIX_NOT_VALIDATION' and c["schema"]==m["schema"] and
            m["chromosome"]==c["chromosome"]==chrom and m["genome_build"]==c["genome_build"]=="GRCh38" and
            c["coordinate_system"]=="1-based-inclusive" and m["input_contract_sha256"]==sha256(cp) and
            c["analytical_samples_sha256"]==people["sample_hash"] and m["analytical_samples"]==len(people["samples"]),
            "Annotation identity/build/units differ")
    require(m["phase_declared_source"]==c["phase_declared_source"] and c["phase_declared_source"]["same_source"] is True and
            c["phase_declared_source"]["haplotype_mapping"]=={"1":0,"2":1} and
            c["population_order"]==["African","European","Native-American"],"Phase/ancestry identity differs")
    parents=ledger.adjacent(mp,m,"parents.tsv.gz");children=ledger.adjacent(mp,m,"children.tsv.gz")
    output=Path(output);require(not output.exists(),"No overwrite allowed");output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="ibd-ancestry-",dir=guard.scratch) as folder:
        folder=Path(folder);raw=folder/"compact.tsv";ordered=folder/"ordered.tsv"
        counts=compact_rows(parents,children,chrom,people,raw,guard,max_temp_mb)
        require(counts["parents"]==m["analytical_records"] and counts["children"]==m["children"],"Source row counts differ")
        require(raw.stat().st_size*3+guard.reserve*2**20 <= __import__('shutil').disk_usage(folder).free,
                "Insufficient sort space above protected reserve")
        with ordered.open("x") as out, (output/"sort.stderr").open("x") as errors:
            process=subprocess.Popen(["sort","-k1,1n","-k2,2n","-k3,3n","--parallel=1","-S",f"{sort_memory_mb}M",
                "-T",str(folder),str(raw)],stdout=out,stderr=errors,env={**os.environ,"LC_ALL":"C"})
            try:
                while process.poll() is None:
                    require(time.monotonic()-started < sort_timeout,"Chromosome wall-time limit exceeded")
                    guard.check();time.sleep(.5)
                require(process.returncode==0,"External sort failed")
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=15)
        pairs=write_table(output/"pairs.private.tsv.gz",sorted_pair_rows(ordered,people),["sample_a","sample_b",*FIELDS])
        totals=summarize_pairs(output/"pairs.private.tsv.gz",people,guard,block_size)
    rows=list(result_rows(totals,people));write_table(output/"composition.tsv",rows)
    return finish(output,ledger,PART_SCHEMA,"CHROMOSOME_REPORTED_IBD_ANCESTRY_COMPOSITION",chromosome=chrom,
        input_contract_sha256=expected_sha,samples_sha256=people["sample_hash"],reported_pairs=pairs,
        source_counts=counts,phase_accuracy=m["phase_accuracy"],n_rows=len(rows),seconds=time.monotonic()-started,
        limitations=LIMITATIONS,parameters=dict(sort_memory_mb=sort_memory_mb,max_temp_mb=max_temp_mb,block_size=block_size,
        max_memory_mb=guard.memory,min_free_mb=guard.reserve,sort_timeout_seconds=sort_timeout))


def aggregate_stage(path,expected_sha,part_paths,output,guard):
    contract,ledger,people=load_inputs(path,expected_sha);totals=empty_totals(people);seen=set();part_records=[]
    expected_keys=set(totals);parameter_signature=None
    expected_code={name:sha256(Path(__file__).parent/name) for name in
        ('r02_community_ibd_ancestry.py','r02_community_ibd.py','r02_community_levels.py')}
    partition_lookup={p['partition_id']:p for p in people['partitions']}
    for p in part_paths:
        p=Path(p).resolve(strict=True);m=strict_json(p)
        require(m["schema"]==PART_SCHEMA and m['status']=='CHROMOSOME_REPORTED_IBD_ANCESTRY_COMPOSITION' and
                m['code_sha256']==expected_code and m["input_contract_sha256"]==expected_sha and
                m["samples_sha256"]==people["sample_hash"] and m["chromosome"] in contract["chromosomes"] and
                m["chromosome"] not in seen,"Unexpected/repeated ancestry-composition partial")
        require(set(m['parameters'])=={'sort_memory_mb','max_temp_mb','block_size','max_memory_mb','min_free_mb','sort_timeout_seconds'},
                'Unexpected partial parameters')
        if parameter_signature is None:parameter_signature=m['parameters']
        require(m['parameters']==parameter_signature,'Inconsistent partial parameters')
        ledger.add(dict(path=str(p),sha256=sha256(p)));seen.add(m["chromosome"])
        part_records.append(dict(chromosome=m["chromosome"],manifest_sha256=sha256(p),source_counts=m["source_counts"]))
        data=ledger.adjacent(p,m,"composition.tsv")
        part_keys=set();n_rows=0
        for r in table(data):
            key=r["partition_id"],r["support"],None if r["phi"]==NA else float(r["phi"]),r["relation"],r["scope"]
            require(key in expected_keys and key not in part_keys,'Unknown or duplicated partial grid key')
            part_keys.add(key);n_rows+=1
            partition=partition_lookup[key[0]]
            require(r['graph_id']==partition['graph_id'] and r['family']==partition['family'] and
                    float(r['resolution'])==partition['resolution'],'Partial graph identity differs')
            values=np.asarray([int(r[f]) for f in FIELDS]+[int(r["pair_chromosome_observations"])],dtype=np.int64)
            require(np.all(values>=0) and values[:5].sum()==values[6] and values[5]<=values[6] and
                    values[7]<=m['reported_pairs'],"Invalid partial lengths/counts")
            totals[key]+=values
        require(part_keys==expected_keys and n_rows==m['n_rows'],'Partial grid or row count incomplete')
        guard.check()
    require(seen==set(contract["chromosomes"]),"Incomplete autosomes: no partial aggregate accepted")
    output=Path(output);require(not output.exists(),"No overwrite allowed");output.mkdir(parents=True)
    rows=list(result_rows(totals,people));write_table(output/"composition.tsv",rows)
    summary=dict(schema=SCHEMA,status=STATUS,chromosomes=contract["chromosomes"],n_samples=len(people["samples"]),
        n_partitions=len(people["partitions"]),n_rows=len(rows),parts=part_records,
        denominator="REPORTED_HAPLOTYPE_PAIR_EDGE_UNION_BP_SUM",phase_accuracy="NOT_VALIDATED",
        rare_phase_assigned=False,ancestry_adjustment=False,posterior_threshold=None,
        posterior_probabilities_multiplied=False,no_winner_selected=True,limitations=LIMITATIONS)
    save_json(output/"summary.json",summary)
    with (output/"INFORME.md").open("x") as out:
        out.write("# Composición ancestral del IBD reportado\n\n")
        out.write(f"Se reutilizaron {len(seen)} cromosomas anotados y {len(people['partitions'])} particiones, sin repetir IBD ni Gnomix.\n\n")
        out.write("Cada fracción divide bp con etiqueta concordante AFR/EUR/NAM, discordancia o etiqueta desconocida "
                  "por territorio de aristas haplotípicas IBD reportadas. Se unieron solapamientos dentro de la misma "
                  "pareja de personas/copias y cromosoma; aristas diferentes conservan su identidad. "
                  "No es fracción del genoma ancestral evaluable ni clasificación IBD1/IBD2.\n\n")
        out.write("La disponibilidad de dos vectores de probabilidad no significa alta confianza. "
                  "Se conserva UNKNOWN; no se multiplicaron probabilidades ni se fasearon alelos raros. "
                  "Una concentración ancestral de IBD puede deberse a composición global, historia genealógica o medición; "
                  "esta descripción no distingue por sí sola esas explicaciones.\n\n")
        out.write("## Límites\n\n"+"\n".join("- "+v for v in LIMITATIONS)+"\n")
    return finish(output,ledger,SCHEMA,STATUS,input_contract_sha256=expected_sha,chromosomes=contract["chromosomes"],
                  samples_sha256=people["sample_hash"],n_rows=len(rows),limitations=LIMITATIONS)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("chromosome","aggregate"));parser.add_argument("--contract",required=True)
    parser.add_argument("--contract-sha256",required=True);parser.add_argument("--chromosome")
    parser.add_argument("--parts",nargs="*");parser.add_argument("--output-dir",required=True)
    parser.add_argument("--scratch",required=True);parser.add_argument("--max-memory-mb",type=int,required=True)
    parser.add_argument("--min-free-mb",type=int,required=True);parser.add_argument("--sort-memory-mb",type=int,default=64)
    parser.add_argument("--max-temp-mb",type=int,default=1024);parser.add_argument("--block-size",type=int,default=5000)
    parser.add_argument("--sort-timeout",type=int,default=3600)
    args=parser.parse_args()
    require(0<args.max_memory_mb<=2048 and 0<args.sort_memory_mb<=256 and 0<args.max_temp_mb<=2048 and
            0<args.block_size<=10000 and 0<args.sort_timeout<=7200,"Invalid bounded resources")
    guard=Guard(args.scratch,args.max_memory_mb,1,args.min_free_mb)
    if args.mode=="chromosome":
        result=chromosome_stage(args.contract,args.contract_sha256,args.chromosome,args.output_dir,guard,
             args.sort_memory_mb,args.max_temp_mb,args.block_size,args.sort_timeout)
    else:
        require(args.parts,"Explicit partial manifests required")
        result=aggregate_stage(args.contract,args.contract_sha256,args.parts,args.output_dir,guard)
    print(json.dumps({k:result[k] for k in ("schema","status","n_rows")},allow_nan=False))


if __name__=="__main__":main()
