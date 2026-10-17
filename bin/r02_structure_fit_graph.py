#!/usr/bin/env python3
"""Freeze the existing FIT spectrum once and reuse it for authenticated seeds.

No graph, eigensolver, Leiden objective, or projection algorithm is duplicated.
The stored representation is accepted only under an external manifest pin and
the exact producing implementation, with bounded, non-pickle array decoding.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import numpy as np
import r02_structure_fit_partitions as learner
import r02_resolution_profile as profile
import r02_structure_search_policy as policy
from r02_structure_affinity_io import (AuthenticatedAffinityView,
    AuthenticatedCompositeAffinityView)

io = learner.discrete
require = io.require
CODE_FILES = tuple(dict.fromkeys(('r02_structure_fit_graph.py','r02_resolution_profile.py',
    'r02_structure_search_policy.py','r02_structure_h_affinities.py','r02_structure_h_core.py',
    'r02_m14_chains.py','rare_allele_sharing_painter.py',*learner.CODE_FILES)))
ARRAYS = ('fit_indices','adjacency','degree','active_fit','inverse_sqrt_degree','normalized',
    'connected_component','eigenvalues','eigenvectors','trivial_modes','zero_modes',
    'projectable_modes','fit_scores','shared_component_ids','shared_component_volumes',
    'shared_component_basis','shared_component_fit_scores','shared_fit_states')
SCALARS = ('memory_bound','persons_sha256','source_sha256','query_indices','graph_sha256','numerical_tolerance')
FAMILY_GRAPH_REQUEST='r02_fit_family_graph_request_v2'
FAMILY_CANDIDATE_REQUEST='r02_fit_family_cached_candidate_request_v2'
FAMILY_PROJECTION_REQUEST='r02_fit_family_projection_request_v1'
FAMILY_GRAPH_SCHEMA='r02_fit_family_graph_frozen_v2'
FAMILY_STATE_SCHEMA='r02_fit_family_graph_state_v2'


def implementation_hashes():
    return {name:io.sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def json_write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    path.chmod(0o600)


def pin(path, expected, maximum):
    path=Path(path)
    require(io.valid_hash(expected) and path.is_file() and path.stat().st_size<=maximum
            and io.sha(path)==expected,'Graph input identity/size changed')
    return path


def _spectrum_semantics(value):
    """Reduce a rich provider declaration to the numerical graph contract."""
    require(isinstance(value,dict) and all(k in value for k in ('family','measure','units')),
            'Affinity family/measure/units required')
    if 'range' in value:
        require(set(value['range'])=={'minimum','maximum'},'Exact affinity range required')
        minimum,maximum=value['range']['minimum'],value['range']['maximum']
    else:
        require(all(k in value for k in ('minimum','maximum')),'Affinity numerical range required')
        minimum,maximum=value['minimum'],value['maximum']
    return dict(family=value['family'],measure=value['measure'],units=value['units'],
                minimum=minimum,maximum=maximum)


def _provider_files(provider):
    """Actual pinned files used by a provider, for cache/provenance binding."""
    if hasattr(provider,'common') and hasattr(provider,'rare'):
        files={'descriptor':dict(path=str(provider.spec_path),sha256=provider.source_sha256)}
        for prefix,source in (('common',provider.common),('rare',provider.rare)):
            files.update({prefix+':'+name:dict(path=str(path),sha256=source.pins[name])
                          for name,path in source.inputs.items()})
        return files
    paths=getattr(provider,'paths',None)
    if paths is not None:
        files={'manifest':dict(path=str(provider.manifest_path),sha256=provider.source_sha256)}
        files.update({name:dict(path=str(path),sha256=provider.pins[name]) for name,path in paths.items()})
        return files
    return {name:dict(path=str(path),sha256=provider.pins[name])
            for name,path in provider.inputs.items()}


def _persons_path(provider):
    if hasattr(provider,'paths'): return provider.paths['persons']
    if hasattr(provider,'common'): return provider.common.inputs['persons.tsv']
    return provider.inputs['persons.tsv']


def _provider_memory_bound(provider):
    return 3*sum(a.nbytes for group in (getattr(provider,'support_arrays',{}),
                                        getattr(provider,'affinity_arrays',{})) for a in group.values())


def load_provider(spec, *, evidence_scope, max_input_bytes, max_metadata_bytes,
                  max_memory_bytes, need_queries):
    """Instantiate one explicit family provider without inferring from filenames."""
    require(isinstance(spec,dict) and spec.get('schema')=='r02_family_affinity_provider_v1'
            and spec.get('kind') in {'FAMILY_TABLE','LAZY_COMPOSITE','H_COMPACT'}
            and isinstance(spec.get('family'),str) and spec['family'],
            'Explicit supported family provider required')
    common=dict(evidence_scope=evidence_scope,max_input_bytes=max_input_bytes,
                max_metadata_bytes=max_metadata_bytes)
    if spec['kind']=='FAMILY_TABLE':
        require(set(spec)=={'schema','kind','family','inputs'},'Exact family-table provider required')
        source=AuthenticatedAffinityView(spec['inputs'],expected_family=spec['family'],**common)
    elif spec['kind']=='LAZY_COMPOSITE':
        require(set(spec)=={'schema','kind','family','descriptor'} and spec['family']=='R_PLUS_C',
                'Exact R+C provider required')
        source=AuthenticatedCompositeAffinityView(spec['descriptor'],expected_family=spec['family'],**common)
    else:
        require(set(spec)=={'schema','kind','family','root','manifest'} and spec['family']=='H'
                and set(spec['manifest'])=={'path','sha256'},'Exact compact H provider required')
        from r02_structure_h_affinities import AuthenticatedHCompactView
        source=AuthenticatedHCompactView(spec['root'],spec['manifest']['path'],spec['manifest']['sha256'],
            max_memory_bytes=max_memory_bytes,**common)
    require(source.graph_semantics()['family']==spec['family']
            and (not need_queries or callable(getattr(source,'query_rows',None))),
            'Provider family or query capability differs')
    actual=_provider_files(source)
    unique={record['path'] for record in actual.values()}
    require(sum(Path(path).stat().st_size for path in unique)<=max_input_bytes,
            'Combined provider files exceed input byte budget')
    source.verify()
    return source,actual


def _territory(source):
    value=source.upstream.get('territory')
    require(isinstance(value,dict) and value,'Explicit provider territory required')
    return value


def _graph_identity(graph, *, with_semantics):
    value=dict(source_sha256=graph.source_sha256,persons_sha256=graph.persons_sha256,
        fit_indices=graph.fit_indices.tolist(),adjacency=graph.adjacency.tolist(),
        same_component_fit_edges='EXCLUDE',missing_edge_policy='UNKNOWN_NOT_ZERO')
    if with_semantics:value['affinity_semantics']=graph.affinity_semantics
    return io.materialize.digest(value)


def write_graph(output, graph, source, request_sha256, *, evidence_scope, max_output_bytes, min_free_bytes):
    graph.verify(source.source_sha256);source.verify()
    require(tuple(graph._arrays)==ARRAYS,'FIT graph schema changed; explicit adapter review required')
    out=Path(output)
    estimated=sum(getattr(graph,n).nbytes for n in ARRAYS)+source.inputs['persons.tsv'].stat().st_size+2**20
    require(estimated<=max_output_bytes,'Frozen graph exceeds output budget')
    require(not out.exists() and out.parent.is_dir() and shutil.disk_usage(out.parent).free>=min_free_bytes+estimated,
            'New graph directory and explicit disk reserve required')
    out.mkdir(mode=0o700)
    shutil.copyfile(source.inputs['persons.tsv'],out/'persons.tsv')
    np.savez(out/'spectrum.private.npz',**{n:getattr(graph,n) for n in ARRAYS})
    scalars={n:getattr(graph,n) for n in SCALARS}
    metadata=dict(schema='r02_fit_graph_state_v1',scalars=scalars,
        arrays={n:dict(dtype=str(getattr(graph,n).dtype),shape=list(getattr(graph,n).shape),
                       sha256=graph._pins[n]) for n in ARRAYS})
    json_write(out/'state.json',metadata)
    source.verify();graph.verify(source.source_sha256)
    upstream=source.upstream
    result=dict(schema='r02_fit_graph_frozen_v1',stage='FIT_GRAPH_ONLY_NO_COMMUNITIES_OR_SELECTION',
        request_sha256=request_sha256,inputs={n:dict(path=str(p),sha256=source.pins[n])for n,p in source.inputs.items()},
        implementation_sha256=implementation_hashes(),source_sha256=source.source_sha256,
        graph_sha256=graph.graph_sha256,persons_sha256=graph.persons_sha256,
        view_id=upstream['exclusion_scope']+':'+upstream['block_id'],
        exclusion_scope=upstream['exclusion_scope'],block_id=upstream['block_id'],
        n_fit=len(graph.fit_indices),n_query=len(graph.query_indices),
        workspace_bound_bytes=graph.memory_bound,arrays_bytes=sum(getattr(graph,n).nbytes for n in ARRAYS),
        outputs_sha256={n:io.sha(out/n)for n in ('persons.tsv','spectrum.private.npz','state.json')},
        evidence_scope=evidence_scope,query_edges_used=False,evaluate_scored=False,biological_validation=False)
    for p in out.iterdir():p.chmod(0o600)
    require(sum(p.stat().st_size for p in out.iterdir())<=max_output_bytes,'Graph serialized output exceeds budget')
    json_write(out/'graph_manifest.json',result)
    return result


def write_family_graph(output,graph,source,provider,source_files,request_sha256,*,
                       evidence_scope,max_output_bytes,min_free_bytes):
    """Freeze a family-explicit graph while retaining the existing arrays."""
    graph.verify(source.source_sha256);source.verify()
    semantics=source.graph_semantics(); numerical=_spectrum_semantics(semantics)
    require(graph.affinity_semantics==numerical and tuple(graph._arrays)==ARRAYS,
            'Family graph semantics or array schema changed')
    out=Path(output);persons_path=_persons_path(source)
    estimated=sum(getattr(graph,n).nbytes for n in ARRAYS)+persons_path.stat().st_size+2**20
    require(estimated<=max_output_bytes and not out.exists() and out.parent.is_dir()
            and shutil.disk_usage(out.parent).free>=min_free_bytes+estimated,
            'New family graph output exceeds disk/byte budget')
    out.mkdir(mode=0o700);shutil.copyfile(persons_path,out/'persons.tsv')
    np.savez(out/'spectrum.private.npz',**{n:getattr(graph,n)for n in ARRAYS})
    state=dict(schema=FAMILY_STATE_SCHEMA,scalars={n:getattr(graph,n)for n in SCALARS},
        affinity_semantics=numerical,
        arrays={n:dict(dtype=str(getattr(graph,n).dtype),shape=list(getattr(graph,n).shape),
                       sha256=graph._pins[n])for n in ARRAYS})
    json_write(out/'state.json',state)
    source.verify();graph.verify(source.source_sha256)
    result=dict(schema=FAMILY_GRAPH_SCHEMA,stage='FIT_GRAPH_ONLY_NO_COMMUNITIES_OR_SELECTION',
        request_sha256=request_sha256,provider=provider,source_files=source_files,
        implementation_sha256=implementation_hashes(),source_sha256=source.source_sha256,
        graph_sha256=graph.graph_sha256,persons_sha256=graph.persons_sha256,
        family=numerical['family'],affinity_semantics=semantics,
        spectrum_semantics=numerical,territory=_territory(source),
        same_component_fit_edges='EXCLUDE',missing_edge_policy='UNKNOWN_NOT_ZERO',
        view_id=io.materialize.digest(dict(family=numerical['family'],territory=_territory(source),
                                           source_sha256=source.source_sha256)),
        n_fit=len(graph.fit_indices),n_query=len(graph.query_indices),
        workspace_bound_bytes=graph.memory_bound,arrays_bytes=sum(getattr(graph,n).nbytes for n in ARRAYS),
        outputs_sha256={n:io.sha(out/n)for n in ('persons.tsv','spectrum.private.npz','state.json')},
        evidence_scope=evidence_scope,query_edges_used=False,evaluate_scored=False,biological_validation=False)
    require(sum(p.stat().st_size for p in out.iterdir())<=max_output_bytes,
            'Family graph serialized output exceeds budget')
    for p in out.iterdir():p.chmod(0o600)
    json_write(out/'graph_manifest.json',result)
    return result


def load_graph(directory, manifest_sha256, *, max_input_bytes, max_metadata_bytes, max_memory_bytes):
    root=Path(directory)
    m=io.load_json(pin(root/'graph_manifest.json',manifest_sha256,max_metadata_bytes))
    require(m['schema'] in {'r02_fit_graph_frozen_v1',FAMILY_GRAPH_SCHEMA}
            and m['stage']=='FIT_GRAPH_ONLY_NO_COMMUNITIES_OR_SELECTION'
            and m['implementation_sha256']==implementation_hashes(),'FIT graph producing implementation differs')
    require(set(m['outputs_sha256'])=={'persons.tsv','spectrum.private.npz','state.json'},'Exact graph artifact inventory required')
    require(sum((root/n).stat().st_size for n in m['outputs_sha256'])<=max_input_bytes,'Graph input budget exceeded')
    for n,h in m['outputs_sha256'].items():pin(root/n,h,max_input_bytes if n.endswith('.npz')else max_metadata_bytes)
    state=io.load_json(root/'state.json')
    family=m['schema']==FAMILY_GRAPH_SCHEMA
    require(state['schema']==(FAMILY_STATE_SCHEMA if family else 'r02_fit_graph_state_v1')
            and set(state['scalars'])==set(SCALARS) and set(state['arrays'])==set(ARRAYS)
            and (not family or set(state)=={'schema','scalars','affinity_semantics','arrays'}),
            'Exact graph state fields required')
    with (root/'persons.tsv').open() as fh:persons=list(csv.DictReader(fh,delimiter='\t'))
    roles=io.shared.validate_roles(persons)
    require(io.materialize.digest(persons)==m['persons_sha256'],'Graph persons content differs')
    decoded=0
    with zipfile.ZipFile(root/'spectrum.private.npz') as z:
        require(set(z.namelist())=={n+'.npy'for n in ARRAYS} and len(z.namelist())==len(ARRAYS),
                'Unexpected or duplicate graph array')
        for name in ARRAYS:
            info=z.getinfo(name+'.npy')
            with z.open(info) as fh:
                version=np.lib.format.read_magic(fh)
                require(version in {(1,0),(2,0)},'Unsupported frozen graph array format version')
                reader=np.lib.format.read_array_header_1_0 if version==(1,0)else np.lib.format.read_array_header_2_0
                shape,fortran,dtype=reader(fh)
                require(not dtype.hasobject and all(type(v)is int and v>=0 for v in shape),'Unsafe graph array dtype/shape')
                expected=state['arrays'][name]
                size=math.prod(shape)*dtype.itemsize
                require(list(shape)==expected['shape'] and str(dtype)==expected['dtype'] and
                        fh.tell()+size==info.file_size,'Graph array header differs from frozen metadata')
                decoded+=size
    require(3*decoded<=max_memory_bytes and decoded==m['arrays_bytes'],'Graph decode workspace exceeds budget')
    graph=learner.FitRareGraphSpectrum.__new__(learner.FitRareGraphSpectrum)
    for name,value in state['scalars'].items():setattr(graph,name,tuple(value)if name=='query_indices'else value)
    if family:
        require(state['affinity_semantics']==m['spectrum_semantics']
                and _spectrum_semantics(m['affinity_semantics'])==m['spectrum_semantics']
                and m['family']==m['spectrum_semantics']['family'],
                'Frozen family graph semantic binding differs')
        graph.affinity_semantics=dict(state['affinity_semantics'])
    with np.load(root/'spectrum.private.npz',allow_pickle=False) as z:
        for name in ARRAYS:
            array=z[name]
            require(hashlib.sha256(array.tobytes()).hexdigest()==state['arrays'][name]['sha256'],'Graph array content differs')
            array.setflags(write=False);setattr(graph,name,array)
    graph._arrays=ARRAYS;graph._pins={n:state['arrays'][n]['sha256']for n in ARRAYS}
    require(graph.persons_sha256==m['persons_sha256'] and graph.source_sha256==m['source_sha256']
            and graph.graph_sha256==m['graph_sha256'],'Frozen graph scalar provenance differs')
    require(graph.graph_sha256==_graph_identity(graph,with_semantics=family),
            'Frozen graph identity cannot be reproduced from arrays and semantics')
    expected_query=tuple(sorted(int(i)for role in ('SELECT','EVALUATE')for i in roles[role]))
    require(np.array_equal(graph.fit_indices,roles['FIT']) and graph.query_indices==expected_query
            and graph.adjacency.shape==(len(roles['FIT']),len(roles['FIT'])),'Graph roster/shape differs')
    graph.verify(m['source_sha256'])
    pin(root/'graph_manifest.json',manifest_sha256,max_metadata_bytes)
    return graph,persons,m


def admission(path, expected, request_sha256, scientific_spec_sha256, mode, evidence_scope, maximum):
    a=io.load_json(pin(path,expected,maximum))
    require(a['schema']=='r02_fit_graph_task_admission_v1' and a['request_sha256']==request_sha256
        and a['scientific_spec_sha256']==scientific_spec_sha256 and a['mode']==mode
        and a['evidence_scope']==evidence_scope and a['status']=='FIT_ONLY_ADMITTED'
        and a['evaluate_allowed'] is False,'Explicit FIT graph/candidate admission required')


def write_family_candidate(output,candidate,persons,graph_manifest,episode_sha256):
    """Write labels with family identity; numerical learning stays in learner."""
    payload={k:v for k,v in candidate.items()if k!='candidate_sha256'}
    require(candidate.get('candidate_sha256')==io.materialize.digest(payload)
            and candidate['status']=='FIT_ZERO_IMPROVEMENT_ITERATION_REACHED',
            'Only an unchanged converged candidate can be frozen')
    roles=io.shared.validate_roles(persons)
    require(io.materialize.digest(persons)==candidate['persons_sha256']
            and np.array_equal(roles['FIT'],candidate['fit_indices']),
            'Family candidate roster differs')
    out=Path(output);out.mkdir(parents=True,mode=0o700)
    table='sample_id\tcommunity\n'+''.join(
        persons[i]['sample_id']+'\t'+candidate['labels'][str(i)]+'\n'for i in roles['FIT'])
    (out/'communities.private.tsv').write_text(table)
    identity=dict(family=graph_manifest['family'],affinity_semantics=graph_manifest['affinity_semantics'],
        territory=graph_manifest['territory'],source_sha256=graph_manifest['source_sha256'],
        graph_sha256=graph_manifest['graph_sha256'])
    method=dict(schema='r02_fit_family_community_method_v2',algorithm='Leiden RBConfigurationVertexPartition',
        identity=identity,parameters={k:v for k,v in candidate.items()if k not in ('labels','code_sha256')},
        code_sha256=candidate['code_sha256'],fit_only=True)
    json_write(out/'community_method.json',method)
    receipt=dict(schema='r02_fit_family_communities_frozen_v2',status='FIT_ONLY_FROZEN_BEFORE_PROJECTION',
        identity=identity,episode_sha256=episode_sha256,
        fit_ids_sha256=io.ordered_ids_hash(persons[i]['sample_id']for i in roles['FIT']),
        communities_sha256=io.sha(out/'communities.private.tsv'),
        method_manifest_sha256=io.sha(out/'community_method.json'),learning_roles=['FIT'],
        target_used=False,heldout_edges_used=False)
    json_write(out/'frozen_communities.json',receipt)
    return receipt


def project_family(output,graph,persons,manifest,source,roles,*,request_sha256,
                   implementation_sha256,graph_reference,source_files,
                   max_output_bytes,min_free_bytes):
    """Project explicit held-out roles against the frozen FIT graph only."""
    require(isinstance(roles,list) and roles and len(set(roles))==len(roles)
            and set(roles)<={'SELECT','EVALUATE'},'Explicit unique held-out roles required')
    require(source.source_sha256==graph.source_sha256
            and io.materialize.digest(source.persons)==graph.persons_sha256
            and source.graph_semantics()==manifest['affinity_semantics']
            and _territory(source)==manifest['territory'],'Projection provider differs from frozen graph')
    indices=[];scores=[];shared=[];statuses=[];shared_statuses=[];observed=[];totals=[];degrees=[];support=[]
    for query,observations,diagnostics in source.query_rows(roles=tuple(roles)):
        row=graph.project(query,observations,source_sha256=source.source_sha256)
        indices.append(query);scores.append(row['scores']);shared.append(row['shared_component_scores'])
        statuses.append(row['status']);shared_statuses.append(row['shared_state'])
        observed.append(row['observed_fit_count']);totals.append(row['total_fit_count'])
        degrees.append(np.nan if row['query_degree'] is None else row['query_degree'])
        support.append(diagnostics)
    role_indices=io.shared.validate_roles(persons)
    expected=sorted(i for role in roles for i in role_indices[role])
    require(indices==expected,'Projection provider omitted or reordered held-out people')
    out=Path(output);require(not out.exists()and out.parent.is_dir()
        and shutil.disk_usage(out.parent).free>=min_free_bytes,'New projection output and disk reserve required')
    out.mkdir(mode=0o700)
    arrays=dict(query_indices=np.asarray(indices,dtype=np.int64),
        scores=np.asarray(scores,dtype=np.float64),shared_component_scores=np.asarray(shared,dtype=np.float64),
        status=np.asarray(statuses,dtype=np.str_),shared_state=np.asarray(shared_statuses,dtype=np.str_),
        observed_fit_count=np.asarray(observed,dtype=np.int64),total_fit_count=np.asarray(totals,dtype=np.int64),
        query_degree=np.asarray(degrees,dtype=np.float64),component_ids=graph.shared_component_ids)
    np.savez(out/'projections.private.npz',**arrays)
    json_write(out/'support.json',dict(schema='r02_fit_family_projection_support_v1',rows=support))
    result=dict(schema='r02_fit_family_projection_result_v1',stage='QUERY_TO_FIT_PROJECTED_NO_SELECTION',
        request_sha256=request_sha256,implementation_sha256=implementation_sha256,
        graph=graph_reference,source_files=source_files,
        graph_sha256=graph.graph_sha256,source_sha256=source.source_sha256,family=manifest['family'],
        affinity_semantics=manifest['affinity_semantics'],territory=manifest['territory'],roles=roles,
        query_count=len(indices),incomplete_fit_support=sum(s=='INCOMPLETE_FIT_SUPPORT'for s in statuses),
        outputs_sha256={n:io.sha(out/n)for n in ('projections.private.npz','support.json')},
        missing_edge_policy='UNKNOWN_NOT_ZERO',query_query_used=False,target_outcomes_read=False,
        evaluate_scored=False,biological_validation=False)
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    require(sum(p.stat().st_size for p in out.iterdir())+len(encoded.encode())<=max_output_bytes,
            'Projection output exceeds budget')
    (out/'projection_manifest.json').write_text(encoded);(out/'projection_manifest.json').chmod(0o600)
    return result


def run_request(request_path,request_sha256,admission_path,admission_sha256,output):
    request_path=pin(request_path,request_sha256,2**20);r=io.load_json(request_path)
    schemas={'r02_fit_graph_request_v1','r02_fit_cached_candidate_request_v1',
             FAMILY_GRAPH_REQUEST,FAMILY_CANDIDATE_REQUEST,FAMILY_PROJECTION_REQUEST}
    require(r['schema'] in schemas
        and r['implementation_sha256']==implementation_hashes() and r['evidence_scope'] in {'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'}
        and io.valid_hash(r['scientific_spec_sha256']),'Exact admitted graph/candidate specification required')
    mode=('GRAPH'if r['schema'] in {'r02_fit_graph_request_v1',FAMILY_GRAPH_REQUEST}else
          'CANDIDATE'if r['schema'] in {'r02_fit_cached_candidate_request_v1',FAMILY_CANDIDATE_REQUEST}
          else'PROJECTION')
    limits=r['resources']
    require(set(limits)=={'max_input_bytes','max_metadata_bytes','max_memory_bytes','max_output_bytes','min_free_bytes'}
        and all(type(v)is int and v>0 for v in limits.values()),'Explicit graph task byte budgets required')
    admission(admission_path,admission_sha256,request_sha256,r['scientific_spec_sha256'],mode,r['evidence_scope'],limits['max_metadata_bytes'])
    if r['schema']=='r02_fit_graph_request_v1':
        require(set(r)=={'schema','evidence_scope','scientific_spec_sha256','implementation_sha256','inputs','resources'},'Exact graph request fields required')
        source=learner.AuthenticatedFitAffinities(r['inputs'],evidence_scope=r['evidence_scope'],
            max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'])
        graph=learner.FitRareGraphSpectrum(source.persons,source.fit_edges(),source_sha256=source.source_sha256,
                                          max_memory_bytes=limits['max_memory_bytes'])
        answer=write_graph(output,graph,source,request_sha256,evidence_scope=r['evidence_scope'],
            max_output_bytes=limits['max_output_bytes'],min_free_bytes=limits['min_free_bytes'])
    elif r['schema']==FAMILY_GRAPH_REQUEST:
        require(set(r)=={'schema','evidence_scope','scientific_spec_sha256','implementation_sha256',
                         'provider','source_files','resources'},'Exact family graph request fields required')
        source,files=load_provider(r['provider'],evidence_scope=r['evidence_scope'],
            max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'],
            max_memory_bytes=limits['max_memory_bytes'],need_queries=False)
        require(files==r['source_files'],'Family graph request source inventory differs')
        semantics=source.graph_semantics()
        remaining=limits['max_memory_bytes']-_provider_memory_bound(source)
        require(remaining>0,'No graph memory remains after provider decode')
        graph=learner.continuous.FitAffinityGraphSpectrum(source.persons,source.fit_edges(),
            source_sha256=source.source_sha256,max_memory_bytes=remaining,
            affinity_semantics=_spectrum_semantics(semantics))
        answer=write_family_graph(output,graph,source,r['provider'],files,request_sha256,
            evidence_scope=r['evidence_scope'],max_output_bytes=limits['max_output_bytes'],
            min_free_bytes=limits['min_free_bytes'])
    elif mode=='CANDIDATE':
        require(set(r)=={'schema','evidence_scope','scientific_spec_sha256','implementation_sha256','graph','episode','parameters','resources'},'Exact cached-candidate request fields required')
        graph,persons,m=load_graph(r['graph']['directory'],r['graph']['manifest_sha256'],
            max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'],max_memory_bytes=limits['max_memory_bytes'])
        require(m['evidence_scope']==r['evidence_scope'],'Graph evidence scope differs')
        e=io.load_json(pin(r['episode']['path'],r['episode']['sha256'],limits['max_metadata_bytes']));policy.verify_episode(e)
        params=r['parameters']
        require(set(params)=={'resolution','seed','max_iterations'} and e['view_id']==m['view_id']
            and e['source_sha256']==graph.source_sha256 and e['gamma_hex']==float(params['resolution']).hex()
            and params['seed']in e['seeds'],'Candidate seed/gamma differs from precommitted episode')
        candidate=learner.learn_candidate(graph,source_sha256=graph.source_sha256,
            max_working_bytes=limits['max_memory_bytes']-3*m['arrays_bytes'],**params)
        out=Path(output)
        require(not out.exists() and out.parent.is_dir() and shutil.disk_usage(out.parent).free>=limits['min_free_bytes'],
                'New candidate output and disk reserve required')
        if r['schema']==FAMILY_CANDIDATE_REQUEST:
            require(m['schema']==FAMILY_GRAPH_SCHEMA,'Family candidate requires a family-explicit graph')
            write_family_candidate(out,candidate,persons,m,e['episode_sha256'])
        else:
            require(m['schema']=='r02_fit_graph_frozen_v1','Legacy candidate requires legacy rare graph')
            learner.write_frozen_candidate(out,candidate,persons=persons,persons_sha256=m['inputs']['persons.tsv']['sha256'],
                fit_affinities_sha256=m['inputs']['fit_affinities.private.tsv']['sha256'],exclusion_sha256=m['inputs']['exclusion.json']['sha256'])
        json_write(out/'candidate.json',candidate)
        line=profile.make_line(graph,candidate['labels'])
        answer=dict(schema=('r02_fit_family_cached_candidate_result_v2' if r['schema']==FAMILY_CANDIDATE_REQUEST
                            else'r02_fit_cached_candidate_result_v1'),stage='FIT_CANDIDATE_COMPLETE_NOT_SELECTED',
            request_sha256=request_sha256,implementation_sha256=implementation_hashes(),graph=r['graph'],
            episode_sha256=e['episode_sha256'],episode_file_sha256=r['episode']['sha256'],source_sha256=graph.source_sha256,
            gamma_hex=e['gamma_hex'],seed=params['seed'],membership_key=line.key,rb_line=vars(line),
            status=candidate['status'],evidence_scope=r['evidence_scope'],
            outputs_sha256={p.name:io.sha(p)for p in out.iterdir()},query_edges_used=False,evaluate_scored=False,biological_validation=False)
        if r['schema']==FAMILY_CANDIDATE_REQUEST:
            identity=dict(family=m['family'],affinity_semantics=m['affinity_semantics'],territory=m['territory'],
                          source_sha256=m['source_sha256'],graph_sha256=m['graph_sha256'])
            answer.update(identity=identity,profile_identity_sha256=io.materialize.digest(
                dict(identity=identity,membership_key=line.key,rb_line=vars(line))))
        encoded=json.dumps(answer,indent=2,allow_nan=False)+'\n'
        require(sum(p.stat().st_size for p in out.iterdir())+len(encoded.encode())<=limits['max_output_bytes'],
                'Candidate output exceeds budget')
        (out/'candidate_manifest.json').write_text(encoded);(out/'candidate_manifest.json').chmod(0o600)
    else:
        require(set(r)=={'schema','evidence_scope','scientific_spec_sha256','implementation_sha256',
                         'graph','provider','source_files','parameters','resources'},
                'Exact family projection request fields required')
        graph,persons,m=load_graph(r['graph']['directory'],r['graph']['manifest_sha256'],
            max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'],
            max_memory_bytes=limits['max_memory_bytes'])
        require(m['schema']==FAMILY_GRAPH_SCHEMA and m['evidence_scope']==r['evidence_scope'],
                'Projection requires matching family graph')
        provider_memory=limits['max_memory_bytes']-3*m['arrays_bytes']
        require(provider_memory>0,'No provider memory remains after graph decode')
        source,files=load_provider(r['provider'],evidence_scope=r['evidence_scope'],
            max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'],
            max_memory_bytes=provider_memory,need_queries=True)
        require(files==r['source_files'],'Projection source inventory differs')
        require(set(r['parameters'])=={'roles'},'Exact projection role parameters required')
        answer=project_family(output,graph,persons,m,source,r['parameters']['roles'],
            request_sha256=request_sha256,implementation_sha256=r['implementation_sha256'],
            graph_reference=r['graph'],source_files=files,
            max_output_bytes=limits['max_output_bytes'],min_free_bytes=limits['min_free_bytes'])
    require(io.sha(request_path)==request_sha256 and implementation_hashes()==r['implementation_sha256'],
            'Graph task source/request changed during execution')
    return answer


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('request','request-sha256','admission','admission-sha256','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run_request(a.request,a.request_sha256,a.admission,a.admission_sha256,a.output)
