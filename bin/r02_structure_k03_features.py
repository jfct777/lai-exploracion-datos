#!/usr/bin/env python3
"""K03 representation assembly; no target reader, fitting or model selection.

Callers authenticate every upstream view/profile and the fixed source roster.
This module preserves full available ranks and explicit nonprojectable states.
"""
import hashlib
import numpy as np
import r02_structure_continuous as continuous
import r02_structure_discrete as discrete
import r02_structure_k03_design as design

require = discrete.require
RARE_STATES = ('SUPPORTED','ZERO_DEGREE','INCOMPLETE_FIT_SUPPORT','AFFINITY_TO_ZERO_DEGREE_FIT')


def _state(value):
    if value in {'POSITIVE_DEGREE_FIT','PROJECTED'}:return 'SUPPORTED'
    if value in {'ZERO_DEGREE_FIT','ZERO_QUERY_DEGREE'}:return 'ZERO_DEGREE'
    require(value in RARE_STATES, 'Unknown rare projection state')
    return value


class RareProjectionView:
    """Freeze rare coordinates and query features once, shared by all candidates.

    query_rows is the authenticated complete query-to-FIT iterator. The matrix
    retained here is features only, not within-query targets. Nonprojectable
    coordinates use the weighted mean of observed FIT scores. The reason and
    raw availability remain explicit; no genotype/affinity is imputed as zero.
    """
    def __init__(self, persons, graph, query_rows, *, source_sha256, weighting, max_memory_bytes):
        graph.verify(source_sha256)
        require(graph.persons_sha256 == discrete.materialize.digest(persons), 'Rare graph roster differs')
        self.persons = persons;self.roles = discrete.shared.validate_roles(persons)
        self.graph = graph;self.source_sha256 = source_sha256
        n=len(persons);f=len(graph.fit_indices)
        width=graph.fit_scores.shape[1]+graph.shared_component_fit_scores.shape[1]
        # Inputs/graph are owned separately; allow arrays, copies and one-hot workspace.
        self.memory_bound=8*(4*n*width+3*len(graph.query_indices)*f+4*n*f+32*n)
        require(type(max_memory_bytes)is int and 0<self.memory_bound<=max_memory_bytes,
                'Rare projection workspace exceeds explicit byte bound')
        coordinates=np.full((n,width),np.nan)
        coordinates[graph.fit_indices]=np.concatenate((graph.fit_scores,graph.shared_component_fit_scores),axis=1)
        states=np.full(n,'',dtype='<U40')
        states[graph.fit_indices]=[_state(v)for v in graph.shared_fit_states]
        raw_states=np.full(n,'',dtype='<U40');raw_states[graph.fit_indices]=graph.shared_fit_states
        query=np.full((len(graph.query_indices),f),np.nan)
        iterator=iter(query_rows);diagnostics=[]
        for row_index,expected in enumerate(graph.query_indices):
            item=next(iterator,None)
            require(item is not None and len(item)==3 and item[0]==expected,'Exact complete ordered query roster required')
            index,observations,denominators=item
            require(set(observations)==set(map(int,graph.fit_indices)),'Explicit query-to-FIT support required')
            result=graph.project(index,observations,source_sha256=source_sha256)
            states[index]=_state(result['shared_state']);raw_states[index]=result['shared_state']
            coordinates[index]=np.concatenate((result['scores'],result['shared_component_scores']))
            query[row_index]=[np.nan if observations[int(j)]is None else observations[int(j)]for j in graph.fit_indices]
            diagnostics.append(dict(query_index=index,rare_state=str(states[index]),
                observed_fit_count=result['observed_fit_count'],total_fit_count=result['total_fit_count'],
                query_degree=result['query_degree'],
                affinity_to_isolated_fit_sum=float(np.nansum(query[row_index,~graph.active_fit])),
                source_denominators=denominators))
        require(next(iterator,None)is None,'Unexpected extra query row')
        supported=states=='SUPPORTED'
        require(np.all(np.isfinite(coordinates[supported])) and np.isnan(coordinates[~supported]).all(),
                'Rare coordinate availability differs from shared state')
        w=continuous.fit_weights(persons,graph.fit_indices,weighting)
        valid=supported[graph.fit_indices]
        require(valid.any(),'No supported FIT coordinates for neutralization')
        mean=np.average(coordinates[graph.fit_indices[valid]],axis=0,weights=w[valid])
        coordinates[~supported]=mean
        self.coordinates=coordinates;self.fit_imputation_mean=mean
        self.states=states;self.raw_states=raw_states;self.query_affinities=query
        self.query_indices=np.asarray(graph.query_indices,np.int64)
        self.state_points=np.column_stack([states==name for name in RARE_STATES[1:]]).astype(float)
        self.diagnostics=diagnostics;self.weighting=weighting
        self.graph_sha256=graph.graph_sha256
        self._pins={}
        for name in ['coordinates','fit_imputation_mean','states','raw_states','query_affinities','query_indices','state_points']:
            array=getattr(self,name);array.setflags(write=False)
            self._pins[name]=hashlib.sha256(array.tobytes()).hexdigest()
        graph.verify(source_sha256)

    def verify(self):
        self.graph.verify(self.source_sha256)
        require(all(hashlib.sha256(getattr(self,k).tobytes()).hexdigest()==v for k,v in self._pins.items()),
                'Frozen rare projection changed')
        require(self.graph.persons_sha256==discrete.materialize.digest(self.persons),'Projection roster changed')

    def project_communities(self, fit_labels):
        self.verify();fit=list(map(int,self.graph.fit_indices))
        require(isinstance(fit_labels,dict) and set(fit_labels)==set(fit) and
                all(isinstance(v,str)and v.strip()==v and v for v in fit_labels.values()),'Exact FIT community labels required')
        require(all((fit_labels[i]=='UNASSIGNED')==(not self.graph.active_fit[k])for k,i in enumerate(fit)),
                'FIT isolated/assigned community states differ')
        categories=sorted(set(fit_labels.values())-{'UNASSIGNED'})
        require(categories,'At least one assigned FIT community required')
        index={name:k for k,name in enumerate(categories)}
        labels=['UNASSIGNED']*len(self.persons);reasons=['']*len(labels)
        for i in fit:labels[i]=fit_labels[i];reasons[i]='FIT_FROZEN' if labels[i]!='UNASSIGNED' else 'ZERO_DEGREE_FIT'
        for k,person in enumerate(self.query_indices):
            i=int(person)
            if self.states[i]!='SUPPORTED':reasons[i]=str(self.raw_states[i]);continue
            row=self.query_affinities[k]
            result=discrete.observed_community_means(fit_labels,{j:float(row[t])for t,j in enumerate(fit)if np.isfinite(row[t])})
            labels[i]=result['community']or'UNASSIGNED';reasons[i]=result['status']
        points=np.zeros((len(labels),len(categories)),dtype=float)
        for i,label in enumerate(labels):
            if label!='UNASSIGNED':points[i,index[label]]=1.
        assigned=np.asarray([v!='UNASSIGNED'for v in labels],bool)
        self.verify()
        return dict(one_hot=points,labels=tuple(labels),categories=tuple(categories),
                    assigned=assigned,reasons=tuple(reasons),graph_sha256=self.graph_sha256,
                    target_outcomes_read=False,biological_validation=False)


def conditional_sham_labels(graph,fit_labels,*,source_sha256,seed):
    """Uniform label assignment conditional on sizes inside FIT components.

    This is a representation control, not a genealogy/demographic null. Number
    of draws and selection policy are external explicit scientific decisions.
    """
    graph.verify(source_sha256)
    fit=list(map(int,graph.fit_indices))
    require(type(seed)is int and 0<=seed<2**64 and set(fit_labels)==set(fit),'Explicit seed and exact FIT labels required')
    require(all((fit_labels[i]=='UNASSIGNED')==(not graph.active_fit[k])for k,i in enumerate(fit)),
            'SHAM preserves exactly isolated FIT members')
    owners={}
    for local,i in enumerate(fit):
        label=fit_labels[i]
        if label=='UNASSIGNED':continue
        component=int(graph.connected_component[local])
        require(label not in owners or owners[label]==component,'Community spans disconnected FIT components')
        owners[label]=component
    rng=np.random.Generator(np.random.PCG64(seed));result=dict(fit_labels)
    for component in sorted(set(map(int,graph.connected_component[graph.active_fit]))):
        indices=[i for local,i in enumerate(fit)if graph.active_fit[local]and graph.connected_component[local]==component]
        values=[fit_labels[i]for i in indices]
        for i,value in zip(indices,rng.permutation(values).tolist()):result[i]=value
    graph.verify(source_sha256)
    return result


def pair_inputs(common_points,rare,community,left,right,*,arm,paired_controls,max_output_bytes):
    """Exact arrays for PairDesign, before any FIT scaling/alpha decision.

    Continuous baseline includes the SAME partition-dependent availability as
    A/B. A adds one same-community column; B adds K one-hot individual axes,
    hence all 3K symmetric columns without changing Ridge's penalty. No output
    from this function is a fitted/predictively accepted model.
    """
    rare.verify();common=np.asarray(common_points,dtype=float);controls=np.asarray(paired_controls,dtype=float)
    left=np.asarray(left);right=np.asarray(right);n=len(rare.persons)
    require(arm in {'CONTINUOUS','COMMUNITY_A','COMMUNITY_B'},'Explicit K03 arm required')
    require(common.ndim==2 and common.shape[0]==n and np.isfinite(common).all(),'Finite full common view required')
    require(left.ndim==right.ndim==1 and left.shape==right.shape and len(left)>0 and
            left.dtype.kind in 'iu'and right.dtype.kind in 'iu'and np.all(left>=0)and np.all(right>=0)and
            np.all(left<n)and np.all(right<n)and np.all(left!=right),'Explicit valid pair keys required')
    require(controls.ndim==2 and len(controls)==len(left)and np.isfinite(controls).all(),'Finite aligned baseline controls required')
    hot=np.asarray(community['one_hot']);assigned=np.asarray(community['assigned'])
    require(community['graph_sha256']==rare.graph_sha256 and hot.ndim==2 and hot.shape[0]==n and
            assigned.shape==(n,)and assigned.dtype==np.bool_ and np.all((hot==0)|(hot==1))and
            np.array_equal(hot.sum(axis=1),assigned.astype(int)),'Community encoding/support differs')
    labels=community['labels'];categories=community['categories']
    require(len(labels)==n and len(categories)==hot.shape[1]and len(set(categories))==len(categories)and
            all(v!='UNASSIGNED'for v in categories),'Community category inventory differs')
    index={v:k for k,v in enumerate(categories)}
    require(all((label=='UNASSIGNED'and not assigned[i])or(label in index and assigned[i]and hot[i,index[label]]==1)
                for i,label in enumerate(labels)), 'Community labels and one-hot differ')
    require(not np.any(assigned[rare.states!='SUPPORTED']), 'Unsupported rare projection cannot use community label')
    codes=np.asarray([index.get(label,-1)for label in labels],dtype=np.int64)
    base=np.concatenate((common,rare.coordinates,rare.state_points),axis=1)
    result=design.assemble(base,codes,len(categories),left,right,controls,arm=arm,max_output_bytes=max_output_bytes)
    return dict(result,common_axes=common.shape[1],rare_axes=rare.coordinates.shape[1],rare_state_axes=rare.state_points.shape[1])
