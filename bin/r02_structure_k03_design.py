#!/usr/bin/env python3
"""Shared K03 A/B construction from a frozen base and categorical memberships.

Integer codes are storage keys only: they enter the predictor through one-hot
or equality, never as ordinal covariates. Unknown codes are exactly -1.
"""
import hashlib
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


def assemble(base_points,codes,n_categories,left,right,controls,*,arm,max_output_bytes):
    base=np.asarray(base_points,dtype=float);codes=np.asarray(codes)
    left=np.asarray(left);right=np.asarray(right);controls=np.asarray(controls,dtype=float)
    require(base.ndim==2 and np.isfinite(base).all(),'Finite shared base points required')
    n=len(base)
    require(codes.shape==(n,)and codes.dtype.kind in 'iu'and type(n_categories)is int and n_categories>0
        and np.all(codes>=-1)and np.all(codes<n_categories),'Exact categorical codes and inventory required')
    require(arm in {'CONTINUOUS','COMMUNITY_A','COMMUNITY_B'},'Explicit K03 arm required')
    require(left.ndim==right.ndim==1 and left.shape==right.shape and len(left)>0
        and left.dtype.kind in 'iu'and right.dtype.kind in 'iu'and np.all(left>=0)and np.all(right>=0)
        and np.all(left<n)and np.all(right<n)and np.all(left!=right),'Explicit valid pair keys required')
    require(controls.ndim==2 and len(controls)==len(left)and np.isfinite(controls).all(),'Finite aligned baseline controls required')
    width=base.shape[1]+1+(n_categories if arm=='COMMUNITY_B'else 0)
    extra_width=controls.shape[1]+int(arm=='COMMUNITY_A')
    require(type(max_output_bytes)is int and 8*(n*width+len(left)*extra_width)<=max_output_bytes,
        'K03 feature allocation bound exceeded')
    assigned=codes>=0;unknown=(~assigned).astype(float)[:,None]
    parts=[base,unknown]
    if arm=='COMMUNITY_B':
        hot=np.zeros((n,n_categories),dtype=float);rows=np.flatnonzero(assigned);hot[rows,codes[rows]]=1.;parts.append(hot)
    points=np.concatenate(parts,axis=1);extra=controls.copy()
    if arm=='COMMUNITY_A':
        same=(assigned[left]&assigned[right]&(codes[left]==codes[right])).astype(float)
        extra=np.column_stack((extra,same))
    return dict(points=points,extra=extra,individual_axes=width,pair_columns=3*width+extra_width,
        arm=arm,community_axes=n_categories if arm=='COMMUNITY_B'else 0,community_availability_axes=1,
        neutral_community_code='ALL_ZERO_WITH_SHARED_UNASSIGNED_INDICATOR',
        exact_same_controls_required_for_baseline_reuse=True,target_outcomes_read=False)


def baseline_identity(base_manifest_sha256,codes):
    """Only identical prepared data and all-person availability share a task.

    The base manifest binds both role masks, outcomes, FIT weights and paired
    controls. Availability is person-level, so equality also binds all pairs.
    """
    require(isinstance(base_manifest_sha256,str)and len(base_manifest_sha256)==64
        and all(c in '0123456789abcdef'for c in base_manifest_sha256),'Base manifest hash required')
    codes=np.asarray(codes);require(codes.ndim==1 and codes.dtype.kind in 'iu'and np.all(codes>=-1),'Membership codes required')
    h=hashlib.sha256();h.update(base_manifest_sha256.encode());h.update((codes>=0).tobytes())
    return h.hexdigest()
