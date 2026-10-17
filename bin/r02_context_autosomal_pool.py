#!/usr/bin/env python3
"""Exact 22-autosome pooling for the existing structure DeepSets path.

This component accepts already encoded common values and rare ``event_values``.
It can expose their exact autosomal sufficient sums before ``person_rho`` so a
trainer may recompute one tile at a time during backward instead of retaining
22 chromosome graphs. It does not enable AUTOSOMAL_22 admission, choose blocks,
average losses, or access EVALUATE.
"""
from dataclasses import dataclass
import torch

AUTOSOMES = tuple(range(1, 23))

def require(condition, message):
    if not condition:
        raise ValueError(message)

@dataclass(frozen=True)
class AutosomalPoolResult:
    embedding: torch.Tensor
    common_mean: torch.Tensor
    rare_embedding: torch.Tensor
    common_valid_loci: torch.Tensor
    common_missing_loci: torch.Tensor
    rare_valid_loci: torch.Tensor
    rare_missing_loci: torch.Tensor
    rare_valid_zero_dose_loci: torch.Tensor
    chromosomes: tuple

@dataclass(frozen=True)
class AutosomalSufficientStatistics:
    common_sum: torch.Tensor
    rare_sum: torch.Tensor
    common_valid_loci: torch.Tensor
    common_missing_loci: torch.Tensor
    rare_valid_loci: torch.Tensor
    rare_missing_loci: torch.Tensor
    rare_valid_zero_dose_loci: torch.Tensor
    chromosomes: tuple

def finalize_statistics(statistics, person_rho):
    """Apply the downstream person map once to exact autosomal sums."""
    denominator=statistics.common_valid_loci.clamp_min(1).to(statistics.common_sum.dtype)[:,None]
    common=statistics.common_sum/denominator
    rare=person_rho(statistics.rare_sum)
    rare=torch.where((statistics.rare_valid_loci>0)[:,None],rare,0)
    return AutosomalPoolResult(common+rare,common,rare,
        statistics.common_valid_loci,statistics.common_missing_loci,
        statistics.rare_valid_loci,statistics.rare_missing_loci,
        statistics.rare_valid_zero_dose_loci,statistics.chromosomes)

class AutosomalPool:
    """Accumulate exact per-person autosomal sufficient statistics.

    Person IDs and roles are immutable ordered identities.  Each chromosome is
    admitted once.  Missingness is ``core & ~valid``; an observed dosage of zero
    remains in the rare sum because inclusion depends only on ``core & valid``.
    """
    def __init__(self, person_ids, roles, chromosomes=AUTOSOMES):
        self.person_ids=tuple(str(x) for x in person_ids);self.roles=tuple(str(x) for x in roles)
        self.expected=tuple(chromosomes);self.seen=[];self.finalized=False;self.current=None;self.next_tile=0;self.last_core_end=None
        require(self.person_ids and len(self.person_ids)==len(set(self.person_ids)),"unique nonempty ordered person IDs required")
        require(len(self.roles)==len(self.person_ids) and all(self.roles),"one nonempty frozen role per person required")
        require(self.expected==AUTOSOMES,"exact ordered chromosomes 1..22 required")
        self.common_sum=self.rare_sum=None
        self.common_valid=self.common_missing=self.rare_valid=self.rare_missing=self.rare_zero=None

    def add(self, chromosome, person_ids, roles, common_values, common_core, common_valid,
            rare_values, rare_core, rare_valid, rare_dosage, tile_index=0, last_tile=True, core_bp=None):
        require(not self.finalized,"pool already finalized")
        require(type(chromosome)is int and chromosome in self.expected and chromosome not in self.seen,
                "chromosome must be an unfinished autosome")
        require(type(tile_index)is int and type(last_tile)is bool,"tile_index integer and last_tile boolean required")
        if self.current is None:
            require(chromosome==self.expected[len(self.seen)]and tile_index==0,"autosomes/tiles must arrive in exact order")
            self.current=chromosome;self.next_tile=0;self.last_core_end=None
        require(chromosome==self.current and tile_index==self.next_tile,"autosome tile order differs")
        if core_bp is not None:
            require(type(core_bp)in(tuple,list)and len(core_bp)==2 and all(type(x)is int for x in core_bp)
                    and core_bp[0]<=core_bp[1]and(self.last_core_end is None or core_bp[0]==self.last_core_end+1),
                    "tile cores must be contiguous and nonoverlapping")
            self.last_core_end=core_bp[1]
        require(core_bp is not None or(last_tile and self.last_core_end is None),
                "every tile of a multi-tile chromosome requires an explicit core interval")
        require(tuple(map(str,person_ids))==self.person_ids and tuple(map(str,roles))==self.roles,
                "person ID/role order differs across autosomes")
        p=len(self.person_ids)
        for name,values,core,valid in (("common",common_values,common_core,common_valid),("rare",rare_values,rare_core,rare_valid)):
            require(torch.is_tensor(values)and values.ndim==3 and values.shape[0]==p,"%s values shape differs"%name)
            require(torch.is_tensor(core)and torch.is_tensor(valid)and core.dtype==torch.bool and valid.dtype==torch.bool
                    and core.shape==valid.shape==values.shape[:2],"%s core/valid mask differs"%name)
            require(bool(torch.isfinite(values).all()),"%s values must be finite"%name)
        require(common_values.shape[2]==rare_values.shape[2],"common/rare embedding widths differ")
        require(torch.is_tensor(rare_dosage)and rare_dosage.shape==rare_valid.shape,
                "rare dosage shape differs")
        require(bool(torch.isfinite(rare_dosage[rare_valid]).all()) and
                bool(torch.isin(rare_dosage[rare_valid],torch.tensor((0,1,2),device=rare_dosage.device)).all()),
                "observed rare dosage must be 0/1/2")
        cmask=common_core&common_valid;rmask=rare_core&rare_valid
        csum=torch.where(cmask[...,None],common_values,0).sum(1)
        rsum=torch.where(rmask[...,None],rare_values,0).sum(1)
        require(csum.device==rsum.device and csum.dtype==rsum.dtype,"autosomal value device/dtype differs")
        counts=(cmask.sum(1), (common_core&~common_valid).sum(1), rmask.sum(1),
                (rare_core&~rare_valid).sum(1), (rmask&(rare_dosage==0)).sum(1))
        if self.common_sum is None:
            self.common_sum,self.rare_sum=csum,rsum
            self.common_valid,self.common_missing,self.rare_valid,self.rare_missing,self.rare_zero=counts
        else:
            require(csum.shape==self.common_sum.shape and csum.device==self.common_sum.device and
                    csum.dtype==self.common_sum.dtype,"autosomal accumulator shape/device/dtype differs")
            self.common_sum=self.common_sum+csum;self.rare_sum=self.rare_sum+rsum
            self.common_valid=self.common_valid+counts[0];self.common_missing=self.common_missing+counts[1]
            self.rare_valid=self.rare_valid+counts[2];self.rare_missing=self.rare_missing+counts[3];self.rare_zero=self.rare_zero+counts[4]
        self.next_tile+=1
        if last_tile:
            self.seen.append(chromosome);self.current=None;self.next_tile=0;self.last_core_end=None

    def finalize_sums(self):
        require(not self.finalized and self.current is None and tuple(self.seen)==self.expected,"all autosomes 1..22 required exactly once")
        self.finalized=True
        return AutosomalSufficientStatistics(self.common_sum,self.rare_sum,
            self.common_valid,self.common_missing,self.rare_valid,self.rare_missing,
            self.rare_zero,tuple(self.seen))

    def finalize(self, person_rho):
        return finalize_statistics(self.finalize_sums(),person_rho)
