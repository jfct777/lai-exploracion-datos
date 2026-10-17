#!/usr/bin/env python3
"""Bounded FIT-only continuous representations for the K03 comparison.

This numerical component does not select a number of components, learn
communities, score an outcome, or establish that a continuous model is adequate.
It factors a weighted, FIT-centred genotype matrix through its small dual Gram
matrix; held-out people are projected without changing the FIT solution. A
separate numerical rare-graph API exposes the full normalized FIT spectrum and
fixed inductive projection; it is not wired into the production CLI or selection.
"""
from __future__ import annotations

from collections import Counter
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

import r02_structure_discrete as discrete

require = discrete.require
POLICIES = {"INDIVIDUAL_EQUAL", "COMPONENT_EQUAL"}
SCHEMA = "r02_structure_continuous_common_v1"
CODE_FILES = ("r02_structure_continuous.py", *discrete.CODE_FILES)
LOCUS_COLUMNS = ["block_id", "chromosome", "position_bp", "variant_id", "ref", "alt",
                 "position_cm", "fit_called", "fit_an", "fit_alt_af"]
EVENT_FIELDS = {"dosage", "observed", "position_bp", "position_cm", "ref", "alt",
                "counted_allele", "allele_count", "allele_number", "frequency"}
CACHE_ARRAYS = {"fit_indices", "fit_weights", "gram", "cross", "observed_counts"}
CACHE_METADATA = {"schema", "chromosome", "source_sha256", "persons_sha256",
                  "weighting", "n_people", "n_sites"}


class FitRareGraphSpectrum:
    """Numerical instrument for the SAME rare FIT graph, not a K03 selector.

    Edges are (global FIT index, global FIT index, observed Jaccard), in strict
    upper-triangle order, excluding within-component pairs. Every eligible pair
    must be observed, including zeros. Unknowns are rejected, not set to zero.
    This complete-support restriction bounds this API; it does not authorize a
    new scientific cohort filter. Upstream consumers must authenticate territory,
    allele, denominators and the pinned source. A hash alone does not prove them.

    Retain ALL eigenvalues of D^-1/2 A D^-1/2 for diagnosis. Eigenvectors at one
    are trivial connected-component modes. A deterministic basis of these modes
    is exposed separately for BOTH continuous and community arms; their omission
    from the legacy nontrivial scores is not permission to omit connectivity
    information from the comparator. Isolated vertices and numerical zero modes
    cannot be projected. All other modes, including negatives, are exposed
    in ascending eigenvalue order. No predictive rank or sign-based selection is
    made here. Dense cost is O(n_FIT^2) memory and O(n_FIT^3) decomposition.
    """
    def __init__(self, persons, edges, *, source_sha256, max_memory_bytes,
                 affinity_semantics=None, inner_fit_indices=None,
                 inner_project_indices=None):
        roles = discrete.shared.validate_roles(persons)
        if (inner_fit_indices is None) != (inner_project_indices is None):
            raise ValueError("Both inner-FIT and inner-project indices are required together")
        if inner_fit_indices is None:
            fit_indices = np.asarray(roles['FIT'], dtype=np.int64)
            query_indices = tuple(sorted(int(i) for r in ('SELECT', 'EVALUATE') for i in roles[r]))
        else:
            fit_indices = np.asarray(inner_fit_indices, dtype=np.int64)
            query_indices = tuple(int(i) for i in inner_project_indices)
            outer_fit = set(map(int, roles['FIT']))
            require(fit_indices.ndim == 1 and len(fit_indices) > 1
                    and np.all(fit_indices[1:] > fit_indices[:-1])
                    and tuple(sorted(set(query_indices))) == query_indices
                    and set(map(int, fit_indices)).isdisjoint(query_indices)
                    and set(map(int, fit_indices)) | set(query_indices) == outer_fit,
                    "Inner partition must exactly divide frozen outer FIT")
            train_components = {persons[int(i)]['component'] for i in fit_indices}
            project_components = {persons[i]['component'] for i in query_indices}
            require(train_components.isdisjoint(project_components),
                    "A family component crosses inner FIT/project roles")
        default_rare_semantics = affinity_semantics is None
        if affinity_semantics is None:
            affinity_semantics = dict(
                family="R", measure="SOURCE_MINOR_JACCARD",
                units="DIMENSIONLESS_INTERSECTION_OVER_UNION",
                minimum=0.0, maximum=1.0)
        require(isinstance(affinity_semantics, dict)
                and set(affinity_semantics) == {"family", "measure", "units", "minimum", "maximum"}
                and isinstance(affinity_semantics["family"], str) and affinity_semantics["family"]
                and isinstance(affinity_semantics["measure"], str) and affinity_semantics["measure"]
                and isinstance(affinity_semantics["units"], str) and affinity_semantics["units"]
                and type(affinity_semantics["minimum"]) in (int, float)
                and np.isfinite(affinity_semantics["minimum"])
                and (affinity_semantics["maximum"] is None
                     or (type(affinity_semantics["maximum"]) in (int, float)
                         and np.isfinite(affinity_semantics["maximum"])
                         and affinity_semantics["maximum"] >= affinity_semantics["minimum"])),
                "Explicit affinity family, measure, units and finite range required")
        require(affinity_semantics["minimum"] >= 0,
                "Spectral affinity graph requires nonnegative weights")
        require(discrete.valid_hash(source_sha256), "Pinned affinity source required")
        require(type(max_memory_bytes) is int and max_memory_bytes > 0,
                "Explicit positive rare-spectrum memory budget required")
        self.fit_indices = fit_indices
        n = len(self.fit_indices)
        require(n > 1, "At least two FIT people required")
        # Conservative allowance for dense adjacency, eigensolver, diagnostics
        # and single-query workspaces. Caller owns input/decompression memory.
        self.memory_bound = (20*n*n + 128*n)*8
        require(self.memory_bound <= max_memory_bytes, "Affinity-spectrum workspace exceeds memory budget")
        self.persons_sha256 = discrete.materialize.digest(persons)
        self.source_sha256 = source_sha256
        self.affinity_semantics = dict(affinity_semantics)
        self.query_indices = query_indices
        local = {int(i): k for k, i in enumerate(self.fit_indices)}
        components = [persons[i]['component'] for i in self.fit_indices]
        counts = Counter(components)
        expected = n*(n-1)//2 - sum(v*(v-1)//2 for v in counts.values())
        adjacency = np.zeros((n, n), dtype=np.float64)
        previous, seen = None, 0
        for i, j, value in edges:
            require(type(i) is int and type(j) is int and i in local and j in local and i < j,
                    "Affinity edges must be canonical off-diagonal FIT pairs")
            require(previous is None or previous < (i, j), "FIT affinity edges must be sorted and unique")
            previous = (i, j)
            a, b = local[i], local[j]
            require(components[a] != components[b], "Within-component FIT edge forbidden")
            require(type(value) in (int, float) and np.isfinite(value)
                    and value >= affinity_semantics["minimum"]
                    and (affinity_semantics["maximum"] is None
                         or value <= affinity_semantics["maximum"]),
                    "Unknown or out-of-contract affinity; cannot substitute zero")
            adjacency[a, b] = adjacency[b, a] = value
            seen += 1
        require(seen == expected, "Complete observed eligible FIT-pair support required")
        self.adjacency = adjacency
        self.degree = adjacency.sum(axis=1)
        self.active_fit = self.degree > 0
        require(np.any(self.active_fit), "FIT affinity graph has no positive-degree support")
        self.inverse_sqrt_degree = np.zeros(n, dtype=np.float64)
        self.inverse_sqrt_degree[self.active_fit] = 1/np.sqrt(self.degree[self.active_fit])
        self.normalized = adjacency*self.inverse_sqrt_degree[:, None]*self.inverse_sqrt_degree[None, :]
        _, self.connected_component = connected_components(adjacency, directed=False)
        values, vectors = eigh(self.normalized, check_finite=False, driver='evd')
        self.numerical_tolerance = 8*np.finfo(np.float64).eps*n
        require(np.all(np.abs(values) <= 1+self.numerical_tolerance), "Normalized affinity spectrum outside [-1,1]")
        trivial = np.abs(values-1) <= self.numerical_tolerance
        zero = np.abs(values) <= self.numerical_tolerance
        require(int(trivial.sum()) == len(set(self.connected_component[self.active_fit])),
                "Trivial spectral multiplicity differs from positive-degree components")
        for column in range(n):
            pivot = np.argmax(np.abs(vectors[:, column]))
            if vectors[pivot, column] < 0:
                vectors[:, column] *= -1
        self.eigenvalues, self.eigenvectors = values, vectors
        self.trivial_modes, self.zero_modes = trivial, zero
        self.projectable_modes = ~trivial & ~zero
        self.fit_scores = vectors[:, self.projectable_modes].copy()
        self.fit_scores[~self.active_fit] = np.nan
        # Explicit component basis avoids arbitrary rotations of the repeated
        # eigenvalue-one eigenspace. Column order follows minimum global FIT
        # index; each column is sqrt(d_i / volume(C)) on that component.
        component_groups = sorted((np.flatnonzero((self.connected_component == label) & self.active_fit)
                                   for label in set(self.connected_component[self.active_fit])),
                                  key=lambda group: int(self.fit_indices[group[0]]))
        self.shared_component_ids = np.asarray([int(self.fit_indices[group[0]]) for group in component_groups], dtype=np.int64)
        self.shared_component_volumes = np.asarray([self.degree[group].sum() for group in component_groups])
        self.shared_component_basis = np.zeros((n, len(component_groups)), dtype=np.float64)
        for column, group in enumerate(component_groups):
            self.shared_component_basis[group, column] = np.sqrt(self.degree[group]/self.shared_component_volumes[column])
        require(np.allclose(self.normalized @ self.shared_component_basis, self.shared_component_basis,
                            rtol=0, atol=self.numerical_tolerance),
                'Deterministic connected-component basis is not an eigenvalue-one basis')
        self.shared_component_fit_scores = self.shared_component_basis.copy()
        self.shared_component_fit_scores[~self.active_fit] = np.nan
        self.shared_fit_states = np.where(self.active_fit, 'POSITIVE_DEGREE_FIT', 'ZERO_DEGREE_FIT')
        graph_identity = dict(
            source_sha256=source_sha256, persons_sha256=self.persons_sha256,
            fit_indices=self.fit_indices.tolist(), adjacency=adjacency.tolist(),
            same_component_fit_edges='EXCLUDE', missing_edge_policy='UNKNOWN_NOT_ZERO')
        # Preserve the exact historical R graph digest. New families bind their
        # explicit semantics so equal numbers with different units cannot alias.
        if not default_rare_semantics:
            graph_identity['affinity_semantics'] = self.affinity_semantics
        self.graph_sha256 = discrete.materialize.digest(graph_identity)
        self._arrays = ('fit_indices', 'adjacency', 'degree', 'active_fit', 'inverse_sqrt_degree',
                        'normalized', 'connected_component', 'eigenvalues', 'eigenvectors',
                        'trivial_modes', 'zero_modes', 'projectable_modes', 'fit_scores',
                        'shared_component_ids', 'shared_component_volumes', 'shared_component_basis',
                        'shared_component_fit_scores', 'shared_fit_states')
        self._pins = {}
        for name in self._arrays:
            value = getattr(self, name)
            value.setflags(write=False)
            self._pins[name] = hashlib.sha256(value.tobytes()).hexdigest()

    def verify(self, source_sha256):
        require(source_sha256 == self.source_sha256, "Projection source differs from frozen FIT source")
        require(all(hashlib.sha256(getattr(self, name).tobytes()).hexdigest() == digest
                    for name, digest in self._pins.items()), "Frozen FIT affinity representation changed")

    def _query_row(self, query_index, observations, source_sha256):
        self.verify(source_sha256)
        require(type(query_index) is int and query_index in self.query_indices,
                "Projection row must belong to a frozen held-out role")
        require(isinstance(observations, dict) and
                all(type(i) is int and i in self.fit_indices for i in observations),
                "Query affinity references must be FIT only")
        row = np.zeros(len(self.fit_indices), dtype=np.float64)
        observed = np.zeros(len(row), dtype=bool)
        for k, i in enumerate(self.fit_indices):
            value = observations.get(int(i))
            if value is not None:
                semantics = getattr(self, "affinity_semantics", dict(minimum=0.0, maximum=1.0))
                require(type(value) in (int, float) and np.isfinite(value)
                        and value >= semantics["minimum"]
                        and (semantics["maximum"] is None or value <= semantics["maximum"]),
                        "Invalid observed query affinity")
                row[k], observed[k] = value, True
        status = ('INCOMPLETE_FIT_SUPPORT' if not observed.all() else
                  'AFFINITY_TO_ZERO_DEGREE_FIT' if np.any(row[~self.active_fit] > 0) else
                  'ZERO_QUERY_DEGREE' if row.sum() == 0 else 'PROJECTED')
        return row, observed, status

    def _shared_projection(self, row, status):
        scores = np.full(len(self.shared_component_ids), np.nan)
        if status == 'PROJECTED':
            # Nyström with lambda=1 and the deterministic basis above:
            # sum_j in C A_xj / sqrt(d_x * volume(C)). FIT degrees stay frozen.
            scores = ((row*self.inverse_sqrt_degree) @ self.shared_component_basis / np.sqrt(row.sum()))
            require(np.all(np.isfinite(scores)), 'Nonfinite shared component projection')
        return scores

    def project_shared_components(self, query_index, observations, *, source_sha256):
        """The identical connectivity features/states for both predictive arms.

        This is not community assignment. It carries the trivial FIT connected
        components, even when there are no nontrivial spectral directions. No
        query-to-query edge, outcome or updated FIT degree is used.
        """
        row, observed, status = self._query_row(query_index, observations, source_sha256)
        return dict(shared_component_scores=self._shared_projection(row, status), shared_state=status,
                    component_ids=self.shared_component_ids.copy(), query_index=query_index,
                    observed_fit_count=int(observed.sum()), total_fit_count=len(row),
                    query_degree=float(row.sum()) if observed.all() else None,
                    graph_sha256=self.graph_sha256, source_sha256=self.source_sha256,
                    target_read=False, biological_validation=False)

    def project(self, query_index, observations, *, source_sha256):
        """Nyström row, retaining the legacy nontrivial API plus shared modes.

        Unknown observations stay unknown. Query affinity to isolated FIT rows
        makes normalization undefined and rejects the row in both arms. Shared
        component scores are separately explicit, never inferred from arbitrary
        eigenvectors of the repeated eigenvalue-one subspace.
        """
        row, observed, shared_status = self._query_row(query_index, observations, source_sha256)
        status = ('NO_PROJECTABLE_NONTRIVIAL_MODES' if shared_status == 'PROJECTED'
                  and not np.any(self.projectable_modes) else shared_status)
        scores = np.full(int(self.projectable_modes.sum()), np.nan)
        if status == 'PROJECTED':
            normalized_row = row*self.inverse_sqrt_degree/np.sqrt(row.sum())
            scores = (normalized_row @ self.eigenvectors[:, self.projectable_modes]
                      / self.eigenvalues[self.projectable_modes])
            require(np.all(np.isfinite(scores)), "Nonfinite affinity inductive projection")
        return dict(scores=scores, status=status, query_index=query_index,
                    shared_component_scores=self._shared_projection(row, shared_status),
                    shared_state=shared_status, component_ids=self.shared_component_ids.copy(),
                    observed_fit_count=int(observed.sum()), total_fit_count=len(row),
                    query_degree=float(row.sum()) if observed.all() else None,
                    graph_sha256=self.graph_sha256, source_sha256=self.source_sha256,
                    target_read=False, biological_validation=False,
                    capacity_selected=False, complete_support_is_api_limit=True)


# The historical public name remains valid for frozen R jobs.  New family
# adapters use the semantic name and must pass ``affinity_semantics`` explicitly.
FitAffinityGraphSpectrum = FitRareGraphSpectrum


def fit_weights(persons, fit_indices, policy):
    """Unit-sum FIT weights; a component is not a certified biological family.

    COMPONENT_EQUAL gives each recorded kinship component equal total mass.
    It is an explicit sensitivity, not a correction which proves independence.
    Role validation prevents one component spanning FIT and a held-out role.
    """
    roles = discrete.shared.validate_roles(persons)
    fit_indices = np.asarray(fit_indices)
    require(np.array_equal(fit_indices, roles["FIT"]), "FIT roster/order differs")
    require(policy in POLICIES, "Explicit FIT weighting policy required")
    if policy == "INDIVIDUAL_EQUAL":
        raw = np.ones(len(fit_indices), dtype=np.float64)
    else:
        counts = Counter(persons[i]["component"] for i in fit_indices)
        raw = np.asarray([1.0/counts[persons[i]["component"]] for i in fit_indices])
    return raw/raw.sum()


def standardized_common(dosage, observed, frequency, fit_indices):
    """Return dosage-standardized columns, retaining mean-imputation semantics.

    Frequency is the already selected FIT ALT frequency. It is verified against
    observed FIT dosages, never recomputed using query people. Missing genotype
    contributes z=0 before centring, i.e. an imputed dose of 2*p_FIT, not REF.
    Selected sites/geometry/provenance are responsibilities of the caller.
    """
    dose, mask, freq = np.asarray(dosage), np.asarray(observed), np.asarray(frequency)
    fit = np.asarray(fit_indices)
    require(dose.ndim == 2 and dose.dtype == np.uint8 and mask.dtype == np.bool_
            and mask.shape == dose.shape, "Diploid uint8 dosage and boolean mask required")
    require(freq.dtype.kind == "f" and freq.shape == (dose.shape[1],)
            and np.all(np.isfinite(freq)) and np.all((freq > 0) & (freq < 1)),
            "Polymorphic finite FIT frequencies required")
    require(fit.ndim == 1 and fit.dtype.kind in "iu" and len(fit) > 1
            and np.array_equal(fit, np.unique(fit)) and fit[-1] < dose.shape[0]
            and fit[0] >= 0, "Sorted unique FIT indices required")
    require(np.all(dose[mask] <= 2) and np.all(dose[~mask] == 255),
            "Missing sentinel or complete diploid genotype contract differs")
    fit_calls = mask[fit].sum(axis=0)
    require(np.all(fit_calls > 0), "No FIT observation at selected common locus")
    fit_sum = np.where(mask[fit], dose[fit], 0).sum(axis=0, dtype=np.float64)
    require(np.allclose(fit_sum/(2*fit_calls), freq, rtol=0, atol=1e-12),
            "Frequency not equal to observed FIT allele count/number")
    result = np.zeros(dose.shape, dtype=np.float64)
    np.subtract(dose, 2*freq, out=result, where=mask)
    np.divide(result, np.sqrt(2*freq*(1-freq)), out=result)
    return result


class DualContinuousAccumulator:
    """Stream columns without concatenating a genome-sized dense matrix.

    For z=(g-2p)/sqrt(2p(1-p)) with mean-imputed missing entries, subtract its
    FIT weighted column mean mu. Let X=z-mu and F=sqrt(w)*X_FIT. Accumulate
    G=F*F.T and H=X*F.T. Eigenpairs G=U diag(lambda) U.T give projected scores
    H*U/sqrt(lambda), exactly the right-singular-vector projection X*V.

    Query genotypes enter H only. They never enter means, weights, G, U, or
    eigenvalue/rank decisions. This is not PC-AiR and is not a replacement for
    PC-AiR/PC-Relate kinship estimation. Weighting may limit family domination
    but does not establish independence or ancestry interpretation.
    """
    def __init__(self, persons, *, weighting, max_chunk_sites, max_memory_bytes):
        roles = discrete.shared.validate_roles(persons)
        self.fit = np.asarray(roles["FIT"], dtype=np.int64)
        self.n = len(persons)
        require(type(max_chunk_sites) is int and max_chunk_sites > 0
                and type(max_memory_bytes) is int and max_memory_bytes > 0,
                "Positive explicit memory and chunk bounds required")
        self.max_chunk_sites = max_chunk_sites
        # Includes eigensolver/scores workspaces, row arrays and standardized
        # chunks. A caller must add its source decompression/prefetch buffers.
        self.memory_bound = (12*len(self.fit)**2 + 8*self.n*len(self.fit)
                             + 10*self.n*max_chunk_sites + 64*self.n)*8
        require(self.memory_bound <= max_memory_bytes, "Continuous workspace exceeds memory budget")
        self.weighting = weighting
        self.persons_sha256 = discrete.materialize.digest(persons)
        self.weights = fit_weights(persons, self.fit, weighting)
        self.sqrt_weights = np.sqrt(self.weights)
        self.gram = np.zeros((len(self.fit), len(self.fit)), dtype=np.float64)
        self.cross = np.zeros((self.n, len(self.fit)), dtype=np.float64)
        self.observed_counts = np.zeros(self.n, dtype=np.int64)
        self.n_sites = 0
        self.closed = False

    def chromosome_statistics(self, chromosome, source_sha256):
        """Freeze additive statistics, not independently computed PC scores.

        source_sha256 must bind this chromosome's genotype chunks and FIT marker
        selection. The file-level reader authenticates that identity separately.
        The same FIT roster, weighting and scaling permit sums across autosomes.
        """
        require(not self.closed and self.n_sites > 0 and str(chromosome) in {str(i) for i in range(1,23)}
                and discrete.valid_hash(source_sha256), 'Explicit chromosome source identity required')
        return dict(schema='r02_common_gram_chromosome_v1',chromosome=str(chromosome),
                    source_sha256=source_sha256,persons_sha256=self.persons_sha256,
                    weighting=self.weighting,n_people=self.n,n_sites=self.n_sites,
                    fit_indices=self.fit.copy(),fit_weights=self.weights.copy(),
                    gram=self.gram.copy(),cross=self.cross.copy(),observed_counts=self.observed_counts.copy())

    def add(self, dosage, observed, frequency):
        require(not self.closed, "Accumulator already finalized")
        require(np.ndim(dosage) == 2 and np.shape(dosage)[0] == self.n
                and 0 < np.shape(dosage)[1] <= self.max_chunk_sites,
                "Chunk person count or site budget differs")
        x = standardized_common(dosage, observed, frequency, self.fit)
        mean = self.weights @ x[self.fit]
        x -= mean
        f = self.sqrt_weights[:, None] * x[self.fit]
        self.gram += f @ f.T
        self.cross += x @ f.T
        self.observed_counts += np.asarray(observed).sum(axis=1, dtype=np.int64)
        self.n_sites += x.shape[1]

    def finalize(self):
        """Return all numerically identifiable directions; no scientific rank.

        The tolerance scales with IEEE float64 precision, matrix dimension and
        leading eigenvalue. It only excludes numerical zero directions. Choosing
        predictive capacity is deferred to preregistered FIT/SELECT selection.
        """
        require(not self.closed and self.n_sites > 0, "No data or already finalized")
        self.closed = True
        require(np.all(np.isfinite(self.gram)) and np.all(np.isfinite(self.cross)),
                "Nonfinite continuous Gram/cross product")
        values, vectors = eigh(self.gram, check_finite=False, driver="evd")
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        scale = max(float(values[0]), np.finfo(np.float64).tiny)
        tolerance = np.finfo(np.float64).eps * len(self.fit) * scale
        require(values[-1] >= -8*tolerance, "Gram matrix has a material negative eigenvalue")
        selected = values > tolerance
        values, vectors = values[selected], vectors[:, selected]
        require(len(values) > 0, "No numerically identifiable continuous direction")
        # Fix eigenvector signs for reproducible files. Rotations in a degenerate
        # eigenspace remain possible; rank views cannot cut through that space.
        for column in range(vectors.shape[1]):
            pivot = np.argmax(np.abs(vectors[:, column]))
            if vectors[pivot, column] < 0:
                vectors[:, column] *= -1
        dual_loadings = vectors / np.sqrt(values)
        scores = self.cross @ dual_loadings
        require(np.all(np.isfinite(scores)), "Nonfinite projected score")
        return dict(scores=scores, dual_loadings=dual_loadings, eigenvalues=values,
                    numerical_tolerance=tolerance, fit_weights=self.weights.copy(),
                    fit_indices=self.fit.copy(), observed_counts=self.observed_counts.copy(),
                    n_sites=self.n_sites, weighting=self.weighting,
                    mean_imputation="FROZEN_FIT_ALT_FREQUENCY_NOT_REF",
                    fit_centering="WEIGHTED_FIT_MEAN_AFTER_FIT_FREQUENCY_SCALING",
                    source_geometry_verified=False, biological_validation=False,
                    method="WEIGHTED_FIT_DUAL_SVD_WITH_INDUCTIVE_PROJECTION")


def rank_view(representation, rank):
    """Explicit candidate rank; a numerical tie cannot be cut arbitrarily."""
    vals = representation["eigenvalues"]
    require(type(rank) is int and 0 < rank <= len(vals), "Rank outside identifiable spectrum")
    if rank < len(vals):
        require(vals[rank-1] - vals[rank] > 8*representation["numerical_tolerance"],
                "Rank cuts a numerically degenerate eigenspace")
    return representation["scores"][:, :rank]


def merge_chromosome_statistics(persons, records, *, weighting, included_chromosomes,
                               expected_sources, max_memory_bytes):
    """Sum authenticated per-chromosome G/H once, then perform ONE dual SVD.

    Each chromosome contributes one authenticated selection. The caller must
    prove its exact allowed intervals/LD list; a whole-chromosome cache cannot
    stand in for a differently pruned block-plus-margin cache. Records may be
    a generator to bound memory.
    """
    expected=list(included_chromosomes)
    require(expected and expected==sorted(set(expected),key=int)
            and set(expected)==set(expected_sources)
            and all(v in {str(i) for i in range(1,23)} for v in expected)
            and all(discrete.valid_hash(v) for v in expected_sources.values()),
            'Explicit unique chromosome/source inventory required')
    accumulator=DualContinuousAccumulator(persons,weighting=weighting,max_chunk_sites=1,
                                          max_memory_bytes=max_memory_bytes)
    seen=set()
    fields={'schema','chromosome','source_sha256','persons_sha256','weighting','n_people','n_sites',
            'fit_indices','fit_weights','gram','cross','observed_counts'}
    for record in records:
        require(set(record)==fields and record['schema']=='r02_common_gram_chromosome_v1', 'Gram-cache schema differs')
        chrom=record['chromosome']
        require(chrom in expected_sources and chrom not in seen and record['source_sha256']==expected_sources[chrom],
                'Missing, repeated, excluded or changed chromosome statistics')
        require(record['persons_sha256']==accumulator.persons_sha256 and record['weighting']==weighting
                and record['n_people']==len(persons) and np.array_equal(record['fit_indices'],accumulator.fit)
                and np.array_equal(record['fit_weights'],accumulator.weights), 'Gram cache FIT/weight/cohort differs')
        nsites=record['n_sites']
        require(type(nsites) is int and nsites>0 and nsites <= np.iinfo(np.int64).max-accumulator.n_sites,
                'Invalid site count or overflow in Gram cache')
        gram,cross,observed=(np.asarray(record[k]) for k in ('gram','cross','observed_counts'))
        require(gram.dtype==cross.dtype==np.float64 and gram.shape==accumulator.gram.shape
                and cross.shape==accumulator.cross.shape and np.all(np.isfinite(gram)) and np.all(np.isfinite(cross)),
                'Gram-cache array shape/dtype/finite value differs')
        require(np.allclose(gram,gram.T,rtol=0,atol=np.finfo(float).eps*max(1,float(np.max(np.abs(gram))))*len(gram))
                and observed.dtype==np.int64 and observed.shape==(len(persons),)
                and np.all((observed>=0)&(observed<=nsites)), 'Gram symmetry/call denominator differs')
        accumulator.gram+=gram;accumulator.cross+=cross
        accumulator.observed_counts+=observed;accumulator.n_sites+=nsites;seen.add(chrom)
    require(seen==set(expected),'Incomplete chromosome statistics; no partial-genome result')
    result=accumulator.finalize()
    result.update(included_chromosomes=expected,source_sha256=expected_sources,
                  aggregation='SUM_CHROMOSOME_GRAM_AND_CROSS_BEFORE_SINGLE_DECOMPOSITION')
    return result


def save_chromosome_statistics(directory, record, *, max_output_bytes, min_free_bytes):
    """Persist reusable G/H without pickle; commit the manifest last.

    An incomplete directory has no valid manifest and cannot be reused. Existing
    directories are never overwritten. The consumer must supply the manifest
    hash obtained from the parent execution, then verify cohort/source identities
    through ``merge_chromosome_statistics`` before using its matrices.
    """
    directory = Path(directory)
    require(not directory.exists(), "Chromosome cache exists; preserve its data")
    require(type(max_output_bytes) is int and max_output_bytes > 0
            and type(min_free_bytes) is int and min_free_bytes > 0,
            "Positive cache output and disk-reserve bounds required")
    require(set(record) == CACHE_ARRAYS | CACHE_METADATA
            and record['schema'] == 'r02_common_gram_chromosome_v1', "Cache fields differ")
    arrays = {k: np.asarray(record[k]) for k in CACHE_ARRAYS}
    require(all(a.dtype.kind in 'if' for a in arrays.values()), "Object/non-numeric cache array prohibited")
    estimated_bytes = sum(a.nbytes for a in arrays.values()) + 16384
    require(estimated_bytes <= max_output_bytes
            and shutil.disk_usage(directory.parent).free >= max_output_bytes + min_free_bytes,
            "Cache exceeds output bound or disk reserve")
    directory.mkdir(mode=0o700)
    path = directory/'statistics.private.npz'
    with path.open('xb') as handle:
        np.savez(handle, **arrays)
    require(path.stat().st_size <= max_output_bytes, "Cache output bound exceeded")
    manifest = {k: record[k] for k in CACHE_METADATA}
    manifest.update(arrays_file=path.name, arrays_sha256=discrete.sha(path),
                    arrays_bytes=path.stat().st_size,
                    array_contract={k: dict(shape=list(a.shape), dtype=a.dtype.str)
                                    for k, a in arrays.items()})
    discrete.materialize.write_json(directory/'manifest.json', manifest)
    return discrete.sha(directory/'manifest.json')


def load_chromosome_statistics(directory, manifest_sha256, *, max_input_bytes):
    """Authenticate an immutable numeric cache before allocating its arrays."""
    directory = Path(directory).resolve()
    require(type(max_input_bytes) is int and max_input_bytes > 0, "Positive cache input bound required")
    manifest_path = pinned(directory/'manifest.json', manifest_sha256, 2**20)
    manifest = discrete.load_json(manifest_path)
    require(set(manifest) == CACHE_METADATA | {'arrays_file', 'arrays_sha256', 'arrays_bytes', 'array_contract'}
            and manifest['arrays_file'] == 'statistics.private.npz'
            and set(manifest['array_contract']) == CACHE_ARRAYS, "Cache manifest schema differs")
    path = pinned(directory/manifest['arrays_file'], manifest['arrays_sha256'], max_input_bytes)
    require(path.stat().st_size == manifest['arrays_bytes'], "Cache byte count differs")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) == len(CACHE_ARRAYS)
                and {m.filename for m in members} == {k+'.npy' for k in CACHE_ARRAYS}
                and sum(m.file_size for m in members) <= max_input_bytes,
                "Cache inventory or decompression bound differs")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in CACHE_ARRAYS}
    require(all(a.dtype.kind in 'if' and
                manifest['array_contract'][k] == dict(shape=list(a.shape), dtype=a.dtype.str)
                for k, a in arrays.items()), "Cache dtype/shape differs")
    require(discrete.sha(path) == manifest['arrays_sha256']
            and discrete.sha(manifest_path) == manifest_sha256, "Cache changed while being read")
    return {**{k: manifest[k] for k in CACHE_METADATA}, **arrays}


def implementation_hashes():
    return {name: discrete.sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def pinned(path, digest, limit):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and discrete.valid_hash(digest)
            and path.stat().st_size <= limit, "Missing/oversized absolute pinned source")
    require(discrete.sha(path) == digest, "Source SHA256 changed: " + str(path))
    return path


def accumulate_selected_chunks(persons, fit_indices, selected, chunks, accumulator,
                               *, min_fit_called):
    """Join authenticated selected loci to original genotype chunks exactly.

    Both existing source-interval selections and materialized feature-interval
    selections use this reader. The caller authenticates their different
    producer manifests and proves allowed territory before calling it.
    """
    seen={c:set() for c in selected}
    for chrom,path,desc in chunks:
        with np.load(path,allow_pickle=False) as data:
            require(set(data.files)==EVENT_FIELDS, 'Genotype chunk inventory differs')
            pos=data['position_bp']
            require(pos.dtype==np.int64 and pos.shape==(desc['n_sites'],)
                    and np.all(np.diff(pos)>0) and pos[0]==desc['start_bp'] and pos[-1]==desc['end_bp'],
                    'Genotype positions/order differ')
            indices=np.asarray([i for i,p in enumerate(pos) if int(p) in selected[chrom]],dtype=int)
            require(len(indices)>0, 'Unexpected empty selected chunk')
            require(data['dosage'].shape==data['observed'].shape==(len(persons),desc['n_sites']),
                    'Original genotype shape differs from roster/chunk')
            require(data['dosage'].dtype==np.uint8 and data['observed'].dtype==np.bool_,
                    'Original genotype dtype/mask differs')
            require(all(data[k].shape==(desc['n_sites'],) for k in EVENT_FIELDS-{'dosage','observed'}),
                    'Locus array shape differs')
            for i in indices:
                p=int(pos[i]);row=selected[chrom][p]
                require(p not in seen[chrom], 'Selected locus seen twice')
                require(int(data['ref'][i])=='ACGT'.index(row['ref'])
                        and int(data['alt'][i])=='ACGT'.index(row['alt'])
                        and int(data['counted_allele'][i])==1
                        and abs(float(data['frequency'][i])-float(row['fit_alt_af']))<=1e-12,
                        'Selected genotype allele/frequency differs from FIT LD catalogue')
                calls=data['observed'][fit_indices,i]
                called=int(calls.sum());ac=int(data['dosage'][fit_indices,i][calls].sum())
                require(called>=min_fit_called and int(data['allele_number'][i])==2*called
                        and int(data['allele_count'][i])==ac,
                        'Selected FIT allele count/called denominator differs')
                if 'fit_called' in row:
                    require(called==int(row['fit_called']) and 2*called==int(row['fit_an']),
                            'Selected FIT catalogue call denominator differs')
                seen[chrom].add(p)
            accumulator.add(data['dosage'][:,indices],data['observed'][:,indices],data['frequency'][indices])
    require(all(seen[c]==set(selected[c]) for c in selected),
            'FIT retained loci missing from genotype store; no silent support restriction')


class CommonView:
    """Read actual event-cache chunks plus source-interval FIT LD selections.

    This first adapter deliberately accepts the already planned whole-chromosome
    sensitivity only. A chromosome source's LD list cannot be relabelled as the
    target-specific block-plus-margin LD list. Frozen producer hashes are bound
    to the request, rather than compared with the latest repository source.
    """
    def __init__(self, bundle, request, request_sha256):
        self.request_path = pinned(Path(request).resolve(), request_sha256, 2**20)
        self.request_sha256 = request_sha256
        self.r = r = discrete.load_json(self.request_path)
        require(set(r) == {"schema", "evidence_scope", "source_contract_sha256", "implementation_sha256",
                "block_id", "exclusion_scope", "selection_scope", "weighting", "chromosomes", "resources"}
                and r["schema"] == SCHEMA, "Exact common-continuous request fields required")
        require(r["evidence_scope"] in {"DNABR_R02", "SYNTHETIC_INSTRUMENT_ONLY"}, "Explicit evidence scope required")
        require(r["implementation_sha256"] == implementation_hashes(), "Continuous consumer code changed")
        require(r["exclusion_scope"] == "WHOLE_CHROMOSOME"
                and r["selection_scope"] == "PER_SOURCE_INTERVAL_FIT_LD",
                "Source-interval LD lists support only declared whole-chromosome sensitivity")
        self.limits = limits = r["resources"]
        require(set(limits) == {"max_metadata_bytes", "max_input_bytes", "max_chunk_bytes", "max_chunk_sites",
                "max_selected_sites", "max_memory_bytes", "max_output_bytes", "min_free_bytes"}
                and all(type(v) is int and v > 0 for v in limits.values()), "Positive explicit resource bounds required")
        self.bundle = Path(bundle).resolve()
        pinned(self.bundle/"materialize.json", r["source_contract_sha256"], limits["max_metadata_bytes"])
        self.source, self.persons, self.roles, blocks, _ = discrete.materialize.load_context(self.bundle)
        require(self.source["roles_state"] == "EXPLICITLY_FROZEN", "Frozen roles required")
        if r["evidence_scope"] == "DNABR_R02":
            require(len(self.persons) == 2619 and {k:len(v) for k,v in self.roles.items()} ==
                    {"FIT":1460,"SELECT":476,"EVALUATE":683} and len(blocks) == 44,
                    "DNABR frozen cohort/roles/targets differ")
        geometry = discrete.materialize.input_plan.plan_regions(blocks,
                    {k:v["length_bp"] for k,v in self.source["sources"].items()}, r["exclusion_scope"])
        matched = [v for v in geometry if v["block_id"] == r["block_id"]]
        require(len(matched) == 1, "Unknown target block")
        self.exclusion = matched[0]
        require(set(r["chromosomes"]) == set(self.source["sources"]), "Exact source chromosome inventory required")
        self.included = sorted(set(r["chromosomes"])-{self.exclusion["target"][0]}, key=int)
        require(self.included, "No included chromosome")
        self.source_hashes = {str(self.bundle/n):discrete.sha(self.bundle/n)
                              for n in ("materialize.json", *discrete.materialize.META)}
        self.chunks, self.selected = [], {}
        total_bytes = 0
        for chrom in self.included:
            d = r["chromosomes"][chrom]
            require(set(d) == {"event_directory", "event_manifest_sha256", "producer_code_sha256",
                    "fit_manifest", "fit_manifest_sha256", "retained", "retained_sha256"}, "Exact chromosome descriptor required")
            directory = Path(d["event_directory"]).resolve()
            mp = pinned(directory/"manifest.json", d["event_manifest_sha256"], limits["max_metadata_bytes"])
            fp = pinned(d["fit_manifest"], d["fit_manifest_sha256"], limits["max_metadata_bytes"])
            rp = pinned(d["retained"], d["retained_sha256"], limits["max_metadata_bytes"])
            m, f = discrete.load_json(mp), discrete.load_json(fp)
            require(m["schema"] == "r02_genomic_event_chromosome_v1" and
                    m["stage"] == "COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING" and m["chromosome"] == chrom
                    and m["source_contract_sha256"] == r["source_contract_sha256"]
                    and m["n_people"] == len(self.persons) and m["sample_ids_sha256"] ==
                    discrete.materialize.evidence.sample_hash([p["sample_id"] for p in self.persons])
                    and m["code_sha256"] == d["producer_code_sha256"], "Event producer/cohort/source differs")
            require(isinstance(d["producer_code_sha256"],dict) and d["producer_code_sha256"]
                    and all(discrete.valid_hash(v) for v in d["producer_code_sha256"].values()), "Frozen producer pins required")
            require(f["schema"] == "r02_structure_region_support_v1" and
                    f["stage"] == "FIT_SUPPORT_MEASURED_NO_BIOLOGICAL_VALIDATION" and f["chromosome"] == chrom
                    and f["genotype_roles_read"] == ["FIT"] and f["common"] == self.source["common"]
                    and f["fit_ids_sha256"] == discrete.materialize.evidence.sample_hash(
                        [self.persons[i]["sample_id"] for i in self.roles["FIT"]])
                    and f["publish_files"]["retained.tsv"] == d["retained_sha256"]
                    and f["source_sha256"]["common_vcf"] == self.source["sources"][chrom]["common_vcf"]["sha256"],
                    "Selected marker list is not the authenticated source FIT-only LD list")
            markers = {}
            block_coordinates={b['block_id']:b for b in blocks if b['chromosome']==chrom}
            for row in discrete.shared.table(rp, LOCUS_COLUMNS):
                pos = int(row["position_bp"])
                require(row["chromosome"] == chrom and pos not in markers and row["ref"] in "ACGT"
                        and row["alt"] in "ACGT" and row["ref"] != row["alt"]
                        and row["variant_id"] == f"{chrom}:{pos}:{row['ref']}:{row['alt']}", "Selected locus identity differs")
                require(row['block_id'] in block_coordinates and
                        int(block_coordinates[row['block_id']]['start_bp'])<=pos<=int(block_coordinates[row['block_id']]['end_bp']),
                        'Selected source-interval locus lies outside its declared block')
                markers[pos] = row
            require(markers, "Empty source FIT LD selection")
            require(len(markers)==sum(int(v['retained_sites']) for v in f['pruning']),
                    "Selected FIT LD catalogue count differs from manifest")
            self.selected[chrom] = markers
            require(sum(len(v) for v in self.selected.values()) <= limits['max_selected_sites'],
                    'Selected-site limit exceeded before genotype allocation')
            marker_positions=np.asarray(sorted(markers),dtype=np.int64)
            self.source_hashes.update({str(p):discrete.sha(p) for p in (mp,fp,rp)})
            previous = 0
            for c in m["channels"]["common"]:
                require(set(c) == {"path", "sha256", "n_sites", "bytes", "start_bp", "end_bp"}
                        and previous < c["start_bp"] <= c["end_bp"] and 0 < c["n_sites"] <= limits["max_chunk_sites"],
                        "Invalid ordered common chunk descriptor")
                previous = c["end_bp"]
                if np.searchsorted(marker_positions,c['start_bp']) == np.searchsorted(marker_positions,c['end_bp'],side='right'):
                    continue
                rel=Path(c["path"])
                require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe genotype path")
                path=(directory/rel).resolve()
                require(path.is_relative_to(directory) and m["outputs_sha256"].get(c["path"]) == c["sha256"],
                        "Chunk escaped producer root or is not producer-authenticated")
                pinned(path,c["sha256"],limits["max_chunk_bytes"])
                require(path.stat().st_size == c["bytes"], "Chunk byte count differs")
                with zipfile.ZipFile(path) as archive:
                    info=archive.infolist()
                    require(len(info)==len(EVENT_FIELDS) and {v.filename for v in info}=={k+'.npy' for k in EVENT_FIELDS}
                            and sum(v.file_size for v in info)*4 <= limits["max_chunk_bytes"], "Chunk decompression budget/inventory differs")
                total_bytes += c["bytes"]
                require(total_bytes <= limits["max_input_bytes"], "Input byte budget exceeded")
                self.chunks.append((chrom,path,c))
                self.source_hashes[str(path)]=c["sha256"]
        self.n_selected=sum(len(v) for v in self.selected.values())
        require(0 < self.n_selected <= limits["max_selected_sites"], "Selected-site limit exceeded")
        self.accumulator=DualContinuousAccumulator(self.persons,weighting=r["weighting"],
                max_chunk_sites=limits["max_chunk_sites"],max_memory_bytes=limits["max_memory_bytes"]-limits["max_chunk_bytes"])

    def verify(self):
        require(discrete.sha(self.request_path)==self.request_sha256 and
                self.r["implementation_sha256"]==implementation_hashes(), "Request/consumer changed")
        require(all(discrete.sha(p)==v for p,v in self.source_hashes.items()), "Authenticated input changed")

    def accumulate(self):
        accumulate_selected_chunks(self.persons,self.roles['FIT'],self.selected,self.chunks,self.accumulator,
                                   min_fit_called=self.source['common']['min_fit_called'])
        self.verify()
        return self.accumulator.finalize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--stage',choices=('audit','fit'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();os.umask(0o077)
    require(not args.output_dir.exists(), 'Output exists; preserve reusable results')
    view=CommonView(args.bundle,args.request,args.request_sha256)
    require(shutil.disk_usage(args.output_dir.parent).free >= view.limits['min_free_bytes']+view.limits['max_output_bytes'],
            'Insufficient disk reserve for output budget')
    args.output_dir.mkdir(mode=0o700)
    result=dict(schema=SCHEMA,stage='AUDITED_NO_SCORES',request_sha256=args.request_sha256,
                source_hashes=view.source_hashes,implementation_sha256=implementation_hashes(),
                exclusion=view.exclusion,exclusion_scope='WHOLE_CHROMOSOME',included_chromosomes=view.included,
                selected_sites=view.n_selected,genotype_chunks=len(view.chunks),targets_read=False,
                biological_validation=False,adequate_continuous_comparator_certified=False,
                fitted_roles=['FIT'] if args.stage=='fit' else [],outputs_sha256={})
    if args.stage=='fit':
        model=view.accumulate()
        numeric={k:v for k,v in model.items() if isinstance(v,np.ndarray)}
        output=args.output_dir/'continuous.private.npz'
        np.savez(output,**numeric)
        require(output.stat().st_size <= view.limits['max_output_bytes'], 'Output budget exceeded')
        result.update(stage='COMMON_CONTINUOUS_SCORES_NOT_K03_EVALUATION',
                      numerical_rank=len(model['eigenvalues']),numerical_tolerance=model['numerical_tolerance'],
                      weighting=model['weighting'],outputs_sha256={output.name:discrete.sha(output)})
    view.verify()
    discrete.materialize.write_json(args.output_dir/'manifest.json',result)
    print(json.dumps({k:result[k] for k in ('schema','stage','biological_validation')}))


if __name__=='__main__':main()
