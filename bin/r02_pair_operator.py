"""Bounded symmetric pair design and standardized weighted Ridge with LSQR.

This is a numerical backend, not a scientific admission or a parameter search.
The caller must authenticate roles, representations, targets, and pair weights.
Scaling and Ridge both use FIT pair weights, matching the actual fit callers in
r02_structure_increment. No missing-value policy
is inferred here: representations and extra columns must already be finite.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def require(condition, message):
    if not condition:
        raise ValueError(message)


RIDGE_PATH_CHECKPOINT_SCHEMA = 'r02_pair_ridge_path_checkpoint_v1'
LSQR_PARTIAL_CHECKPOINT_SCHEMA = 'r02_pair_lsqr_partial_checkpoint_v1'


def _hash_array(digest, value):
    value = np.ascontiguousarray(value)
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(value.shape, separators=(',', ':')).encode())
    digest.update(memoryview(value).cast('B'))


def ridge_path_fingerprint(design, response, weights, alphas, settings, identity):
    """Bind resumable numerical state to exact data, code and solver settings."""
    require(isinstance(identity, str) and identity, 'Nonempty checkpoint identity required')
    digest = hashlib.sha256()
    digest.update(RIDGE_PATH_CHECKPOINT_SCHEMA.encode())
    digest.update(identity.encode())
    digest.update(hashlib.sha256(Path(__file__).read_bytes()).digest())
    for value in (design.points, design.left, design.right, design.extra,
                  response, weights, np.asarray(alphas, dtype=np.float64)):
        _hash_array(digest, value)
    digest.update(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode())
    digest.update(json.dumps(dict(shape=design.shape, backend=design.backend,
        preprocessing_backend=design.preprocessing_backend, workers=design.workers),
        sort_keys=True, separators=(',', ':')).encode())
    return digest.hexdigest()


def write_ridge_path_checkpoint(path, fingerprint, iteration, q, basis, images,
                                operator_applications, models, alpha_values):
    """Atomically replace one private checkpoint after a complete iteration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    accepted = [a for a in alpha_values if a in models]
    columns = len(q)
    coefficients = (np.vstack([models[a].coefficient for a in accepted])
                    if accepted else np.empty((0, columns), dtype=np.float64))
    if accepted:
        center=models[accepted[0]].center;scale=models[accepted[0]].scale
        offset=models[accepted[0]].fit_standardized_offset
        require(all(np.array_equal(models[a].center,center) and
                    np.array_equal(models[a].scale,scale) and
                    np.array_equal(models[a].fit_standardized_offset,offset) for a in accepted),
                'Ridge path candidates do not share frozen preprocessing')
    else:
        center=np.zeros(columns);scale=np.ones(columns);offset=np.zeros(columns)
    diagnostics = np.asarray([json.dumps(models[a].diagnostics, sort_keys=True,
                                         separators=(',', ':')) for a in accepted])
    state = dict(schema=np.asarray(RIDGE_PATH_CHECKPOINT_SCHEMA),
        fingerprint=np.asarray(fingerprint), iteration=np.asarray(iteration, dtype=np.int64),
        q=np.asarray(q), basis=np.column_stack(basis) if basis else np.empty((columns, 0)),
        images=np.column_stack(images) if images else np.empty((columns, 0)),
        operator_applications=np.asarray(operator_applications, dtype=np.int64),
        accepted_alphas=np.asarray(accepted, dtype=np.float64), coefficients=coefficients,
        center=np.asarray(center), scale=np.asarray(scale), offset=np.asarray(offset),
        response_means=np.asarray([models[a].response_mean for a in accepted], dtype=np.float64),
        diagnostics=diagnostics)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    try:
        with temporary.open('xb') as handle:
            np.savez(handle, **state)
            handle.flush();os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():temporary.unlink()


def load_ridge_path_checkpoint(path, fingerprint, alpha_values, columns):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        required = {'schema','fingerprint','iteration','q','basis','images',
                    'operator_applications','accepted_alphas','coefficients','center',
                    'scale','offset','response_means','diagnostics'}
        require(set(saved.files) == required, 'Ridge path checkpoint fields differ')
        require(str(saved['schema']) == RIDGE_PATH_CHECKPOINT_SCHEMA and
                str(saved['fingerprint']) == fingerprint,
                'Ridge path checkpoint input/code identity differs')
        iteration = int(saved['iteration']);q=saved['q'].copy()
        basis=saved['basis'].copy();images=saved['images'].copy()
        accepted=saved['accepted_alphas'].astype(float).tolist()
        coefficients=saved['coefficients'].copy();center=saved['center'].copy()
        scale=saved['scale'].copy();offset=saved['offset'].copy()
        response_means=saved['response_means'].copy();diagnostics=saved['diagnostics'].tolist()
        applications=int(saved['operator_applications'])
    rows=(len(accepted),columns)
    require(iteration >= 0 and q.shape == (columns,) and
            basis.shape == images.shape == (columns, iteration) and
            coefficients.shape == rows and center.shape == scale.shape == offset.shape == (columns,) and
            response_means.shape == (len(accepted),) and len(diagnostics)==len(accepted) and
            len(set(accepted))==len(accepted) and set(accepted)<=set(alpha_values) and
            all(np.isfinite(v).all() for v in (q,basis,images,coefficients,center,scale,
                                               offset,response_means)) and
            np.all(scale>0) and applications >= iteration,
            'Invalid Ridge path checkpoint content')
    models={a:PairRidge(coefficients[k],center.copy(),scale.copy(),response_means[k],
                       json.loads(diagnostics[k]),offset.copy()) for k,a in enumerate(accepted)}
    return dict(iteration=iteration,q=q,basis=[basis[:,k] for k in range(iteration)],
                images=[images[:,k] for k in range(iteration)],models=models,
                operator_applications=applications)


def write_lsqr_partial_checkpoint(path, fingerprint, alpha, x0, iterations, segments, status):
    """Persist one non-converged LSQR iterate without promoting it to a model."""
    path=Path(path);x0=np.asarray(x0,dtype=np.float64).reshape(-1)
    require(isinstance(fingerprint,str) and len(fingerprint)==64 and
            math.isfinite(alpha) and alpha>0 and x0.size>0 and np.isfinite(x0).all() and
            type(iterations) is int and iterations>0 and type(segments) is int and segments>0 and status==7,
            'Invalid LSQR partial checkpoint')
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    try:
        with temporary.open('xb') as handle:
            np.savez(handle,schema=np.asarray(LSQR_PARTIAL_CHECKPOINT_SCHEMA),
                fingerprint=np.asarray(fingerprint),alpha=np.asarray(alpha,dtype=np.float64),
                x0=x0,iterations=np.asarray(iterations,dtype=np.int64),
                segments=np.asarray(segments,dtype=np.int64),
                status=np.asarray(status,dtype=np.int64))
            handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def load_lsqr_partial_checkpoint(path, fingerprint, alpha_values, columns):
    path=Path(path)
    if not path.exists():return None
    with np.load(path,allow_pickle=False) as saved:
        require(set(saved.files)=={'schema','fingerprint','alpha','x0','iterations','segments','status'},
                'LSQR partial checkpoint fields differ')
        require(str(saved['schema'])==LSQR_PARTIAL_CHECKPOINT_SCHEMA and
                str(saved['fingerprint'])==fingerprint,
                'LSQR partial checkpoint input/code identity differs')
        alpha=float(saved['alpha']);x0=saved['x0'].copy();iterations=int(saved['iterations'])
        segments=int(saved['segments']);status=int(saved['status'])
    require(alpha in set(alpha_values) and x0.shape==(columns,) and np.isfinite(x0).all() and
            iterations>0 and segments>0 and status==7,'Invalid LSQR partial checkpoint content')
    return dict(alpha=alpha,x0=x0,iterations=iterations,segments=segments,status=status)


def clear_lsqr_partial_checkpoint(path):
    path=Path(path)
    if path.exists():path.unlink()


class PairDesign:
    """Rows [(x_i+x_j)/2, abs(x_i-x_j), x_i*x_j, optional paired columns]."""

    def __init__(self, points, left, right, *, extra=None, rows_per_chunk,
                 max_working_mib, backend='chunked', preprocessing_backend='chunked',
                 workers=1):
        self.points = np.asarray(points, dtype=np.float64)
        self.left, self.right = np.asarray(left), np.asarray(right)
        require(self.points.ndim == 2 and np.isfinite(self.points).all(),
                'Finite individual representation required')
        require(self.left.ndim == self.right.ndim == 1 and
                self.left.shape == self.right.shape and len(self.left) > 0 and
                self.left.dtype.kind in 'iu' and self.right.dtype.kind in 'iu',
                'Nonempty integer pair indices required')
        require(np.all(self.left >= 0) and np.all(self.right >= 0) and
                np.all(self.left < len(self.points)) and
                np.all(self.right < len(self.points)) and
                np.all(self.left != self.right), 'Invalid pair endpoint')
        self.extra = (np.empty((len(self.left), 0), dtype=np.float64) if extra is None
                      else np.asarray(extra, dtype=np.float64))
        require(self.extra.ndim == 2 and len(self.extra) == len(self.left) and
                np.isfinite(self.extra).all(), 'Extra columns must be finite and aligned')
        require(type(rows_per_chunk) is int and rows_per_chunk > 0 and
                math.isfinite(max_working_mib) and max_working_mib > 0,
                'Positive explicit chunk and memory limits required')
        self.rows_per_chunk = min(rows_per_chunk, len(self.left))
        self.shape = (len(self.left), 3*self.points.shape[1]+self.extra.shape[1])
        require(self.shape[1] > 0, 'Empty design')
        # Conservative six simultaneous dense chunk buffers. Full pair design
        # is never allocated. Inputs and O(n_pairs) solver vectors are separate.
        self.chunk_bytes_bound = 6*self.rows_per_chunk*self.shape[1]*8
        require(backend in {'chunked', 'factorized', 'factorized_sparse_binary',
                            'factorized_admitted_pairs'},
                'Unknown numerical backend')
        self.backend = backend
        require(preprocessing_backend in {'chunked', 'moments'}, 'Unknown preprocessing backend')
        self.preprocessing_backend = preprocessing_backend
        require(type(workers) is int and workers > 0, 'Positive explicit worker count required')
        require(workers == 1 or backend.startswith('factorized') or preprocessing_backend == 'moments',
                'Parallel workers require an implemented factorized operation')
        self.workers = workers
        n, d = self.points.shape
        factor_bound = (8*((workers+2)*n*n+6*n*d+8*self.shape[0]+self.shape[1]*4)+65536*workers
                        if (backend.startswith('factorized') and backend != 'factorized_admitted_pairs')
                        or preprocessing_backend == 'moments' else 0)
        if backend == 'factorized_admitted_pairs':
            # CSR product operator plus dense person result and three admitted
            # pair chunk buffers.  Pair indices and input points are separate.
            factor_bound = max(factor_bound,
                32*self.shape[0] + 8*(n+1) + 16*n*d + 32*self.rows_per_chunk*d + 8*self.shape[1])
        admitted_bound = (32*self.shape[0] + 8*(n+1) + 16*n*d +
                          32*self.rows_per_chunk*d + 8*self.shape[1])
        factor_operation_bound = (8*((workers+2)*n*n+6*n*d+8*self.shape[0]+self.shape[1]*4)
                                  + 65536*workers)
        self.operation_working_bytes_bound = (
            admitted_bound if backend == 'factorized_admitted_pairs' else
            factor_operation_bound if backend.startswith('factorized') else self.chunk_bytes_bound)
        if backend == 'factorized_sparse_binary':
            # CSR operands/results, a dense person matrix, and gather buffers.
            # Bound also covers dense binary inputs, not only sparse one-hot.
            factor_bound += 8*(4*n*n+4*n*d)
        self.working_bytes_bound = max(self.chunk_bytes_bound, factor_bound)
        require(self.working_bytes_bound <= max_working_mib*2**20,
                'Declared chunk memory bound exceeded')
        if backend == 'factorized_sparse_binary':
            constant = np.all(self.points == self.points[:1], axis=0)
            binary = np.all((self.points == 0.) | (self.points == 1.), axis=0) & ~constant
            self._constant_axes = np.flatnonzero(constant)
            self._binary_axes = np.flatnonzero(binary)
            self._dense_axes = np.flatnonzero(~(constant | binary))
            self._constant_values = self.points[0, self._constant_axes].copy()
            self._binary_points = csr_matrix(self.points[:, self._binary_axes])
            self._dense_points = np.ascontiguousarray(self.points[:, self._dense_axes])
            extra_constant = np.flatnonzero(np.all(self.extra == self.extra[:1], axis=0))
            self._constant_columns = np.r_[self._constant_axes, d+self._constant_axes,
                                           2*d+self._constant_axes, 3*d+extra_constant]
            self._constant_raw_values = np.r_[self._constant_values,
                np.zeros(len(self._constant_axes)), self._constant_values**2,
                self.extra[0, extra_constant]]

    def _parallel_ranges(self, count, function):
        """Bounded independent row/axis blocks; no concurrent reductions."""
        if count == 0:
            return
        width = max(1, (count+self.workers-1)//self.workers)
        ranges = [(a, min(a+width, count)) for a in range(0, count, width)]
        if self.workers == 1:
            function(ranges[0])
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(function, ranges))

    def _weighted_cityblock(self, weights, points=None):
        x = self.points if points is None else points
        distance = np.empty((len(x), len(x)), dtype=np.float64)
        def rows(bounds):
            start, stop = bounds
            cdist(x[start:stop], x, metric='cityblock', w=weights,
                  out=distance[start:stop])
        self._parallel_ranges(len(x), rows)
        return distance

    def _raw_matvec(self, vector):
        """Same three feature families, evaluated with person-sized matrices."""
        if self.backend == 'factorized_sparse_binary':
            return self._sparse_binary_matvec(vector)
        if self.backend == 'factorized_admitted_pairs':
            return self._admitted_pair_matvec(vector)
        x = self.points
        d = x.shape[1]
        mean = x @ vector[:d]
        result = (mean[self.left]+mean[self.right])/2
        product = (x*vector[2*d:3*d]) @ x.T
        result += product[self.left, self.right]
        del product
        # scipy requires nonnegative metric weights; subtraction preserves
        # arbitrary signed regression coefficients without changing features.
        for sign in (1., -1.):
            weights = np.maximum(sign*vector[d:2*d], 0)
            if np.any(weights):
                distance = self._weighted_cityblock(weights)
                result += sign*distance[self.left, self.right]
                del distance
        if self.extra.shape[1]:
            result += self.extra @ vector[3*d:]
        return result

    def _raw_rmatvec(self, vector):
        if self.backend == 'factorized_sparse_binary':
            return self._sparse_binary_rmatvec(vector)
        if self.backend == 'factorized_admitted_pairs':
            return self._admitted_pair_rmatvec(vector)
        x = self.points
        n, d = x.shape
        result = np.zeros(self.shape[1])
        endpoint = np.bincount(self.left, weights=vector, minlength=n)
        endpoint += np.bincount(self.right, weights=vector, minlength=n)
        result[:d] = (x.T @ endpoint)/2
        paired = np.zeros((n, n))
        # add.at retains repeated or reversed pair rows and arbitrary masks.
        np.add.at(paired, (self.left, self.right), vector/2)
        np.add.at(paired, (self.right, self.left), vector/2)
        result[2*d:3*d] = np.sum(x*(paired @ x), axis=0)
        def axes(bounds):
            start, stop = bounds
            for axis in range(start, stop):
                distance = cdist(x[:, axis:axis+1], x[:, axis:axis+1], metric='cityblock')
                result[d+axis] = np.einsum('ij,ij->', paired, distance, optimize=False)
        self._parallel_ranges(d, axes)
        if self.extra.shape[1]:
            result[3*d:] = self.extra.T @ vector
        return result

    def _admitted_pair_matvec(self, vector):
        """Evaluate exact symmetric columns only for the admitted pair rows."""
        x = self.points
        d = x.shape[1]
        result = np.empty(self.shape[0], dtype=np.float64)
        for start in range(0, self.shape[0], self.rows_per_chunk):
            sl = slice(start, min(start+self.rows_per_chunk, self.shape[0]))
            a, b = x[self.left[sl]], x[self.right[sl]]
            value = ((a+b)/2.) @ vector[:d]
            value += np.abs(a-b) @ vector[d:2*d]
            value += (a*b) @ vector[2*d:3*d]
            if self.extra.shape[1]:
                value += self.extra[sl] @ vector[3*d:]
            result[sl] = value
        return result

    def _admitted_pair_rmatvec(self, vector):
        """Adjoint using admitted rows; duplicates, inversions and signs remain."""
        x = self.points
        n, d = x.shape
        result = np.zeros(self.shape[1], dtype=np.float64)
        endpoint = np.bincount(self.left, weights=vector, minlength=n)
        endpoint += np.bincount(self.right, weights=vector, minlength=n)
        result[:d] = (x.T @ endpoint)/2.
        # CSR sums duplicate oriented entries and accepts arbitrary signed data.
        paired = csr_matrix((vector, (self.left, self.right)), shape=(n, n))
        paired.sum_duplicates()
        result[2*d:3*d] = np.sum(x*(paired @ x), axis=0, dtype=np.float64)
        for start in range(0, self.shape[0], self.rows_per_chunk):
            sl = slice(start, min(start+self.rows_per_chunk, self.shape[0]))
            difference = np.abs(x[self.left[sl]]-x[self.right[sl]])
            result[d:2*d] += difference.T @ vector[sl]
        if self.extra.shape[1]:
            result[3*d:] = self.extra.T @ vector
        return result

    def _sparse_binary_matvec(self, vector):
        """Exact binary identities; every logical feature/penalty is retained."""
        d = self.points.shape[1]
        dense, binary, constant = self._dense_axes, self._binary_axes, self._constant_axes
        c = self._constant_values
        result = np.full(self.shape[0], c @ vector[constant] + (c*c) @ vector[2*d+constant])
        x = self._dense_points
        if len(dense):
            mean = x @ vector[dense]
            result += (mean[self.left]+mean[self.right])/2
            product = (x*vector[2*d+dense]) @ x.T
            result += product[self.left, self.right]
            del product
            for sign in (1., -1.):
                weights = np.maximum(sign*vector[d+dense], 0.)
                if np.any(weights):
                    distance = self._weighted_cityblock(weights, x)
                    result += sign*distance[self.left, self.right]
                    del distance
        if len(binary):
            b = self._binary_points
            endpoint = np.asarray(b @ (vector[binary]/2+vector[d+binary])).reshape(-1)
            result += endpoint[self.left]+endpoint[self.right]
            product = b.multiply(vector[2*d+binary]-2*vector[d+binary]) @ b.T
            result += np.asarray(product.tocsr()[self.left, self.right]).reshape(-1)
        if self.extra.shape[1]:
            result += self.extra @ vector[3*d:]
        return result

    def _sparse_binary_rmatvec(self, vector):
        n, d = self.points.shape
        dense, binary, constant = self._dense_axes, self._binary_axes, self._constant_axes
        result = np.zeros(self.shape[1])
        total = float(np.sum(vector))
        result[constant] = self._constant_values*total
        result[2*d+constant] = self._constant_values**2*total
        endpoint = np.bincount(self.left, weights=vector, minlength=n)
        endpoint += np.bincount(self.right, weights=vector, minlength=n)
        paired = np.zeros((n, n))
        np.add.at(paired, (self.left, self.right), vector/2)
        np.add.at(paired, (self.right, self.left), vector/2)
        x = self._dense_points
        if len(dense):
            result[dense] = (x.T @ endpoint)/2
            result[2*d+dense] = np.sum(x*(paired @ x), axis=0)
            def axes(bounds):
                for k in range(*bounds):
                    distance = cdist(x[:, k:k+1], x[:, k:k+1], metric='cityblock')
                    result[d+dense[k]] = np.einsum('ij,ij->', paired, distance, optimize=False)
            self._parallel_ranges(len(dense), axes)
        if len(binary):
            b = self._binary_points
            mean_twice = np.asarray(b.T @ endpoint).reshape(-1)
            product = np.asarray(b.multiply(paired @ b).sum(axis=0)).reshape(-1)
            result[binary] = mean_twice/2
            result[d+binary] = mean_twice-2*product
            result[2*d+binary] = product
        if self.extra.shape[1]:
            result[3*d:] = self.extra.T @ vector
        return result

    def chunks(self, columns=None):
        """Generate all or an explicit subset of the same symmetric columns.

        The subset path lets stable scaling process near-constant columns
        together. It never removes those columns from the statistical design.
        Its buffers are subsets of the admitted full-row chunk memory bound.
        """
        d = self.points.shape[1]
        columns = (np.arange(self.shape[1], dtype=np.int64) if columns is None
                   else np.asarray(columns))
        require(columns.ndim == 1 and columns.dtype.kind in 'iu' and
                len(columns) > 0 and np.all(columns >= 0) and
                np.all(columns < self.shape[1]) and
                len(np.unique(columns)) == len(columns), 'Invalid feature column subset')
        for start in range(0, self.shape[0], self.rows_per_chunk):
            sl = slice(start, min(start+self.rows_per_chunk, self.shape[0]))
            out = np.empty((sl.stop-sl.start, len(columns)), dtype=np.float64)
            for kind in range(3):
                mask = (columns >= kind*d) & (columns < (kind+1)*d)
                if not np.any(mask):
                    continue
                axes = columns[mask]-kind*d
                a = self.points[np.ix_(self.left[sl], axes)]
                b = self.points[np.ix_(self.right[sl], axes)]
                out[:, mask] = (a+b)/2 if kind == 0 else np.abs(a-b) if kind == 1 else a*b
                del a, b
            mask = columns >= 3*d
            if np.any(mask):
                out[:, mask] = self.extra[sl][:, columns[mask]-3*d]
            require(np.isfinite(out).all(), 'Nonfinite symmetric feature')
            yield sl, out

    def fit_preprocessing(self, weights):
        """FIT only: weighted scaler and center for the unpenalized intercept."""
        weights = check_weights(weights, self.shape[0])
        if self.preprocessing_backend == 'moments':
            return self._fit_moments(weights)
        scaler = StandardScaler()
        for sl, values in self.chunks():
            scaler.partial_fit(values, sample_weight=weights[sl])
        center = scaler.mean_.copy()
        require(np.isfinite(center).all() and np.isfinite(scaler.scale_).all() and
                np.all(scaler.scale_ > 0), 'Invalid FIT scaling')
        return self._exact_constant_preprocessing(center, scaler.scale_.copy())

    def _exact_constant_preprocessing(self, center, scale):
        if self.backend == 'factorized_sparse_binary':
            # Exact equality proof above, never StandardScaler's variance guard.
            # Positive weights and a free intercept imply these centered FIT
            # columns are identically zero, whatever the numerical mean sum.
            center[self._constant_columns] = self._constant_raw_values
            scale[self._constant_columns] = 1.
        return center, scale

    def _fit_moments(self, weights):
        """Exact weighted moments, with stable two-pass fallback near constants.

        E|Xi-Xj|^2 = E(Xi-Xj)^2; product second moments use X^2.
        Only the first absolute moment requires featurewise pair distances.
        Bounds detect cancellation; uncertain columns use StandardScaler on
        their actual values rather than clipping a negative variance to zero.
        The algebra retains repeated/reversed/masked pairs and original weights.
        """
        x = self.points
        n, d = x.shape
        total = float(weights.sum())
        center = self._raw_rmatvec(weights)/total
        endpoint = np.bincount(self.left, weights=weights, minlength=n)
        endpoint += np.bincount(self.right, weights=weights, minlength=n)
        paired = np.zeros((n, n))
        np.add.at(paired, (self.left, self.right), weights/2)
        np.add.at(paired, (self.right, self.left), weights/2)
        squared = x*x
        endpoint_square = squared.T @ endpoint/total
        second = np.empty(self.shape[1])
        second[:d] = endpoint_square/4 + center[2*d:3*d]/2
        second[d:2*d] = endpoint_square - 2*center[2*d:3*d]
        if self.backend == 'factorized_sparse_binary':
            dense, binary, constant = self._dense_axes, self._binary_axes, self._constant_axes
            sq = squared[:, dense]
            second[2*d+dense] = np.sum(sq*(paired @ sq), axis=0)/total
            second[2*d+binary] = center[2*d+binary]
            second[2*d+constant] = self._constant_values**4
        else:
            second[2*d:3*d] = np.sum(squared*(paired @ squared), axis=0)/total
        if self.extra.shape[1]:
            second[3*d:] = np.average(self.extra*self.extra, weights=weights, axis=0)
        variance = second-center*center
        eps = np.finfo(float).eps
        # Conservative forward-error guard for sums/products and subtraction.
        magnitude = np.abs(second)
        magnitude[:d] = endpoint_square/4 + np.abs(center[2*d:3*d])/2
        magnitude[d:2*d] = endpoint_square + 2*np.abs(center[2*d:3*d])
        bound = 64*eps*(self.shape[0]+n)*np.maximum(magnitude, center*center)
        unstable = (~np.isfinite(variance)) | (variance <= bound)
        scale = np.sqrt(np.maximum(variance, 0))
        columns = np.flatnonzero(unstable)
        if len(columns):
            scaler = StandardScaler()
            for sl, values in self.chunks(columns):
                scaler.partial_fit(values, sample_weight=weights[sl])
            center[columns], scale[columns] = scaler.mean_, scaler.scale_
        require(np.isfinite(center).all() and np.isfinite(scale).all()
                and np.all(scale > 0), 'Invalid factorized FIT scaling')
        return self._exact_constant_preprocessing(center, scale)

    def operator(self, weights, center, scale):
        weights = check_weights(weights, self.shape[0])
        center, scale = np.asarray(center), np.asarray(scale)
        require(center.shape == scale.shape == (self.shape[1],) and
                np.isfinite(center).all() and np.isfinite(scale).all() and
                np.all(scale > 0), 'Invalid frozen preprocessing')
        root = np.sqrt(weights)
        stable_columns = np.empty(0, dtype=np.int64)
        if self.backend == 'factorized_sparse_binary':
            # Cancellation guard changes arithmetic only. Near-constant columns
            # stay in the model and are centered locally, just like chunks().
            stable_columns = np.union1d(self._constant_columns,
                np.flatnonzero(np.abs(center)*np.sqrt(np.finfo(float).eps) > scale))

        def matvec(vector):
            v = np.asarray(vector, dtype=np.float64).reshape(-1)
            require(v.shape == (self.shape[1],), 'Coefficient dimension differs')
            if self.backend.startswith('factorized'):
                raw = v/scale
                raw[stable_columns] = 0.
                result = self._raw_matvec(raw)-center@raw
                if len(stable_columns):
                    for sl, values in self.chunks(stable_columns):
                        values -= center[stable_columns]
                        values /= scale[stable_columns]
                        result[sl] += values @ v[stable_columns]
                return result*root
            result = np.empty(self.shape[0])
            for sl, values in self.chunks():
                values -= center
                values /= scale
                result[sl] = (values @ v)*root[sl]
            return result

        def rmatvec(vector):
            v = np.asarray(vector, dtype=np.float64).reshape(-1)
            require(v.shape == (self.shape[0],), 'Response dimension differs')
            if self.backend.startswith('factorized'):
                raw = v*root
                result = (self._raw_rmatvec(raw)-center*raw.sum())/scale
                if len(stable_columns):
                    result[stable_columns] = 0.
                    for sl, values in self.chunks(stable_columns):
                        values -= center[stable_columns]
                        values /= scale[stable_columns]
                        result[stable_columns] += values.T @ raw[sl]
                return result
            result = np.zeros(self.shape[1])
            for sl, values in self.chunks():
                values -= center
                values /= scale
                result += values.T @ (v[sl]*root[sl])
            return result

        return LinearOperator(self.shape, matvec=matvec, rmatvec=rmatvec,
                              dtype=np.dtype(np.float64))


    def ridge_operator(self, weights, center, scale):
        """Center standardized FIT columns for the unpenalized intercept.

        A nearly constant raw feature can retain a substantial mean after FP64
        scaling. This projection retains every column and its Ridge penalty.
        Older backends preserve their frozen numerical behavior.
        """
        op = self.operator(weights, center, scale)
        offset = np.zeros(self.shape[1])
        if self.backend != 'factorized_sparse_binary':
            return op, offset
        root = np.sqrt(weights)
        offset = op.rmatvec(root)/float(np.sum(weights))
        def matvec(v):
            value = np.asarray(v).reshape(-1)
            return op @ value-root*float(offset@value)
        def rmatvec(v):
            value = np.asarray(v).reshape(-1)
            return op.rmatvec(value)-offset*float(root@value)
        return LinearOperator(self.shape, matvec=matvec, rmatvec=rmatvec,
                              dtype=np.dtype(np.float64)), offset


def check_weights(weights, n):
    weights = np.asarray(weights, dtype=np.float64)
    require(weights.shape == (n,) and np.isfinite(weights).all() and
            np.all(weights > 0) and math.isfinite(float(weights.sum())),
            'Finite positive aligned weights required; no silent normalization')
    return weights


@dataclass
class PairRidge:
    coefficient: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    response_mean: float
    diagnostics: dict
    fit_standardized_offset: np.ndarray | None = None

    def predict(self, design):
        require(design.shape[1] == len(self.coefficient), 'Prediction columns differ')
        if design.backend.startswith('factorized'):
            values = self.response_mean + design.operator(
                np.ones(design.shape[0]), self.center, self.scale) @ self.coefficient
            require(np.isfinite(values).all(), 'Nonfinite prediction')
            return values
        values = np.empty(design.shape[0])
        for sl, x in design.chunks():
            x -= self.center
            x /= self.scale
            values[sl] = self.response_mean + x @ self.coefficient
        require(np.isfinite(values).all(), 'Nonfinite prediction')
        return values

    def numerical_loss_interval(self, design, response, weights, *,
                                residual_roundoff_allowance):
        """Bound loss of the exact Ridge solution, conditional on residual accuracy.

        H=Z'WZ+alpha*I has lambda_min>=alpha, hence ||beta-beta*|| <=
        ||H beta-rhs||/alpha. Each query row then has error <= ||z|| times
        that coefficient bound. These are deterministic numerical bounds,
        NOT confidence intervals or uncertainty about biological observations.
        The caller must supply a nonnegative bound on residual computation
        error. Zero explicitly means exact-arithmetic diagnostic only; this
        implementation does not certify a floating-point roundoff allowance.
        Fixed FIT preprocessing and response mean are treated as given.
        """
        require(design.shape[1] == len(self.coefficient), 'Prediction columns differ')
        w = check_weights(weights, design.shape[0])
        y = np.asarray(response, dtype=np.float64)
        require(y.shape == w.shape and np.isfinite(y).all(), 'Invalid query response')
        allowance = float(residual_roundoff_allowance)
        require(math.isfinite(allowance) and allowance >= 0,
                'Explicit nonnegative residual roundoff allowance required')
        residual_norm = self.diagnostics.get('absolute_penalized_gradient')
        alpha = self.diagnostics.get('alpha')
        require(residual_norm is not None and alpha is not None and
                math.isfinite(residual_norm) and residual_norm >= 0 and alpha > 0,
                'Absolute residual and positive penalty required')
        coefficient_bound = (residual_norm+allowance)/alpha
        offset = (np.zeros(len(self.coefficient)) if self.fit_standardized_offset is None
                  else np.asarray(self.fit_standardized_offset))
        require(offset.shape == self.coefficient.shape and np.isfinite(offset).all(),
                'Invalid frozen standardized FIT mean')
        lower, upper, observed, maximum_prediction_bound = 0., 0., 0., 0.
        for sl, x in design.chunks():
            x -= self.center
            x /= self.scale
            predicted = self.response_mean+x@self.coefficient
            error = np.abs(predicted-y[sl])
            # The free intercept changes with beta: query errors use z-offset.
            bound = np.linalg.norm(x-offset, axis=1)*coefficient_bound
            require(np.isfinite(error).all() and np.isfinite(bound).all(),
                    'Nonfinite numerical loss bound')
            lower += float(w[sl]@np.maximum(error-bound, 0.)**2)
            upper += float(w[sl]@(error+bound)**2)
            observed += float(w[sl]@(error*error))
            maximum_prediction_bound = max(maximum_prediction_bound, float(bound.max()))
        total = float(w.sum())
        require(all(math.isfinite(v) for v in (lower, upper, observed)),
                'Nonfinite numerical loss interval')
        return dict(weighted_mse=observed/total,
                    exact_ridge_mse_lower=lower/total,
                    exact_ridge_mse_upper=upper/total,
                    coefficient_l2_error_bound=coefficient_bound,
                    maximum_prediction_error_bound=maximum_prediction_bound,
                    residual_roundoff_allowance=allowance,
                    bound_scope='CONDITIONAL_ON_RESIDUAL_ACCURACY_AND_FIXED_PREPROCESSING',
                    floating_point_roundoff_certified=False,
                    statistical_confidence_interval=False)


def fit_ridge(design, response, weights, alpha, *, atol, btol, max_iter,
              gradient_tolerance, preprocessing=None):
    """Minimize sum(w*(y-intercept-z beta)^2) + alpha*||beta||^2.

    LSQR damping is sqrt(alpha), not alpha. Its status alone is insufficient:
    recompute the normalized penalized normal-equation residual explicitly.
    Tolerances and iteration cap are required numerical settings, not biology.
    A capped or ill-conditioned solve cannot become a successful model.
    """
    weights = check_weights(weights, design.shape[0])
    y = np.asarray(response, dtype=np.float64)
    require(y.shape == weights.shape and np.isfinite(y).all(), 'Invalid response')
    require(math.isfinite(alpha) and alpha > 0 and 0 < atol < 1 and
            0 < btol < 1 and 0 < gradient_tolerance < 1 and
            type(max_iter) is int and max_iter > 0, 'Invalid explicit solver settings')
    if preprocessing is None:
        center, scale = design.fit_preprocessing(weights)
    else:
        center, scale = preprocessing
    op, standardized_offset = design.ridge_operator(weights, center, scale)
    ymean = float(np.average(y, weights=weights))
    b = np.sqrt(weights)*(y-ymean)
    result = lsqr(op, b, damp=math.sqrt(alpha), atol=atol, btol=btol,
                  conlim=0, iter_lim=max_iter, show=False)
    beta, status, iterations = result[:3]
    require(status in (0, 1, 2, 4, 5), 'LSQR did not converge: status='+str(status))
    residual = op @ beta-b
    gradient = op.rmatvec(residual)+alpha*beta
    rhs = op.rmatvec(b)
    ratio = float(np.linalg.norm(gradient)/max(np.linalg.norm(rhs),
                   alpha*np.linalg.norm(beta), np.finfo(float).tiny))
    require(math.isfinite(ratio) and ratio <= gradient_tolerance,
            'Penalized gradient exceeds declared numerical tolerance: '+str(ratio))
    diagnostics = dict(alpha=float(alpha), damping=math.sqrt(alpha),
        lsqr_status=int(status), iterations=int(iterations),
        condition_estimate=float(result[6]), relative_penalized_gradient=ratio,
        absolute_penalized_gradient=float(np.linalg.norm(gradient)),
        coefficient_l2_error_bound_exact_arithmetic=float(np.linalg.norm(gradient)/alpha),
        penalized_objective=float(residual@residual + alpha*(beta@beta)),
        scale_policy='WEIGHTED_FIT_STANDARD_SCALER',
        intercept_policy='WEIGHTED_FIT_FREE_INTERCEPT',
        rows=design.shape[0], columns=design.shape[1],
        chunk_bytes_bound=design.chunk_bytes_bound,
        numerical_backend=design.backend, working_bytes_bound=design.working_bytes_bound,
        numerical_workers=design.workers,
        preprocessing_backend=design.preprocessing_backend,
        atol=atol, btol=btol, max_iter=max_iter, gradient_tolerance=gradient_tolerance,
        standardized_fit_center_l2=float(np.linalg.norm(standardized_offset)))
    return PairRidge(beta, np.asarray(center).copy(), np.asarray(scale).copy(),
                     ymean-float(standardized_offset@beta), diagnostics, standardized_offset.copy())


def fit_ridge_path(design, response, weights, alphas, *, max_iter,
                   gradient_tolerance, max_basis_bytes, preprocessing=None,
                   checkpoint_path=None, checkpoint_identity=None,
                   checkpoint_every_iterations=1):
    """Reuse one fully reorthogonalized Krylov space for positive Ridge shifts.

    All systems share A=Z'WZ and rhs=Z'W(y-ybar); only alpha*I differs.
    The Galerkin solution in span(rhs,A rhs,...) is computed for every requested
    alpha. Each returned model must also pass its own explicit full-operator
    gradient check. No truncation is accepted as a new model or biological
    dimension. A cap/memory/numerical failure rejects the entire requested path.
    This reduces repeated operator applications, never the parameter domain.

    Normal-equation Krylov can be less stable than LSQR for weak penalties;
    double reorthogonalization and actual gradient checks do not guarantee
    coefficient accuracy in an ill-conditioned system. Dense/SVD equivalence
    and numerical sensitivity remain required before admitting a human use.
    """
    weights = check_weights(weights, design.shape[0])
    y = np.asarray(response, dtype=np.float64)
    require(y.shape == weights.shape and np.isfinite(y).all(), 'Invalid response')
    alpha_values = [float(a) for a in alphas]
    require(alpha_values and len(set(alpha_values)) == len(alpha_values) and
            all(math.isfinite(a) and a > 0 for a in alpha_values),
            'Distinct positive finite penalties required')
    require(type(max_iter) is int and max_iter > 0 and 0 < gradient_tolerance < 1
            and type(max_basis_bytes) is int and max_basis_bytes > 0 and
            type(checkpoint_every_iterations) is int and checkpoint_every_iterations > 0,
            'Explicit positive path solver limits required')
    require((checkpoint_path is None)==(checkpoint_identity is None),
            'Checkpoint path and identity are required together')
    n = design.shape[1]
    limit = min(max_iter, n)
    # Q and AQ plus their temporary growth copies and small projected systems.
    bound = 8*(8*n*limit + 6*limit*limit + 12*n)
    require(bound <= max_basis_bytes, 'Krylov basis memory bound exceeded')
    center, scale = (design.fit_preprocessing(weights) if preprocessing is None
                     else preprocessing)
    op, standardized_offset = design.ridge_operator(weights, center, scale)
    ymean = float(np.average(y, weights=weights))
    b = np.sqrt(weights)*(y-ymean)
    rhs = op.rmatvec(b)
    rhsnorm = float(np.linalg.norm(rhs))
    require(np.isfinite(rhs).all() and math.isfinite(rhsnorm), 'Nonfinite path rhs')
    settings=dict(max_iter=max_iter,gradient_tolerance=gradient_tolerance,
                  max_basis_bytes=max_basis_bytes,
                  checkpoint_every_iterations=checkpoint_every_iterations)
    fingerprint=(None if checkpoint_path is None else ridge_path_fingerprint(
        design,y,weights,alpha_values,settings,checkpoint_identity))
    saved=(None if checkpoint_path is None else load_ridge_path_checkpoint(
        checkpoint_path,fingerprint,alpha_values,n))
    if saved is None:
        models={};q=np.zeros(n) if rhsnorm==0 else rhs/rhsnorm
        basis,images=[],[];operator_applications=0;completed=0
    else:
        models=saved['models'];q=saved['q'];basis=saved['basis'];images=saved['images']
        operator_applications=saved['operator_applications'];completed=saved['iteration']
        if len(models)==len(alpha_values):return {a:models[a] for a in alpha_values}
    pending=set(alpha_values)-set(models)
    for iteration in range(completed+1, limit+1):
        if rhsnorm == 0:
            coefficients = {a: np.zeros(n) for a in pending}
        else:
            aq = op.rmatvec(op @ q)
            operator_applications += 1
            require(np.isfinite(aq).all(), 'Nonfinite normal operator')
            basis.append(q.copy())
            images.append(aq)
            Q, AQ = np.column_stack(basis), np.column_stack(images)
            projected = Q.T @ AQ
            projected = (projected+projected.T)/2
            values, vectors = eigh(projected, check_finite=True)
            transformed = vectors.T @ (Q.T @ rhs)
            coefficients = {}
            for alpha in sorted(pending, reverse=True):
                require(np.all(values+alpha > 0), 'Projected Ridge system not positive definite')
                small = vectors @ (transformed/(values+alpha))
                beta = Q @ small
                estimated = AQ @ small + alpha*beta-rhs
                denominator = max(rhsnorm, alpha*np.linalg.norm(beta), np.finfo(float).tiny)
                # Numerical headroom before a separately computed full check.
                if np.linalg.norm(estimated)/denominator <= gradient_tolerance/4:
                    coefficients[alpha] = beta
        for alpha, beta in coefficients.items():
            residual = op @ beta-b
            gradient = op.rmatvec(residual)+alpha*beta
            operator_applications += 1
            ratio = float(np.linalg.norm(gradient)/max(rhsnorm,
                          alpha*np.linalg.norm(beta), np.finfo(float).tiny))
            if not math.isfinite(ratio) or ratio > gradient_tolerance:
                continue
            diagnostic = dict(alpha=alpha, iterations=iteration,
                relative_penalized_gradient=ratio,
                absolute_penalized_gradient=float(np.linalg.norm(gradient)),
                coefficient_l2_error_bound_exact_arithmetic=float(np.linalg.norm(gradient)/alpha),
                penalized_objective=float(residual@residual+alpha*(beta@beta)),
                scale_policy='WEIGHTED_FIT_STANDARD_SCALER',
                intercept_policy='WEIGHTED_FIT_FREE_INTERCEPT',
                solver='SHARED_REORTHOGONALIZED_KRYLOV_EXPLICIT_GRADIENT',
                numerical_backend=design.backend, preprocessing_backend=design.preprocessing_backend,
                numerical_workers=design.workers,
                rows=design.shape[0], columns=n, gradient_tolerance=gradient_tolerance,
                max_iter=max_iter, basis_bytes_bound=bound,
                normal_operator_applications_to_acceptance=operator_applications,
                alpha_domain=alpha_values,
                standardized_fit_center_l2=float(np.linalg.norm(standardized_offset)))
            models[alpha] = PairRidge(beta, np.asarray(center).copy(),
                                     np.asarray(scale).copy(),
                                     ymean-float(standardized_offset@beta), diagnostic,
                                     standardized_offset.copy())
            pending.remove(alpha)
        if not pending:
            for model in models.values():
                model.diagnostics['path_normal_operator_applications'] = operator_applications
                model.diagnostics['path_basis_dimension'] = len(basis)
            if checkpoint_path is not None:
                write_ridge_path_checkpoint(checkpoint_path,fingerprint,iteration,q,basis,images,
                                            operator_applications,models,alpha_values)
            return {a: models[a] for a in alpha_values}
        if rhsnorm == 0:
            break
        new = aq.copy()
        # Two modified projection passes prevent loss of orthogonality when
        # the spectral range is large; no eigenmode is intentionally dropped.
        for _ in range(2):
            new -= Q @ (Q.T @ new)
        norm = float(np.linalg.norm(new))
        if norm <= 64*np.finfo(float).eps*max(np.linalg.norm(aq), np.finfo(float).tiny):
            break
        q = new/norm
        if checkpoint_path is not None and iteration%checkpoint_every_iterations==0:
            write_ridge_path_checkpoint(checkpoint_path,fingerprint,iteration,q,basis,images,
                                        operator_applications,models,alpha_values)
    raise ValueError('Ridge path did not converge for penalties: '+str(sorted(pending)))
