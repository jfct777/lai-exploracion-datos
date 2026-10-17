#!/usr/bin/env python3
"""Exact mean-feature block preconditioning; the original Ridge objective stays fixed.

This is a numerical experiment, not an admitted human K03 fitting backend.
The remaining feature blocks receive scalar preconditioning. No feature,
singular value, person or penalty is removed from the statistical problem.
"""
import math
import time
import numpy as np
from scipy.linalg import eigh, svd
from scipy.sparse.linalg import LinearOperator, lsqr
from r02_pair_operator import PairRidge, check_weights, require


class MeanBlockGeometry:
    """Alpha-independent exact mean-block factorization reusable by a Ridge path."""
    def __init__(self, design, weights, center, scale, *, max_working_mib,
                 standardized_offset=None):
        w = check_weights(weights, design.shape[0])
        require(math.isfinite(max_working_mib) and max_working_mib > 0,
                'Explicit positive memory limit required')
        n, d = design.points.shape
        columns = design.shape[1]
        center, scale = np.asarray(center), np.asarray(scale)
        require(center.shape == scale.shape == (columns,) and
                np.isfinite(center).all() and np.isfinite(scale).all() and
                np.all(scale > 0), 'Frozen preprocessing differs')
        # Conservative input copies, square eigensolver and thin SVD workspaces.
        # Caller must add existing inputs, PairDesign and LSQR vectors separately.
        self.working_bytes_bound = 8*(10*n*n+12*n*d+6*d*min(n, d)+8*columns)
        require(self.working_bytes_bound <= max_working_mib*2**20,
                'Mean-block workspace exceeds explicit bound')
        started = time.monotonic()
        adjacency = np.zeros((n, n))
        np.add.at(adjacency, (design.left, design.right), w)
        np.add.at(adjacency, (design.right, design.left), w)
        degree = adjacency.sum(axis=1)
        adjacency[np.diag_indices(n)] += degree
        adjacency *= .25
        values, vectors = eigh(adjacency, check_finite=False, driver='evd')
        rounding_bound = 64*np.finfo(float).eps*n*max(float(np.max(degree))/2, 1.)
        require(float(values.min()) >= -rounding_bound,
                'Signless Laplacian is not positive semidefinite within rounding bound')
        # Only roundoff-negative eigenvalues of the analytically PSD row matrix
        # are clipped. All statistical features and all Ridge penalties remain.
        root = np.sqrt(np.maximum(values, 0.))[:, None]*vectors.T
        points = (design.points-center[:d])/scale[:d]
        if standardized_offset is not None:
            offset = np.asarray(standardized_offset, dtype=float)
            require(offset.shape == (columns,) and np.isfinite(offset).all(),
                    'Invalid standardized FIT mean')
            points -= offset[:d]
        factor = root @ points
        _, singular_values, right = svd(factor, full_matrices=False,
                                        check_finite=False, lapack_driver='gesdd')
        self.vectors = right.T
        self.singular_values = singular_values
        self.weight_sum = float(w.sum())
        self.axes, self.columns = d, columns
        self.diagnostics = dict(kind='EXACT_MEAN_BLOCK_SCALAR_REMAINDER',
            mean_axes=d, row_space=n,
            retained_singular_values=len(singular_values),
            smallest_mean_singular_value=float(singular_values.min()) if len(singular_values) else 0.,
            psd_roundoff_negative_eigenvalues=int(np.count_nonzero(values < 0)),
            psd_roundoff_bound=rounding_bound, working_bytes_bound=self.working_bytes_bound,
            seconds=time.monotonic()-started, scientific_features_removed=0)

    def apply(self, vector, alpha):
        value = np.asarray(vector, dtype=float).reshape(-1)
        require(value.shape == (self.columns,) and np.isfinite(value).all() and
                math.isfinite(alpha) and alpha > 0,
                'Invalid preconditioner vector')
        result = value/math.sqrt(self.weight_sum+alpha)
        coordinates = self.vectors.T @ value[:self.axes]
        # Explicit complement preserves a full-rank inverse when d > n.
        complement = value[:self.axes]-self.vectors @ coordinates
        result[:self.axes] = (self.vectors @ (coordinates/np.sqrt(self.singular_values**2+alpha)) +
                             complement/math.sqrt(alpha))
        return result


class MeanBlockPreconditioner:
    """Backward-compatible one-alpha view over reusable exact geometry."""
    def __init__(self, design, weights, center, scale, alpha, *, max_working_mib,
                 standardized_offset=None):
        require(math.isfinite(alpha) and alpha > 0, 'Explicit positive penalty required')
        self.geometry=MeanBlockGeometry(design,weights,center,scale,
            max_working_mib=max_working_mib,standardized_offset=standardized_offset)
        self.alpha=float(alpha)
        self.working_bytes_bound=self.geometry.working_bytes_bound
        self.diagnostics=dict(self.geometry.diagnostics,alpha=self.alpha)

    def apply(self, vector):
        return self.geometry.apply(vector,self.alpha)


def fit_mean_preconditioned(design, response, weights, alpha, *, atol, btol,
                           max_iter, gradient_tolerance, max_preconditioner_mib,
                           preprocessing=None):
    """Solve the same augmented weighted Ridge system, then check its residual."""
    return fit_mean_preconditioned_path(design,response,weights,[alpha],atol=atol,btol=btol,
        max_iter=max_iter,gradient_tolerance=gradient_tolerance,
        max_preconditioner_mib=max_preconditioner_mib,preprocessing=preprocessing)[float(alpha)]


def fit_mean_preconditioned_path(design, response, weights, alphas, *, atol, btol,
                                 max_iter, gradient_tolerance, max_preconditioner_mib,
                                 preprocessing=None, completed_models=None,
                                 candidate_callback=None, max_total_iter=None,
                                 pending_state=None, partial_callback=None):
    """Solve an exact finite Ridge family while reusing preprocessing and geometry."""
    w = check_weights(weights, design.shape[0])
    y = np.asarray(response, dtype=float)
    require(y.shape == w.shape and np.isfinite(y).all(), 'Invalid response')
    alpha_values=[float(a) for a in alphas]
    max_total_iter=max_iter if max_total_iter is None else max_total_iter
    require(alpha_values and len(set(alpha_values))==len(alpha_values) and
            all(math.isfinite(a) and a>0 for a in alpha_values) and 0 < atol < 1 and
            0 < btol < 1 and 0 < gradient_tolerance < 1 and
            type(max_iter) is int and max_iter > 0 and type(max_total_iter) is int and
            max_total_iter>=max_iter,'Invalid explicit numerical settings')
    center, scale = design.fit_preprocessing(w) if preprocessing is None else preprocessing
    op, standardized_offset = design.ridge_operator(w, center, scale)
    geometry=MeanBlockGeometry(design,w,center,scale,max_working_mib=max_preconditioner_mib,
                               standardized_offset=standardized_offset)
    # Preserve the exact zero-residual case.  A weighted average of an exactly
    # constant floating response can differ by one ulp and create a spurious,
    # scale-free gradient ratio even though beta=0 is the exact optimum.
    constant_response=bool(np.all(y==y[0]))
    mean = float(y[0]) if constant_response else float(np.average(y, weights=w))
    b = np.zeros_like(y) if constant_response else np.sqrt(w)*(y-mean)
    rows, columns = design.shape
    rhs = op.rmatvec(b)
    models={} if completed_models is None else dict(completed_models)
    require(set(models)<=set(alpha_values) and all(isinstance(models[a],PairRidge) for a in models),
            'Recovered Ridge candidates differ from requested path')
    require(pending_state is None or (pending_state['alpha'] in alpha_values and
            pending_state['alpha'] not in models and pending_state['x0'].shape==(columns,) and
            np.isfinite(pending_state['x0']).all() and
            0<pending_state['iterations']<max_total_iter and pending_state['segments']>0 and
            pending_state['status']==7),
            'Recovered LSQR iterate differs from requested path')
    for alpha in alpha_values:
        if alpha in models:
            continue
        root_alpha=math.sqrt(alpha)
        def matvec(vector):
            beta=geometry.apply(vector,alpha)
            return np.concatenate((op @ beta,root_alpha*beta))
        def rmatvec(vector):
            value=np.asarray(vector).reshape(-1)
            return geometry.apply(op.rmatvec(value[:rows])+root_alpha*value[rows:],alpha)
        augmented=LinearOperator((rows+columns,columns),matvec=matvec,rmatvec=rmatvec,dtype=np.dtype(float))
        x0=None;total_iterations=0;segments=0
        if pending_state is not None:
            require(pending_state['alpha']==alpha,
                    'Recovered LSQR iterate is not the next unfinished candidate')
            x0=pending_state['x0'];total_iterations=pending_state['iterations']
            segments=pending_state['segments'];pending_state=None
        augmented_rhs=np.concatenate((b,np.zeros(columns)))
        while True:
            remaining=max_total_iter-total_iterations
            require(remaining>0,'LSQR total iteration budget exhausted')
            result=lsqr(augmented,augmented_rhs,damp=0.,atol=atol,btol=btol,
                        conlim=0.,iter_lim=min(max_iter,remaining),show=False,x0=x0)
            status=int(result[1]);segment_iterations=int(result[2])
            require(segment_iterations>=0 and np.isfinite(result[0]).all() and
                    (status!=7 or segment_iterations>0),'LSQR returned invalid progress')
            total_iterations+=segment_iterations;segments+=1
            if status!=7:break
            state=dict(alpha=alpha,x0=np.asarray(result[0]).copy(),iterations=total_iterations,
                       segments=segments,status=7)
            if partial_callback is not None:partial_callback(state)
            if total_iterations>=max_total_iter:
                raise ValueError('Preconditioned LSQR did not converge: status=7; total_iterations='+
                                 str(total_iterations))
            x0=state['x0']
        require(result[1] in (0,1,2,4,5),'Preconditioned LSQR did not converge: status='+str(result[1]))
        beta=geometry.apply(result[0],alpha)
        residual=op @ beta-b
        gradient=op.rmatvec(residual)+alpha*beta
        ratio=float(np.linalg.norm(gradient)/max(np.linalg.norm(rhs),alpha*np.linalg.norm(beta),
                    np.finfo(float).tiny))
        require(math.isfinite(ratio) and ratio<=gradient_tolerance,
                'Original penalized gradient exceeds declared tolerance: '+str(ratio))
        diagnostics=dict(alpha=alpha,damping=root_alpha,lsqr_status=int(result[1]),
            iterations=total_iterations,last_segment_iterations=segment_iterations,
            continuation_segments=segments,max_total_iter=max_total_iter,
            condition_estimate=float(result[6]),
            relative_penalized_gradient=ratio,absolute_penalized_gradient=float(np.linalg.norm(gradient)),
            coefficient_l2_error_bound_exact_arithmetic=float(np.linalg.norm(gradient)/alpha),
            penalized_objective=float(residual@residual+alpha*(beta@beta)),
            scale_policy='WEIGHTED_FIT_STANDARD_SCALER',intercept_policy='WEIGHTED_FIT_FREE_INTERCEPT',
            rows=rows,columns=columns,numerical_backend=design.backend,numerical_workers=design.workers,
            working_bytes_bound=design.working_bytes_bound,preprocessing_backend=design.preprocessing_backend,
            atol=atol,btol=btol,max_iter=max_iter,gradient_tolerance=gradient_tolerance,
            preconditioner=dict(geometry.diagnostics,alpha=alpha),preconditioner_geometry_reused=len(alpha_values)>1,
            ridge_path_size=len(alpha_values),solver='LSQR_EXPLICIT_RIGHT_PRECONDITIONED_AUGMENTED_RIDGE',
            original_objective_verified=True,
            standardized_fit_center_l2=float(np.linalg.norm(standardized_offset)))
        models[alpha]=PairRidge(beta,np.asarray(center).copy(),np.asarray(scale).copy(),
            mean-float(standardized_offset@beta),diagnostics,standardized_offset.copy())
        if candidate_callback is not None:
            candidate_callback(models)
    return {a:models[a] for a in alpha_values}
