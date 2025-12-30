import numpy as np
import scipy.io as scio
import scipy.sparse as spar
import scipy.linalg as la
import scipy.sparse.linalg as sparla
from scipy.sparse.linalg import spsolve_triangular, LinearOperator, cg, spilu
import qdldl
from typing import TypeAlias, Callable, Optional
from matplotlib import pyplot as plt
import pickle as pkl


spmatrix : TypeAlias = spar.coo_matrix | spar.csc_matrix | spar.csr_matrix


paths = [
    '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/weir-20k/abs-tol/t=0.00263982s/amd_true',
    '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/weir-20k/abs-tol/t=25.8342s/amd_true',
    # '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/cap-rise-64k/abs-tol/t=0.019081s/amd_true',
    # '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/sloshing-266k/abs-tol/t=0.00117647s/amd_true',
    # '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/sloshing-2123k/abs-tol/t=0.00117647s/amd_true',
    # '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/cap-rise-1024k/abs-tol/t=1.22302e-05s/amd_true',
]


def linear_system_name(p: str):
    parts = p.split('erdc-staged/')
    return parts[1]


def read_and_validate_A(p: str) -> spmatrix:
    A = scio.mmread(p + '/A.mtx')
    d = A.diagonal()
    A_x_ones = A @ np.ones(d.size)
    offd_ok = np.all((A - spar.diags(d)).data <= 0)
    symm_ok = np.all((A - A.T).data == 0)
    diag_ok = np.all(d > 0)
    sums_ok = np.all(A_x_ones > - 1e-12 * d )
    if not all([diag_ok, sums_ok, offd_ok, symm_ok]):
        print('\nFailed SDDM check for\n' + p)
        print('diag ' + str(diag_ok))
        print('sums ' + str(sums_ok))
        print('offd ' + str(offd_ok))
        print('symm ' + str(symm_ok))
        raise ValueError()
    return A


def read_and_validate_C(p: str) -> spmatrix:
    C = scio.mmread(p + '/richol_C_seed_0.mtx')
    diag_ok = np.all(C.diagonal() > 0)
    triu_ok = np.all((C - spar.tril(C)).data == 0)
    if (not diag_ok) or (not triu_ok):
        print('\nInvalid preconditioner for\n' + p)
        print('diag : ' + str(diag_ok))
        print('triu : ' + str(triu_ok))
        raise ValueError()
    return C


def openfoam_residual(_x: np.ndarray, _A: spmatrix, _b: np.ndarray) -> np.floating:
    num = la.norm(_b - _A @ _x, 1)
    den = (la.norm(_A @ (_x - np.mean(_x)*np.ones(_x.shape)), 1) + la.norm(_b - _A @ (np.mean(_x)*np.ones(_x.shape)), 1))
    return num/den


def stateful_cg_callback(_A: spmatrix, _b: np.ndarray, x_direct: Optional[np.ndarray]=None) -> tuple[list, Callable[[np.ndarray],None]]:
    if x_direct is None:
        x_direct = qdldl.Solver(_A).solve(_b)
    log = []
    def callback(_x):
        vec = np.array([
            openfoam_residual(_x, _A, _b),
            np.sqrt((x_direct - _x) @ _A @ (x_direct - _x)),
            la.norm(_A @ _x - _b) / _x.size**0.5
        ])
        log.append(vec)
    return log, callback


def jacobi_factory(_A: spmatrix) -> LinearOperator:
    d = _A.diagonal()
    linop = LinearOperator(
        dtype=np.double, shape=_A.shape,
        matvec  = lambda v: v / d,
        rmatvec = lambda v: v / d
    )
    return linop


def ssor_factory(_A: spmatrix) -> LinearOperator:
    d = A.diagonal()**-0.5
    lower : spar.csc_matrix = spar.tril(_A, format='csc') @ spar.diags(d)
    upper : spar.csr_matrix = lower.T # type: ignore
    return inv_ctc_factory(upper)


def dic_factory(_A: spmatrix) -> LinearOperator:
    #
    #   See Figure 3.3 of https://www.netlib.org/templates/templates.pdf
    #
    #   The "D" in that algorithm is really inv(D) for the preconditioner M
    #   defined on page 40, under the heading **Simple cases: ILU(0) and D-ILU.**
    #
    d = _A.diagonal()
    A_csc : spar.csc_matrix = _A.tocsc()
    inds = A_csc.indices.copy()
    ptrs = A_csc.indptr.copy()
    vals = A_csc.data.copy()
    n = d.size
    for i in range(n):
        d[i] = 1/d[i]
        js = inds[ptrs[i]:ptrs[i+1]]
        js = js[js > i]
        if js.size == 0:
            continue
        d[js] -= d[i] * vals[js]**2
    lower : spar.csc_matrix = spar.tril(A_csc, format='csc')
    if not lower.has_canonical_format:
        lower.sum_duplicates()
        lower.sort_indices()
    lower.data[lower.indptr[:n]] = 1/d
    upper : spar.csr_matrix = lower.T # type: ignore
    return inv_ctc_factory(upper, d)


def inv_ctc_factory(c_upper: spmatrix, d=None) -> LinearOperator:
    if isinstance(c_upper, (spar.coo_array, spar.coo_matrix)):
        c_upper = c_upper.tocsr()
        # For some reason that I cannot FATHOM, using .tocsr()
        # instead of .tocsc() radically changes the iterate
        # trajectories from PCG. 
        #
        # Maybe related:
        #   https://github.com/scipy/scipy/issues/6603.
        #   https://github.com/scipy/scipy/issues/14091.
        #
        #   Apparently spsolve_triangular operates on a view
        #   of the input matrix. So a lower-triangular view
        #   of a matrix that is in fact upper-triangular would
        #   be a diagonal matrix.
        #
        #       Maybe /Users/rjmurr/mico3/envs/rb311b/lib/python3.11/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py::672 is ... wrong ???
        #       That seems like madness ???
        #
        #   x = np.random.randn(c_upper.shape[0])
        #   c_upper_csr = c_upper.copy().tocsr()
        #   c_upper_csc = c_upper.copy().tocsc()
        #   y_csc = spsolve_triangular(c_upper_csc.T, x, lower=True)
        #   y_csr = spsolve_triangular(c_upper_csr.T, x, lower=True)
        #   z_csc = spsolve_triangular(c_upper_csc, y_csc, lower=False)
        #   z_csr = spsolve_triangular(c_upper_csr, y_csr, lower=False)
        #   la.norm(y_csc - y_csr)  # very big
    def spcho_solve(vec):
        y = spsolve_triangular( c_upper.T, vec, lower=True  )
        if d is not None:
            y *= d
        x = spsolve_triangular( c_upper,     y, lower=False )
        return x
    linop = LinearOperator(
        dtype=np.double, shape=c_upper.shape,
        matvec  = lambda v: spcho_solve(v),
        rmatvec = lambda v: spcho_solve(v)
    )
    return linop


def ilu_factory(_A: spar.spmatrix) -> LinearOperator:
    # I assume the linear operator represented by
    # ilu_factorization is positive definite when 
    # _A is positive definite.
    ilu_factorization = spilu(_A)
    linop = LinearOperator(
        dtype=np.double, shape=_A.shape,
        matvec  = lambda v: ilu_factorization.solve(v),
        rmatvec = lambda v: ilu_factorization.solve(v)
    )
    return linop


def read_system(p: str) -> tuple[spmatrix, np.ndarray, np.ndarray]:
    A  = read_and_validate_A(p) 
    b  = scio.mmread(p + '/b.mtx').ravel()
    scale_a = la.norm(A.data, ord=np.inf)
    scale_b = la.norm(b, ord=np.inf)
    scale = scale_a ** 0.5 * scale_b ** 0.5
    x0 = scio.mmread(p + '/x0.mtx').ravel()
    A.data /= scale
    b /= scale
    return A, b, x0


def read_richol_preconditioner(p: str) -> LinearOperator:
    C = read_and_validate_C(p)
    M = inv_ctc_factory(C)
    return M


factories = {
    'SSOR preconditioning' : ssor_factory,
    'DIC preconditioning' : dic_factory,
    'ILU preconditioning' : ilu_factory,
    'Jacobi preconditioning' : jacobi_factory
}

curve_colors = {
    'DIC preconditioning':    'k',
    'RIC preconditioning':    'b',
    'Jacobi preconditioning': 'r',
    'SSOR preconditioning':    'orange'
}


iters = 500

for p in paths:
    A, b, x0 = read_system(p)
    x_direct = qdldl.Solver(A).solve(b)

    residuals = dict()

    # run PCG with classical preconditioners
    for k, f in {'DIC preconditioning' : dic_factory, 
                 'SSOR preconditioning' : ssor_factory,
                 'Jacobi preconditioning' : jacobi_factory }.items():
        cur_resid, callback = stateful_cg_callback(A, b, x_direct)
        callback(x0)
        M = f(A)
        _ = cg(A, b, x0, rtol=0, atol=0, maxiter=iters, M=M, callback=callback)
        residuals[k] = cur_resid
    
    cur_resid, callback = stateful_cg_callback(A, b, x_direct)
    callback(x0)
    M = read_richol_preconditioner(p)
    _ = cg(A, b, x0, rtol=0, atol=0, maxiter=iters, M=M, callback=callback)
    residuals['RIC preconditioning'] = cur_resid

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6), sharex=True)
    ax1.set_title('OpenFOAM residual of $x_k$ (see $\S 5.4$ of $Notes~on~CFD$)')
    ax2.set_title('CG minimization objective $\\| A^{1/2}(x_k - A^{-1}b)\\|$')
    ax3.set_title('Root MSE of residual $\\| A x_k - b\\|_2 / \\sqrt{n}$')
    fig.suptitle('Error trajectories of PCG iterates in solving the "p_rgh" system from ' + linear_system_name(p))
    xaxis = np.arange(iters + 1)

    for i, ax in enumerate([ax1, ax2, ax3]):
        for precname in ['DIC preconditioning', 'SSOR preconditioning', 'RIC preconditioning', 'Jacobi preconditioning']:
            resids = residuals[precname]
            arr = np.array([a[i] for a in resids])
            ax.semilogy(xaxis, arr, color=curve_colors[precname], label=precname)
        ax.yaxis.minorticks_on()
        ax.xaxis.minorticks_on()
        ax.grid(which='major', linestyle='--', linewidth=0.5, color='k', alpha=0.75)
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='k', alpha=0.333)
        ax.legend(facecolor='white', framealpha=1, loc='lower left')
        ax.set_xlabel('k')

    with open(p + '/residuals.pkl', 'wb') as jar:
        pkl.dump(residuals, jar)
    plt.tight_layout()
    plt.savefig(p + '/fig.pdf')
    continue


print()
