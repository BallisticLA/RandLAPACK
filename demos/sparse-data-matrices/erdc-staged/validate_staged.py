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
    '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/cap-rise-64k/abs-tol/t=0.019081s/amd_true',
    '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/sloshing-266k/abs-tol/t=0.00117647s/amd_true',
    # '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/sloshing-2123k/abs-tol/t=0.00117647s/amd_true',
    # '/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/cap-rise-1024k/abs-tol/t=1.22302e-05s/amd_true',
]


def linear_system_name(p: str):
    parts = p.split('erdc-staged/')
    return parts[1]


def is_sddm(A: spmatrix, sdd_reltol: float) -> bool:
    d = A.diagonal()
    A_x_ones = A @ np.ones(d.size)
    A_x_ones /= la.norm(A_x_ones, np.inf)
    offd_ok = np.all((A - spar.diags(d)).data <= 0)
    symm_ok = np.all((A - A.T).data == 0)
    diag_ok = np.all(d > 0)
    sums_ok = np.all(A_x_ones >= - sdd_reltol )
    all_ok  = bool(offd_ok and symm_ok and diag_ok and sums_ok)
    return all_ok


def read_and_validate_A(p: str, sdd_rtol=1e-12) -> spmatrix:
    A : spar.coo_matrix = scio.mmread(
        p + '/A.mtx', spmatrix=True
    ) # type: ignore
    d = A.diagonal()
    A_x_ones = A @ np.ones(d.size)
    offd_ok = np.all((A - spar.diags(d)).data <= 0)
    symm_ok = np.all((A - A.T).data == 0)
    diag_ok = np.all(d > 0)
    sums_ok = np.all(A_x_ones >= - sdd_rtol * d )
    if not all([diag_ok, sums_ok, offd_ok, symm_ok]):
        print('\nFailed SDDM check for\n' + p)
        print('diag ' + str(diag_ok))
        print('sums ' + str(sums_ok))
        print('offd ' + str(offd_ok))
        print('symm ' + str(symm_ok))
        raise ValueError()
    A.setdiag(d - np.minimum(A_x_ones, 0))
    return A


def read_and_validate_c_lower(p: str) -> spmatrix:
    c_lower : spar.coo_matrix = scio.mmread(
        p + '/richol_C_seed_0.mtx', spmatrix=True
    ) # type: ignore
    diag_ok = np.all(c_lower.diagonal() > 0)
    triu_ok = np.all(spar.triu(c_lower, k=1).data == 0)
    if (not diag_ok) or (not triu_ok):
        print('\nInvalid preconditioner for\n' + p)
        print('diag : ' + str(diag_ok))
        print('triu : ' + str(triu_ok))
        raise ValueError()
    return c_lower



def openfoam_residual(x: np.ndarray, A: spmatrix, b: np.ndarray) -> np.floating:
    x_const = np.mean(x)*np.ones(x.shape)
    num = la.norm(b - A @ x, 1)
    den = la.norm(b - A @ x_const, 1) + la.norm(A @ (x - x_const), 1)
    return num/den


def stateful_cg_callback(A: spmatrix, b: np.ndarray, x_direct: Optional[np.ndarray]=None) -> tuple[list, Callable[[np.ndarray], None]]:
    if x_direct is None:
        x_direct = qdldl.Solver(A).solve(b)
    log = []
    def callback(x):
        vec = np.array([
            openfoam_residual(x, A, b),
            np.sqrt((x_direct - x) @ A @ (x_direct - x)),
            la.norm(A @ x - b) / x.size**0.5
        ])
        log.append(vec)
    return log, callback


def jacobi_factory(A: spmatrix) -> LinearOperator:
    d = A.diagonal()
    linop = LinearOperator(
        dtype=np.double, shape=A.shape,
        matvec  = lambda v: v / d,
        rmatvec = lambda v: v / d
    )
    return linop


def as_compressed(A: spmatrix, format:str='csc'):
    if A.format in {'csc', 'csr'}:
        return A
    elif format == 'csc':
        return A.tocsc()
    elif format == 'csr':
        return A.tocsr()
    raise ValueError()


def ssor_factory(A: spmatrix) -> LinearOperator:
    L : spar.csc_matrix = spar.tril(A, format='csc') # type: ignore
    d = A.diagonal()
    L = L @ spar.diags(d ** -0.5)
    return inv_cct_factory(L)


def dic_diagonal(A: spmatrix) -> np.ndarray:
    """
    Figure 3.3 of https://www.netlib.org/templates/templates.pdf shows
    how to compute the diagonal of "inv(D)" for the D-ILU preconditioner.
    This function adapts that method for SDDM matrices and constructs the
    diagonal of D (rather than that of inv(D)).
    """
    assert is_sddm(A, sdd_reltol=1e-15)
    a : spar.csc_matrix = as_compressed(A, 'csc')  # type: ignore
    d = a.diagonal()
    inds = a.indices
    ptrs = a.indptr
    vals = a.data
    for i in range(d.size):
        j = inds[ptrs[i]:ptrs[i+1]]
        v = vals[ptrs[i]:ptrs[i+1]]
        selector = j > i
        j = j[selector]
        v = v[selector]
        d[j] -= v**2 / d[i]
    return d


def dic_factory(A: spmatrix) -> LinearOperator:
    d = dic_diagonal(A)
    L : spar.csc_matrix = spar.tril(A, format='csc') # type: ignore
    L.setdiag(d)
    L = L @ spar.diags(d ** -0.5)
    return inv_cct_factory(L)


def inv_cct_factory(c_lower: spmatrix) -> LinearOperator:
    if isinstance(c_lower, (spar.coo_array, spar.coo_matrix)):
        c_lower = c_lower.tocsr() # type: ignore
    def spcho_solve(vec):
        y = spsolve_triangular( c_lower,   vec, lower=True  )
        x = spsolve_triangular( c_lower.T,   y, lower=False )
        return x
    linop = LinearOperator(
        dtype=np.double, shape=c_lower.shape,
        matvec  = lambda v: spcho_solve(v),
        rmatvec = lambda v: spcho_solve(v)
    )
    return linop


def ilu_factory(A: spar.spmatrix) -> LinearOperator:
    # I assume the linear operator represented by
    # ilu_factorization is positive definite when 
    # A is positive definite.
    ilu_factorization = spilu(A)
    linop = LinearOperator(
        dtype=np.double, shape=A.shape,
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
    C = read_and_validate_c_lower(p)
    M = inv_cct_factory(C)
    return M


if __name__ == '__main__':

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
