import numpy as np
import scipy.io as scio
import scipy.sparse as spar
import scipy.linalg as la
import qdldl
from typing import TypeAlias, Callable, Optional
from matplotlib import pyplot as plt
import pickle as pkl


spmatrix : TypeAlias = spar.coo_matrix | spar.csc_matrix | spar.csr_matrix


paths : list[tuple[str, int]] = [
    ('/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/weir-20k/abs-tol/t=0.00263982s/amd_false', 247,),
    ('/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/weir-20k/abs-tol/t=25.8342s/amd_false', 294),
    ('/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/cap-rise-64k/abs-tol/t=0.019081s/amd_false',541),
    ('/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/sloshing-266k/abs-tol/t=0.00117647s/amd_false',500),
    ('/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/sloshing-2123k/abs-tol/t=0.00117647s/amd_false',5000),
    ('/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/erdc-staged/cap-rise-1024k/abs-tol/t=1.22302e-05s/amd_false',2138)
]


__LSNS__ = {
    'weir-20k/abs-tol/t=0.00263982s/amd_true': 'weir-1\nAMD',
    'weir-20k/abs-tol/t=25.8342s/amd_true': 'weir-2\nAMD',
    'cap-rise-64k/abs-tol/t=0.019081s/amd_true': 'cap-1\nAMD',
    'cap-rise-1024k/abs-tol/t=1.22302e-05s/amd_true': 'cap-2\nAMD',
    'sloshing-266k/abs-tol/t=0.00117647s/amd_true': 'slosh-1\nAMD',
    'sloshing-2123k/abs-tol/t=0.00117647s/amd_true': 'slosh-2\nAMD',
    'weir-20k/abs-tol/t=0.00263982s/amd_false': 'weir-1\nNatural',
    'weir-20k/abs-tol/t=25.8342s/amd_false': 'weir-2\nNatural',
    'cap-rise-64k/abs-tol/t=0.019081s/amd_false': 'cap-1\nNatural',
    'cap-rise-1024k/abs-tol/t=1.22302e-05s/amd_false': 'cap-2\nNatural',
    'sloshing-266k/abs-tol/t=0.00117647s/amd_false': 'slosh-1\nNatural',
    'sloshing-2123k/abs-tol/t=0.00117647s/amd_false': 'slosh-2\nNatural'
}

def linear_system_name(p: str):
    parts = p.split('erdc-staged/')
    return __LSNS__[parts[1]]


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


def as_compressed(A: spmatrix, format:str='csc'):
    if A.format in {'csc', 'csr'}:
        return A
    elif format == 'csc':
        return A.tocsc()
    elif format == 'csr':
        return A.tocsr()
    raise ValueError()


def read_system(p: str) -> tuple[spmatrix, np.ndarray, np.ndarray, np.ndarray]:
    A  = read_and_validate_A(p) 
    b  = scio.mmread(p + '/b.mtx').ravel()
    scale_a = la.norm(A.data, ord=np.inf)
    scale_b = la.norm(b, ord=np.inf)
    scale = scale_a ** 0.5 * scale_b ** 0.5
    x0 = scio.mmread(p + '/x0.mtx').ravel()
    A.data /= scale
    b /= scale
    xf = scio.mmread(p + '/xf.mtx').ravel()
    print(p)
    print(openfoam_residual(x0, A, b))
    print(openfoam_residual(xf, A, b))
    return A, b, x0, xf


def parsed_residuals(t: tuple[str, int]):
    p, iters = t
    iters = max(iters, 550)
    if iters == 5000:
        iters = 1000
    with open(p + '/residuals.pkl', 'rb') as jar:
        residuals = pkl.load(jar)
        parsed = dict()
        for k, v in residuals.items():
            parsed[k] = v[:iters+1]
    xaxis = np.arange(iters + 1)
    return parsed, xaxis


if __name__ == '__main__':

    curve_colors = {
        'DIC preconditioning':    'k',
        'RIC preconditioning':    'b',
        'Jacobi preconditioning': 'r',
        'SSOR preconditioning':    'orange'
    }

    for t in paths:
        p = t[0]
        residuals, xaxis = parsed_residuals(t)

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(8.5, 3), sharex=True, constrained_layout=True)
        ax1.set_title('OpenFOAM residual')
        ax3.set_title('CG loss')
        lsn = linear_system_name(p)
        fig.suptitle(f'{lsn}', x=0.01, ha='left', y=0.9)
        log_order_to_ax_order = {0: 0, 1: 2, 2: 1}

        precnames = ['DIC', 'SSOR', 'RIC', 'Jacobi']
        for i, ax in enumerate([ax1, None, ax3]):
            if ax is None:
                continue
            lines = []
            for precname in precnames:
                resids = residuals[precname + ' preconditioning']
                arr = np.array([a[log_order_to_ax_order[i]] for a in resids])
                ell = ax.semilogy(xaxis, arr, color=curve_colors[precname + ' preconditioning'], label=precname)
                lines.extend(ell)
            if i == 0:
                ax.set_ylim([5e-13, None])
            ax.yaxis.minorticks_on()
            ax.xaxis.minorticks_on()
            ax.grid(which='major', linestyle='--', linewidth=0.5, color='k', alpha=0.75)
            ax.grid(which='minor', linestyle='--', linewidth=0.5, color='k', alpha=0.333)
            ax.set_xlabel('iteration')
        fig.legend(handles=lines, labels=precnames, loc='outside center left', ncol=1)

        print('Saving figure.')
        lsn = lsn.replace("\n","-")
        plt.savefig(p + f'/{lsn}.pdf')
        continue

    print()
