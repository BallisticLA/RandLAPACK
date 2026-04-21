import sys
sys.path.insert(0, '../build/python')
import numpy as np
import scipy.linalg as la
from qrbbrp import run_basic



def validated_qrbbrp(A, b_sz):
    """Run QRBBRP and check the results."""
    A_copy = A.copy(order='F')
    n = A.shape[1]
    J, _, _ = run_basic(A_copy, b_sz, overwrite_a=True)
    # The call to QRBBRP should have overwritten the first n columns of A_copy with R
    # from the R-factor of unpivoted QR on A_copy[:, J]. We can check this against the R-factor from unpivoted QR on A[:, J].
    R_boring = la.qr(A[:, J], mode='r')[0]
    R_qrbbrp = np.triu(A_copy[:n, :])
    diffs = []
    for i in range(n):
        rb = R_boring[i, :]
        rq = R_qrbbrp[i, :]
        x = la.norm(rb - rq)
        y = la.norm(rb + rq)
        diffs.append(min(x, y))
    diffs = np.array(diffs)
    print(f"QRBBRP vs. QR difference: {diffs.max():.2e}")
    return (J, R_qrbbrp, diffs)


rng = np.random.default_rng(0)
m, n, b = 300, 120, 6
# Matrix with polynomial singular value decay so pivot order is non-trivial
U, _, Vt = np.linalg.svd(rng.standard_normal((m, n)), full_matrices=False)
s = (1 + np.arange(n, dtype=float)) ** -0.5   # sigma_k ~ k^{-0.5}
A = np.asfortranarray((U * s) @ Vt)            # column-major float64

temp = validated_qrbbrp(A, 6)
# QRBBRP vs. QR difference: 3.61e-16
