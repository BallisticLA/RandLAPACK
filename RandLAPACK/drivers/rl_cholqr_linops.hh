#pragma once

#include "rl_cholqr.hh"

#include <chrono>
#include <cstdint>

using namespace std::chrono;

namespace RandLAPACK {

/// Cholesky QR for abstract linear operators: G = A^T A, G = R^T R.
///
/// A may be any type satisfying the LinearOperator concept (dense, sparse, composite, ...);
/// the Gram is formed through A * B and A^T * B on column blocks, so at most m x block_size of
/// A is formed at a time when block_size < n (all of it otherwise).
/// Q is not formed; with test_mode on, Q = A * R^{-1} is formed for verification, from A's
/// columns (see form_q_from_r) and a triangular solve, outside the timed region.
///
/// The first attempt is unshifted. If potrf breaks down, the Cholesky is retried on
/// G + s I with s seeded at eps * trace(G) and grown by shift_growth (see cholqr_primitive),
/// in which case R factors the shifted Gram and the shift record says so.
///
/// call() returns 0 on success, or 1 when the factorization failed; chol_status then holds
/// cholqr_primitive's return code.
///
/// times (6 entries): alloc (always 0), fwd, adj, chol, rest, total. For an operator that provides
/// materialize_cols, fwd times its column copy.
template <typename T>
class CholQR_linops : public CholQRLinopsBase<T, 1> {
    public:
        /// @param ep  unused; kept for call-site compatibility with the other QR drivers.
        CholQR_linops(bool time_subroutines, T ep, bool enable_test_mode = false)
            : CholQRLinopsBase<T, 1>(time_subroutines, enable_test_mode) { (void)ep; }

        /// Computes the R factor of A = QR (Q not formed), A m x n with m >= n.
        /// @param[out] R    n x n buffer; on exit the upper-triangular factor, zeros below.
        /// @param[in]  ldr  leading dimension of R.
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(GLO& A, T* R, int64_t ldr) {
            return this->run_passes(A, R, ldr, /*sf1=*/T(0), /*sf_rest=*/T(0), /*gram_left_gemm=*/false);
        }
};


/// CholQR2 for abstract linear operators: CholQR, then a second CholQR pass preconditioned on
/// the first factor (trsm_identity), returning the accumulated R = R_2 R_1.
///
/// Both passes start unshifted (shift_factor_iter1 = shift_factor_iter2 = 0); a potrf breakdown
/// is rescued by the adaptive shift as in CholQR_linops.
///
/// call() returns 0 on success, or the 1-based pass that failed (1 or 2; an invalid shift factor
/// fails as pass 1 before any work); chol_status then holds cholqr_primitive's return code for
/// that pass.
///
/// times (13 entries): alloc (always 0), fwd1, adj1, chol1, upd1 (always 0), fwd2, adj2,
/// left2, chol2, upd2, q_mat, rest, total. For an operator that provides materialize_cols, fwd1
/// times its column copy.
template <typename T>
class CholQR2_linops : public CholQRLinopsBase<T, 2> {
    public:
        /// First-attempt shift factors of passes 1 and 2 (s = factor * trace(G)). No sentinel: a
        /// negative or NaN value fails the call with kCholQRInvalidShift.
        T    shift_factor_iter1 = T(0);
        T    shift_factor_iter2 = T(0);
        bool gram_left_gemm     = false;  ///< pass 2: apply R_pre^T by GEMM per block instead of one TRSM

        /// @param ep  unused; kept for call-site compatibility with the other QR drivers.
        CholQR2_linops(bool time_subroutines, T ep, bool enable_test_mode = false)
            : CholQRLinopsBase<T, 2>(time_subroutines, enable_test_mode) { (void)ep; }

        template <RandLAPACK::linops::LinearOperator GLO>
        int call(GLO& A, T* R, int64_t ldr) {
            return this->run_passes(A, R, ldr, shift_factor_iter1, shift_factor_iter2, gram_left_gemm);
        }
};


} // end namespace RandLAPACK
