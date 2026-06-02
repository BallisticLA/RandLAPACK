#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "../comps/rl_cholqr.hh"

#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std::chrono;

namespace RandLAPACK {

/// Cholesky QR factorization for abstract linear operators.
///
/// Thin wrapper around `cholqr_primitive` (Algorithm 1 from the collaborator's spec):
///     G = A^T A,  G = R^T R via Cholesky.
///
/// A may be any type satisfying the LinearOperator concept (dense, sparse, composite,
/// etc.). Q is not computed by default; when test_mode is enabled, Q = A * R^{-1}
/// is materialized for verification.
///
template <typename T>
class CholQR_linops {
    public:

        bool timing;
        bool test_mode;
        T eps;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // 6 entries: alloc, fwd, adj, chol, rest, total
        std::vector<long> times;

        // Column-block size for Gram and Q materialization. <=0 or >=n means no blocking.
        int64_t block_size;

        CholQR_linops(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            eps = ep;
            block_size = 0;
            test_mode = enable_test_mode;
            Q = nullptr;
            Q_rows = 0;
            Q_cols = 0;
        }

        ~CholQR_linops() {
            if (Q != nullptr) {
                delete[] Q;
            }
        }

        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr
        ) {
            steady_clock::time_point t0, t1, total_t_start, total_t_stop;
            long alloc_dur = 0, fwd_dur = 0, adj_dur = 0, chol_dur = 0, total_dur = 0;
            long q_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            int info = cholqr_primitive<T, GLO>(
                A, R, ldr,
                /*shift_factor=*/T(0),
                this->block_size,
                fwd_dur, adj_dur, chol_dur, this->timing);

            if (info != 0) {
                return info;
            }

            // Test mode: materialize Q = A * R^{-1}. Recompute A * I into a full m×n buffer
            // (outside the timing region) then apply R^{-1} via trsm.
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();

                T* Q_buf = new T[m * n];
                T* I_mat = new T[n * n]();
                RandLAPACK::util::eye(n, n, I_mat);

                for (int64_t j = 0; j < n; j += b_eff) {
                    int64_t b_j = std::min(b_eff, n - j);
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, b_j, n, (T)1.0, I_mat + j * n, n, (T)0.0, Q_buf + j * m, m);
                }
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, m, n, (T)1.0, R, ldr, Q_buf, m);

                delete[] I_mat;
                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;

                if (this->timing) { t1 = steady_clock::now(); q_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            if (this->timing) {
                total_t_stop = steady_clock::now();
                total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                // Subtract test-mode Q materialization (not part of algorithmic cost).
                total_dur -= q_dur;

                long rest_dur = total_dur - (alloc_dur + fwd_dur + adj_dur + chol_dur);
                this->times = {alloc_dur, fwd_dur, adj_dur, chol_dur, rest_dur, total_dur};
            }

            return 0;
        }
};

} // end namespace RandLAPACK
