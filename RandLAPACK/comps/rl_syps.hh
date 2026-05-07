#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_orth.hh"
#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <concepts>


namespace RandLAPACK {


template <typename T, typename RNG>
class SYPS {
    public:
        using scalar_t = T;
        using RNG_t    = RNG;
        int64_t passes_over_data;
        int64_t passes_per_stab;
        bool verbose;
        bool cond_check;
        std::vector<T> cond_nums;
        // 0 = Gaussian (DenseDist, default); 1 = SASO (SparseDist, vec_nnz per column)
        int sketch_type = 0;
        int vec_nnz     = 4;
        // Internal stabilization (applied between power iterations).
        // nullptr → use the historical hardcoded geqrf+ungqr path (HQRQ-equivalent).
        // Non-null → invoke the supplied Stabilization (e.g. CholQRQ) and fall back to
        //   the hardcoded geqrf+ungqr if it returns nonzero (e.g. CholQR Cholesky failure).
        // Power iteration produces increasingly column-correlated matrices; CholQR can
        // hit potrf failure on ill-conditioned inputs, so the fallback is intentional.
        Stabilization<T>* internal_stab = nullptr;
    
        SYPS(
            int64_t p, // number of passes
            int64_t q, // passes per stabilization
            bool verb,
            bool cond
        ) {
            verbose = verb;
            cond_check = cond;
            passes_over_data = p;
            passes_per_stab = q;
        }

        /// Return an m-by-k matrix Y for use in sketching a symmetric matrix mat(A).
        /// The qualitative goal is that the range of Y should be well-aligned with
        /// the dominant k eigenvectors of A. This function takes "passes_over_data" steps
        /// of a power method that starts with a random Gaussian matrix.
        /// We stabilize the power method with a user-defined function.
        ///
        /// @param[in] uplo
        ///     Uplo::Upper or Uplo::Lower.
        ///     The triangular part of mat(A) that we can read from A.
        ///
        /// @param[in] m
        ///     The matrix mat(A) is m-by-m.
        ///
        /// @param[in] A
        ///     Buffer that partially defines the matrix mat(A).
        ///
        /// @param[in] lda
        ///     Leading dimension of mat(A) when reading from A.
        ///
        /// @param[in] k
        ///     The number of columns in the sketch.
        ///
        /// @param[in] state
        ///     The RNGState used to define the data-oblivious sketching operator at the
        ///     start of the power iteration.a
        ///
        /// @param[out] skop_buff
        ///     Optional. If provided, must have length >= m*k, and upon exit its contents
        ///     will underpin the returned sketching operator. If not provided, we will
        ///     allocate this memory and assign ownership to the returned sketching operator.
        ///
        /// @param[in,out] work_buff
        ///     Optional. If provided, must have length >= m*k. If not provided, we
        ///     will allocate a buffer of size m*k and deallocate before we return.
        ///
        /// @returns
        ///     An RNGState that the calling function should use the next
        ///     time it needs an RNGState.
        ///
        int call(
            Uplo uplo,
            int64_t m,
            const T* A,
            int64_t lda,
            int64_t k,
            RandBLAS::RNGState<RNG> &state,
            T* &skop_buff,
            T* work_buff
        ) {
            linops::ExplicitSymLinOp<T> A_linop(m, uplo, A, lda, Layout::ColMajor);
            return call(A_linop, k, state, skop_buff, work_buff);
        }

        template <linops::SymmetricLinearOperator SLO>
        int call(
            SLO &A,
            int64_t k,
            RandBLAS::RNGState<RNG> &state,
            T* &skop_buff,
            T* work_buff
        ) {
            int64_t m = A.dim;
            int64_t p = this->passes_over_data;
            int64_t q = this->passes_per_stab;
            int64_t p_done = 0;

            bool callers_skop_buff = skop_buff != nullptr;
            if (!callers_skop_buff)
                skop_buff = new T[m * k];

            bool callers_work_buff = work_buff != nullptr;
            if (!callers_work_buff)
                work_buff = new T[m * k];
            RandBLAS::util::safe_scal(m * k, (T) 0.0, work_buff);

            T *symm_out = work_buff;
            T *symm_in  = skop_buff;
            T *tau = new T[k]{};

            // Inline lambda for the per-iter stabilization (deduplicates the path used
            // for both the sparse-first-iter shortcut and the regular dense loop).
            auto stabilize = [&](T* buf) {
                bool fallback = (this->internal_stab == nullptr);
                if (!fallback) {
                    fallback = (this->internal_stab->call(m, k, buf) != 0);
                }
                if (fallback) {
                    if (lapack::geqrf(m, k, buf, m, tau)) {
                        delete[] tau;
                        throw std::runtime_error("GEQRF failed.");
                    }
                    lapack::ungqr(m, k, k, buf, m, tau);
                }
            };

            if (sketch_type == 1) {
                // Sparse sketch path — apply A to the sparse Ω directly via the
                // SkOp overload of A (dense·sparse spmm), skipping the n×k
                // densification of Ω. T and sint_t (int64_t) must be explicit
                // since SparseDist::sample can't deduce T from the RNGState.
                auto S = RandBLAS::SparseDist(m, k, this->vec_nnz).sample<T, RNG, int64_t>(state);
                state = S.next_state;
                RandBLAS::fill_sparse(S);

                // SLOs that implement the SkOp-taking operator() get the fast
                // sparse-aware first matvec. Other SLOs (RegExplicitSymLinOp,
                // SparseSymLinOp without the SkOp overload, etc.) fall back
                // to the legacy densification.
                if constexpr (requires (SLO& a, decltype(S)& s) {
                                  a(Layout::ColMajor, k, (T)1.0, s, (T)0.0, symm_out, m);
                              }) {
                    A(Layout::ColMajor, k, (T)1.0, S, (T)0.0, symm_out, m);
                } else {
                    auto Scoo = RandBLAS::coo_view_of_skop(S);
                    std::fill(skop_buff, skop_buff + m * k, (T)0.0);
                    RandLAPACK::util::sparse_to_dense(Scoo, Layout::ColMajor, skop_buff);
                    A(Layout::ColMajor, k, (T)1.0, skop_buff, m, (T)0.0, symm_out, m);
                }
                ++p_done;
                if (p_done % q == 0) stabilize(symm_out);
                symm_out = (p_done % 2 == 1) ? skop_buff : work_buff;
                symm_in  = (p_done % 2 == 1) ? work_buff : skop_buff;
            } else {
                // Dense Gaussian sketch — fill skop_buff in place; first matvec
                // happens uniformly in the loop below.
                RandBLAS::DenseDist D(m, k);
                state = RandBLAS::fill_dense(D, skop_buff, state);
            }

            // Remaining iterations are pure dense matvecs through the SLO interface.
            while (p - p_done > 0) {
                A(Layout::ColMajor, k, 1.0, symm_in, m, 0.0, symm_out, m);
                ++p_done;
                if (p_done % q == 0) stabilize(symm_out);
                symm_out = (p_done % 2 == 1) ? skop_buff : work_buff;
                symm_in  = (p_done % 2 == 1) ? work_buff : skop_buff;
            }
            delete[] tau;
            if (p % 2 == 1)
                blas::copy(m * k, work_buff, 1, skop_buff, 1);

            if (!callers_work_buff)
                delete[] work_buff;

            return 0;
        }
    
};


} // end namespace RandLAPACK
