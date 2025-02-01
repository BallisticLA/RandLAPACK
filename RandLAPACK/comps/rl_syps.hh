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

namespace RandLAPACK {


template <typename SYPS_t, typename Op, typename T, typename RNG>
concept SymmetricPowerSketchConcept = 
    linops::SymmetricLinearOperator<Op> && // Ensure Op conforms to the concept
    requires(SYPS_t obj, Uplo uplo, int64_t m, const T* A, int64_t lda, int64_t k, RandBLAS::RNGState<RNG> &state, T* skop_buff, T* work_buff) {
        // First version of call
        { obj.call(uplo, m, A, lda, k, state, skop_buff, work_buff) } -> std::same_as<int>;
        // Second version of call, templated on a type that satisfies SymmetricLinearOperator
        { obj.call(std::declval<Op>(), k, state, skop_buff, work_buff) } -> std::same_as<int>;
};


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
            int64_t m = A.m;
            int64_t p = this->passes_over_data;
            int64_t q = this->passes_per_stab;
            int64_t p_done = 0;

            bool callers_skop_buff = skop_buff != nullptr;
            if (!callers_skop_buff)
                skop_buff = new T[m * k];
            RandBLAS::DenseDist D(m, k);
            state = RandBLAS::fill_dense(D, skop_buff, state);

            bool callers_work_buff = work_buff != nullptr;
            if (!callers_work_buff)
                work_buff = new T[m * k];
            RandBLAS::util::safe_scal(m * k, (T) 0.0, work_buff, 1);

            T *symm_out = work_buff;
            T *symm_in  = skop_buff;
            T *tau = new T[k]{};
            while (p - p_done > 0) {
                A(Layout::ColMajor, k, 1.0, symm_in, m, 0.0, symm_out, m);
                ++p_done;
                if (p_done % q == 0) {
                    if(lapack::geqrf(m, k, symm_out, m, tau)) {
                        delete [] tau;
                        throw std::runtime_error("GEQRF failed.");
                    }
                    lapack::ungqr(m, k, k, symm_out, m, tau);
                }
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
