#pragma once

#include "rl_syps.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_orth.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <stdexcept>


namespace RandLAPACK {


template <typename SYRF_t, typename Op, typename T, typename RNG>
concept SymmetricRangeFinderConcept = 
    // Ensure Op conforms to the SymmetricLinearOperator concept
    linops::SymmetricLinearOperator<Op> &&
    requires(SYRF_t obj, Uplo uplo, int64_t m, const T* A, int64_t lda, int64_t k, std::vector<T> Q, RandBLAS::RNGState<RNG> &state, T* work_buff) {
        // First version of call
        { obj.call(uplo, m, A, lda, k, Q, state, work_buff) } -> std::same_as<int>;
        // Second version of call, templated on a type that satisfies SymmetricLinearOperator
        { obj.call(std::declval<Op>(), k, Q, state, work_buff) } -> std::same_as<int>;
};


template <typename SYPS_t>
class SYRF {
    public:
        using T = typename SYPS_t::scalar_t;
        using RNG = typename SYPS_t::RNG_t;
        SYPS_t &SYPS_Obj;
        RandLAPACK::Stabilization<T> &Orth_Obj;
        bool verbose;
        bool cond_check;
        std::vector<T> cond_work_mat;
        std::vector<T> cond_work_vec;
        std::vector<T> cond_nums; // Condition nubers of sketches

        SYRF(
            SYPS_t &syps_obj,
            RandLAPACK::Stabilization<T> &orth_obj,
            bool verb = false,
            bool cond = false
        ) : SYPS_Obj(syps_obj), Orth_Obj(orth_obj) {
            verbose = verb;
            cond_check = cond;
        }

        /// This is an analog of the RangeFinder class RF for symmetric matrices.
        ///
        /// @param[in] uplo
        ///     Uplo::Upper or Uplo::Lower.
        ///     The triangular part of mat(A) that we can read from A.
        ///
        /// @param[in] m
        ///     The matrix mat(A) is m-by-m.
        ///
        /// @param[in] A
        ///     The m-by-m matrix A, stored in a column-major format.
        ///
        /// @param[in] k
        ///     Column size of the sketch.
        ///
        /// @param[in,out] Q
        ///     On output, stores an m-by-k column-major column-orthonormal matrix,
        ///     where range(Q) is "reasonably" well aligned with A's dominant
        ///     k-dimensional eigenspace.
        ///
        /// @param[in] work_buff
        ///     Optional. If provided, must be size at least m*k.
        ///     If not provided, then we will allocate a buffer of size m*k and
        ///     deallocate before returning.
        ///
        /// @returns
        ///     An RNGState that the calling function should use the next
        ///     time it needs an RNGState.
        ///
        int call(
            Uplo uplo,
            int64_t m,
            const T* A,
            int64_t k,
            std::vector<T> &Q,
            RandBLAS::RNGState<RNG> &state,
            T* work_buff
        ) {
            linops::ExplicitSymLinOp<T> A_linop(m, uplo, A, m, Layout::ColMajor);
            return this->call(A_linop, k, Q, state, work_buff);
        }

        template <linops::SymmetricLinearOperator SLO>
        int call(
            SLO &A,
            int64_t k,
            std::vector<T> &Q,
            RandBLAS::RNGState<RNG> &state,
            T* work_buff
        ) {
            int64_t m = A.m;
            bool callers_work_buff = work_buff != nullptr;
            if (!callers_work_buff)
                work_buff = new T[m * k];

            RandBLAS::util::safe_scal(m * k, (T) 0.0, work_buff, 1);

            T* Q_dat = util::upsize(m * k, Q);
            SYPS_Obj.call(A, k, state, work_buff, Q_dat);

            // Q = orth(A * Omega)
            A(Layout::ColMajor, k, (T) 1.0, work_buff, m, (T) 0.0, Q_dat, m);
            if(this->cond_check) {
                util::upsize(m * k, this->cond_work_mat);
                util::upsize(k, this->cond_work_vec);
                this->cond_nums.push_back(
                    util::cond_num_check(m, k, Q.data(), this->verbose)
                );
            }
            if(this->Orth_Obj.call(m, k, Q.data()))
                throw std::runtime_error("Orthogonalization failed.");
            
            if (!callers_work_buff)
                delete[] work_buff;

            return 0;
        }

};



} // end namespace RandLAPACK
