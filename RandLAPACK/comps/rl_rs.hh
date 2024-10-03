#pragma once

#include "rl_orth.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <cstdio>

namespace RandLAPACK {

template <typename T, typename RNG>
class RowSketcher
{
    public:
        virtual ~RowSketcher() {}

        virtual int call(
            int64_t m,
            int64_t n,
            const T* &A,
            int64_t k,
            T* &Omega,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class RS : public RowSketcher<T, RNG>
{
    public:

        RS(
            // Requires a stabilization algorithm object.
            RandLAPACK::Stabilization<T> &stab_obj,
            int64_t p,
            int64_t q,
            bool verb,
            bool cond
        ) : Stab_Obj(stab_obj) {
            verbose = verb;
            cond_check = cond;
            passes_over_data = p;
            passes_per_stab = q;
        }

        /// Return an n-by-k matrix Omega for use in sketching the rows of the m-by-n
        /// matrix A. (I.e., for computing a sketch Y = A @ Omega.) The qualitative goal
        /// is that the range of Omega should be well-aligned with the top-k right
        /// singular vectors of A.
        /// This function works by taking "passes_over_data" steps of a power method that
        /// starts with a random Gaussian matrix, and then makes alternating
        /// applications of A and A.T. We stabilize the power method with a user-defined method.
        /// This algorithm is shown in "the RandLAPACK book" book as Algorithm 8.
        ///
        ///    This implementation is inspired by [ZM:2020, Algorithm 3.3]. The most
        ///    significant difference is that this function stops one step "early",
        ///    so that it returns a matrix Omega for use in sketching Y = A @ Omega, rather than
        ///    returning an orthonormal basis for a sketched matrix Y. Here are the
        ///    differences between this implementation and [ZM:2020, Algorithm 3.3],
        ///    assuming the latter algorithm was modified to stop "one step early" like
        ///    this algorithm:
        ///       (1) We make no assumptions on the distribution of the initial
        ///            (oblivious) sketching matrix. [ZM:2020, Algorithm 3.3] uses
        ///            a Gaussian distribution.
        ///        (2) We allow any number of passes over A, including zero passes.
        ///            [ZM2020: Algorithm 3.3] requires at least one pass over A.
        ///        (3) We let the user provide the stabilization method. [ZM:2020,
        ///            Algorithm 3.3] uses LU for stabilization.
        ///        (4) We let the user decide how many applications of A or A.T
        ///            can be made between calls to the stabilizer.
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] n
        ///     The number of columns in the matrix A.
        ///
        /// @param[in] A
        ///     The m-by-n matrix A, stored in a column-major format.
        ///
        /// @param[in] k
        ///     Column size of the sketch.
        ///
        /// @param[in] Omega
        ///     Sketching operator buffer.
        ///
        /// @param[out] Omega
        ///     Stores m-by-k matrix, range(Omega) is
        ///     "reasonably" well aligned with A's leading left singular vectors.
        ///
        /// @return = 0: successful exit
        ///

        int call(
            int64_t m,
            int64_t n,
            const T* &A,
            int64_t k,
            T* &Omega,
            RandBLAS::RNGState<RNG> &state
        ) override;

        RandLAPACK::Stabilization<T> &Stab_Obj;
        int64_t passes_over_data;
        int64_t passes_per_stab;
        bool verbose;
        bool cond_check;
        std::vector<T> cond_nums;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int RS<T, RNG>::call(
    int64_t m,
    int64_t n,
    const T* &A,
    int64_t k,
    T* &Omega,
    RandBLAS::RNGState<RNG> &state
){

    int64_t p = this->passes_over_data;
    int64_t q = this->passes_per_stab;
    int64_t p_done= 0;

    T* Omega_1  = ( T * ) calloc( m * k, sizeof( T ) );

    if (p % 2 == 0) {
        // Fill n by k Omega
        RandBLAS::DenseDist D(n, k);
        state = RandBLAS::fill_dense(D, Omega, state);
    } else {
        // Fill m by k Omega_1
        RandBLAS::DenseDist D(m, k);
        state = RandBLAS::fill_dense(D, Omega_1, state);

        // multiply A' by Omega results in n by k omega
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, Omega_1, m, 0.0, Omega, n);

        ++ p_done;
        if ((p_done % q == 0) && (this->Stab_Obj.call(n, k, Omega)))
            return 1; // Scheme failure
    }

    while (p - p_done > 0) {
        // Omega = A * Omega
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Omega, n, 0.0, Omega_1, m);
        ++ p_done;

        if(this->cond_check)
            this->cond_nums.push_back(util::cond_num_check(m, k, Omega_1, this->verbose));

        if ((p_done % q == 0) && (this->Stab_Obj.call(m, k, Omega_1)))
            return 1;

        // Omega = A' * Omega
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, Omega_1, m, 0.0, Omega, n);
        ++ p_done;

        if (this->cond_check)
            this->cond_nums.push_back(util::cond_num_check(n, k, Omega, this->verbose));

        if ((p_done % q == 0) && (this->Stab_Obj.call(n, k, Omega)))
            return 1;
    }

    free(Omega_1);
    //successful termination
    return 0;
}

} // end namespace RandLAPACK
