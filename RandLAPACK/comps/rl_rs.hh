#ifndef randlapack_comps_rs_h
#define randlapack_comps_rs_h

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
            const std::vector<T> &A,
            int64_t k,
            std::vector<T> &Omega,
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
            verbosity = verb;
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
            const std::vector<T> &A,
            int64_t k,
            std::vector<T> &Omega,
            RandBLAS::RNGState<RNG> &state
        ) override;

        RandLAPACK::Stabilization<T> &Stab_Obj;
        int64_t passes_over_data;
        int64_t passes_per_stab;
        bool verbosity;
        bool cond_check;
        std::vector<T> Omega_1;
        std::vector<T> cond_nums;

        std::vector<T> Omega_cpy;
        std::vector<T> Omega_1_cpy;
        std::vector<T> s;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int RS<T, RNG>::call(
    int64_t m,
    int64_t n,
    const std::vector<T> &A,
    int64_t k,
    std::vector<T> &Omega,
    RandBLAS::RNGState<RNG> &state
){

    int64_t p = this->passes_over_data;
    int64_t q = this->passes_per_stab;
    int64_t p_done= 0;

    const T* A_dat = A.data();
    T* Omega_dat   = Omega.data();
    T* Omega_1_dat = util::upsize(m * k, this->Omega_1);

    if (p % 2 == 0) {
        // Fill n by k Omega
        RandBLAS::DenseDist D(n, k);
        state = RandBLAS::fill_dense(D, Omega_dat, state).second;
    } else {
        // Fill m by k Omega_1
        RandBLAS::DenseDist D(m, k);
        state = RandBLAS::fill_dense(D, Omega_1_dat, state).second;

        // multiply A' by Omega results in n by k omega
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);

        ++ p_done;
        if ((p_done % q == 0) && (this->Stab_Obj.call(n, k, Omega)))
            return 1; // Scheme failure
    }

    while (p - p_done > 0) {
        // Omega = A * Omega
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Omega_1_dat, m);
        ++ p_done;

        if(this->cond_check)
            this->cond_nums.push_back(util::cond_num_check(m, k, Omega_1.data(), (this->Omega_1_cpy).data(), (this->s).data(), this->verbosity));

        if ((p_done % q == 0) && (this->Stab_Obj.call(m, k, Omega_1)))
            return 1;

        // Omega = A' * Omega
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);
        ++ p_done;

        if (this->cond_check)
            this->cond_nums.push_back(util::cond_num_check(n, k, Omega.data(), (this->Omega_cpy).data(), (this->s).data(), this->verbosity));

        if ((p_done % q == 0) && (this->Stab_Obj.call(n, k, Omega)))
            return 1;
    }
    //successful termination
    return 0;
}

} // end namespace RandLAPACK
#endif
