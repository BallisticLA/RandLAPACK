#ifndef randlapack_comps_rs_h
#define randlapack_comps_rs_h

#include "rl_orth.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>

#include <vector>
#include <cstdint>
#include <cstdio>

namespace RandLAPACK {

template <typename T>
class RowSketcher
{
    public:
        virtual ~RowSketcher() {}

        virtual int call(
            int64_t m,
            int64_t n,
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Omega
        ) = 0;
};

template <typename T, typename RNG>
class RS : public RowSketcher<T>
{
    public:

        RS(
            // Requires a stabilization algorithm object.
            RandLAPACK::Stabilization<T>& stab_obj,
            RandBLAS::base::RNGState<RNG> s,
            int64_t p,
            int64_t q,
            bool verb,
            bool cond
        ) : Stab_Obj(stab_obj) {
            verbosity = verb;
            cond_check = cond;
            st = s;
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
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Omega
        ) override;

        RandLAPACK::Stabilization<T>& Stab_Obj;
        RandBLAS::base::RNGState<RNG> st;
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
    const std::vector<T>& A,
    int64_t k,
    std::vector<T>& Omega
){

    int64_t p = this->passes_over_data;
    int64_t q = this->passes_per_stab;
    int64_t p_done= 0;

    const T* A_dat = A.data();
    T* Omega_dat = Omega.data();
    T* Omega_1_dat = util::upsize(m * k, this->Omega_1);
    auto state = this->st;

    if (p % 2 == 0) {
        // Fill n by k Omega
        RandBLAS::dense::DenseDist  D{.n_rows = n, .n_cols = k};
        RandBLAS::dense::fill_buff(Omega_dat, D, state);
    } else {
        // Fill m by k Omega_1
        RandBLAS::dense::DenseDist D{.n_rows = m, .n_cols = k};
        RandBLAS::dense::fill_buff(Omega_1_dat, D, state);

        // multiply A' by Omega results in n by k omega
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);

        ++ p_done;
        if ((p_done % q == 0) && (this->Stab_Obj.call(n, k, Omega_dat)))
            return 1; // Scheme failure
    }

    while (p - p_done > 0) {
        // Omega = A * Omega
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Omega_1_dat, m);
        ++ p_done;

        if(this->cond_check)
            this->cond_nums.push_back(util::cond_num_check(m, k, Omega_1, this->Omega_1_cpy, this->s, this->verbosity));

        if ((p_done % q == 0) && (this->Stab_Obj.call(m, k, Omega_1_dat)))
            return 1;

        // Omega = A' * Omega
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);
        ++ p_done;

        if (this->cond_check)
            this->cond_nums.push_back(util::cond_num_check(n, k, Omega, this->Omega_cpy, this->s, this->verbosity));

        if ((p_done % q == 0) && (this->Stab_Obj.call(n, k, Omega_dat)))
            return 1;
    }
    //successful termination
    return 0;
}

template <typename T, typename RNG>
class BK : public RowSketcher<T>
{
    public:

        BK(
            // Requires a stabilization algorithm object.
            RandLAPACK::Stabilization<T>& stab_obj,
            RandBLAS::base::RNGState<RNG> s,
            int64_t p,
            int64_t q,
            bool verb,
            bool cond
        ) : Stab_Obj(stab_obj) {
            verbosity = verb;
            cond_check = cond;
            st = s;
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
        /// @param[in] block_sz
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
            const std::vector<T>& A,
            int64_t block_sz,
            std::vector<T>& Omega
        ) override;

        RandLAPACK::Stabilization<T>& Stab_Obj;
        RandBLAS::base::RNGState<RNG> st;
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
int BK<T, RNG>::call(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    int64_t block_sz,
    std::vector<T>& Omega
){

    int64_t p = this->passes_over_data;
    int64_t q = this->passes_per_stab;
    int64_t p_done= 0;

    const T* A_dat = A.data();
    T* Omega_dat = Omega.data();
    auto state = this->st;

    std::vector<T> A_cpy(m * n, 0.0);
    blas::copy(m * n, A_dat, 1, A_cpy.data(), 1);

    char name1 [] = "A";
    RandBLAS::util::print_colmaj(m, n, A_cpy.data(), name1);

    // Number of columns in a sketching operator
    int64_t k = 0;
    // Number of Krylov iterations done
    int64_t q_done = 0;
    if (p % 2 == 0) {
        // Compute the sketch size from the number of passes & block size.
        // In this case, we have an expression (A'A)^q S.
        // Even number of passes over data, so k = block_sz / (p / 2).
        // When block_sz is not evenly divisible by q, we allow for 
        // a portion of the last (A'A)^q AS to be written into Krylov matrix.
        k = (int64_t) std::ceil((float) 2 * block_sz / p);

        // Allocate space for a computational buffer 
        T* Omega_1_dat = util::upsize(m * k, this->Omega_1);
        
        // Place an n by k Sketching operator buffer into the full Omega matrix, n by block_sz
        RandBLAS::dense::DenseDist  D{.n_rows = n, .n_cols = k};
        RandBLAS::dense::fill_buff(Omega_dat, D, state);
    } else {

        // Compute the sketch size from the number of passes & block size.
        // In this case, we have an expression (A'A)^q AS.
        // This means that there are potential q + 1 elements in the Block Krylov matrix
        // q = (p - 1) / 2
        // k = block_sz / q + 1 = block_sz / [(p - 1) / 2 + 1].
        // When block_sz is not evenly divisible by q+1, we allow for 
        // a portion of the last (A'A)^q AS to be written into Krylov matrix.
        k = (p == 1) ? block_sz : (int64_t) std::ceil(block_sz / (1 + ((p - 1) / (float) 2)));

        // Allocate space for a computational buffer 
        T* Omega_1_dat = util::upsize(m * k, this->Omega_1);
        // Fill m by k Omega_1
        RandBLAS::dense::DenseDist D{.n_rows = m, .n_cols = k};
        RandBLAS::dense::fill_buff(Omega_1_dat, D, state);

        char name2 [] = "Omega";
        RandBLAS::util::print_colmaj(m, block_sz, Omega_1.data(), name2);

        // Write an n by k product of A' Omega_1 into the full Omega matrix
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);

        char name4 [] = "A' * Omega";
        RandBLAS::util::print_colmaj(n, block_sz, Omega.data(), name4);

        ++ p_done;
        ++ q_done;
        // Need to take in a pointer
        if ((p_done % q == 0) && (this->Stab_Obj.call(n, k, Omega_dat)))
            return 1; // Scheme failure
    }

    char name3 [] = "qr(A' * Omega)";
    RandBLAS::util::print_colmaj(n, block_sz, Omega.data(), name3);

    // In case with (A'A)^q S, we will be performing block_sz / k iterations.
    // In case with (A'A)^q A'S, we will be performing block_sz / k - 1 iterations,
    // we account for that inside of a previous if statement.
    // We always allow for an additional iteration of the loop below
    // 'offset' - size of a matrix by which shift in the block Krylov matrix.
    // Need a separate variable for it, as k may change at the last iteration.
    int64_t offset = n * k;
    while (p - p_done > 0 && q_done < std::ceil(block_sz / (float) k)) {
        // Omega_1 = A * Omega[:, k * q_done : k * (q_done + 1)]
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega.data() + offset * q_done, n, 0.0, Omega_1.data(), m);
        ++ p_done;

        if(this->cond_check)
            this->cond_nums.push_back(util::cond_num_check(m, k, Omega_1, this->Omega_1_cpy, this->s, this->verbosity));

        // Need to take in a pointer
        if ((p_done % q == 0) && (this->Stab_Obj.call(m, k, Omega_1.data())))
            return 1;

        // At the last iteration, we may not be able to fit k columns into block Krylov matrix.
        // We then need to alter k.
        // Computation above is still done for a full k.
        if ((q_done + 1) * k > block_sz)
            k = block_sz - q_done * k;

        // Omega[:, k * (q_done + 1) : k * (q_done + 2)] = A' * Omega_1
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1.data(), m, 0.0, Omega.data() + offset * q_done, n);
        ++ p_done;

        if (this->cond_check)
            this->cond_nums.push_back(util::cond_num_check(n, k, Omega, this->Omega_cpy, this->s, this->verbosity));

        // Need to take in a pointer
        if ((p_done % q == 0) && (this->Stab_Obj.call(n, k, Omega.data() + offset * q_done)))
            return 1;

        ++q_done;
        RandBLAS::util::print_colmaj(n, block_sz, Omega.data(), name1);
    }

    //successful termination
    return 0;
}

} // end namespace RandLAPACK
#endif
