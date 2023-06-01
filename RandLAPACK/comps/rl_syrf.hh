#ifndef randlapack_comps_syrf_h
#define randlapack_comps_syrf_h

#include "rl_syps.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_orth.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <stdexcept>


namespace RandLAPACK {

template <typename T, typename RNG>
class SymmetricRangeFinder {
    public:
        virtual ~SymmetricRangeFinder() {}

        virtual RandBLAS::base::RNGState<RNG> call(
            blas::Uplo uplo,
            int64_t m,
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Q,
            RandBLAS::base::RNGState<RNG> state,
            T* work_buff
        ) = 0;
};

template <typename T, typename RNG>
class SYRF : public SymmetricRangeFinder<T, RNG> {
    public:

        SYRF(
            RandLAPACK::SymmetricPowerSketch<T, RNG>& syps_obj,
            RandLAPACK::Stabilization<T>& orth_obj,
            bool verb = false,
            bool cond = false
        ) : SYPS_Obj(syps_obj), Orth_Obj(orth_obj) {
            verbose = verb;
            cond_check = cond;
        }

        /// This is an analog of the RangeFinder class RF for symmetric matrices.
        ///
        /// @param[in] uplo
        ///     blas::Uplo::Upper or blas::Uplo::Lower.
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
        RandBLAS::base::RNGState<RNG> call(
            blas::Uplo uplo,
            int64_t m,
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Q,
            RandBLAS::base::RNGState<RNG> state,
            T* work_buff
        ) override;

    public:
       // Instantiated in the constructor
       RandLAPACK::SymmetricPowerSketch<T, RNG>& SYPS_Obj;
       RandLAPACK::Stabilization<T>& Orth_Obj;
       bool verbose;
       bool cond_check;
       std::vector<T> cond_work_mat;
       std::vector<T> cond_work_vec;
       std::vector<T> cond_nums; // Condition nubers of sketches
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
RandBLAS::base::RNGState<RNG> SYRF<T, RNG>::call(
    blas::Uplo uplo,
    int64_t m,
    const std::vector<T>& A,
    int64_t k,
    std::vector<T>& Q,
    RandBLAS::base::RNGState<RNG> state,
    T* work_buff
){

    std::vector<T> Omega;
    T* Omega_dat = util::upsize(m * k, Omega);
    T* Q_dat = Q.data();

    // Basic version-works
    //RandBLAS::dense::DenseDist D{.n_rows = m, .n_cols = k};
    //RandBLAS::dense::fill_buff(Omega_dat, D, state);

    auto next_state = SYPS_Obj.call(uplo, m, A, m, k, state, Omega_dat, Q.data());

    // Q = orth(A * Omega)
    blas::symm(Layout::ColMajor, Side::Left, uplo, m, k, 1.0, A.data(), m, Omega_dat, m, 0.0, Q_dat, m);

    this->Orth_Obj.call(m, k, Q);

    // Normal termination
    return state;
}

} // end namespace RandLAPACK
#endif
