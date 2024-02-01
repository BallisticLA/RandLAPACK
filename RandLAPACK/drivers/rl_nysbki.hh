#ifndef randlapack_NysBKI_h
#define randlapack_NysBKI_h

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>
#include <climits>

using namespace std::chrono;

namespace RandLAPACK {

template <typename T, typename RNG>
class NysBKIalg {
    public:
        virtual ~NysBKIalg() {}
        virtual int call(
            int64_t m,
            T* A,
            int64_t lda,
            int64_t k,
            T* V,
            T* Lambda,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class NysBKI : public NysBKIalg<T, RNG> {
    public:
        NysBKI(
            bool verb,
            bool time_subroutines,
            T ep
        ) {
            verbosity = verb;
            timing = time_subroutines;
            tol = ep;
            max_krylov_iters = INT_MAX;
        }
        int call(
            int64_t m,
            T* A,
            int64_t lda,
            int64_t k,
            T* V,
            T* Lambda,
            RandBLAS::RNGState<RNG> &state
        ) override;
    public:
        bool verbosity;
        bool timing;
        T tol;
        int num_krylov_iters;
        int max_krylov_iters;
        std::vector<long> times;
        T norm_R_end;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int NysBKI<T, RNG>::call(
    int64_t m,
    T* A,
    int64_t lda,
    int64_t k,
    T* V,
    T* Lambda,
    RandBLAS::RNGState<RNG> &state
){
    int iter = 0;

    T* X   = ( T * ) calloc( m * (m + k), sizeof( T ) );
    T* X_i = X;
    T* Y   = ( T * ) calloc( m * (m + k), sizeof( T ) );
    T* Y_i = Y;

    // tau space for QR
    T* tau = ( T * ) calloc( k,           sizeof( T ) );


    // Generate a dense Gaussian random matrx.
    RandBLAS::DenseDist D(m, k);
    state = RandBLAS::fill_dense(D, X_i, state).second;
    // [X_i, ~] = qr(randn(m, m), 0)
    lapack::geqrf(m, k, X_i, m, tau);
    // Y_i = A * X_i
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, m, 1.0, A, m, X_i, m, 0.0, Y_i, m);

    while(iter < max_krylov_iters) {
        // Advance X_i pointer
        X_i = X_i + (m * k);
        lapack::lacpy(MatrixType::Upper, m, k, X, m, X_i, m);

        if (!iter) {
            // X_i+1 = Y_i + tol * X_i;
            blas::scal(m * k, this->tol, X_i, 1);	
            blas::axpy(m * k, 1.0, Y_i, 1, X_i, 1);
        } else {

        }



    }

    return 0;
}
} // end namespace RandLAPACK
#endif