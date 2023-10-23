#ifndef randlapack_rbki_h
#define randlapack_rbki_h

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>

using namespace std::chrono;

namespace RandLAPACK {

template <typename T, typename RNG>
class RBKIalg {
    public:

        virtual ~RBKIalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T> &A,
            int64_t d,
            std::vector<T> &R,
            std::vector<int64_t> &J,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class RBKI : public RBKIalg<T, RNG> {
    public:

        RBKI(
            bool verb,
            bool time_subroutines,
            T ep
        ) {
            verbosity = verb;
            timing = time_subroutines;
            eps = ep;
            no_hqrrp = 1;
            nb_alg = 64;
            oversampling = 10;
            use_cholqr = 0;
            panel_pivoting = 1;
            naive_rank_estimate = 1;
            use_fro_norm = 1;
            cond_check = 0;
        }

        int call(
            int64_t m,
            int64_t n,
            std::vector<T> &A,
            int64_t d,
            std::vector<T> &R,
            std::vector<int64_t> &J,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
        bool verbosity;
        bool timing;
        bool cond_check;
        T eps;
        int64_t rank;

        // 10 entries
        std::vector<long> times;

        // tuning SASOS
        int num_threads;
        int64_t nnz;

        // Buffers
        std::vector<T> A_hat;
        std::vector<T> tau;
        std::vector<T> R_sp;

        // HQRRP-related
        int no_hqrrp;
        int64_t nb_alg;
        int64_t oversampling;
        int64_t panel_pivoting;
        int64_t use_cholqr;

        // Rank estimate-related
        int naive_rank_estimate;
        int use_fro_norm;

        // Preconditioning-related
        T cond_num_A_pre;
        T cond_num_A_norm_pre;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int RBKI<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* &A,
    int64_t lda,
    int64_t k,
    T tol,
    T* &U,
    T* &V,
    T* &S,
    RandBLAS::RNGState<RNG> &state
){
    int64_t iter = 0;

    // Sketching operator space.
    T* Y = ( T * ) calloc( n * k, sizeof( T ) );
    // X_i space
    T* Y = ( T * ) calloc( m * k, sizeof( T ) );
    // tau space for QR
    T* tau = ( T * ) calloc( k, sizeof( T ) );

    // Pre-conpute Fro norm of an input matrix.
    T norm_A = lapack::lange(Norm::Fro, m, n, A, lda);
    T sq_tol = std::pow(tol, 2);

    // Generate a dense Gaussian random matrx.
    RandBLAS::DenseDist D(n, k);
    state = RandBLAS::fill_dense(D, Y, state).second;

    // [X_i, ~] = qr(A * Y_i, 0)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Y, n, 0.0, X, m);
    lapack::geqrf(m, k, X, k, tau);
    // Y_i = A' * X_i 
    // Below operation will instead return Y_i' because ORMQR() does not have an option to transpose
    // a non-inplicit matrix.
    lapack::ormqr(Side::Left, Op::Trans, k, n, m, X, m, tau, A, lda);

    // Iterate until in-loop termination criteria is met.
    while(1) {
        if (i % 2 == 0) {

        }
        else {

        }
    }

    return 0;
}
} // end namespace RandLAPACK
#endif