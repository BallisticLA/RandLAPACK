#include <RandLAPACK/comps/rf.hh>
#include <RandLAPACK/comps/rs.hh>
#include <RandLAPACK/comps/qb.hh>
#include <RandLAPACK.hh>
#include <RandBLAS.hh>
#include <iostream>

#include <lapack.hh>
#include <math.h>

namespace RandLAPACK::comps::qb {

// Generic scheme
template <typename T>
void qb1(
        int64_t m,
        int64_t n,
        T* const A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        T* Q, // m by k
        T* B, // k by n
	bool use_lu,
	uint64_t seed
){
    using namespace blas;

    RandLAPACK::comps::rf::rf1<T>(m, n, A, k, p, passes_per_stab, Q, use_lu, seed);
    //char name_5[] = "Q afret RF1";
    //RandBLAS::util::print_colmaj(m, k, Q, name_5);

    gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, 1.0, Q, m, A, m, 0.0, B, k);
}

// Blocked scheme
template <typename T>
void qb2(
        int64_t m,
        int64_t n,
        T* const A,
        int64_t k, // Here, serves as a backup termination criteria
        int64_t block_sz,
        T tol,
        int64_t p,
        int64_t passes_per_stab,
        T* Q, // m by k
        T* B, // k by n
	bool use_lu,
	uint64_t seed
){
    using namespace blas;
    using namespace lapack;

    // pre-compute squarred error
    T norm_A = lange(Norm::Fro, m, n, A, m);
    // Immediate termination criteria
    if(norm_A == 0.0)
    {
        return;
    }

    // Copy the initial data to avoid unwanted modification
    std::vector<T> A_cpy (m * n, 0.0);
    lacpy(MatrixType::General, m, n, A, m, A_cpy.data(), m);

    T norm_B = 0.0;
    T prev_err = 0.0;
    T approx_err = 0.0;
    int curr_sz = 0;

    while(k > curr_sz)
    {
        // Dynamically changing block size
        block_sz = std::min(block_sz, k - curr_sz);
        std::vector<T> Q_i(m * block_sz, 0.0);
        std::vector<T> B_i(block_sz * n, 0.0);
        std::vector<T> Q_i_small(k * block_sz, 0.0);
        std::vector<T> tau(block_sz, 2.0);

        RandLAPACK::comps::rf::rf1<T>(m, n, A_cpy.data(), block_sz, p, passes_per_stab, Q_i.data(), use_lu, ++seed);

        char nameQ_i1[] = "Q_i";
        RandBLAS::util::print_colmaj<T>(m, block_sz, Q_i.data(), nameQ_i1);

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0)
        {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, block_sz, m, 1.0, Q, m, Q_i.data(), m, 0.0, Q_i_small.data(), k);
            gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, block_sz, k, -1.0, Q, m, Q_i_small.data(), k, 1.0, Q_i.data(), m);

            // Done via regular LAPACK's QR
            geqrf(m, block_sz, Q_i.data(), m, tau.data());
            ungqr(m, block_sz, block_sz, Q_i.data(), m, tau.data());

            // Done via CholQR
            //RandLAPACK::comps::util::chol_QR<T>(m, block_sz, Q_i.data());
            // Performing the alg twice for better orthogonality	
            //RandLAPACK::comps::util::chol_QR<T>(m, block_sz, Q_i.data());
        }
        //B_i = Q_i' * A
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, block_sz, n, m, 1.0, Q_i.data(), m, A_cpy.data(), m, 0.0, B_i.data(), block_sz);
        
        char nameQ_i[] = "Q_ii";
        RandBLAS::util::print_colmaj<T>(m, block_sz, Q_i.data(), nameQ_i);

        //char nameB_i[] = "B_i";
        //RandBLAS::util::print_colmaj<T>(block_sz, n, B_i.data(), nameB_i);

        // Updating B norm estimation
        T norm_B_i = lange(Norm::Fro, block_sz, n, B_i.data(), block_sz);
        norm_B = hypot(norm_B, norm_B_i);
        // Updating approximation error
        prev_err = approx_err;
        approx_err = sqrt(abs(norm_A - norm_B) * (norm_A + norm_B)) / norm_A;

        // Early termination - handling round-off error accumulation
        if ((curr_sz > 0) && (approx_err > prev_err))
        {
            break;
        } 

        // Update the matrices Q and B
        lacpy(MatrixType::General, m, block_sz, Q_i.data(), m, Q + (m * curr_sz), m);	
        lacpy(MatrixType::General, block_sz, n, B_i.data(), block_sz, B + curr_sz, k);
        
        char nameQ[] = "Q";
        RandBLAS::util::print_colmaj<T>(m, k, Q, nameQ);

        //char nameB[] = "B";
        //RandBLAS::util::print_colmaj<T>(k, n, B, nameB);

        curr_sz += block_sz;
        // Termination criteria
        if (approx_err < tol)
        {
            break;
        }
        
        //char name_final1[] = "A_cpy_pre";
	    //RandBLAS::util::print_colmaj<T>(m, n, A_cpy.data(), name_final1);

        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        std::vector<T> BUFFER(m * n, 0.0);
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, -1.0, Q, m, B, k, 1.0, A_cpy.data(), m);

        //char name_final[] = "A_cpy";
	    //RandBLAS::util::print_colmaj<T>(m, n, A_cpy.data(), name_final);
    }
    printf("Normal termination\n");
    // still need to shink output sizes from k to curr_sz
}


template void qb1<float>( int64_t m, int64_t n, float* const A, int64_t k, int64_t p, int64_t passes_per_stab, float* Q, float* B, bool use_lu, uint64_t seed);
template void qb1<double>( int64_t m, int64_t n, double* const A, int64_t k, int64_t p, int64_t passes_per_stab, double* Q, double* B, bool use_lu, uint64_t seed);

template void qb2( int64_t m, int64_t n, float* const A, int64_t k, int64_t block_sz, float tol, int64_t p, int64_t passes_per_stab, float* Q, float* B, bool use_lu, uint64_t seed);
template void qb2( int64_t m, int64_t n, double* const A, int64_t k, int64_t block_sz, double tol, int64_t p, int64_t passes_per_stab, double* Q, double* B, bool use_lu, uint64_t seed);
}// end namespace RandLAPACK::comps::qb
