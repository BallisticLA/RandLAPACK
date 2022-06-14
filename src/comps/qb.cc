#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>

#include <math.h>

#define USE_QR
#define ORTHONORM_CHECKS
#define COND_CHECK
//#define VERBOSE

namespace RandLAPACK::comps::qb {

// Generic scheme
template <typename T>
void qb1(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // m by k
        std::vector<T>& B, // k by n
	    uint32_t seed
){
    using namespace blas;

    // Adjust the expected rank
    if(k == 0)
    {
        k = std::min(m, n);
        Q.resize(m * k);
        B.resize(n * k);
    }

    T buf = 0;
    RandLAPACK::comps::rf::rf1<T>(m, n, A, k, p, passes_per_stab, Q, seed, buf);
    gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, 1.0, Q.data(), m, A.data(), m, 0.0, B.data(), k);
}

// Blocked scheme
template <typename T>
int qb2(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t& k, // Here, serves as a backup termination criteria
        int64_t block_sz,
        T tol,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // m by k
        std::vector<T>& B, // k by n
	    uint32_t seed,
        std::vector<T>& cond_nums
){
    using namespace blas;
    using namespace lapack;

    int curr_sz = 0;

    T* A_dat = A.data();
    // pre-compute norm
    T norm_A = lange(Norm::Fro, m, n, A_dat, m);

    // Immediate termination criteria
    if(norm_A == 0.0)
    {
        // Zero matrix termination
        RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
        return 1;
    }

    // Copy the initial data to avoid unwanted modification
    std::vector<T> A_cpy (m * n, 0.0);
    T* A_cpy_dat = A_cpy.data();
    lacpy(MatrixType::General, m, n, A_dat, m, A_cpy_dat, m);

    T norm_B = 0.0;
    T prev_err = 0.0;
    T approx_err = 0.0;

    // Adjust the expected rank
    if(k == 0)
    {
        k = std::min(m, n);
        Q.resize(m * k);
        B.resize(n * k);
    }

#ifdef ORTHONORM_CHECKS
#ifdef VERBOSE
    printf("\nQ ORTHOGONALITY CHECK ENABLED\n\n");
#endif    
    std::vector<T> Q_gram(k * k, 0.0);
    T* Q_gram_dat = Q_gram.data();
#endif

    std::vector<T> Q_i(m * block_sz, 0.0);
    std::vector<T> B_i(block_sz * n, 0.0);
    std::vector<T> QtQi(k * block_sz, 0.0); 
    std::vector<T> tau(block_sz, 2.0);

    T* Q_dat = Q.data();
    T* B_dat = B.data();
    T* Q_i_dat = Q_i.data();
    T* B_i_dat = B_i.data();
    T* QtQi_dat = QtQi.data();
    T* tau_dat = tau.data();
    T* cond_nums_dat = cond_nums.data();

    while(k > curr_sz)
    {
        // Dynamically changing block size
        block_sz = std::min(block_sz, k - curr_sz);
        int next_sz = curr_sz + block_sz;

#ifdef COND_CHECK
        RandLAPACK::comps::rf::rf1<T>(m, n, A_cpy, block_sz, p, passes_per_stab, Q_i, ++seed, *cond_nums_dat);
        ++cond_nums_dat;
#else
        RandLAPACK::comps::rf::rf1<T>(m, n, A_cpy, block_sz, p, passes_per_stab, Q_i, ++seed, NULL);
#endif


#ifdef ORTHONORM_CHECKS
        //needs reallocated
        std::vector<T> Q_i_gram(block_sz * block_sz, 0.0);
        T* Q_i_gram_dat = Q_i_gram.data();

        gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
            block_sz, block_sz, m,
            1.0, Q_i_dat, m, Q_i_dat, m,
            0.0, Q_i_gram_dat, block_sz
        );
        for (int oi = 0; oi < block_sz; ++oi) {
            Q_i_gram_dat[oi * block_sz + oi] -= 1.0;
        }
        T orth_i_err = lange(Norm::Fro, block_sz, block_sz, Q_i_gram_dat, block_sz);
#ifdef VERBOSE
        printf("Q_i ERROR: %e\n", orth_i_err);
#endif
        if (orth_i_err > 1.0e-10)
        {
            // Lost orthonormality of Q_i
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            return 4;
        }
#endif

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0)
        {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, next_sz, block_sz, m, 1.0, Q_dat, m, Q_i_dat, m, 0.0, QtQi_dat, k);
            gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, block_sz, next_sz, -1.0, Q_dat, m, QtQi_dat, k, 1.0, Q_i_dat, m);

#ifdef USE_QR
            // Done via regular LAPACK's QR
            geqrf(m, block_sz, Q_i.data(), m, tau.data());
            ungqr(m, block_sz, block_sz, Q_i.data(), m, tau.data());
#else
            // Done via CholQR
            RandLAPACK::comps::orth::chol_QR<T>(m, block_sz, Q_i);
            // Performing the alg twice for better orthogonality	
            RandLAPACK::comps::orth::chol_QR<T>(m, block_sz, Q_i);
#endif
        }
        //B_i = Q_i' * A
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, block_sz, n, m, 1.0, Q_i_dat, m, A_cpy_dat, m, 0.0, B_i_dat, block_sz);

        //char name_3[] = "Qi";
        //RandBLAS::util::print_colmaj(m, block_sz, Q_i_dat, name_3);

        // Updating B norm estimation
        T norm_B_i = lange(Norm::Fro, block_sz, n, B_i_dat, block_sz);
        norm_B = hypot(norm_B, norm_B_i);

        // Updating approximation error
        prev_err = approx_err;
        approx_err = sqrt(abs(norm_A - norm_B) * (norm_A + norm_B)) / norm_A;

        // Early termination - handling round-off error accumulation
        if ((curr_sz > 0) && (approx_err > prev_err))
        {
            // Early termination - error growth
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            return 2;
        } 

        // Update the matrices Q and B
        lacpy(MatrixType::General, m, block_sz, Q_i_dat, m, Q_dat + (m * curr_sz), m);	
        lacpy(MatrixType::General, block_sz, n, B_i_dat, block_sz, B_dat + curr_sz, k);

#ifdef ORTHONORM_CHECKS
        gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
            k, k, m,
            1.0, Q_dat, m, Q_dat, m,
            0.0, Q_gram_dat, k
        );
        for (int oi = 0; oi < next_sz; ++oi) {
            Q_gram_dat[oi * k + oi] -= 1.0;
        }
        T orth_err = lange(Norm::Fro, k, k, Q_gram_dat, k);
#ifdef VERBOSE
        printf("Q ERROR:   %e\n\n", orth_err);
#endif
        if (orth_err > 1.0e-10)
        {
            // Lost orthonormality of Q
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            return 5;
        }
#endif

        curr_sz += block_sz;
        // Termination criteria
        if (approx_err < tol)
        {
            // Expected ternimation - tolerance achieved
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            return 0;
        }
        
        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, block_sz, -1.0, Q_i_dat, m, B_i_dat, block_sz, 1.0, A_cpy_dat, m);
    }
    // Reached expected rank without achieving the tolerance
    return 3;
}

// Blocked scheme utilizing CholQR by default. If CholQR fails, falls back on Householder QR for the rest of the run.
template <typename T>
int qb2_safe(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t& k, // Here, serves as a backup termination criteria
        int64_t block_sz,
        T tol,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // m by k
        std::vector<T>& B, // k by n
	    uint32_t seed
){
    using namespace blas;
    using namespace lapack;

    int curr_sz = 0;

    T* A_dat = A.data();
    // pre-compute nrom
    T norm_A = lange(Norm::Fro, m, n, A_dat, m);
    // Immediate termination criteria
    if(norm_A == 0.0)
    {
        // Zero matrix termination
        RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
        return 1;
    }

    // Copy the initial data to avoid unwanted modification
    std::vector<T> A_cpy (m * n, 0.0);
    T* A_cpy_dat = A_cpy.data();
    lacpy(MatrixType::General, m, n, A_dat, m, A_cpy_dat, m);

    T norm_B = 0.0;
    T prev_err = 0.0;
    T approx_err = 0.0;

    // Adjust the expected rank
    if(k == 0)
    {
        k = std::min(m, n);
        Q.resize(m * k);
        B.resize(n * k);
    }

#ifdef ORTHONORM_CHECKS
#ifdef VERBOSE
    printf("\nQ ORTHOGONALITY CHECK ENABLED\n\n");
#endif
    std::vector<T> Q_gram(k * k, 0.0);
    T* Q_gram_dat = Q_gram.data();
#endif

    std::vector<T> QtQi(k * block_sz, 0.0); 
    std::vector<T> Q_i(m * block_sz, 0.0);
    std::vector<T> B_i(block_sz * n, 0.0);
    std::vector<T> tau(block_sz, 2.0);

    T* Q_dat = Q.data();
    T* B_dat = B.data();
    T* Q_i_dat = Q_i.data();
    T* B_i_dat = B_i.data();
    T* QtQi_dat = QtQi.data();
    T* tau_dat = tau.data();

    // Set flag for Householder QR usage
    // TODO:: propagate the flag down to RF1 (OO interface stuff?)
    bool use_qr = 0;

    while(k > curr_sz)
    {
        // Dynamically changing block size
        block_sz = std::min(block_sz, k - curr_sz);
        int next_sz = curr_sz + block_sz;

        use_qr = RandLAPACK::comps::rf::rf1_safe<T>(m, n, A_cpy, block_sz, p, passes_per_stab, Q_i, use_qr, ++seed);

#ifdef ORTHONORM_CHECKS
        //needs reallocated
        std::vector<T> Q_i_gram(block_sz * block_sz, 0.0);
        T* Q_i_gram_dat = Q_i_gram.data();

        gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
            block_sz, block_sz, m,
            1.0, Q_i_dat, m, Q_i_dat, m,
            0.0, Q_i_gram_dat, block_sz
        );
        for (int oi = 0; oi < block_sz; ++oi) {
            Q_i_gram_dat[oi * block_sz + oi] -= 1.0;
        }
        T orth_i_err = lange(Norm::Fro, block_sz, block_sz, Q_i_gram_dat, block_sz);
#ifdef VERBOSE
        printf("Q_i ERROR: %e\n", orth_i_err);
#endif
        if (orth_i_err > 1.0e-10)
        {
            // Lost orthonormality of Q_i
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            return 4;
        }
#endif

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0)
        {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, next_sz, block_sz, m, 1.0, Q_dat, m, Q_i_dat, m, 0.0, QtQi_dat, k);
            gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, block_sz, next_sz, -1.0, Q_dat, m, QtQi_dat, k, 1.0, Q_i_dat, m);


            if (!use_qr)
            {
                // No need to perform failure check here, as RF1 will notify in advance
                // Done via CholQR
                RandLAPACK::comps::orth::chol_QR<T>(m, block_sz, Q_i);
                // Performing the alg twice for better orthogonality	
                RandLAPACK::comps::orth::chol_QR<T>(m, block_sz, Q_i);
            }
            else
            {
                // Done via regular LAPACK's QR
                geqrf(m, block_sz, Q_i_dat, m, tau_dat);
                ungqr(m, block_sz, block_sz, Q_i_dat, m, tau_dat);
            }
        }
        //B_i = Q_i' * A
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, block_sz, n, m, 1.0, Q_i_dat, m, A_cpy_dat, m, 0.0, B_i_dat, block_sz);

        // Updating B norm estimation
        T norm_B_i = lange(Norm::Fro, block_sz, n, B_i_dat, block_sz);
        norm_B = hypot(norm_B, norm_B_i);
        // Updating approximation error
        prev_err = approx_err;
        approx_err = sqrt(abs(norm_A - norm_B) * (norm_A + norm_B)) / norm_A;

        // Early termination - handling round-off error accumulation
        if ((curr_sz > 0) && (approx_err > prev_err))
        {
            // Early termination - error growth
            // Resizing the output arrays
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            return 2;
        } 

        // Update the matrices Q and B
        lacpy(MatrixType::General, m, block_sz, Q_i_dat, m, Q_dat + (m * curr_sz), m);	
        lacpy(MatrixType::General, block_sz, n, B_i_dat, block_sz, B_dat + curr_sz, k);

#ifdef ORTHONORM_CHECKS
        gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
            k, k, m,
            1.0, Q_dat, m, Q_dat, m,
            0.0, Q_gram_dat, k
        );
        for (int oi = 0; oi < next_sz; ++oi) {
            Q_gram_dat[oi * k + oi] -= 1.0;
        }
        T orth_err = lange(Norm::Fro, k, k, Q_gram_dat, k);
#ifdef VERBOSE
        printf("Q ERROR:   %e\n\n", orth_err);
#endif
        if (orth_err > 1.0e-10)
        {
            // Lost orthonormality of Q
            // Cut off the last iteration
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            return 5;
        }
#endif

        curr_sz += block_sz;
        // Termination criteria
        if (approx_err < tol)
        {
            // Expected ternimation - tolerance achieved
            // Resizing the output arrays 
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            return 0;
        }
        
        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, block_sz, -1.0, Q_i_dat, m, B_i_dat, block_sz, 1.0, A_cpy_dat, m);
    }
    // Reached expected rank without achieving the tolerance
    return 3;
}

template void qb1<float>(int64_t m, int64_t n, std::vector<float>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<float>& Q, std::vector<float>& B, uint32_t seed);
template void qb1<double>(int64_t m, int64_t n, std::vector<double>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<double>& Q, std::vector<double>& B, uint32_t seed);

template int qb2(int64_t m, int64_t n, std::vector<float>& A, int64_t& k, int64_t block_sz, float tol, int64_t p, int64_t passes_per_stab, std::vector<float>& Q, std::vector<float>& B, uint32_t seed, std::vector<float>& cond_nums);
template int qb2(int64_t m, int64_t n, std::vector<double>& A, int64_t& k, int64_t block_sz, double tol, int64_t p, int64_t passes_per_stab, std::vector<double>& Q, std::vector<double>& B, uint32_t seed, std::vector<double>& cond_nums);

template int qb2_safe(int64_t m, int64_t n, std::vector<float>& A, int64_t& k, int64_t block_sz, float tol, int64_t p, int64_t passes_per_stab, std::vector<float>& Q, std::vector<float>& B, uint32_t seed);
template int qb2_safe(int64_t m, int64_t n, std::vector<double>& A, int64_t& k, int64_t block_sz, double tol, int64_t p, int64_t passes_per_stab, std::vector<double>& Q, std::vector<double>& B, uint32_t seed);
}// end namespace RandLAPACK::comps::qb
