#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>

#include <math.h>

namespace RandLAPACK::comps::qb {

template <typename T>
void QB<T>::QB2(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t& k,
    int64_t block_sz,
    T tol,
    std::vector<T>& Q,
    std::vector<T>& B
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
        QB::termination = 1;
        return;
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

    if(QB::verbosity && QB::orth_check)
    {
        printf("\nQ ORTHOGONALITY CHECK ENABLED\n\n");
    }
    std::vector<T> Q_gram(k * k, 0.0);
    T* Q_gram_dat = Q_gram.data();

    std::vector<T> QtQi(k * block_sz, 0.0); 
    std::vector<T> Q_i(m * block_sz, 0.0);
    std::vector<T> B_i(block_sz * n, 0.0);

    QB::Orth_Obj.tau.resize(block_sz);

    T* Q_dat = Q.data();
    T* B_dat = B.data();
    T* Q_i_dat = Q_i.data();
    T* B_i_dat = B_i.data();
    T* QtQi_dat = QtQi.data();

    while(k > curr_sz)
    {
        // Dynamically changing block size
        block_sz = std::min(block_sz, k - curr_sz);
        int next_sz = curr_sz + block_sz;

        QB::RF_Obj.call(m, n, A_cpy, block_sz, Q_i);

        if(QB::orth_check)
        {
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

            if(QB::verbosity)
            {
                printf("Q_i ERROR: %e\n", orth_i_err);
            }

            if (orth_i_err > 1.0e-10)
            {
                // Lost orthonormality of Q_i
                RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
                QB::termination = 4;
                return;
            }
        }

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0)
        {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, next_sz, block_sz, m, 1.0, Q_dat, m, Q_i_dat, m, 0.0, QtQi_dat, k);
            gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, block_sz, next_sz, -1.0, Q_dat, m, QtQi_dat, k, 1.0, Q_i_dat, m);

            // If CholQR succeeded in the rangefinder
            if (!QB::RF_Obj.Orth_Obj.chol_fail)
            {
                // No need to perform failure check here, as RF1 will notify in advance
                // Done via CholQR
                QB::Orth_Obj.call(m, block_sz, Q_i);
                // Performing the alg twice for better orthogonality	
                QB::Orth_Obj.call(m, block_sz, Q_i);
            }
            else
            {
                QB::Orth_Obj.decision_orth = 1;
                // Done via regular LAPACK's QR
                QB::Orth_Obj.call(m, block_sz, Q_i);
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
            QB::termination = 2;
            return;
        } 

        // Update the matrices Q and B
        lacpy(MatrixType::General, m, block_sz, Q_i_dat, m, Q_dat + (m * curr_sz), m);	
        lacpy(MatrixType::General, block_sz, n, B_i_dat, block_sz, B_dat + curr_sz, k);

        if(QB::orth_check)
        {
            gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                k, k, m,
                1.0, Q_dat, m, Q_dat, m,
                0.0, Q_gram_dat, k
            );
            for (int oi = 0; oi < next_sz; ++oi) {
                Q_gram_dat[oi * k + oi] -= 1.0;
            }
            T orth_err = lange(Norm::Fro, k, k, Q_gram_dat, k);
    
            if(QB::verbosity)
            {
               printf("Q ERROR:   %e\n\n", orth_err);
            }

            if (orth_err > 1.0e-10)
            {
                // Lost orthonormality of Q
                // Cut off the last iteration
                RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
                QB::termination = 5;
                return;
            }
        }

        curr_sz += block_sz;
        // Termination criteria
        if (approx_err < tol)
        {
            // Expected ternimation - tolerance achieved
            // Resizing the output arrays 
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            QB::termination = 0;
            return;
        }
        
        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, block_sz, -1.0, Q_i_dat, m, B_i_dat, block_sz, 1.0, A_cpy_dat, m);
    }
    // Reached expected rank without achieving the tolerance
    QB::termination = 3;
}

template <typename T>
void QB<T>::QB2_test_mode(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t& k,
    int64_t block_sz,
    T tol,
    std::vector<T>& Q,
    std::vector<T>& B
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
        QB::termination = 1;
        return;
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

    if(QB::verbosity && QB::orth_check)
    {
        printf("\nQ ORTHOGONALITY CHECK ENABLED\n\n");
    } 
    std::vector<T> Q_gram(k * k, 0.0);
    T* Q_gram_dat = Q_gram.data();

    std::vector<T> Q_i(m * block_sz, 0.0);
    std::vector<T> B_i(block_sz * n, 0.0);
    std::vector<T> QtQi(k * block_sz, 0.0); 

    QB::Orth_Obj.tau.resize(block_sz);

    T* Q_dat = Q.data();
    T* B_dat = B.data();
    T* Q_i_dat = Q_i.data();
    T* B_i_dat = B_i.data();
    T* QtQi_dat = QtQi.data();
    T* cond_nums_dat = QB::cond_nums.data();

    // No matter what parameters the user provided, always switch to test mode and using default Householder QR
    //QB::RF_Obj.decision_RF = 1;
    QB::Orth_Obj.decision_orth = 1;

    while(k > curr_sz)
    {
        // Dynamically changing block size
        block_sz = std::min(block_sz, k - curr_sz);
        int next_sz = curr_sz + block_sz;

        QB::RF_Obj.call(m, n, A_cpy, block_sz, Q_i);
        // Would be nice to update through pointers instead
        QB::cond_nums[curr_sz / block_sz] = QB::RF_Obj.cond_num;

        if(QB::orth_check)
        {
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
            if(QB::verbosity)
            {
                printf("Q_i ERROR: %e\n", orth_i_err);
            }
            if (orth_i_err > 1.0e-10)
            {
                // Lost orthonormality of Q_i
                RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
                QB::termination = 4;
                return;
            }
        }

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0)
        {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, next_sz, block_sz, m, 1.0, Q_dat, m, Q_i_dat, m, 0.0, QtQi_dat, k);
            gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, block_sz, next_sz, -1.0, Q_dat, m, QtQi_dat, k, 1.0, Q_i_dat, m);
            // Always call HQR in test mode
            QB::Orth_Obj.call(m, block_sz, Q_i);
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
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            QB::termination = 2;
            return;
        } 

        // Update the matrices Q and B
        lacpy(MatrixType::General, m, block_sz, Q_i_dat, m, Q_dat + (m * curr_sz), m);	
        lacpy(MatrixType::General, block_sz, n, B_i_dat, block_sz, B_dat + curr_sz, k);

        if(QB::orth_check)
        {
            gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                k, k, m,
                1.0, Q_dat, m, Q_dat, m,
                0.0, Q_gram_dat, k
            );
            for (int oi = 0; oi < next_sz; ++oi) {
                Q_gram_dat[oi * k + oi] -= 1.0;
            }
            T orth_err = lange(Norm::Fro, k, k, Q_gram_dat, k);
            
            if(QB::verbosity)
            {
                printf("Q ERROR:   %e\n\n", orth_err);
            }
            
            if (orth_err > 1.0e-10)
            {
                // Lost orthonormality of Q
                RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
                QB::termination = 5;
                return;
            }
        }

        curr_sz += block_sz;
        // Termination criteria
        if (approx_err < tol)
        {
            // Expected ternimation - tolerance achieved
            RandLAPACK::comps::util::qb_resize( m, n, Q, B, k, curr_sz);
            QB::termination = 0;
            return;
        }
        
        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, block_sz, -1.0, Q_i_dat, m, B_i_dat, block_sz, 1.0, A_cpy_dat, m);
    }
    // Reached expected rank without achieving the tolerance
    QB::termination = 3;
}

template void QB<float>::QB2(int64_t m, int64_t n, std::vector<float>& A, int64_t& k, int64_t block_sz, float tol, std::vector<float>& Q, std::vector<float>& B);
template void QB<double>::QB2(int64_t m, int64_t n, std::vector<double>& A, int64_t& k, int64_t block_sz, double tol, std::vector<double>& Q, std::vector<double>& B);

template void QB<float>::QB2_test_mode(int64_t m, int64_t n, std::vector<float>& A, int64_t& k, int64_t block_sz, float tol, std::vector<float>& Q, std::vector<float>& B);
template void QB<double>::QB2_test_mode(int64_t m, int64_t n, std::vector<double>& A, int64_t& k, int64_t block_sz, double tol, std::vector<double>& Q, std::vector<double>& B);
}// end namespace RandLAPACK::comps::qb
