/*
TODO #1: Update implementation so that no copy of the original data is needed.

TODO #3: Need a test case with switching between different orthogonalization types

On early termination, data in B is moved, but not sized down
*/

#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>

#include <math.h>

using namespace RandLAPACK::comps::util;

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

    int64_t curr_sz = 0;
    int64_t next_sz = 0;

    T* A_dat = A.data();
    // pre-compute nrom
    T norm_A = lange(Norm::Fro, m, n, A_dat, m);
    // Immediate termination criteria
    if(norm_A == 0.0)
    {
        // Zero matrix termination
        k = curr_sz;
        this->termination = 1;
        return;
    }

    // tolerance check
    if (tol < 100 * std::numeric_limits<T>::epsilon())
    {
        tol = 100 * std::numeric_limits<T>::epsilon();
    }
    
    // If the space allocated for col in Q and row in B is insufficient for any iterations ...
    if(std::max( Q.size() / m, B.size() / n) < k)
    {
        // ... allocate more!
        this->curr_lim = std::min(this->dim_growth_factor * block_sz, k);
        // No need for data movement in this case
        upsize<T>(m * this->curr_lim, Q);
        upsize<T>(this->curr_lim * n, B);
    }
    else
    {
        this->curr_lim = k;
    }
    
    // Copy the initial data to avoid unwanted modification TODO #1
    std::vector<T> A_cpy (m * n, 0.0);
    T* A_cpy_dat = A_cpy.data();
    lacpy(MatrixType::General, m, n, A_dat, m, A_cpy_dat, m);

    T norm_B = 0.0;
    T prev_err = 0.0;
    T approx_err = 0.0;

    T* Q_gram_dat;
    T* Q_i_gram_dat;

    if(this->orth_check)
    {
        Q_gram_dat = upsize<T>(this->curr_lim * this->curr_lim, this->Q_gram);
        Q_i_gram_dat = upsize<T>(block_sz * block_sz, this->Q_i_gram);
    }

    T* QtQi_dat = upsize<T>(this->curr_lim * block_sz, this->QtQi);
    T* Q_i_dat = upsize<T>(m * block_sz, this->Q_i);
    T* B_i_dat = upsize<T>(block_sz * n, this->B_i);

    T* Q_dat = Q.data();
    T* B_dat = B.data();
    
    while(k > curr_sz)
    {
        // Dynamically changing block size
        block_sz = std::min(block_sz, k - curr_sz);
        next_sz = curr_sz + block_sz;

        // Make sure we have enough space for everything
        if(next_sz > this->curr_lim)
        {
            this->curr_lim = std::min(2 * this->curr_lim, k);
            Q_dat = upsize<T>(this->curr_lim * m, Q);
            B_dat = row_resize<T>(curr_sz, n, B, this->curr_lim);
            QtQi_dat = upsize<T>(this->curr_lim * block_sz, QtQi);
            if(this->orth_check)
                Q_gram_dat = upsize<T>(this->curr_lim * this->curr_lim, Q_gram);
        }

        // Calling RangeFinder
        this->RF_Obj.call(m, n, A_cpy, block_sz, this->Q_i);

        if(this->orth_check)
        {
            if (orthogonality_check<T>(m, block_sz, block_sz, Q_i, Q_i_gram, this->verbosity))
            {
                // Lost orthonormality of Q
                row_resize<T>(this->curr_lim, n, B, curr_sz);
                k = curr_sz;
                this->termination = 4;
                return;
            }
        }

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0)
        {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, curr_sz, block_sz, m, 1.0, Q_dat, m, Q_i_dat, m, 0.0, QtQi_dat, this->curr_lim);
            gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, block_sz, curr_sz, -1.0, Q_dat, m, QtQi_dat, this->curr_lim, 1.0, Q_i_dat, m);

            this->Orth_Obj.call(m, block_sz, this->Q_i);
        }

        //B_i = Q_i' * A
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, block_sz, n, m, 1.0, Q_i_dat, m, A_cpy_dat, m, 0.0, B_i_dat, block_sz);

        // Updating B norm estimation
        T norm_B_i = lange(Norm::Fro, block_sz, n, B_i_dat, block_sz);
        norm_B = hypot(norm_B, norm_B_i);
        // Updating approximation error
        prev_err = approx_err;
        approx_err = sqrt(abs(norm_A - norm_B)) * (sqrt(norm_A + norm_B) / norm_A);

        // Early termination - handling round-off error accumulation
        if ((curr_sz > 0) && (approx_err > prev_err))
        {
            // Early termination - error growth
            // Only need to move B's data, no resizing
            row_resize<T>(this->curr_lim, n, B, curr_sz);
            k = curr_sz;
            this->termination = 2;
            return;
        } 

        // Update the matrices Q and B
        lacpy(MatrixType::General, m, block_sz, &Q_i_dat[0], m, &Q_dat[m * curr_sz], m);	
        lacpy(MatrixType::General, block_sz, n, &B_i_dat[0], block_sz, &B_dat[curr_sz], this->curr_lim);
        
        if(this->orth_check)
        {
            if (orthogonality_check<T>(m, this->curr_lim, next_sz, Q, Q_gram, this->verbosity))
            {
                // Lost orthonormality of Q
                row_resize<T>(this->curr_lim, n, B, curr_sz);
                k = curr_sz;
                this->termination = 5;
                return;
            }
        }
        
        curr_sz += block_sz;
        // Termination criteria
        if (approx_err < tol)
        {
            // Reached the required error tol
            row_resize<T>(this->curr_lim, n, B, curr_sz);
            k = curr_sz;
            this->termination = 0;
            return;
        }
        
        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, block_sz, -1.0, Q_i_dat, m, B_i_dat, block_sz, 1.0, A_cpy_dat, m);
    }
    // Reached expected rank without achieving the tolerance
    this->termination = 3;
}

template void QB<float>::QB2(int64_t m, int64_t n, std::vector<float>& A, int64_t& k, int64_t block_sz, float tol, std::vector<float>& Q, std::vector<float>& B);
template void QB<double>::QB2(int64_t m, int64_t n, std::vector<double>& A, int64_t& k, int64_t block_sz, double tol, std::vector<double>& Q, std::vector<double>& B);
}// end namespace RandLAPACK::comps::qb
