#ifndef randlapack_comps_util_h
#define randlapack_comps_util_h

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace RandLAPACK::util {

/// Generates an identity matrix. Assuming col-maj
template <typename T>
void eye(
    int64_t m,
    int64_t n,
    T* A
) {
    int64_t min = std::min(m, n);
    for (int i = 0; i < m*n; ++i)
        A[i] = 0.0;
    for(int j = 0; j < min; ++j)
        A[(m * j) + j] = 1.0;
}

template <typename T>
void eye(
    int64_t m,
    int64_t n,
    std::vector<T> &A
) {
    eye(m, n, A.data());
}

/// Diagonalization - turns a vector into a diagonal matrix. Overwrites the
/// diagonal entries of matrix S with those stored in s.
template <typename T>
void diag(
    int64_t m,
    int64_t n,
    T* s,
    int64_t k, // size of s, < min(m, n)
    T* S // Assuming S is m by n
) {

    if(k > std::min(m, n)) 
        throw std::runtime_error("Invalid rank parameter.");
    // size of s
    blas::copy(k, s, 1, S, m + 1);
}

/// Zeros-out the upper-triangular portion of A
template <typename T>
void get_L(
    int64_t m,
    int64_t n,
    T* A,
    int overwrite_diagonal
) {
    for(int i = 0; i < n; ++i) {
        std::fill(&A[m * i], &A[i + m * i], 0.0);
        
        if(overwrite_diagonal)
            A[i + m * i] = 1.0;
    }
}

/// Zeros-out the lower-triangular portion of A
template <typename T>
void get_U(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda
) {
    for(int i = 0; i < n - 1; ++i) {
        std::fill(&A[i * (lda + 1) + 1], &A[(i * lda) + m], 0.0);
    }
}

/// Positions columns of A in accordance with idx vector of length k.
/// idx array modified ONLY within the scope of this function.
template <typename T>
void col_swap(
    int64_t m,
    int64_t n,
    int64_t k,
    T* A,
    int64_t lda,
    std::vector<int64_t> idx
) {
    if(k > n) 
        throw std::runtime_error("Invalid rank parameter.");

    int64_t i, j; //, l;
    for (i = 0, j = 0; i < k; ++i) {
        j = idx[i] - 1;
        blas::swap(m, &A[i * lda], 1, &A[j * lda], 1);

        // swap idx array elements
        // Find idx element with value i and assign it to j
        auto it = std::find(idx.begin() + i, idx.begin() + k, i + 1);
        idx[it - (idx.begin())] = j + 1;
    }
}

/// A version of the above function to be used on a vector of integers
template <typename T>
void col_swap(
    int64_t n,
    int64_t k,
    int64_t* A,
    std::vector<int64_t> idx
) {
    if(k > n) 
        throw std::runtime_error("Incorrect rank parameter.");

    int64_t* idx_dat = idx.data();

    int64_t i, j;
    for (i = 0, j = 0; i < k; ++i) {
        j = idx_dat[i] - 1;
        std::swap(A[i], A[j]);

        // swap idx array elements
        // Find idx element with value i and assign it to j
        auto it = std::find(idx.begin() + i, idx.begin() + k, i + 1);
        idx[it - (idx.begin())] = j + 1;
    }
}

/// Checks if the given size is larger than available. 
/// If so, resizes the vector.
template <typename T>
T* upsize(
    int64_t target_sz,
    std::vector<T> &A
) {
    if ((int64_t) A.size() < target_sz)
        A.resize(target_sz, 0);

    return A.data();
}


/// Changes the number of rows of a column-major matrix.
template <typename T>
T* row_resize(
    int64_t m,
    int64_t n,
    std::vector<T> &A,
    int64_t k
) {

    T* A_dat = A.data();

    // SIZING DOWN - just moving data
    if(m > k) {
        uint64_t end = k;
        for (int i = 1; i < n; ++i) {
            // Place ith column (of k entries) after the (i - 1)st column
            blas::copy(k, &A_dat[m * i], 1, &A_dat[end], 1);
            end += k;
        }
    } else { //SIZING UP
        // How many rows are being added: k - m
        A_dat = upsize(k * n, A);

        int64_t end = k * (n - 1);
        for(int i = n - 1; i > 0; --i) {
            // Copy in reverse order to avoid overwriting
            blas::copy(m, &A_dat[m * i], -1, &A_dat[end], -1);
            std::fill(&A_dat[m * i], &A_dat[end], 0.0);
            end -= k;
        }
    }

    return A_dat;
}

/// Find the condition number of a given matrix A.
template <typename T>
T cond_num_check(
    int64_t m,
    int64_t n,
    const T* A,
    T* A_cpy,
    T* s,
    bool verbose
) {

    // TODO: GET RID OF THE INTERNAL ALLOCATIONS
    A_cpy = ( T * ) calloc( m * n, sizeof( T ) );
    s     = ( T * ) calloc( n, sizeof( T ) );

    lapack::lacpy(MatrixType::General, m, n, A, m, A_cpy, m);
    lapack::gesdd(Job::NoVec, m, n, A_cpy, m, s, NULL, m, NULL, n);

    T cond_num = s[0] / s[n - 1];

    if (verbose)
        printf("CONDITION NUMBER: %f\n", cond_num);

    free(A_cpy);
    free(s);

    return cond_num;
}

// Computes the numerical rank of a given matirx
template <typename T>
int64_t rank_check(
    int64_t m,
    int64_t n,
    const T* A
) {
    T* A_pre_cpy = ( T * ) calloc( m * n, sizeof( T ) );
    T* s     = ( T * ) calloc( n, sizeof( T ) );

    RandLAPACK::util::cond_num_check(m, n, A, A_pre_cpy, s, false);

    for(int i = 0; i < n; ++i) {
        if (s[i] / s[0] <= 5 * std::numeric_limits<T>::epsilon())
            return i - 1;
    }

    free(A_pre_cpy);
    free(s);

    return n;
}

/// Checks whether matrix A has orthonormal columns.
template <typename T>
bool orthogonality_check(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<T> &A,
    std::vector<T> &A_gram,
    bool verbose
) {

    const T* A_dat = A.data();
    T* A_gram_dat = A_gram.data();

    blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, n, m, 1.0, A_dat, m, 0.0, A_gram_dat, n);

    for (int oi = 0; oi < k; ++oi) {
        A_gram_dat[oi * n + oi] -= 1.0;
    }
    T orth_err = lapack::lange(Norm::Fro, n, n, A_gram_dat, k);

    if(verbose) {
        printf("Q ERROR:   %e\n\n", orth_err);
    }

    if (orth_err > 1.0e-10)
        return true;

    return false;
}

/// Computes an L-2 norm of a given matrix using
/// p steps of power iteration.
template <typename T, typename RNG>
T estimate_spectral_norm(
    int64_t m,
    int64_t n,
    T const* A_dat,
    int p,
    RandBLAS::RNGState<RNG>& state
) {

    std::vector<T> buf (n, 0.0);
    std::vector<T> buf1 (m, 0.0);

    RandBLAS::DenseDist DV(n, 1);
    state = RandBLAS::fill_dense(DV, buf.data(), state).second;

    T prev_norm_inv = 1.0;
    for(int i = 0; i < p; ++i) {
        // A * v
        gemv(Layout::ColMajor, Op::NoTrans, m, n, 1.0, A_dat, m, buf.data(), 1, 0.0, buf1.data(), 1);
        // prev_norm_inv * A' * A * v
        gemv(Layout::ColMajor, Op::Trans, m, n, prev_norm_inv, A_dat, m, buf1.data(), 1, 0.0, buf.data(), 1);
        prev_norm_inv = 1 / blas::nrm2(n, buf.data(), 1);
    }

    return std::sqrt(blas::nrm2(n, buf.data(), 1));
}

/// Uses recursion to find the rank of the matrix pointed to by A_dat.
/// Does so by attempting to find the smallest k such that 
/// ||A[k:, k:]||_F <= tau_trunk * ||A||.
/// ||A|| can be either 2 or Fro.
/// Finding such k is done via binary search in range [1, n], which is 
/// controlled by ||A[k:, k:]||_F (<)(>) tau_trunk * ||A||. 
/// We first attempt to find k that results in an expression closest to 
/// ||A[k:, k:]||_F == tau_trunk * ||A|| and then ensure that ||A[k:, k:]||_F
/// is not smaller than tau_trunk * ||A|| to avoid rank underestimation.
template <typename T>
int64_t rank_search_binary(
    int64_t lo,
    int64_t hi,
    int64_t k,
    int64_t n,
    T norm_A,
    T tau_trunc,
    T const* A_dat
) {
    T norm_R_sub = lapack::lange(Norm::Fro, n - k, n, &A_dat[k * n], n - k);

    if(((k - lo) / 2) == 0) {
        // Need to make sure we are not underestimating rank
        while(norm_R_sub > tau_trunc * norm_A)
        {
            ++k;
            norm_R_sub = lapack::lange(Norm::Fro, n - k, n, &A_dat[k * n], n - k);
        }
        return k;
    } else if (norm_R_sub > tau_trunc * norm_A) {
        // k is larger
        k = rank_search_binary(k, hi, k + ((k - lo) / 2), n, norm_A, tau_trunc, A_dat);
    } else { //(norm_R_sub < tau_trunc * norm_A) {
        // k is smaller
        k = rank_search_binary(lo, k, lo + ((k - lo) / 2), n, norm_A, tau_trunc, A_dat);
    }
    return k;
}

/// Normalizes columns of a given matrix, writes the result into a buffer
template <typename T>
void normc(
    int64_t m,
    int64_t n,
    const std::vector<T> &A,
    std::vector<T> &A_norm
) {
    util::upsize(m * n, A_norm);

    T col_nrm = 0.0;
    for(int i = 0; i < n; ++i) {
        col_nrm = blas::nrm2(m, &A[m * i], 1);
        if(col_nrm != 0) {
            for (int j = 0; j < m; ++j) {
                A_norm[m * i + j] = A[m * i + j] / col_nrm;
            }
        }
    }
}


/**
 * In-place transpose of square matrix of order n, with leading dimension n.
 * Turns out that "layout" doesn't matter here.
*/
template <typename T>
void transpose_square(T* H, int64_t n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            T val_ij = H[i + j*n];
            T val_ji = H[j + i*n];
            H[i + j*n] = val_ji;
            H[j + i*n] = val_ij;
        }
    }
    return;
}

/**
 * 
*/
template <typename T>
void eat_lda_slack(
    T* buff,
    int64_t vec_len,
    int64_t num_vecs,
    int64_t inter_vec_stride
) {
    if (vec_len == inter_vec_stride)
        return;
    T* work = new T[vec_len]{};
    for (int i = 0; i < num_vecs; ++i) {
        blas::copy(vec_len, &buff[i*inter_vec_stride], 1, work, 1);
        blas::copy(vec_len, work, 1, &buff[i*vec_len], 1);
    }
    delete [] work;
}

// Perform an explicit transposition of a given matrix, 
// write the transpose into a buffer.
// WARNING: OMP parallelism occasionally tanks the performance.
template <typename T>
void transposition(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    T* AT,
    int64_t ldat,
    int copy_upper_triangle
) {
    if (copy_upper_triangle) {
        // Only transposing the upper-triangular portion of the original
        #pragma omp parallel for
        for(int i = 0; i < n; ++i)
            blas::copy(i + 1, &A[i * lda], 1, &AT[i], ldat);
    } else {
        #pragma omp parallel for
        for(int i = 0; i < n; ++i)
            blas::copy(m, &A[i * lda], 1, &AT[i], ldat);
    }
}

} // end namespace util
#endif
