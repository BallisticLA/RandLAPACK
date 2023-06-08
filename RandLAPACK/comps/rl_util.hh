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
    std::vector<T>& A
) {
    T* A_dat = A.data();
    int64_t min = std::min(m, n);
    for(int j = 0; j < min; ++j) {
        A_dat[(m * j) + j] = 1.0;
    }
}

/// Diagonalization - turns a vector into a diagonal matrix. Overwrites the
/// diagonal entries of matrix S with those stored in s.
template <typename T>
void diag(
    int64_t m,
    int64_t n,
    const std::vector<T>& s,
    int64_t k, // size of s, < min(m, n)
    std::vector<T>& S // Assuming S is m by n
) {

    if(k > n) {
        // Throw an error
    }
    // size of s
    blas::copy(k, s.data(), 1, S.data(), m + 1);
}

/// Captures k diagonal elements of A and stores them in buf.
template <typename T>
void extract_diag(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<T>& A,
    std::vector<T>& buf
) {
    const T* A_dat = A.data();
    if (k == 0) {
        k = std::min(m, n);
    }
    for(int i = 0; i < k; ++i) {
        buf[i] = A_dat[(i * m) + i];
    }
}

/// Displays the first k diagonal elements.
template <typename T>
void disp_diag(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<T>& A
) {
    const T* A_dat = A.data();
    if (k == 0) {
        k = std::min(m, n);
    }
    printf("DISPLAYING THE MAIN DIAGONAL OF A GIVEN MATRIX: \n");
    for(int i = 0; i < k; ++i) {
        printf("ELEMENT %d: %f\n", i, *(A_dat + (i * m) + i));
    }
}

/// Extracts the l-portion of the GETRF result, places 1's on the main diagonal.
/// Overwrites the passed-in matrix.
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

template <typename T>
void get_L(
    int64_t m,
    int64_t n,
    std::vector<T> &L,
    int overwrite_diagonal
) {
    get_L(m, n, L.data(), overwrite_diagonal);
}

/// Stores the upper-triangualr portion of A in U.
template <typename T>
void get_U(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    std::vector<T>& U // We are assuming U is n by n
) {
    // Vector end pointer
    int size = m * n;

    const T* A_dat = A.data();
    T* U_dat = U.data();

    for(int i = 0, j = 1, k = 0; i < size && j <= m; i += m, k +=n, ++j) {
        blas::copy(j, &A_dat[i], 1, &U_dat[k], 1);
    }
}

/// Zeros-out the lower-triangular portion of A
template <typename T>
void get_U(
    int64_t m,
    int64_t n,
    std::vector<T>& A
) {
    T* A_dat = A.data();

    for(int i = 0; i < n - 1; ++i) {
        std::fill(&A_dat[i * (m + 1) + 1], &A_dat[(i + 1) * m], 0.0);
    }
}

/// Positions columns of A in accordance with idx vector of length k.
/// idx array modified ONLY within the scope of this function.
template <typename T>
void col_swap(
    int64_t m,
    int64_t n,
    int64_t k,
    std::vector<T>& A,
    std::vector<int64_t> idx
) {

    if(k > n) {
        // Throw error
    }

    int64_t* idx_dat = idx.data();
    T* A_dat = A.data();

    int64_t i, j, l;
    for (i = 0, j = 0; i < k; ++i) {
        j = idx_dat[i] - 1;
        blas::swap(m, &A_dat[i * m], 1, &A_dat[j * m], 1);

        // swap idx array elements
        // Find idx element with value i and assign it to j
        for(l = i; l < k; ++l) {
            if(idx[l] == i + 1) {
                    idx[l] = j + 1;
                    break;
            }
        }
        idx[i] = i + 1;
    }
}


/// Checks if the given size is larger than available. If so, resizes the vector.
template <typename T>
T* upsize(
    int64_t target_sz,
    std::vector<T>& A
) {
    if ((int64_t) A.size() < target_sz)
        A.resize(target_sz, 0);

    return A.data();
}


/// Changes the number of rows of a column-major matrix.
/// Resulting array is to be k by n - THIS IS SIZING DOWN
template <typename T>
T* row_resize(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
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
    const std::vector<T>& A,
    std::vector<T>& A_cpy,
    std::vector<T>& s,
    bool verbose
) {

    // Copy to avoid any changes
    T* A_cpy_dat = upsize(m * n, A_cpy);
    T* s_dat = upsize(n, s);

    // Packed storage check
    if (A.size() < A_cpy.size()) {
        // Convert to normal format
        lapack::tfttr(Op::NoTrans, Uplo::Upper, n, A.data(), A_cpy_dat, m);
    } else {
        lapack::lacpy(MatrixType::General, m, n, A.data(), m, A_cpy_dat, m);
    }
    lapack::gesdd(Job::NoVec, m, n, A_cpy_dat, m, s_dat, NULL, m, NULL, n);

    T cond_num = s_dat[0] / s_dat[n - 1];

    if (verbose)
        printf("CONDITION NUMBER: %f\n", cond_num);

    return cond_num;
}

// Computes the numerical rank of a given matirx
template <typename T>
int64_t rank_check(
    int64_t m,
    int64_t n,
    const std::vector<T>& A
) {
    std::vector<T> A_pre_cpy;
    std::vector<T> s;
    RandLAPACK::util::cond_num_check(m, n, A, A_pre_cpy, s, false);

    for(int i = 0; i < n; ++i) {
        if (s[i] / s[0] <= 5 * std::numeric_limits<T>::epsilon())
            return i - 1;
    }
    return n;
}

/// Checks whether matrix A has orthonormal columns.
template <typename T>
bool orthogonality_check(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<T>& A,
    std::vector<T>& A_gram,
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
    RandBLAS::RNGState<RNG> state
) {

    std::vector<T> buf (n, 0.0);
    std::vector<T> buf1 (m, 0.0);

    RandBLAS::DenseDist DV{.n_rows = n, .n_cols = 1};
    state = RandBLAS::fill_dense(DV, buf.data(), state);

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
    const std::vector<T>& A,
    std::vector<T>& A_norm
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

} // end namespace util
#endif
