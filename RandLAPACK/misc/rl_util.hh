#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <ctime>
#include <iomanip>

namespace RandLAPACK::util {

template <typename T>
void print_colmaj(int64_t n_rows, int64_t n_cols, T *a, int64_t lda, char label[])
{
	int64_t i, j;
    T val;
	std::cout << "\n" << label << std::endl;
    for (i = 0; i < n_rows; ++i) {
        std::cout << "\t";
        for (j = 0; j < n_cols - 1; ++j) {
            val = a[i + lda * j];
            if (val < 0) {
				//std::cout << string_format("  %2.4f,", val);
                printf("  %2.20f,", val);
            } else {
				//std::cout << string_format("   %2.4f", val);
				printf("   %2.20f,", val);
            }
        }
        // j = n_cols - 1
        val = a[i + lda * j];
        if (val < 0) {
   			//std::cout << string_format("  %2.4f,", val); 
			printf("  %2.20f,", val);
		} else {
            //std::cout << string_format("   %2.4f,", val);
			printf("   %2.20f,", val);
		}
        printf("\n");
    }
    printf("\n");
    return;
}

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

/// Find the condition number of a given matrix A.
template <typename T>
T cond_num_check(
    int64_t m,
    int64_t n,
    const T* A,
    bool verbose
) {
    T* A_cpy = ( T * ) calloc( m * n, sizeof( T ) );
    T* s     = ( T * ) calloc( n, sizeof( T ) );

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
    T* A_cpy = ( T * ) calloc( m * n, sizeof( T ) );
    T* s     = ( T * ) calloc( n, sizeof( T ) );

    lapack::lacpy(MatrixType::General, m, n, A, m, A_cpy, m);
    lapack::gesdd(Job::NoVec, m, n, A_cpy, m, s, NULL, m, NULL, n);

    for(int i = 0; i < n; ++i) {
        if (s[i] / s[0] <= 5 * std::numeric_limits<T>::epsilon())
            return i - 1;
    }

    free(A_cpy);
    free(s);

    return n;
}

/// Checks whether matrix A has orthonormal columns.
template <typename T>
bool orthogonality_check(
    int64_t m,
    int64_t k,
    T* A,
    bool verbose
) {

    T* A_gram  = ( T * ) calloc( k * k, sizeof( T ) );

    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A, m, 0.0, A_gram, k);

    for (int i = 0; i < k; ++i) {
        A_gram[i * k + i] -= 1.0;
    }
    T orth_err = lapack::lange(Norm::Fro, k, k, A_gram, k);

    if(verbose) {
        printf("Q ERROR:   %e\n\n", orth_err);
    }

    if (orth_err > 1.0e-10) {
        free(A_gram);
        return true;
    }

    free(A_gram);
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

// Custom implementation of orhr_col.
// Allows to choose whether to output T or tau.
template <typename T>
void rl_orhr_col(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    T* T_dat,
    T* D,
    bool output_tau
) {
    // We assume that the space for S, D has ben pre-allocated
    T buf = 0;

    int i;
    for(i = 0; i < n; ++i) {
        // S(i, i) = − sgn(Q(i, i)); = 1 if Q(i, i) == 0
        buf = A[i * lda + i];
        buf == 0 ? D[i] = 1 : D[i] = -((T(0) < buf) - (buf < T(0)));
        A[i * lda + i] -= D[i];
        // Scale ith column if L by diagonal element
        blas::scal(m - (i + 1), 1 / A[i * (lda + 1)], &A[(lda + 1) * i + 1], 1);
        // Perform Schur compliment update
        // A(i+1:m, i+1:n) = A(i+1:m, i+1:n) - (A(i+1:m, i) * A(i, i+1:n))
        
        char name [] = "In";
        RandLAPACK::util::print_colmaj(m, n, A, m, name);

        blas::ger(Layout::ColMajor, m - (i + 1), n - (i + 1), (T) -1.0, &A[(lda + 1) * i + 1], 1, &A[lda * (i + 1) + i], m, &A[(lda + 1) * (i + 1)], lda);	
    }

    if(output_tau) {
        // In this case, we are assuming that T_dat stores a vector tau of length n.
        blas::copy(n, A, lda + 1, T_dat, 1);
        #pragma omp parallel for
        for(i = 0; i < n; ++i)
            T_dat[i] *= -D[i];
    } else {
        // In this case, we are assuming that T_dat stores matrix T of size n by n.
        // Fing T = -R * diag(D) * Q_11^{-T}
        lapack::lacpy(MatrixType::Upper, n, n, A, lda, T_dat, n);
        for(i = 0; i < n; ++i) {
            blas::scal(i + 1, -D[i], &T_dat[n * i], 1);
        }
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Lower, Op::Trans, Diag::Unit, n, n, 1.0, A, lda, T_dat, n);	
    }
}

template <typename T>
// Function returns current date
std::string getCurrentDate() {
    // Get the current time
    std::time_t now = std::time(nullptr);
    // Convert to local time
    std::tm* localTime = std::localtime(&now);

    // Create a string stream to format the date
    std::ostringstream dateStream;
    dateStream << std::setw(4) << std::setfill('0') << (1900 + localTime->tm_year) << "_"  // Year
               << std::setw(2) << std::setfill('0') << (localTime->tm_mon + 1) << "_"      // Month
               << std::setw(2) << std::setfill('0') << localTime->tm_mday << "_";          // Day

    return dateStream.str();
}

} // end namespace util
