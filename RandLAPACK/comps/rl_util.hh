#ifndef randlapack_comps_util_h
#define randlapack_comps_util_h

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

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
        // Throw error
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
template <typename T> void get_L( int64_t m, int64_t n, std::vector<T>& L);

template <typename T>
void get_L(
    int64_t m,
    int64_t n,
    std::vector<T>& L
) {
    // Vector end pointer
    int size = m * n;
    // The unit diagonal elements of L were not stored.
    T* L_dat = L.data();
    L_dat[0] = 1;

    for(int i = m, j = 1; i < size && j < m; i += m, ++j) {
        std::for_each(&L_dat[i], &L_dat[i + j],
            // Lambda expression begins
            [](T& entry) {
                entry = 0.0;
            }
        );
        // The unit diagonal elements of L were not stored.
        L_dat[i + j] = 1;
    }
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

/// Generates left and right singular vectors for the three matrix types above.
/// Note: Printed matrix A may have different rank from actual generated matrix A
template <typename T>
void gen_mat( int64_t m, int64_t n, std::vector<T>& A, int64_t k, std::vector<T>& S, int32_t seed);
template <typename T>
void gen_mat(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t k,
    std::vector<T>& S,
    int32_t seed
) {

    std::vector<T> U(m * k, 0.0);
    std::vector<T> V(n * k, 0.0);
    std::vector<T> tau(k, 2.0);
    std::vector<T> Gemm_buf(m * k, 0.0);

    // Data pointer predeclarations for whatever is accessed more than once
    T* U_dat = U.data();
    T* V_dat = V.data();
    T* tau_dat = tau.data();
    T* Gemm_buf_dat = Gemm_buf.data();

    auto state = RandBLAS::base::RNGState(seed, 0);
    RandBLAS::dense::DenseDist DU{.n_rows = m, .n_cols = k};
    RandBLAS::dense::DenseDist DV{.n_rows = n, .n_cols = k};
    state = RandBLAS::dense::fill_buff(U_dat, DU, state);
    state = RandBLAS::dense::fill_buff(V_dat, DV, state);

    lapack::geqrf(m, k, U_dat, m, tau_dat);
    lapack::ungqr(m, k, k, U_dat, m, tau_dat);

    lapack::geqrf(n, k, V_dat, n, tau_dat);
    lapack::ungqr(n, k, k, V_dat, n, tau_dat);

    blas::copy(m * k, U_dat, 1, Gemm_buf_dat, 1);
    for(int i = 0; i < k; ++i) {
        blas::scal(m, S[i + k * i], &Gemm_buf_dat[i * m], 1);
    }

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, k, 1.0, Gemm_buf_dat, m, V_dat, n, 0.0, A.data(), m);
}

/// Generates matrix with the following singular values:
/// sigma_i = 1 / (i + 1)^pow (first k * 0.2 sigmas = 1
/// Can either be a diagonal matrix, or a full one.
/// In later case, left and right singular vectors are randomly-generated
/// and orthogonaized.
template <typename T>
void gen_poly_mat(
    int64_t& m,
    int64_t& n,
    std::vector<T>& A,
    int64_t k,
    T cond,
    bool diagon,
    int32_t seed
) {

    // Predeclare to all nonzero constants, start decay where needed
    std::vector<T> s(k, 1.0);
    std::vector<T> S(k * k, 0.0);

    // The first 10% of the singular values will be =1
    int offset = (int) floor(k * 0.1);

    // We have a set condition number, so need to find an exponent parameter
    // The higher the value, the faster the decay
    T t = log2(cond) / log2(k - offset);

    T cnt = 0.0;
    // apply lambda function to every entry of s
    std::for_each(s.begin() + offset, s.end(),
        // Lambda expression begins
        [&t, &cnt](T& entry) {
                entry = 1 / std::pow(++cnt, t);
        }
    );

    // form a diagonal S
    diag(k, k, s, k, S);

    if (diagon) {
        if (!(m == k || n == k)) {
            m = k;
            n = k;
            A.resize(k * k);
        }
        lapack::lacpy(MatrixType::General, k, k, S.data(), k, A.data(), k);
    } else {
        gen_mat(m, n, A, k, S, seed);
    }
}

/// Generates matrix with the following singular values:
/// sigma_i = e^((i + 1) * -pow) (first k * 0.2 sigmas = 1
/// Can either be a diagonal matrix, or a full one.
/// In later case, left and right singular vectors are randomly-generated
/// and orthogonaized.
template <typename T>
void gen_exp_mat(
    int64_t& m,
    int64_t& n,
    std::vector<T>& A,
    int64_t k,
    T cond,
    bool diagon,
    int32_t seed
) {

    std::vector<T> s(k, 1.0);
    std::vector<T> S(k * k, 0.0);

    // The first 10% of the singular values will be =1
    int offset = (int) floor(k * 0.1);

    T t = -log(1 / cond) / (k - offset);

    T cnt = 0.0;
    // apply lambda function to every entry of s
    // Please make sure that the first singular value is always 1
    std::for_each(s.begin() + offset, s.end(),
        // Lambda expression begins
        [&t, &cnt](T& entry) {
                entry = (std::exp(++cnt * -t));
        }
    );

    // form a diagonal S
    diag(k, k, s, k, S);
    if (diagon) {
        if (!(m == k || n == k)) {
                m = k;
                n = k;
                A.resize(k * k);
        }
        lapack::lacpy(MatrixType::General, k, k, S.data(), k, A.data(), k);
    } else {
        gen_mat(m, n, A, k, S, seed);
    }
}

/// Dimensions m and n may change if we want the diagonal matrix of rank k < min(m, n).
/// In that case, it would be of size k by k.
template <typename T>
void gen_mat_type(
    int64_t& m, // These may change
    int64_t& n,
    std::vector<T>& A,
    int64_t k,
    int32_t seed,
    const std::tuple<int, T, bool>& type
) {
    T* A_dat = A.data();
    auto state = RandBLAS::base::RNGState(seed, 0);

    switch(std::get<0>(type)) {
        /*
        First 3 cases are identical, varying ony in the entry generation function.
        is there a way to propagate the lambda expression or somehowe pass several parameners into a undary function in foreach?
        */
        case 0:
                // Generating matrix with polynomially decaying singular values
                //printf("TEST MATRIX: POLYNOMIAL DECAY sigma_i = (i + 1)^-pow (first k * 0.2 sigmas = 1)\n");
                RandLAPACK::util::gen_poly_mat(m, n, A, k, std::get<1>(type), std::get<2>(type), seed);
                break;
        case 1:
                // Generating matrix with exponentially decaying singular values
                //printf("TEST MATRIX: EXPONENTIAL DECAY sigma_i = e^((i + 1) * -pow) (first k * 0.2 sigmas = 1)\n");
                RandLAPACK::util::gen_exp_mat(m, n, A, k, std::get<1>(type), std::get<2>(type), seed);
                break;
        case 2: {
                // A = [A A]
                //printf("TEST MATRIX: A = [A A]\n");
                RandBLAS::dense::DenseDist D{.n_rows = m, .n_cols = k};
                RandBLAS::dense::fill_buff(A_dat, D, state);
                if (2 * k <= n)
                {
                    blas::copy(m * (n / 2), &A_dat[0], 1, &A_dat[(n / 2) * m], 1);
                }
            }
            break;
        case 3:
                // Zero matrix
                //printf("TEST MATRIX: ZERO\n");
                break;
        case 4: {
                // Random diagonal A of rank k
                //printf("TEST MATRIX: GAUSSIAN RANDOM DIAGONAL\n");
                std::vector<T> buf(k, 0.0);
                RandBLAS::dense::DenseDist D{.n_rows = k, .n_cols = 1};
                RandBLAS::dense::fill_buff(buf.data(), D, state);
                // Fills the first k diagonal elements
                diag(m, n, buf, k, A);
            }
            break;
        case 5: {
                // A = diag(sigma), where sigma_1 = ... = sigma_l > sigma_{l + 1} = ... = sigma_n
                // In the case below, sigma = 1 | 0.5
                //printf("TEST MATRIX: A = diag(sigma), where sigma_1 = ... = sigma_l > sigma_{l + 1} = ... = sigma_n\n");
                std::vector<T> buf(n, 1.0);
                std::for_each(buf.begin() + (n / 2), buf.end(),
                    // Lambda expression begins
                    [](T& entry)
                    {
                            entry -= 0.5;
                    }
                );
                diag(m, n, buf, n, A);
            }
            break;
        case 6: {
                // Gaussian random matrix
                //printf("TEST MATRIX: GAUSSIAN RANDOM\n");
                RandBLAS::dense::DenseDist D{.n_rows = m, .n_cols = n};
                RandBLAS::dense::fill_buff(A_dat, D, state);
            }
            break;
        default:
            throw std::runtime_error(std::string("Unrecognized case."));
            break;
    }
}

/// Find the condition number of a given matrix A.
template <typename T>
T cond_num_check(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    std::vector<T>& A_cpy,
    std::vector<T>& s,
    bool verbosity
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

    if (verbosity)
        printf("CONDITION NUMBER: %f\n", cond_num);

    return cond_num;
}

/// Checks whether matrix A has orthonormal columns.
template <typename T>
bool orthogonality_check(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<T>& A,
    std::vector<T>& A_gram,
    bool verbosity
) {

    const T* A_dat = A.data();
    T* A_gram_dat = A_gram.data();

    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A_dat, m, A_dat, m, 0.0, A_gram_dat, n);
    for (int oi = 0; oi < k; ++oi) {
        A_gram_dat[oi * n + oi] -= 1.0;
    }
    T orth_err = lapack::lange(Norm::Fro, n, n, A_gram_dat, k);

    if(verbosity) {
        printf("Q ERROR:   %e\n\n", orth_err);
    }

    if (orth_err > 1.0e-10)
        return true;

    return false;
}

} // end namespace util
#endif