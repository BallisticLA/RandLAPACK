#ifndef randlapack_gen_h
#define randlapack_gen_h

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <sstream>
#include <fstream>

namespace RandLAPACK::gen {

/// An enumeration describing various matrix types by name.
/// Each matrix type can be generated bt mat_gen() utility function.
/// This list is expected to grow.
enum mat_type {
    polynomial, 
    exponential, 
    gaussian, 
    step, 
    spiked, 
    adverserial, 
    bad_cholqr,
    kahan,
    custom_input};

/// A struct containing info about a given matrix to be generated by mat_gen().
/// Requires only the size and type of a matrix by default, but can have other optional parameters.
template <typename T>
struct mat_gen_info {
    int64_t rows;
    int64_t cols;
    int64_t rank;
    mat_type m_type;
    T cond_num;
    T scaling;
    T exponent;
    bool diag;
    bool check_true_rank;
    T theta;
    T perturb;
    char* filename;
    int workspace_query_mod;
    T frac_spectrum_one;

    mat_gen_info(int64_t& m, int64_t& n, mat_type t) {
        rows = m;
        cols = n;
        m_type = t;
        /// default values
        diag = false;
        rank = n;
        cond_num = 1.0;
        scaling = 1.0;
        exponent = 1.0;
        theta = 1.0;
        perturb = 1.0;
        check_true_rank = false;
        frac_spectrum_one = 0.1;
    }
};

/// Given singular values, generates left and right singular vectors and combines all into a single matrix.
/// Note: Printed matrix A may have different rank from actual generated matrix A
template <typename T, typename RNG>
void gen_singvec(
    int64_t m,
    int64_t n,
    T* A,
    int64_t k,
    T* S,
    RandBLAS::RNGState<RNG> &state
) {
    T* U        = ( T * ) calloc( m * k, sizeof( T ) );
    T* V        = ( T * ) calloc( n * k, sizeof( T ) );
    T* tau      = ( T * ) calloc( k    , sizeof( T ) );
    T* Gemm_buf = ( T * ) calloc( m * k, sizeof( T ) );

    RandBLAS::DenseDist DU(m, k);
    RandBLAS::DenseDist DV(n, k);
    state = RandBLAS::fill_dense(DU, U, state).second;
    state = RandBLAS::fill_dense(DV, V, state).second;

    lapack::geqrf(m, k, U, m, tau);
    lapack::ungqr(m, k, k, U, m, tau);

    lapack::geqrf(n, k, V, n, tau);
    lapack::ungqr(n, k, k, V, n, tau);

    blas::copy(m * k, U, 1, Gemm_buf, 1);
    for(int i = 0; i < k; ++i)
        blas::scal(m, S[i + k * i], &Gemm_buf[i * m], 1);

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, k, 1.0, Gemm_buf, m, V, n, 0.0, A, m);

    free(U);
    free(V);
    free(tau);
    free(Gemm_buf);
}

/// Generates a matrix with polynomially-decaying spectrum of the following form:
/// s_i = a(i + b)^p, where p is the user-defined exponent constant, a and b are computed
/// using p and the user-defined condition number parameter and the first 
/// (100 * frac_spectrum_one) percent of the  singular values are equal to one.
/// User can optionally choose for the matrix to be diagonal.
/// The output matrix has k singular values. 
template <typename T, typename RNG>
void gen_poly_mat(
    int64_t &m,
    int64_t &n,
    T* A,
    int64_t k,
    T frac_spectrum_one,
    T cond,
    T p,
    bool diagon,
    RandBLAS::RNGState<RNG> &state
) {

    // Predeclare to all nonzero constants, start decay where needed
    T* s = ( T * ) calloc( k,     sizeof( T ) );
    T* S = ( T * ) calloc( k * k, sizeof( T ) );

    // The first 10% of the singular values will be equal to one
    int offset = (int) floor(k * frac_spectrum_one);
    T first_entry = 1.0;
    T last_entry = first_entry / cond;
    T neg_invp = -((T)1.0)/p;
    T a = std::pow((std::pow(last_entry, neg_invp) - std::pow(first_entry, neg_invp)) / (k - offset), p);
    T b = std::pow(a * first_entry, neg_invp) - offset;
    // apply lambda function to every entry of s
    std::fill(s, s + offset, 1.0);
    for (int i = offset; i < k; ++i) {
        s[i] = 1 / (a * std::pow(offset + b, p));
        ++offset;
    }

    // form a diagonal S
    RandLAPACK::util::diag(k, k, s, k, S);

    if (diagon) {
        lapack::lacpy(MatrixType::General, k, k, S, k, A, k);
    } else {
        RandLAPACK::gen::gen_singvec(m, n, A, k, S, state);
    }

    free(s);
    free(S);
}

/// Generates a matrix with exponentially-decaying spectrum of the following form:
/// s_i = e^((i + 1) * -t), where t is computed using the user-defined cndition number parameter;
/// the first 10 percent of the singular values are equal to one.
/// User can optionally choose for the matrix to be diagonal.
/// The output matrix has k singular values. 
template <typename T, typename RNG>
void gen_exp_mat(
    int64_t &m,
    int64_t &n,
    T* A,
    int64_t k,
    T cond,
    bool diagon,
    RandBLAS::RNGState<RNG> &state
) {
    T* s = ( T * ) calloc( k,     sizeof( T ) );
    T* S = ( T * ) calloc( k * k, sizeof( T ) );

    // The first 10% of the singular values will be =1
    int offset = (int) floor(k * 0.1);

    T t = -log(1 / cond) / (k - offset);

    T cnt = 0.0;
    // apply lambda function to every entry of s
    // Please make sure that the first singular value is always 1
    std::fill(s, s + offset, 1.0);
    for (int i = offset; i < k; ++i) {
        s[i] = (std::exp(++cnt * -t));
        ++offset;
    }

    // form a diagonal S
    RandLAPACK::util::diag(k, k, s, k, S);
    if (diagon) {
        lapack::lacpy(MatrixType::General, k, k, S, k, A, k);
    } else {
        RandLAPACK::gen::gen_singvec(m, n, A, k, S, state);
    }

    free(s);
    free(S);
}

/// Generates matrix with a staircase spectrum with 4 steps.
/// Output matrix is m by n of rank k.
/// Boolean parameter 'diag' signifies whether the matrix is to be
/// generated as diagonal.
/// Parameter 'cond' signfies the condition number of a generated matrix.
template <typename T, typename RNG>
void gen_step_mat(
    int64_t &m,
    int64_t &n,
    T* A,
    int64_t k,
    T cond,
    bool diagon,
    RandBLAS::RNGState<RNG> &state
) {

    // Predeclare to all nonzero constants, start decay where needed
    T* s = ( T * ) calloc( k,     sizeof( T ) );
    T* S = ( T * ) calloc( k * k, sizeof( T ) );

    // We will have 4 steps controlled by the condition number size and starting with 1
    int offset = (int) (k / 4);

    std::fill(s, s + offset, 1.0);
    std::fill(s + offset + 1, s + 2 * offset, 8.0 / cond);
    std::fill(s + 2 * offset + 1, s + 3 * offset, 4.0 / cond);
    std::fill(s + 3 * offset + 1, s + k, 1.0 / cond);

    // form a diagonal S
    RandLAPACK::util::diag(k, k, s, k, S);

    if (diagon) {
        lapack::lacpy(MatrixType::General, k, k, S, k, A, k);
    } else {
        RandLAPACK::gen::gen_singvec(m, n, A, k, S, state);
    }

    free(s);
    free(S);
}

/// Generates a matrix with high coherence between the left singular vectors.
/// Output matrix is m by n, full-rank.
/// Such matrix would be difficult to sketch.
/// Right singular vectors are sampled uniformly at random.
template <typename T, typename RNG>
void gen_spiked_mat(
    int64_t &m,
    int64_t &n,
    T* A,
    T spike_scale,
    RandBLAS::RNGState<RNG> &state
) {
    int64_t num_rows_sampled = n / 2;

    /// sample from [m] without replacement. Get the row indices for a tall LASO with a single column.
    RandBLAS::SparseDist DS = {.n_rows = m, .n_cols = 1, .vec_nnz = num_rows_sampled, .major_axis = RandBLAS::MajorAxis::Long};
    RandBLAS::SparseSkOp<T> S(DS, state);
    state = RandBLAS::fill_sparse(S);

    T* V   = ( T * ) calloc( n * n, sizeof( T ) );
    T* tau = ( T * ) calloc( n,     sizeof( T ) );

    RandBLAS::DenseDist DV(n, n);
    state = RandBLAS::fill_dense(DV, V, state).second;

    lapack::geqrf(n, n, V, n, tau);
    lapack::ungqr(n, n, n, V, n, tau);

    // Fill A with stacked copies of V
    int start = 0;
    while(start + n <= m){
        for(int j = 0; j < n; ++j) {
            blas::copy(n, &V[m * j], 1, &A[start + (m * j)], 1);
        }
        start += n;
    }
    // Scale randomly sampled rows
    start = 0;
    while (start + m <= m * n) {
        for(int i = 0; i < num_rows_sampled; ++i) {
            A[start + (S.cols)[i] - 1] *= spike_scale;
        }
        start += m;
    }

    free(V);
    free(tau);
}

/// Generates a numerically rank-deficient matrix.
/// Added per Oleg's suggestion.
/// Output matrix is m by n of some rank k < n.
/// Generates a matrix of the form A = UV, where
/// U is an m by n Gaussian matrix whose first row was scaled by a factor sigma, and that then 
/// was orthonormalized with a Householder QR. 
/// The matrix V is the upper triangular part of an n × n 
/// orthonormalized Gaussian matrix with modified diagonal entries to diag(V) *= [1, 10^-15, . . . , 10^-15, 10^-15].
template <typename T, typename RNG>
void gen_oleg_adversarial_mat(
    int64_t &m,
    int64_t &n,
    T* A,
    T sigma,
    RandBLAS::RNGState<RNG> &state
) {

    T scaling_factor_U = sigma;
    T scaling_factor_V = 10e-3;

    T* U    = ( T * ) calloc( m * n, sizeof( T ) );
    T* V    = ( T * ) calloc( n * n, sizeof( T ) );
    T* tau1 = ( T * ) calloc( n,     sizeof( T ) );
    T* tau2 = ( T * ) calloc( n,     sizeof( T ) );

    RandBLAS::DenseDist DU(m, n);
    state = RandBLAS::fill_dense(DU, U, state).second;

    RandBLAS::DenseDist DV(n, n);
    state = RandBLAS::fill_dense(DV, V, state).second;

    for(int i = 0; i < n; ++i) {
        //U_dat[m * i + 1] *= scaling_factor_U;
        for(int j = 0; j < 10; ++j) {
            U[m * i + j] *= scaling_factor_U;
        }
    }

    lapack::geqrf(m, n, U, m, tau1);
    lapack::ungqr(m, n, n, U, m, tau1);

    lapack::geqrf(n, n, V, n, tau2);
    lapack::ungqr(n, n, n, V, n, tau2);

    // Grab an upper-triangular portion of V
    RandLAPACK::util::get_U(n, n, V, n);

    for(int i = 11; i < n; ++i)
        V[n * i + i] *= scaling_factor_V;

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U, m, V, n, 0.0, A, m);

    free(U);
    free(V);
    free(tau1);
    free(tau2);
}

/// Per Oleg Balabanov's suggestion, this matrix is supposed to break QB with Cholesky QR.
/// Output matrix is m by n, full-rank.
/// Parameter 'k' signifies the dimension of a sketching operator.
/// Boolean parameter 'diag' signifies whether the matrix is to be
/// generated as diagonal.
/// Parameter 'cond' signfies the condition number of a generated matrix.
template <typename T, typename RNG>
void gen_bad_cholqr_mat(
    int64_t &m,
    int64_t &n,
    T* A,
    int64_t k,
    T cond,
    bool diagon,
    RandBLAS::RNGState<RNG> &state
) {
    T* s = ( T * ) calloc( n,     sizeof( T ) );
    T* S = ( T * ) calloc( n * n, sizeof( T ) );

    // The first k singular values will be =1
    int offset = k;
    std::fill(s, s + offset, 1.0);

    // Then, we start with 10^-8 and decrease exponentially
    T t = log(std::pow(10, 8) / cond) / (1 - (n - offset));

    T cnt = 0.0;
    // apply lambda function to every entry of s
    // Please make sure that the first singular value is always 1
    std::for_each(s + offset, s + k,
        // Lambda expression begins
        [&t, &cnt](T &entry) {
                entry = (std::exp(t) / std::pow(10, 8)) * (std::exp(++cnt * -t));
        }
    );

    // form a diagonal S
    RandLAPACK::util::diag(k, k, s, k, S);
    if (diagon) {
        lapack::lacpy(MatrixType::General, k, k, S, k, A, k);
    } else {
        RandLAPACK::gen::gen_singvec(m, n, A, k, S, state);
    }

    free(s);
    free(S);
}

/// Generates Kahan matrix
template <typename T>
void gen_kahan_mat(
    int64_t m,
    int64_t n,
    T* A,
    T theta,
    T perturb
) {
    T* S = ( T * ) calloc( m * m, sizeof( T ) );
    T* C = ( T * ) calloc( m * m, sizeof( T ) );

    for (int i = 0; i < n; ++i) {
        A[(m + 1) * i] = perturb * std::numeric_limits<double>::epsilon() * (m - i);
        S[(m + 1) * i] = std::pow(std::sin(i), i);
        for(int j = 0; j < i; ++ j)
            C[(m * i) + j] = std::cos(theta); 
        C[m * i + i] = 1.0;
    }

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, m, m, 1.0, S, m, C, m, 1.0, A, m);

    free(S);
    free(C);
}

/// Reads a matrix from a file
template <typename T>
void process_input_mat(
    int64_t &m,
    int64_t &n,
    T* A,
    char* filename,
    int& workspace_query_mod
) {
    // We only check the size of the input data.
    if (workspace_query_mod) {
        std::string line;
        std::string line_entry;

        // Read input file
        std::ifstream inputMat(filename);

        // Count numcols.
        std::getline(inputMat, line);
        std::istringstream lineStream(line);
        while (lineStream >> line_entry)
            ++n;

        // Count numrows - already got through row 1.
        ++m;
        while (std::getline(inputMat, line))
            ++m;

        // Exit querying mod.
        workspace_query_mod = 0;
    } else {
        double value;
        int i, j;
        // Read input file
        std::ifstream inputMat(filename);

        // Place the contents of a file into the matrix space.
        // Matrix is input in a row-major order, we process data in column-major.
        // Reads here are, unfortunately, sequential;
        for(j = 0; j < m; ++j) {
            for(i = 0; i < n; ++i) {
                inputMat >> value;
                A[m * i + j] = value;
            }
        }
    }
}

/// 'Entry point' routine for matrix generation.
/// Calls functions for different mat type to fill the contents of a provided standard vector.
template <typename T, typename RNG>
void mat_gen(
    mat_gen_info<T> &info,
    T* A,
    RandBLAS::RNGState<RNG> &state
) {

    switch(info.m_type) {
        case polynomial:
                // Generating matrix with polynomially decaying singular values
                RandLAPACK::gen::gen_poly_mat(info.rows, info.cols, A, info.rank, info.frac_spectrum_one, info.cond_num, info.exponent, info.diag, state);
                break;
        case exponential:
                // Generating matrix with exponentially decaying singular values
                RandLAPACK::gen::gen_exp_mat(info.rows, info.cols, A, info.rank, info.cond_num, info.diag, state);
                break;
            break;
        case gaussian: {
                // Gaussian random matrix
                RandBLAS::DenseDist D(info.rows, info.cols);
                state = RandBLAS::fill_dense(D, A, state).second;
            }
            break;
        case step: {
                // Generating matrix with a staircase-like spectrum
                RandLAPACK::gen::gen_step_mat(info.rows, info.cols, A, info.rank, info.cond_num, info.diag, state);
            }    
            break;
        case spiked: {
                // This matrix may be numerically rank deficient
                RandLAPACK::gen::gen_spiked_mat(info.rows, info.cols, A, info.scaling, state);
                if(info.check_true_rank)
                    info.rank = RandLAPACK::util::rank_check(info.rows, info.cols, A);
            }
            break;
        case adverserial: {
                // This matrix may be numerically rank deficient
                RandLAPACK::gen::gen_oleg_adversarial_mat(info.rows, info.cols, A, info.scaling, state);
                if(info.check_true_rank)
                    info.rank = RandLAPACK::util::rank_check(info.rows, info.cols, A);
            }
            break;
        case bad_cholqr: {
                // Per Oleg's suggestion, this is supposed to make QB fail with CholQR for orth/stab
                RandLAPACK::gen::gen_bad_cholqr_mat(info.rows, info.cols, A, info.rank, info.cond_num, info.diag, state);
            }
            break;
        case kahan: {
                // Generates Kahan Matrix
                RandLAPACK::gen::gen_kahan_mat(info.rows, info.cols, A, info.theta, info.perturb);
            }
            break;
        case custom_input: {
                // Generates Kahan Matrix
                RandLAPACK::gen::process_input_mat(info.rows, info.cols, A, info.filename, info.workspace_query_mod);
            }
            break;
        default:
            throw std::runtime_error(std::string("Unrecognized case."));
            break;
    }
}
}
#endif
