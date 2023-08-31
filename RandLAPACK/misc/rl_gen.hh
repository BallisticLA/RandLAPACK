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
    kahan};

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
    T pertrub;

    mat_gen_info(int64_t m, int64_t n, mat_type t) {
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
        pertrub = 1.0;
    }
};

/// Given singular values, generates left and right singular vectors and combines all into a single matrix.
/// Note: Printed matrix A may have different rank from actual generated matrix A
template <typename T, typename RNG>
void gen_singvec(
    int64_t m,
    int64_t n,
    std::vector<T> &A,
    int64_t k,
    std::vector<T> &S,
    RandBLAS::RNGState<RNG> &state
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

    RandBLAS::DenseDist DU{.n_rows = m, .n_cols = k};
    RandBLAS::DenseDist DV{.n_rows = n, .n_cols = k};
    state = RandBLAS::fill_dense(DU, U_dat, state);
    state = RandBLAS::fill_dense(DV, V_dat, state);

    lapack::geqrf(m, k, U_dat, m, tau_dat);
    lapack::ungqr(m, k, k, U_dat, m, tau_dat);

    lapack::geqrf(n, k, V_dat, n, tau_dat);
    lapack::ungqr(n, k, k, V_dat, n, tau_dat);

    blas::copy(m * k, U_dat, 1, Gemm_buf_dat, 1);
    for(int i = 0; i < k; ++i)
        blas::scal(m, S[i + k * i], &Gemm_buf_dat[i * m], 1);

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, k, 1.0, Gemm_buf_dat, m, V_dat, n, 0.0, A.data(), m);
}

/// Generates a matrix with polynomially-decaying spectrum of the following form:
/// s_i = a(i + b)^p, where p is the user-defined exponent constant, a and b are computed
/// using p and the user-defined condition number parameter and the first 10 percent of the 
/// singular values are equal to one.
/// User can optionally choose for the matrix to be diagonal.
/// The output matrix has k singular values. 
template <typename T, typename RNG>
void gen_poly_mat(
    int64_t &m,
    int64_t &n,
    std::vector<T> &A,
    int64_t k,
    T cond,
    T p,
    bool diagon,
    RandBLAS::RNGState<RNG> &state
) {

    // Predeclare to all nonzero constants, start decay where needed
    std::vector<T> s(k, 1.0);
    std::vector<T> S(k * k, 0.0);

    // The first 10% of the singular values will be equal to one
    int offset = (int) floor(k * 0.1);
    T first_entry = 1.0;
    T last_entry = first_entry / cond;
    T a = std::pow((std::pow(last_entry, -1 / p) - std::pow(first_entry, -1 / p)) / (k - offset), p);
    T b = std::pow(a * first_entry, -1 / p) - offset;
    // apply lambda function to every entry of s
    std::for_each(s.begin() + offset, s.end(),
        // Lambda expression begins
        [&p, &offset, &a, &b](T &entry) {
                entry = 1 / (a * std::pow(offset + b, p));
                ++offset;
        }
    );

    // form a diagonal S
    RandLAPACK::util::diag(k, k, s, k, S);

    if (diagon) {
        if (!(m == k || n == k)) {
            m = k;
            n = k;
            A.resize(k * k);
        }
        lapack::lacpy(MatrixType::General, k, k, S.data(), k, A.data(), k);
    } else {
        RandLAPACK::gen::gen_singvec(m, n, A, k, S, state);
    }
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
    std::vector<T> &A,
    int64_t k,
    T cond,
    bool diagon,
    RandBLAS::RNGState<RNG> &state
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
        [&t, &cnt](T &entry) {
                entry = (std::exp(++cnt * -t));
        }
    );

    // form a diagonal S
    RandLAPACK::util::diag(k, k, s, k, S);
    if (diagon) {
        if (!(m == k || n == k)) {
                m = k;
                n = k;
                A.resize(k * k);
        }
        lapack::lacpy(MatrixType::General, k, k, S.data(), k, A.data(), k);
    } else {
        RandLAPACK::gen::gen_singvec(m, n, A, k, S, state);
    }
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
    std::vector<T> &A,
    int64_t k,
    T cond,
    bool diagon,
    RandBLAS::RNGState<RNG> &state
) {

    // Predeclare to all nonzero constants, start decay where needed
    std::vector<T> s(k, 1.0);
    std::vector<T> S(k * k, 0.0);

    // We will have 4 steps controlled by the condition number size and starting with 1
    int offset = (int) (k / 4);

    std::fill(s.begin(), s.begin() + offset, 1);
    std::fill(s.begin() + offset + 1, s.begin() + 2 * offset, 8.0 / cond);
    std::fill(s.begin() + 2 * offset + 1, s.begin() + 3 * offset, 4.0 / cond);
    std::fill(s.begin() + 3 * offset + 1, s.end(), 1.0 / cond);

    // form a diagonal S
    RandLAPACK::util::diag(k, k, s, k, S);

    if (diagon) {
        if (!(m == k || n == k)) {
            m = k;
            n = k;
            A.resize(k * k);
        }
        lapack::lacpy(MatrixType::General, k, k, S.data(), k, A.data(), k);
    } else {
        gen_singvec(m, n, A, k, S, state);
    }
}

/// Generates a matrix with high coherence between the left singular vectors.
/// Output matrix is m by n, full-rank.
/// Such matrix would be difficult to sketch.
/// Right singular vectors are sampled uniformly at random.
template <typename T, typename RNG>
void gen_spiked_mat(
    int64_t &m,
    int64_t &n,
    std::vector<T> &A,
    T spike_scale,
    RandBLAS::RNGState<RNG> &state
) {
    int64_t num_rows_sampled = n / 2;

    /// sample from [m] without replacement. Get the row indices for a tall LASO with a single column.
    RandBLAS::SparseDist DS = {.n_rows = m, .n_cols = 1, .vec_nnz = num_rows_sampled, .major_axis = RandBLAS::MajorAxis::Long};
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    state = RandBLAS::fill_sparse(S);

    std::vector<T> V(n * n, 0.0);
    std::vector<T> tau(n, 0.0);

    RandBLAS::DenseDist DV{.n_rows = n, .n_cols = n};
    state = RandBLAS::fill_dense(DV, V.data(), state);

    lapack::geqrf(n, n, V.data(), n, tau.data());
    lapack::ungqr(n, n, n, V.data(), n, tau.data());

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
    std::vector<T> &A,
    T sigma,
    RandBLAS::RNGState<RNG> &state
) {

    T scaling_factor_U = sigma;
    T scaling_factor_V = 10e-3;

    std::vector<T> U(m * n, 0.0);
    std::vector<T> V(n * n, 0.0);
    std::vector<T> tau1(n, 0.0);
    std::vector<T> tau2(n, 0.0);

    RandBLAS::DenseDist DU{.n_rows = m, .n_cols = n};
    state = RandBLAS::fill_dense(DU, U.data(), state);

    RandBLAS::DenseDist DV{.n_rows = n, .n_cols = n};
    state = RandBLAS::fill_dense(DV, V.data(), state);

    T* U_dat = U.data();
    for(int i = 0; i < n; ++i) {
        //U_dat[m * i + 1] *= scaling_factor_U;
        for(int j = 0; j < 10; ++j) {
            U_dat[m * i + j] *= scaling_factor_U;
        }
    }

    lapack::geqrf(m, n, U.data(), m, tau1.data());
    lapack::ungqr(m, n, n, U.data(), m, tau1.data());

    lapack::geqrf(n, n, V.data(), n, tau2.data());
    lapack::ungqr(n, n, n, V.data(), n, tau2.data());

    // Grab an upper-triangular portion of V
    RandLAPACK::util::get_U(n, n, V.data(), n);

    T* V_dat = V.data();
    for(int i = 11; i < n; ++i)
        V_dat[n * i + i] *= scaling_factor_V;

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U.data(), m, V.data(), n, 0.0, A.data(), m);
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
    std::vector<T> &A,
    int64_t k,
    T cond,
    bool diagon,
    RandBLAS::RNGState<RNG> &state
) {

    std::vector<T> s(n, 1.0);
    std::vector<T> S(n * n, 0.0);

    // The first k singular values will be =1
    int offset = k;

    // Then, we start with 10^-8 and decrease exponentially
    T t = log(std::pow(10, 8) / cond) / (1 - (n - offset));

    T cnt = 0.0;
    // apply lambda function to every entry of s
    // Please make sure that the first singular value is always 1
    std::for_each(s.begin() + offset, s.end(),
        // Lambda expression begins
        [&t, &cnt](T &entry) {
                entry = (std::exp(t) / std::pow(10, 8)) * (std::exp(++cnt * -t));
        }
    );

    // form a diagonal S
    RandLAPACK::util::diag(k, k, s, k, S);
    if (diagon) {
        if (!(m == k || n == k)) {
                m = k;
                n = k;
                A.resize(k * k);
        }
        lapack::lacpy(MatrixType::General, k, k, S.data(), k, A.data(), k);
    } else {
        gen_singvec(m, n, A, k, S, state);
    }
}

/// Generates Kahan matrix
template <typename T, typename RNG>
void gen_kahan_mat(
    int64_t &m,
    int64_t &n,
    std::vector<T> &A,
    T theta,
    T pertrub,
    RandBLAS::RNGState<RNG> &state
) {

    std::vector<T> S(m * m, 0.0);
    std::vector<T> P(m * m, 0.0);
    std::vector<T> C(m * m, 0.0);

    for (int i = 0; i < n; ++i) {
        P[(m + 1) * i] = pertrub * std::numeric_limits<double>::epsilon() * (m - i);
        S[(m + 1) * i] = std::pow(std::sin(i), i);
        for(int j = 0; j < i; ++ j)
            C[(m * i) + j] = std::cos(theta); 
        C[m * i + i] = 1.0;
    }

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, m, m, 1.0, P.data(), m, S.data(), m, 0.0, A.data(), m);
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, m, 1.0, C.data(), m, A.data(), m);
}

/// 'Entry point' routine for matrix generation.
/// Calls functions for different mat type to fill the contents of a provided standard vector.
template <typename T, typename RNG>
void mat_gen(
    mat_gen_info<T> info,
    std::vector<T> &A,
    RandBLAS::RNGState<RNG> &state
) {
    // Base parameters
    int64_t m = info.rows;
    int64_t n = info.cols;
    int64_t k = info.rank;
    T* A_dat = RandLAPACK::util::upsize(m * n, A);

    switch(info.m_type) {
        case polynomial:
                // Generating matrix with polynomially decaying singular values
                RandLAPACK::gen::gen_poly_mat(m, n, A, k, info.cond_num, info.exponent, info.diag, state);
                break;
        case exponential:
                // Generating matrix with exponentially decaying singular values
                RandLAPACK::gen::gen_exp_mat(m, n, A, k, info.cond_num, info.diag, state);
                break;
            break;
        case gaussian: {
                // Gaussian random matrix
                RandBLAS::DenseDist D{.n_rows = m, .n_cols = n};
                state = RandBLAS::fill_dense(D, A_dat, state);
            }
            break;
        case step: {
                // Generating matrix with a staircase-like spectrum
                RandLAPACK::gen::gen_step_mat(m, n, A, k, info.cond_num, info.diag, state);
            }    
            break;
        case spiked: {
                // This matrix may be numerically rank deficient
                RandLAPACK::gen::gen_spiked_mat(m, n, A, info.scaling, state);
                if(info.check_true_rank)
                    k = RandLAPACK::util::rank_check(m, n, A);
            }
            break;
        case adverserial: {
                // This matrix may be numerically rank deficient
                RandLAPACK::gen::gen_oleg_adversarial_mat(m, n, A, info.scaling, state);
                if(info.check_true_rank)
                    k = RandLAPACK::util::rank_check(m, n, A);
            }
            break;
        case bad_cholqr: {
                // Per Oleg's suggestion, this is supposed to make QB fail with CholQR for orth/stab
                RandLAPACK::gen::gen_bad_cholqr_mat(m, n, A, k, info.cond_num, info.diag, state);
            }
            break;
        case kahan: {
                // Generates Kahan Matrix
                RandLAPACK::gen::gen_kahan_mat(m, n, A, info.theta, info.pertrub, state);
            }
            break;
        default:
            throw std::runtime_error(std::string("Unrecognized case."));
            break;
    }
}
}
#endif
