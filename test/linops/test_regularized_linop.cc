// Unit tests for RegularizedLinOp<JLO>:
//   verifies that A_aug = [J; λI] applies correctly for forward, adjoint,
//   and sparse-SASO sketch paths, by comparing against an explicitly-built
//   (m+n) × n dense matrix.

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include <random>
#include <vector>


using RandLAPACK::linops::KroneckerOperator;
using RandLAPACK::linops::RegularizedLinOp;
using RandLAPACK::linops::DenseLinOp;
using blas::Layout;
using blas::Op;
using blas::Side;


// Build A = kron(A2, A1) explicitly in ColMajor.
template <typename T>
static void materialize_kron(int64_t m1, int64_t n1, int64_t m2, int64_t n2,
                             const T* A1, const T* A2, T* A_out)
{
    int64_t M = m1 * m2;
    for (int64_t j_outer = 0; j_outer < n2; ++j_outer) {
        for (int64_t j_inner = 0; j_inner < n1; ++j_inner) {
            int64_t j = j_outer * n1 + j_inner;
            for (int64_t i_outer = 0; i_outer < m2; ++i_outer) {
                T a2 = A2[i_outer + j_outer * m2];
                for (int64_t i_inner = 0; i_inner < m1; ++i_inner) {
                    int64_t i = i_outer * m1 + i_inner;
                    A_out[i + j * M] = a2 * A1[i_inner + j_inner * m1];
                }
            }
        }
    }
}


// Build A_aug = [A; λI] explicitly in ColMajor.
template <typename T>
static void materialize_augmented(int64_t M, int64_t N, T lambda,
                                   const T* A, T* A_aug)
{
    int64_t M_aug = M + N;
    // Top: copy A into rows [0..M).
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t i = 0; i < M; ++i) {
            A_aug[i + j * M_aug] = A[i + j * M];
        }
        // Bottom: λI in rows [M..M+N).
        for (int64_t i = 0; i < N; ++i) {
            A_aug[(M + i) + j * M_aug] = (i == j) ? lambda : (T)0;
        }
    }
}


template <typename T>
static void fill_random(std::vector<T>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (auto& x : v) x = dist(rng);
}


template <typename T>
static T frobenius_diff(const std::vector<T>& a, const std::vector<T>& b) {
    T sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        T d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}


class TestRegularizedLinOp : public ::testing::Test {};


// Forward + adjoint single column on a Kronecker-backed RegularizedLinOp
// must match the explicitly-materialized A_aug = [kron(A2,A1); λI].
TEST_F(TestRegularizedLinOp, kron_forward_adjoint_single_column) {
    using T = double;
    int64_t m1 = 4, n1 = 3, m2 = 5, n2 = 2;
    int64_t M = m1 * m2;
    int64_t N = n1 * n2;
    int64_t M_aug = M + N;
    T lambda = 0.7;

    std::vector<T> A1(m1 * n1), A2(m2 * n2);
    fill_random(A1, 11);
    fill_random(A2, 22);

    std::vector<T> A_full(M * N), A_aug_full(M_aug * N);
    materialize_kron(m1, n1, m2, n2, A1.data(), A2.data(), A_full.data());
    materialize_augmented(M, N, lambda, A_full.data(), A_aug_full.data());

    KroneckerOperator<T> J(m1, n1, m2, n2, A1.data(), A2.data());
    RegularizedLinOp<KroneckerOperator<T>> J_aug(J, lambda);
    DenseLinOp<T> A_aug_dense(M_aug, N, A_aug_full.data(), M_aug, Layout::ColMajor);

    // Forward: y = A_aug x
    {
        std::vector<T> x(N), y_aug(M_aug, 0), y_dense(M_aug, 0);
        fill_random(x, 33);
        J_aug(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
              M_aug, 1, N, (T)1.0, x.data(), N, (T)0.0, y_aug.data(), M_aug);
        A_aug_dense(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    M_aug, 1, N, (T)1.0, x.data(), N, (T)0.0, y_dense.data(), M_aug);
        EXPECT_LT(frobenius_diff(y_aug, y_dense), 1e-12);
    }
    // Adjoint: z = A_aug^T y
    {
        std::vector<T> y(M_aug), z_aug(N, 0), z_dense(N, 0);
        fill_random(y, 44);
        J_aug(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              N, 1, M_aug, (T)1.0, y.data(), M_aug, (T)0.0, z_aug.data(), N);
        A_aug_dense(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                    N, 1, M_aug, (T)1.0, y.data(), M_aug, (T)0.0, z_dense.data(), N);
        EXPECT_LT(frobenius_diff(z_aug, z_dense), 1e-12);
    }
}


// alpha/beta + multi-column on forward and adjoint.
TEST_F(TestRegularizedLinOp, kron_alpha_beta_multi_column) {
    using T = double;
    int64_t m1 = 3, n1 = 2, m2 = 4, n2 = 3;
    int64_t M = m1 * m2;
    int64_t N = n1 * n2;
    int64_t M_aug = M + N;
    int64_t k = 5;
    T lambda = 0.4;

    std::vector<T> A1(m1 * n1), A2(m2 * n2);
    fill_random(A1, 51);
    fill_random(A2, 52);
    std::vector<T> A_full(M * N), A_aug_full(M_aug * N);
    materialize_kron(m1, n1, m2, n2, A1.data(), A2.data(), A_full.data());
    materialize_augmented(M, N, lambda, A_full.data(), A_aug_full.data());

    KroneckerOperator<T> J(m1, n1, m2, n2, A1.data(), A2.data());
    RegularizedLinOp<KroneckerOperator<T>> J_aug(J, lambda);
    DenseLinOp<T> A_aug_dense(M_aug, N, A_aug_full.data(), M_aug, Layout::ColMajor);

    // Forward with non-trivial alpha/beta.
    {
        std::vector<T> B(N * k), C_aug(M_aug * k), C_dense(M_aug * k);
        fill_random(B, 61);
        fill_random(C_aug, 71);
        C_dense = C_aug;
        T alpha = 0.7, beta = -0.3;
        J_aug(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
              M_aug, k, N, alpha, B.data(), N, beta, C_aug.data(), M_aug);
        A_aug_dense(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    M_aug, k, N, alpha, B.data(), N, beta, C_dense.data(), M_aug);
        EXPECT_LT(frobenius_diff(C_aug, C_dense), 1e-11);
    }
    // Adjoint with non-trivial alpha/beta.
    {
        std::vector<T> B(M_aug * k), C_aug(N * k), C_dense(N * k);
        fill_random(B, 81);
        fill_random(C_aug, 91);
        C_dense = C_aug;
        T alpha = 1.3, beta = 0.2;
        J_aug(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              N, k, M_aug, alpha, B.data(), M_aug, beta, C_aug.data(), N);
        A_aug_dense(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                    N, k, M_aug, alpha, B.data(), M_aug, beta, C_dense.data(), N);
        EXPECT_LT(frobenius_diff(C_aug, C_dense), 1e-11);
    }
}


// Sparse SASO sketch via the COO-filter dispatch path:
// J_aug(Side::Right, NoTrans, NoTrans, S, ...) must match DenseLinOp on the
// materialized A_aug.
TEST_F(TestRegularizedLinOp, sparse_sketch_apply) {
    using T = double;
    using RNG = r123::Philox4x32;
    int64_t m1 = 6, n1 = 4, m2 = 5, n2 = 3;
    int64_t M = m1 * m2;
    int64_t N = n1 * n2;
    int64_t M_aug = M + N;
    int64_t d = 8;
    int64_t saso_nnz = 3;
    T lambda = 0.6;

    std::vector<T> A1(m1 * n1), A2(m2 * n2);
    fill_random(A1, 101);
    fill_random(A2, 102);
    std::vector<T> A_full(M * N), A_aug_full(M_aug * N);
    materialize_kron(m1, n1, m2, n2, A1.data(), A2.data(), A_full.data());
    materialize_augmented(M, N, lambda, A_full.data(), A_aug_full.data());

    KroneckerOperator<T> J(m1, n1, m2, n2, A1.data(), A2.data());
    RegularizedLinOp<KroneckerOperator<T>> J_aug(J, lambda);
    DenseLinOp<T> A_aug_dense(M_aug, N, A_aug_full.data(), M_aug, Layout::ColMajor);

    RandBLAS::RNGState<RNG> state(42);
    RandBLAS::SparseDist DS(d, M_aug, saso_nnz);
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    RandBLAS::fill_sparse(S);
    auto state2 = state;
    RandBLAS::SparseSkOp<T, RNG> S2(DS, state2);
    RandBLAS::fill_sparse(S2);

    std::vector<T> A_hat_aug(d * N, 0);
    std::vector<T> A_hat_dense(d * N, 0);

    J_aug(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          d, N, M_aug, (T)1.0, S, (T)0.0, A_hat_aug.data(), d);
    A_aug_dense(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                d, N, M_aug, (T)1.0, S2, (T)0.0, A_hat_dense.data(), d);

    EXPECT_LT(frobenius_diff(A_hat_aug, A_hat_dense), 1e-12);
}
