// Unit tests for KroneckerOperator<T>:
//   forward / adjoint dense apply (single column, multi-column),
//   sparse SASO sketching (S * A) via the SkOp overload.
// All tests verify against an explicitly-materialized A = kron(A2, A1).

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include <random>
#include <vector>


using RandLAPACK::linops::KroneckerOperator;
using RandLAPACK::linops::DenseLinOp;
using blas::Layout;
using blas::Op;
using blas::Side;


// Build A = kron(A2, A1) explicitly in ColMajor as an (m1*m2) × (n1*n2) buffer.
template <typename T>
static void materialize_kron(int64_t m1, int64_t n1, int64_t m2, int64_t n2,
                             const T* A1, const T* A2, T* A_out)
{
    int64_t M = m1 * m2;
    int64_t N = n1 * n2;
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
    (void)M; (void)N;
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


class TestKroneckerOperator : public ::testing::Test {};


// Test forward apply (NoTrans) and adjoint (Trans) for single-column input.
TEST_F(TestKroneckerOperator, forward_and_adjoint_single_column) {
    using T = double;
    int64_t m1 = 4, n1 = 3, m2 = 5, n2 = 2;
    int64_t M = m1 * m2;  // 20
    int64_t N = n1 * n2;  // 6

    std::vector<T> A1(m1 * n1), A2(m2 * n2), A_full(M * N);
    fill_random(A1, 11);
    fill_random(A2, 22);
    materialize_kron(m1, n1, m2, n2, A1.data(), A2.data(), A_full.data());

    KroneckerOperator<T> kop(m1, n1, m2, n2, A1.data(), A2.data());
    DenseLinOp<T> dop(M, N, A_full.data(), M, Layout::ColMajor);

    // forward: y = A x
    {
        std::vector<T> x(N), y_kop(M, 0), y_dense(M, 0);
        fill_random(x, 33);
        kop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            M, 1, N, (T)1.0, x.data(), N, (T)0.0, y_kop.data(), M);
        dop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            M, 1, N, (T)1.0, x.data(), N, (T)0.0, y_dense.data(), M);
        EXPECT_LT(frobenius_diff(y_kop, y_dense), 1e-12);
    }
    // adjoint: z = A^T y
    {
        std::vector<T> y(M), z_kop(N, 0), z_dense(N, 0);
        fill_random(y, 44);
        kop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
            N, 1, M, (T)1.0, y.data(), M, (T)0.0, z_kop.data(), N);
        dop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
            N, 1, M, (T)1.0, y.data(), M, (T)0.0, z_dense.data(), N);
        EXPECT_LT(frobenius_diff(z_kop, z_dense), 1e-12);
    }
}


// Test alpha and beta correctness on a multi-column input.
TEST_F(TestKroneckerOperator, alpha_beta_multi_column) {
    using T = double;
    int64_t m1 = 3, n1 = 2, m2 = 4, n2 = 3;
    int64_t M = m1 * m2, N = n1 * n2;
    int64_t k = 5;

    std::vector<T> A1(m1 * n1), A2(m2 * n2), A_full(M * N);
    fill_random(A1, 51);
    fill_random(A2, 52);
    materialize_kron(m1, n1, m2, n2, A1.data(), A2.data(), A_full.data());

    KroneckerOperator<T> kop(m1, n1, m2, n2, A1.data(), A2.data());
    DenseLinOp<T> dop(M, N, A_full.data(), M, Layout::ColMajor);

    // forward: B is N × k, C accumulator nonzero
    {
        std::vector<T> B(N * k), C_kop(M * k), C_dense(M * k);
        fill_random(B, 61);
        fill_random(C_kop, 71);
        C_dense = C_kop;
        T alpha = 0.7, beta = -0.3;
        kop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            M, k, N, alpha, B.data(), N, beta, C_kop.data(), M);
        dop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            M, k, N, alpha, B.data(), N, beta, C_dense.data(), M);
        EXPECT_LT(frobenius_diff(C_kop, C_dense), 1e-12);
    }
    // adjoint: B is M × k, C accumulator nonzero
    {
        std::vector<T> B(M * k), C_kop(N * k), C_dense(N * k);
        fill_random(B, 81);
        fill_random(C_kop, 91);
        C_dense = C_kop;
        T alpha = 1.3, beta = 0.2;
        kop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
            N, k, M, alpha, B.data(), M, beta, C_kop.data(), N);
        dop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
            N, k, M, alpha, B.data(), M, beta, C_dense.data(), N);
        EXPECT_LT(frobenius_diff(C_kop, C_dense), 1e-12);
    }
}


// Test sparse-SASO sketch: A_hat = S * A.
// The SkOp overload should match what DenseLinOp produces on the materialized A.
TEST_F(TestKroneckerOperator, sparse_sketch_apply) {
    using T = double;
    using RNG = r123::Philox4x32;
    int64_t m1 = 6, n1 = 4, m2 = 5, n2 = 3;
    int64_t M = m1 * m2;  // 30
    int64_t N = n1 * n2;  // 12
    int64_t d = 8;
    int64_t saso_nnz = 3;

    std::vector<T> A1(m1 * n1), A2(m2 * n2), A_full(M * N);
    fill_random(A1, 101);
    fill_random(A2, 102);
    materialize_kron(m1, n1, m2, n2, A1.data(), A2.data(), A_full.data());

    KroneckerOperator<T> kop(m1, n1, m2, n2, A1.data(), A2.data());
    DenseLinOp<T> dop(M, N, A_full.data(), M, Layout::ColMajor);

    // Build a sparse SASO sketch and apply via both linops.
    RandBLAS::RNGState<RNG> state(42);
    RandBLAS::SparseDist DS(d, M, saso_nnz);
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    RandBLAS::fill_sparse(S);
    auto state2 = state;
    RandBLAS::SparseSkOp<T, RNG> S2(DS, state2);
    RandBLAS::fill_sparse(S2);

    std::vector<T> A_hat_kop(d * N, 0);
    std::vector<T> A_hat_dense(d * N, 0);

    kop(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, N, M, (T)1.0, S, (T)0.0, A_hat_kop.data(), d);
    dop(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, N, M, (T)1.0, S2, (T)0.0, A_hat_dense.data(), d);

    EXPECT_LT(frobenius_diff(A_hat_kop, A_hat_dense), 1e-12);
}
