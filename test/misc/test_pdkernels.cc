#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include "/Users/rjmurr/Documents/randnla/RandLAPACK/RandBLAS/test/comparison.hh"

#include <math.h>
#include <gtest/gtest.h>

using RandBLAS::RNGState;
using blas::Layout;
using std::vector;

template <typename T>
vector<T> random_gaussian_mat(int64_t m, int64_t n, uint32_t seed) {
    RandBLAS::DenseDist D(m, n);
    RNGState state(seed);
    vector<T> mat(m*n);
    RandBLAS::fill_dense(D, mat.data(), state);
    return mat;
}

template <typename T, typename RNG>
RNGState<RNG> left_multiply_by_orthmat(int64_t m, int64_t n, std::vector<T> &A, RNGState<RNG> state) {
    using std::vector;
    vector<T> U(m * m, 0.0);
    RandBLAS::DenseDist DU(m, m);
    auto out_state = RandBLAS::fill_dense(DU, U.data(), state).second;
    vector<T> tau(m, 0.0);
    lapack::geqrf(m, m, U.data(), m, tau.data());
    lapack::ormqr(blas::Side::Left, blas::Op::NoTrans, m, n, m, U.data(), m, tau.data(), A.data(), m);
    return out_state;
}

class TestPDK_SquaredExponential : public ::testing::Test {
    protected:
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    /**
     * Test that squared_exp_kernel_submatrix gives the same result
     * as calls to squared_exp_kernel.
     */
    template <typename T>
    void run_same_blockimpl_vs_entrywise(int64_t d, int64_t n, T bandwidth, uint32_t seed) {
        vector<T> K_blockimpl(n*n, 0.0);
        vector<T> K_entrywise(n*n, 0.0);
        vector<T> X = random_gaussian_mat<T>(d, n, seed);
        vector<T> squared_norms(n, 0.0);
        T* X_ = X.data();
        for (int64_t i = 0; i < n; ++i) {
            squared_norms[i] = std::pow(blas::nrm2(d, X_ + i*d, 1), 2);
        }
        RandLAPACK::squared_exp_kernel_submatrix(
            d, n, X_, squared_norms.data(), n, n, K_blockimpl.data(), 0, 0, bandwidth
        );
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                T* xi = X.data() + i*d;
                T* xj = X.data() + j*d;
                K_entrywise[i + j*n] = RandLAPACK::squared_exp_kernel(d, xi, xj, bandwidth);
            }
        }
        T atol = d * std::numeric_limits<T>::epsilon() * (1.0 + std::pow(bandwidth, -2));
        test::comparison::matrices_approx_equal(
            blas::Layout::ColMajor, blas::Op::NoTrans, n, n, K_blockimpl.data(), n,
            K_entrywise.data(), n, __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, atol
        );
        return;
    }

    /**
     * Test that if all of X's columns are the same then the squared exponential kernel
     * gives a matrix of all ones.
     */
    template <typename T>
    void run_all_same_column(int64_t d, int64_t n, uint32_t seed) {
        vector<T> c = random_gaussian_mat<T>(d, 1, seed);
        vector<T> X(d*n, 0.0);
        T* _X = X.data();
        T* _c = c.data();
        for (int64_t i = 0; i < n; ++i) {
            blas::copy(d, _c, 1, _X + i*d, 1);
        }
        T sqnorm = std::pow(blas::nrm2(d, _c, 1), 2);
        vector<T> squarednorms(n, sqnorm);
        vector<T> K(n*n, 0.0);
        T bandwidth = 2.3456;
        RandLAPACK::squared_exp_kernel_submatrix(
            d, n, _X, squarednorms.data(), n, n, K.data(), 0, 0, bandwidth
        );
        vector<T> expected(n*n, 1.0);
        test::comparison::matrices_approx_equal(
            blas::Layout::ColMajor, blas::Op::NoTrans, n, n, K.data(), n,
            expected.data(), n, __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;
    }

    /**
     * Test that if the columns of X are orthonormal then the diagonal
     * will be all ones and the off-diagonal will be exp(-bandwidth^{-2});
     * this needs to vary with different values for the bandwidth.
     */
    template <typename T>
    void run_orthogonal(int64_t n, T bandwidth, uint32_t seed) {
        std::vector<T> X(n*n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            X[i+i*n] = 1.0;
        RNGState state(seed);
        left_multiply_by_orthmat(n, n, X, state);
        vector<T> squarednorms(n, 1.0);
        vector<T> K(n*n, 0.0);
        RandLAPACK::squared_exp_kernel_submatrix(
            n, n, X.data(), squarednorms.data(), n, n, K.data(), 0, 0, bandwidth
        );
        T offdiag = std::exp(-std::pow(bandwidth, -2));
        std::vector<T> expect(n*n);
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                if (i == j) {
                    expect[i+j*n] = 1.0;
                } else {
                    expect[i+j*n] = offdiag;
                }
            }
        }
        T atol = 50 * std::numeric_limits<T>::epsilon();
        test::comparison::matrices_approx_equal(
            blas::Layout::ColMajor, blas::Op::NoTrans, n, n, K.data(), n,
            expect.data(), n,  __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, atol
        );
        return;
    }

};

TEST_F(TestPDK_SquaredExponential, test_repeated_columns) {
    for (uint32_t i = 10; i < 15; ++i) {
        run_all_same_column<float>(3, 9, i);
        run_all_same_column<float>(9, 3, i);
    }
}


TEST_F(TestPDK_SquaredExponential, test_blockimpl_vs_entrywise_full_matrix_d_3_n_10) {
    for (uint32_t i = 2; i < 7; ++i) {
        run_same_blockimpl_vs_entrywise<float>(3, 10, 1.0, i);
        run_same_blockimpl_vs_entrywise<float>(3, 10, 0.2, i);
        run_same_blockimpl_vs_entrywise<float>(3, 10, 5.9, i);
    }
}

TEST_F(TestPDK_SquaredExponential, test_blockimpl_vs_entrywise_full_matrix_d_10_n_3) {
    for (uint32_t i = 2; i < 7; ++i) {
        run_same_blockimpl_vs_entrywise<float>(10, 3, 1.0, i);
        run_same_blockimpl_vs_entrywise<float>(10, 3, 0.2, i);
        run_same_blockimpl_vs_entrywise<float>(10, 3, 5.9, i);
    }
}

TEST_F(TestPDK_SquaredExponential, test_orthogonal_columns) {
    for (uint32_t i = 70; i < 75; ++i) {
        run_orthogonal(5, 0.5, i);
        run_orthogonal(5, 1.1, i);
        run_orthogonal(5, 3.0, i);
    }
}
