#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include "../RandLAPACK/RandBLAS/test/comparison.hh"
#include "../moremats.hh"

#include <math.h>
#include <gtest/gtest.h>

using RandBLAS::RNGState;
using RandBLAS::DenseDist;
using blas::Layout;
using std::vector;

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
        vector<T> X = RandLAPACK_Testing::random_gaussian_mat<T>(d, n, seed);
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
        T atol = 3 * d * std::numeric_limits<T>::epsilon() * (1.0 + std::pow(bandwidth, -2));
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
        vector<T> c = RandLAPACK_Testing::random_gaussian_mat<T>(d, 1, seed);
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
        RandLAPACK_Testing::left_multiply_by_orthmat(n, n, X, state);
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


class TestPDK_RBFKernelMatrix : public ::testing::Test {
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    void run(T bandwidth, T reg, int64_t m, int64_t d, uint32_t seed, bool use_reg = true) {
        RNGState state_x(seed);
        DenseDist D(d, m);
        vector<T> X_vec(d*m);
        T* X = X_vec.data();
        RandBLAS::fill_dense(D, X, state_x);
        vector<T> regs(1,reg); 
        RandLAPACK::linops::RBFKernelMatrix K(m, X, d, bandwidth, regs);
        K.set_eval_includes_reg(use_reg);

        vector<T> eye(m * m, 0.0);
        vector<T> sq_colnorms(m, 0.0);
        for (int64_t i = 0; i < m; ++i) {
            eye[i + m*i] = 1.0;
            sq_colnorms[i] = std::pow(blas::nrm2(d, X + i*d, 1), 2);
        }
        vector<T> K_out_expect(m * m, 0.0);

        // (alpha, beta) = (0.25, 0.0), 
        T alpha = 0.25;
        RandLAPACK::squared_exp_kernel_submatrix(
            d, m, X, sq_colnorms.data(), m, m, K_out_expect.data(), 0, 0, bandwidth
        );
        blas::scal(m * m, alpha, K_out_expect.data(), 1);
        if (use_reg) {
            for (int i = 0; i < m; ++i)
                K_out_expect[i + i*m] += alpha * reg;
        }
        vector<T> K_out_actual1(m * m, 1.0);
        K(blas::Layout::ColMajor, m, alpha, eye.data(), m, 0.0, K_out_actual1.data(), m);

        T atol = d * std::numeric_limits<T>::epsilon() * (1.0 + std::pow(bandwidth, -2));
        test::comparison::matrices_approx_equal(
            blas::Layout::ColMajor, blas::Op::NoTrans, m, m, K_out_actual1.data(), m, 
            K_out_expect.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, atol
        );
        
        // Expected output when (alpha, beta) = (0.25, 0.3)
        T beta = 0.3;
        for (int i = 0; i < m*m; ++i)
            K_out_expect[i] += beta;
        vector<T> K_out_actual2(m * m, 1.0);
        K(blas::Layout::ColMajor, m, alpha, eye.data(), m, beta, K_out_actual2.data(), m);

        test::comparison::matrices_approx_equal(
            blas::Layout::ColMajor, blas::Op::NoTrans, m, m, K_out_actual2.data(), m, 
            K_out_expect.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, atol
        );
        return;
    }

};

TEST_F(TestPDK_RBFKernelMatrix, apply_to_eye_m100_d3) {
    double mu = 0.123;
    for (uint32_t i = 77; i < 80; ++i) {
        run(1.0,      mu, 100, 3, i, false);
        run(2.0,      mu, 100, 3, i, false);
        run(2.345678, mu, 100, 3, i, false);
    }
}

TEST_F(TestPDK_RBFKernelMatrix, apply_to_eye_m256_d4) {
    double mu = 0.123;
    for (uint32_t i = 77; i < 80; ++i) {
        run(1.0,      mu, 256, 4, i, false);
        run(2.0,      mu, 256, 4, i, false);
        run(2.345678, mu, 256, 4, i, false);
    }
}

TEST_F(TestPDK_RBFKernelMatrix, apply_to_eye_m999_d7) {
    double mu = 0.123;
    for (uint32_t i = 77; i < 80; ++i) {
        run(1.0,      mu, 999, 7, i, false);
        run(2.0,      mu, 999, 7, i, false);
        run(2.345678, mu, 999, 7, i, false);
    }
}

TEST_F(TestPDK_RBFKernelMatrix, reg_apply_to_eye_m100_d3) {
    double bandwidth = 1.1;
    for (uint32_t i = 77; i < 80; ++i) {
        run(bandwidth, 0.1,      100, 3, i);
        run(bandwidth, 1.0,      100, 3, i);
        run(bandwidth, 7.654321, 100, 3, i);
    }
}

TEST_F(TestPDK_RBFKernelMatrix, reg_apply_to_eye_m256_d4) {
    double bandwidth = 1.1;
    for (uint32_t i = 77; i < 80; ++i) {
        run(bandwidth, 0.1,      256, 4, i);
        run(bandwidth, 1.0,      256, 4, i);
        run(bandwidth, 7.654321, 256, 4, i);
    }
}

TEST_F(TestPDK_RBFKernelMatrix, reg_apply_to_eye_m257_d5) {
    double bandwidth = 1.1;
    for (uint32_t i = 77; i < 80; ++i) {
        run(bandwidth, 0.1,      257, 5, i);
        run(bandwidth, 1.0,      257, 5, i);
        run(bandwidth, 7.654321, 257, 5, i);
    }
}
