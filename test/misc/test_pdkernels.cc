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

class TestPDK_SquaredExponential : public ::testing::Test {
    protected:
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T = float>
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
        T sqnormtol = d * std::numeric_limits<T>::epsilon() / std::min(1.0, std::pow(bandwidth, 2));
        T atol = sqnormtol;
        std::cout << "atol : " << atol;
        test::comparison::matrices_approx_equal(
            blas::Layout::ColMajor, blas::Op::NoTrans, n, n, K_blockimpl.data(), n,
            K_entrywise.data(), n, __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, atol
        );
        return;
    }

    template <typename T>
    void run_all_same_column(int64_t d, int64_t n, uint32_t seed) {
        vector<T> x(d, 0.0);
        RandBLAS::util::genmat(d, 1, x.data(), seed);
        vector<T> X(d*n, 0.0);
        T* _X = X.data();
        T* _x = x.data();
        for (int64_t i = 0; i < n; ++i) {
            std::fill(_X + i*d, _X + (i+1)*d, _x);
        }
        return;
    }

};

/**
 * Test that if all of X's columns are the same then the squared exponential kernel
 * gives a matrix of all ones.
 */


/**
 * Test that squared_exp_kernel_submatrix gives the same result
 * as calls to squared_exp_kernel.
 */

TEST_F(TestPDK_SquaredExponential, test_blockimpl_vs_entrywise_full_matrix_d_3_n_10) {
    for (uint32_t i = 2; i < 7; ++i) {
        run_same_blockimpl_vs_entrywise(3, 10, 1.0, i);
        run_same_blockimpl_vs_entrywise(3, 10, 0.2, i);
        run_same_blockimpl_vs_entrywise(3, 10, 5.9, i);
    }
}

TEST_F(TestPDK_SquaredExponential, test_blockimpl_vs_entrywise_full_matrix_d_10_n_3) {
    for (uint32_t i = 2; i < 7; ++i) {
        run_same_blockimpl_vs_entrywise(10, 3, 1.0, i);
        run_same_blockimpl_vs_entrywise(10, 3, 0.2, i);
        run_same_blockimpl_vs_entrywise(10, 3, 5.9, i);
    }
}

/**
 * Test that if the columns of X are orthonormal then the diagonal
 * will be all ones and the off-diagonal will be exp(-bandwidth^{-2});
 * this needs to vary with different values for the bandwidth.
 */

