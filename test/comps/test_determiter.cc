#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "../RandLAPACK/RandBLAS/test/comparison.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <gtest/gtest.h>

template <typename T>
std::vector<T> eye(int64_t n) {
    std::vector<T> A(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        A[i + n*i] = 1.0;
    return A;
}


class TestDetermiterOLS : public ::testing::Test {
    protected:
        int64_t m = 201;
        int64_t n = 12;
        std::vector<uint64_t> keys = {42, 0, 1};
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    virtual void run(uint64_t key_index) {   
        std::vector<double> A(m * n);
        RandBLAS::RNGState state0(keys[key_index]);
        auto state1 = RandBLAS::fill_dense({m, n}, A.data(), state0);
        
        std::vector<double> b(m);
        RandBLAS::fill_dense({m, 1}, b.data(), state1);
        
        std::vector<double> c(n, 0.0);
        std::vector<double> x0(n, 0.0);
        std::vector<double> x(n, 0.0);
        std::vector<double> y(m, 0.0);
        std::vector<double> resid_vec(10*n, -1.0);

        std::vector<double> M(n*n, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            M[i + n*i] = 1.0;
        }

        double delta = 0.1;
	    double tol = 1e-8;

        RandLAPACK::pcg(
            m, n, A.data(), m, b.data(), c.data(), delta,
            resid_vec, tol, n, M.data(), n, x0.data(), x.data(), y.data());
        

        int64_t iter_count = 0;
        for (double res: resid_vec) {
            if (res < 0) {
                break;
            }
            else {
                iter_count += 1;
                std::cout << res << std::endl;
            }
        }
        ASSERT_LE(iter_count, 2*n);
        ASSERT_GE(iter_count, 2);
    }
};

TEST_F(TestDetermiterOLS, Trivial) {
    for (int64_t k_idx : {0, 1, 2}) {
        run(k_idx);
    }
}
