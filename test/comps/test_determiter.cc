#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "/Users/rjmurr/Documents/randnla/RandLAPACK/RandBLAS/test/comparison.hh"

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
        RandBLAS::util::genmat(m, n, A.data(), keys[key_index]);
        
        std::vector<double> b(m);
        RandBLAS::util::genmat(m, 1, b.data(), keys[key_index] + (uint64_t) 1);
        
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

        RandLAPACK::pcg_saddle(
            m, n, A.data(), m, b.data(), c.data(), delta,
            resid_vec, tol, n, M.data(), n, x0.data(), x.data(), y.data()
        );
        

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


class TestDetermiterLockBlockPCG : public ::testing::Test {
    protected:
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    void run_simple_block(int64_t m, int64_t s, T coeff, uint32_t seed) {   
        using std::vector;
        auto layout = blas::Layout::ColMajor;
        vector<T> G_buff(m*m, 0.0);
        vector<T> H(m*s, 0.0);
        randblas_require((int64_t) H.size() == m*s);
        vector<T> X_star(m*s, 0.0);
        vector<T> X_init(m*s, 0.0);
        RandBLAS::RNGState state0(seed);
        vector<T> temp(2*m*m);
        auto D = RandBLAS::DenseDist {2*m, m, RandBLAS::DenseDistName::Gaussian};
        auto out1 = RandBLAS::fill_dense(D, temp.data(), state0);
        auto state1 = std::get<1>(out1);
        blas::syrk(layout, blas::Uplo::Upper, blas::Op::Trans, m, 2*m, 1.0, temp.data(), 2*m, 0.0, G_buff.data(), m);

        vector<T> regs(1, coeff);
        RandLAPACK::linops::RegExplicitSymLinOp G(m, G_buff.data(), m, regs);
        RandBLAS::DenseDist DX_star {m, s, RandBLAS::DenseDistName::Gaussian};
        auto Xsd = X_star.data();
        auto out2 = RandBLAS::fill_dense(DX_star, Xsd, state1);
        auto state2 = std::get<1>(out2);
        G(layout, s, 1.0, X_star.data(), m, 0.0, H.data(), m);

        RandLAPACK::StatefulFrobeniusNorm<T> seminorm{};

        auto I_buff = eye<T>(m);
        vector<T> zeros(1, 0.0);
        RandLAPACK::linops::RegExplicitSymLinOp I(m, I_buff.data(), m, zeros);

        T tol = 100*std::numeric_limits<T>::epsilon();
        RandLAPACK::lockorblock_pcg(G, H, tol, m, I, seminorm, X_init, true);

        T tol_scale = std::sqrt((T)m);
        T atol = tol_scale * std::pow(std::numeric_limits<T>::epsilon(), 0.5);
        T rtol = tol_scale * atol;
        test::comparison::buffs_approx_equal(X_init.data(), X_star.data(), m * s,
            __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
        );
        return;
    }

    virtual void run_simple_lockstep(int64_t m, int64_t s, uint32_t seed) {  
        using T = double;
        randblas_require(s <= 4);
        using std::vector;
        vector<T> reg_coeffs{};
        reg_coeffs.push_back(100);
        if (s > 1)
            reg_coeffs.push_back(7);
        if (s > 2)
            reg_coeffs.push_back(0.1);
        if (s > 3)
            reg_coeffs.push_back(0.5483);
        auto layout = blas::Layout::ColMajor;
        vector<T> G_buff(m*m, 0.0);
        vector<T> H(m*s, 0.0);
        vector<T> X_star(m*s, 0.0);
        vector<T> X_init(m*s, 0.0);
        RandBLAS::RNGState state0(seed);
        vector<T> temp(2*m*m);
        
        auto D = RandBLAS::DenseDist {2*m, m, RandBLAS::DenseDistName::Gaussian};
        auto out1 = RandBLAS::fill_dense(D, temp.data(), state0);
        auto state1 = std::get<1>(out1);
        blas::syrk(layout, blas::Uplo::Upper, blas::Op::Trans, m, 2*m, 1.0, temp.data(), 2*m, 0.0, G_buff.data(), m);

        vector<T> regs(reg_coeffs);
        RandLAPACK::linops::RegExplicitSymLinOp G(m, G_buff.data(), m, regs);
        RandBLAS::DenseDist DX_star {m, s, RandBLAS::DenseDistName::Gaussian};
        auto Xsd = X_star.data();
        auto out2 = RandBLAS::fill_dense(DX_star, Xsd, state1);
        auto state2 = std::get<1>(out2);
        G(layout, s, 1.0, X_star.data(), m, 0.0, H.data(), m);

        RandLAPACK::StatefulFrobeniusNorm<T> seminorm{};

        auto I_buff = eye<T>(m);
        vector<T> zeros(s, 0.0);
        RandLAPACK::linops::RegExplicitSymLinOp I(m, I_buff.data(), m, zeros);

        T tol = 100*std::numeric_limits<T>::epsilon();
        RandLAPACK::lockorblock_pcg(G, H, tol, m, I, seminorm, X_init, true);

        T tol_scale = std::sqrt((T)m);
        T atol = tol_scale * std::pow(std::numeric_limits<T>::epsilon(), 0.5);
        T rtol = tol_scale * atol;
        test::comparison::buffs_approx_equal(X_init.data(), X_star.data(), m * s,
            __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
        );
        return;
    }
};


TEST_F(TestDetermiterLockBlockPCG, test_run_simple_block_5_1) {
    run_simple_block<double>(5, 1, 0.5, 1997);
}

TEST_F(TestDetermiterLockBlockPCG, test_run_simple_block_6_2) {
    run_simple_block<double>(6, 2, 0.5, 1997);
}

TEST_F(TestDetermiterLockBlockPCG, test_run_simple_block_5_4) {
    run_simple_block<double>(5, 4, 0.5, 1997);
}

TEST_F(TestDetermiterLockBlockPCG, test_run_simple_lockstep_5_1) {
    run_simple_lockstep(5, 1, 1997);
    run_simple_lockstep(5, 1, 2024);
}

TEST_F(TestDetermiterLockBlockPCG, test_run_simple_lockstep_6_2) {
    run_simple_lockstep(6, 2, 1997);
    run_simple_lockstep(6, 2, 2024);
}

TEST_F(TestDetermiterLockBlockPCG, test_run_simple_lockstep_5_4) {
    run_simple_lockstep(5, 4, 1997);
    run_simple_lockstep(5, 4, 2024);
}
