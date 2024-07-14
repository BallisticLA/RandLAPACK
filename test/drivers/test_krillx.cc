#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>

#include "../moremats.hh"
#include "../RandLAPACK/RandBLAS/test/comparison.hh"


using std::vector;
using blas::Layout;
using blas::Op;
using RandBLAS::DenseDist;
using RandBLAS::SparseDist;
using RandBLAS::RNGState;
using RandLAPACK::linops::RegExplicitSymLinOp;
using RandLAPACK::linops::SEKLO;
using RandLAPACK_Testing::polynomial_decay_psd;


class TestKrillIsh: public ::testing::Test {

    protected:
        static inline int64_t m = 1000;
        static inline vector<uint32_t> keys = {42, 1};
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    void run_common(T mu_min, vector<T> &V, vector<T> &lambda, RegExplicitSymLinOp<T> &G_linop) {
        RandLAPACK::linops::SpectralPrecond<T> invP(m);
        vector<T> mus {mu_min, mu_min/10, mu_min/100};
        G_linop.regs = mus;
        G_linop.set_eval_includes_reg(true);
        invP.prep(V, lambda, mus, mus.size());
        int64_t s = mus.size();

        vector<T> X_star(m*s, 0.0);
        vector<T> X_init(m*s, 0.0);
        vector<T> H(m*s, 0.0);
        RNGState state0(101);
        DenseDist DX_star {m, s, RandBLAS::DenseDistName::Gaussian};
        auto Xsd = X_star.data();
        auto out1 = RandBLAS::fill_dense(DX_star, Xsd, state0);
        auto state1 = std::get<1>(out1);
        G_linop(blas::Layout::ColMajor, s, 1.0, X_star.data(), m, 0.0, H.data(), m);

        std::cout << "\nFrobenius norm of optimal solution : " << blas::nrm2(m*s, X_star.data(), 1);
        std::cout << "\nFrobenius norm of right-hand-side  : " << blas::nrm2(m*s, H.data(), 1) << std::endl;
        RandLAPACK::StatefulFrobeniusNorm<T> seminorm{};
        T tol = 100*std::numeric_limits<T>::epsilon();
        int64_t max_iters = 30;
        RandLAPACK::lockorblock_pcg(G_linop, H, tol, max_iters, invP, seminorm, X_init, true);

        T tol_scale = std::sqrt((T)m);
        T atol = tol_scale * std::pow(std::numeric_limits<T>::epsilon(), 0.5);
        T rtol = tol_scale * atol;
        test::comparison::buffs_approx_equal(X_init.data(), X_star.data(), m * s,
            __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
        );
        return;
    }

    template <typename T = double>
    void run_nystrom(int key_index, vector<T> &G) {
        /* Run the algorithm under test */
        RNGState alg_state(keys[key_index]);
        alg_state.key.incr();
        vector<T> V(0);
        vector<T> lambda(0);
        int64_t k = 64;
        T mu_min = 1e-5;
        vector<T> regs{};
        RegExplicitSymLinOp G_linop(m, G.data(), m, regs);
        RandLAPACK::nystrom_pc_data(
            G_linop, V, lambda, k, mu_min/10, alg_state
        ); // k has been updated.
        EXPECT_TRUE(k > 5);
        EXPECT_TRUE(k < m);
        run_common(mu_min, V, lambda, G_linop);
    }

    template <typename T = double>
    void run_rpchol(int key_index, vector<T> &G) {
        RNGState alg_state(keys[key_index]);
        alg_state.key.incr();
        int64_t k = 128;
        vector<T> V(m*k);
        vector<T> lambda(k);
        T mu_min = 1e-5;
        int64_t rp_chol_block_size = 4;
        vector<T> regs{};
        RegExplicitSymLinOp G_linop(m, G.data(), m, regs);
        RandLAPACK::rpchol_pc_data(m, G_linop, k, rp_chol_block_size, V.data(), lambda.data(), alg_state);
        EXPECT_TRUE(k == 128);
        run_common(mu_min, V, lambda, G_linop);
    }
};

TEST_F(TestKrillIsh, test_manual_lockstep_nystrom) {
    for (int64_t decay = 2; decay < 4; ++decay) {
        auto G = polynomial_decay_psd(m, 1e12, (double) decay, 99);
        run_nystrom(0, G);
        run_nystrom(1, G);
    }
}

TEST_F(TestKrillIsh, test_manual_lockstep_rpchol) {
    auto G = polynomial_decay_psd(m, 1e12, 2.0, 99);
    run_rpchol(0, G);
    run_rpchol(1, G);
}


class TestKrillx: public ::testing::Test {

    protected:
        static inline int64_t m = 1000;
        static inline vector<uint32_t> keys = {42, 1};
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename RELO>
    void run_krill_separable(int key_index, RELO &G_linop, int64_t k) {
        using T = typename RELO::scalar_t;
        int64_t s = G_linop.regs.size();

        vector<T> X_star(m*s, 0.0);
        vector<T> X_init(m*s, 0.0);
        vector<T> H(m*s, 0.0);
        RNGState state0(101);
        DenseDist DX_star {m, s, RandBLAS::DenseDistName::Gaussian};
        auto Xsd = X_star.data();
        auto out1 = RandBLAS::fill_dense(DX_star, Xsd, state0);
        auto state1 = std::get<1>(out1);
        G_linop.set_eval_includes_reg(true);
        G_linop(blas::Layout::ColMajor, s, 1.0, X_star.data(), m, 0.0, H.data(), m);
        std::cout << "\nFrobenius norm of optimal solution : " << blas::nrm2(m*s, X_star.data(), 1);
        std::cout << "\nFrobenius norm of right-hand-side  : " << blas::nrm2(m*s, H.data(), 1) << std::endl;

        RandLAPACK::StatefulFrobeniusNorm<T> seminorm{};
        T tol = 100*std::numeric_limits<T>::epsilon();
        int64_t max_iters = 30;
        int64_t rpc_blocksize = 16;
        RNGState state2(keys[key_index]);
        RandLAPACK::krill_full_rpchol(
            m, G_linop, H, X_init, tol, state2, seminorm, rpc_blocksize, max_iters, k
        );
        T tol_scale = std::sqrt((T)m);
        T atol = tol_scale * std::pow(std::numeric_limits<T>::epsilon(), 0.5);
        T rtol = tol_scale * atol;
        test::comparison::buffs_approx_equal(X_init.data(), X_star.data(), m * s,
            __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
        );
        return;
    }
};

TEST_F(TestKrillx, test_krill_full_rpchol) {
    using T = double;
    T mu_min = 1e-5;
    vector<T> mus {mu_min, mu_min/10, mu_min/100};
    for (int64_t decay = 2; decay < 4; ++decay) {
        auto G = polynomial_decay_psd(m, 1e12, (T) decay, 99);
        RegExplicitSymLinOp G_linop(m, G.data(), m, mus);
        int64_t k = 128;
        run_krill_separable(0, G_linop, k);
        run_krill_separable(1, G_linop, k);
    }
}

TEST_F(TestKrillx, test_krill_separable_squared_exp_kernel) {
    using T = double;
    T mu_min = 1e-2;
    vector<T> mus {mu_min, mu_min*10, mu_min*100};
    for (uint32_t key = 0; key < 5; ++key) {
        //auto G = polynomial_decay_psd(m, 1e12, (T) decay, key);
        //RegExplicitSymLinOp G_linop(m, G.data(), m, mus);
        vector<T> X0 = RandLAPACK_Testing::random_gaussian_mat<T>(5, m, key);
        SEKLO G_linop(m, X0.data(), 5, 3.0, mus);
        int64_t k = 128;
        run_krill_separable(0, G_linop, k);
        run_krill_separable(1, G_linop, k);
    }
}
