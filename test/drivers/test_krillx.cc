#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "/Users/rjmurr/Documents/randnla/RandLAPACK/RandBLAS/test/comparison.hh"


using std::vector;
using blas::Layout;
using blas::Op;
using RandBLAS::DenseDist;
using RandBLAS::SparseDist;
using RandBLAS::RNGState;


template <typename T>
vector<T> polynomial_decay_psd(int64_t m, T cond_num, T exponent, uint32_t seed) {
    RandLAPACK::gen::mat_gen_info<T> mat_info(m, m, RandLAPACK::gen::polynomial);
    mat_info.cond_num = std::sqrt(cond_num);
    mat_info.rank = m;
    mat_info.exponent = exponent;
    vector<T> A(m * m, 0.0);
    RNGState data_state(seed);
    RandLAPACK::gen::mat_gen(mat_info, A.data(), data_state);
    vector<T> G(m * m, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::NoTrans, m, m, 1.0,
        A.data(), m, 0.0, G.data(), m
    ); // Note: G is PSD with squared spectrum of A.
    RandBLAS::util::symmetrize(Layout::ColMajor, Uplo::Upper, G.data(), m, m);
    return G;
}


/***
 * This actually assesses quality of the Nystrom preconditioner.
 */
class TestKrillIsh: public ::testing::Test {

    protected:
        static inline int64_t m = 1000;
        static inline vector<uint32_t> keys = {42, 1};
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    void run_common(T mu_min, vector<T> &V, vector<T> &lambda, vector<T> &G) {
        RandLAPACK::OOPreconditioners::SpectralPrecond<T> invP(m);
        vector<T> mus {mu_min, mu_min/10, mu_min/100};
        invP.prep(V, lambda, mus, mus.size());
        int64_t s = mus.size();

        vector<T> X_star(m*s, 0.0);
        vector<T> X_init(m*s, 0.0);
        vector<T> H(m*s, 0.0);
        RNGState state0(101);
        RandLAPACK::RegExplicitSymLinOp G_linop(m, G.data(), m, mus);
        DenseDist DX_star {m, s, RandBLAS::DenseDistName::Gaussian};
        auto Xsd = X_star.data();
        auto out1 = RandBLAS::fill_dense(DX_star, Xsd, state0);
        auto state1 = std::get<1>(out1);
        G_linop(blas::Layout::ColMajor, s, 1.0, X_star.data(), m, 0.0, H.data(), m);

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
        RandLAPACK::nystrom_pc_data(
            Uplo::Lower, G.data(), m, V, lambda, k, mu_min, alg_state
        ); // k has been updated.
        EXPECT_TRUE(k > 5);
        EXPECT_TRUE(k < m);
        run_common(mu_min, V, lambda, G);
    }

    template <typename T = double>
    void run_rpchol(int key_index, vector<T> &G) {
        RNGState alg_state(keys[key_index]);
        alg_state.key.incr();
        int64_t k = 128;
        vector<T> V(m*k);
        vector<T> lambda(k);
        T mu_min = 1e-5;
        T* Gd = G.data();
        auto G_ij_callable = [Gd](int64_t i, int64_t j) {return Gd[i + m*j]; };
        int64_t rp_chol_block_size = 4;
        RandLAPACK::rpchol_pc_data(m, G_ij_callable, k, rp_chol_block_size, V.data(), lambda.data(), alg_state);
        EXPECT_TRUE(k == 128);
        run_common(mu_min, V, lambda, G);
    }
};

TEST_F(TestKrillIsh, test_separable_lockstep_nystrom) {
    auto G = polynomial_decay_psd<double>(m, 1e12, 2.0, 99);
    run_nystrom(0, G);
    run_nystrom(1, G);
}

TEST_F(TestKrillIsh, test_separable_lockstep_rpchol) {
    auto G = polynomial_decay_psd<double>(m, 1e12, 2.0, 99);
    run_rpchol(0, G);
    run_rpchol(1, G);
}
