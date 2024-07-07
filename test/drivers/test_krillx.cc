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

    template <typename T = double>
    void run(int key_index, vector<T> &G) {
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

        /* Verify algorithm output */
        EXPECT_TRUE(k > 5);
        EXPECT_TRUE(k < m);
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
    }
};

TEST_F(TestKrillIsh, test_separable_lockstep) {
    auto G = polynomial_decay_psd<double>(m, 1e12, 2.0, 99);
    run(0, G);
    run(1, G);
}
