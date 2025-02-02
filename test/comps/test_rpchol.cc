#include "RandLAPACK.hh"
#include "rl_rpchol.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"
#include "../RandLAPACK/RandBLAS/test/comparison.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <gtest/gtest.h>


using RandBLAS::RNGState;

template <typename T, typename RNG>
RNGState<RNG> left_multiply_by_orthmat(int64_t m, int64_t n, std::vector<T> &A, RNGState<RNG> state) {
    using std::vector;
    vector<T> U(m * m, 0.0);
    RandBLAS::DenseDist DU(m, m);
    auto out_state = RandBLAS::fill_dense(DU, U.data(), state);
    vector<T> tau(m, 0.0);
    lapack::geqrf(m, m, U.data(), m, tau.data());
    lapack::ormqr(blas::Side::Left, blas::Op::NoTrans, m, n, m, U.data(), m, tau.data(), A.data(), m);
    return out_state;
}

template <typename T>
void full_gram(int64_t n, std::vector<T> &A, blas::Op op, int64_t k = -1) {
    std::vector<T> work(A);
    auto uplo   = blas::Uplo::Upper;
    auto layout = blas::Layout::ColMajor;
    if (k == -1) {
        k = n;
    } else {
        randblas_require(op == blas::Op::NoTrans);
    }
    blas::syrk(layout, uplo, op, n, k, 1.0, work.data(), n, 0.0, A.data(), n); 
    RandBLAS::symmetrize(layout, uplo, n, A.data(), n);
}

class TestRPCholesky : public ::testing::Test {
    protected:
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T, typename FUNC>
    void run_exact(int64_t n, FUNC &A, T* Abuff, int64_t b, T atol, T rtol, uint32_t seed) {
        using std::vector;

        int64_t k = n;
        vector<T> F(n*k, 0.0);
        vector<int64_t> selection(k, -1);
        RandBLAS::RNGState state_in(seed);
        auto state_out = RandLAPACK::rp_cholesky(n, A, k, selection.data(), F.data(), b, state_in);

        vector<T> Arecovered(F);
        full_gram(n, Arecovered, blas::Op::NoTrans, k);
        test::comparison::matrices_approx_equal(
            blas::Layout::ColMajor, blas::Op::NoTrans, n, n, Abuff, n, Arecovered.data(), n, __PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        );
        // Check that the pivots are reasonable and nontrivial (i.e., not the sequence from 0 to n-1).
        std::set<int64_t> selection_unique{};
        for (auto pivot : selection) {
            if (pivot != -1)
                selection_unique.insert(pivot);
        }
        ASSERT_EQ(selection_unique.size(), k) << "using seed " << seed;
        if (n > 4)
            ASSERT_FALSE(std::is_sorted(selection.begin(), selection.end())) <<  "using seed " << seed;
        // ^ is_sorted() checks if we're in increasing order
        return;
    }

    template <typename T>
    void run_exact_diag(int64_t n, int64_t b, int64_t power, uint32_t seed) {  
        std::vector<T> Avec(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            Avec[i + n*i] = std::pow((T) i + 1, power);
        auto Abuff = Avec.data();
        auto A = [Abuff, n](int64_t i, int64_t j) { return Abuff[i + n*j]; };

        T atol = std::sqrt(n) * std::numeric_limits<T>::epsilon();
        T rtol = std::sqrt(n) * std::numeric_limits<T>::epsilon();
        run_exact(n, A, Abuff, b, atol, rtol, seed);
        return;
    }

    template <typename T>
    void run_exact_kahan_gram(int64_t n, int64_t b, uint32_t seed) {
        using std::vector;
        vector<T> Avec(n * n, 0.0);
        T theta = 1.2;
        T perturb = 10;
        RandLAPACK::gen::gen_kahan_mat(n, n, Avec.data(), theta, perturb);
        vector<T> kahan(Avec);
        full_gram(n, Avec, blas::Op::Trans);
        // ^ Avec now represents the Gram matrix of the Kahan matrix.

        std::vector<T> gk_chol(Avec); 
        // ^ We'll run Cholesky on the Gram matrix of the Kahan matrix,
        //   and compare to the Kahan matrix itself. This helps us get
        //   a realistic tolerance considering the numerical nastyness
        //   of the Kahan matrix.
        auto status = lapack::potrf(blas::Uplo::Upper, n, gk_chol.data(), n);
        randblas_require(status == 0);
        T atol = 0.0;
        RandLAPACK::util::get_U(n, n, gk_chol.data(), n);
        for (int64_t i = 0; i < n*n; ++i) {
            T val1 = std::abs(kahan[i] - gk_chol[i]);
            T val2 = std::abs(kahan[i] + gk_chol[i]);
            atol = std::max(atol, std::min(val1, val2));
        }
        atol = std::sqrt(n) * atol;

        T* Abuff = Avec.data();
        auto A = [Abuff, n](int64_t i, int64_t j) { return Abuff[i + n*j]; };
        run_exact(n, A, Abuff, b, atol, atol, seed);
        // ^ use the same value for rtol and atol
        return;
    }
};


TEST_F(TestRPCholesky, test_exact_diag_b1) {
    using T = float;
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_diag<T>(5,   1, 2, i);
        run_exact_diag<T>(10,  1, 1, i);
        run_exact_diag<T>(10,  1, 2, i);
        run_exact_diag<T>(13,  1, 2, i);
        run_exact_diag<T>(100, 1, 2, i);
    }
}

TEST_F(TestRPCholesky, test_exact_diag_b2) {
    using T = float;
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_diag<T>(10,  2, 1, i);
        run_exact_diag<T>(10,  2, 2, i);
        run_exact_diag<T>(100, 2, 2, i);
    }
}

TEST_F(TestRPCholesky, test_exact_kahan_gram_b1) {
    using T = float;
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_kahan_gram<T>(5,  1, i);
        run_exact_kahan_gram<T>(10, 1, i);
    }
}

TEST_F(TestRPCholesky, test_exact_kahan_gram_b2) {
    using T = float;
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_kahan_gram<T>(10,  2, i);
        run_exact_kahan_gram<T>(11,  2, i);
        run_exact_kahan_gram<T>(12,  2, i);
    }
}

TEST_F(TestRPCholesky, test_exact_kahan_gram_b3) {
    using T = float;
    for (uint32_t i = 2012; i < 2019; ++i) {
        // run_exact_kahan_gram<T>(9,   3, i);
        run_exact_kahan_gram<T>(10,  3, i);
        // run_exact_kahan_gram<T>(11,  3, i);
        // run_exact_kahan_gram<T>(12,  3, i);
    }
}
