#include "RandLAPACK.hh"
#include "rl_rpchol.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"
#include "/Users/rjmurr/Documents/randnla/RandLAPACK/RandBLAS/test/comparison.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <gtest/gtest.h>

// template <typename T>
// std::vector<T> eye(int64_t n) {
//     std::vector<T> A(n * n, 0.0);
//     for (int i = 0; i < n; ++i)
//         A[i + n*i] = 1.0;
//     return A;
// }

using RandBLAS::RNGState;

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
    RandBLAS::util::symmetrize(layout, uplo, A.data(), n, n);
}

class TestRPCholesky : public ::testing::Test {
    protected:
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T, typename FUNC>
    void run_exact(int64_t N, FUNC &A, T* Abuff, int64_t B, T atol, T rtol, uint32_t seed) {
        using std::vector;

        int64_t k = N;
        vector<T> F(N*k, 0.0);
        vector<int64_t> selection(k, -1);
        RandBLAS::RNGState state_in(seed);
        auto state_out = RandLAPACK::rp_cholesky(N, A, k, selection.data(), F.data(), B, state_in);

        vector<T> Arecovered(F);
        full_gram(N, Arecovered, blas::Op::NoTrans, k);
        test::comparison::matrices_approx_equal(
            blas::Layout::ColMajor, blas::Op::NoTrans, N, N, Abuff, N, Arecovered.data(), N, __PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        );
        // Check that the pivots are reasonable and nontrivial (i.e., not the sequence from 0 to N-1).
        std::set<int64_t> selection_unique{};
        for (auto pivot : selection) {
            if (pivot != -1)
                selection_unique.insert(pivot);
        }
        ASSERT_EQ(selection_unique.size(), k) << "using seed " << seed;
        if (N > 4)
            ASSERT_FALSE(std::is_sorted(selection.begin(), selection.end())) <<  "using seed " << seed;
        // ^ is_sorted() checks if we're in increasing order
        return;
    }

    template <typename T>
    void run_exact_diag(int64_t N, int64_t B, int64_t power, uint32_t seed) {  
        std::vector<T> Avec(N * N, 0.0);
        for (int64_t i = 0; i < N; ++i)
            Avec[i + N*i] = std::pow((T) i + 1, power);
        auto Abuff = Avec.data();
        auto A = [Abuff, N](int64_t i, int64_t j) { return Abuff[i + N*j]; };

        T atol = std::sqrt(N) * std::numeric_limits<T>::epsilon();
        T rtol = std::sqrt(N) * std::numeric_limits<T>::epsilon();
        run_exact(N, A, Abuff, B, atol, rtol, seed);
        return;
    }

    template <typename T>
    void run_exact_kahan_gram(int64_t N, int64_t B, uint32_t seed) {
        using std::vector;
        vector<T> Avec(N * N, 0.0);
        T theta = 1.2;
        T perturb = 10;
        RandLAPACK::gen::gen_kahan_mat(N, N, Avec.data(), theta, perturb);
        vector<T> kahan(Avec);
        full_gram(N, Avec, blas::Op::Trans);
        // ^ Avec now represents the Gram matrix of the Kahan matrix.

        std::vector<T> gk_chol(Avec); 
        // ^ We'll run Cholesky on the Gram matrix of the Kahan matrix,
        //   and compare to the Kahan matrix itself. This helps us get
        //   a realistic tolerance considering the numerical nastyness
        //   of the Kahan matrix.
        auto status = lapack::potrf(blas::Uplo::Upper, N, gk_chol.data(), N);
        randblas_require(status == 0);
        T atol = 0.0;
        RandLAPACK::util::get_U(N, N, gk_chol.data(), N);
        for (int64_t i = 0; i < N*N; ++i) {
            T val1 = std::abs(kahan[i] - gk_chol[i]);
            T val2 = std::abs(kahan[i] + gk_chol[i]);
            atol = std::max(atol, std::min(val1, val2));
        }
        atol = std::sqrt(N) * atol;

        T* Abuff = Avec.data();
        auto A = [Abuff, N](int64_t i, int64_t j) { return Abuff[i + N*j]; };
        run_exact(N, A, Abuff, B, atol, atol, seed);
        // ^ use the same value for rtol and atol
        return;
    }
};


TEST_F(TestRPCholesky, test_exact_diag_B1) {
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_diag<float>(5,   1, 2, i);
        run_exact_diag<float>(10,  1, 1, i);
        run_exact_diag<float>(10,  1, 2, i);
        run_exact_diag<float>(13,  1, 2, i);
        run_exact_diag<float>(100, 1, 2, i);
    }
}

TEST_F(TestRPCholesky, test_exact_diag_B2) {
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_diag<float>(10,  2, 1, i);
        run_exact_diag<float>(10,  2, 2, i);
        run_exact_diag<float>(100, 2, 2, i);
    }
}

TEST_F(TestRPCholesky, test_exact_kahan_gram_B1) {
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_kahan_gram<float>(5,  1, i);
        run_exact_kahan_gram<float>(10, 1, i);
    }
}

TEST_F(TestRPCholesky, test_exact_kahan_gram_B2) {
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_kahan_gram<float>(10,  2, i);
        run_exact_kahan_gram<float>(11,  2, i);
        run_exact_kahan_gram<float>(12,  2, i);
    }
}

TEST_F(TestRPCholesky, test_exact_kahan_gram_B3) {
    for (uint32_t i = 2012; i < 2019; ++i) {
        run_exact_kahan_gram<float>(9,   3, i);
        run_exact_kahan_gram<float>(10,  3, i);
        run_exact_kahan_gram<float>(11,  3, i);
        run_exact_kahan_gram<float>(12,  3, i);
    }
}
