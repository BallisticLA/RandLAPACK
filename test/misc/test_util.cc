#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <RandBLAS/testing/comparison.hh>

#include <math.h>
#include <chrono>
#include <random>
#include <gtest/gtest.h>
/*
TODO #1: Resizing tests.

TODO #2: Diagonalization tests.

TODO #4: L & pivotig tests.
*/
using namespace std::chrono;


class TestUtil : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct SpectralTestData {
        int64_t row;
        int64_t col;
        std::vector<T> A;
        std::vector<T> A_cpy;
        std::vector<T> s;

        SpectralTestData(int64_t m, int64_t n) :
        A(m * n, 0.0),
        A_cpy(m * n, 0.0), 
        s(n, 0.0)
        {
            row = m;
            col = n;
        }
    };

    template <typename T>
    struct NormcTestData {
        int64_t row;
        int64_t col;
        std::vector<T> A;
        std::vector<T> A_norm;

        NormcTestData(int64_t m, int64_t n) :
        A(m * n, 0.0), 
        A_norm(m * n, 0.0) 
        {
            row = m;
            col = n;
        }
    };

    template <typename T>
    struct ColSwpTestData {
        int64_t row;
        int64_t col;
        std::vector<T> A;
        std::vector<T> A_cpy;
        std::vector<T> Ident;
        std::vector<T> tau;
        std::vector<int64_t> J;

        ColSwpTestData(int64_t m, int64_t n) :
        A(m * n, 0.0),
        A_cpy(m * n, 0.0),
        Ident(m * n, 0.0),
        tau(n, 0.0),
        J(n, 0.0)
        {
            row = m;
            col = n;
        }
    };

    template <typename T>
    struct OrhrColTestData {
        int64_t row;
        int64_t col;
        std::vector<T> A;
        std::vector<T> B;
        std::vector<T> B1;
        std::vector<T> A1;
        std::vector<T> R;
        std::vector<T> T_mat;
        std::vector<T> D;
        std::vector<T> D1;
        std::vector<T> tau;
        std::vector<T> tau1;

        OrhrColTestData(int64_t m, int64_t n) :
        A(m * n, 0.0),
        B(m * n, 0.0),
        B1(m * n, 0.0),
        A1(m * n, 0.0),
        R(n * n, 0.0),
        T_mat(n * n, 0.0),
        D(n, 0.0),
        D1(n, 0.0),
        tau(n, 0.0),
        tau1(n, 0.0)
        {
            row = m;
            col = n;
        }
    };

    template <typename T, typename RNG>
    static void 
    test_spectral_norm(RandBLAS::RNGState<RNG> state, SpectralTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;

        T norm = RandLAPACK::util::estimate_spectral_norm(m, n, all_data.A.data(), 10000, state);
        // Get an SVD -> first singular value == 2_norm
        lapack::gesdd(Job::NoVec, m, n, all_data.A_cpy.data(), m, all_data.s.data(), NULL, m, NULL, n);

        std::cout << "Computed norm:  " << std::scientific << norm << "\nComputed s_max: " << std::scientific << all_data.s[0] << "\n";
        ASSERT_NEAR(norm, all_data.s[0], std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_normc(NormcTestData<T> &all_data) {
        
        auto m = all_data.row;

        int cnt = 0;
        // we have a vector with entries 10:m, first 10 entries = 0
        std::for_each(all_data.A.begin() + 10, all_data.A.end(), [&cnt](T &entry) { entry = ++cnt;});

        // We expect A_norm to have all 1's
        RandLAPACK::util::normc(1, m, all_data.A, all_data.A_norm);

        // We expect this to be 1;
        T norm = blas::nrm2(m, all_data.A_norm.data(), 1);
        std::cout << "norm is" << std::fixed << norm << "\n";
        ASSERT_NEAR(norm, std::sqrt(m - 10), std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_binary_rank_search_zero_mat(int64_t m, int64_t n, std::vector<T> &A) {
        
        int64_t k = RandLAPACK::util::rank_search_binary(0, m, 0, n, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75), A.data());

        std::cout << "K IS " << k << "\n";
        ASSERT_EQ(k, 0);
    }

    template <typename T>
    static void 
    test_col_swp(ColSwpTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
    
        // Perform Pivoted QR
        lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());

        // Swap columns in A's copy
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy.data(), m, all_data.J.data());

        // Create an identity and store Q in it.
        RandLAPACK::util::eye(m, n, all_data.Ident.data());
        lapack::ormqr(Side::Left, Op::NoTrans, m, n, n, all_data.A.data(), m,  all_data.tau.data(),  all_data.Ident.data(), m);

        // Q * R -> Identity space
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.A.data(), m, all_data.Ident.data(), m);

        // A_piv - A_cpy
        for(int i = 0; i < m * n; ++i)
            all_data.A_cpy[i] -= all_data.Ident[i];

        T norm = lapack::lange(Norm::Fro, m, n, all_data.A_cpy.data(), m);
        std::cout << "||A_piv - QR||_F:  " << std::scientific << norm << "\n";
        ASSERT_NEAR(norm, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    }

    // Pins the gather contract of col_swap (a delegate to LAPACK's LAPMT):
    // on exit, column i of A holds what was column idx[i] - 1, and the pivot
    // vector, which serves as LAPMT's internal scratch, is restored on exit.
    template <typename T>
    static void 
    test_col_swp_gather(ColSwpTestData<T> &all_data, int64_t k) {

        auto m = all_data.row;
        auto n = all_data.col;
        std::vector<int64_t> J_copy(all_data.J);

        RandLAPACK::util::col_swap(m, n, k, all_data.A.data(), m, all_data.J.data());

        for (int64_t i = 0; i < n; ++i)
            ASSERT_EQ(all_data.J[i], J_copy[i]) << "pivot vector not restored at " << i;
        for (int64_t j = 0; j < k; ++j)
            for (int64_t i = 0; i < m; ++i)
                ASSERT_EQ(all_data.A[i + j * m], all_data.A_cpy[i + (all_data.J[j] - 1) * m])
                    << "col " << j << " row " << i << " m=" << m << " n=" << n << " k=" << k;
    }

    // Structured permutations that stress the cycle-following logic directly:
    // identity (no cycles), reversal (n/2 two-cycles), rotation (one n-cycle).
    template <typename T>
    static void 
    test_col_swp_structured(ColSwpTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        T* A = all_data.A.data();
        T const* orig = all_data.A_cpy.data();
        std::vector<int64_t> &J = all_data.J;

        std::iota(J.begin(), J.end(), 1);                       // identity
        RandLAPACK::util::col_swap(m, n, n, A, m, J.data());
        for (int64_t i = 0; i < m * n; ++i) ASSERT_EQ(A[i], orig[i]);

        for (int64_t i = 0; i < n; ++i) J[i] = n - i;           // reversal
        RandLAPACK::util::col_swap(m, n, n, A, m, J.data());
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                ASSERT_EQ(A[i + j * m], orig[i + (n - 1 - j) * m]);

        lapack::lacpy(MatrixType::General, m, n, orig, m, A, m);
        for (int64_t i = 0; i < n; ++i) J[i] = (i + 1) % n + 1; // one n-cycle
        RandLAPACK::util::col_swap(m, n, n, A, m, J.data());
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                ASSERT_EQ(A[i + j * m], orig[i + ((j + 1) % n) * m]);
    }

    // The matrix overload must respect lda > m (operating on a submatrix of
    // a taller allocation) and leave the rows below m untouched.
    template <typename T>
    static void 
    test_col_swp_lda(int64_t m, int64_t lda, int64_t n) {

        std::vector<T> A(lda * n);
        std::iota(A.begin(), A.end(), 0.0);
        std::vector<T> orig(A);
        std::vector<int64_t> J = {3, 1, 6, 2, 5, 4};

        RandLAPACK::util::col_swap(m, n, n, A.data(), lda, J.data());

        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < m; ++i)
                ASSERT_EQ(A[i + j * lda], orig[i + (J[j] - 1) * lda]);
            for (int64_t i = m; i < lda; ++i)
                ASSERT_EQ(A[i + j * lda], orig[i + j * lda]) << "padding row touched";
        }
    }

    // The integer-vector overload implements LAPMT-style cycle-following
    // directly (LAPMT itself only handles real matrices). Its pivot vector
    // is a permutation of 1..k, and entries of A beyond the first k must
    // stay untouched: the GPU counterpart and CQRRPT-style subvector use
    // depend on that prefix-only contract.
    static void 
    test_col_swp_int_vector(int64_t n, int64_t k, uint32_t seed) {

        std::vector<int64_t> A(n);
        std::iota(A.begin(), A.end(), 100);
        std::vector<int64_t> orig(A);
        std::vector<int64_t> J(k);
        std::iota(J.begin(), J.end(), 1);
        std::mt19937_64 gen(seed);
        std::shuffle(J.begin(), J.end(), gen);
        std::vector<int64_t> J_copy(J);

        RandLAPACK::util::col_swap(n, k, A.data(), J.data());

        for (int64_t i = 0; i < k; ++i)
            ASSERT_EQ(J[i], J_copy[i]) << "pivot vector not restored at " << i;
        for (int64_t i = 0; i < k; ++i)
            ASSERT_EQ(A[i], orig[J[i] - 1]) << "entry " << i;
        for (int64_t i = k; i < n; ++i)
            ASSERT_EQ(A[i], orig[i]) << "entry beyond k touched at " << i;
    }

    // Efficiency canary. The pre-LAPMT implementation searched the pivot
    // vector inside its swap loop, costing O(n^2): about 40 seconds at this
    // size (measured 35 ms at n = 30,000, quadratic scaling). The LAPMT
    // path runs in milliseconds, so the generous bound below only trips if
    // the O(n^2) behavior ever comes back.
    static void 
    test_col_swp_speed_canary(int64_t n) {

        std::vector<double> A(n);
        std::iota(A.begin(), A.end(), 0.0);
        std::vector<int64_t> J(n);
        for (int64_t i = 0; i < n; ++i) J[i] = (i + 1) % n + 1; // one n-cycle

        auto t0 = steady_clock::now();
        RandLAPACK::util::col_swap(1, n, n, A.data(), 1, J.data());
        double sec = std::chrono::duration<double>(steady_clock::now() - t0).count();

        for (int64_t j = 0; j < n; ++j)
            ASSERT_EQ(A[j], (double) ((j + 1) % n));
        std::cout << "col_swap n=" << n << " single-cycle: " << sec * 1e3 << " ms\n";
        ASSERT_LT(sec, 10.0) << "col_swap has lost its O(n) behavior";
    }

    template <typename T>
    static void 
    test_orhr_col(OrhrColTestData<T> &all_data) {

        int i;
        auto m = all_data.row;
        auto n = all_data.col;

        T* A      = all_data.A.data();
        T* B      = all_data.B.data();
        T* B1     = all_data.B1.data();
        T* A1     = all_data.A1.data();
        T* R      = all_data.R.data();
        T* T_mat  = all_data.T_mat.data();
        T* D      = all_data.D.data();
        T* D1     = all_data.D1.data();
        T* tau    = all_data.tau.data();
        T* tau1   = all_data.tau1.data();
    
        // Perform Cholesky QR  
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A, m, 0.0, R, n);
        lapack::potrf(Uplo::Upper, n, R, n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R, n, A, m);
        // Copy Q, R, B
        lapack::lacpy(MatrixType::General, m, n, A, m, A1, m);
        lapack::lacpy(MatrixType::General, m, n, B, m, B1, m);

        // built-in orhr_col
        lapack::orhr_col(m, n, n, A, m, T_mat, n, D);
        // own orhr_col
        RandLAPACK::util::rl_orhr_col(m, n, A1, m, tau1, D1, 1);

        for(i = 0; i < n; ++i)
            tau[i] = T_mat[(n + 1) * i];

        auto start_std = steady_clock::now();
        lapack::ormqr(Side::Left, Op::Trans, m, n, n, A,  m, tau,  B,  m);
        auto stop_std = steady_clock::now();
        long dur_std = duration_cast<microseconds>(stop_std - start_std).count();

        auto start_own = steady_clock::now();
        lapack::ormqr(Side::Left, Op::Trans, m, n, n, A1, m, tau1, B1, m);
        auto stop_own = steady_clock::now();
        long dur_own = duration_cast<microseconds>(stop_own - start_own).count();

        std::cout << "Own is " << std::fixed << (T) dur_std / dur_own << "x faster than std.\n";

        // A_piv - A_cpy
        for(i = 0; i < m * n; ++i) {
            A[i] -= A1[i];
            B[i] -= B1[i];
        }

        T norm1 = lapack::lange(Norm::Fro, m, n, A, m);
        T norm2 = lapack::lange(Norm::Fro, m, n, B, m);
        std::cout << "||Q_std - Q_own||_F:  " << std::scientific << norm1 << "\n";
        std::cout << "||B - B1||_F:         " << std::scientific << norm2 << "\n";
        
        //ASSERT_NEAR(norm, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    }
};

TEST_F(TestUtil, test_spectral_norm_polynomial_decay_double_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::RNGState();
    SpectralTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    test_spectral_norm(state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_rank_def_mat_double_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::RNGState();
    SpectralTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::adverserial);
    m_info.scaling = std::pow(10, 15);
    m_info.rank = n;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    test_spectral_norm(state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_polynomial_decay_single_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::RNGState();
    SpectralTestData<float> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<float> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    test_spectral_norm(state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_rank_def_mat_single_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::RNGState();
    SpectralTestData<float> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<float> m_info(m, n, RandLAPACK::gen::adverserial);
    m_info.scaling = std::pow(10, 7);
    m_info.rank = n;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    test_spectral_norm(state, all_data);
}

TEST_F(TestUtil, test_normc) {
    int64_t m = 1000;
    int64_t n = 1;
    NormcTestData<double> all_data(m, n);

    test_normc(all_data);
}

TEST_F(TestUtil, test_binary_rank_search_zero_mat) {
    int64_t m = 1000;
    int64_t n = 100;
    std::vector<double> A(m * n, 0.0); 

    test_binary_rank_search_zero_mat(m, n, A);
}

class Test_Inplace_Square_Transpose : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    virtual void apply(blas::Layout layout) {
        int64_t n = 37;
        RandBLAS::DenseDist D(n, n);
        RandBLAS::RNGState state(1);
        double *A1 = new double[n*n];
        state = RandBLAS::fill_dense(D, A1, state);
        double *A2 = new double[n*n];
        blas::copy(n*n, A1, 1, A2, 1);
        RandLAPACK::util::transpose_square(A2, n);
        RandBLAS::testing::matrices_approx_equal(
            layout, blas::Op::Trans, n, n, A1, n, A2, n, 
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        delete [] A1;
        delete [] A2;
    }

};


TEST_F(Test_Inplace_Square_Transpose, random_matrix_colmajor) {
    apply(blas::Layout::ColMajor);
}

TEST_F(Test_Inplace_Square_Transpose, random_matrix_rowmajor) {
    apply(blas::Layout::RowMajor);
}

TEST_F(TestUtil, test_col_swp) {
    
    int64_t m = 1000;
    int64_t n = 1000;
    auto state = RandBLAS::RNGState();
    ColSwpTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);

    test_col_swp<double>(all_data);
}

TEST_F(TestUtil, test_col_swp_gather_full_and_truncated) {

    for (auto [m, n, k, seed] : {std::tuple<int64_t, int64_t, int64_t, uint32_t>
            {10, 7, 7, 0}, {10, 7, 4, 1}, {1000, 200, 200, 2}, {8, 12, 5, 3},
            {5, 1, 1, 5}, {6, 9, 1, 6}}) {
        auto state = RandBLAS::RNGState(seed);
        ColSwpTestData<double> all_data(m, n);
        RandBLAS::DenseDist D(m, n);
        RandBLAS::fill_dense(D, all_data.A.data(), state);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
        std::iota(all_data.J.begin(), all_data.J.end(), 1);
        std::mt19937_64 gen(seed);
        std::shuffle(all_data.J.begin(), all_data.J.end(), gen);

        test_col_swp_gather<double>(all_data, k);
    }
}

TEST_F(TestUtil, test_col_swp_structured_permutations) {

    int64_t m = 3;
    int64_t n = 8;
    ColSwpTestData<double> all_data(m, n);
    std::iota(all_data.A.begin(), all_data.A.end(), 0.0);
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);

    test_col_swp_structured<double>(all_data);
}

TEST_F(TestUtil, test_col_swp_respects_lda) {
    test_col_swp_lda<double>(4, 7, 6);
}

TEST_F(TestUtil, test_col_swp_int_vector) {
    test_col_swp_int_vector(7, 7, 0);
    test_col_swp_int_vector(200, 200, 1);
    test_col_swp_int_vector(2800, 100, 2); // subvector pattern, k << n
}

TEST_F(TestUtil, test_col_swp_large_permutation_is_fast) {
    test_col_swp_speed_canary(1000000);
}

#if !defined(__APPLE__) || defined(ACCELERATE_NEW_LAPACK)
TEST_F(TestUtil, test_orhr_col) {
    
    int64_t m = 4;//std::pow(2, 10);
    int64_t n = 4;//std::pow(2, 9);
    auto state = RandBLAS::RNGState();
    OrhrColTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.B.data(), state);

    test_orhr_col<double>(all_data);
}
#endif
