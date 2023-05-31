#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <RandBLAS/test_util.hh>

#include <math.h>
#include <chrono>
#include <gtest/gtest.h>
/*
TODO #1: Resizing tests.

TODO #2: Diagonalization tests.

TODO #4: L & pivotig tests.
*/
using namespace std::chrono;

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

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

    template <typename T, typename RNG>
    static void 
    test_spectral_norm(RandBLAS::base::RNGState<RNG> state, SpectralTestData<T>& all_data) {

        auto m = all_data.row;
        auto n = all_data.col;

        T norm = RandLAPACK::util::estimate_spectral_norm(m, n, all_data.A.data(), 10000, state);
        // Get an SVD -> first singular value == 2_norm
        lapack::gesdd(Job::NoVec, m, n, all_data.A_cpy.data(), m, all_data.s.data(), NULL, m, NULL, n);

        printf("Computed norm:  %e\nComputed s_max: %e\n", norm, all_data.s[0]);
        ASSERT_NEAR(norm, all_data.s[0], std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_normc(NormcTestData<T>& all_data) {
        
        auto m = all_data.row;

        int cnt = 0;
        // we have a vector with entries 10:m, first 10 entries = 0
        std::for_each(all_data.A.begin() + 10, all_data.A.end(), [&cnt](T &entry) { entry = ++cnt;});

        // We expect A_norm to have all 1's
        RandLAPACK::util::normc(1, m, all_data.A, all_data.A_norm);

        // We expect this to be 1;
        T norm = blas::nrm2(m, all_data.A_norm.data(), 1);
        printf("norm is%f\n", norm);
        ASSERT_NEAR(norm, std::sqrt(m - 10), std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_binary_rank_search_zero_mat(int64_t m, int64_t n, std::vector<T>& A) {
        
        int64_t k = RandLAPACK::util::rank_search_binary(0, m, 0, n, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75), A.data());

        printf("K IS %ld\n", k);
        ASSERT_EQ(k, 0);
    }
};

TEST_F(TestUtil, test_spectral_norm_polynomial_decay_double_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::base::RNGState();
    SpectralTestData<double> all_data(m, n);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, n, state, std::make_tuple(0, 2025, false));
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    test_spectral_norm<double, r123::Philox4x32>(state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_rank_def_mat_double_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::base::RNGState();
    SpectralTestData<double> all_data(m, n);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, n, state, std::make_tuple(9, std::pow(10, 15), false));
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    test_spectral_norm<double, r123::Philox4x32>(state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_polynomial_decay_single_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::base::RNGState();
    SpectralTestData<float> all_data(m, n);

    RandLAPACK::util::gen_mat_type<float, r123::Philox4x32>(m, n, all_data.A, n, state, std::make_tuple(0, 2, false));
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    test_spectral_norm<float, r123::Philox4x32>(state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_rank_def_mat_single_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::base::RNGState();
    SpectralTestData<float> all_data(m, n);

    RandLAPACK::util::gen_mat_type<float, r123::Philox4x32>(m, n, all_data.A, n, state, std::make_tuple(9, std::pow(10, 7), false));
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    test_spectral_norm<float, r123::Philox4x32>(state, all_data);
}

TEST_F(TestUtil, test_normc) {
    int64_t m = 1000;
    int64_t n = 1;
    NormcTestData<double> all_data(m, n);

    test_normc<double>(all_data);
}

TEST_F(TestUtil, test_binary_rank_search_zero_mat) {
    int64_t m = 1000;
    int64_t n = 100;
    std::vector<double> A(m * n, 0.0); 

    test_binary_rank_search_zero_mat<double>(m, n, A);
}


class Test_Inplace_Square_Transpose : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    virtual void apply(blas::Layout layout) {
        int64_t n = 37;
        RandBLAS::dense::DenseDist D{n, n};
        RandBLAS::base::RNGState state(1);
        double *A1 = new double[n*n];
        state = RandBLAS::dense::fill_buff(A1, D, state);
        double *A2 = new double[n*n];
        blas::copy(n*n, A1, 1, A2, 1);
        RandLAPACK::util::transpose_square(A2, n);
        RandBLAS_Testing::Util::matrices_approx_equal(
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
