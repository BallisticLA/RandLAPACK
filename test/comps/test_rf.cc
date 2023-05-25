#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>

#include <fstream>
#include <gtest/gtest.h>

class TestRF : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct QBTestData {
        std::vector<T> A;
        std::vector<T> Q;

        QBTestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0), 
        Q(m * k, 0.0) 
        {}
    };

    /// General test for QB:
    /// Computes QB factorzation, and checks:
    /// 1. A - QB
    /// 2. B - \transpose{Q}A
    /// 3. I - \transpose{Q}Q
    /// 4. A_k - QB = U_k\Sigma_k\transpose{V_k} - QB
    template <typename T, typename RNG>
    static void test_RF_general(int64_t m, int64_t n, int64_t k, int64_t p, RandBLAS::base::RNGState<RNG> state, QBTestData<T>& all_data) {

        printf("|==================================TEST QB2 GENERAL BEGIN==================================|\n");

        //Subroutine parameters
        bool verbosity = false;
        bool cond_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        RandLAPACK::PLUL<T> Stab(cond_check, verbosity);
        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::RS<T, RNG> RS(Stab, state, p, passes_per_iteration, verbosity, cond_check);
        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::HQRQ<T> Orth_RF(cond_check, verbosity);
        // RangeFinder constructor - Choose default (rf1)
        RandLAPACK::RF<T> RF(RS, Orth_RF, verbosity, cond_check);

        RF.call(m, n, all_data.A, k, all_data.Q);

        // Reassing pointers because Q, B have been resized
        T* Q_dat = all_data.Q.data();

        std::vector<T> Ident(k * k, 0.0);
        T* Ident_dat = Ident.data();
        // Generate a reference identity
        RandLAPACK::util::eye(k, k, Ident);

        // TEST 1: Q'Q = I
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, Ident_dat, k);

        T test_tol = std::pow(std::numeric_limits<T>::epsilon(), 0.625);
        // Test 1 Output
        T norm_test_1 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, Ident_dat, k);
        printf("FRO NORM OF Q'Q - I:   %e\n", norm_test_1);
        ASSERT_NEAR(norm_test_1, 0, test_tol);
    }
};

TEST_F(TestRF, Polynomial_Decay_general1)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    auto state = RandBLAS::base::RNGState();

    QBTestData<double> all_data(m, n, k);
    
    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2025, false));
    test_RF_general<double, r123::Philox4x32>(m, n, k, p, state, all_data);
}

TEST_F(TestRF, Polynomial_Decay_general2)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    auto state = RandBLAS::base::RNGState();

    QBTestData<double> all_data(m, n, k);
    
    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 6.7, false));
    test_RF_general<double, r123::Philox4x32>(m, n, k, p, state, all_data);
}

TEST_F(TestRF, Rand_diag_general)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    auto state = RandBLAS::base::RNGState();

    QBTestData<double> all_data(m, n, k);
    
    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(4, 0, false));
    test_RF_general<double, r123::Philox4x32>(m, n, k, p, state, all_data);
}
