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
    struct RFTestData {
        std::vector<T> A;
        std::vector<T> A_cpy;
        std::vector<T> Q;
        std::vector<T> Buf1;
        std::vector<T> Buf2;
        std::vector<T> Q_cpy;
        std::vector<T> Q_hat_cpy;

        RFTestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0), 
        A_cpy(m * n, 0.0),
        Q(m * k, 0.0), 
        Buf1(m * m, 0.0), 
        Buf2(m * m, 0.0), 
        Q_cpy(m * k, 0.0), 
        Q_hat_cpy(m * n, 0.0) 
        {}
    };

    template <typename T, typename RNG>
    struct algorithm_objects {
        RandLAPACK::PLUL<T> Stab;
        RandLAPACK::RS<T, RNG> RS;
        RandLAPACK::HQRQ<T> Orth_RF;
        RandLAPACK::RF<T> RF;

        algorithm_objects(bool verbosity, 
                            bool cond_check, 
                            int64_t p, 
                            int64_t passes_per_iteration, 
                            RandBLAS::base::RNGState<RNG> state) :
                            Stab(cond_check, verbosity),
                            RS(Stab, state, p, passes_per_iteration, verbosity, cond_check),
                            Orth_RF(cond_check, verbosity),
                            RF(RS, Orth_RF, verbosity, cond_check)
                            {}
    };

    template <typename T, typename RNG>
    static void orth_and_copy_computational_helper(int64_t m, int64_t n, RFTestData<T>& all_data) {
        
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
        
        RandLAPACK::CholQRQ<T> CholQRQ(false, false);
        CholQRQ.call(m, n, all_data.A_cpy);

        lapack::lacpy(MatrixType::General, m, n, all_data.A_cpy.data(), m, all_data.Q_hat_cpy.data(), m);
    }

    /// General test for QB:
    /// Computes QB factorzation, and checks:
    /// 1. A - QB
    /// 2. B - \transpose{Q}A
    /// 3. I - \transpose{Q}Q
    /// 4. A_k - QB = U_k\Sigma_k\transpose{V_k} - QB
    template <typename T, typename RNG>
    static void test_RF_general(
        int64_t m, 
        int64_t n, 
        int64_t k, 
        RFTestData<T>& all_data, 
        algorithm_objects<T, RNG>& all_algs) {

        all_algs.RF.call(m, n, all_data.A, k, all_data.Q);

        // Reassing pointers because Q, B have been resized
        T* Q_dat = all_data.Q.data();
        T* Q_cpy_dat = all_data.Q_cpy.data();
        T* Buf1_dat = all_data.Buf1.data();
        T* Buf2_dat = all_data.Buf2.data();
        T* Q_hat_dat = all_data.A_cpy.data();
        T* Q_hat_cpy_dat = all_data.Q_hat_cpy.data();

        lapack::lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);

        std::vector<T> Ident(k * k, 0.0);
        T* Ident_dat = Ident.data();
        // Generate a reference identity
        RandLAPACK::util::eye(k, k, Ident);

        // TEST 1: Q'Q = I
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, Ident_dat, k);

        // TEST 2: QQ' * Q_hat = Q_hat
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::NoTrans, m, m, 1.0, Q_dat, m, -1.0, Buf1_dat, m);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m, 1.0, Buf1_dat, m, Q_hat_dat, m, -1.0, Q_hat_cpy_dat, m);

        // TEST 2: Q_hat Q_hat' * Q = Q
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::NoTrans, m, m, 1.0, Q_dat, m, -1.0, Buf2_dat, m);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, m, 1.0, Buf2_dat, m, Q_dat, m, -1.0, Q_cpy_dat, m);

        T test_tol = std::pow(std::numeric_limits<T>::epsilon(), 0.625);
        // Test 1 Output
        T norm_test_1 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, Ident_dat, k);
        printf("FRO NORM OF Q'Q - I:   %e\n", norm_test_1);
        ASSERT_NEAR(norm_test_1, 0, test_tol);

        // Test 1 Output
        T norm1_test_2 = lapack::lange(Norm::Fro, m, n, Q_hat_cpy_dat, m);
        T norm2_test_2 = lapack::lange(Norm::Fro, m, k, Q_cpy_dat, m);
        printf("FRO NORM OF QQ' * Q_hat - Q_hat:   %e\n", norm1_test_2);
        printf("FRO NORM OF Q_hat Q_hat' * Q = Q:   %e\n", norm2_test_2);
        if(k == n) {
            ASSERT_NEAR(norm1_test_2, 0, test_tol);
            ASSERT_NEAR(norm1_test_2, 0, test_tol);
        }
    }
};

TEST_F(TestRF, Polynomial_Decay_general1)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 100;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    auto state = RandBLAS::base::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;

    RFTestData<double> all_data(m, n, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration, state);
    
    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2025, false));
    orth_and_copy_computational_helper<double, r123::Philox4x32>(m, n, all_data);
    
    test_RF_general<double, r123::Philox4x32>(m, n, k, all_data, all_algs);
}

TEST_F(TestRF, Polynomial_Decay_general2)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    auto state = RandBLAS::base::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;

    RFTestData<double> all_data(m, n, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration, state);
    
    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2025, false));
    orth_and_copy_computational_helper<double, r123::Philox4x32>(m, n, all_data);
    
    test_RF_general<double, r123::Philox4x32>(m, n, k, all_data, all_algs);
}

TEST_F(TestRF, Rand_diag_general)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    auto state = RandBLAS::base::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;

    RFTestData<double> all_data(m, n, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration, state);
    
    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2025, false));
    orth_and_copy_computational_helper<double, r123::Philox4x32>(m, n, all_data);
    
    test_RF_general<double, r123::Philox4x32>(m, n, k, all_data, all_algs);
}
