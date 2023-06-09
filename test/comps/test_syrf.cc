#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>

#include <fstream>
#include <gtest/gtest.h>

class TestSYRF : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct SYRFTestData {
        int64_t row;
        int64_t col;
        int64_t rank;
        std::vector<T> A;
        std::vector<T> A_cpy;
        std::vector<T> Q;
        std::vector<T> Buf1;
        std::vector<T> Buf2;
        std::vector<T> Q_cpy;
        std::vector<T> Q_hat_cpy;

        SYRFTestData(int64_t m, int64_t k) :
        A(m * m, 0.0),
        A_cpy(m * m, 0.0),
        Q(m * k, 0.0), 
        Buf1(k * m, 0.0), 
        Buf2(m * m, 0.0), 
        Q_cpy(m * k, 0.0), 
        Q_hat_cpy(m * m, 0.0) 
        {
            row = m;
            rank = k;
        }
    };

    template <typename T, typename RNG>
    struct algorithm_objects {
        RandLAPACK::SYPS<T, RNG> SYPS;
        RandLAPACK::HQRQ<T> Orth_RF; 
        RandLAPACK::SYRF<T, RNG> SYRF;

        algorithm_objects(
            bool verbosity, 
            bool cond_check, 
            int64_t p, 
            int64_t passes_per_iteration
        ) :
            SYPS(p, passes_per_iteration, verbosity, cond_check),
            Orth_RF(cond_check, verbosity),
            SYRF(SYPS, Orth_RF, verbosity, cond_check)
            {}
    };

    template <typename T, typename RNG>
    static void orth_and_copy_computational_helper(SYRFTestData<T>& all_data) {
        
        auto m = all_data.row;

        blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, m, m, 1.0, all_data.A_cpy.data(), m, 0.0, all_data.A.data(), m);

        T* A_cpy_dat = all_data.A_cpy.data();
        lapack::lacpy(MatrixType::General, m, m, all_data.A.data(), m, all_data.A_cpy.data(), m);
        // Note that the matrix that is being used in SYRF call can remain triangular
        for(int i = 1; i < m; ++i)
            blas::copy(m - i, &A_cpy_dat[i + ((i-1) * m)], 1, &A_cpy_dat[(i - 1) + (i * m)], m);

        RandLAPACK::HQRQ<T> HQRQ(false, false);
        HQRQ.call(m, m, all_data.A_cpy);

        lapack::lacpy(MatrixType::General, m, m, all_data.A_cpy.data(), m, all_data.Q_hat_cpy.data(), m);
    }

    /// General test for QB:
    /// Computes QB factorzation, and checks:
    /// 1. A - QB
    /// 2. B - \transpose{Q}A
    /// 3. I - \transpose{Q}Q
    /// 4. A_k - QB = U_k\Sigma_k\transpose{V_k} - QB
    template <typename T, typename RNG>
    static void test_SYRF_general(
        RandBLAS::RNGState<RNG> state,
        SYRFTestData<T>& all_data, 
        algorithm_objects<T, RNG>& all_algs) {

        auto m = all_data.row;
        auto k = all_data.rank;

        all_algs.SYRF.call(Uplo::Upper, m, all_data.A, k, all_data.Q, state, NULL);

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

        // TEST 2:
        // Q' * Q_hat = Buf1
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, m, m, 1.0, Q_dat, m, Q_hat_dat, m, 0.0, Buf1_dat, k);
        // Q * Buf1 - Q_hat
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, Q_dat, m, Buf1_dat, k, -1.0, Q_hat_cpy_dat, m);

        // Q_hat' * Q = Buf2
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, m, k, m, 1.0, Q_hat_dat, m, Q_dat, m, 0.0, Buf2_dat, m);
        // Q_hat * Buf2 - Q
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, m, 1.0, Q_hat_dat, m, Buf2_dat, m, -1.0, Q_cpy_dat, m);

        T test_tol = std::pow(std::numeric_limits<T>::epsilon(), 0.625);
        // Test 1 Output
        T norm_test_1 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, Ident_dat, k);
        printf("FRO NORM OF Q'Q - I:   %e\n", norm_test_1);
        ASSERT_NEAR(norm_test_1, 0, test_tol);

        // Test 1 Output
        T norm1_test_2 = lapack::lange(Norm::Fro, m, m, Q_hat_cpy_dat, m);
        T norm2_test_2 = lapack::lange(Norm::Fro, m, k, Q_cpy_dat, m);
        printf("FRO NORM OF QQ' * Q_hat - Q_hat:   %e\n", norm1_test_2);
        printf("FRO NORM OF Q_hat Q_hat' * Q = Q:   %e\n", norm2_test_2);
        ASSERT_NEAR(std::min(norm1_test_2, norm2_test_2), 0, test_tol);
    }
};

TEST_F(TestSYRF, Polynomial_Decay_general1)
{
    int64_t m = 100;
    int64_t k = 100;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    auto state = RandBLAS::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;

    SYRFTestData<double> all_data(m, k);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, m, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration);
    orth_and_copy_computational_helper<double, r123::Philox4x32>(all_data);
    test_SYRF_general<double, r123::Philox4x32>(state, all_data, all_algs);
}

TEST_F(TestSYRF, Polynomial_Decay_general2)
{
    int64_t m = 100;
    int64_t k = 50;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    auto state = RandBLAS::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;

    SYRFTestData<double> all_data(m, k);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, m, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration);
    orth_and_copy_computational_helper<double, r123::Philox4x32>(all_data);
    test_SYRF_general<double, r123::Philox4x32>(state, all_data, all_algs);
}
