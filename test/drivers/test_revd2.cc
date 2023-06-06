#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

class TestREVD2 : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct REVD2TestData {
        int64_t dim;
        int64_t rank;
        std::vector<T> A;
        std::vector<T> A_cpy;
        std::vector<T> V;
        std::vector<T> eigvals;
        std::vector<T> E;
        std::vector<T> Buf;

        REVD2TestData(
            int64_t m, int64_t k
        ) : 
            A(m * m, 0.0), 
            A_cpy(m * m, 0.0),  
            V(m * k, 0.0), 
            eigvals(k, 0.0), 
            E(k * k, 0.0), 
            Buf(m * k, 0.0)
        {
            dim = m;
            rank = k;
        }
    };

        template <typename T>
    struct REVD2UploTestData {
        int64_t dim;
        int64_t rank;
        std::vector<T> work;
        std::vector<T> A_u;
        std::vector<T> V_u;
        std::vector<T> eigvals_u;
        std::vector<T> A_l;
        std::vector<T> V_l;
        std::vector<T> eigvals_l;
        std::vector<T> E_u;
        std::vector<T> E_l;

        REVD2UploTestData(
            int64_t m, int64_t k
        ) : 
            work(m * m, 0.0),
            A_u(m * m, 0.0),
            V_u(m * k, 0.0), 
            eigvals_u(k, 0.0), 
            A_l(m * m, 0.0),
            V_l(m * k, 0.0), 
            eigvals_l(k, 0.0),
            E_u(k * k, 0.0), 
            E_l(k * k, 0.0)
        {
            dim = m;
            rank = k;
        }
    };

    template <typename T, typename RNG>
    struct algorithm_objects {
        RandLAPACK::SYPS<T, RNG> SYPS;
        RandLAPACK::HQRQ<T> Orth_RF; 
        RandLAPACK::SYRF<T, RNG> SYRF;
        //  ^ Needs a symmetric power skether and an orthogonalizer
        RandLAPACK::REVD2<T, RNG> REVD2;
        //  ^ Needs a symmetric rangefinder.


        algorithm_objects(
            bool verbose, 
            bool cond_check, 
            int64_t num_syps_passes, 
            int64_t passes_per_syps_stabilization, 
            int64_t num_steps_power_iter_error_est
        ) : 
            SYPS(num_syps_passes, passes_per_syps_stabilization, verbose, cond_check),
            Orth_RF(cond_check, verbose),
            SYRF(SYPS, Orth_RF, verbose, cond_check),
            REVD2(SYRF, num_steps_power_iter_error_est, verbose)
            {}
    };

    template <typename T, typename RNG>
    static void symm_mat_and_copy_computational_helper(T& norm_A, REVD2TestData<T>& all_data) {
        auto m = all_data.dim;
        // We're using Nystrom, the original must be positive semidefinite
        blas::syrk(
            Layout::ColMajor, Uplo::Lower, Op::Trans,
            m, m, 1.0, all_data.A_cpy.data(), m, 0.0, all_data.A.data(), m
        );
        T* A_dat = all_data.A.data();
        // shouldn't need the line below if things are implemented properly (maybe need lansy?)
        for(int i = 1; i < m; ++i)
            blas::copy(m - i, &A_dat[i + ((i-1) * m)], 1, &A_dat[(i - 1) + (i * m)], m);
        blas::copy(m * m, all_data.A.data(), 1, all_data.A_cpy.data(), 1);
        norm_A = lapack::lange(Norm::Fro, m, m, all_data.A_cpy.data(), m);
    }

    template <typename T, typename RNG>
    static void uplo_computational_helper(REVD2UploTestData<T>& all_data) {
        auto m = all_data.dim;
        T* A_u_dat = all_data.A_u.data();
        T* A_l_dat = all_data.A_l.data();

       // Filling the lower-triangular matrix
        blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, m, m, 1.0, all_data.work.data(), m, 0.0, A_l_dat, m);
        // Filling the upper-triangular matrix
        for(int i = 0; i < m; ++i)
            blas::copy(m - i, &A_l_dat[i + (i * m)], 1, &A_u_dat[i + (i * m)], m);

        // Fill the unused space with NANs
        std::fill(&A_u_dat[1], &A_u_dat[m], NAN);
        for(int i = 1; i < m; ++i) {
            std::fill(&A_l_dat[m * i], &A_l_dat[i + m * i], NAN);
            std::fill(&A_u_dat[i + 1 + m * i], &A_u_dat[m * (i + 1)], NAN);
        }
    }

    /// General test for REVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T, typename RNG>
    static void test_REVD2_general(
        int64_t k_start, 
        T tol,
        int rank_expectation, 
        T err_expectation, 
        T& norm_A, 
        REVD2TestData<T>& all_data,
        algorithm_objects<T, RNG>& all_algs,
        RandBLAS::RNGState<RNG> state
    ) {
        
        auto m = all_data.dim;

        int64_t k = k_start;
        all_algs.REVD2.call(blas::Uplo::Upper, m, all_data.A, k, tol, all_data.V, all_data.eigvals, state);

        T* E_dat = RandLAPACK::util::upsize(k * k, all_data.E);
        T* Buf_dat = RandLAPACK::util::upsize(m * k, all_data.Buf);

        T* A_cpy_dat = all_data.A_cpy.data();
        T* V_dat = all_data.V.data();

        // Construnct A_hat = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, all_data.eigvals, k, all_data.E);
        // V * E = Buf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, V_dat, m, E_dat, k, 0.0, Buf_dat, m);
        // A - Buf * V' - should be close to 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, m, k, 1.0, Buf_dat, m, V_dat, m, -1.0, A_cpy_dat, m);

        T norm_0 = lapack::lange(Norm::Fro, m, m, A_cpy_dat, m);
        printf("||A - VEV'||_F / ||A||_F:  %e\n", norm_0 / norm_A);
        ASSERT_NEAR(norm_0 / norm_A, err_expectation, 10 * err_expectation);
        ASSERT_NEAR(k, rank_expectation, std::numeric_limits<T>::epsilon());
    }

        /// General test for REVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T, typename RNG>
    static void test_REVD2_uplo(
        int64_t k_start, 
        T tol,
        T err_expectation, 
        REVD2UploTestData<T>& all_data,
        algorithm_objects<T, RNG>& all_algs,
        RandBLAS::RNGState<RNG> state
    ) {
        
        auto m = all_data.dim;

        int64_t k = k_start;
        all_algs.REVD2.call(blas::Uplo::Upper, m, all_data.A_u, k, tol, all_data.V_u, all_data.eigvals_u, state);
        all_algs.REVD2.call(blas::Uplo::Lower, m, all_data.A_l, k, tol, all_data.V_l, all_data.eigvals_l, state);

        T* E_u_dat = RandLAPACK::util::upsize(k * k, all_data.E_u);
        T* E_l_dat = RandLAPACK::util::upsize(k * k, all_data.E_l);
        T* V_u_dat = all_data.V_u.data();
        T* V_l_dat = all_data.V_l.data();
        T* work_u_dat = all_data.A_u.data();
        T* work_l_dat = all_data.A_l.data();
        T* A_approx_dat = all_data.work.data();

        RandLAPACK::util::diag(k, k, all_data.eigvals_u, k, all_data.E_u);
        RandLAPACK::util::diag(k, k, all_data.eigvals_l, k, all_data.E_l);

        // Reconstruct factorizations, compare the result
        // V_u * E_u = work_u
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, V_u_dat, m, E_u_dat, k, 0.0, work_u_dat, m);
        // work_u * V_u' = A_approx
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, m, k, 1.0, work_u_dat, m, V_u_dat, m, 0.0, A_approx_dat, m);
        // V_l * E_l = work_l
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, V_l_dat, m, E_l_dat, k, 0.0, work_l_dat, m);
        // work_l * V_l' - A_approx
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, m, k, 1.0, work_l_dat, m, V_l_dat, m, -1.0, A_approx_dat, m);

        T norm = lapack::lange(Norm::Fro, m, m, A_approx_dat, m);
        printf("||V_u*E_u*V_u' - V_l*E_l*V_l'||_F  %e\n", norm);
        ASSERT_NEAR(norm, 0.0, 10 * err_expectation);
    }
};

TEST_F(TestREVD2, Underestimation1) { 
    using RNG = r123::Philox4x32;

    int64_t m = 1000;
    int64_t k = 100;
    int64_t k_start = 1;
    int64_t rank_expectation = 64;
    double norm_A = 0;
    double tol = std::pow(10, -14);
    double err_expectation = std::pow(10, -13);
    int64_t num_syps_passes = 3;
    int64_t passes_per_syps_stabilization = 1;
    int64_t num_steps_power_iter_error_est = 10;
    auto state = RandBLAS::RNGState(0);

    //Subroutine parameters 
    bool verbose = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::util::gen_mat_type<double, RNG>(
        m, m, all_data.A_cpy, k, state, std::make_tuple(0, std::pow(10, 8), false)
    );
    symm_mat_and_copy_computational_helper<double, RNG>(norm_A, all_data);
    test_REVD2_general<double, RNG>(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestREVD2, Underestimation2) { 
    using RNG = r123::Philox4x32;

    int64_t m = 1000;
    int64_t k = 100;
    int64_t k_start = 10;
    int64_t rank_expectation = 80;
    double norm_A = 0;
    double tol = std::pow(10, -14);
    double err_expectation = std::pow(10, -13);
    int64_t num_syps_passes = 3;
    int64_t passes_per_syps_stabilization = 1;
    int64_t num_steps_power_iter_error_est = 10;
    auto state = RandBLAS::RNGState(0);
    //Subroutine parameters 
    bool verbose = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::util::gen_mat_type<double, RNG>(
        m, m, all_data.A_cpy, k, state, std::make_tuple(0, std::pow(10, 8), false)
    );
    symm_mat_and_copy_computational_helper<double, RNG>(norm_A, all_data);
    test_REVD2_general<double, RNG>(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestREVD2, Overestimation1) { 
    using RNG = r123::Philox4x32;

    int64_t m = 1000;
    int64_t k = 100;
    int64_t k_start = 10;
    int64_t rank_expectation = 160;
    double norm_A = 0;
    double tol = std::pow(10, -14);
    double err_expectation = std::pow(10, -13);
    int64_t num_syps_passes = 3;
    int64_t passes_per_syps_stabilization = 1;
    int64_t num_steps_power_iter_error_est = 10;
    auto state = RandBLAS::RNGState(0);
    //Subroutine parameters 
    bool verbose = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::util::gen_mat_type<double, RNG>(m, m, all_data.A_cpy, k, state, std::make_tuple(0, std::pow(10, 2), false));
    symm_mat_and_copy_computational_helper<double, RNG>(norm_A, all_data);
    test_REVD2_general<double, RNG>(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestREVD2, Oversetimation2) { 
    using RNG = r123::Philox4x32;

    int64_t m = 1000;
    int64_t k = 159;
    int64_t k_start = 10;
    int64_t rank_expectation = 160;
    double norm_A = 0;
    double tol = std::pow(10, -14);
    double err_expectation = std::pow(10, -13);
    int64_t num_syps_passes = 3;
    int64_t passes_per_syps_stabilization = 1;
    int64_t num_steps_power_iter_error_est = 10;
    auto state = RandBLAS::RNGState(0);
    //Subroutine parameters 
    bool verbose = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::util::gen_mat_type<double, RNG>(
        m, m, all_data.A_cpy, k, state, std::make_tuple(0, std::pow(10, 2), false)
    );
    symm_mat_and_copy_computational_helper<double, RNG>(norm_A, all_data);
    test_REVD2_general<double, RNG>(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestREVD2, Exactness) { 
    using RNG = r123::Philox4x32;

    int64_t m = 100;
    int64_t k = 100;
    int64_t k_start = 10;
    int64_t rank_expectation = 100;
    double norm_A = 0;
    double tol = std::pow(10, -14);
    double err_expectation = std::pow(10, -13);
    int64_t num_syps_passes = 3;
    int64_t passes_per_syps_stabilization = 1;
    int64_t num_steps_power_iter_error_est = 10;
    auto state = RandBLAS::RNGState(0);
    //Subroutine parameters 
    bool verbose = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::util::gen_mat_type<double, RNG>(m, m, all_data.A_cpy, k, state, std::make_tuple(0, std::pow(10, 2), false));
    symm_mat_and_copy_computational_helper<double, RNG>(norm_A, all_data);
    test_REVD2_general<double, RNG>(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestREVD2, Uplo) { 
    using RNG = r123::Philox4x32;

    int64_t m = 100;
    int64_t k = 50;
    int64_t k_start = 1;
    double tol = std::pow(10, -14);
    double err_expectation = std::pow(10, -13);
    int64_t num_syps_passes = 3;
    int64_t passes_per_syps_stabilization = 1;
    int64_t num_steps_power_iter_error_est = 10;
    auto state = RandBLAS::RNGState(0);
    //Subroutine parameters 
    bool verbose = false;
    bool cond_check = false;

    REVD2UploTestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::util::gen_mat_type<double, RNG>(m, m, all_data.work, k, state, std::make_tuple(0, std::pow(10, 2), false));

    uplo_computational_helper<double, RNG>(all_data);
    
    test_REVD2_uplo<double, RNG>(k_start, tol, err_expectation, all_data, all_algs, state);
}
