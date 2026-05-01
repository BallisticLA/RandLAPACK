#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


class TestNystromEVD : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct NystromEVDTestData {
        int64_t dim;
        int64_t rank;
        std::vector<T> A;
        std::vector<T> A_cpy;
        T* V = nullptr; int64_t V_sz = 0;
        T* eigvals = nullptr; int64_t eigvals_sz = 0;
        std::vector<T> E;
        std::vector<T> Buf;

        NystromEVDTestData(
            int64_t m, int64_t k
        ) :
            A(m * m, 0.0),
            A_cpy(m * m, 0.0),
            E(k * k, 0.0),
            Buf(m * k, 0.0)
        {
            dim = m;
            rank = k;
        }

        ~NystromEVDTestData() { delete[] V; delete[] eigvals; }
    };

    template <typename T>
    struct NystromEVDUploTestData {
        int64_t dim;
        int64_t rank;
        std::vector<T> work;
        std::vector<T> A_u;
        T* V_u = nullptr; int64_t V_u_sz = 0;
        T* eigvals_u = nullptr; int64_t eigvals_u_sz = 0;
        std::vector<T> A_l;
        T* V_l = nullptr; int64_t V_l_sz = 0;
        T* eigvals_l = nullptr; int64_t eigvals_l_sz = 0;
        std::vector<T> E_u;
        std::vector<T> E_l;

        NystromEVDUploTestData(
            int64_t m, int64_t k
        ) :
            work(m * m, 0.0),
            A_u(m * m, 0.0),
            A_l(m * m, 0.0),
            E_u(k * k, 0.0),
            E_l(k * k, 0.0)
        {
            dim = m;
            rank = k;
        }

        ~NystromEVDUploTestData() {
            delete[] V_u; delete[] eigvals_u;
            delete[] V_l; delete[] eigvals_l;
        }
    };

    template <typename T, typename RNG>
    struct algorithm_objects {
        using SYPS_t = RandLAPACK::SYPS<T, RNG>;
        using Orth_t = RandLAPACK::HQRQ<T>;
        using SYRF_t = RandLAPACK::SYRF<SYPS_t, Orth_t>;
        SYPS_t syps;
        Orth_t orth; 
        SYRF_t syrf;
        RandLAPACK::NystromEVD<SYRF_t> nystrom_evd;


        algorithm_objects(
            bool verbose, 
            bool cond_check, 
            int64_t num_syps_passes, 
            int64_t passes_per_syps_stabilization, 
            int64_t num_steps_power_iter_error_est
        ) : 
            syps(num_syps_passes, passes_per_syps_stabilization, verbose, cond_check),
            orth(cond_check, verbose),
            syrf(syps, orth, verbose, cond_check),
            nystrom_evd(syrf, num_steps_power_iter_error_est, verbose)
            {}
    };

    template <typename T>
    static void symm_mat_and_copy_computational_helper(T &norm_A, NystromEVDTestData<T> &all_data) {
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

    template <typename T>
    static void uplo_computational_helper(NystromEVDUploTestData<T> &all_data) {
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
    static void test_NystromEVD_general(
        int64_t k_start, 
        T tol,
        int rank_expectation, 
        T err_expectation, 
        T &norm_A, 
        NystromEVDTestData<T> &all_data,
        algorithm_objects<T, RNG> &all_algs,
        RandBLAS::RNGState<RNG> state
    ) {
        
        auto m = all_data.dim;

        int64_t k = k_start;
        all_algs.nystrom_evd.call(blas::Uplo::Upper, m, all_data.A.data(), k, tol,
                            all_data.V, all_data.V_sz, all_data.eigvals, all_data.eigvals_sz, state);

        T* E_dat = RandLAPACK::util::resize(k * k, all_data.E);
        T* Buf_dat = RandLAPACK::util::resize(m * k, all_data.Buf);

        T* A_cpy_dat = all_data.A_cpy.data();
        T* V_dat = all_data.V;

        // Construct A_hat = V * diag(eigvals) * V'

        // Turn array into diagonal matrix
        RandLAPACK::util::diag(k, k, all_data.eigvals, k, all_data.E.data());
        // V * E = Buf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, V_dat, m, E_dat, k, 0.0, Buf_dat, m);
        // A - Buf * V' - should be close to 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, m, k, 1.0, Buf_dat, m, V_dat, m, -1.0, A_cpy_dat, m);

        T norm_0 = lapack::lange(Norm::Fro, m, m, A_cpy_dat, m);
        printf("||A - VEV'||_F / ||A||_F:  %e\n", norm_0 / norm_A);
        ASSERT_NEAR(norm_0 / norm_A, err_expectation, 10 * err_expectation);
        ASSERT_EQ(k, rank_expectation);
    }

        /// General test for REVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T, typename RNG>
    static void test_NystromEVD_uplo(
        int64_t k_start, 
        T tol,
        T err_expectation, 
        NystromEVDUploTestData<T> &all_data,
        algorithm_objects<T, RNG> &all_algs,
        RandBLAS::RNGState<RNG> state
    ) {
        
        auto m = all_data.dim;

        int64_t k = k_start;
        all_algs.nystrom_evd.call(blas::Uplo::Upper, m, all_data.A_u.data(), k, tol,
                            all_data.V_u, all_data.V_u_sz, all_data.eigvals_u, all_data.eigvals_u_sz, state);
        all_algs.nystrom_evd.call(blas::Uplo::Lower, m, all_data.A_l.data(), k, tol,
                            all_data.V_l, all_data.V_l_sz, all_data.eigvals_l, all_data.eigvals_l_sz, state);

        T* E_u_dat = RandLAPACK::util::resize(k * k, all_data.E_u);
        T* E_l_dat = RandLAPACK::util::resize(k * k, all_data.E_l);
        T* V_u_dat = all_data.V_u;
        T* V_l_dat = all_data.V_l;
        T* work_u_dat = all_data.A_u.data();
        T* work_l_dat = all_data.A_l.data();
        T* A_approx_dat = all_data.work.data();

        RandLAPACK::util::diag(k, k, all_data.eigvals_u, k, all_data.E_u.data());
        RandLAPACK::util::diag(k, k, all_data.eigvals_l, k, all_data.E_l.data());

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

TEST_F(TestNystromEVD, Underestimation1) { 
    using RNG = r123::Philox4x32;

    int64_t m = 1000;
    int64_t k = 100;
    int64_t k_start = 1;
    int64_t rank_expectation = 32;
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

    NystromEVDTestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::gen::mat_gen_info<double> m_info(m, m, RandLAPACK::gen::polynomial);
    m_info.cond_num = std::pow(10, 8);
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A_cpy.data(), state);

    symm_mat_and_copy_computational_helper(norm_A, all_data);
    test_NystromEVD_general(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestNystromEVD, Underestimation2) { 
    using RNG = r123::Philox4x32;

    int64_t m = 1000;
    int64_t k = 100;
    int64_t k_start = 10;
    int64_t rank_expectation = 40;
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

    NystromEVDTestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::gen::mat_gen_info<double> m_info(m, m, RandLAPACK::gen::polynomial);
    m_info.cond_num = std::pow(10, 8);
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A_cpy.data(), state);

    symm_mat_and_copy_computational_helper(norm_A, all_data);
    test_NystromEVD_general(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestNystromEVD, Overestimation1) { 
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

    NystromEVDTestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::gen::mat_gen_info<double> m_info(m, m, RandLAPACK::gen::polynomial);
    m_info.cond_num = std::pow(10, 2);
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A_cpy.data(), state);

    symm_mat_and_copy_computational_helper(norm_A, all_data);
    test_NystromEVD_general(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestNystromEVD, Overestimation2) {
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

    NystromEVDTestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::gen::mat_gen_info<double> m_info(m, m, RandLAPACK::gen::polynomial);
    m_info.cond_num = std::pow(10, 2);
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A_cpy.data(), state);

    symm_mat_and_copy_computational_helper(norm_A, all_data);
    test_NystromEVD_general(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

TEST_F(TestNystromEVD, Exactness) { 
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

    NystromEVDTestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::gen::mat_gen_info<double> m_info(m, m, RandLAPACK::gen::polynomial);
    m_info.cond_num = std::pow(10, 2);
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A_cpy.data(), state);

    symm_mat_and_copy_computational_helper(norm_A, all_data);
    test_NystromEVD_general(
        k_start, tol, rank_expectation, err_expectation, norm_A, all_data, all_algs, state
    );
}

// Verify that error_est_power_iters=0 and tol=0 produce fixed-rank behavior:
// k must not increase, since FunNystromPP's (n-k)*f(0) tail correction depends on k staying fixed.
TEST_F(TestNystromEVD, FixedRank) {
    using RNG = r123::Philox4x32;
    int64_t m = 40, k = 10;
    auto state = RandBLAS::RNGState(5);

    std::vector<double> A(m * m, 0.0);
    for (int64_t i = 0; i < m; ++i)
        A[i + i * m] = (double)(i + 1);

    algorithm_objects<double, RNG> all_algs(false, false, 3, 1, /*error_est_p=*/0);
    double* V_out = nullptr; int64_t V_out_sz = 0;
    double* eigvals_out = nullptr; int64_t eigvals_out_sz = 0;
    int64_t k_before = k;
    all_algs.nystrom_evd.call(blas::Uplo::Upper, m, A.data(), k, 0.0, V_out, V_out_sz, eigvals_out, eigvals_out_sz, state);
    delete[] V_out;
    delete[] eigvals_out;

    printf("NystromEVD fixed-k: k_before=%lld, k_after=%lld\n", (long long)k_before, (long long)k);
    ASSERT_EQ(k, k_before);
}

TEST_F(TestNystromEVD, Uplo) {
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

    NystromEVDUploTestData<double> all_data(m, k);
    algorithm_objects<double, RNG> all_algs(
        verbose, cond_check,
        num_syps_passes, 
        passes_per_syps_stabilization, 
        num_steps_power_iter_error_est
    );

    RandLAPACK::gen::mat_gen_info<double> m_info(m, m, RandLAPACK::gen::polynomial);
    m_info.cond_num = std::pow(10, 2);
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.work.data(), state);

    uplo_computational_helper(all_data);
    
    test_NystromEVD_uplo(k_start, tol, err_expectation, all_data, all_algs, state);
}
