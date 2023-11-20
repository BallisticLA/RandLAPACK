#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


class TestRBKI : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct RBKITestData {
        int64_t row;
        int64_t col;
        int64_t rank; // has to be modifiable
        std::vector<T> A;
        std::vector<T> U;
        std::vector<T> V;
        std::vector<T> Sigma;
        std::vector<T> A_cpy;
        std::vector<T> Sigma_exact;

        RBKITestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0),
        U(m * n, 0.0),
        V(n * n, 0.0),
        Sigma(n, 0.0),
        A_cpy(m * n, 0.0),
        Sigma_exact(n, 0.0)
        {
            row = m;
            col = n;
            rank = k;
        }
    };

    template <typename T, typename RNG>
    static void norm_and_copy_computational_helper(T &norm_A, RBKITestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }

    template <typename T, typename RNG, typename alg_type>
    static void test_RBKI_general(
        T norm_A,
        RBKITestData<T> &all_data,
        alg_type &RBKI,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        RBKI.call(m, n, all_data.A.data(), m, k, all_data.U.data(), all_data.V.data(), all_data.Sigma.data(), state);
        // Compute singular values via a deterministic method
        lapack::gesdd(Job::NoVec, m, n, all_data.A_cpy.data(), m, all_data.Sigma_exact.data(), NULL, m, NULL, n);
        
        // Find diff between singular values computed by two methods
        int cnt = -1;

        for(int i = 0; i < n; ++i) {
            //printf("%e, %e\n", all_data.Sigma[i], all_data.Sigma_exact[i]);
        }

        std::for_each(all_data.Sigma.data(), all_data.Sigma.data() + k,
            // Lambda expression begins
            [&cnt, &all_data](T &entry) {
                    entry -= all_data.Sigma_exact[++cnt];
            }
        );
        T norm = blas::nrm2(k, all_data.Sigma.data(), 1);
        printf("||A_svd - A_rbki||_F: %e\n", norm);

    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestRBKI, RBKI_basic) {
    int64_t m = 4000;
    int64_t n = 200;
    int64_t k = 100;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    RBKITestData<double> all_data(m, n, k);
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, false, tol);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_RBKI_general<double, r123::Philox4x32, RandLAPACK::RBKI<double, r123::Philox4x32>>(norm_A, all_data, RBKI, state);
}
