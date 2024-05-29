#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


class TestCQRRP : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct CQRRPTestData {
        int64_t row;
        int64_t col;
        int64_t rank; // has to be modifiable
        std::vector<T> A;
        std::vector<T> Q;
        std::vector<T> R;
        std::vector<T> tau;
        std::vector<int64_t> J;
        std::vector<T> A_cpy1;
        std::vector<T> A_cpy2;
        std::vector<T> I_ref;

        CQRRPTestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0),
        Q(m * n, 0.0),
        tau(n, 0.0),
        J(n, 0),
        A_cpy1(m * n, 0.0),
        A_cpy2(m * n, 0.0),
        I_ref(k * k, 0.0) 
        {
            row = m;
            col = n;
            rank = k;
        }
    };

    template <typename T>
    static void norm_and_copy_computational_helper(T &norm_A, CQRRPTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, CQRRPTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        RandLAPACK::util::upsize(k * k, all_data.I_ref);
        RandLAPACK::util::eye(k, k, all_data.I_ref);

        T* A_dat         = all_data.A_cpy1.data();
        T const* A_cpy_dat = all_data.A_cpy2.data();
        T const* Q_dat   = all_data.Q.data();
        T const* R_dat   = all_data.R.data();
        T* I_ref_dat     = all_data.I_ref.data();

        // Check orthogonality of Q
        // Q' * Q  - I = 0
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, I_ref_dat, k);
        T norm_0 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);

        // A - QR
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, R_dat, k, -1.0, A_dat, m);
        
        // Implementing max col norm metric
        T max_col_norm = 0.0;
        T col_norm = 0.0;
        int max_idx = 0;
        for(int i = 0; i < n; ++i) {
            col_norm = blas::nrm2(m, &A_dat[m * i], 1);
            if(max_col_norm < col_norm) {
                max_col_norm = col_norm;
                max_idx = i;
            }
        }
        T col_norm_A = blas::nrm2(n, &A_cpy_dat[m * max_idx], 1);
        T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, m);
        
        printf("REL NORM OF AP - QR:    %14e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC:    %14e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF (Q'Q - I):  %14e\n\n", norm_0 / std::sqrt((T) n));

        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_NEAR(norm_AQR / norm_A,         0.0, atol);
        ASSERT_NEAR(max_col_norm / col_norm_A, 0.0, atol);
        ASSERT_NEAR(norm_0, 0.0, atol);
    }

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRP_general(
        T d_factor, 
        T norm_A,
        CQRRPTestData<T> &all_data,
        alg_type &CQRRP,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state);
        all_data.rank = CQRRP.rank;

        RandLAPACK::util::upsize(all_data.rank * n, all_data.R);
        lapack::lacpy(MatrixType::Upper, all_data.rank, n, all_data.A.data(), m, all_data.R.data(), all_data.rank);

        lapack::ungqr(m, std::min(m, n), std::min(m, n), all_data.A.data(), m, all_data.tau.data());
        
        lapack::lacpy(MatrixType::General, m, all_data.rank, all_data.A.data(), m, all_data.Q.data(), m);

        printf("RANK AS RETURNED BY CQRRP %4ld\n", all_data.rank);

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);

        error_check(norm_A, all_data);
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRP, CQRRP_blocked_full_rank_basic) {
    int64_t m = 60;//5000;
    int64_t n = 60;//2000;
    int64_t k = 60;
    double d_factor = 1;//1.0;
    int64_t b_sz = 10;//500;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(true, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    //RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    //m_info.cond_num = 2;
    //m_info.rank = k;
    //m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
#if !defined(__APPLE__)
    test_CQRRP_general(d_factor, norm_A, all_data, CQRRP_blocked, state);
#endif
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRP, CQRRP_blocked_full_rank_block_change) {
    int64_t m = 32;//5000;
    int64_t n = 32;//2000;
    int64_t k = 32;
    double d_factor = 1;//1.0;
    int64_t b_sz = 7;//500;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(true, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    //RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    //m_info.cond_num = 2;
    //m_info.rank = k;
    //m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
#if !defined(__APPLE__)
    test_CQRRP_general(d_factor, norm_A, all_data, CQRRP_blocked, state);
#endif
}


// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRP, CQRRP_blocked_low_rank) {
    int64_t m = 5000;
    int64_t n = 2000;
    int64_t k = 1500;
    double d_factor = 2.0;
    int64_t b_sz = 256;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(true, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    //RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    //m_info.cond_num = 2;
    //m_info.rank = k;
    //m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
#if !defined(__APPLE__)
    test_CQRRP_general(d_factor, norm_A, all_data, CQRRP_blocked, state);
#endif
}


// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRP, CQRRP_sprand) {
    int64_t m = 9;
    int64_t n = 20;
    int64_t k = 0;
    double d_factor = 2.0;
    int64_t b_sz = 20;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(true, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;

    all_data.A = {-0.0827167038890927,-0.0206918849698518,-0.0259836084477108,-0.0438206513004983,-0.0343778544544599,-0.000805121862082783,0.00829667953992122,7.50061367145073e-06,0.000292651243166796,
0,0.0246199713916022,-0.0247394922338517,0.0422655798717098,0.0137021990650297,-0.000310116925136409,-0.0474561859694144,-1.72615928409158e-06,0.000107707472419336,
0,-0.0198261379990644,-0.00706539193084746,0.0815405675577816,0.0394525645899178,-6.52010266991435e-05,-0.131454334934239,3.0311453142746e-06,-0.000102413508511469,
0,0.00576512750202491,-0.00543648729501218,0.0483907241687058,0.0209624360603151,0.000107778521176888,-0.08859446923744,7.45862007014111e-06,0.000121801485614043,
0,-0.017892736836812,0.0257726395495156,0,0.0068403892363079,-6.52571101406832e-05,0.0345818020386983,-6.44764002048003e-06,-3.44003324131636e-05,
0,0,0.0163408182078844,0,0.00640266145766524,-2.82945109232014e-05,0.017588886423778,-8.22705912920404e-06,-2.27451110932393e-05,
0,0,-0.0231518612205992,0,-0.00777297708659133,-0.00237745273118921,-0.0276850905752793,-1.51775330654909e-05,0.000886171334764636,
0,0,-0.000220278518946396,0,-0.00127746791431439,-0.0803575867738797,-0.00189497325167734,2.3003242126639e-05,0.000921709468770216,
0,0,0.00400482798890193,0,0.000686364993451384,-0.0100336904044529,-0.00130329867352651,4.67757417619112e-05,0.000837582731020458,
0,0,0.000943237721400181,0,-0.000166499596848855,-0.015671608894064,0,8.43734241568531e-05,0.00187520280839173,
-0.210466060980831,-0.177541298659471,-0.0864288278396223,0.126336525283444,-0.0195133955383448,-0.0282267422608998,0.0591316561076056,0.0136482211184706,-0.00402126113511845,
-0.319819203005284,0,0.0159202071735808,0.0844855814967015,0.0312234541545924,0.0497228456223766,0.00417151803553716,-0.00014966336519997,3.41664079325782e-05,
0,0,0.0606499330826978,0.344912938918324,0.03502324642053,-0.0321473315917248,-0.00027651601433613,0.000171432175776961,-0.000604130267690244,
0,0,0.0254718356376048,-0.0704115413581921,0.00600803208631311,-0.0347730228521616,0.0152009446685347,-0.0281822263691903,-4.68523585704173e-05,
0,0,-0.109613546151845,0,-0.0145298579558023,0.171664080992915,0.00153320601514659,-0.000163679800977026,0.00093184978966319,
0,0,-0.0270681823274668,0,-0.0213599013939551,0,0.00105554375784269,0.00123568162380067,0.00149658048683344,
0,0,0,0,0.00815086997491705,0,0.000192256334156304,-0.000254488422047733,-0.000667409306888673,
0,0,0,0,-0.00462324111160224,0,-0.000172630153829677,5.01775634974473e-05,-0.000404854763904511,
0,0,0,0,0.00257465095454797,0,9.58479421600021e-05,1.97836675949495e-05,5.80262358256747e-05,
-0.0435170992627711,-0.0166575236060233,0.0114940835473573,-0.579098908415182,-0.00029326036984175,2.36399585779379e-05,2.25975864285562e-05,6.06996441734969e-05,0.000121914473450061
};

    norm_and_copy_computational_helper(norm_A, all_data);
#if !defined(__APPLE__)
    test_CQRRP_general(d_factor, norm_A, all_data, CQRRP_blocked, state);
#endif
}



// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRP, something) {
    int64_t m = 10;
    int64_t n = 5;
    auto state = RandBLAS::RNGState();

    std::vector<double> A(m * n, 0.0);
    std::vector<double> B(m * n, 0.0);
    std::vector<double> C(m * 2 * n, 0.0);
    std::vector<double> D(m * n, 0.0);
    std::vector<double> D_cpy(m * n, 0.0);
    std::vector<double> D_space(m * n, 0.0);

    std::vector<double> tau(n * 2, 0.0);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, A.data(), state);
    RandLAPACK::gen::mat_gen(m_info, B.data(), state);
    RandLAPACK::gen::mat_gen(m_info, D.data(), state);
    lapack::lacpy(MatrixType::General, m, n, D.data(), m, D_cpy.data(), m);

    lapack::geqrf(m, n, A.data(), m, tau.data());
    lapack::geqrf(m, n, B.data(), m, tau.data() + n);

    // Method 1
    lapack::lacpy(MatrixType::Lower, m, n, A.data(), m, C.data(), m);
    lapack::lacpy(MatrixType::Lower, m, n, B.data(), m, C.data() + (m * n), m);
    lapack::ormqr(Side::Left, Op::NoTrans, m, n, m, C.data(), m, tau.data(), D.data(), m);

    char name [] = "D through ormqr";
    RandBLAS::util::print_colmaj(m, n, D.data(), name);

    // Method 2
    lapack::ungqr(m, n, n, A.data(), m, tau.data());
    lapack::ungqr(m, n, n, B.data(), m, tau.data() + n);

    lapack::lacpy(MatrixType::General, m, n, A.data(), m, C.data(), m);
    lapack::lacpy(MatrixType::General, m, n, B.data(), m, C.data() + (m * n), m);

    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, m, n, m, 1.0, C.data(), m, D_cpy.data(), m, 0.0, D_space.data(), m);

    char name1 [] = "D through gemm";
    RandBLAS::util::print_colmaj(m, n, D_space.data(), name1);
}
