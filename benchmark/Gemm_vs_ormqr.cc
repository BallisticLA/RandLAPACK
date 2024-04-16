#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <chrono>
/*
Auxillary benchmark routine, computes flops using GEMM for a given system
*/

using namespace std::chrono;
using namespace RandLAPACK;

template <typename T>
static void 
test_speed(int64_t m, int64_t n, int64_t runs, RandBLAS::RNGState<> const_state) {

    // Matrix to decompose.
    std::vector<T> A(m * n, 0.0);
    // Matrix to apply the Q-factor to.
    std::vector<T> B1(m * n, 0.0);
    std::vector<T> B2(m * n, 0.0);
    std::vector<T> Product(n * n, 0.0);
    std::vector<T> tau(n, 0.0);

    T* A_dat = A.data();
    T* B1_dat = B1.data();
    T* B2_dat = B2.data();
    T* Product_dat = Product.data();
    T* tau_dat = tau.data();

    T mean_gflop_rate_gemm  = 0;
    T mean_gflop_rate_ormqr = 0;

    for (int i = 0; i < runs; ++i) {
        auto state = const_state;

        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
        RandLAPACK::gen::mat_gen(m_info, A, state);
        RandLAPACK::gen::mat_gen(m_info, B1, state);
        lapack::lacpy(MatrixType::General, m, n, B1_dat, m, B2_dat, m);

        // Get the implicit Q-factor in A_dat
        lapack::geqrf(m, n, A_dat, m, tau_dat);

        auto start_ormqr = high_resolution_clock::now();
        lapack::ormqr(Side::Left, Op::Trans, m, n, n, A_dat, m, tau_dat, B1_dat, m);
        auto stop_ormqr = high_resolution_clock::now();
        long dur_ormqr = duration_cast<microseconds>(stop_ormqr - start_ormqr).count();

        auto start_gemm = high_resolution_clock::now();
        lapack::ungqr(m, n, n, A_dat, m, tau_dat);
        gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A_dat, m, B2_dat, m, 0.0, Product_dat, n);
        auto stop_gemm = high_resolution_clock::now();
        long dur_gemm = duration_cast<microseconds>(stop_gemm - start_gemm).count();

        T gflop_count_gemm     = (2 * std::pow(n, 2) * m) / std::pow(10, 9);
        if (i != 0) {
            mean_gflop_rate_gemm  += gflop_count_gemm / dur_gemm;
            mean_gflop_rate_ormqr += gflop_count_gemm / dur_ormqr;
        }
    }

    printf("%f  %f\n", mean_gflop_rate_gemm / (runs - 1), mean_gflop_rate_ormqr / (runs - 1));
}

int main() {
    auto state = RandBLAS::RNGState();
    test_speed<double>(std::pow(2, 10), std::pow(2, 5),  10, state);
    test_speed<double>(std::pow(2, 11), std::pow(2, 6),  10, state);
    test_speed<double>(std::pow(2, 12), std::pow(2, 7),  10, state);
    test_speed<double>(std::pow(2, 13), std::pow(2, 8),  10, state);
    test_speed<double>(std::pow(2, 14), std::pow(2, 9),  10, state);
    test_speed<double>(std::pow(2, 15), std::pow(2, 10), 10, state);
    return 0;
}
