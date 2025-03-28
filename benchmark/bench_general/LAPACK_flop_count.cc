#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <chrono>
#include <climits> 
/*
Auxillary benchmark routine, computes flops using GEQRF for a given system
*/

using namespace std::chrono;
using namespace RandLAPACK;

template <typename T, typename RNG>
static void 
geqrf_flops(
    int64_t rows, 
    int64_t cols, 
    int64_t numruns,
    RandBLAS::RNGState<RNG> state) {

    // Sourced from LAWN (LAPACK Work Notes) 41
    T flop_count = 0;
    if (rows >= cols) {
        flop_count = 2. * rows * std::pow(cols, 2) - (2./3.) * std::pow(cols, 3) + rows * cols + std::pow(cols, 2) + (14./3.) * cols;
    } else {
        flop_count = 2. * cols * std::pow(rows, 2) - (2./3.) * std::pow(rows, 3) + 3. * rows * cols - std::pow(rows, 2) + (14./3.) * cols;
    }
    printf("%f\n", flop_count);

    long dur_microsec_best = LONG_MAX;

    T* A   = new T[rows * cols]();
    T* tau = new T[cols]();

    RandLAPACK::gen::mat_gen_info<double> m_info(rows, cols, RandLAPACK::gen::gaussian);  

    for (int i = 0; i < numruns; ++i) {
        RandLAPACK::gen::mat_gen(m_info, A, state);

        // Get the timing
        auto start = steady_clock::now();
        lapack::geqrf(rows, cols, A, rows, tau);
        auto stop = steady_clock::now();
        long dur  = duration_cast<microseconds>(stop - start).count();

        if(dur < dur_microsec_best)
            dur_microsec_best = dur;
    }

    T dur_s_best = dur_microsec_best / 1e+6;
    T GFLOP_rate_best = (flop_count / dur_s_best) / 1e+9;

    printf("THE SYSTEM IS CAPABLE OF %f GFLOPs/sec RUNNING GEQRF.\n", GFLOP_rate_best);

    delete[] A;
    delete[] tau;
}

template <typename T, typename RNG>
static void 
getrf_flops(
    int64_t rows, 
    int64_t cols, 
    int64_t numruns,
    RandBLAS::RNGState<RNG> state) {

    // Sourced from LAWN (LAPACK Work Notes) 41
    T flop_count = rows * std::pow(cols, 2) - (1./3.) * std::pow(cols, 3) - (1./2.) * std::pow(cols, 2) + (5./6.) * cols;

    long dur_microsec_best = LONG_MAX;

    T* A          = new T[rows * cols]();
    int64_t* ipiv = new int64_t[cols]();

    RandLAPACK::gen::mat_gen_info<double> m_info(rows, cols, RandLAPACK::gen::gaussian);  

    for (int i = 0; i < numruns; ++i) {
        RandLAPACK::gen::mat_gen(m_info, A, state);

        // Get the timing
        auto start = steady_clock::now();
        lapack::getrf(rows, cols, A, rows, ipiv);
        auto stop = steady_clock::now();
        long dur  = duration_cast<microseconds>(stop - start).count();

        if(dur < dur_microsec_best)
            dur_microsec_best = dur;
    }

    T dur_s_best = dur_microsec_best / 1e+6;
    T GFLOP_rate_best = (flop_count / dur_s_best) / 1e+9;

    printf("THE SYSTEM IS CAPABLE OF %f GFLOPs/sec RUNNING GETRF.\n", GFLOP_rate_best);

    delete[] A;
    delete[] ipiv;
}

template <typename T, typename RNG>
static void 
potrf_flops(
    int64_t dim, 
    int64_t numruns,
    RandBLAS::RNGState<RNG> state) {

    // Sourced from LAWN (LAPACK Work Notes) 41
    T flop_count = (1./3.) * std::pow(dim, 3) + (1./2.) * std::pow(dim, 2) + (1./6.) * dim;

    long dur_microsec_best = LONG_MAX;

    T* A   = new T[dim * dim]();

    RandLAPACK::gen::mat_gen_info<double> m_info(dim, dim, RandLAPACK::gen::gaussian);  

    for (int i = 0; i < numruns; ++i) {
        RandLAPACK::gen::mat_gen(m_info, A, state);

        // Get the timing
        auto start = steady_clock::now();
        lapack::potrf(Uplo::Upper, dim, A, dim);
        auto stop = steady_clock::now();
        long dur  = duration_cast<microseconds>(stop - start).count();

        if(dur < dur_microsec_best)
            dur_microsec_best = dur;
    }

    T dur_s_best = dur_microsec_best / 1e+6;
    T GFLOP_rate_best = (flop_count / dur_s_best) / 1e+9;

    printf("THE SYSTEM IS CAPABLE OF %f GFLOPs/sec RUNNING POTRF.\n", GFLOP_rate_best);

    delete[] A;
}

int main(int argc, char *argv[]) {

    if(argc < 4) {
        throw std::runtime_error(
            std::string("Improper input provided.\n Please, specify the name of the function to be tested,")
            + "the size of the input matrix and the number of consecutive runs of the given algorithm to be performed.\n"
            + "Example input: GEQRF 1000 1000 20\n");
    }
    std::string fname = argv[1];
    auto rows         = std::stol(argv[2]);
    auto cols         = std::stol(argv[3]);
    auto numruns      = std::stol(argv[4]);

    auto state = RandBLAS::RNGState();

    if(fname == "geqrf" || fname == "GEQRF" || fname == "qrf") {
        geqrf_flops<double>(rows, cols, numruns, state);
    } else if(fname == "getrf" || fname == "GETRF" || fname == "trf") {
        getrf_flops<double>(rows, cols, numruns, state);
    } else if(fname == "potrf" || fname == "POTRF") {
        if (rows != cols) {
            printf("Cholesky factorization required a square input. \n Using the smaller dimension provided.");
        }
        potrf_flops<double>(std::min(rows, cols), numruns, state);
    } else {
        throw std::runtime_error("Invalid LAPACK function name.");
    }
    return 0;
}