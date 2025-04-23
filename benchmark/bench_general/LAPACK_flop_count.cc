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
/*
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

*/


template <typename T>
void _LAPACK_gejsv(
    lapack::Job joba, lapack::Job jobu, lapack::Job jobv, lapack::Job jobr,
    lapack::Job jobt, lapack::Job jobp,
    int64_t m, int64_t n,
    T *A, int64_t lda,
    T *S,
    T *U, int64_t ldu,
    T *V, int64_t ldv,
    T* work, int64_t* lwork,
    int64_t* iwork,
    int64_t* info
){

    char joba_ = lapack::to_char( joba );
    char jobu_ = lapack::to_char( jobu );
    char jobv_ = lapack::to_char( jobv );
    char jobr_ = lapack::to_char( jobr );
    char jobt_ = lapack::to_char( jobt );;
    char jobp_ = lapack::to_char( jobp );

    lapack_int m_   = (lapack_int) m;
    lapack_int n_   = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldv_ = (lapack_int) ldv;
    
    lapack_int *lwork_ = (lapack_int *) lwork;
    lapack_int *iwork_ = (lapack_int *) iwork;
    lapack_int *info_  = (lapack_int *) info;

    LAPACK_dgejsv( & joba_, & jobu_, & jobv_, & jobr_,
        & jobt_, & jobp_,
        & m_, & n_,
        A, & lda_,
        S,
        U, & ldu_,
        V, & ldv_,
        work, lwork_,
        iwork_,
        info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        //, 1, 1, 1, 1, 1, 1
        #endif
        );

    return;
}

int main(int argc, char *argv[]) {

    int64_t m = 16384;
    int64_t n = 16384;
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::kahan);
    m_info.theta   = 1.2;
    m_info.perturb = 1e3;

    double* A = new double[m * n]();
    double* U = new double[m * n]();
    double* VT = new double[m * n]();
    double* Sigma = new double[m]();
    double* work = new double[m * n]();
    auto state = RandBLAS::RNGState<r123::Philox4x32>();
    RandLAPACK::gen::mat_gen(m_info, A, state);

    //lapack::gesvd(Job::SomeVec, Job::SomeVec, m, n, A, m, Sigma, U, m, VT, n);

    int64_t info[1];
    double work_query[1];
    int64_t lwork[1];
    int64_t iwork_qry[1];
    lwork[0] = -1;

    _LAPACK_gejsv(
        Job::AllVec, Job::AllVec, Job::AllVec, Job::AllVec,
        Job::AllVec, Job::AllVec,
        m, n,
        A, m,
        Sigma,
        U, m,
        VT, m,
        work_query, lwork,
        iwork_qry,
        info
    );

    lwork[0] = std::max((int64_t) blas::real(work_query[0]), n);
    double* buff_workspace  = new double[lwork[0]]();
    int64_t iwork[8 * std::min(m,n)];

    _LAPACK_gejsv(
        Job::AllVec, Job::AllVec, Job::AllVec, Job::AllVec,
        Job::AllVec, Job::AllVec,
        m, n,
        A, m,
        Sigma,
        U, m,
        VT, m,
        buff_workspace, lwork,
        iwork,
        info
    );

    std::ofstream file("Kahan_spectrum.txt", std::ios::out | std::ios::app);
    for (int i = 0; i < n; ++i){
        file << Sigma[i] << ",  ";
    }
    file  << "\n";
}
