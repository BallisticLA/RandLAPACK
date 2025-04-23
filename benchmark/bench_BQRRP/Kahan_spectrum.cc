#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <chrono>
#include <climits> 
/*
Auxillary benchmark routine, finds the spectrum of RandLAPACK-generated Kahan matrix via Jacobi SVD (accuratemost version of SVD).
*/

using namespace std::chrono;
using namespace RandLAPACK;

template <typename T>
void _LAPACK_gejsv(
    char joba, char jobu, char jobv, char jobr,
    char jobt, char jobp,
    int64_t m, int64_t n,
    T *A, int64_t lda,
    T *S,
    T *U, int64_t ldu,
    T *V, int64_t ldv,
    T* work, int64_t* lwork,
    int64_t* iwork,
    int64_t* info
){

    char joba_ = joba; //lapack::to_char( joba );
    char jobu_ = jobu; //lapack::to_char( jobu );
    char jobv_ = jobv; //lapack::to_char( jobv );
    char jobr_ = jobr; //lapack::to_char( jobr );
    char jobt_ = jobt; //lapack::to_char( jobt );;
    char jobp_ = jobp; //lapack::to_char( jobp );

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

    if (argc < 3) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <num_rows> <num_cols>" << std::endl;
        return 1;
    }

    int64_t m = std::stol(argv[2]);
    int64_t n = std::stol(argv[3]);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::kahan);
    m_info.theta   = 1.2;
    m_info.perturb = 1e3;

    double* A     = new double[m * n]();
    double* U     = new double[m * n]();
    double* VT    = new double[m * n]();
    double* Sigma = new double[m]();
    double* work  = new double[m * n]();
    auto state    = RandBLAS::RNGState<r123::Philox4x32>();
    RandLAPACK::gen::mat_gen(m_info, A, state);

    char joba = 'C'; 
    char jobu = 'U';
    char jobv = 'V';
    char jobr = 'N';
    char jobt = 'N';
    char jobp = 'N';
    
    double* buff_workspace  = new double[8 * m * n]();
    int64_t lwork[1]; 
    lwork[0] = 8 * m * n;
    int64_t iwork[8 * std::min(m,n)];
    int64_t info[1];

    _LAPACK_gejsv(
        joba, jobu, jobv, jobr,
        jobt, jobp,
        m, n,
        A, m,
        Sigma,
        U, m,
        VT, m,
        buff_workspace, lwork,
        iwork,
        info
    );

    std::string output_filename = "_BQRRP_runtime_breakdown_num_info_lines_" + std::to_string(7) + ".txt";
    std::string path;
    if (std::string(argv[1]) != ".") {
        path = std::string(argv[1]) + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    file << "Description: Spectrum of the Kahan matrix (generated in RandLAPACK) found via Jacobi SVD"
    "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
    "\nInput type:"       + std::to_string(m_info.m_type) +
    "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
    "\n";
    file.flush();

    for (int i = 0; i < n; ++i){
        file << Sigma[i] << ",  ";
    }
    file  << "\n";
}
