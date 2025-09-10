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

template <typename T>
void get_spectrum( 
    int64_t m,
    int64_t n,
    RandLAPACK::gen::mat_gen_info<T> m_info,
    std::string path       
) {
    double* A     = new double[m * n]();
    double* U     = new double[m * n]();
    double* VT    = new double[m * n]();
    double* Sigma = new double[m]();
    
    auto state    = RandBLAS::RNGState<r123::Philox4x32>();
    RandLAPACK::gen::mat_gen(m_info, A, state);

    char joba = 'C'; 
    char jobu = 'N';
    char jobv = 'N';
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

    std::ofstream file(path, std::ios::out | std::ios::app);

    /*
    file << "Description: Spectrum of the matrix of a given type generated in RandLAPACK found via Jacobi SVD"
    "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
    "\nInput type:"       + std::to_string(m_info.m_type) +
    "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
    "\n";
    file.flush();
    */

    for (int i = 0; i < n; ++i){
        file << Sigma[i] << ",  ";
    }
    file  << "\n";

    free(A);
    free(U);
    free(VT);
    free(Sigma);
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <num_rows> <num_cols>" << std::endl;
        return 1;
    }

    int64_t m = std::stol(argv[2]);
    int64_t n = std::stol(argv[3]);

    // Set the input matrices
    // Polynomial matrix
    RandLAPACK::gen::mat_gen_info<double> m_info_poly(m, n, RandLAPACK::gen::spiked);
    m_info_poly.cond_num = std::pow(10, 10);
    m_info_poly.exponent = 2.0;
    // Matrix with staircase spectrum
    RandLAPACK::gen::mat_gen_info<double> m_info_stair(m, n, RandLAPACK::gen::spiked);
    m_info_stair.cond_num = std::pow(10, 10);
    // Matrix with spiked spectrum
    RandLAPACK::gen::mat_gen_info<double> m_info_spiked(m, n, RandLAPACK::gen::spiked);
    m_info_spiked.scaling = std::pow(10, 10);
    // Kahan matrix 
    RandLAPACK::gen::mat_gen_info<double> m_info_kahan(m, n, RandLAPACK::gen::spiked);
    m_info_kahan.theta   = 1.2;
    m_info_kahan.perturb = 1e3;

    std::string output_filename1 = "_poly_spectrum_num_info_lines_" + std::to_string(4) + ".txt";
    std::string output_filename2 = "_stair_spectrum_num_info_lines_" + std::to_string(4) + ".txt";
    std::string output_filename3 = "_spike_spectrum_num_info_lines_" + std::to_string(4) + ".txt";
    std::string output_filename4 = "_kahan_spectrum_num_info_lines_" + std::to_string(4) + ".txt";

    std::string path1;
    std::string path2;
    std::string path3;
    std::string path4;
    if (std::string(argv[1]) != ".") {
        path1 = std::string(argv[1]) + output_filename1;
        path2 = std::string(argv[1]) + output_filename2;
        path3 = std::string(argv[1]) + output_filename3;
        path4 = std::string(argv[1]) + output_filename4;
    } else {
        path1 = output_filename1;
        path2 = output_filename2;
        path3 = output_filename3;
        path4 = output_filename4;
    }

    get_spectrum(m, n, m_info_poly,   path1); 
    get_spectrum(m, n, m_info_stair,  path2); 
    get_spectrum(m, n, m_info_spiked, path3); 
    get_spectrum(m, n, m_info_kahan,  path4); 
}
