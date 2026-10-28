#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_bench_utils.hh"

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
    int64_t* iwork = new int64_t[8 * std::min(m, n)]();
    int64_t info[1];

    RandLAPACK::bench::gejsv(
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

    file << "Description: Spectrum of the matrix of a given type generated in RandLAPACK found via Jacobi SVD"
    "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
    "\nInput type:"       + std::to_string(m_info.m_type) +
    "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
    "\n";
    file.flush();

    for (int i = 0; i < n; ++i){
        file << Sigma[i] << ",  ";
    }
    file  << "\n";

    delete[] buff_workspace;
    delete[] iwork;
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

    // Set the input matrices: polynomial, staircase, spiked and Kahan
    auto tests_info = RandLAPACK::bench::qr_test_matrices<double>(m, n);

    std::string output_filename1 = "_poly_spectrum_num_info_lines_" + std::to_string(4) + ".txt";
    std::string output_filename2 = "_stair_spectrum_num_info_lines_" + std::to_string(4) + ".txt";
    std::string output_filename3 = "_spike_spectrum_num_info_lines_" + std::to_string(4) + ".txt";
    std::string output_filename4 = "_kahan_spectrum_num_info_lines_" + std::to_string(4) + ".txt";

    std::string path1 = RandLAPACK::bench::output_path(argv[1], output_filename1);
    std::string path2 = RandLAPACK::bench::output_path(argv[1], output_filename2);
    std::string path3 = RandLAPACK::bench::output_path(argv[1], output_filename3);
    std::string path4 = RandLAPACK::bench::output_path(argv[1], output_filename4);

    get_spectrum(m, n, tests_info[0], path1);
    std::cout << "Done with the polynomial matrix\n";
    get_spectrum(m, n, tests_info[1], path2);
    std::cout << "Done with the staircase matrix\n";
    get_spectrum(m, n, tests_info[2], path3);
    std::cout << "Done with the spiked matrix\n";
    get_spectrum(m, n, tests_info[3], path4);
    std::cout << "Done with the kahan matrix\n";
}
