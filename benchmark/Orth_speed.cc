#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <fstream>
/*
Note: this benchmark attempts to save files into a specific location.
If the required folder structure does not exist, the files will not be saved.
*/
/*
TODO #1: Switch tuples to vectors.
*/


using namespace std::chrono;


#if !defined(__APPLE__)
template <typename T>
class GEQR : public RandLAPACK::Stabilization<T> {
    public:
        std::vector<T> tvec;
        bool cond_check;
        bool verbosity;

        // Constructor
        GEQR(bool c_check, bool verb) {
            cond_check = c_check;
            verbosity = verb;
        };

        int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        );
};

// -----------------------------------------------------------------------------
/// Performs a QR factorization. Outputs the implicitly-stored Q and R factors.
/// This routine is only defined in Intel MKL.
///
/// Templated for `float` and `double` types.
///
/// @param[in] m
///     The number of rows in the matrix A.
///
/// @param[in] n
///     The number of columns in the matrix A.
///
/// @param[in] A
///     The m-by-n matrix, stored in a column-major format.
///
/// @param[in] tau
///     Buffer for the scalar factor array.
///     
/// @param[out] A
///     Lower-triangular portion represents householder reflectors. 
///     Upper- stores the R-factor. 
///
/// @param[out] tau.
///     Array of length n.
///
/// @return = 0: successful exit
///
template <typename T> 
int GEQR<T>::call(
    int64_t m,
    int64_t n,
    std::vector<T>& A
){
    auto tvec = this->tvec;
    tvec.resize(5);

    T* A_dat = A.data();

    lapack::geqr(m, n, A_dat, m, tvec.data(), -1);
    int64_t tsize = (int64_t) tvec[0]; 
    tvec.resize(tsize);
    if(lapack::geqr(m, n, A_dat, m, tvec.data(), tsize))
        return 1;

    return 0;
}
#endif

template <typename T, typename RNG>
static std::tuple<long, long, long, long> 
test_speed_helper(int64_t m, int64_t n, RandBLAS::base::RNGState<RNG> state) {

    int64_t size = m * n;
    std::vector<T> A(size, 0.0);
    std::vector<T> A_cpy(size, 0.0);
    std::vector<T> A_cpy_2(size, 0.0);
    std::vector<T> A_cpy_3(size, 0.0);
    
    T* A_dat = A.data();
    T* A_cpy_dat = A_cpy.data();
    T* A_cpy_2_dat = A_cpy_2.data();
    T* A_cpy_3_dat = A_cpy_3.data();

    // Random Gaussian test matrix
    RandLAPACK::util::gen_mat_type(m, n, A, n, state, std::tuple(6, 0., false));
    // Make a copy
    std::copy(A_dat, A_dat + size, A_cpy_dat);
    std::copy(A_dat, A_dat + size, A_cpy_2_dat);
    std::copy(A_dat, A_dat + size, A_cpy_3_dat);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Stabilization Constructor
    RandLAPACK::PLUL<T> Stab_PLU(false, false);
    RandLAPACK::CholQRQ<T> Orth_CholQR(false, false);
    RandLAPACK::HQRQ<T> Orth_HQR(false, false);
#if !defined(__APPLE__)
    GEQR<T> Orth_GEQR(false, false);
#endif
    // PIV LU
    // Stores L, U into Omega
    auto start_lu = high_resolution_clock::now();
    Stab_PLU.call(m, n, A_cpy);
    auto stop_lu = high_resolution_clock::now();
    long dur_lu = duration_cast<microseconds>(stop_lu - start_lu).count();

    // HQR
    auto start_qr = high_resolution_clock::now();
    Orth_HQR.call(m, n, A_cpy_2);
    auto stop_qr = high_resolution_clock::now();
    long dur_qr = duration_cast<microseconds>(stop_qr - start_qr).count();

    // CHOL QR
    // Orthonormalize A
    auto start_chol = high_resolution_clock::now();
    Orth_CholQR.call(m, n, A);
    auto stop_chol = high_resolution_clock::now();
    long dur_chol = duration_cast<microseconds>(stop_chol - start_chol).count();

    // GEQR
    auto start_geqr = high_resolution_clock::now();
#if !defined(__APPLE__)
    Orth_GEQR.call(m, n, A);
#endif
    auto stop_geqr = high_resolution_clock::now();
    long dur_geqr = duration_cast<microseconds>(stop_geqr - start_geqr).count();

    return std::make_tuple(dur_chol, dur_lu, dur_qr, dur_geqr);
}

template <typename T, typename RNG>
static void 
test_speed(int r_pow, int r_pow_max, int c_pow, int c_pow_max, int runs, RandBLAS::base::RNGState<RNG> state) {
    int64_t rows = 0;
    int64_t cols = 0;

    T chol_avg = 0;
    T lu_avg = 0;
    T qr_avg = 0;
    T geqr_avg = 0;

    for(; r_pow <= r_pow_max; ++r_pow) {
        rows = std::pow(2, r_pow);
        int c_buf = c_pow;

        for (; c_buf <= c_pow_max; ++c_buf) {
            cols = std::pow(2, c_buf);

            std::tuple<long, long, long, long> res;
            long t_chol = 0;
            long t_lu   = 0;
            long t_qr   = 0;
            long t_geqr = 0;

            long curr_t_chol = 0;
            long curr_t_lu   = 0;
            long curr_t_qr   = 0;
            long curr_t_geqr = 0;

            std::ofstream file("../../build/test_plots/test_speed/raw_data/test_" + std::to_string(rows) + "_" + std::to_string(cols) + ".dat");
            for(int i = 0; i < runs; ++i) {
                res = test_speed_helper<T>(rows, cols, state);
                curr_t_chol = std::get<0>(res);
                curr_t_lu   = std::get<1>(res);
                curr_t_qr   = std::get<2>(res);
                curr_t_geqr = std::get<3>(res);

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0) {
                    // Save the output into .dat file
                    file << curr_t_chol << "  " << curr_t_lu << "  " << curr_t_qr << "  " << curr_t_geqr << "\n";
            
                    t_chol += curr_t_chol;
                    t_lu   += curr_t_lu;
                    t_qr   += curr_t_qr;
                    t_geqr += curr_t_geqr;
                }
            }

            chol_avg = (T)t_chol / (T)(runs - 1);
            lu_avg   = (T)t_lu   / (T)(runs - 1);
            qr_avg   = (T)t_qr   / (T)(runs - 1);
            geqr_avg   = (T)t_geqr   / (T)(runs - 1);

            printf("\nMatrix size: %ld by %ld.\n", rows, cols);
            printf("Average timing of Chol QR for %d runs: %f μs.\n", runs, chol_avg);
            printf("Average timing of Pivoted LU for %d runs: %f μs.\n", runs, lu_avg);
            printf("Average timing of Householder QR for %d runs: %f μs.\n", runs, qr_avg);
            printf("Average timing of GEQR for %d runs: %f μs.\n", runs, geqr_avg);
            printf("\nResult: cholQR is %f times faster then HQR, %f times faster then GEQR and %f times faster then PLU.\n", qr_avg / chol_avg, geqr_avg / chol_avg, lu_avg / chol_avg);
        }
    }
}

template <typename T, typename RNG>
static void 
test_speed_mean(int r_pow, int r_pow_max, int col, int col_max, int runs, RandBLAS::base::RNGState<RNG> state)
{

    // Clear all files
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf) {
        //int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        //ofs.open("../../build/test_plots/test_speed/raw_data/test_mean_time_QR_" + std::to_string(rows) + ".dat", std::ofstream::out | std::ofstream::trunc);
        //ofs.close();
    }

    int64_t rows = 0;

    T chol_avg = 0;
    T lu_avg = 0;
    T qr_avg = 0;
    T geqr_avg = 0;

    for(; r_pow <= r_pow_max; ++r_pow) {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols += 64) {
            std::tuple<long, long, long, long> res;
            long t_chol = 0;
            long t_lu   = 0;
            long t_qr   = 0;
            long t_geqr = 0;

            long curr_t_chol = 0;
            long curr_t_lu   = 0;
            long curr_t_qr   = 0;
            long curr_t_geqr = 0;

            for(int i = 0; i < runs; ++i) {
                res = test_speed_helper<T>(rows, cols, state);
                curr_t_chol = std::get<0>(res);
                curr_t_lu   = std::get<1>(res);
                curr_t_qr   = std::get<2>(res);
                curr_t_geqr = std::get<3>(res);

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0) {
                    t_chol += curr_t_chol;
                    t_lu   += curr_t_lu;
                    t_qr   += curr_t_qr;
                    t_geqr += curr_t_geqr;
                }
            }

            chol_avg = (T)t_chol / (T)(runs - 1);
            lu_avg   = (T)t_lu   / (T)(runs - 1);
            qr_avg   = (T)t_qr   / (T)(runs - 1);
            geqr_avg   = (T)t_geqr   / (T)(runs - 1);

            // Save the output into .dat file
            //std::ofstream file("../../build/test_plots/test_speed/raw_data/test_mean_time_" + std::to_string(rows) + ".dat");
            //std::fstream file;
            //file.open("../../build/test_plots/test_speed/raw_data/test_mean_time_QR_" + std::to_string(rows) + ".dat", std::fstream::app);
            //file << chol_avg << "  " << lu_avg << "  " << qr_avg << "  " << geqr_avg << "\n";

            printf("\nMatrix size: %ld by %ld.\n", rows, cols);
            printf("Average timing of Chol QR for %d runs: %f μs.\n", runs, chol_avg);
            printf("Average timing of Pivoted LU for %d runs: %f μs.\n", runs, lu_avg);
            printf("Average timing of Householder QR for %d runs: %f μs.\n", runs, qr_avg);
            printf("Average timing of GEQR for %d runs: %f μs.\n", runs, geqr_avg);
            printf("\nResult: cholQR is %f times faster then HQR, %f times faster then GEQR and %f times faster then PLU.\n", qr_avg / chol_avg, geqr_avg / chol_avg, lu_avg / chol_avg);
        }
    }
}

int main() {
    auto state = RandBLAS::base::RNGState(0, 0);
    test_speed_mean<double, r123::Philox4x32>(12, 12, 64, 64, 3, state);
    return 0;
}
