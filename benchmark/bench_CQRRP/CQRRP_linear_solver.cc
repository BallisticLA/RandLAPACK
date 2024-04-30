/*
ICQRRP linear system solver benchmark - runs:
    1. GELS
    2. ICQRRP (single) + iterative refinement
for a matrix with fixed number of rows and columns and a varying ICQRRP block size.
Records the best timing, saves that into a file.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct QR_solver_benchmark_data {
    int64_t row;
    int64_t col;
    float       tolerance;
    T sampling_factor;
    std::vector<T> A;
    std::vector<float> A_single;
    std::vector<float> tau;
    std::vector<int64_t> J;
    std::vector<T> b;
    std::vector<T> x;
    std::vector<float> work;
    std::vector<T> work_double;

    QR_solver_benchmark_data(int64_t m, int64_t n, float tol, T d_factor) :
    A(m * n, 0.0),
    A_single(m * n, 0.0),
    tau(n, 0.0),
    J(n, 0),
    b(m, 0),
    x(n, 0),
    work(m, 0),
    work_double(m, 0)
    {
        row             = m;
        col             = n;
        tolerance       = tol;
        sampling_factor = d_factor;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_solver_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int apply_itoa) {

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    if (apply_itoa) {
        std::iota(all_data.J.begin(), all_data.J.end(), 1);
    } else {
        std::fill(all_data.J.begin(), all_data.J.end(), 0);
    }
}


// Re-generate and clear data
template <typename T, typename RNG>
static long ICQRRP_refinement(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_solver_benchmark_data<T> &all_data,
                                        int64_t b_sz, 
                                        RandBLAS::RNGState<RNG> state) {
    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Set up a single-precision ICQRRP
    RandLAPACK::CQRRP_blocked<float, r123::Philox4x32> CQRRP_blocked(false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 8;

    T*       A           = all_data.A.data();           // double
    float*   A_single    = all_data.A_single.data();    // single
    float*   tau         = all_data.tau.data();         // single
    int64_t* J           = all_data.J.data();           // single
    T*       b           = all_data.b.data();           // double
    T*       x           = all_data.x.data();           // double
    float*   work        = all_data.work.data();        // single 
    T*       work_double = all_data.work_double.data(); // double

    long dur_cqrrp_refine = 0;

    auto start_cqrrp_refine = high_resolution_clock::now();
    // Cast the input matrix A down to single precision.
    // Copy columns in parallel.
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        std::transform(&A[m * i], &A[m * i + m], &A_single[m * i], [](double d) { return static_cast<float>(d); });

    // Call single-precision ICQRRP
    CQRRP_blocked.call(m, n, A_single, m, d_factor, tau, J, state);
    // Vector all_data.x has already been initialized to all 0;
    // repeat below until some tolerance threshold is reached:
    while (1) {
        // 1. Solve r = b - Ax for r in double precision.
        // Need to make sure that work = b before this.
        // After this, r will be stored in work (m by 1).
        std::transform(b, &b[m], work, [](double d) { return static_cast<float>(d); });
        blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, n, A, m, x, 1, -1.0, work, 1);
        // 2. Solve Qy = Pr for y in single precision.
        // Since Q' = inv(Q); y = Q'Pr.
        // After this, y will be stored in work (m by 1).
        lapack::laswp(1, work, m, 1, n, J, 1);
        lapack::ormqr(Side::Left, Op::Trans, m, 1, n, A_single, m, tau, work, m);
        // 3. Solve Rz = y for z in single precision.
        // A_single stores single-precision R.
        // After this, z will be stored in work (n by 1).
        blas::trmv(Layout::ColMajor, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, A_single, m, work, 1);	
        // 4. Check if ||z|| <= tol.
        if (blas::nrm2(n, work, 1) <= tol)
            break;
        // 5. Transform work into a double-precision array
        std::transform(work, &work[n], work_double, [](float f) { return static_cast<double>(f); });
        // 6. x = x + z.
        blas::axpy(n, 1.0, x, 1, work, 1);	
    }
    auto stop_cqrrp_refine = high_resolution_clock::now();
    dur_cqrrp_refine = duration_cast<microseconds>(stop_cqrrp_refine - start_cqrrp_refine).count();

    return dur_cqrrp_refine;
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    QR_solver_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // timing vars
    long dur_gels         = 0;
    long dur_cqrrp_refine = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d\n", i);
        // Testing GELS
        auto start_gels = high_resolution_clock::now();
        lapack::gels(Op::NoTrans, m, n, 1, all_data.A.data(), m, all_data.b.data(), m);	
        auto stop_gels = high_resolution_clock::now();
        dur_gels = duration_cast<microseconds>(stop_gels - start_gels).count();
        printf("TOTAL TIME FOR GELS %ld\n", dur_gels);

        data_regen(m_info, all_data, state_gen, 0);
        state_gen = state;

        // Testing ICQRRP + iterative refinement
        dur_cqrrp_refine = ICQRRP_refinement( m_info, all_data, b_sz, state);
        printf("TOTAL TIME FOR ICQRRP + refinement %ld\n", dur_cqrrp_refine);

        data_regen(m_info, all_data, state_gen, 0);
        state_gen = state;

        std::ofstream file(output_filename, std::ios::app);
        file << m << ",  " << b_sz <<  ",  " << dur_gels << ",  " << dur_cqrrp_refine << ",\n";
    }
}

int main() {
    // Declare parameters
    int64_t m          = 100;
    int64_t n          = 50;
    double d_factor   = 1.25;
    int64_t b_sz_start = 10;
    int64_t b_sz_end   = 10;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    QR_solver_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "ICQRRP_solver_time_raw_rows_"   + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz_start)
                                    + "_b_sz_end_"     + std::to_string(b_sz_end)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat";
#if !defined(__APPLE__)
    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        call_all_algs(m_info, numruns, b_sz_start, all_data, state_constant, output_filename);
    }
#endif
}