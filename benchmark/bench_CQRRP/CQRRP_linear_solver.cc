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
    T       tol_d;
    float   tol_s;
    T sampling_factor;
    std::vector<T> A;
    std::vector<float> A_single;
    std::vector<float> tau;
    std::vector<int64_t> J;
    std::vector<T> b;
    std::vector<T> x;
    std::vector<float> work;
    std::vector<T> work_double;
    std::vector<T> x_solution;
    std::vector<T> b_constant;
    std::vector<float> R_buffer;

    QR_solver_benchmark_data(int64_t m, int64_t n, T tolerance_d, float tolerance_s, T d_factor) :
    A(m * n, 0.0),
    A_single(m * n, 0.0),
    tau(n, 0.0),
    J(n, 0),
    b(m, 0),
    x(n, 0),
    work(m, 0),
    work_double(m, 0),
    x_solution(n, 0),
    b_constant(m, 0),
    R_buffer(n * n, 0)
    {
        row             = m;
        col             = n;
        tol_d           = tolerance_d;
        tol_s           = tolerance_s;
        sampling_factor = d_factor;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_solver_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int reset_refinement) {
    if (reset_refinement) {
        // No need to re-generate A, x_sln, as they will remain untouched
        std::fill(all_data.A_single.begin(), all_data.A_single.end(), 0.0);
        std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
        std::fill(all_data.work.begin(), all_data.work.end(), 0.0);
    } else {
        RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    }
    // Vector b needs to be updated always
    blas::copy(m_info.rows, all_data.b_constant.data(), 1, all_data.b.data(), 1);
    std::fill(all_data.x.begin(), all_data.x.end(), 0.0);
    std::fill(all_data.work_double.begin(), all_data.work_double.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

template <typename T, typename RNG>
static long ICQRRP_refinement(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_solver_benchmark_data<T> &all_data,
                                        int64_t b_sz, 
                                        RandBLAS::RNGState<RNG> state) {
    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol_d    = all_data.tol_d;
    auto tol_s    = all_data.tol_s;
    auto d_factor = all_data.sampling_factor;

    // Set up a single-precision ICQRRP
    RandLAPACK::CQRRP_blocked<float, r123::Philox4x32> CQRRP_blocked(false, tol_d, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 8;

    T*       A           = all_data.A.data();           // double
    float*   A_single    = all_data.A_single.data();    // single
    float*   tau         = all_data.tau.data();         // single
    T*       b           = all_data.b.data();           // double
    T*       x           = all_data.x.data();           // double
    float*   work        = all_data.work.data();        // single 
    T*       work_double = all_data.work_double.data(); // double
    int64_t* J           = all_data.J.data();
    float* R_buf          = all_data.R_buffer.data();   // single

    T residual_inf_nrm = 0;
    T solution_inf_nrm = 0;
    T A_inf_nrm        = lapack::lange(Norm::Fro, m, n, A, m);

    long dur_icqrrp_refine = 0;
    auto start_icqrrp_refine = high_resolution_clock::now();
    // Cast the input matrix A down to single precision.
    // Copy columns in parallel.
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        std::transform(&A[m * i], &A[m * i + m], &A_single[m * i], [](double d) { return static_cast<float>(d); });

    // Call single-precision ICQRRP
    CQRRP_blocked.call(m, n, A_single, m, d_factor, tau, J, state);
    // Address Pivoting by permuting columns of A
    RandLAPACK::util::col_swap(m, n, n, A, m, all_data.J);

    // Vector all_data.x has already been initialized to all 0;
    // repeat below until some tolerance threshold is reached:
    int ctr = 0;
    while (ctr < 30) {
        // 1. Solve r = b - Ax for r in double precision.
        // Need to make sure that work = b before this.
        // After this, r will be stored in work (m by 1).
        std::transform(b, &b[m], work, [](double d) { return static_cast<float>(d); });
        blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, -1.0, A, m, x, 1, 1.0, work, 1);
        residual_inf_nrm = lapack::lange(Norm::Inf, m, 1, work, m);
        // 2. Solve Qy = r for y in single precision.
        // Since Q' = inv(Q); y = Q'r.
        // After this, y will be stored in work (m by 1).
        //lapack::laswp(1, work, m, 1, n, J, 1);
        lapack::ormqr(Side::Left, Op::Trans, m, 1, n, A_single, m, tau, work, m);
        // 3. Solve Rz = y for z in single precision.
        // A_single stores single-precision R.
        // After this, z will be stored in work (n by 1).
        blas::trsv(Layout::ColMajor, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, A_single, m, work, 1);	
        // 4. Transform work into a double-precision array
        std::transform(work, &work[n], work_double, [](float f) { return static_cast<double>(f); });
        // 5. x = x + z.
        blas::axpy(n, 1.0, work, 1, x, 1);

        //printf("Iteration %d\n", ctr);
        //printf("Ratio %.20e\n", blas::nrm2(n, work_double, 1) / blas::nrm2(n, x, 1));
        //printf("Tol %.20e\n\n", tol_d);

        //if(blas::nrm2(n, work_double, 1) / blas::nrm2(n, x, 1) < tol_s)
            //break;

        // Check termination criteria
        //solution_inf_nrm = lapack::lange(Norm::Fro, n, 1, x, n);
        //printf("residual_inf_nrm is %.20e\n", residual_inf_nrm);
        //printf("expr is %.20e\n\n", tol_d * std::sqrt(n) * solution_inf_nrm * A_inf_nrm);
        //if(residual_inf_nrm < tol_d * std::sqrt(n) * solution_inf_nrm * A_inf_nrm)
        //    break;

        ++ctr;	
    }
    auto stop_icqrrp_refine = high_resolution_clock::now();
    dur_icqrrp_refine = duration_cast<microseconds>(stop_icqrrp_refine - start_icqrrp_refine).count();

    return dur_icqrrp_refine;
}

template <typename T>
static long GEQRF_refinement(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_solver_benchmark_data<T> &all_data) {
    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol_s    = all_data.tol_s;
    auto tol_d    = all_data.tol_d;

    T*       A           = all_data.A.data();           // double
    float*   A_single    = all_data.A_single.data();    // single
    float*   tau         = all_data.tau.data();         // single
    T*       b           = all_data.b.data();           // double
    T*       x           = all_data.x.data();           // double
    float*   work        = all_data.work.data();        // single 
    T*       work_double = all_data.work_double.data(); // double

    T residual_inf_nrm = 0;
    T solution_inf_nrm = 0;
    T A_inf_nrm        = lapack::lange(Norm::Fro, m, n, A, m);

    long dur_geqrf_refine = 0;
    auto start_geqrf_refine = high_resolution_clock::now();
    // Cast the input matrix A down to single precision.
    // Copy columns in parallel.
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        std::transform(&A[m * i], &A[m * i + m], &A_single[m * i], [](double d) { return static_cast<float>(d); });

    // Call single-precision GEQRF
    lapack::geqrf(m, n, A_single, m, tau);
    // Vector all_data.x has already been initialized to all 0;
    // repeat below until some tolerance threshold is reached:
    int ctr = 0;
    while (ctr < 30) {
        // 1. Solve r = b - Ax for r in double precision.
        // Need to make sure that work = b before this.
        // After this, r will be stored in work (m by 1).
        std::transform(b, &b[m], work, [](double d) { return static_cast<float>(d); });
        blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, -1.0, A, m, x, 1, 1.0, work, 1);
        residual_inf_nrm = lapack::lange(Norm::Inf, m, 1, work, m);
        // 2. Solve Qy = r for y in single precision.
        // Since Q' = inv(Q); y = Q'r.
        // After this, y will be stored in work (m by 1).
        lapack::ormqr(Side::Left, Op::Trans, m, 1, n, A_single, m, tau, work, m);
        // 3. Solve Rz = y for z in single precision.
        // A_single stores single-precision R.
        // After this, z will be stored in work (n by 1).
        blas::trsv(Layout::ColMajor, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, A_single, m, work, 1);	
        // 4. Transform work into a double-precision array
        std::transform(work, &work[n], work_double, [](float f) { return static_cast<double>(f); });
        // 5. x = x + z.
        blas::axpy(n, 1.0, work_double, 1, x, 1);	

        //printf("Iteration %d\n", ctr);
        //printf("Ratio %.20e\n", blas::nrm2(n, work_double, 1) / blas::nrm2(n, x, 1));
        //printf("Tol %.20e\n\n", tol_d);

        //if(blas::nrm2(n, work_double, 1) / blas::nrm2(n, x, 1) < tol_s)
            //break;

        // Check termination criteria
        //solution_inf_nrm = lapack::lange(Norm::Fro, n, 1, x, n);
        //printf("residual_inf_nrm is %.20e\n", residual_inf_nrm);
        //printf("expr is %.20e\n\n", tol_d * std::sqrt(n) * solution_inf_nrm * A_inf_nrm);
        //if(residual_inf_nrm < tol_d * std::sqrt(n) * solution_inf_nrm * A_inf_nrm)
        //    break;

        ++ctr;
    }
    auto stop_geqrf_refine = high_resolution_clock::now();
    dur_geqrf_refine = duration_cast<microseconds>(stop_geqrf_refine - start_geqrf_refine).count();

    return dur_geqrf_refine;
}

template <typename T>
static long GETRF_refinement(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_solver_benchmark_data<T> &all_data) {
    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol_s    = all_data.tol_s;
    auto tol_d    = all_data.tol_d;

    T*       A           = all_data.A.data();           // double
    float*   A_single    = all_data.A_single.data();    // single
    float*   tau         = all_data.tau.data();         // single
    T*       b           = all_data.b.data();           // double
    T*       x           = all_data.x.data();           // double
    float*   work        = all_data.work.data();        // single 
    T*       work_double = all_data.work_double.data(); // double
    int64_t* J           = all_data.J.data();

    T residual_inf_nrm = 0;
    T solution_inf_nrm = 0;
    T A_inf_nrm        = lapack::lange(Norm::Fro, m, n, A, m);

    long dur_getrf_refine = 0;
    auto start_getrf_refine = high_resolution_clock::now();
    // Cast the input matrix A down to single precision.
    // Copy columns in parallel.
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        std::transform(&A[m * i], &A[m * i + m], &A_single[m * i], [](double d) { return static_cast<float>(d); });

    // Call single-precision getrf
    lapack::getrf(m, n, A_single, m, J);
    // Vector all_data.x has already been initialized to all 0;
    // repeat below until some tolerance threshold is reached:
    int ctr = 0;
    while (ctr < 30) {
        // 1. Solve r = b - Ax for r in double precision.
        // Need to make sure that work = b before this.
        // After this, r will be stored in work (m by 1).
        std::transform(b, &b[m], work, [](double d) { return static_cast<float>(d); });
        blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, -1.0, A, m, x, 1, 1.0, work, 1);
        residual_inf_nrm = lapack::lange(Norm::Inf, m, 1, work, m);
        // 2. Solve Qy = Pr for y in single precision.
        // Since Q' = inv(Q); y = Q'Pr.
        // After this, y will be stored in work (m by 1).
        lapack::laswp(1, work, m, 1, n, J, 1);
        lapack::ormqr(Side::Left, Op::Trans, m, 1, n, A_single, m, tau, work, m);
        // 3. Solve Rz = y for z in single precision.
        // A_single stores single-precision R.
        // After this, z will be stored in work (n by 1).
        blas::trsv(Layout::ColMajor, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, A_single, m, work, 1);	
        // 4. Transform work into a double-precision array
        std::transform(work, &work[n], work_double, [](float f) { return static_cast<double>(f); });
        // 5. x = x + z.
        blas::axpy(n, 1.0, work_double, 1, x, 1);	

        //printf("Iteration %d\n", ctr);
        //printf("Ratio %.20e\n", blas::nrm2(n, work_double, 1) / blas::nrm2(n, x, 1));
        //printf("Tol %.20e\n\n", tol_d);

        //if(blas::nrm2(n, work_double, 1) / blas::nrm2(n, x, 1) < tol_s)
            //break;

        // Check termination criteria
        //solution_inf_nrm = lapack::lange(Norm::Fro, n, 1, x, n);
        //printf("residual_inf_nrm is %.20e\n", residual_inf_nrm);
        //printf("expr is %.20e\n\n", tol_d * std::sqrt(n) * solution_inf_nrm * A_inf_nrm);
        //if(residual_inf_nrm < tol_d * std::sqrt(n) * solution_inf_nrm * A_inf_nrm)
        //    break;

        ++ctr;
    }
    auto stop_getrf_refine = high_resolution_clock::now();
    dur_getrf_refine = duration_cast<microseconds>(stop_getrf_refine - start_getrf_refine).count();

    return dur_getrf_refine;
}

template <typename T>
static T forward_error(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_solver_benchmark_data<T> &all_data) {
    auto n        = all_data.col;
    T* x_solution = all_data.x_solution.data();
    T* x          = all_data.x.data();

    // Return ||x - x_sln||.
    blas::axpy(n, -1.0, x_solution, 1, x, 1);	
    return blas::nrm2(n, x, 1);
}

template <typename T>
static T backward_error(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_solver_benchmark_data<T> &all_data) {
    auto m        = all_data.row;
    auto n        = all_data.col;

    T* A            = all_data.A.data();
    T* x            = all_data.x.data();
    T* b            = all_data.b.data();
    T* work_double  = all_data.work_double.data();

    // Compute Ax - b
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, 1.0, A, m, x, 1, -1.0, b, 1);
    T nrm1 = blas::nrm2(m, b, 1);
    // Compute ||A'Ax - A'Ab||
    blas::gemv(Layout::ColMajor, Op::Trans, m, n, 1.0, A, m, b, 1, 0.0, work_double, 1);
    T nrm_numerator = blas::nrm2(n, work_double, 1);
    // Compute ||Ax-b||||A||_F
    T nrm_denominator = nrm1 * lapack::lange(Norm::Fro, m, n, A, m);

    //return (nrm_numerator / nrm_denominator);
    return nrm1;
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

    // timing vars
    long dur_gels         = 0;
    long dur_gesv         = 0;
    long dur_cqrrp_refine = 0;
    long dur_geqrf_refine = 0;
    long dur_getrf_refine = 0;

    // error vars
    T forward_err_geqrf  = 0;
    T backward_err_geqrf = 0;
    T forward_err_getrf  = 0;
    T backward_err_getrf = 0;
    T forward_err_cqrrp  = 0;
    T backward_err_cqrrp = 0;
    T forward_err_gesv   = 0;
    T backward_err_gesv  = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;

    char name [] = "A";
    char name1 [] = "x";
    char name2 [] = "b";

    //RandBLAS::util::print_colmaj(m, n, all_data.A.data(), name); 
    //RandBLAS::util::print_colmaj(m, 1, all_data.b.data(), name2); 

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d\n", i);
        // Testing GELS
        auto start_gels = high_resolution_clock::now();
        lapack::gels(Op::NoTrans, m, n, 1, all_data.A.data(), m, all_data.b.data(), m);	
        auto stop_gels = high_resolution_clock::now();
        dur_gels = duration_cast<microseconds>(stop_gels - start_gels).count();
        printf("\nTOTAL TIME FOR GELS %ld\n", dur_gels);

        // b now stores the solution vector.
        // For some reason, the original solution is actually bad.
        blas::copy(n, all_data.b.data(), 1, all_data.x_solution.data(), 1);

        data_regen(m_info, all_data, state_gen, 0);
        state_gen = state;

        // Testing GESV
        auto start_gesv = high_resolution_clock::now();
        lapack::gesv(n, 1, all_data.A.data(), m, all_data.J.data(), all_data.b.data(), m);	
        auto stop_gesv = high_resolution_clock::now();
        dur_gesv = duration_cast<microseconds>(stop_gesv - start_gesv).count();
        printf("\nTOTAL TIME FOR GESV %ld\n", dur_gesv);

        // Copy everything into the right places
        blas::copy(m, all_data.b.data(), 1, all_data.x.data(), 1);
        blas::copy(m, all_data.b_constant.data(), 1, all_data.b.data(), 1);
        backward_err_gesv = backward_error(m_info, all_data);
        forward_err_gesv  = forward_error(m_info, all_data);
        printf("F_err_GESV: %e\n", backward_err_gesv);
        printf("B_err_GESV: %e\n", forward_err_gesv);

        data_regen(m_info, all_data, state_gen, 0);
        state_gen = state;

        // Testing ICQRRP + iterative refinement
        dur_cqrrp_refine = ICQRRP_refinement( m_info, all_data, b_sz, state);
        printf("\nTOTAL TIME FOR ICQRRP + refinement %ld\n", dur_cqrrp_refine);


        //RandBLAS::util::print_colmaj(m, n, all_data.A.data(), name); 
        //RandBLAS::util::print_colmaj(n, 1, all_data.x.data(), name1); 

        backward_err_geqrf = backward_error(m_info, all_data);
        forward_err_geqrf  = forward_error(m_info, all_data);
        printf("F_err_ICQRRP: %e\n", forward_err_geqrf);
        printf("B_err_ICQRRP: %e\n", backward_err_geqrf);

        data_regen(m_info, all_data, state_gen, 1);
        state_gen = state;

        // Testing GEQRF + iterative refinement
        dur_geqrf_refine = GEQRF_refinement( m_info, all_data);
        printf("\nTOTAL TIME FOR GEQRF + refinement %ld\n", dur_cqrrp_refine);

        //RandBLAS::util::print_colmaj(m, n, all_data.A.data(), name); 
        //RandBLAS::util::print_colmaj(n, 1, all_data.x.data(), name1); 

        backward_err_cqrrp = backward_error(m_info, all_data);
        forward_err_cqrrp  = forward_error(m_info, all_data);
        printf("F_err_GEQRF: %e\n", forward_err_cqrrp);
        printf("B_err_GEQRF: %e\n", backward_err_cqrrp);

        data_regen(m_info, all_data, state_gen, 1);
        state_gen = state;

        // Testing GETRF + iterative refinement
        dur_getrf_refine = GETRF_refinement( m_info, all_data);
        printf("\nTOTAL TIME FOR GETRF + refinement %ld\n", dur_getrf_refine);

        //RandBLAS::util::print_colmaj(m, n, all_data.A.data(), name); 
        //RandBLAS::util::print_colmaj(n, 1, all_data.x.data(), name1); 

        //lapack::laswp(n, all_data.A.data(), m, 1, n, all_data.J.data(), 1);

        backward_err_getrf = backward_error(m_info, all_data);
        forward_err_getrf  = forward_error(m_info, all_data);
        printf("F_err_GETRF: %e\n", backward_err_getrf);
        printf("B_err_GETRF: %e\n", forward_err_getrf);

        data_regen(m_info, all_data, state_gen, 1);
        state_gen = state;

        std::ofstream file(output_filename, std::ios::app);
        file << m << ",  " << b_sz << ",  " << forward_err_geqrf << ",  " << backward_err_geqrf << ",  " << forward_err_cqrrp << ",  " << backward_err_cqrrp << ",  " << forward_err_gesv << ",  " << backward_err_gesv << ",  " << dur_gels << ",  " << dur_cqrrp_refine << ",  " << dur_geqrf_refine << ",  " << dur_gesv << ",\n";
    }
}


int main() {
    // Declare parameters
    int64_t m           = 5;
    int64_t n           = 5;
    double  d_factor    = 1.25;
    int64_t b_sz_start  = 5;
    int64_t b_sz_end    = 5;
    double tol_d        = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    float tol_s         = std::pow(std::numeric_limits<float>::epsilon(), 0.95);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 1;

    // Allocate basic workspace
    QR_solver_benchmark_data<double> all_data(m, n, tol_d, tol_s, d_factor);
    // Generate the input matrix
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = std::pow(10, -16);
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Randomly the solution vector
    RandBLAS::DenseDist D(m, 1);
    RandBLAS::fill_dense(D, all_data.b.data(), state).second;
    blas::copy(m, all_data.b.data(), 1, all_data.b_constant.data(), 1);

    // Declare a data file
    std::string output_filename = "ICQRRP_solver_time_raw_rows_"   + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz_start)
                                    + "_b_sz_end_"     + std::to_string(b_sz_end)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat";

    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        call_all_algs(m_info, numruns, b_sz_start, all_data, state_constant, output_filename);
    }
}
/*
int main(int argc, char *argv[]) {

    printf("Function begin\n");

    if(argc <= 1) {
        printf("No input provided\n");
        return 0;
    }

    // Declare parameters
    int64_t m           = 0;
    int64_t n           = 0;
    double d_factor     = 1.25;
    int64_t b_sz_start  = 8;
    int64_t b_sz_end    = 8;
    double tol_d        = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    float tol_s         = std::pow(std::numeric_limits<float>::epsilon(), 0.95);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 1;

    // Read in the input matrix.
    // Data is stored as [A, x, b] array, where x is padded with zeros at the end.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[1];
    m_info.workspace_query_mod = 1;
    // Workspace query;
    RandLAPACK::gen::mat_gen<double>(m_info, NULL, state);
  
    // Fill the Axb buffer
    double* Axb_buf = ( double * ) calloc( m_info.rows * m_info.cols, sizeof( double ) );
    RandLAPACK::gen::mat_gen(m_info, Axb_buf, state);
    m_info.cols = m_info.cols - 2;
    m = m_info.rows;
    n = m_info.cols;

    // Allocate basic workspace.
    QR_solver_benchmark_data<double> all_data(m, n, tol_d, tol_s, d_factor);

    // Copy A, x, b over
    lapack::lacpy(MatrixType::General, m, n, Axb_buf, m, all_data.A.data(), m);
    blas::copy(n, &Axb_buf[m * n], 1, all_data.x_solution.data(), 1);
    blas::copy(m, &Axb_buf[m * (n + 1)], 1, all_data.b.data(), 1);
    blas::copy(m, &Axb_buf[m * (n + 1)], 1, all_data.b_constant.data(), 1);
    // clear the buffer
    free(Axb_buf);

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
}*/