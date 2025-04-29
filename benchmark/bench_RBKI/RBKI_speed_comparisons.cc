/*
Additional RBKI speed comparison benchmark - runs RBKI, RSVD and SVDS from Spectra library.
The user is required to provide a matrix file to be read, set min and max numbers of large gemms (Krylov iterations) that the algorithm is allowed to perform min and max block sizes that RBKI is to use; 
furthermore, the user is to provide a 'custom rank' parameter (number of singular vectors to approximate by RBKI). 
The benchmark outputs the basic data of a given run, as well as the RBKI runtime and singular vector residual error, 
which is computed as "sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F / sqrt(target_rank)" (for "custom rank" singular vectors and values).
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>

// External libs includes
#include <Eigen/Dense>
#include <Spectra/contrib/PartialSVDSolver.h>
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

template <typename T>
struct RBKI_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    T* A;
    T* U;
    T* VT; 
    T* V;  
    T* Sigma;
    T* A_lowrank_svd;
    T* A_lowrank_svd_const;
    T* Buffer;
    T* Sigma_cpy;
    T* U_cpy;
    T* V_cpy;
    Matrix A_spectra;

    RBKI_benchmark_data(int64_t m, int64_t n, T tol) :
    A_spectra(m, n)
    {
        A          = new T[m * n]();
        U          = nullptr;
        VT         = nullptr;
        V          = nullptr;
        Sigma      = nullptr;
        U_cpy      = nullptr;
        V_cpy      = nullptr;

        A_lowrank_svd       = nullptr;
        A_lowrank_svd_const = nullptr;
        row                 = m;
        col                 = n;
        tolerance           = tol;
    }

    ~RBKI_benchmark_data() {
        delete[] A;
        delete[] U;
        delete[] VT;
        delete[] V;
        delete[] Sigma;
        delete[] U_cpy;
        delete[] V_cpy;
        delete[] A_lowrank_svd;
        delete[] A_lowrank_svd_const;
    }
};

template <typename T, typename RNG>
struct RBKI_algorithm_objects {
    RandLAPACK::PLUL<T> Stab;
    RandLAPACK::RS<T, RNG> RS;
    RandLAPACK::CholQRQ<T> Orth_RF;
    RandLAPACK::RF<T, RNG> RF;
    RandLAPACK::CholQRQ<T> Orth_QB;
    RandLAPACK::QB<T, RNG> QB;
    RandLAPACK::RSVD<T, RNG> RSVD;
    RandLAPACK::RBKI<T, RNG> RBKI;

    RBKI_algorithm_objects(
        bool verbosity, 
        bool cond_check, 
        bool orth_check, 
        bool time_subroutines, 
        int64_t p, 
        int64_t passes_per_iteration, 
        int64_t block_sz,
        T tol
    ) :
        Stab(cond_check, verbosity),
        RS(Stab, p, passes_per_iteration, verbosity, cond_check),
        Orth_RF(cond_check, verbosity),
        RF(RS, Orth_RF, verbosity, cond_check),
        Orth_QB(cond_check, verbosity),
        QB(RF, Orth_QB, verbosity, orth_check),
        RSVD(QB, block_sz),
        RBKI(verbosity, time_subroutines, tol)
        {}
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        RBKI_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int overwrite_A) {

    auto m = all_data.row;
    auto n = all_data.col;

    if (overwrite_A) {
        RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
        Eigen::Map<Eigen::MatrixXd>(all_data.A_spectra.data(), all_data.A_spectra.rows(), all_data.A_spectra.cols()) = Eigen::Map<const Eigen::MatrixXd>(all_data.A, m, n);
        if (all_data.A_lowrank_svd != nullptr)
            lapack::lacpy(MatrixType::General, m, n, all_data.A_lowrank_svd_const, m, all_data.A_lowrank_svd, m);
    }

    delete[] all_data.U;
    delete[] all_data.VT;
    delete[] all_data.V;
    delete[] all_data.Sigma;
    delete[] all_data.U_cpy;
    delete[] all_data.V_cpy;

    all_data.U     = nullptr;
    all_data.VT    = nullptr;
    all_data.V     = nullptr;
    all_data.Sigma = nullptr;
    all_data.U_cpy = nullptr;
    all_data.V_cpy = nullptr;
}

// This routine computes the residual norm error, consisting of two parts (one of which) vanishes
// in exact precision. Target_rank defines size of U, V as returned by RBKI; target_rank <= target_rank.
template <typename T, typename TestData>
static T
residual_error_comp(TestData &all_data, int64_t target_rank) {
    auto m = all_data.row;
    auto n = all_data.col;

    all_data.U_cpy = new T[m * target_rank]();
    all_data.V_cpy = new T[n * target_rank]();

    lapack::lacpy(MatrixType::General, m, target_rank, all_data.U, m, all_data.U_cpy, m);
    lapack::lacpy(MatrixType::General, n, target_rank, all_data.V, n, all_data.V_cpy, n);

    // AV - US
    // Scale columns of U by S
    for (int i = 0; i < target_rank; ++i)
        blas::scal(m, all_data.Sigma[i], &all_data.U_cpy[m * i], 1);

    // Compute AV(:, 1:target_rank) - SU(1:target_rank)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, target_rank, n, 1.0, all_data.A, m, all_data.V, n, -1.0, all_data.U_cpy, m);

    // A'U - VS
    // Scale columns of V by S
    for (int i = 0; i < target_rank; ++i)
        blas::scal(n, all_data.Sigma[i], &all_data.V_cpy[i * n], 1);
    // Compute A'U(:, 1:target_rank) - VS(1:target_rank).
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, target_rank, m, 1.0, all_data.A, m, all_data.U, m, -1.0, all_data.V_cpy, n);

    T nrm1 = lapack::lange(Norm::Fro, m, target_rank, all_data.U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, n, target_rank, all_data.V_cpy, n);

    return std::hypot(nrm1, nrm2);
}

template <typename T>
static T
approx_error_comp(RBKI_benchmark_data<T> &all_data, int64_t target_rank, T norm_A_lowrank) {
    
    auto m = all_data.row;
    auto n = all_data.col;

    all_data.U_cpy = new T[m * target_rank]();
    lapack::lacpy(MatrixType::General, m, target_rank, all_data.U, m, all_data.U_cpy, m);

    // U * S; scale the columns of U by S
    for (int i = 0; i < target_rank; ++i)
    blas::scal(m, all_data.Sigma[i], &all_data.U_cpy[i * m], 1);
    
    // U * S * V' - A_cpy ~= 0?
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, target_rank, 1.0, all_data.U_cpy, m, all_data.V, n, -1.0, all_data.A_lowrank_svd, m);

    T nrm = lapack::lange(Norm::Fro, m, n, all_data.A_lowrank_svd, m);
    printf("||A_hat_cursom_rank - A_svd_target_rank||_F / ||A_svd_target_rank||_F: %e\n", nrm / norm_A_lowrank);

    return nrm / norm_A_lowrank;
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t num_runs,
    int64_t b_sz,
    int64_t num_matmuls,
    int64_t target_rank,
    RBKI_algorithm_objects<T, RNG> &all_algs,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename, 
    T norm_A_lowrank) {

    int i;
    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;

    // Additional params setup.
    all_algs.RSVD.block_sz = b_sz;
    // Matrices R or S that give us the singular value spectrum returned by RBKI will be of size b_sz * num_krylov_iters / 2.
    // These matrices will be full-rank.
    // Hence, target_rank = b_sz * num_krylov_iters / 2 
    // RBKI.max_krylov_iters = (int) ((target_rank * 2) / b_sz);
    // 
    // Instead of the above approach, we now pre-specify the maximum number of Krylov iters that we allow for in num_matmuls.
    all_algs.RBKI.max_krylov_iters = (int) num_matmuls;
    all_algs.RBKI.num_threads_min = 4;
    all_algs.RBKI.num_threads_max = RandLAPACK::util::get_omp_threads();
    
    // timing vars
    long dur_rbki = 0;
    long dur_rsvd = 0;
    long dur_svds = 0;
    long dur_svd  = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    T residual_err_custom_SVD  = 0;
    T residual_err_custom_RBKI = 0;
    T residual_err_custom_RSVD = 0;
    T residual_err_custom_SVDS = 0;

    T lowrank_err_SVD  = 0;
    T lowrank_err_RBKI = 0;
    T lowrank_err_RSVD = 0;
    T lowrank_err_SVDS = 0;

    int64_t singular_triplets_target_RBKI = 0;
    int64_t singular_triplets_found_RSVD  = 0;
    int64_t singular_triplets_target_RSVD = 0;
    int64_t singular_triplets_target_SVDS = 0;

    for (i = 0; i < num_runs; ++i) {
        printf("\nBlock size %ld, num matmuls %ld. Iteration %d start.\n", b_sz, num_matmuls, i);
        
        // Running RBKI
        auto start_rbki = steady_clock::now();
        all_algs.RBKI.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state_alg);
        auto stop_rbki = steady_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();
        printf("TOTAL TIME FOR RBKI %ld\n", dur_rbki);

        // This is in case the number of singular triplets is smaller than the target rank
        singular_triplets_target_RBKI = std::min(target_rank, all_algs.RBKI.singular_triplets_found);

        residual_err_custom_RBKI = residual_error_comp<T>(all_data, singular_triplets_target_RBKI);
        printf("RBKI sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(target_rank): %.16e\n", residual_err_custom_RBKI);

        if (all_data.A_lowrank_svd != nullptr)
            lowrank_err_RBKI = approx_error_comp(all_data, singular_triplets_target_RBKI, norm_A_lowrank);
        
        state_alg = state;
        state_gen = state;
        data_regen(m_info, all_data, state_gen, 1);
        
        // Running RSVD
        auto start_rsvd = steady_clock::now();
        singular_triplets_found_RSVD = (int64_t ) (b_sz * num_matmuls / 2);

        all_data.U     = new T[m * singular_triplets_found_RSVD]();
        all_data.V     = new T[n * singular_triplets_found_RSVD]();
        all_data.Sigma = new T[singular_triplets_found_RSVD]();

        all_algs.RSVD.call(m, n, all_data.A, singular_triplets_found_RSVD, tol, all_data.U, all_data.Sigma, all_data.V, state_alg);
        auto stop_rsvd = steady_clock::now();
        dur_rsvd = duration_cast<microseconds>(stop_rsvd - start_rsvd).count();
        printf("TOTAL TIME FOR RSVD %ld\n", dur_rsvd);

        // This is in case the number of singular triplets is smaller than the target rank
        singular_triplets_target_RSVD = std::min(singular_triplets_found_RSVD, target_rank);

        residual_err_custom_RSVD = residual_error_comp<T>(all_data, singular_triplets_target_RSVD);
        printf("RSVD sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(target_rank): %.16e\n", residual_err_custom_RSVD);

        if (all_data.A_lowrank_svd != nullptr)
            lowrank_err_RSVD = approx_error_comp(all_data, singular_triplets_target_RSVD, norm_A_lowrank);
        
        state_alg = state;
        state_gen = state;
        data_regen(m_info, all_data, state_gen, 1);
        
        // There is no reason to run SVDS many times, as it always outputs the same result.
        //if ((num_matmuls == 2) && ((i == 0) || (i == 1))) {
            // Running SVDS
            auto start_svds = steady_clock::now();
            // This is in case the number of singular triplets is smaller than the target rank
            singular_triplets_target_SVDS = std::min(target_rank, n-2);

            Spectra::PartialSVDSolver<Matrix> svds(all_data.A_spectra, singular_triplets_target_SVDS, std::min(2 * target_rank, n-1));
            svds.compute();
            auto stop_svds = steady_clock::now();
            dur_svds = duration_cast<microseconds>(stop_svds - start_svds).count();
            printf("TOTAL TIME FOR SVDS %ld\n", dur_svds);

            // Copy data from Spectra (Eigen) format to the nomal C++.
            Matrix U_spectra = svds.matrix_U(singular_triplets_target_SVDS);
            Matrix V_spectra = svds.matrix_V(singular_triplets_target_SVDS);
            Vector S_spectra = svds.singular_values();

            all_data.U     = new T[m * singular_triplets_target_SVDS]();
            all_data.V     = new T[n * singular_triplets_target_SVDS]();
            all_data.Sigma = new T[singular_triplets_target_SVDS]();

            Eigen::Map<Matrix>(all_data.U, m, singular_triplets_target_SVDS)  = U_spectra;
            Eigen::Map<Matrix>(all_data.V, n, singular_triplets_target_SVDS)  = V_spectra;
            Eigen::Map<Vector>(all_data.Sigma, singular_triplets_target_SVDS) = S_spectra;

            residual_err_custom_SVDS = residual_error_comp<T>(all_data, singular_triplets_target_SVDS);
            printf("SVDS sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(target_rank): %.16e\n", residual_err_custom_SVDS);

            if (all_data.A_lowrank_svd != nullptr)
                lowrank_err_SVDS = approx_error_comp(all_data, singular_triplets_target_SVDS, norm_A_lowrank);
            
            state_alg = state;
            state_gen = state;
            data_regen(m_info, all_data, state_gen, 1);
        //}
        
        // There is no reason to run SVD many times, as it always outputs the same result.
        //if ((b_sz == 16) && (num_matmuls == 2) && ((i == 0) || (i == 1))) {
            // Running SVD
            auto start_svd = steady_clock::now();
            all_data.U     = new T[m * n]();
            all_data.Sigma = new T[n]();
            all_data.VT    = new T[n * n]();
            all_data.V     = new T[n * n]();
            lapack::gesdd(Job::SomeVec, m, n, all_data.A, m, all_data.Sigma, all_data.U, m, all_data.VT, n);
            auto stop_svd = steady_clock::now();
            dur_svd = duration_cast<microseconds>(stop_svd - start_svd).count();
            printf("TOTAL TIME FOR SVD %ld\n", dur_svd);

            /*
            char name[] = "A";
            char name1[] = "U";
            char name2[] = "VT";
            char name3[] = "S";
            RandLAPACK::util::print_colmaj(m, n, all_data.A, m, name);
            RandLAPACK::util::print_colmaj(m, n, all_data.U, m, name1);
            RandLAPACK::util::print_colmaj(n, n, all_data.VT, n, name2);
            RandLAPACK::util::print_colmaj(n, 1, all_data.Sigma, n, name3);
            */

            // Standard SVD destorys matrix A, need to re-read it before running accuracy tests.
            state_gen = state;
            RandLAPACK::gen::mat_gen(m_info, all_data.A, state_gen);
            RandLAPACK::util::transposition(n, n, all_data.VT, n, all_data.V, n, 0);

            residual_err_custom_SVD = residual_error_comp<T>(all_data, target_rank);
            printf("SVD sqrt(||AV - US||^2_F + ||A'U - VS||^2_F) / sqrt(target_rank): %.16e\n", residual_err_custom_SVD);

            if (all_data.A_lowrank_svd != nullptr)
                lowrank_err_SVD = approx_error_comp(all_data, target_rank, norm_A_lowrank);

            state_alg = state;
            state_gen = state;
            data_regen(m_info, all_data, state_gen, 1);
        //}

        std::ofstream file(output_filename, std::ios::app);
        file << b_sz << ",  " << all_algs.RBKI.max_krylov_iters  <<  ",  " << target_rank << ",  " 
        << residual_err_custom_RBKI << ",  " << lowrank_err_RBKI <<  ",  " << dur_rbki    << ",  " 
        << residual_err_custom_RSVD << ",  " << lowrank_err_RSVD <<  ",  " << dur_rsvd    << ",  "
        << residual_err_custom_SVDS << ",  " << lowrank_err_SVDS <<  ",  " << dur_svds    << ",  " 
        << residual_err_custom_SVD  << ",  " << lowrank_err_SVD  <<  ",  " << dur_svd     << ",\n";
    }
}

int main(int argc, char *argv[]) {

    if (argc < 12) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <output_directory_path> <input_matrix_path> <lowrank_matrix_path> <num_runs> <num_rows> <num_cols> <target_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return 1;
    }

    int num_runs              = std::stol(argv[4]);
    int64_t m_expected        = std::stol(argv[5]);
    int64_t n_expected        = std::stol(argv[6]);
    int64_t target_rank       = std::stol(argv[7]);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[8]); ++i)
        b_sz.push_back(std::stoi(argv[i + 10]));
    // Save elements in string for logging purposes
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[9]); ++i)
        matmuls.push_back(std::stoi(argv[i + 10 + std::stol(argv[8])]));
    // Save elements in string for logging purposes
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();
    double tol                = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                = RandBLAS::RNGState();
    auto state_constant       = state;
    double norm_A_lowrank     = 0;
    int64_t m = 0, n = 0;

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[2];
    m_info.workspace_query_mod = 1;
    // Workspace query;
    RandLAPACK::gen::mat_gen<double>(m_info, NULL, state);

    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    if (m_expected != m || n_expected != n) {
        std::cerr << "Expected input size (" << m_expected << ", " << n_expected << ") did not matrch actual input size (" << m << ", " << n << "). Aborting." << std::endl;
        return 1;
    }

    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, tol);
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    // Declare objects for RSVD and RBKI
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    // Block size will need to be altered.
    int64_t block_sz = 0;
    RBKI_algorithm_objects<double, r123::Philox4x32> all_algs(false, false, false, false, p, passes_per_iteration, block_sz, tol);

    // Copying input data into a Spectra (Eigen) matrix object
    Eigen::Map<Eigen::MatrixXd>(all_data.A_spectra.data(), all_data.A_spectra.rows(), all_data.A_spectra.cols()) = Eigen::Map<const Eigen::MatrixXd>(all_data.A, m, n);

    // Optional pass of lowrank SVD matrix into the benchmark
    if (std::string(argv[3]) != ".") {
        printf("Lowrank A input.\n");
        RandLAPACK::gen::mat_gen_info<double> m_info_A_svd(m, n, RandLAPACK::gen::custom_input);
        m_info_A_svd.filename            = argv[3];
        m_info_A_svd.workspace_query_mod = 0;
        all_data.A_lowrank_svd       = new double[m * n]();
        all_data.A_lowrank_svd_const = new double[m * n]();
        RandLAPACK::gen::mat_gen<double>(m_info_A_svd, all_data.A_lowrank_svd_const, state);
        lapack::lacpy(MatrixType::General, m, n, all_data.A_lowrank_svd_const, m, all_data.A_lowrank_svd, m);
    
        // Pre-compute norm(A lowrank) for future benchmarking
        norm_A_lowrank = lapack::lange(Norm::Fro, m, n, all_data.A_lowrank_svd, m);
    }

    printf("Finished data preparation\n");
    // Declare a data file
    std::string output_filename = "_ABRIK_speed_comparisons_num_info_lines_" + std::to_string(6) + ".txt";
    std::string path;
    if (std::string(argv[1]) != ".") {
        path = argv[1] + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the ABRIK speed comparison benchmark, recording the time it takes to perform RBKI and alternative methods for low-rank SVD."
              "\nFile format: 15 columns, showing krylov block size, nummber of matmuls permitted, and num svals and svecs to approximate, followed by the residual error, standard lowrank error and execution time for all algorithms (ABRIK, RSVD, SVDS, SVD)"
              "\n Rows correspond to algorithm runs with Krylov block sizes varying as specified, and numbers of matmuls varying as specified per eah block size, with num_runs repititions of each number of matmuls."
              "\nInput type:"       + std::string(argv[2]) +
              "\nInput size:"       + std::to_string(m) + " by "             + std::to_string(n) +
              "\nAdditional parameters: Krylov block sizes "                 + b_sz_string +
                                        " matmuls: "                         + matmuls_string +
                                        " num runs per size "                + std::to_string(num_runs) +
                                        " num singular values and vectors approximated " + std::to_string(target_rank) +
              "\n";
    file.flush();

    size_t i = 0, j = 0;
    for (;i < b_sz.size(); ++i) {
        for (;j < matmuls.size(); ++j) {
            call_all_algs(m_info, num_runs, b_sz[i], matmuls[j], target_rank, all_algs, all_data, state_constant, path, norm_A_lowrank);
        }
        j = 0;
    }
}
