/*
Additional ABRIK speed comparison benchmark - runs ABRIK, RSVD and SVDS from Spectra library.
The user is required to provide a matrix file to be read, set min and max numbers of large gemms (Krylov iterations) that the algorithm is allowed to perform min and max block sizes that ABRIK is to use; 
furthermore, the user is to provide a 'custom rank' parameter (number of singular vectors to approximate by ABRIK). 
The benchmark outputs the basic data of a given run, as well as the ABRIK runtime and singular vector residual error, 
which is computed as "sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F / sqrt(target_rank)" (for "custom rank" singular vectors and values).
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>

// External libs includes
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Spectra/contrib/PartialSVDSolver.h>

using SpMatrix = Eigen::SparseMatrix<double>;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

using Subroutines = RandLAPACK::ABRIKSubroutines;

template <typename T, RandBLAS::sparse_data::SparseMatrix SpMat>
struct ABRIK_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    SpMat* A_input;
    RandLAPACK::linops::SpLinOp<SpMat> A_linop;
    T* U;
    T* V; 
    T* Sigma;
    T* U_cpy;
    T* V_cpy;
    SpMatrix A_spectra;

    ABRIK_benchmark_data(SpMat& input_mat, int64_t m, int64_t n, T tol) :
    A_spectra(m, n),
    A_linop(m, n, input_mat)
    {
        U     = nullptr;
        V     = nullptr;
        Sigma = nullptr;
        U_cpy = nullptr;
        V_cpy = nullptr;

        row       = m;
        col       = n;
        tolerance = tol;
    }

    ~ABRIK_benchmark_data() {
        delete[] U;
        delete[] V;
        delete[] Sigma;
        delete[] U_cpy;
        delete[] V_cpy;
    }
};

template <typename T, typename RNG>
struct ABRIK_algorithm_objects {
    RandLAPACK::ABRIK<T, RNG> ABRIK;

    ABRIK_algorithm_objects(
        bool verbosity, 
        bool time_subroutines, 
        T tol
    ) :
        ABRIK(verbosity, time_subroutines, tol)
        {}
};

template <typename T>
RandBLAS::sparse_data::coo::COOMatrix<T> from_matrix_market(std::string fn) {

    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows,  cols, vals
    );

    RandBLAS::sparse_data::coo::COOMatrix<T> out(n_rows, n_cols);
    reserve_coo(vals.size(),out);
    for (int i = 0; i < out.nnz; ++i) {
        out.rows[i] = rows[i];
        out.cols[i] = cols[i];
        out.vals[i] = vals[i];
    }

    return out;
}

template <typename T>
void from_matrix_market(std::string fn, SpMatrix& A) {

    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows,  cols, vals
    );

    std::vector<Eigen::Triplet<T>> tripletList;
    for (int i = 0; i < vals.size(); ++i) 
        tripletList.push_back(Eigen::Triplet<T>(rows[i], cols[i], vals[i]));

    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

template <typename T>
void from_matrix_market_leading_submatrix(std::string fn, SpMatrix& A, T submatrix_dim_ratio) {
    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows, cols, vals
    );

    // Use half-size for the leading principal submatrix
    int64_t m = n_rows * submatrix_dim_ratio;
    int64_t n = n_cols * submatrix_dim_ratio;

    // Create triplets only for entries within the submatrix
    std::vector<Eigen::Triplet<T>> tripletList;
    for (size_t i = 0; i < vals.size(); ++i) {
        if (rows[i] < m && cols[i] < n) {
            tripletList.emplace_back(rows[i], cols[i], vals[i]);
        }
    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

template <typename T>
RandBLAS::CSCMatrix<T> format_conversion(int64_t m, int64_t n, RandBLAS::COOMatrix<T>& input_mat_coo)
{
    // Grab the leading principal submatrix of the size of half the input
    RandBLAS::COOMatrix<double> input_mat_transformed(m, n);

    // check how many nonzeros are in the left principal submatrix
    int64_t nnz_sub = 0;
    for (int64_t i = 0; i < input_mat_coo.nnz; ++i) {
        if (input_mat_coo.rows[i] < m && input_mat_coo.cols[i] < n) {
            ++nnz_sub;
        }
    }

    // Allocate
    reserve_coo(nnz_sub, input_mat_transformed);

    int64_t ell = 0;
    for (int64_t i = 0; i < input_mat_coo.nnz; ++i) {
        if (input_mat_coo.rows[i] < m && input_mat_coo.cols[i] < n) {
            input_mat_transformed.rows[ell] = input_mat_coo.rows[i];
            input_mat_transformed.cols[ell] = input_mat_coo.cols[i];
            input_mat_transformed.vals[ell] = input_mat_coo.vals[i];
            ++ell;
        }
    }

    // Convert the sparse matrix format for performance
    RandBLAS::CSCMatrix<double> input_mat_csc(m, n);
    //RandBLAS::conversions::coo_to_csc(input_mat_coo, input_mat_csc);
    RandBLAS::conversions::coo_to_csc(input_mat_transformed, input_mat_csc);

    return input_mat_csc;
}


// Re-generate and clear data
template <typename T, typename RNG, RandBLAS::sparse_data::SparseMatrix SpMat>
static void data_regen(ABRIK_benchmark_data<T, SpMat> &all_data, 
                        RandBLAS::RNGState<RNG> &state) {

    delete[] all_data.U;
    delete[] all_data.V;
    delete[] all_data.Sigma;
    delete[] all_data.U_cpy;
    delete[] all_data.V_cpy;
    all_data.U     = nullptr;
    all_data.V     = nullptr;
    all_data.Sigma = nullptr;
    all_data.U_cpy = nullptr;
    all_data.V_cpy = nullptr;
}

// This routine computes the residual norm error, consisting of two parts (one of which) vanishes
// in exact precision. Target_rank defines size of U, V as returned by ABRIK; target_rank <= target_rank.
template <typename T, typename TestData>
static T
residual_error_vectors_comp(TestData &all_data, int64_t target_rank) {
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
    all_data.A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, target_rank, n, 1.0, all_data.V, n, -1.0, all_data.U_cpy, m);

    // A'U - VS
    // Scale columns of V by S
    for (int i = 0; i < target_rank; ++i)
        blas::scal(n, all_data.Sigma[i], &all_data.V_cpy[i * n], 1);
    // Compute A'U(:, 1:target_rank) - VS(1:target_rank).
    all_data.A_linop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, target_rank, m, 1.0, all_data.U, m, -1.0, all_data.V_cpy, n);

    T nrm1 = lapack::lange(Norm::Fro, m, target_rank, all_data.U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, n, target_rank, all_data.V_cpy, n);

    return std::hypot(nrm1, nrm2);
}

// Assesses the quality of approximation of singular values specifically
template <typename T, typename TestData>
static T
residual_error_values_comp(TestData &all_data, int64_t target_rank, T triplet_error) {
    
    T spectral_gap;
    if (target_rank == 1) {
        spectral_gap = all_data.Sigma[0];
    } else {
        spectral_gap = all_data.Sigma[target_rank - 2] - all_data.Sigma[target_rank - 1];
    }

    return triplet_error * spectral_gap;
}

template <typename T, typename RNG, RandBLAS::sparse_data::SparseMatrix SpMat>
static void call_all_algs(
    int64_t num_runs,
    int64_t b_sz,
    int64_t num_matmuls,
    int64_t target_rank,
    ABRIK_algorithm_objects<T, RNG> &all_algs,
    ABRIK_benchmark_data<T, SpMat> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename, 
    std::string input_path) {

    int i;
    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;

    // Additional params setup.
    // Matrices R or S that give us the singular value spectrum returned by ABRIK will be of size b_sz * num_krylov_iters / 2.
    // These matrices will be full-rank.
    // Hence, target_rank = b_sz * num_krylov_iters / 2 
    // ABRIK.max_krylov_iters = (int) ((target_rank * 2) / b_sz);
    // 
    // Instead of the above approach, we now pre-specify the maximum number of Krylov iters that we allow for in num_matmuls.
    all_algs.ABRIK.max_krylov_iters = (int) num_matmuls;
    all_algs.ABRIK.num_threads_min = 4;
    all_algs.ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();
    // Useful for all sparse matrices except 0.
    //all_algs.ABRIK.qr_exp = Subroutines::QR_explicit::cqrrt;
    
    // timing vars
    long dur_ABRIK = 0;
    long dur_svds = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    T residual_err_vec_SVDS  = 0;
    T residual_err_val_SVDS  = 0;
    T residual_err_vec_ABRIK = 0;
    T residual_err_val_ABRIK = 0;

    int64_t singular_triplets_target_ABRIK = 0;
    int64_t singular_triplets_found_SVDS   = 0;
    int64_t singular_triplets_target_SVDS  = 0;

    for (i = 0; i < num_runs; ++i) {
        printf("\nBlock size %ld, num matmuls %ld. Iteration %d start.\n", b_sz, num_matmuls, i);

        // Running ABRIK
        auto start_ABRIK = steady_clock::now();
        all_algs.ABRIK.call(m, n, *all_data.A_input, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state_alg);
        auto stop_ABRIK = steady_clock::now();
        dur_ABRIK = duration_cast<microseconds>(stop_ABRIK - start_ABRIK).count();
        printf("TOTAL TIME FOR ABRIK %ld\n", dur_ABRIK);

        // This is in case the number of singular triplets is smaller than the target rank
        singular_triplets_target_ABRIK = std::min(target_rank, all_algs.ABRIK.singular_triplets_found);
        printf("Singular triplets: %ld\n", singular_triplets_target_ABRIK);

        residual_err_vec_ABRIK = residual_error_vectors_comp<T>(all_data, singular_triplets_target_ABRIK);
        printf("ABRIK sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(target_rank): %.16e\n", residual_err_vec_ABRIK);
        residual_err_val_ABRIK = residual_error_values_comp<T>(all_data, singular_triplets_target_ABRIK, residual_err_vec_ABRIK);
        printf("ABRIK resigual error * spectral gap: %.16e\n", residual_err_val_ABRIK);

        state_alg = state;
        state_gen = state;
        data_regen(all_data, state_gen);


        // Running SVDS
        auto start_svds = steady_clock::now();

        // Despite my earlier expectations, estimating a larger number of 
        // singular triplets via SVDS does improve the quality of the first singular triplets.
        // As such, aiming for just the "target rank" would be unfair.

        // Below line also accounts for the case when number of singular triplets is smaller than the target rank.
        singular_triplets_found_SVDS = std::min((int64_t ) (b_sz * num_matmuls / 2), n-2);

        Spectra::PartialSVDSolver<SpMatrix> svds(all_data.A_spectra, singular_triplets_found_SVDS, std::min(2 * singular_triplets_found_SVDS, n-1));
        svds.compute();
        auto stop_svds = steady_clock::now();
        dur_svds = duration_cast<microseconds>(stop_svds - start_svds).count();
        printf("TOTAL TIME FOR SVDS %ld\n", dur_svds);

        // Copy data from Spectra (Eigen) format to the nomal C++.
        Matrix U_spectra = svds.matrix_U(singular_triplets_found_SVDS);
        Matrix V_spectra = svds.matrix_V(singular_triplets_found_SVDS);
        Vector S_spectra = svds.singular_values();

        all_data.U     = new T[m * singular_triplets_found_SVDS]();
        all_data.V     = new T[n * singular_triplets_found_SVDS]();
        all_data.Sigma = new T[m * singular_triplets_found_SVDS]();

        Eigen::Map<Matrix>(all_data.U, m, singular_triplets_found_SVDS)  = U_spectra;
        Eigen::Map<Matrix>(all_data.V, n, singular_triplets_found_SVDS)  = V_spectra;
        Eigen::Map<Vector>(all_data.Sigma, singular_triplets_found_SVDS) = S_spectra;

        singular_triplets_target_SVDS = std::min(target_rank, singular_triplets_found_SVDS);

        residual_err_vec_SVDS = residual_error_vectors_comp<T>(all_data, singular_triplets_target_SVDS);
        printf("SVDS sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(target_rank): %.16e\n", residual_err_vec_SVDS);
        residual_err_val_SVDS = residual_error_values_comp<T>(all_data, singular_triplets_target_SVDS, residual_err_vec_SVDS);
        printf("SVDS resigual error * spectral gap: %.16e\n", residual_err_val_SVDS);        

        state_alg = state;
        state_gen = state;
        data_regen(all_data, state_gen);

        std::ofstream file(output_filename, std::ios::app);
        file << b_sz << ",  " << all_algs.ABRIK.max_krylov_iters  <<  ",  " << target_rank << ",  " 
        << residual_err_vec_ABRIK << ",  " << residual_err_val_ABRIK <<  ",  " << dur_ABRIK    << ",  " 
        << residual_err_vec_SVDS << ",  " << residual_err_val_SVDS << ",  " << dur_svds    << ",\n";
    }
}

int main(int argc, char *argv[]) {

    if (argc < 9) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << "<output_directory_path> <input_matrix_path> <num_runs> <target_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return 1;
    }

    double submatrix_dim_ratio = 0.5;

    int num_runs              = std::stol(argv[3]);
    int64_t target_rank       = std::stol(argv[4]);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[5]); ++i)
        b_sz.push_back(std::stoi(argv[i + 7]));
    // Save elements in string for logging purposes
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[6]); ++i)
        matmuls.push_back(std::stoi(argv[i + 7 + std::stol(argv[5])]));
    // Save elements in string for logging purposes
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();
    double tol                 = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                 = RandBLAS::RNGState();
    auto state_constant        = state;

    // Read the input fast matrix market data
    // The idea is that input_mat_coo will be automatically freed at the end of function execution
    auto input_mat_coo = from_matrix_market<double>(std::string(argv[2]));
    auto m = input_mat_coo.n_rows * submatrix_dim_ratio;
    auto n = input_mat_coo.n_cols * submatrix_dim_ratio;

    // Convert coo into csc matrix - this will grab the leading principal submatrix
    // depending on what m and n were set to.
    auto input_mat_csc = format_conversion<double>(m, n, input_mat_coo);

    // Allocate basic workspace.
    ABRIK_benchmark_data<double, RandBLAS::sparse_data::CSCMatrix<double>> all_data(input_mat_csc, m, n, tol);
    all_data.A_input = &input_mat_csc;
    // Read matrix into spectra format
    //from_matrix_market<double>(std::string(argv[2]), all_data.A_spectra);
    from_matrix_market_leading_submatrix<double>(std::string(argv[2]), all_data.A_spectra, submatrix_dim_ratio);

    // Declare ABRIK object
    ABRIK_algorithm_objects<double, r123::Philox4x32> all_algs(false, false, tol);

    printf("Finished data preparation\n");
    // Declare a data file
    std::string output_filename = "_ABRIK_speed_comparisons_sparse_num_info_lines_" + std::to_string(6) + ".txt";
    std::string path;
    if (std::string(argv[1]) != ".") {
        path = argv[1] + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the ABRIK speed comparison benchmark, recording the time it takes to perform ABRIK and alternative methods for low-rank SVD, specifically on sparse matrices."
              "\nFile format: 15 columns, showing krylov block size, nummber of matmuls permitted, and num svals and svecs to approximate, followed by the residual error, standard lowrank error and execution time for all algorithms (ABRIK, SVDS)"
              "\n Rows correspond to algorithm runs with Krylov block sizes varying as specified, and numbers of matmuls varying as specified per each block size, with num_runs repititions of each number of matmuls."
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
            call_all_algs(num_runs, b_sz[i], matmuls[j], target_rank, all_algs, all_data, state_constant, path, std::string(argv[2]));
        }
        j = 0;
    }
}
