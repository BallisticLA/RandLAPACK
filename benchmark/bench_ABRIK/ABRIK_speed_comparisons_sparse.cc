/*
ABRIK speed comparison benchmark (sparse) - runs ABRIK and optionally SVDS (Spectra) / GESDD
on sparse input matrices read from Matrix Market format.
Precision (float or double) is specified as the first CLI argument.

Output: CSV file with '#'-prefixed metadata header, column names, then data rows.
ABRIK allocates U/V/Sigma with new[] internally -> cleanup with delete[].

Residual metrics:
  - Vector: sqrt(||AV - US||^2_F + ||A'U - VS||^2_F) / sigma_{target_rank}
  - Value:  vector_error * spectral_gap / sigma_1
Timings in microseconds.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>

// External libs includes
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Spectra/contrib/PartialSVDSolver.h>

// Traits struct mapping scalar type T to Eigen matrix/vector/sparse types.
template <typename T> struct EigenTypes;
template <> struct EigenTypes<double> {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using SpMatrix = Eigen::SparseMatrix<double>;
};
template <> struct EigenTypes<float> {
    using Matrix = Eigen::MatrixXf;
    using Vector = Eigen::VectorXf;
    using SpMatrix = Eigen::SparseMatrix<float>;
};

using Subroutines = RandLAPACK::ABRIKSubroutines;

// Helper function to ensure directory exists (creates parent directories if needed)
void ensure_directory_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        if (info.st_mode & S_IFDIR) return;
        std::cerr << "Error: " << path << " exists but is not a directory" << std::endl;
        return;
    }
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos && pos > 0) {
        ensure_directory_exists(path.substr(0, pos));
    }
    if (mkdir(path.c_str(), 0755) != 0) {
        std::cerr << "Warning: Could not create directory " << path
                  << " (error: " << strerror(errno) << ")" << std::endl;
    } else {
        std::cout << "Created output directory: " << path << std::endl;
    }
}

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
    typename EigenTypes<T>::SpMatrix A_spectra;

    ABRIK_benchmark_data(SpMat& input_mat, int64_t m, int64_t n, T tol) :
    A_spectra(m, n),
    A_linop(m, n, input_mat)
    {
        U     = nullptr;
        V     = nullptr;
        Sigma = nullptr;
        row       = m;
        col       = n;
        tolerance = tol;
    }

    ~ABRIK_benchmark_data() {}
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
        file_stream, n_rows, n_cols, rows, cols, vals
    );

    RandBLAS::sparse_data::coo::COOMatrix<T> out(n_rows, n_cols);
    reserve_coo(vals.size(), out);
    for (int i = 0; i < out.nnz; ++i) {
        out.rows[i] = rows[i];
        out.cols[i] = cols[i];
        out.vals[i] = vals[i];
    }

    return out;
}

template <typename T>
void from_matrix_market(std::string fn, typename EigenTypes<T>::SpMatrix& A) {
    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows, cols, vals
    );

    std::vector<Eigen::Triplet<T>> tripletList;
    for (size_t i = 0; i < vals.size(); ++i)
        tripletList.push_back(Eigen::Triplet<T>(rows[i], cols[i], vals[i]));

    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

template <typename T>
void from_matrix_market_leading_submatrix(std::string fn, typename EigenTypes<T>::SpMatrix& A, T submatrix_dim_ratio) {
    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows, cols, vals
    );

    int64_t m = n_rows * submatrix_dim_ratio;
    int64_t n = n_cols * submatrix_dim_ratio;

    std::vector<Eigen::Triplet<T>> tripletList;
    for (size_t i = 0; i < vals.size(); ++i) {
        if (rows[i] < m && cols[i] < n)
            tripletList.emplace_back(rows[i], cols[i], vals[i]);
    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

template <typename T>
RandBLAS::CSCMatrix<T> format_conversion(int64_t m, int64_t n, RandBLAS::COOMatrix<T>& input_mat_coo)
{
    RandBLAS::COOMatrix<T> input_mat_transformed(m, n);

    int64_t nnz_sub = 0;
    for (int64_t i = 0; i < input_mat_coo.nnz; ++i) {
        if (input_mat_coo.rows[i] < m && input_mat_coo.cols[i] < n)
            ++nnz_sub;
    }

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

    RandBLAS::CSCMatrix<T> input_mat_csc(m, n);
    RandBLAS::conversions::coo_to_csc(input_mat_transformed, input_mat_csc);

    return input_mat_csc;
}

// Computes the residual norm error: sqrt(||AV - US||^2_F + ||A'U - VS||^2_F) / sigma_{target_rank}.
// Uses SpLinOp for sparse matvec. Scratch buffers are allocated and freed locally.
template <typename T, typename LinOp>
static T
residual_error_vectors_comp(LinOp& A_linop, int64_t m, int64_t n,
                            T* U, T* V, T* Sigma, int64_t target_rank) {

    T* U_cpy = new T[m * target_rank]();
    T* V_cpy = new T[n * target_rank]();

    lapack::lacpy(MatrixType::General, m, target_rank, U, m, U_cpy, m);
    lapack::lacpy(MatrixType::General, n, target_rank, V, n, V_cpy, n);

    // AV - US: scale columns of U_cpy by Sigma, then compute AV - US
    for (int i = 0; i < target_rank; ++i)
        blas::scal(m, Sigma[i], &U_cpy[m * i], 1);
    A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, target_rank, n,
            (T)1, V, n, (T)-1, U_cpy, m);

    // A'U - VS: scale columns of V_cpy by Sigma, then compute A'U - VS
    for (int i = 0; i < target_rank; ++i)
        blas::scal(n, Sigma[i], &V_cpy[i * n], 1);
    A_linop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, target_rank, m,
            (T)1, U, m, (T)-1, V_cpy, n);

    T nrm1 = lapack::lange(Norm::Fro, m, target_rank, U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, n, target_rank, V_cpy, n);

    delete[] U_cpy;
    delete[] V_cpy;

    return std::hypot(nrm1, nrm2) / Sigma[target_rank - 1];
}

// Assesses the quality of approximation of singular values specifically.
template <typename T>
static T
residual_error_values_comp(T* Sigma, int64_t target_rank, T triplet_error) {
    T spectral_gap;
    if (target_rank == 1) {
        spectral_gap = Sigma[0];
    } else {
        spectral_gap = Sigma[target_rank - 2] - Sigma[target_rank - 1];
    }
    return triplet_error * spectral_gap / Sigma[0];
}

// Helper function to write matrices to Matrix Market format (MATLAB-compatible)
template <typename T>
void write_matrix_to_file(const std::string& filename, const T* matrix, int64_t rows, int64_t cols, bool is_vector = false) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    file << "%%MatrixMarket matrix array real general\n";
    file << rows << " " << cols << "\n";
    file << std::scientific << std::setprecision(16);
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            file << matrix[i + j * rows] << "\n";
        }
    }
    file.close();
    std::cout << "Successfully wrote " << (is_vector ? "vector" : "matrix")
              << " (" << rows << " x " << cols << ") to " << filename << std::endl;
}

// Perform direct SVD using LAPACK's GESDD on dense matrix
template <typename T>
void compute_direct_gesdd(const typename EigenTypes<T>::SpMatrix& A_sparse, int64_t m, int64_t n, int64_t target_rank,
                          std::string output_dir, bool write_output_matrices) {
    printf("\n========== Running Direct GESDD ==========\n");

    // Convert sparse to dense
    printf("Converting sparse matrix to dense...\n");
    T* A_dense = new T[m * n]();

    for (int k = 0; k < A_sparse.outerSize(); ++k) {
        for (typename EigenTypes<T>::SpMatrix::InnerIterator it(A_sparse, k); it; ++it) {
            A_dense[it.row() + it.col() * m] = it.value();
        }
    }

    int64_t min_mn = std::min(m, n);

    printf("Computing full SVD with GESDD...\n");
    auto start_gesdd = steady_clock::now();

    T* S_full = new T[min_mn]();
    T* U_full = new T[m * min_mn]();
    T* VT_full = new T[min_mn * n]();

    T* A_copy = new T[m * n]();
    lapack::lacpy(MatrixType::General, m, n, A_dense, m, A_copy, m);

    lapack::gesdd(lapack::Job::SomeVec, m, n, A_copy, m, S_full, U_full, m, VT_full, min_mn);

    auto stop_gesdd = steady_clock::now();
    long dur_gesdd = duration_cast<microseconds>(stop_gesdd - start_gesdd).count();
    printf("TOTAL TIME FOR GESDD: %ld microseconds\n", dur_gesdd);

    // Transpose VT to get V
    T* V_full = new T[n * min_mn]();
    for (int64_t i = 0; i < min_mn; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            V_full[j + i * n] = VT_full[i + j * min_mn];
        }
    }

    printf("GESDD completed. First 5 singular values:\n");
    for (int64_t i = 0; i < std::min((int64_t)5, min_mn); ++i) {
        printf("  Sigma[%ld] = %.16e\n", i, S_full[i]);
    }

    if (write_output_matrices) {
        std::string prefix = output_dir + "/GESDD";
        write_matrix_to_file(prefix + "_U.mtx", U_full, m, min_mn, false);
        write_matrix_to_file(prefix + "_V.mtx", V_full, n, min_mn, false);
        write_matrix_to_file(prefix + "_Sigma.mtx", S_full, min_mn, 1, true);
    }

    delete[] A_dense;
    delete[] A_copy;
    delete[] VT_full;
    delete[] V_full;
    delete[] S_full;
    delete[] U_full;

    printf("==========================================\n\n");
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
    std::ofstream &outfile,
    std::string input_path,
    std::string output_dir,
    bool write_output_matrices) {

    auto m   = all_data.row;
    auto n   = all_data.col;

    // Additional params setup.
    all_algs.ABRIK.max_krylov_iters = (int) num_matmuls;
    all_algs.ABRIK.num_threads_min = 4;
    all_algs.ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();

    // timing vars
    long dur_ABRIK = 0;
    long dur_svds = 0;

    auto state_alg = state;

    T residual_err_vec_SVDS  = 0;
    T residual_err_val_SVDS  = 0;
    T residual_err_vec_ABRIK = 0;
    T residual_err_val_ABRIK = 0;

    int64_t singular_triplets_target_ABRIK = 0;
    int64_t singular_triplets_found_SVDS   = 0;
    int64_t singular_triplets_target_SVDS  = 0;

    for (int i = 0; i < num_runs; ++i) {
        printf("\nBlock size %ld, num matmuls %ld. Iteration %d start.\n", b_sz, num_matmuls, i);

        // ---- ABRIK ----
        // ABRIK allocates U, V, Sigma with new[] internally.
        auto start_ABRIK = steady_clock::now();
        all_algs.ABRIK.call(m, n, *all_data.A_input, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state_alg);
        auto stop_ABRIK = steady_clock::now();
        dur_ABRIK = duration_cast<microseconds>(stop_ABRIK - start_ABRIK).count();
        printf("TOTAL TIME FOR ABRIK %ld\n", dur_ABRIK);

        singular_triplets_target_ABRIK = std::min(target_rank, all_algs.ABRIK.singular_triplets_found);
        printf("Singular triplets: %ld\n", singular_triplets_target_ABRIK);

        residual_err_vec_ABRIK = residual_error_vectors_comp<T>(all_data.A_linop, m, n, all_data.U, all_data.V, all_data.Sigma, singular_triplets_target_ABRIK);
        printf("ABRIK sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sigma_{k}: %.16e\n", residual_err_vec_ABRIK);
        residual_err_val_ABRIK = residual_error_values_comp<T>(all_data.Sigma, singular_triplets_target_ABRIK, residual_err_vec_ABRIK);
        printf("ABRIK residual error * spectral gap / sigma[0]: %.16e\n", residual_err_val_ABRIK);

        // Write ABRIK output matrices to files if requested (only on first run)
        if (write_output_matrices && i == 0) {
            int64_t full_abrik_output_size = all_algs.ABRIK.singular_triplets_found;
            std::string prefix = output_dir + "/ABRIK_bsz" + std::to_string(b_sz) + "_mm" + std::to_string(num_matmuls);
            write_matrix_to_file(prefix + "_U.mtx", all_data.U, m, full_abrik_output_size, false);
            write_matrix_to_file(prefix + "_V.mtx", all_data.V, n, full_abrik_output_size, false);
            write_matrix_to_file(prefix + "_Sigma.mtx", all_data.Sigma, full_abrik_output_size, 1, true);
        }

        // Cleanup ABRIK outputs (new[])
        delete[] all_data.U;     all_data.U     = nullptr;
        delete[] all_data.V;     all_data.V     = nullptr;
        delete[] all_data.Sigma; all_data.Sigma = nullptr;

        state_alg = state;

        /*
        // ---- SVDS (Spectra, sparse) ----
        // When un-commenting, use EigenTypes<T>::SpMatrix for the Spectra solver.
        using ESpMatrix = typename EigenTypes<T>::SpMatrix;
        using EMatrix   = typename EigenTypes<T>::Matrix;
        using EVector   = typename EigenTypes<T>::Vector;

        singular_triplets_found_SVDS = std::min((int64_t) (b_sz * num_matmuls / 2), n - 2);

        auto start_svds = steady_clock::now();
        Spectra::PartialSVDSolver<ESpMatrix> svds(all_data.A_spectra, singular_triplets_found_SVDS, std::min(2 * singular_triplets_found_SVDS, n - 1));
        svds.compute();
        auto stop_svds = steady_clock::now();
        dur_svds = duration_cast<microseconds>(stop_svds - start_svds).count();
        printf("TOTAL TIME FOR SVDS %ld\n", dur_svds);

        EMatrix U_spectra = svds.matrix_U(singular_triplets_found_SVDS);
        EMatrix V_spectra = svds.matrix_V(singular_triplets_found_SVDS);
        EVector S_spectra = svds.singular_values();

        all_data.U     = new T[m * singular_triplets_found_SVDS]();
        all_data.V     = new T[n * singular_triplets_found_SVDS]();
        all_data.Sigma = new T[singular_triplets_found_SVDS]();

        Eigen::Map<EMatrix>(all_data.U, m, singular_triplets_found_SVDS)  = U_spectra;
        Eigen::Map<EMatrix>(all_data.V, n, singular_triplets_found_SVDS)  = V_spectra;
        Eigen::Map<EVector>(all_data.Sigma, singular_triplets_found_SVDS) = S_spectra;

        singular_triplets_target_SVDS = std::min(target_rank, singular_triplets_found_SVDS);

        residual_err_vec_SVDS = residual_error_vectors_comp<T>(all_data.A_linop, m, n, all_data.U, all_data.V, all_data.Sigma, singular_triplets_target_SVDS);
        printf("SVDS sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sigma_{k}: %.16e\n", residual_err_vec_SVDS);
        residual_err_val_SVDS = residual_error_values_comp<T>(all_data.Sigma, singular_triplets_target_SVDS, residual_err_vec_SVDS);
        printf("SVDS residual error * spectral gap / sigma[0]: %.16e\n", residual_err_val_SVDS);

        // Cleanup SVDS outputs (new[])
        delete[] all_data.U;     all_data.U     = nullptr;
        delete[] all_data.V;     all_data.V     = nullptr;
        delete[] all_data.Sigma; all_data.Sigma = nullptr;

        state_alg = state;
        */

        // Write CSV data row
        outfile << b_sz << ", " << all_algs.ABRIK.max_krylov_iters << ", " << target_rank << ", "
                << residual_err_vec_ABRIK << ", " << residual_err_val_ABRIK << ", " << dur_ABRIK << ", "
                << residual_err_vec_SVDS << ", " << residual_err_val_SVDS << ", " << dur_svds << "\n";
        outfile.flush();
    }
}

template <typename T>
static void run_benchmark(int argc, char *argv[]) {

    if (argc < 12) {
        std::cerr << "Usage: " << argv[0] << " <precision> <output_directory_path> <input_matrix_path> <num_runs> <target_rank> <run_gesdd> <write_matrices> <submatrix_dim_ratio> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        std::cerr << "  run_gesdd: 1 to run direct GESDD, 0 to skip" << std::endl;
        std::cerr << "  write_matrices: 1 to write U,V,Sigma to files, 0 to skip" << std::endl;
        std::cerr << "  submatrix_dim_ratio: ratio of input matrix to use (e.g., 0.5 for half, 1.0 for full matrix)" << std::endl;
        return;
    }

    T submatrix_dim_ratio = (T)std::stod(argv[8]);

    int num_runs              = std::stol(argv[4]);
    int64_t target_rank       = std::stol(argv[5]);
    bool run_gesdd            = (std::stoi(argv[6]) != 0);
    bool write_matrices       = (std::stoi(argv[7]) != 0);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[9]); ++i)
        b_sz.push_back(std::stoi(argv[i + 11]));
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[10]); ++i)
        matmuls.push_back(std::stoi(argv[i + 11 + std::stol(argv[9])]));
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();
    T tol                      = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state                 = RandBLAS::RNGState();
    auto state_constant        = state;

    // Ensure output directory exists if we're writing matrices
    if (write_matrices) {
        ensure_directory_exists(std::string(argv[2]));
    }

    // Read the input matrix market data
    auto input_mat_coo = from_matrix_market<T>(std::string(argv[3]));
    auto m = (int64_t)(input_mat_coo.n_rows * submatrix_dim_ratio);
    auto n = (int64_t)(input_mat_coo.n_cols * submatrix_dim_ratio);

    // Convert COO to CSC (grabs leading principal submatrix)
    auto input_mat_csc = format_conversion<T>(m, n, input_mat_coo);

    // Allocate basic workspace.
    ABRIK_benchmark_data<T, RandBLAS::sparse_data::CSCMatrix<T>> all_data(input_mat_csc, m, n, tol);
    all_data.A_input = &input_mat_csc;
    // Read matrix into Spectra (Eigen sparse) format
    from_matrix_market_leading_submatrix<T>(std::string(argv[3]), all_data.A_spectra, submatrix_dim_ratio);

    // Declare ABRIK object
    ABRIK_algorithm_objects<T, r123::Philox4x32> all_algs(false, false, tol);

    printf("Finished data preparation\n");

    // Build output file path
    std::string output_filename = "_ABRIK_speed_comparisons_sparse.csv";
    std::string path;
    if (std::string(argv[2]) != ".") {
        path = argv[2] + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Write metadata header (prefixed with # for easy parsing)
    file << "# ABRIK Speed Comparison Benchmark (Sparse)\n"
         << "# Precision: " << argv[1] << "\n"
         << "# Input matrix: " << argv[3] << "\n"
         << "# Input size: " << m << " x " << n << " (submatrix ratio: " << submatrix_dim_ratio << ")\n"
         << "# Target rank: " << target_rank << "\n"
         << "# Krylov block sizes: " << b_sz_string << "\n"
         << "# Matmul counts: " << matmuls_string << "\n"
         << "# Runs per configuration: " << num_runs << "\n"
         << "# Tolerance: " << tol << "\n"
         << "# Run GESDD: " << (run_gesdd ? "yes" : "no") << "\n"
         << "# Write matrices: " << (write_matrices ? "yes" : "no") << "\n"
         << "# Residual metric (vec): sqrt(||AV - US||^2_F + ||A'U - VS||^2_F) / sigma_{target_rank}\n"
         << "# Residual metric (val): vec_error * spectral_gap / sigma_1\n"
         << "# Timings in microseconds\n";
    // Write CSV column header
    file << "b_sz, num_matmuls, target_rank, "
         << "err_vec_ABRIK, err_val_ABRIK, dur_ABRIK, "
         << "err_vec_SVDS, err_val_SVDS, dur_SVDS\n";
    file.flush();

    // Run direct GESDD if requested
    if (run_gesdd) {
        compute_direct_gesdd<T>(all_data.A_spectra, m, n, target_rank, std::string(argv[2]), write_matrices);
    }

    size_t i = 0, j = 0;
    for (; i < b_sz.size(); ++i) {
        for (; j < matmuls.size(); ++j) {
            call_all_algs(num_runs, b_sz[i], matmuls[j], target_rank, all_algs, all_data, state_constant, file, std::string(argv[3]), std::string(argv[2]), write_matrices);
        }
        j = 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <precision: double|float> <output_directory_path> <input_matrix_path> <num_runs> <target_rank> <run_gesdd> <write_matrices> <submatrix_dim_ratio> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return 1;
    }

    std::string precision = argv[1];
    if (precision == "double") {
        run_benchmark<double>(argc, argv);
    } else if (precision == "float") {
        run_benchmark<float>(argc, argv);
    } else {
        std::cerr << "Error: precision must be 'double' or 'float', got '" << precision << "'" << std::endl;
        return 1;
    }
    return 0;
}
