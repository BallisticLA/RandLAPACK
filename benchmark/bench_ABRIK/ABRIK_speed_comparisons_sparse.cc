/*
ABRIK speed comparison benchmark (sparse) - runs ABRIK and optionally SVDS (Spectra) / GESDD
on sparse input matrices read from Matrix Market format.
Precision (float or double) is specified as the first CLI argument.

Output: CSV file with '#'-prefixed metadata header, column names, then data rows.
ABRIK allocates U/V/Sigma with new[] internally -> cleanup with delete[].

Residual metric: sqrt(||S^{-1}AV - U||^2_F + ||(A'U)S^{-1} - V||^2_F).
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
#include <ctime>
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

    // S^{-1}AV - U
    A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, target_rank, n,
            (T)1, V, n, (T)0, U_cpy, m);
    for (int i = 0; i < target_rank; ++i)
        blas::scal(m, (T)1 / Sigma[i], &U_cpy[m * i], 1);
    blas::axpy(m * target_rank, (T)-1, U, 1, U_cpy, 1);

    // (A'U)S^{-1} - V
    A_linop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, target_rank, m,
            (T)1, U, m, (T)0, V_cpy, n);
    for (int i = 0; i < target_rank; ++i)
        blas::scal(n, (T)1 / Sigma[i], &V_cpy[i * n], 1);
    blas::axpy(n * target_rank, (T)-1, V, 1, V_cpy, 1);

    T nrm1 = lapack::lange(Norm::Fro, m, target_rank, U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, n, target_rank, V_cpy, n);

    delete[] U_cpy;
    delete[] V_cpy;

    return std::hypot(nrm1, nrm2);
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

// ---- ABRIK sweep (sparse) ----
// Loops over (b_sz, num_matmuls, run). ABRIK allocates with new[] internally.
// Sparse input is not modified — no regen needed.
template <typename T, typename RNG, RandBLAS::sparse_data::SparseMatrix SpMat>
static void run_abrik_sweep_sparse(
    int64_t num_runs,
    std::vector<int64_t> &b_sz_vec,
    std::vector<int64_t> &matmuls_vec,
    int64_t target_rank,
    ABRIK_algorithm_objects<T, RNG> &all_algs,
    ABRIK_benchmark_data<T, SpMat> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile,
    std::string output_dir,
    bool write_output_matrices) {

    auto m = all_data.row;
    auto n = all_data.col;

    for (auto b_sz : b_sz_vec) {
        for (auto num_matmuls : matmuls_vec) {
            all_algs.ABRIK.max_krylov_iters = (int) num_matmuls;

            for (int i = 0; i < num_runs; ++i) {
                auto state_alg = state;
                printf("\nABRIK: b_sz=%ld, mm=%ld, run %d\n", b_sz, num_matmuls, i);

                auto start = steady_clock::now();
                all_algs.ABRIK.call(m, n, *all_data.A_input, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state_alg);
                auto stop = steady_clock::now();
                long dur = duration_cast<microseconds>(stop - start).count();

                int64_t k_found = std::min(target_rank, all_algs.ABRIK.singular_triplets_found);
                T err = residual_error_vectors_comp<T>(all_data.A_linop, m, n, all_data.U, all_data.V, all_data.Sigma, k_found);
                printf("  err=%.16e, time=%ld us\n", err, dur);

                outfile << "ABRIK, " << b_sz << ", " << num_matmuls << ", 0, "
                        << target_rank << ", " << err << ", " << dur << "\n";
                outfile.flush();

                // Write ABRIK output matrices to files if requested (only on first run of first config)
                if (write_output_matrices && i == 0) {
                    int64_t full_size = all_algs.ABRIK.singular_triplets_found;
                    std::string prefix = output_dir + "/ABRIK_bsz" + std::to_string(b_sz) + "_mm" + std::to_string(num_matmuls);
                    write_matrix_to_file(prefix + "_U.mtx", all_data.U, m, full_size, false);
                    write_matrix_to_file(prefix + "_V.mtx", all_data.V, n, full_size, false);
                    write_matrix_to_file(prefix + "_Sigma.mtx", all_data.Sigma, full_size, 1, true);
                }

                delete[] all_data.U;     all_data.U     = nullptr;
                delete[] all_data.V;     all_data.V     = nullptr;
                delete[] all_data.Sigma; all_data.Sigma = nullptr;
            }
        }
    }
}

// ---- SVDS sweep (Spectra, sparse) ----
// Runs num_runs times with nev = target_rank.
template <typename T, RandBLAS::sparse_data::SparseMatrix SpMat>
static void run_svds_sweep_sparse(
    int64_t num_runs,
    int64_t target_rank,
    ABRIK_benchmark_data<T, SpMat> &all_data,
    std::ofstream &outfile) {

    using ESpMatrix = typename EigenTypes<T>::SpMatrix;
    using EMatrix   = typename EigenTypes<T>::Matrix;
    using EVector   = typename EigenTypes<T>::Vector;

    auto m = all_data.row;
    auto n = all_data.col;
    int64_t nev = target_rank;
    int64_t ncv = std::min(2 * nev + 1, n - 1);

    for (int i = 0; i < num_runs; ++i) {
        printf("\nSVDS: nev=%ld, ncv=%ld, run %d\n", nev, ncv, i);

        auto start = steady_clock::now();
        Spectra::PartialSVDSolver<ESpMatrix> svds(all_data.A_spectra, nev, ncv);
        svds.compute(1000, all_data.tolerance);
        auto stop = steady_clock::now();
        long dur = duration_cast<microseconds>(stop - start).count();

        EMatrix U_spectra = svds.matrix_U(nev);
        EMatrix V_spectra = svds.matrix_V(nev);
        EVector S_spectra = svds.singular_values();

        all_data.U     = new T[m * nev]();
        all_data.V     = new T[n * nev]();
        all_data.Sigma = new T[nev]();

        Eigen::Map<EMatrix>(all_data.U, m, nev) = U_spectra;
        Eigen::Map<EMatrix>(all_data.V, n, nev) = V_spectra;
        Eigen::Map<EVector>(all_data.Sigma, nev) = S_spectra;

        T err = residual_error_vectors_comp<T>(all_data.A_linop, m, n, all_data.U, all_data.V, all_data.Sigma, nev);
        printf("  err=%.16e, time=%ld us\n", err, dur);

        outfile << "SVDS, 0, 0, 0, "
                << target_rank << ", " << err << ", " << dur << "\n";
        outfile.flush();

        delete[] all_data.U;     all_data.U     = nullptr;
        delete[] all_data.V;     all_data.V     = nullptr;
        delete[] all_data.Sigma; all_data.Sigma = nullptr;
    }
}

template <typename T>
static void run_benchmark(int argc, char *argv[]) {

    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input_matrix_path> <num_runs>"
                  << " <target_rank> <run_gesdd> <write_matrices> <submatrix_dim_ratio>"
                  << " <num_block_sizes> <num_matmul_sizes>"
                  << " <block_sizes...> <matmul_sizes...>" << std::endl;
        std::cerr << "  run_gesdd: 1 to run direct GESDD, 0 to skip" << std::endl;
        std::cerr << "  write_matrices: 1 to write U,V,Sigma to files, 0 to skip" << std::endl;
        std::cerr << "  submatrix_dim_ratio: ratio of input matrix to use (e.g., 0.5 for half, 1.0 for full)" << std::endl;
        return;
    }

    T submatrix_dim_ratio = (T)std::stod(argv[8]);

    int num_runs              = std::stol(argv[4]);
    int64_t target_rank       = std::stol(argv[5]);
    bool run_gesdd            = (std::stoi(argv[6]) != 0);
    bool write_matrices       = (std::stoi(argv[7]) != 0);
    int64_t num_block_sizes   = std::stol(argv[9]);
    int64_t num_matmul_sizes  = std::stol(argv[10]);

    int64_t expected_argc = 11 + num_block_sizes + num_matmul_sizes;
    if (argc < expected_argc) {
        std::cerr << "Error: expected " << expected_argc << " arguments, got " << argc << std::endl;
        return;
    }

    // Parse block sizes (for ABRIK)
    int64_t offset = 11;
    std::vector<int64_t> b_sz;
    for (int64_t i = 0; i < num_block_sizes; ++i)
        b_sz.push_back(std::stol(argv[offset + i]));
    offset += num_block_sizes;

    // Parse matmul counts (for ABRIK)
    std::vector<int64_t> matmuls;
    for (int64_t i = 0; i < num_matmul_sizes; ++i)
        matmuls.push_back(std::stol(argv[offset + i]));

    // Build display strings for metadata
    auto vec_to_string = [](const std::vector<int64_t> &v) {
        std::ostringstream oss;
        for (const auto &val : v) oss << val << ", ";
        return oss.str();
    };
    std::string b_sz_string    = vec_to_string(b_sz);
    std::string matmuls_string = vec_to_string(matmuls);

    T tol              = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state         = RandBLAS::RNGState();
    auto state_constant = state;

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

    // Generate date/time prefix for output filename
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    std::string output_filename = std::string(date_prefix) + "ABRIK_speed_comparisons_sparse.csv";
    std::string path;
    if (std::string(argv[2]) != ".") {
        path = std::string(argv[2]) + "/" + output_filename;
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
         << "# Residual metric: sqrt(||S^{-1}AV - U||^2_F + ||(A'U)S^{-1} - V||^2_F)\n"
         << "# Timings in microseconds\n";

    // Write CSV column header (unified format matching dense benchmark)
    file << "algorithm, b_sz, num_matmuls, p, target_rank, error, duration_us\n";
    file.flush();

    // Run direct GESDD if requested (separate from CSV output)
    if (run_gesdd) {
        compute_direct_gesdd<T>(all_data.A_spectra, m, n, target_rank, std::string(argv[2]), write_matrices);
    }

    // Run each algorithm sweep independently
    printf("\n=== ABRIK sweep (%zu block sizes x %zu matmul counts x %d runs) ===\n",
           b_sz.size(), matmuls.size(), num_runs);
    run_abrik_sweep_sparse(num_runs, b_sz, matmuls, target_rank, all_algs, all_data, state_constant, file, std::string(argv[2]), write_matrices);

    printf("\n=== SVDS sweep (%d runs, nev=%ld) ===\n", num_runs, target_rank);
    run_svds_sweep_sparse(num_runs, target_rank, all_data, file);

    printf("\nBenchmark complete. Output: %s\n", path.c_str());
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input_matrix_path> <num_runs>"
                  << " <target_rank> <run_gesdd> <write_matrices> <submatrix_dim_ratio>"
                  << " <num_block_sizes> <num_matmul_sizes>"
                  << " <block_sizes...> <matmul_sizes...>" << std::endl;
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
