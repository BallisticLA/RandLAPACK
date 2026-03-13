#pragma once

// RandLAPACK Extras - Utilities
//
// Format conversion utilities for working with external matrix libraries
// (Eigen, fast_matrix_market). Kept separate from core RandLAPACK to avoid
// adding external dependencies to the main library.

#include <RandLAPACK.hh>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iomanip>

// External library includes
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <Eigen/Sparse>
#include <Eigen/Core>

namespace RandLAPACK_extras {

/***********************************************************************
 *                                                                     *
 *              MATRIX MARKET I/O (RandBLAS COO format)                *
 *                                                                     *
 ***********************************************************************/

/// Read a Matrix Market file into a RandBLAS COO matrix.
/// Loads the entire matrix from a .mtx file into COO (coordinate) format.
///
/// @tparam T - Scalar type (double, float, etc.)
///
/// @param[in] filename - Path to Matrix Market file (.mtx)
///
/// @return COO matrix containing the loaded data
///
template <typename T>
RandBLAS::sparse_data::coo::COOMatrix<T> coo_from_matrix_market(const std::string& filename) {
    using namespace RandBLAS::sparse_data;

    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        throw std::runtime_error("Cannot open Matrix Market file: " + filename);
    }

    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows, cols, vals
    );

    coo::COOMatrix<T> output(n_rows, n_cols);
    coo::reserve_coo(vals.size(), output);

    for (int64_t i = 0; i < output.nnz; ++i) {
        output.rows[i] = rows[i];
        output.cols[i] = cols[i];
        output.vals[i] = vals[i];
    }

    return output;
}

/***********************************************************************
 *                                                                     *
 *                MATRIX MARKET I/O (Eigen format)                     *
 *                                                                     *
 ***********************************************************************/

/// Read a Matrix Market file into an Eigen sparse matrix.
/// Loads the entire matrix from a .mtx file into Eigen's SparseMatrix format.
///
/// @tparam T - Scalar type
///
/// @param[in] filename - Path to Matrix Market file
/// @param[out] A - Eigen SparseMatrix to populate (will be resized)
///
template <typename T>
void eigen_sparse_from_matrix_market(
    const std::string& filename,
    Eigen::SparseMatrix<T>& A
) {
    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        throw std::runtime_error("Cannot open Matrix Market file: " + filename);
    }

    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows, cols, vals
    );

    // Convert to Eigen triplets
    std::vector<Eigen::Triplet<T>> tripletList;
    tripletList.reserve(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        tripletList.emplace_back(rows[i], cols[i], vals[i]);
    }

    // Build sparse matrix
    A.resize(n_rows, n_cols);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

/// Read a leading principal submatrix from Matrix Market file into Eigen sparse matrix.
/// Extracts only entries within the specified submatrix dimensions.
///
/// @tparam T - Scalar type
///
/// @param[in] filename - Path to Matrix Market file
/// @param[out] A - Eigen SparseMatrix to populate (will be resized)
/// @param[in] submatrix_dim_ratio - Fraction of dimensions to extract (e.g., 0.5 for half)
///
template <typename T>
void eigen_sparse_submatrix_from_matrix_market(
    const std::string& filename,
    Eigen::SparseMatrix<T>& A,
    T submatrix_dim_ratio
) {
    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        throw std::runtime_error("Cannot open Matrix Market file: " + filename);
    }

    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows, cols, vals
    );

    // Calculate submatrix dimensions
    int64_t m = static_cast<int64_t>(n_rows * submatrix_dim_ratio);
    int64_t n = static_cast<int64_t>(n_cols * submatrix_dim_ratio);

    // Create triplets only for entries within submatrix
    std::vector<Eigen::Triplet<T>> tripletList;
    for (size_t i = 0; i < vals.size(); ++i) {
        if (rows[i] < m && cols[i] < n) {
            tripletList.emplace_back(rows[i], cols[i], vals[i]);
        }
    }

    // Build sparse matrix
    A.resize(m, n);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

/***********************************************************************
 *                                                                     *
 *           SPARSE FORMAT CONVERSIONS (RandBLAS formats)              *
 *                                                                     *
 ***********************************************************************/

/// Extract leading principal submatrix from COO matrix and convert to CSC.
/// This function extracts an m×n leading principal submatrix from a larger
/// COO matrix and converts it to CSC format for efficient sparse operations.
///
/// @tparam T - Scalar type
///
/// @param[in] m - Number of rows in output submatrix
/// @param[in] n - Number of columns in output submatrix
/// @param[in] input_mat_coo - Input COO matrix (must be at least m×n)
///
/// @return CSC matrix containing the m×n leading principal submatrix
///
template <typename T>
RandBLAS::sparse_data::csc::CSCMatrix<T> coo_submatrix_to_csc(
    int64_t m,
    int64_t n,
    const RandBLAS::sparse_data::coo::COOMatrix<T>& input_mat_coo
) {
    using namespace RandBLAS::sparse_data;

    // Create submatrix
    coo::COOMatrix<T> submatrix(m, n);

    // Count nonzeros in leading principal submatrix
    int64_t nnz_sub = 0;
    for (int64_t i = 0; i < input_mat_coo.nnz; ++i) {
        if (input_mat_coo.rows[i] < m && input_mat_coo.cols[i] < n) {
            ++nnz_sub;
        }
    }

    // Allocate storage for submatrix
    coo::reserve_coo(nnz_sub, submatrix);

    // Copy entries within submatrix bounds
    int64_t ell = 0;
    for (int64_t i = 0; i < input_mat_coo.nnz; ++i) {
        if (input_mat_coo.rows[i] < m && input_mat_coo.cols[i] < n) {
            submatrix.rows[ell] = input_mat_coo.rows[i];
            submatrix.cols[ell] = input_mat_coo.cols[i];
            submatrix.vals[ell] = input_mat_coo.vals[i];
            ++ell;
        }
    }

    // Convert to CSC format
    csc::CSCMatrix<T> output_csc(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csc(submatrix, output_csc);

    return output_csc;
}

/***********************************************************************
 *                                                                     *
 *               SPD MATRIX GENERATION AND I/O                         *
 *                                                                     *
 ***********************************************************************/

/// Generate a random SPD matrix and save to Matrix Market file.
/// Uses RandLAPACK::gen::gen_spd_mat() to generate the matrix, then writes
/// it to a Matrix Market coordinate format file.
///
/// @tparam T - Scalar type (double, float, etc.)
/// @tparam RNG - Random number generator type
///
/// @param[in] filename - Path to output Matrix Market file
/// @param[in] n - Dimension of square SPD matrix (n × n)
/// @param[in] cond_num - Target condition number for the matrix
/// @param[in] state - RNG state for reproducible generation
///
/// @brief Helper function to generate SPD matrix with specified condition number
///
/// Generates an SPD matrix A = Q * D * Q^T where:
/// - D is diagonal with eigenvalues: λ_i = 1 + (cond_num - 1) * (i / (n-1))^2
///   This gives λ_1 = 1, λ_n = cond_num, so κ(A) = cond_num
/// - Q is a random orthogonal matrix generated via QR of a Gaussian matrix
///
/// This implementation does NOT sum duplicates (avoids the PR 115 bug).
///
// Note: gen_spd_mat is now in RandLAPACK::gen namespace (rl_gen.hh)
// This is just a convenience wrapper for extras
template <typename T, typename RNG>
void gen_spd_mat(
    int64_t n,
    T cond_num,
    T* A,
    RandBLAS::RNGState<RNG> &state
) {
    RandLAPACK::gen::gen_spd_mat(n, cond_num, A, state);
}

template <typename T, typename RNG>
void generate_spd_matrix_file(
    const std::string& filename,
    int64_t n,
    T cond_num,
    RandBLAS::RNGState<RNG> &state
) {
    // Generate SPD matrix using helper function
    std::vector<T> A(n * n);
    gen_spd_mat(n, cond_num, A.data(), state);

    // Write to Matrix Market file (coordinate format)
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Set full double precision (17 significant digits) to preserve SPD property
    file << std::scientific << std::setprecision(17);

    file << "%%MatrixMarket matrix coordinate real general\n";

    // Count nonzeros (entries with magnitude > threshold)
    int64_t nnz = 0;
    for (int64_t i = 0; i < n * n; ++i) {
        if (std::abs(A[i]) > 1e-14) {
            ++nnz;
        }
    }

    file << n << " " << n << " " << nnz << "\n";

    // Write nonzero entries (Matrix Market uses 1-based indexing)
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < n; ++i) {
            if (std::abs(A[i + j * n]) > 1e-14) {
                file << (i + 1) << " " << (j + 1) << " " << A[i + j * n] << "\n";
            }
        }
    }

    file.close();
}

/***********************************************************************
 *                                                                     *
 *       GENERAL INVERTIBLE MATRIX GENERATION AND I/O                  *
 *                                                                     *
 ***********************************************************************/

/// Generate a random invertible (non-symmetric) matrix and save to Matrix Market file.
///
/// Generates A = Q1 * D * Q2^T where:
/// - D is diagonal with singular values: sigma_i = 1 + (cond_num - 1) * (i / (n-1))^2
///   This gives sigma_1 = 1, sigma_n = cond_num, so kappa(A) = cond_num
/// - Q1, Q2 are independent random orthogonal matrices (from QR of Gaussian matrices)
///
/// Unlike gen_spd_mat, this produces a general (non-symmetric) matrix since Q1 != Q2.
///
/// @tparam T - Scalar type (double, float, etc.)
/// @tparam RNG - Random number generator type
///
/// @param[in] filename - Path to output Matrix Market file
/// @param[in] n - Dimension of square matrix (n x n)
/// @param[in] cond_num - Target condition number for the matrix
/// @param[in] state - RNG state for reproducible generation
///
template <typename T, typename RNG>
void generate_invertible_matrix_file(
    const std::string& filename,
    int64_t n,
    T cond_num,
    RandBLAS::RNGState<RNG> &state
) {
    // Step 1: Generate singular values with desired condition number
    std::vector<T> singvals(n);
    singvals[0] = 1.0;  // Smallest singular value
    if (n > 1) {
        singvals[n-1] = cond_num;  // Largest singular value
        for (int64_t i = 1; i < n - 1; ++i) {
            T t = static_cast<T>(i) / static_cast<T>(n - 1);
            singvals[i] = 1.0 + (cond_num - 1.0) * t * t;
        }
    }

    // Step 2: Generate two independent random orthogonal matrices Q1, Q2
    std::vector<T> Q1(n * n);
    std::vector<T> Q2(n * n);
    std::vector<T> tau(n);

    auto d1 = RandBLAS::DenseDist(n, n);
    state = RandBLAS::fill_dense(d1, Q1.data(), state);
    lapack::geqrf(n, n, Q1.data(), n, tau.data());
    lapack::orgqr(n, n, n, Q1.data(), n, tau.data());

    auto d2 = RandBLAS::DenseDist(n, n);
    state = RandBLAS::fill_dense(d2, Q2.data(), state);
    lapack::geqrf(n, n, Q2.data(), n, tau.data());
    lapack::orgqr(n, n, n, Q2.data(), n, tau.data());

    // Step 3: Form A = Q1 * D * Q2^T
    // First compute Q1_scaled = Q1 * D (scale columns of Q1 by singular values)
    std::vector<T> Q1_scaled(n * n);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < n; ++i) {
            Q1_scaled[i + j * n] = Q1[i + j * n] * singvals[j];
        }
    }

    // Then A = Q1_scaled * Q2^T
    std::vector<T> A(n * n);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
               n, n, n, (T)1.0, Q1_scaled.data(), n, Q2.data(), n, (T)0.0, A.data(), n);

    // Step 4: Write to Matrix Market file (coordinate format)
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Set full double precision to preserve invertibility
    file << std::scientific << std::setprecision(17);

    file << "%%MatrixMarket matrix coordinate real general\n";

    // Count nonzeros (entries with magnitude > threshold)
    int64_t nnz = 0;
    for (int64_t i = 0; i < n * n; ++i) {
        if (std::abs(A[i]) > 1e-14) {
            ++nnz;
        }
    }

    file << n << " " << n << " " << nnz << "\n";

    // Write nonzero entries (Matrix Market uses 1-based indexing)
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < n; ++i) {
            if (std::abs(A[i + j * n]) > 1e-14) {
                file << (i + 1) << " " << (j + 1) << " " << A[i + j * n] << "\n";
            }
        }
    }

    file.close();
}

/***********************************************************************
 *                                                                     *
 *              LINEAR OPERATOR DIAGNOSTICS                            *
 *                                                                     *
 ***********************************************************************/

/// Compute and print condition number diagnostics for a linear operator.
///
/// Materializes the operator, computes condition number both raw and after
/// column normalization, and prints a diagnostic summary.
///
template <typename T, typename LinOp>
void print_condition_diagnostics(LinOp& A_linop, int64_t m, int64_t n,
                                 const std::string& label = "operator") {
    printf("\nCondition number diagnostics for %s:\n", label.c_str());

    auto A_dense = RandLAPACK::testing::materialize_linop<T>(A_linop, m, n);

    // Compute column norms
    std::vector<T> col_norms(n);
    for (int64_t j = 0; j < n; ++j)
        col_norms[j] = blas::nrm2(m, &A_dense[j * m], 1);

    T cn_min = *std::min_element(col_norms.begin(), col_norms.end());
    T cn_max = *std::max_element(col_norms.begin(), col_norms.end());
    printf("  Column norm range: [%.6e, %.6e], ratio: %.6e\n",
           (double)cn_min, (double)cn_max, (double)(cn_max / cn_min));

    // Copy for column-normalized version (gesdd is destructive)
    std::vector<T> A_normed(A_dense);
    for (int64_t j = 0; j < n; ++j)
        blas::scal(m, (T)1.0 / col_norms[j], &A_normed[j * m], 1);

    // SVD on raw matrix
    auto sigma = RandLAPACK::testing::compute_singular_values<T>(A_dense.data(), m, n);
    printf("  Raw:     kappa = %.6e (sigma_max=%.6e, sigma_min=%.6e)\n",
           (double)(sigma[0] / sigma[n - 1]), (double)sigma[0], (double)sigma[n - 1]);

    // SVD on column-normalized matrix
    auto sigma_normed = RandLAPACK::testing::compute_singular_values<T>(A_normed.data(), m, n);
    printf("  ColNorm: kappa = %.6e (sigma_max=%.6e, sigma_min=%.6e)\n",
           (double)(sigma_normed[0] / sigma_normed[n - 1]),
           (double)sigma_normed[0], (double)sigma_normed[n - 1]);

    printf("\n");
}

} // namespace RandLAPACK_extras
