#pragma once

// RandLAPACK Demos - Utilities
//
// This header-only library provides format conversion utilities for working with
// external matrix libraries (Eigen, fast_matrix_market) in demos and benchmarks.
//
// These functions are kept separate from core RandLAPACK to avoid adding external
// dependencies to the main library.

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

namespace RandLAPACK_demos {

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
template <typename T, typename RNG>
void generate_spd_matrix_file(
    const std::string& filename,
    int64_t n,
    T cond_num,
    RandBLAS::RNGState<RNG> &state
) {
    // Generate SPD matrix using RandLAPACK core function
    std::vector<T> A(n * n);
    RandLAPACK::gen::gen_spd_mat(n, cond_num, A.data(), state);

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

} // namespace RandLAPACK_demos
