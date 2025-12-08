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

} // namespace RandLAPACK_demos
