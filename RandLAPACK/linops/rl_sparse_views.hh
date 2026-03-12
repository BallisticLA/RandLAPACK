#pragma once

#include "RandBLAS/sparse_data/base.hh"

#include <RandBLAS.hh>
#include <vector>
#include <cstdint>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*          Sparse Block View Utilities                  */
/*                                                       */
/*********************************************************/

/// @brief Non-owning view of a contiguous row range of a CSR matrix.
template <typename T, typename sint_t = int64_t>
struct CSRRowBlockView {
    std::vector<sint_t> rowptr;
    T* vals;
    sint_t* colidxs;
    int64_t n_rows;
    int64_t n_cols;
    int64_t nnz;
    RandBLAS::sparse_data::CSRMatrix<T, sint_t> as_csr() {
        return RandBLAS::sparse_data::CSRMatrix<T, sint_t>(
            n_rows, n_cols, nnz, vals, rowptr.data(), colidxs);
    }
};

template <typename T, typename sint_t>
CSRRowBlockView<T, sint_t> csr_row_block(
    RandBLAS::sparse_data::CSRMatrix<T, sint_t>& A,
    int64_t row_start, int64_t row_count
) {
    randblas_require(row_start >= 0 && row_count > 0);
    randblas_require(row_start + row_count <= A.n_rows);
    sint_t base = A.rowptr[row_start];
    int64_t block_nnz = A.rowptr[row_start + row_count] - base;
    std::vector<sint_t> rebased(row_count + 1);
    for (int64_t i = 0; i <= row_count; ++i)
        rebased[i] = A.rowptr[row_start + i] - base;
    return CSRRowBlockView<T, sint_t>{std::move(rebased), A.vals + base, A.colidxs + base, row_count, A.n_cols, block_nnz};
}

template <typename T, typename sint_t = int64_t>
struct CSRColBlock {
    std::vector<T> vals;
    std::vector<sint_t> rowptr;
    std::vector<sint_t> colidxs;
    int64_t n_rows, n_cols, nnz;
    RandBLAS::sparse_data::CSRMatrix<T, sint_t> as_csr() {
        return RandBLAS::sparse_data::CSRMatrix<T, sint_t>(n_rows, n_cols, nnz, vals.data(), rowptr.data(), colidxs.data());
    }
};

template <typename T, typename sint_t>
CSRColBlock<T, sint_t> csr_col_block(
    RandBLAS::sparse_data::CSRMatrix<T, sint_t>& A,
    int64_t col_start, int64_t col_count
) {
    randblas_require(col_start >= 0 && col_count > 0 && col_start + col_count <= A.n_cols);
    int64_t col_end = col_start + col_count;
    std::vector<sint_t> rowptr(A.n_rows + 1, 0);
    for (int64_t i = 0; i < A.n_rows; ++i)
        for (sint_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k)
            if (A.colidxs[k] >= col_start && A.colidxs[k] < col_end) rowptr[i + 1]++;
    for (int64_t i = 0; i < A.n_rows; ++i) rowptr[i + 1] += rowptr[i];
    int64_t block_nnz = rowptr[A.n_rows];
    std::vector<T> vals(block_nnz);
    std::vector<sint_t> colidxs(block_nnz);
    for (int64_t i = 0; i < A.n_rows; ++i) {
        sint_t write_pos = rowptr[i];
        for (sint_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k)
            if (A.colidxs[k] >= col_start && A.colidxs[k] < col_end) {
                vals[write_pos] = A.vals[k];
                colidxs[write_pos] = A.colidxs[k] - col_start;
                write_pos++;
            }
    }
    return CSRColBlock<T, sint_t>{std::move(vals), std::move(rowptr), std::move(colidxs), A.n_rows, col_count, block_nnz};
}

template <typename T, typename sint_t = int64_t>
struct CSCColBlockView {
    std::vector<sint_t> colptr;
    T* vals;
    sint_t* rowidxs;
    int64_t n_rows, n_cols, nnz;
    RandBLAS::sparse_data::CSCMatrix<T, sint_t> as_csc() {
        return RandBLAS::sparse_data::CSCMatrix<T, sint_t>(n_rows, n_cols, nnz, vals, rowidxs, colptr.data());
    }
};

template <typename T, typename sint_t>
CSCColBlockView<T, sint_t> csc_col_block(
    RandBLAS::sparse_data::CSCMatrix<T, sint_t>& A,
    int64_t col_start, int64_t col_count
) {
    randblas_require(col_start >= 0 && col_count > 0 && col_start + col_count <= A.n_cols);
    sint_t base = A.colptr[col_start];
    int64_t block_nnz = A.colptr[col_start + col_count] - base;
    std::vector<sint_t> rebased(col_count + 1);
    for (int64_t i = 0; i <= col_count; ++i)
        rebased[i] = A.colptr[col_start + i] - base;
    return CSCColBlockView<T, sint_t>{std::move(rebased), A.vals + base, A.rowidxs + base, A.n_rows, col_count, block_nnz};
}

template <typename T, typename sint_t = int64_t>
struct CSCRowBlock {
    std::vector<T> vals;
    std::vector<sint_t> colptr;
    std::vector<sint_t> rowidxs;
    int64_t n_rows, n_cols, nnz;
    RandBLAS::sparse_data::CSCMatrix<T, sint_t> as_csc() {
        return RandBLAS::sparse_data::CSCMatrix<T, sint_t>(n_rows, n_cols, nnz, vals.data(), rowidxs.data(), colptr.data());
    }
};

template <typename T, typename sint_t>
CSCRowBlock<T, sint_t> csc_row_block(
    RandBLAS::sparse_data::CSCMatrix<T, sint_t>& A,
    int64_t row_start, int64_t row_count
) {
    randblas_require(row_start >= 0 && row_count > 0 && row_start + row_count <= A.n_rows);
    int64_t row_end = row_start + row_count;
    std::vector<sint_t> colptr(A.n_cols + 1, 0);
    for (int64_t j = 0; j < A.n_cols; ++j)
        for (sint_t k = A.colptr[j]; k < A.colptr[j + 1]; ++k)
            if (A.rowidxs[k] >= row_start && A.rowidxs[k] < row_end) colptr[j + 1]++;
    for (int64_t j = 0; j < A.n_cols; ++j) colptr[j + 1] += colptr[j];
    int64_t block_nnz = colptr[A.n_cols];
    std::vector<T> vals(block_nnz);
    std::vector<sint_t> rowidxs(block_nnz);
    for (int64_t j = 0; j < A.n_cols; ++j) {
        sint_t write_pos = colptr[j];
        for (sint_t k = A.colptr[j]; k < A.colptr[j + 1]; ++k)
            if (A.rowidxs[k] >= row_start && A.rowidxs[k] < row_end) {
                vals[write_pos] = A.vals[k];
                rowidxs[write_pos] = A.rowidxs[k] - row_start;
                write_pos++;
            }
    }
    return CSCRowBlock<T, sint_t>{std::move(vals), std::move(colptr), std::move(rowidxs), row_count, A.n_cols, block_nnz};
}

/// Split a CSR matrix into num_blocks equal-sized row blocks.
/// Requires A.n_rows is divisible by num_blocks.
template <typename T, typename sint_t>
std::vector<CSRRowBlockView<T, sint_t>> csr_split_row_blocks(
    RandBLAS::sparse_data::CSRMatrix<T, sint_t>& A,
    int64_t num_blocks
) {
    randblas_require(num_blocks > 0 && A.n_rows % num_blocks == 0);
    int64_t block_size = A.n_rows / num_blocks;
    std::vector<CSRRowBlockView<T, sint_t>> blocks;
    blocks.reserve(num_blocks);
    for (int64_t b = 0; b < num_blocks; ++b)
        blocks.push_back(csr_row_block(A, b * block_size, block_size));
    return blocks;
}

/// Split a CSC matrix into num_blocks equal-sized column blocks.
/// Requires A.n_cols is divisible by num_blocks.
template <typename T, typename sint_t>
std::vector<CSCColBlockView<T, sint_t>> csc_split_col_blocks(
    RandBLAS::sparse_data::CSCMatrix<T, sint_t>& A,
    int64_t num_blocks
) {
    randblas_require(num_blocks > 0 && A.n_cols % num_blocks == 0);
    int64_t block_size = A.n_cols / num_blocks;
    std::vector<CSCColBlockView<T, sint_t>> blocks;
    blocks.reserve(num_blocks);
    for (int64_t b = 0; b < num_blocks; ++b)
        blocks.push_back(csc_col_block(A, b * block_size, block_size));
    return blocks;
}

/// Split a CSR matrix into num_blocks equal-sized column blocks (cross-direction, owned).
/// Requires A.n_cols is divisible by num_blocks.
template <typename T, typename sint_t>
std::vector<CSRColBlock<T, sint_t>> csr_split_col_blocks(
    RandBLAS::sparse_data::CSRMatrix<T, sint_t>& A,
    int64_t num_blocks
) {
    randblas_require(num_blocks > 0 && A.n_cols % num_blocks == 0);
    int64_t block_size = A.n_cols / num_blocks;
    std::vector<CSRColBlock<T, sint_t>> blocks;
    blocks.reserve(num_blocks);
    for (int64_t b = 0; b < num_blocks; ++b)
        blocks.push_back(csr_col_block(A, b * block_size, block_size));
    return blocks;
}

/// Split a CSC matrix into num_blocks equal-sized row blocks (cross-direction, owned).
/// Requires A.n_rows is divisible by num_blocks.
template <typename T, typename sint_t>
std::vector<CSCRowBlock<T, sint_t>> csc_split_row_blocks(
    RandBLAS::sparse_data::CSCMatrix<T, sint_t>& A,
    int64_t num_blocks
) {
    randblas_require(num_blocks > 0 && A.n_rows % num_blocks == 0);
    int64_t block_size = A.n_rows / num_blocks;
    std::vector<CSCRowBlock<T, sint_t>> blocks;
    blocks.reserve(num_blocks);
    for (int64_t b = 0; b < num_blocks; ++b)
        blocks.push_back(csc_row_block(A, b * block_size, block_size));
    return blocks;
}

} // end namespace RandLAPACK::linops
