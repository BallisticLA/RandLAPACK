// cqrrt_bench_common.hh — shared utilities for CQRRT linop benchmarks
#pragma once

#include "RandLAPACK.hh"
#include "../../extras/misc/ext_util.hh"
#include <RandBLAS.hh>

#include <vector>
#include <string>

// Load a Matrix Market file into a CSRMatrix. Sets m, n, nnz on exit.
template <typename T>
static RandBLAS::sparse_data::csr::CSRMatrix<T> load_csr(
    const std::string& path, int64_t& m, int64_t& n, int64_t& nnz)
{
    auto coo = RandLAPACK_extras::coo_from_matrix_market<T>(path);
    m = coo.n_rows; n = coo.n_cols; nnz = coo.nnz;
    RandBLAS::sparse_data::csr::CSRMatrix<T> csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
    return csr;
}

// Return an n x n identity matrix (column-major).
template <typename T>
static std::vector<T> make_eye(int64_t n) {
    std::vector<T> I(n * n, T(0));
    RandLAPACK::util::eye(n, n, I.data());
    return I;
}

// Fill lower triangle of a column-major symmetric n x n matrix from its upper triangle.
template <typename T>
static void fill_lower_from_upper(T* M, int64_t n) {
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            M[i + j * n] = M[j + i * n];
}
