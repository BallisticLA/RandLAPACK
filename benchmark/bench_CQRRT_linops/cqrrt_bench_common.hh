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

