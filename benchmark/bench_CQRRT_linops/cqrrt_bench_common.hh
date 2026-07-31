// cqrrt_bench_common.hh — shared utilities for CQRRT linop benchmarks
#pragma once

#include "RandLAPACK.hh"
#include "../../extras/misc/ext_util.hh"
#include <RandBLAS.hh>

#include <ctime>
#include <iostream>
#include <string>
#include <vector>

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

// Load a sparse matrix and emit the standard "Loading <label> from <path>... done (m x n, nnz=N)"
// progress messages used across all CQRRT benchmarks.
template <typename T>
static RandBLAS::sparse_data::csr::CSRMatrix<T> load_csr_verbose(
    const std::string& label, const std::string& path,
    int64_t& m, int64_t& n, int64_t& nnz)
{
    std::cout << "Loading " << label << " from " << path << "... " << std::flush;
    auto csr = load_csr<T>(path, m, n, nnz);
    std::cout << "done (" << m << " x " << n << ", nnz=" << nnz << ")\n";
    return csr;
}

// Return a YYYYMMDD_HHMMSS timestamp string suitable for naming output files.
inline std::string make_run_timestamp() {
    char buf[64];
    std::time_t now = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    return std::string(buf);
}

