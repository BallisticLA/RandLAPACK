#pragma once

// Matrix Market input for the benchmarks of this directory (through the extras loader, so it
// needs Eigen and fast_matrix_market).

#include "../../extras/misc/ext_util.hh"
#include <RandBLAS.hh>

#include <cstdint>
#include <string>

namespace RandLAPACK {
namespace bench {

/// Loads a Matrix Market file as CSR; sets m, n, nnz.
template <typename T>
RandBLAS::sparse_data::csr::CSRMatrix<T> load_csr(const std::string& path,
                                                  int64_t& m, int64_t& n, int64_t& nnz) {
    auto coo = RandLAPACK_extras::coo_from_matrix_market<T>(path);
    m = coo.n_rows; n = coo.n_cols; nnz = coo.nnz;
    RandBLAS::sparse_data::csr::CSRMatrix<T> csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
    return csr;
}

} // namespace bench
} // namespace RandLAPACK
