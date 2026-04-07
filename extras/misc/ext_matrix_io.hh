/*
Benchmark matrix loader — auto-detects file format from extension:
  .mtx  → Matrix Market (dense array or sparse coordinate) via fast_matrix_market
  .txt  → Whitespace-delimited text (dense, row-major) via rl_matrix_io.hh

Dense input: provides a T* buffer (column-major).
Sparse input: provides CSC data for SparseLinOp + Eigen::SparseMatrix for Spectra.
No materialization — all algorithms use native representations.
*/

#ifndef BENCH_MATRIX_IO_HH
#define BENCH_MATRIX_IO_HH

#include "RandLAPACK.hh"
#include "rl_linops.hh"
#include "rl_matrix_io.hh"

#include <RandBLAS.hh>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <Eigen/SparseCore>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

namespace BenchIO {

template <typename T>
struct LoadedMatrix {
    int64_t m = 0;
    int64_t n = 0;
    bool is_sparse = false;

    // Dense data (for dense input only)
    std::vector<T> dense_data;

    // Sparse data (for sparse input only)
    RandBLAS::sparse_data::CSCMatrix<T>* csc = nullptr;
    Eigen::SparseMatrix<T>* eigen_sparse = nullptr;  // for Spectra

    ~LoadedMatrix() {
        delete csc;
        delete eigen_sparse;
    }

    T* data() { return dense_data.data(); }
};

/// Load a matrix from file. If sub_ratio < 1.0, extract the top-left
/// (m*sub_ratio) x (n*sub_ratio) submatrix (sparse: filter triplets, dense: truncate).
template <typename T>
LoadedMatrix<T> load_matrix(const std::string& path, double sub_ratio = 1.0) {
    LoadedMatrix<T> result;

    std::string ext = path.substr(path.find_last_of('.'));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".mtx") {
        std::ifstream fs(path);
        if (!fs.is_open())
            throw std::runtime_error("Cannot open: " + path);

        fast_matrix_market::matrix_market_header header;
        fast_matrix_market::read_header(fs, header);
        result.m = header.nrows;
        result.n = header.ncols;
        fs.clear();
        fs.seekg(0);

        if (header.format == fast_matrix_market::coordinate) {
            result.is_sparse = true;
            std::vector<int64_t> rows, cols;
            std::vector<T> vals;
            fast_matrix_market::read_matrix_market_triplet(
                fs, header.nrows, header.ncols, rows, cols, vals);
            fs.close();

            // Apply submatrix ratio: keep only entries in top-left block
            if (sub_ratio < 1.0) {
                int64_t sub_m = (int64_t)(result.m * sub_ratio);
                int64_t sub_n = (int64_t)(result.n * sub_ratio);
                std::vector<int64_t> filt_rows, filt_cols;
                std::vector<T> filt_vals;
                for (size_t i = 0; i < vals.size(); ++i) {
                    if (rows[i] < sub_m && cols[i] < sub_n) {
                        filt_rows.push_back(rows[i]);
                        filt_cols.push_back(cols[i]);
                        filt_vals.push_back(vals[i]);
                    }
                }
                rows = std::move(filt_rows);
                cols = std::move(filt_cols);
                vals = std::move(filt_vals);
                result.m = sub_m;
                result.n = sub_n;
            }

            int64_t nnz = (int64_t) vals.size();

            // Build RandBLAS CSC (for SparseLinOp)
            RandBLAS::sparse_data::COOMatrix<T> coo(result.m, result.n);
            coo.reserve(nnz);
            for (int64_t i = 0; i < nnz; ++i) {
                coo.rows[i] = rows[i];
                coo.cols[i] = cols[i];
                coo.vals[i] = vals[i];
            }
            result.csc = new RandBLAS::sparse_data::CSCMatrix<T>(result.m, result.n);
            RandBLAS::sparse_data::conversions::coo_to_csc(coo, *result.csc);

            // Build Eigen sparse matrix (for Spectra)
            using Triplet = Eigen::Triplet<T>;
            std::vector<Triplet> eigen_trips(nnz);
            for (int64_t i = 0; i < nnz; ++i)
                eigen_trips[i] = Triplet(rows[i], cols[i], vals[i]);
            result.eigen_sparse = new Eigen::SparseMatrix<T>(result.m, result.n);
            result.eigen_sparse->setFromTriplets(eigen_trips.begin(), eigen_trips.end());

            printf("Loaded %s: %ld x %ld, sparse (nnz=%ld%s)\n",
                   path.c_str(), result.m, result.n, nnz,
                   sub_ratio < 1.0 ? ", submatrix" : "");
        } else {
            fast_matrix_market::read_matrix_market_array(
                fs, header.nrows, header.ncols,
                result.dense_data, fast_matrix_market::col_major);
            fs.close();
            printf("Loaded %s: %ld x %ld, dense (mtx)\n",
                   path.c_str(), result.m, result.n);
        }
    } else if (ext == ".bin") {
        const char* cpath = path.c_str();
        int64_t m = 0, n = 0;
        RandLAPACK::gen::read_bin_matrix<T>(m, n, nullptr, cpath, true);
        result.m = m;
        result.n = n;
        result.dense_data.resize(m * n);
        RandLAPACK::gen::read_bin_matrix<T>(m, n, result.data(), cpath, false);
        printf("Loaded %s: %ld x %ld, dense (bin)\n", cpath, result.m, result.n);
    } else {
        const char* cpath = path.c_str();
        int64_t m = 0, n = 0;
        RandLAPACK::gen::read_txt_matrix<T>(m, n, nullptr, cpath, true);
        result.m = m;
        result.n = n;
        result.dense_data.resize(m * n);
        RandLAPACK::gen::read_txt_matrix<T>(m, n, result.data(), cpath, false);
        printf("Loaded %s: %ld x %ld, dense (txt)\n", cpath, result.m, result.n);
    }

    return result;
}

} // namespace BenchIO

#endif // BENCH_MATRIX_IO_HH
