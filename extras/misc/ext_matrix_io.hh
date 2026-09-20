#ifndef BENCH_MATRIX_IO_HH
#define BENCH_MATRIX_IO_HH

#include "rl_exceptions.hh"
#include "rl_matrix_io.hh"

#include <RandBLAS.hh>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace BenchIO {

/// Owns either compact column-major dense data or equivalent CSC and Eigen
/// sparse matrices. Moving transfers ownership; copying is not supported.
template <typename T>
struct LoadedMatrix {
    int64_t m = 0;
    int64_t n = 0;
    bool is_sparse = false;
    std::vector<T> dense_data;
    std::unique_ptr<RandBLAS::sparse_data::CSCMatrix<T>> csc;
    std::unique_ptr<Eigen::SparseMatrix<T>> eigen_sparse;

    LoadedMatrix() = default;
    LoadedMatrix(const LoadedMatrix&) = delete;
    LoadedMatrix& operator=(const LoadedMatrix&) = delete;
    LoadedMatrix(LoadedMatrix&&) noexcept = default;
    LoadedMatrix& operator=(LoadedMatrix&&) noexcept = default;

    T* data() { return dense_data.data(); }
    const T* data() const { return dense_data.data(); }
};

namespace matrix_io_detail {

[[noreturn]] inline void fail(const std::string& path, const std::string& reason) {
    throw RandLAPACK::Error(reason + ": " + path);
}

template <typename T>
size_t dense_size(int64_t m, int64_t n, const std::string& path) {
    if (m <= 0 || n <= 0)
        fail(path, "Matrix dimensions must be positive");
    const uint64_t max_elements = std::min({
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T)),
        static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max() / sizeof(T)),
        static_cast<uint64_t>(std::vector<T>().max_size())});
    if (static_cast<uint64_t>(m) > max_elements / static_cast<uint64_t>(n))
        fail(path, "Matrix dimensions overflow the output size");
    return static_cast<size_t>(m) * static_cast<size_t>(n);
}

inline int64_t sub_dimension(int64_t dimension, double ratio, const std::string& path) {
    if (dimension <= 0)
        fail(path, "Matrix dimensions must be positive");
    if (ratio == 1.0)
        return dimension;
    const double scaled = static_cast<double>(dimension) * ratio;
    if (scaled < 1)
        fail(path, "Submatrix dimensions must be positive");
    // Avoid an out-of-range cast if floating-point rounding reaches dimension.
    if (scaled >= static_cast<double>(dimension))
        return dimension;
    return static_cast<int64_t>(scaled);
}

// fast_matrix_market computes nrows*ncols while parsing an array header.
// Check that product first; its parser still validates the complete header.
template <typename T>
void check_array_header(std::ifstream& file, const std::string& path) {
    std::string line, banner, object, format;
    std::getline(file, line);
    std::istringstream(line) >> banner >> object >> format;
    auto lowercase = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
    std::transform(object.begin(), object.end(), object.begin(), lowercase);
    std::transform(format.begin(), format.end(), format.begin(), lowercase);
    if (object == "matrix" && format == "array") {
        while (std::getline(file, line)) {
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '%')
                continue;
            int64_t m = 0, n = 0;
            if (std::istringstream(line) >> m >> n)
                dense_size<T>(m, n, path);
            break;
        }
    }
    file.clear();
    file.seekg(0);
    if (!file)
        fail(path, "Cannot rewind Matrix Market file");
}

// FMM does not check for truncated symmetric arrays, and an extra skew array
// entry can reach its handler with an invalid coordinate. Matrix Market stores
// one array value per nonblank line; check the triangular count before parsing.
inline void check_symmetric_array_body(
    std::ifstream& file, const fast_matrix_market::matrix_market_header& header,
    const std::string& path
) {
    if (file.bad())
        fail(path, "Failed to read Matrix Market file");
    // A zero-entry skew array may end at its dimension line without a newline.
    file.clear();
    const auto body_start = file.tellg();
    if (body_start == std::streampos(-1))
        fail(path, "Cannot locate Matrix Market array body");
    // The full nrows*ncols product was checked before reading the header.
    int64_t expected = (header.nnz - header.nrows) / 2;
    if (header.symmetry != fast_matrix_market::skew_symmetric)
        expected += header.nrows;
    int64_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (std::all_of(line.begin(), line.end(),
            [](unsigned char c) { return std::isspace(c); }))
            continue;
        if (count == expected)
            fail(path, "Too many values in Matrix Market array");
        ++count;
    }
    if (file.bad())
        fail(path, "Failed to read Matrix Market file");
    if (count != expected)
        fail(path, "Truncated Matrix Market array: expected " + std::to_string(expected) +
                   " values, found " + std::to_string(count));
    file.clear();
    file.seekg(body_start);
    if (!file)
        fail(path, "Cannot rewind Matrix Market array body");
}

} // namespace matrix_io_detail

/// Load .mtx Matrix Market array/coordinate data, .bin native binary data, or
/// whitespace-delimited text for any other extension (including none).
/// Extension matching ignores case. Binary files use the format documented by
/// RandLAPACK::gen::read_bin_matrix.
///
/// Returns the top-left floor(m*sub_ratio) by floor(n*sub_ratio) submatrix.
/// The ratio must be finite and in (0, 1], and output dimensions must be positive.
/// Dense data is column-major. Sparse input remains sparse, with duplicate
/// coordinates contributing their summed value in both representations.
/// Sparse dimensions and expanded entry counts must fit Eigen's StorageIndex.
/// Invalid input and file errors throw RandLAPACK::Error.
template <typename T>
LoadedMatrix<T> load_matrix(const std::string& path, double sub_ratio = 1.0) {
    using namespace matrix_io_detail;
    if (!std::isfinite(sub_ratio) || sub_ratio <= 0.0 || sub_ratio > 1.0)
        fail(path, "Submatrix ratio must be finite and in (0, 1]");

    LoadedMatrix<T> result;
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (ext == ".mtx") {
        std::ifstream file(path);
        if (!file)
            fail(path, "Cannot open Matrix Market file");
        try {
            check_array_header<T>(file, path);
            fast_matrix_market::matrix_market_header header;
            fast_matrix_market::read_header(file, header);
            if (header.object != fast_matrix_market::matrix)
                fail(path, "Expected a Matrix Market matrix");
            if (header.symmetry != fast_matrix_market::general && header.nrows != header.ncols)
                fail(path, "Symmetric Matrix Market matrices must be square");
            result.m = header.nrows;
            result.n = header.ncols;
            const int64_t sub_m = sub_dimension(result.m, sub_ratio, path);
            const int64_t sub_n = sub_dimension(result.n, sub_ratio, path);

            if (header.format == fast_matrix_market::coordinate) {
                using StorageIndex = typename Eigen::SparseMatrix<T>::StorageIndex;
                const int64_t max_index = std::numeric_limits<StorageIndex>::max();
                if (result.m > max_index || result.n > max_index || header.nnz > max_index)
                    fail(path, "Sparse matrix exceeds Eigen storage index limits");
                dense_size<int64_t>(result.n + 1, 1, path);
                if (header.nnz > 0) {
                    dense_size<int64_t>(header.nnz, 1, path);
                    dense_size<T>(header.nnz, 1, path);
                    dense_size<Eigen::Triplet<T>>(header.nnz, 1, path);
                }
                result.is_sparse = true;
                std::vector<int64_t> rows, cols;
                std::vector<T> vals;
                // Read the stored entries first so symmetry expansion can be
                // checked against the storage index limit before it allocates.
                fast_matrix_market::read_options options;
                options.generalize_symmetry = false;
                fast_matrix_market::read_matrix_market_body_triplet(
                    file, header, rows, cols, vals, T(1), options);
                if (header.symmetry != fast_matrix_market::general) {
                    size_t off_diagonal = 0;
                    for (size_t i = 0; i < vals.size(); ++i)
                        off_diagonal += rows[i] != cols[i];
                    if (off_diagonal > static_cast<size_t>(max_index) - vals.size())
                        fail(path, "Expanded sparse matrix exceeds Eigen storage index limits");
                    const int64_t expanded = static_cast<int64_t>(vals.size() + off_diagonal);
                    if (expanded > 0) {
                        dense_size<int64_t>(expanded, 1, path);
                        dense_size<T>(expanded, 1, path);
                        dense_size<Eigen::Triplet<T>>(expanded, 1, path);
                    }
                    fast_matrix_market::generalize_symmetry_triplet(rows, cols, vals, header.symmetry);
                }

                size_t kept = 0;
                for (size_t i = 0; i < vals.size(); ++i) {
                    if (rows[i] < sub_m && cols[i] < sub_n) {
                        rows[kept] = rows[i];
                        cols[kept] = cols[i];
                        vals[kept] = vals[i];
                        ++kept;
                    }
                }
                rows.resize(kept);
                cols.resize(kept);
                vals.resize(kept);
                result.m = sub_m;
                result.n = sub_n;

                std::vector<Eigen::Triplet<T>> triplets;
                triplets.reserve(kept);
                for (size_t i = 0; i < kept; ++i)
                    triplets.emplace_back(static_cast<StorageIndex>(rows[i]),
                                          static_cast<StorageIndex>(cols[i]), vals[i]);
                result.eigen_sparse = std::make_unique<Eigen::SparseMatrix<T>>(result.m, result.n);
                result.eigen_sparse->setFromTriplets(triplets.begin(), triplets.end());
                result.eigen_sparse->makeCompressed();

                // Eigen sums duplicate coordinates. Copy that same compressed
                // representation so CSC conversions and multiplication agree.
                const int64_t nnz = result.eigen_sparse->nonZeros();
                result.csc = std::make_unique<RandBLAS::sparse_data::CSCMatrix<T>>(result.m, result.n);
                if (nnz > 0) {
                    result.csc->reserve(nnz);
                    std::copy_n(result.eigen_sparse->valuePtr(), nnz, result.csc->vals);
                    std::copy_n(result.eigen_sparse->innerIndexPtr(), nnz, result.csc->rowidxs);
                    std::copy_n(result.eigen_sparse->outerIndexPtr(), result.n + 1, result.csc->colptr);
                } else {
                    // CSC multiplication needs every column boundary even when
                    // there are no nonzeros; reserve(0) is not supported.
                    result.csc->colptr = new int64_t[static_cast<size_t>(result.n) + 1]();
                }
            } else {
                const size_t count = dense_size<T>(result.m, result.n, path);
                if (header.symmetry != fast_matrix_market::general)
                    check_symmetric_array_body(file, header, path);
                result.dense_data.resize(count);
                auto handler = fast_matrix_market::dense_adding_parse_handler(
                    result.dense_data.begin(), fast_matrix_market::col_major, result.m, result.n);
                fast_matrix_market::read_matrix_market_body(file, header, handler, T(1));
            }
            if (file.bad())
                fail(path, "Failed to read Matrix Market file");
        } catch (const fast_matrix_market::fmm_error& error) {
            fail(path, error.what());
        }
    } else {
        auto reader = ext == ".bin" ? RandLAPACK::gen::read_bin_matrix<T>
                                    : RandLAPACK::gen::read_txt_matrix<T>;
        reader(blas::Layout::ColMajor, result.m, result.n, nullptr, 0, path.c_str(), true);
        sub_dimension(result.m, sub_ratio, path);
        sub_dimension(result.n, sub_ratio, path);
        result.dense_data.resize(dense_size<T>(result.m, result.n, path));
        reader(blas::Layout::ColMajor, result.m, result.n, result.data(), result.m, path.c_str(), false);
    }

    if (!result.is_sparse && sub_ratio < 1.0) {
        const int64_t sub_m = sub_dimension(result.m, sub_ratio, path);
        const int64_t sub_n = sub_dimension(result.n, sub_ratio, path);
        std::vector<T> submatrix(dense_size<T>(sub_m, sub_n, path));
        for (int64_t col = 0; col < sub_n; ++col)
            std::copy_n(result.data() + col * result.m, sub_m, submatrix.data() + col * sub_m);
        result.dense_data = std::move(submatrix);
        result.m = sub_m;
        result.n = sub_n;
    }
    return result;
}

} // namespace BenchIO

#endif // BENCH_MATRIX_IO_HH
