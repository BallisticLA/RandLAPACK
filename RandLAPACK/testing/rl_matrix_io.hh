#pragma once

// Matrix file I/O utilities.
//
// read_txt_matrix: reads whitespace-delimited text files (one row per line).
//   Two-phase API: first call with query=true to get dimensions, then call
//   with query=false and a pre-allocated buffer to read data.
//   Uses fast parallel parsing (fread + strtod + OpenMP).
//
// read_bin_matrix: reads binary files written by gen_mat_alg971_paper.m
//   (WriteBinary=true). Format: int64_t m, int64_t n (16-byte header), then
//   m*n doubles in row-major order. Two-phase API matches read_txt_matrix.
//
// For Matrix Market (.mtx) support, see fast_matrix_market (external dependency,
// available in benchmark/ and extras/ projects).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace RandLAPACK::gen {

/// Read a dense matrix from a whitespace-delimited text file.
///
/// Two-phase usage:
///   Phase 1 (dimension query): set query=true, A=nullptr. Returns m and n.
///   Phase 2 (data read):       set query=false, A=pre-allocated m*n buffer.
///
/// The file is row-major text: m rows, each with n space-separated values.
/// Output is stored column-major: A[row + m * col].
template <typename T>
void read_txt_matrix(
    int64_t &m,
    int64_t &n,
    T* A,
    const char* filename,
    bool query
) {
    if (query) {
        // Phase 1: count rows and columns
        m = 0; n = 0;
        std::string line, entry;
        std::ifstream file(filename);
        if (!file.is_open())
            throw std::runtime_error(std::string("Cannot open file: ") + filename);

        // Count columns from first row
        std::getline(file, line);
        std::istringstream ls(line);
        while (ls >> entry) ++n;

        // Count rows (already read one)
        m = 1;
        while (std::getline(file, line)) ++m;
    } else {
        // Phase 2: fast parallel read
        // 1. fread entire file into memory
        // 2. Scan for newline boundaries
        // 3. Parse each row via strtod with OpenMP
        FILE* fp = std::fopen(filename, "r");
        if (!fp)
            throw std::runtime_error(std::string("Cannot open file: ") + filename);
        std::fseek(fp, 0, SEEK_END);
        long file_size = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);

        std::vector<char> buf(file_size + 1);
        size_t bytes_read = std::fread(buf.data(), 1, file_size, fp);
        (void) bytes_read;
        buf[file_size] = '\0';
        std::fclose(fp);

        std::vector<char*> row_starts(m);
        row_starts[0] = buf.data();
        int64_t row_idx = 1;
        for (long pos = 0; pos < file_size && row_idx < m; ++pos) {
            if (buf[pos] == '\n')
                row_starts[row_idx++] = buf.data() + pos + 1;
        }

        #pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < m; ++j) {
            char* ptr = row_starts[j];
            char* end;
            for (int64_t i = 0; i < n; ++i) {
                A[m * i + j] = (T) std::strtod(ptr, &end);
                ptr = end;
            }
        }
    }
}

/// Read a dense matrix from a binary file written by gen_mat_alg971_paper.m
/// (WriteBinary=true option).
///
/// File format:
///   int64_t m, int64_t n   (16-byte header)
///   m*n doubles             (row-major: row 0 col 0..n-1, row 1 col 0..n-1, ...)
///
/// Output is stored column-major: A[row + m * col], matching LAPACK convention.
/// Data is fread in one call then scattered with OpenMP — faster than strtod.
///
/// Two-phase API matches read_txt_matrix:
///   Phase 1 (query=true):  fills m, n; A may be nullptr.
///   Phase 2 (query=false): fills A[0..m*n-1]; m and n must already be set.
template <typename T>
void read_bin_matrix(
    int64_t &m,
    int64_t &n,
    T* A,
    const char* filename,
    bool query
) {
    FILE* fp = std::fopen(filename, "rb");
    if (!fp)
        throw std::runtime_error(std::string("Cannot open file: ") + filename);

    int64_t dims[2] = {0, 0};
    if (std::fread(dims, sizeof(int64_t), 2, fp) != 2) {
        std::fclose(fp);
        throw std::runtime_error(std::string("Failed to read header: ") + filename);
    }
    m = dims[0];
    n = dims[1];

    if (!query) {
        size_t total = (size_t)m * (size_t)n;
        std::vector<double> buf(total);
        if (std::fread(buf.data(), sizeof(double), total, fp) != total) {
            std::fclose(fp);
            throw std::runtime_error(std::string("Truncated binary file: ") + filename);
        }
        // Scatter row-major doubles to column-major T array.
        #pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < m; ++j)
            for (int64_t i = 0; i < n; ++i)
                A[m * i + j] = (T) buf[(size_t)j * n + i];
    }

    std::fclose(fp);
}

} // end namespace RandLAPACK::gen
