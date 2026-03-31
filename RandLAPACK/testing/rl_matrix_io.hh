#pragma once

// Matrix file I/O utilities.
//
// read_txt_matrix: reads whitespace-delimited text files (one row per line).
//   Two-phase API: first call with query=true to get dimensions, then call
//   with query=false and a pre-allocated buffer to read data.
//   Uses fast parallel parsing (fread + strtod + OpenMP).
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

} // end namespace RandLAPACK::gen
