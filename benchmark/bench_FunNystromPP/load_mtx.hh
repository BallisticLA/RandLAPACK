#pragma once
// Minimal Matrix Market (.mtx) reader.
// Supports: coordinate real/integer, symmetric or general.
// Returns a CSRMatrix<double, int64_t> with both triangles stored (required by SparseSymLinOp).

#include <RandBLAS.hh>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cctype>

namespace FunNystromPP_bench {

inline RandBLAS::sparse_data::CSRMatrix<double, int64_t>
load_mtx(const std::string& filename, int64_t& out_n) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("load_mtx: cannot open " + filename);

    // Parse banner line.
    std::string banner;
    std::getline(f, banner);
    std::string blo = banner;
    for (char& c : blo) c = (char)std::tolower((unsigned char)c);
    if (blo.find("%%matrixmarket") == std::string::npos)
        throw std::runtime_error("load_mtx: missing %%MatrixMarket banner");
    if (blo.find("coordinate") == std::string::npos)
        throw std::runtime_error("load_mtx: only coordinate format supported");
    if (blo.find("real")    == std::string::npos &&
        blo.find("integer") == std::string::npos &&
        blo.find("double")  == std::string::npos)
        throw std::runtime_error("load_mtx: only real/integer/double field type supported");
    bool is_symm   = blo.find("symmetric") != std::string::npos;
    bool is_skew   = blo.find("skew")      != std::string::npos;

    // Skip comment lines; find the size line.
    std::string line;
    while (std::getline(f, line)) {
        std::string lo = line;
        for (char& c : lo) c = (char)std::tolower((unsigned char)c);
        if (lo.find("%%matrixmarket") != std::string::npos) continue;
        if (!line.empty() && line[0] != '%') break;
    }

    int64_t nrows, ncols, nnz_file;
    {
        std::istringstream ss(line);
        ss >> nrows >> ncols >> nnz_file;
    }
    if (nrows != ncols)
        throw std::runtime_error("load_mtx: matrix must be square (got "
            + std::to_string(nrows) + "x" + std::to_string(ncols) + ")");
    out_n = nrows;

    // Read COO entries; expand symmetric/skew-symmetric off-diagonal entries.
    std::vector<int64_t> ri, ci;
    std::vector<double>  vi;
    ri.reserve(is_symm ? 2*nnz_file : nnz_file);
    ci.reserve(is_symm ? 2*nnz_file : nnz_file);
    vi.reserve(is_symm ? 2*nnz_file : nnz_file);

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream ls(line);
        int64_t r, c; double v;
        if (!(ls >> r >> c >> v)) continue;
        --r; --c;  // .mtx is 1-indexed; convert to 0-indexed
        ri.push_back(r); ci.push_back(c); vi.push_back(v);
        if ((is_symm || is_skew) && r != c) {
            ri.push_back(c); ci.push_back(r);
            vi.push_back(is_skew ? -v : v);
        }
    }
    int64_t total_nnz = (int64_t)vi.size();

    // Sort COO by (row, col) to build CSR.
    std::vector<int64_t> ord(total_nnz);
    std::iota(ord.begin(), ord.end(), (int64_t)0);
    std::sort(ord.begin(), ord.end(), [&](int64_t a, int64_t b) {
        return ri[a] != ri[b] ? ri[a] < ri[b] : ci[a] < ci[b];
    });

    // Build owning CSRMatrix.
    using CSR = RandBLAS::sparse_data::CSRMatrix<double, int64_t>;
    CSR csr(nrows, ncols);
    csr.reserve(total_nnz);  // allocates vals[total_nnz], colidxs[total_nnz], rowptr[nrows+1]; sets nnz

    // First pass: count entries per row into rowptr[1..nrows].
    std::fill(csr.rowptr, csr.rowptr + nrows + 1, (int64_t)0);
    for (int64_t k = 0; k < total_nnz; ++k)
        ++csr.rowptr[ri[ord[k]] + 1];
    // Prefix sum → rowptr[0..nrows] = start of each row.
    for (int64_t i = 0; i < nrows; ++i)
        csr.rowptr[i+1] += csr.rowptr[i];

    // Second pass: fill colidxs and vals; temporarily advance rowptr[row] to track position.
    std::vector<int64_t> pos(csr.rowptr, csr.rowptr + nrows);  // copy of row starts
    for (int64_t k = 0; k < total_nnz; ++k) {
        int64_t row = ri[ord[k]];
        int64_t p   = pos[row]++;
        csr.colidxs[p] = ci[ord[k]];
        csr.vals[p]    = vi[ord[k]];
    }

    return csr;
}

} // namespace FunNystromPP_bench
