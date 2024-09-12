#pragma once

#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <chrono>
#include <unordered_map>
#include <iomanip> 
#include <limits> 
#include <numbers>

#include <iostream>
#include <fstream>

#include <fast_matrix_market/fast_matrix_market.hpp>

using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;

using RandBLAS::RNGState;
using RandLAPACK::rp_cholesky;
using lapack::gesdd;
using lapack::Job;
using std::vector;

double sec_elapsed(timepoint_t tp0, timepoint_t tp1) {
    return ((double) duration_cast<microseconds>(tp1 - tp0).count())/1e6;
}

template <typename T>
void transpose_colmajor(
    int64_t m, int64_t n, const T* A, int64_t lda, T* AT, int64_t ldat
) {
    for(int i = 0; i < n; ++i)
        blas::copy(m, &A[i * lda], 1, &AT[i], ldat);
}


struct array_matrix {
    int64_t nrows = 0, ncols = 0;
    std::vector<double> vals;
};

struct KRR_data {
    array_matrix X_train;
    array_matrix Y_train;
    array_matrix X_test;
    array_matrix Y_test;
};

void standardize(KRR_data &krrd) {
    randblas_require(krrd.X_train.nrows == krrd.X_test.nrows);
    using T = double;
    int64_t d = krrd.X_train.nrows;
    std::vector<T> mu(d, 0.0);
    std::vector<T> sigma(d, 0.0);
    RandLAPACK::standardize_dataset(
        d, krrd.X_train.ncols, krrd.X_train.vals.data(), mu.data(), sigma.data(), false
    );
    RandLAPACK::standardize_dataset(
        d, krrd.X_test.ncols, krrd.X_test.vals.data(), mu.data(), sigma.data(), true
    );
    return;
}

array_matrix mmread_file(std::string fn, bool transpose = true) {
    array_matrix mat{};
    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_array(
        file_stream, mat.nrows, mat.ncols, mat.vals, fast_matrix_market::col_major
    );
    if (transpose) {
        array_matrix tmat{};
        tmat.nrows = mat.ncols;
        tmat.ncols = mat.nrows;
        tmat.vals.resize(mat.vals.size(), 0.0);
        transpose_colmajor(
            mat.nrows, mat.ncols, mat.vals.data(), mat.nrows, tmat.vals.data(), tmat.nrows
        );
        return tmat;
    } else {
        return mat;   
    }
}

KRR_data mmread_krr_data_dir(std::string dn) {
    // mmread_file calls below always apply a transpose; might need to skip transposition for some
    // datasets.
    KRR_data data{};
    data.X_train = mmread_file(dn + "/Xtr.mm");
    data.Y_train = mmread_file(dn + "/Ytr.mm");
    data.X_test  = mmread_file(dn + "/Xts.mm");
    data.Y_test  = mmread_file(dn + "/Yts.mm");
    standardize(data);
    return data;
}
