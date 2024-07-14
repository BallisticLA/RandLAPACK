
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
// #include <omp.h>

// #include <unistd.h>
// #include <math.h>
// #include <time.h>
// #include <stdlib.h>
// #include <stdio.h>

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

#ifndef DOUT
#define DOUT(_d) std::setprecision(std::numeric_limits<double>::max_digits10) << _d
#endif

#ifndef TIMED_LINE
#define TIMED_LINE(_op, _name) { \
        auto _tp0 = std_clock::now(); \
        _op; \
        auto _tp1 = std_clock::now(); \
        double dtime = (double) duration_cast<microseconds>(_tp1 - _tp0).count(); \
        std::cout << _name << DOUT(dtime / 1e6) << std::endl; \
        }
#endif



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

int main() {
    //std::string dn{"/Users/rjmurr/Documents/open-data/kernel-ridge-regression/sensit_vehicle"};
    std::string dn{"/Users/rjmurr/Documents/open-data/kernel-ridge-regression/cod-rna"};
    auto krrd = mmread_krr_data_dir(dn);
    using T = double;
    int64_t m = krrd.X_train.ncols;
    int64_t d = krrd.X_train.nrows;
    std::cout << "cols  : " << m << std::endl; 
    std::cout << "rows  : " << d << std::endl;
    T mu_min = m * 1e-7;
    std::vector<T> mus{mu_min};
    RandLAPACK::linops::SEKLO A_linop(m, krrd.X_train.vals.data(), d, 3.0, mus);
    // int64_t s = mus.size();
    for (int64_t s = 1; s <= 8; s*=2) {
        std::vector<T> H(m*s, 0.0);

        T* Hd = H.data();
        T* hd = krrd.Y_train.vals.data();
        blas::copy(m, hd, 1, Hd, 1);
        if (s > 1) {
            RandBLAS::RNGState state_H(1);
            RandBLAS::DenseDist D(m, s - 1, RandBLAS::DenseDistName::Gaussian);
            RandBLAS::fill_dense(D, Hd + m, state_H);
            T nrm_h = blas::nrm2(m, hd, 1);
            for (int i = 1; i < s; ++i) {
                // T nrm_Hi = blas::nrm2(m, Hd + i*m, 1);
                // T scale = std::pow(2.0*nrm_Hi, -1); 
                // blas::scal(m, scale, Hd + i*m, 1);
                // blas::axpy(m, 1.0, hd, 1, Hd + i*m, 1);
                T nrm_Hi = blas::nrm2(m, Hd + i*m, 1);
                T scale = nrm_h / nrm_Hi;
                blas::scal(m, scale, Hd + i*m, 1);
            }
        }

        std::vector<T> X(m*s, 0.0);
        // solve A_linop X == H
        RandBLAS::RNGState state(0);
        //RandLAPACK::StatefulFrobeniusNorm<T> seminorm{};
        auto seminorm = [](int64_t n, int64_t s, const T* NR){return blas::nrm2(n, NR, 1);};
        int64_t k = 2*1024;
        int64_t rpc_b = 64;
        int64_t eval_block_size = 1024;
        std::cout << "k     : " << k << std::endl;
        std::cout << "s     : " << s << std::endl;
        std::cout << "mu0   : " << mu_min << std::endl;
        std::cout << "rpc_b : " << rpc_b << std::endl << std::endl;
        T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        int64_t max_iters = 25;
        A_linop._eval_block_size = eval_block_size;
        A_linop._eval_work1.resize(A_linop._eval_block_size * m);
        TIMED_LINE(
        RandLAPACK::krill_full_rpchol(
            m, A_linop, H, X, tol, state, seminorm, rpc_b, max_iters, k
        );, "\nKrill : ")
        std::cout << std::endl;
    }
    return 0;
}
