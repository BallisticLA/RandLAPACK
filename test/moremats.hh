#pragma once

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "../RandLAPACK/RandBLAS/test/comparison.hh"


namespace RandLAPACK_Testing {

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Uplo;
using RandBLAS::RNGState;

template <typename T>
vector<T> polynomial_decay_psd(int64_t m, T cond_num, T exponent, uint32_t seed) {
    RandLAPACK::gen::mat_gen_info<T> mat_info(m, m, RandLAPACK::gen::polynomial);
    mat_info.cond_num = std::sqrt(cond_num);
    mat_info.rank = m;
    mat_info.exponent = std::sqrt(exponent);
    mat_info.frac_spectrum_one = 0.05;
    vector<T> A(m * m, 0.0);
    RNGState data_state(seed);
    RandLAPACK::gen::mat_gen(mat_info, A.data(), data_state);
    vector<T> G(m * m, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::NoTrans, m, m, 1.0,
        A.data(), m, 0.0, G.data(), m
    ); // Note: G is PSD with squared spectrum of A.
    RandBLAS::symmetrize(Layout::ColMajor, Uplo::Upper, m, G.data(), m);
    return G;
}

template <typename T>
vector<T> random_gaussian_mat(int64_t m, int64_t n, uint32_t seed) {
    RandBLAS::DenseDist D(m, n);
    RNGState state(seed);
    vector<T> mat(m*n);
    RandBLAS::fill_dense(D, mat.data(), state);
    return mat;
}

template <typename T, typename RNG>
RNGState<RNG> left_multiply_by_orthmat(int64_t m, int64_t n, std::vector<T> &A, RNGState<RNG> state) {
    using std::vector;
    vector<T> U(m * m, 0.0);
    RandBLAS::DenseDist DU(m, m);
    auto out_state = RandBLAS::fill_dense(DU, U.data(), state);
    vector<T> tau(m, 0.0);
    lapack::geqrf(m, m, U.data(), m, tau.data());
    lapack::ormqr(blas::Side::Left, blas::Op::NoTrans, m, n, m, U.data(), m, tau.data(), A.data(), m);
    return out_state;
}


}