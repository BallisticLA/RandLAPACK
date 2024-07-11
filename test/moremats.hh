#pragma once

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "/Users/rjmurr/Documents/randnla/RandLAPACK/RandBLAS/test/comparison.hh"


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
    RandBLAS::util::symmetrize(Layout::ColMajor, Uplo::Upper, G.data(), m, m);
    return G;
}

}