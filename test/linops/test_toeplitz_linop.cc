// Tests for RandLAPACK_extras::linops::ToeplitzLinOp: forward and adjoint FFT
// applies checked against an explicitly formed dense toeplitz(c, r) matrix.
//
// Covers: NoTrans and Trans, a single right-hand-side column (the serial CG-loop
// path) and kFFTBatch+1 columns (crosses the batched-descriptor recommit
// boundary), and alpha/beta combinations including beta == 0 (must not read the
// caller's C buffer) and beta != 0 (accumulate into C).

// RandBLAS.hh must precede the extras header: it defines RandBLAS_HAS_MKL (via
// RandBLAS/config.h), which rl_blas2_threads.hh (included transitively by
// ext_toeplitz_linop.hh) gates its MKL thread-capping code behind.
#include <RandBLAS.hh>
#include "extras/linops/ext_toeplitz_linop.hh"

#include <blas.hh>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using std::vector;
using blas::Layout;
using blas::Op;
using RandBLAS::RNGState;

namespace ext = RandLAPACK_extras::linops;

class TestToeplitzLinOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    // Dense toeplitz(c, r): T(i,j) = c[i-j] for i >= j, r[j-i] for j > i.
    // ColMajor, leading dimension m.
    template <typename T>
    static void build_dense(const T* c, int64_t m, const T* r, int64_t n, vector<T>& Tdense) {
        Tdense.assign((size_t)m * (size_t)n, (T)0);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                Tdense[i + j * m] = (i >= j) ? c[i - j] : r[j - i];
    }

    template <typename T>
    static double rel_fro_err(const T* a, const T* b, int64_t len) {
        double err = 0.0, ref = 0.0;
        for (int64_t i = 0; i < len; ++i) {
            double d = (double)a[i] - (double)b[i];
            err += d * d;
            ref += (double)b[i] * (double)b[i];
        }
        return (ref > 0) ? std::sqrt(err / ref) : std::sqrt(err);
    }

    struct Case {
        int64_t m, n;
        vector<double> c, r, Tdense;
    };

    // c[0] == r[0] is required by the operator; the rest is arbitrary Gaussian data.
    static Case make_case(int64_t m, int64_t n, uint32_t seed) {
        Case cs;
        cs.m = m; cs.n = n;
        cs.c.resize(m);
        cs.r.resize(n);
        RNGState<> state(seed);
        state = RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), cs.c.data(), state);
        RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), cs.r.data(), state);
        cs.r[0] = cs.c[0];
        build_dense(cs.c.data(), m, cs.r.data(), n, cs.Tdense);
        return cs;
    }

    // Applies the operator and blas::gemm on the dense matrix under identical
    // alpha/beta/trans, and checks the results agree to `tol` relative Frobenius
    // error. When poison_c is set, C starts full of NaN and beta must be 0 (the
    // reference is computed against a clean zero buffer, never the poisoned one,
    // so the comparison does not rely on blas::gemm's own beta==0 contract).
    static void run_case(ext::ToeplitzLinOp<double>& Top, const Case& cs,
                          Op trans_self, int64_t nrhs, double alpha, double beta,
                          bool poison_c, uint32_t seed, double tol = 1e-13) {
        int64_t in_rows  = (trans_self == Op::NoTrans) ? cs.n : cs.m;
        int64_t out_rows = (trans_self == Op::NoTrans) ? cs.m : cs.n;

        vector<double> B(in_rows * nrhs);
        RNGState<> bstate(seed);
        RandBLAS::fill_dense(RandBLAS::DenseDist(in_rows, nrhs), B.data(), bstate);

        vector<double> C(out_rows * nrhs);
        vector<double> C_ref(out_rows * nrhs, 0.0);

        if (poison_c) {
            ASSERT_EQ(beta, 0.0) << "poison_c is only meaningful with beta == 0";
            std::fill(C.begin(), C.end(), std::numeric_limits<double>::quiet_NaN());
            // C_ref stays a clean zero buffer: reference is alpha*op(T)*B only.
            blas::gemm(Layout::ColMajor, trans_self, Op::NoTrans, out_rows, nrhs, in_rows,
                       alpha, cs.Tdense.data(), cs.m, B.data(), in_rows, 0.0, C_ref.data(), out_rows);
        } else {
            RNGState<> cstate(seed + 1);
            RandBLAS::fill_dense(RandBLAS::DenseDist(out_rows, nrhs), C.data(), cstate);
            C_ref = C;
            blas::gemm(Layout::ColMajor, trans_self, Op::NoTrans, out_rows, nrhs, in_rows,
                       alpha, cs.Tdense.data(), cs.m, B.data(), in_rows, beta, C_ref.data(), out_rows);
        }

        Top(Layout::ColMajor, trans_self, Op::NoTrans,
            out_rows, nrhs, in_rows, alpha, B.data(), in_rows, beta, C.data(), out_rows);

        for (double v : C) ASSERT_FALSE(std::isnan(v)) << "C retained an unwritten (NaN) entry";
        ASSERT_LE(rel_fro_err(C.data(), C_ref.data(), out_rows * nrhs), tol);
    }
};

TEST_F(TestToeplitzLinOp, notrans_single_column_matches_dense) {
    Case cs = make_case(43, 37, 11);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    ASSERT_EQ(Top.n_rows, cs.m);
    ASSERT_EQ(Top.n_cols, cs.n);
    run_case(Top, cs, Op::NoTrans, /*nrhs=*/1, /*alpha=*/1.0, /*beta=*/0.0,
             /*poison_c=*/false, /*seed=*/100);
}

TEST_F(TestToeplitzLinOp, notrans_multicolumn_crosses_batch_boundary) {
    Case cs = make_case(43, 37, 12);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    const int64_t nrhs = ext::ToeplitzLinOp<double>::kFFTBatch + 1;
    run_case(Top, cs, Op::NoTrans, nrhs, /*alpha=*/1.0, /*beta=*/0.0,
             /*poison_c=*/false, /*seed=*/101);
}

TEST_F(TestToeplitzLinOp, trans_single_column_matches_dense) {
    Case cs = make_case(41, 39, 13);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    run_case(Top, cs, Op::Trans, /*nrhs=*/1, /*alpha=*/1.0, /*beta=*/0.0,
             /*poison_c=*/false, /*seed=*/102);
}

TEST_F(TestToeplitzLinOp, trans_multicolumn_crosses_batch_boundary) {
    Case cs = make_case(41, 39, 14);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    const int64_t nrhs = ext::ToeplitzLinOp<double>::kFFTBatch + 1;
    run_case(Top, cs, Op::Trans, nrhs, /*alpha=*/1.0, /*beta=*/0.0,
             /*poison_c=*/false, /*seed=*/103);
}

TEST_F(TestToeplitzLinOp, notrans_alpha_beta_accumulate_single_column) {
    Case cs = make_case(43, 37, 15);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    run_case(Top, cs, Op::NoTrans, /*nrhs=*/1, /*alpha=*/2.5, /*beta=*/1.5,
             /*poison_c=*/false, /*seed=*/104);
}

TEST_F(TestToeplitzLinOp, notrans_alpha_beta_accumulate_multicolumn) {
    Case cs = make_case(43, 37, 16);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    const int64_t nrhs = ext::ToeplitzLinOp<double>::kFFTBatch + 1;
    run_case(Top, cs, Op::NoTrans, nrhs, /*alpha=*/2.5, /*beta=*/-0.75,
             /*poison_c=*/false, /*seed=*/105);
}

TEST_F(TestToeplitzLinOp, trans_alpha_beta_accumulate_multicolumn) {
    Case cs = make_case(41, 39, 17);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    const int64_t nrhs = ext::ToeplitzLinOp<double>::kFFTBatch + 1;
    run_case(Top, cs, Op::Trans, nrhs, /*alpha=*/-1.25, /*beta=*/0.5,
             /*poison_c=*/false, /*seed=*/106);
}

TEST_F(TestToeplitzLinOp, beta_zero_does_not_read_garbage_single_column) {
    Case cs = make_case(43, 37, 18);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    run_case(Top, cs, Op::NoTrans, /*nrhs=*/1, /*alpha=*/1.0, /*beta=*/0.0,
             /*poison_c=*/true, /*seed=*/107);
}

TEST_F(TestToeplitzLinOp, beta_zero_does_not_read_garbage_multicolumn) {
    Case cs = make_case(43, 37, 19);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    const int64_t nrhs = ext::ToeplitzLinOp<double>::kFFTBatch + 1;
    run_case(Top, cs, Op::NoTrans, nrhs, /*alpha=*/1.0, /*beta=*/0.0,
             /*poison_c=*/true, /*seed=*/108);
}

TEST_F(TestToeplitzLinOp, beta_zero_does_not_read_garbage_trans) {
    Case cs = make_case(41, 39, 20);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    const int64_t nrhs = ext::ToeplitzLinOp<double>::kFFTBatch + 1;
    run_case(Top, cs, Op::Trans, nrhs, /*alpha=*/1.0, /*beta=*/0.0,
             /*poison_c=*/true, /*seed=*/109);
}
