#pragma once

// Public API: ToeplitzLinOp — matrix-free rectangular Toeplitz operator T = toeplitz(c, r)
//             applied via circulant-embedding FFTs (Intel MKL DFTI).
//
// This is an "extras" linear operator because it depends on MKL's FFT (mkl_dfti.h);
// core RandLAPACK is BLAS++/LAPACK++-only. It satisfies the RandLAPACK LinearOperator
// concept (n_rows, n_cols, operator()), so it drops into VStackOp / CholQR / CQRRT /
// Blendenpik / lsqr exactly like a DenseLinOp, but never materializes T.
//
// Circulant embedding (mirrors the reference MATLAB toeplitz FFT operator):
//   T is m x n with first column c (length m) and first row r (length n), c(1)==r(1).
//   L = 2^ceil(log2(m+n-1)).
//   Forward embedding   emb  = [c; zeros(L-(m+n-1)); flipud(r(2:n))],   Femb  = fft(emb).
//   T*x  = real( ifft( Femb  .* fft([x; 0...]) ) )(1:m).
//   Transpose T' = toeplitz(r, c): embT = [r; zeros(L-(m+n-1)); flipud(c(2:m))], FembT=fft(embT).
//   T'*y = real( ifft( FembT .* fft([y; 0...]) ) )(1:n).
//
// Only Side::Left, ColMajor, trans_B == NoTrans are used (VStackOp's sketch overload
// builds W = A_hat * I_block via NoTrans and sketches W itself, so callers only ever
// ask this operator for T*B (NoTrans) or T'*B (Trans)).

#include "rl_blaspp.hh"
#include "rl_exceptions.hh"

#include <mkl_dfti.h>
#include <complex>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace RandLAPACK_extras::linops {

using blas::Layout;
using blas::Op;
using blas::Side;

template <typename T>
struct ToeplitzLinOp {

    using scalar_t = T;     // required by the RandLAPACK LinearOperator concept

    const int64_t n_rows;   // = m (rows of T)
    const int64_t n_cols;   // = n (cols of T)
    int64_t block_size;     // RHS blocking hint (FFTs are applied column-by-column here)

    int64_t L;                       // circulant length, power of two >= m+n-1
    std::complex<T>* Femb  = nullptr; // fft of forward embedding (length L)
    std::complex<T>* FembT = nullptr; // fft of transpose embedding (length L)
    std::complex<T>* work  = nullptr; // scratch (length L), reused per column
    DFTI_DESCRIPTOR_HANDLE desc = nullptr;

    /// @param c  first column of T (length m).
    /// @param r  first row of T (length n).  Requires c[0] == r[0].
    ToeplitzLinOp(const T* c, int64_t m, const T* r, int64_t n, int64_t block_size_ = 8)
        : n_rows(m), n_cols(n), block_size(block_size_)
    {
        randlapack_require(m >= 1 && n >= 1) << "ToeplitzLinOp: m,n must be >= 1";
        randlapack_require(std::abs((double)c[0] - (double)r[0])
                           < 1e2 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs((double)c[0])))
            << "ToeplitzLinOp: c[0] must equal r[0]";

        int64_t span = m + n - 1;
        L = 1; while (L < span) L <<= 1;   // next power of two >= m+n-1

        Femb  = new std::complex<T>[L];
        FembT = new std::complex<T>[L];
        work  = new std::complex<T>[L];

        // Length-L complex FFT descriptor; DftiComputeBackward is unnormalized, so scale by 1/L.
        constexpr DFTI_CONFIG_VALUE prec = std::is_same_v<T, double> ? DFTI_DOUBLE : DFTI_SINGLE;
        DftiCreateDescriptor(&desc, prec, DFTI_COMPLEX, 1, (MKL_LONG)L);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(desc, DFTI_BACKWARD_SCALE, (double)(1.0 / (double)L));
        DftiCommitDescriptor(desc);

        // Forward embedding: [c(0..m-1); zeros; flipud(r(1..n-1))]  -> Femb = fft(emb).
        build_embedding(c, m, r, n, Femb);
        // Transpose embedding (T' = toeplitz(r,c)): [r(0..n-1); zeros; flipud(c(1..m-1))].
        build_embedding(r, n, c, m, FembT);
    }

    ~ToeplitzLinOp() {
        if (desc)  DftiFreeDescriptor(&desc);
        delete[] Femb; delete[] FembT; delete[] work;
    }

    ToeplitzLinOp(const ToeplitzLinOp&) = delete;
    ToeplitzLinOp& operator=(const ToeplitzLinOp&) = delete;

    // Convenience form (Side::Left implied), matching the LinearOperator concept.
    void operator()(Layout layout, Op trans_self, Op trans_B,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc)
    {
        (*this)(Side::Left, layout, trans_self, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    // C := alpha * op(T) * B + beta * C.  op(T)=T (NoTrans) or T^T (Trans); trans_B must be NoTrans.
    void operator()(Side side, Layout layout, Op trans_self, Op trans_B,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Left)       << "ToeplitzLinOp supports Side::Left only";
        randlapack_require(layout == Layout::ColMajor) << "ToeplitzLinOp supports ColMajor only";
        randlapack_require(trans_B == Op::NoTrans)   << "ToeplitzLinOp supports trans_B == NoTrans only";

        const int64_t nrhs = n;                 // number of right-hand-side columns
        const bool notrans = (trans_self == Op::NoTrans);
        const int64_t out_rows = notrans ? n_rows : n_cols;  // rows of C
        const int64_t in_rows  = notrans ? n_cols : n_rows;  // rows of B
        const std::complex<T>* Fuse = notrans ? Femb : FembT;

        randlapack_require(m == out_rows) << "ToeplitzLinOp: output row dim mismatch";
        randlapack_require(k == in_rows)  << "ToeplitzLinOp: inner dim mismatch";

        for (int64_t j = 0; j < nrhs; ++j) {
            const T* bj = B + j * ldb;
            T*       cj = C + j * ldc;
            // work = [ bj(0..in_rows-1) ; 0 ... ]  (complex, imag=0)
            for (int64_t i = 0; i < in_rows; ++i) work[i] = std::complex<T>(bj[i], (T)0);
            for (int64_t i = in_rows; i < L; ++i) work[i] = std::complex<T>((T)0, (T)0);
            DftiComputeForward(desc, work);
            for (int64_t i = 0; i < L; ++i) work[i] *= Fuse[i];
            DftiComputeBackward(desc, work);   // scaled by 1/L
            for (int64_t i = 0; i < out_rows; ++i)
                cj[i] = alpha * work[i].real() + beta * cj[i];
        }
    }

private:
    // Femb_out = fft( [col(0..mc-1); zeros(L-(mc+nr-1)); flipud(row(1..nr-1))] ).
    void build_embedding(const T* col, int64_t mc, const T* row, int64_t nr, std::complex<T>* Femb_out)
    {
        for (int64_t i = 0; i < L; ++i) Femb_out[i] = std::complex<T>((T)0, (T)0);
        for (int64_t i = 0; i < mc; ++i) Femb_out[i] = std::complex<T>(col[i], (T)0);
        // flipud(row(2:end)) placed at the END: positions L-(nr-1) .. L-1 hold row(nr-1) .. row(1).
        for (int64_t k = 1; k < nr; ++k)
            Femb_out[L - k] = std::complex<T>(row[k], (T)0);
        DftiComputeForward(desc, Femb_out);
    }
};

} // namespace RandLAPACK_extras::linops
