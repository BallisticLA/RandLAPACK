#pragma once

// Public API: ToeplitzLinOp, a matrix-free rectangular Toeplitz operator T = toeplitz(c, r)
// applied through circulant-embedding FFTs (Intel MKL DFTI).
//
// An extras operator because it needs MKL's FFT (mkl_dfti.h); core RandLAPACK needs only
// BLAS++/LAPACK++. It satisfies the LinearOperator concept but has no sketching-operator
// overload, so a sketch-based driver needs it wrapped (VStackLinOp provides one). It keeps c and
// r for materialize_cols, which reads columns of T from them with no FFT.
//
// Circulant embedding (0-based): T is m x n with first column c and first row r, c[0] == r[0],
// and L is the smallest even 2^a 3^b 5^c >= m + n - 1. The embedding holds c[0..m-1] at the
// front, zeros, then r[n-1], ..., r[1] at the end; with E its FFT,
// T x = ifft(E .* fft([x; 0]))[0..m-1]. T^T = toeplitz(r, c) uses the embedding of (r, c) the
// same way. Every transform is real-to-complex on the L/2 + 1 bins of the half spectrum.
//
// Side::Left, ColMajor and trans_B == NoTrans only. Not safe to call concurrently on one
// instance: the work buffer and the batched descriptors are shared mutable state.

#include "rl_blaspp.hh"
#include "rl_exceptions.hh"
#include "rl_thread_caps.hh"

// Match MKL's integer width to BLAS++ before mkl_dfti.h fixes it (as RandBLAS does).
#if defined(BLAS_ILP64) && !defined(MKL_ILP64)
#define MKL_ILP64
#endif
#include <mkl_dfti.h>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace RandLAPACK_extras::linops {

using blas::Layout;
using blas::Op;
using blas::Side;

/// Default cap on MKL's threaded FFT: one transform stops improving past about 16 threads and
/// becomes intermittently slow above that.
inline constexpr int kDefaultFFTThreads = 16;

/// Smallest even 2^a 3^b 5^c >= target (at least 2): MKL's mixed-radix FFTs are about as fast
/// per point on these lengths as on powers of two, which can be up to twice as long.
inline int64_t toeplitz_fft_length(int64_t target) {
    int64_t best = 2;
    while (best < target) best *= 2;
    for (int64_t p5 = 1; p5 < best; p5 *= 5)
        for (int64_t p35 = p5; p35 < best; p35 *= 3) {
            int64_t v = 2 * p35;
            while (v < target) v *= 2;
            best = std::min(best, v);
        }
    return best;
}

namespace detail {
/// FFT thread cap; RANDLAPACK_FFT_THREADS overrides the default (read once), <= 0 disables it.
inline int fft_thread_cap() {
    static const int cap =
        RandLAPACK::thread_cap_from_env("RANDLAPACK_FFT_THREADS").value_or(kDefaultFFTThreads);
    return cap;
}

/// Threads of the elementwise passes around the transforms: the FFT width, never more than
/// OpenMP's default team.
inline int elementwise_threads(int fft_cap) {
#if defined(_OPENMP)
    const int max_t = omp_get_max_threads();
    return (fft_cap > 0) ? std::min(fft_cap, max_t) : max_t;
#else
    (void) fft_cap;
    return 1;
#endif
}

/// Throws unless status is in MKL's DFTI_NO_ERROR class.
inline void dfti_check(MKL_LONG status, const char* what) {
    randlapack_require(status == DFTI_NO_ERROR || DftiErrorClass(status, DFTI_NO_ERROR))
        << "ToeplitzLinOp: " << what << " failed: " << DftiErrorMessage(status);
}
} // namespace detail

template <typename T>
struct ToeplitzLinOp {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ToeplitzLinOp supports float and double");

    using scalar_t = T;

    const int64_t n_rows;   // = m (rows of T)
    const int64_t n_cols;   // = n (cols of T)

    int64_t L;                        // circulant length
    int64_t Lh;                       // L/2 + 1 bins of the half spectrum
    std::complex<T>* Femb  = nullptr; // half spectrum of the embedding of T (length Lh)
    std::complex<T>* FembT = nullptr; // half spectrum of the embedding of T^T (length Lh)
    T* c_gen = nullptr;               // first column of T (length m)
    T* r_gen = nullptr;               // first row of T (length n)

    // Applies run up to kFFTBatch columns per transform call, in place in work: column j holds
    // L reals padded to 2 Lh, then its Lh bins. Single columns use desc_f1/desc_b1; wider
    // batches desc_fb/desc_bb, recommitted only when the batch width changes.
    static constexpr int64_t kFFTBatch = 16;
    /// Smallest L at which the elementwise passes run on OpenMP threads.
    static constexpr int64_t kParallelMinLength = int64_t(1) << 15;
    std::complex<T>* work = nullptr;   // work_cols * Lh, grown on demand
    int64_t work_cols = 0;
    DFTI_DESCRIPTOR_HANDLE desc_f1 = nullptr, desc_b1 = nullptr;
    DFTI_DESCRIPTOR_HANDLE desc_fb = nullptr, desc_bb = nullptr;
    int64_t batch_committed = 0;

    /// @param c  first column of T (length m).
    /// @param r  first row of T (length n).  Requires c[0] == r[0].
    ToeplitzLinOp(const T* c, int64_t m, const T* r, int64_t n)
        : n_rows(m), n_cols(n)
    {
        try {
            init(c, m, r, n);
        } catch (...) {
            release();
            throw;
        }
    }

    ~ToeplitzLinOp() { release(); }

    ToeplitzLinOp(const ToeplitzLinOp&) = delete;
    ToeplitzLinOp& operator=(const ToeplitzLinOp&) = delete;

private:
    static constexpr DFTI_CONFIG_VALUE kPrec = std::is_same_v<T, double> ? DFTI_DOUBLE : DFTI_SINGLE;

    // Caps threads at commit time too (MKL fixes a descriptor's threading when it is
    // committed, so a compute-time guard alone is not guaranteed to bind), then commits.
    static void commit(DFTI_DESCRIPTOR_HANDLE d, const char* name) {
        const int cap = detail::fft_thread_cap();
        if (cap > 0)
            detail::dfti_check(DftiSetValue(d, DFTI_THREAD_LIMIT, (MKL_LONG)cap), name);
        detail::dfti_check(DftiCommitDescriptor(d), name);
    }

    // In-place real transform of `cols` columns laid out as in `work`, forward (real to
    // complex) or backward (complex to real, scaled by 1/L). A descriptor's input and output
    // distances refer to the direction it computes, so each direction has its own.
    DFTI_DESCRIPTOR_HANDLE make_desc(int64_t cols, bool forward) const {
        DFTI_DESCRIPTOR_HANDLE d = nullptr;
        try {
            detail::dfti_check(DftiCreateDescriptor(&d, kPrec, DFTI_REAL, 1, (MKL_LONG)L), "create descriptor");
            detail::dfti_check(DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE), "descriptor placement");
            // The legacy default packs the spectrum in CCS format, not an array of std::complex.
            detail::dfti_check(DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX),
                               "descriptor storage");
            if (cols > 1) {
                detail::dfti_check(DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)cols),
                                   "descriptor transforms");
                const MKL_LONG real_dist = (MKL_LONG)(2 * Lh), cplx_dist = (MKL_LONG)Lh;
                detail::dfti_check(DftiSetValue(d, DFTI_INPUT_DISTANCE, forward ? real_dist : cplx_dist),
                                   "descriptor input distance");
                detail::dfti_check(DftiSetValue(d, DFTI_OUTPUT_DISTANCE, forward ? cplx_dist : real_dist),
                                   "descriptor output distance");
            }
            if (!forward)
                detail::dfti_check(DftiSetValue(d, DFTI_BACKWARD_SCALE, (double)(1.0 / (double)L)),
                                   "descriptor backward scale");
            commit(d, "commit descriptor");
        } catch (...) {
            if (d) DftiFreeDescriptor(&d);
            throw;
        }
        return d;
    }

    void grow_work(int64_t cols) {
        if (cols <= work_cols) return;
        delete[] work;
        work = nullptr;
        work_cols = 0;
        work = new std::complex<T>[cols * Lh];
        work_cols = cols;
    }

    void init(const T* c, int64_t m, const T* r, int64_t n) {
        randlapack_require(m >= 1 && n >= 1) << "ToeplitzLinOp: m,n must be >= 1";
        randlapack_require(c[0] == r[0]) << "ToeplitzLinOp: c[0]=" << c[0] << " must equal r[0]=" << r[0];

        L  = toeplitz_fft_length(m + n - 1);
        Lh = L / 2 + 1;
        desc_f1 = make_desc(1, true);
        desc_b1 = make_desc(1, false);
        grow_work(1);

        Femb  = new std::complex<T>[Lh];
        FembT = new std::complex<T>[Lh];
        build_embedding(c, m, r, n, Femb);
        build_embedding(r, n, c, m, FembT);
        c_gen = new T[m];
        r_gen = new T[n];
        std::copy(c, c + m, c_gen);
        std::copy(r, r + n, r_gen);
    }

    void release() {
        for (DFTI_DESCRIPTOR_HANDLE* d : {&desc_f1, &desc_b1, &desc_fb, &desc_bb})
            if (*d) DftiFreeDescriptor(d);
        delete[] Femb; delete[] FembT; delete[] work; delete[] c_gen; delete[] r_gen;
        desc_f1 = desc_b1 = desc_fb = desc_bb = nullptr;
        Femb = FembT = work = nullptr;
        c_gen = r_gen = nullptr;
        work_cols = 0;
        batch_committed = 0;
    }

    // FFT width: the configured cap, narrowed to an active SolveWidthScope's width.
    static int active_fft_cap() {
        int cap = detail::fft_thread_cap();
        const int ctx = RandLAPACK::solve_context_width();
        if (ctx > 0 && (cap <= 0 || ctx < cap)) cap = ctx;
        return cap;
    }

    // out = half spectrum of the embedding of toeplitz(col, row), transformed in place.
    void build_embedding(const T* col, int64_t mc, const T* row, int64_t nr, std::complex<T>* out) {
        T* e = reinterpret_cast<T*>(out);
        std::fill(e, e + 2 * Lh, (T)0);
        std::copy(col, col + mc, e);
        for (int64_t k = 1; k < nr; ++k) e[L - k] = row[k];
        RandLAPACK::ThreadCapGuard fftg(detail::fft_thread_cap());
        detail::dfti_check(DftiComputeForward(desc_f1, out), "embedding FFT");
    }

public:

    // Convenience form (Side::Left implied), matching the LinearOperator concept.
    void operator()(Layout layout, Op trans_A, Op trans_B,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc)
    {
        (*this)(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    // C := alpha * op(T) * B + beta * C.  op(T)=T (NoTrans) or T^T (Trans); trans_B must be NoTrans.
    void operator()(Side side, Layout layout, Op trans_A, Op trans_B,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Left)       << "ToeplitzLinOp supports Side::Left only";
        randlapack_require(layout == Layout::ColMajor) << "ToeplitzLinOp supports ColMajor only";
        randlapack_require(trans_B == Op::NoTrans)   << "ToeplitzLinOp supports trans_B == NoTrans only";

        const int64_t nrhs = n;
        const bool notrans = (trans_A == Op::NoTrans);
        const int64_t out_rows = notrans ? n_rows : n_cols;  // rows of C
        const int64_t in_rows  = notrans ? n_cols : n_rows;  // rows of B
        const std::complex<T>* spec = notrans ? Femb : FembT;

        randlapack_require(m == out_rows) << "ToeplitzLinOp: m=" << m << " must equal " << out_rows;
        randlapack_require(k == in_rows)  << "ToeplitzLinOp: k=" << k << " must equal " << in_rows;
        randlapack_require(n >= 0) << "ToeplitzLinOp: n=" << n << " must be >= 0";
        randlapack_require(ldb >= in_rows && ldc >= out_rows)
            << "ToeplitzLinOp: ldb=" << ldb << " must be >= " << in_rows
            << " and ldc=" << ldc << " >= " << out_rows;

        const int fft_cap = active_fft_cap();
        if (nrhs == 0) return;

        // The elementwise passes are memory-bound and as large as the transforms' data, so
        // they run at the FFT width, not on one thread.
        const int nt = detail::elementwise_threads(fft_cap);
        const bool par = (L >= kParallelMinLength) && nt > 1;
        (void) par;   // read only by the OpenMP pragmas

        grow_work(std::min(nrhs, kFFTBatch));
        for (int64_t j0 = 0; j0 < nrhs; j0 += kFFTBatch) {
            const int64_t bs = std::min<int64_t>(kFFTBatch, nrhs - j0);
            DFTI_DESCRIPTOR_HANDLE df = desc_f1, db = desc_b1;
            if (bs > 1) {
                if (bs != batch_committed) {
                    batch_committed = 0;   // stays 0 if a recommit below throws
                    if (desc_fb) DftiFreeDescriptor(&desc_fb);
                    if (desc_bb) DftiFreeDescriptor(&desc_bb);
                    desc_fb = desc_bb = nullptr;
                    desc_fb = make_desc(bs, true);
                    desc_bb = make_desc(bs, false);
                    batch_committed = bs;
                }
                df = desc_fb;
                db = desc_bb;
            }
            for (int64_t j = 0; j < bs; ++j) {
                const T* bj = B + (j0 + j) * ldb;
                T* wr = reinterpret_cast<T*>(work + j * Lh);
                #pragma omp parallel for num_threads(nt) schedule(static) if(par)
                for (int64_t i = 0; i < L; ++i) wr[i] = (i < in_rows) ? bj[i] : (T)0;
            }
            {
                RandLAPACK::ThreadCapGuard fftg(fft_cap);
                detail::dfti_check(DftiComputeForward(df, work), "forward FFT");
                for (int64_t j = 0; j < bs; ++j) {
                    std::complex<T>* wc = work + j * Lh;
                    #pragma omp parallel for num_threads(nt) schedule(static) if(par)
                    for (int64_t i = 0; i < Lh; ++i) wc[i] *= spec[i];
                }
                detail::dfti_check(DftiComputeBackward(db, work), "backward FFT");   // scaled by 1/L
            }
            // beta == 0 must not read C (BLAS semantics; callers pass uninitialized scratch).
            for (int64_t j = 0; j < bs; ++j) {
                T* cj = C + (j0 + j) * ldc;
                const T* wr = reinterpret_cast<const T*>(work + j * Lh);
                if (beta == (T)0) {
                    #pragma omp parallel for num_threads(nt) schedule(static) if(par)
                    for (int64_t i = 0; i < out_rows; ++i) cj[i] = alpha * wr[i];
                } else {
                    #pragma omp parallel for num_threads(nt) schedule(static) if(par)
                    for (int64_t i = 0; i < out_rows; ++i) cj[i] = alpha * wr[i] + beta * cj[i];
                }
            }
        }
    }

    /// Columns [col_start, col_start + col_count) of T into the column-major buf (ldb >= m),
    /// read from c and r with no FFT (see HasMaterializeCols): T(i, j) = c[i - j] for i >= j,
    /// r[j - i] for j > i.
    void materialize_cols(int64_t col_start, int64_t col_count, T* buf, int64_t ldb) const {
        randlapack_require(col_start >= 0 && col_start <= n_cols
                           && col_count >= 0 && col_count <= n_cols - col_start)
            << "ToeplitzLinOp::materialize_cols: col_start=" << col_start << ", col_count="
            << col_count << " must select columns within [0, " << n_cols << ")";
        randlapack_require(ldb >= n_rows) << "ToeplitzLinOp::materialize_cols: ldb=" << ldb
                                          << " must be >= n_rows=" << n_rows;
        if (col_count == 0) return;   // num_threads below must be positive
        // One thread per column at most; a single column copies on the calling thread.
        const int nt = (int)std::min<int64_t>(detail::elementwise_threads(active_fft_cap()), col_count);
        const bool par = (n_rows * col_count >= kParallelMinLength) && nt > 1;
        (void) par;   // read only by the OpenMP pragma
        #pragma omp parallel for num_threads(nt) schedule(static) if(par)
        for (int64_t jj = 0; jj < col_count; ++jj) {
            const int64_t j = col_start + jj;
            const int64_t above = std::min(j, n_rows);   // rows i < j hold r[j - i]
            T* col = buf + jj * ldb;
            for (int64_t i = 0; i < above; ++i) col[i] = r_gen[j - i];
            std::copy(c_gen, c_gen + (n_rows - above), col + above);
        }
    }
};

} // namespace RandLAPACK_extras::linops
