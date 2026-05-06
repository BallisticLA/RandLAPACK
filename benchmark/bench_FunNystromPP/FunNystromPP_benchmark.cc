#if defined(__APPLE__)
int main() {return 0;}
#else
/*
FunNystromPP benchmark -- compares funNystrom++ against plain Hutchinson+LanczosFA
for estimating tr(f(A)).

Two algorithms are timed and their matvec counts recorded for each matrix size:
  1. FunNystromPP(k, s, d): Nystrom rank-k approximation plus Hutchinson correction
     with s samples, each using d Lanczos steps.
  2. Hutchinson+LanczosFA(s, d): plain Hutchinson with s Rademacher samples,
     each using LanczosFA with d steps to apply f(A). No Nystrom component.
     Uses the same s and d as FunNystromPP (equal correction-phase budget).

Both algorithms use the Lanczos variant selected by lfa_type (see below).

Matrix input (mat_or_file):
  1         = psd_alg:     A = Q diag(λ) Q', λᵢ = 100 * i^{-2}   (algebraic decay, κ ≈ n²/100)
  2         = psd_exp:     A = Q diag(λ) Q', λᵢ = exp(-i/100)    (exponential decay)
  3         = rbf_kernel:  K[i,j] = exp(-||x_i-x_j||²/(2σ²)), x_i~N(0,I_d), σ=sqrt(d)
  4         = poly_kernel: K[i,j] = (x_i·x_j + 1)², x_i~N(0,I_d)
  /path.txt = load dense square symmetric matrix from space-separated text file
              (row-major, full matrix stored; upper triangle used by algorithms)
  /path.mtx = load sparse symmetric matrix in Matrix Market format
              (coordinate real symmetric; both triangles expanded internally)

  For types 1/2: k_mat eigenvalues are nonzero; exact tr(f(A)) is always available.
  For types 3/4: k_mat_ratio reinterpreted as data dimension d = n/k_mat_ratio;
                 reference requires compute_ref=1 (syevd, only feasible for small n).

Function types (func_type):
  0 = sqrt(x)             [f_zero = 0; safe for SPSD matrices]
  1 = log(x)              [requires all eigenvalues > 0; for generated types, noise=1 is applied
                           automatically (A = lowrank + I), so all eigenvalues >= 1]
  2 = x*(x + poly_lambda) [degree-2 polynomial; exact in d=2 Lanczos steps; use for tr(K(K+λI))]

Lanczos variant (lfa_type):
  0 = scalar LanczosFA (s independent Krylov sequences; BLAS-1/2 dominant)
  1 = BlockLanczosFA  (single joint block Krylov subspace; BLAS-3 dominant)

Sketch types (sketch_type):
  0 = Gaussian (DenseDist, default)
  1 = SASO (SparseDist; nonzeros per column set by vec_nnz argument)

compute_ref:
  0 = skip reference (report -1 in output)
  1 = run syevd to compute exact tr(f(A))
  For types 1/2 this flag is ignored; the reference is always available from eigenvalues.
  For types 3/4 and file input, syevd is run only when compute_ref=1.

precision:
  double = run in double precision (default)
  float  = run in single precision (sparse path is always double regardless)

Both algorithms use a fixed seed so comparisons are reproducible.
Matvec counts reflect actual A applications inside each algorithm (not estimates).
The matrix is constructed/loaded once per n; that cost is not timed.

Output format: 32 comma-separated columns per row (trailing comma):
  n, k, k_mat, s, d,
  matvec_pp, matvec_hutch,
  time_pp_total (us), time_pp_phase1 (us), time_pp_phase2 (us),
  time_hutch (us),
  nystrom_alloc, nystrom_syrf, nystrom_matvec, nystrom_gram,
  nystrom_potrf, nystrom_trsm, nystrom_svd, nystrom_post_svd,
  nystrom_error_est, nystrom_rest, nystrom_total (all us),
  lfa_matvec, lfa_run_lanczos, lfa_apply_f, lfa_rest, lfa_total (all us),
  est_fun_pp, est_hutch,
  true_tr (-1 if unavailable), err_fun_pp, err_hutch

  lfa_type is encoded in the output filename: _scalar_ or _block_.

Usage:
  FunNystromPP_benchmark <dir_path> <num_runs> <k> <k_mat> <s>
                         <d_steps> <mat_or_file> <func_type> <compute_ref>
                         <sketch_type> <vec_nnz> <poly_lambda> <precision> <lfa_type> [n_1 n_2 ...]
    dir_path    : output directory (use "." for current directory)
    num_runs    : number of timed repetitions per matrix size
    k           : Nystrom rank (constant; must satisfy k <= n)
    k_mat       : matrix rank for generated types (constant; for kernel types equals data dim d;
                  ignored for file input)
    s           : Hutchinson sample count (constant)
    d_steps     : Lanczos depth per Hutchinson sample
    mat_or_file : 1=psd_alg, 2=psd_exp, 3=rbf_kernel, 4=poly_kernel, or path to .txt/.mtx file
    func_type   : 0=sqrt, 1=log, 2=poly (x*(x+poly_lambda))
    compute_ref : 0=skip eigendecomp reference, 1=run syevd (file input only)
    sketch_type : 0=Gaussian, 1=SASO
    vec_nnz     : nonzeros per column for SASO sketch (sketch_type=1; ignored for Gaussian)
    poly_lambda : lambda parameter for func_type=2 (ignored for func_type=0,1)
    precision   : double or float  (sparse .mtx path is always double)
    lfa_type    : 0=scalar LanczosFA, 1=BlockLanczosFA
    n_1 ...     : matrix dimensions (generated types only; ignored for file input)
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "load_mtx.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace linops = RandLAPACK::linops;
using RNG = r123::Philox4x32;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// ============================================================================
// Benchmark data struct
// ============================================================================

template <typename T>
struct FunNystromPP_benchmark_data {
    int64_t n;
    int64_t k;
    int64_t k_mat;
    int64_t s;
    int64_t d;
    bool from_file;
    T noise;
    int64_t d_dim;
    std::vector<T> A;
    std::vector<T> eigvals;

    FunNystromPP_benchmark_data(int64_t n_max, int64_t k_, int64_t k_mat_,
                                int64_t s_, int64_t d_, bool from_file_) :
        n(n_max), k(k_), k_mat(k_mat_), s(s_), d(d_), from_file(from_file_),
        noise((T)0.0), d_dim(0), A(n_max * n_max, 0.0)
    {}
};

// ============================================================================
// CountingLinOp: wraps any SLO and counts individual matrix-vector products.
// ============================================================================

template <typename SLO_t>
struct CountingLinOp {
    using scalar_t = typename SLO_t::scalar_t;
    const int64_t dim;
    SLO_t& base;
    int64_t count = 0;

    CountingLinOp(int64_t n, SLO_t& slo) : dim(n), base(slo) {}
    void reset() { count = 0; }

    void operator()(Layout layout, int64_t n_vecs, scalar_t alpha,
                    scalar_t* const B, int64_t ldb,
                    scalar_t beta, scalar_t* C, int64_t ldc) {
        count += n_vecs;
        base(layout, n_vecs, alpha, B, ldb, beta, C, ldc);
    }
};

// ============================================================================
// data_regen: build or load A for the current n.
// ============================================================================

template <typename T>
static void data_regen(
    FunNystromPP_benchmark_data<T>& data,
    RandBLAS::RNGState<RNG>& state,
    int mat_type,
    int func_type,
    const std::string& file_path
) {
    int64_t n = data.n;
    std::fill(data.A.begin(), data.A.begin() + n * n, (T)0.0);

    if (data.from_file) {
        int64_t m = n, ncols = n;
        int wq = 0;
        RandLAPACK::gen::process_input_mat<T>(
            m, ncols, data.A.data(),
            const_cast<char*>(file_path.c_str()), wq);
        data.eigvals.clear();
        data.noise = (T)0.0;
        data.d_dim = 0;
    } else if (mat_type >= 3) {
        data.d_dim = data.k_mat;
        data.noise = (T)0.0;
        data.eigvals.clear();
        T bandwidth = std::sqrt((T)data.d_dim);
        if (mat_type == 3)
            RandLAPACK::gen::gen_kernel_matrix<T, RNG>(
                n, data.d_dim, data.A.data(), n, 0, bandwidth, (T)0.0, 0, state);
        else
            RandLAPACK::gen::gen_kernel_matrix<T, RNG>(
                n, data.d_dim, data.A.data(), n, 1, (T)0.0, (T)1.0, 2, state);
    } else {
        data.d_dim = 0;
        data.eigvals.resize(data.k_mat);
        if (mat_type == 1)
            RandLAPACK::gen::gen_alg_decay_singvals(data.k_mat, (T)100.0, (T)2.0,
                                                     data.eigvals.data());
        else
            RandLAPACK::gen::gen_exp_decay_singvals(data.k_mat, (T)1.0, (T)100.0,
                                                     data.eigvals.data());
        if (func_type == 1) {
            data.noise = (T)1.0;
            RandLAPACK::gen::gen_shifted_lowrank_psd(n, data.k_mat, data.A.data(), n,
                                                      data.eigvals.data(), data.noise, state);
        } else {
            data.noise = (T)0.0;
            RandLAPACK::gen::gen_sym_psd_lowrank(n, data.k_mat, data.A.data(), n,
                                                  data.eigvals.data(), state);
        }
    }
}

// ============================================================================
// LFAOp: wraps LanczosFA as a SymmetricLinearOperator for plain Hutchinson.
// ============================================================================

template <typename SLO_t, typename LFA_t, typename F_t>
struct LFAOp {
    using scalar_t = typename SLO_t::scalar_t;
    const int64_t dim;
    SLO_t& A;
    LFA_t& lfa;
    F_t f;
    int64_t d;
    std::vector<scalar_t> Z_buf;

    LFAOp(int64_t n, SLO_t& A_, LFA_t& lfa_, F_t f_, int64_t d_)
        : dim(n), A(A_), lfa(lfa_), f(f_), d(d_), Z_buf() {}

    void operator()([[maybe_unused]] Layout layout, int64_t n_vecs, scalar_t alpha,
                    scalar_t* const B, [[maybe_unused]] int64_t ldb,
                    scalar_t beta, scalar_t* C, [[maybe_unused]] int64_t ldc)
    {
        Z_buf.assign(dim * n_vecs, (scalar_t)0);
        lfa.call(A, B, dim, n_vecs, f, d, Z_buf.data());
        if (beta == (scalar_t)0) {
            for (int64_t i = 0; i < dim * n_vecs; ++i)
                C[i] = alpha * Z_buf[i];
        } else {
            blas::scal(dim * n_vecs, beta, C, 1);
            blas::axpy(dim * n_vecs, alpha, Z_buf.data(), 1, C, 1);
        }
    }
};

// ============================================================================
// call_all_algs: timed + matvec-counted comparison for one matrix size.
// ============================================================================

template <typename T, typename LFA_t>
static void call_all_algs(
    int64_t numruns,
    FunNystromPP_benchmark_data<T>& data,
    const RandBLAS::RNGState<RNG>& state,
    int mat_type,
    int func_type,
    double poly_lambda,
    bool compute_ref,
    int sketch_type,
    int vec_nnz,
    const std::string& output_filename
) {
    int64_t n = data.n, k = data.k, s = data.s, d = data.d;

    std::function<T(T)> f;
    const char* fname;
    T f_zero = (T)0;
    if (func_type == 0) {
        f     = [](T x){ return (T)std::sqrt((double)std::max(x, (T)0)); };
        fname = "sqrt";
    } else if (func_type == 1) {
        f     = [](T x){ return (T)std::log((double)x); };
        fname = "log";
    } else {
        T lam = (T)poly_lambda;
        f     = [lam](T x){ return x * (x + lam); };
        fname = "poly";
    }

    bool ref_available = false;
    T true_tr = (T)0;
    if (!data.from_file && mat_type < 3) {
        for (int64_t j = 0; j < data.k_mat; ++j)
            true_tr += f(data.eigvals[j] + data.noise);
        true_tr += (T)(n - data.k_mat) * f(data.noise);
        ref_available = true;
    } else if (compute_ref) {
        std::vector<T> A_copy(data.A.begin(), data.A.begin() + n * n);
        std::vector<T> ev(n);
        lapack::syevd(lapack::Job::NoVec, lapack::Uplo::Upper, n,
                      A_copy.data(), n, ev.data());
        for (int64_t j = 0; j < n; ++j)
            true_tr += f(ev[j]);
        ref_available = true;
    }

    RandLAPACK::SYPS<T, RNG>                                             syps(3, 1, false, false);
    syps.sketch_type = sketch_type;
    syps.vec_nnz     = vec_nnz;
    RandLAPACK::HQRQ<T>                                                  orth(false, false);
    RandLAPACK::SYRF<RandLAPACK::SYPS<T, RNG>, RandLAPACK::HQRQ<T>>    syrf(syps, orth);
    RandLAPACK::NystromEVD<decltype(syrf)>                               nystrom_evd(syrf, 0);
    nystrom_evd.timing = true;
    LFA_t                                                                lfa_pp;
    lfa_pp.timing = true;
    RandLAPACK::Hutchinson<T, RNG>                                       hutch_pp;
    RandLAPACK::FunNystromPP<decltype(nystrom_evd), LFA_t, decltype(hutch_pp)>
                                                                         driver(nystrom_evd, lfa_pp, hutch_pp);

    LFA_t                           lfa_base;
    RandLAPACK::Hutchinson<T, RNG>  hutch_base;

    using ESLO = linops::ExplicitSymLinOp<T>;
    using CSLO = CountingLinOp<ESLO>;
    using FuncT = std::function<T(T)>;

    const char* mat_str = data.from_file ? "file" :
                          mat_type == 1  ? "psd_alg"     :
                          mat_type == 2  ? "psd_exp"     :
                          mat_type == 3  ? "rbf_kernel"  : "poly_kernel";

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d, N=%lld, K=%lld, K_MAT=%lld, S=%lld, D=%lld  [%s, %s]\n",
               i, (long long)n, (long long)k, (long long)data.k_mat,
               (long long)s, (long long)d, fname, mat_str);

        auto state_alg = state;

        // FunNystromPP
        ESLO A_op(n, blas::Uplo::Upper, data.A.data(), n, Layout::ColMajor);
        CSLO A_pp(n, A_op);

        RandLAPACK::FunNystromPP_timing pp_timing;
        auto start_pp = steady_clock::now();
        T est_pp = driver.call(A_pp, f, f_zero, k, s, d, state_alg, &pp_timing);
        auto stop_pp  = steady_clock::now();
        long dur_pp   = duration_cast<microseconds>(stop_pp - start_pp).count();
        int64_t mv_pp = A_pp.count;

        auto& nt = nystrom_evd.times;
        auto& lt = lfa_pp.times;

        printf("  FunNystromPP:   est=%.6e  time=%lld us  (ph1=%lld ph2=%lld)  matvecs=%lld\n",
               (double)est_pp, (long long)dur_pp,
               (long long)pp_timing.phase1_us, (long long)pp_timing.phase2_us,
               (long long)mv_pp);
        printf("    NystromEVD:   alloc=%lld syrf=%lld matvec=%lld gram=%lld potrf=%lld trsm=%lld svd=%lld post_svd=%lld err_est=%lld rest=%lld total=%lld\n",
               (long long)nt[0], (long long)nt[1], (long long)nt[2], (long long)nt[3],
               (long long)nt[4], (long long)nt[5], (long long)nt[6], (long long)nt[7],
               (long long)nt[8], (long long)nt[9], (long long)nt[10]);
        printf("    LanczosFA:    matvec=%lld lanczos=%lld apply_f=%lld total=%lld\n",
               (long long)lt[0], (long long)lt[1], (long long)lt[2], (long long)lt[4]);

        // Hutchinson + LanczosFA (state_alg continues from where FunNystromPP left it)
        ESLO A_op2(n, blas::Uplo::Upper, data.A.data(), n, Layout::ColMajor);
        CSLO A_hutch(n, A_op2);
        LFAOp<CSLO, LFA_t, FuncT> lfa_op(n, A_hutch, lfa_base, f, d);

        auto start_h = steady_clock::now();
        T est_h = hutch_base.call(lfa_op, s, state_alg);
        auto stop_h  = steady_clock::now();
        long dur_h    = duration_cast<microseconds>(stop_h - start_h).count();
        int64_t mv_h  = A_hutch.count;
        printf("  Hutchinson+LFA: est=%.6e  time=%lld us  matvecs=%lld\n",
               (double)est_h, (long long)dur_h, (long long)mv_h);

        double err_pp = ref_available ? std::abs((double)est_pp - (double)true_tr) / std::abs((double)true_tr) : -1.0;
        double err_h  = ref_available ? std::abs((double)est_h  - (double)true_tr) / std::abs((double)true_tr) : -1.0;
        if (ref_available)
            printf("  Reference:      true=%.6e  err_pp=%e  err_h=%e\n",
                   (double)true_tr, err_pp, err_h);
        else
            printf("  Reference:      N/A\n");

        double out_tr = ref_available ? (double)true_tr : -1.0;
        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        file << n                     << ",  " << k             << ",  " << data.k_mat  << ",  "
             << s                     << ",  " << d             << ",  "
             << mv_pp                 << ",  " << mv_h          << ",  "
             << dur_pp                << ",  " << pp_timing.phase1_us << ",  " << pp_timing.phase2_us << ",  "
             << dur_h                 << ",  "
             << nt[0]  << ",  " << nt[1]  << ",  " << nt[2]  << ",  " << nt[3]  << ",  "
             << nt[4]  << ",  " << nt[5]  << ",  " << nt[6]  << ",  " << nt[7]  << ",  "
             << nt[8]  << ",  " << nt[9]  << ",  " << nt[10] << ",  "
             << lt[0]  << ",  " << lt[1]  << ",  " << lt[2]  << ",  " << lt[3]  << ",  " << lt[4] << ",  "
             << (double)est_pp        << ",  " << (double)est_h << ",  "
             << out_tr                << ",  " << err_pp        << ",  " << err_h  << ",\n";
    }
}

// ============================================================================
// call_all_algs_sparse: sparse path (always double).
// ============================================================================

template <typename T, typename LFA_t>
static void call_all_algs_sparse(
    int64_t numruns,
    int64_t n,
    int64_t k,
    int64_t s,
    int64_t d,
    RandBLAS::sparse_data::CSRMatrix<T, int64_t>& csr,
    const RandBLAS::RNGState<RNG>& state,
    int func_type,
    double poly_lambda,
    int sketch_type,
    int vec_nnz,
    const std::string& output_filename
) {
    std::function<T(T)> f;
    const char* fname;
    T f_zero = (T)0;
    if (func_type == 0) {
        f     = [](T x){ return (T)std::sqrt((double)std::max(x, (T)0)); };
        fname = "sqrt";
    } else if (func_type == 1) {
        f     = [](T x){ return (T)std::log((double)x); };
        fname = "log";
    } else {
        T lam = (T)poly_lambda;
        f     = [lam](T x){ return x * (x + lam); };
        fname = "poly";
    }

    RandLAPACK::SYPS<T, RNG>                                              syps(3, 1, false, false);
    syps.sketch_type = sketch_type;
    syps.vec_nnz     = vec_nnz;
    RandLAPACK::HQRQ<T>                                                   orth(false, false);
    RandLAPACK::SYRF<RandLAPACK::SYPS<T, RNG>, RandLAPACK::HQRQ<T>>     syrf(syps, orth);
    RandLAPACK::NystromEVD<decltype(syrf)>                                nystrom_evd(syrf, 0);
    nystrom_evd.timing = true;
    LFA_t                                                                 lfa_pp;
    lfa_pp.timing = true;
    RandLAPACK::Hutchinson<T, RNG>                                        hutch_pp;
    RandLAPACK::FunNystromPP<decltype(nystrom_evd), LFA_t, decltype(hutch_pp)>
                                                                          driver(nystrom_evd, lfa_pp, hutch_pp);

    LFA_t                          lfa_base;
    RandLAPACK::Hutchinson<T, RNG> hutch_base;

    using SSLO = linops::SparseSymLinOp<RandBLAS::sparse_data::CSRMatrix<T, int64_t>>;
    using CSLO = CountingLinOp<SSLO>;
    using FuncT = std::function<T(T)>;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d, N=%lld, K=%lld, K_MAT=0, S=%lld, D=%lld  [%s, sparse_mtx]\n",
               i, (long long)n, (long long)k, (long long)s, (long long)d, fname);

        auto state_alg = state;

        SSLO A_op_pp(n, csr);
        CSLO A_pp(n, A_op_pp);

        RandLAPACK::FunNystromPP_timing pp_timing;
        auto start_pp = steady_clock::now();
        T est_pp = driver.call(A_pp, f, f_zero, k, s, d, state_alg, &pp_timing);
        auto stop_pp  = steady_clock::now();
        long dur_pp   = duration_cast<microseconds>(stop_pp - start_pp).count();
        int64_t mv_pp = A_pp.count;

        auto& nt = nystrom_evd.times;
        auto& lt = lfa_pp.times;

        printf("  FunNystromPP:   est=%.6e  time=%lld us  (ph1=%lld ph2=%lld)  matvecs=%lld\n",
               (double)est_pp, (long long)dur_pp,
               (long long)pp_timing.phase1_us, (long long)pp_timing.phase2_us,
               (long long)mv_pp);
        printf("    NystromEVD:   alloc=%lld syrf=%lld matvec=%lld gram=%lld potrf=%lld trsm=%lld svd=%lld post_svd=%lld err_est=%lld rest=%lld total=%lld\n",
               (long long)nt[0], (long long)nt[1], (long long)nt[2], (long long)nt[3],
               (long long)nt[4], (long long)nt[5], (long long)nt[6], (long long)nt[7],
               (long long)nt[8], (long long)nt[9], (long long)nt[10]);
        printf("    LanczosFA:    matvec=%lld lanczos=%lld apply_f=%lld total=%lld\n",
               (long long)lt[0], (long long)lt[1], (long long)lt[2], (long long)lt[4]);

        state_alg = state;
        SSLO A_op_h(n, csr);
        CSLO A_hutch(n, A_op_h);
        LFAOp<CSLO, LFA_t, FuncT> lfa_op(n, A_hutch, lfa_base, f, d);

        auto start_h = steady_clock::now();
        T est_h = hutch_base.call(lfa_op, s, state_alg);
        auto stop_h  = steady_clock::now();
        long dur_h   = duration_cast<microseconds>(stop_h - start_h).count();
        int64_t mv_h = A_hutch.count;
        printf("  Hutchinson+LFA: est=%.6e  time=%lld us  matvecs=%lld\n",
               (double)est_h, (long long)dur_h, (long long)mv_h);
        printf("  Reference:      N/A (sparse matrix)\n");

        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        file << n     << ",  " << k  << ",  " << 0   << ",  "
             << s     << ",  " << d  << ",  "
             << mv_pp << ",  " << mv_h << ",  "
             << dur_pp << ",  " << pp_timing.phase1_us << ",  " << pp_timing.phase2_us << ",  "
             << dur_h  << ",  "
             << nt[0]  << ",  " << nt[1]  << ",  " << nt[2]  << ",  " << nt[3]  << ",  "
             << nt[4]  << ",  " << nt[5]  << ",  " << nt[6]  << ",  " << nt[7]  << ",  "
             << nt[8]  << ",  " << nt[9]  << ",  " << nt[10] << ",  "
             << lt[0]  << ",  " << lt[1]  << ",  " << lt[2]  << ",  " << lt[3]  << ",  " << lt[4] << ",  "
             << (double)est_pp << ",  " << (double)est_h << ",  "
             << -1.0 << ",  " << -1.0 << ",  " << -1.0 << ",\n";
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 15) {
        std::cerr << "Usage: " << argv[0]
                  << " <dir_path> <num_runs> <k> <k_mat> <s>"
                     " <d_steps> <mat_or_file> <func_type> <compute_ref>"
                     " <sketch_type> <vec_nnz> <poly_lambda> <precision> <lfa_type> [n_1 n_2 ...]\n"
                  << "  dir_path    : output directory (use '.' for current dir)\n"
                  << "  num_runs    : timed repetitions per matrix size\n"
                  << "  k           : Nystrom rank (constant; must satisfy k <= n)\n"
                  << "  k_mat       : matrix rank for generated types (for kernel types = data dim)\n"
                  << "  s           : Hutchinson sample count (constant)\n"
                  << "  d_steps     : Lanczos depth per sample\n"
                  << "  mat_or_file : 1=psd_alg, 2=psd_exp, 3=rbf_kernel, 4=poly_kernel, or path\n"
                  << "  func_type   : 0=sqrt, 1=log, 2=poly (x*(x+poly_lambda))\n"
                  << "  compute_ref : 0=skip eigendecomp, 1=run syevd (file input only)\n"
                  << "  sketch_type : 0=Gaussian, 1=SASO\n"
                  << "  vec_nnz     : nonzeros per column for SASO (sketch_type=1; ignored otherwise)\n"
                  << "  poly_lambda : lambda for func_type=2 (ignored otherwise)\n"
                  << "  precision   : double or float  (sparse .mtx path is always double)\n"
                  << "  lfa_type    : 0=scalar LanczosFA, 1=BlockLanczosFA\n"
                  << "  n_1 ...     : matrix sizes (generated types only)\n";
        return 1;
    }

    int64_t numruns     = std::stol(argv[2]);
    int64_t k_const     = std::stol(argv[3]);
    int64_t k_mat_const = std::stol(argv[4]);
    int64_t s_const     = std::stol(argv[5]);
    int64_t d_steps     = std::stol(argv[6]);
    int     func_type   = std::stoi(argv[8]);
    bool    compute_ref = std::stoi(argv[9]) != 0;
    int     sketch_type = std::stoi(argv[10]);
    int     vec_nnz     = std::stoi(argv[11]);
    double  poly_lambda = std::stod(argv[12]);
    std::string precision_str = argv[13];
    bool    use_float   = (precision_str == "float");
    int     lfa_type    = std::stoi(argv[14]);
    const char* lfa_str = lfa_type == 0 ? "scalar" : "block";

    bool from_file = false;
    bool is_mtx    = false;
    std::string mat_file_path;
    int mat_type = 0;
    try {
        mat_type = std::stoi(argv[7]);
    } catch (...) {
        from_file = true;
        mat_file_path = std::string(argv[7]);
        auto& p = mat_file_path;
        is_mtx = p.size() >= 4 && p.substr(p.size() - 4) == ".mtx";
    }

    const char* func_str = func_type == 0 ? "sqrt" : (func_type == 1 ? "log" : "poly");
    const char* mat_str  = from_file    ? mat_file_path.c_str() :
                           mat_type == 1 ? "psd_alg"    :
                           mat_type == 2 ? "psd_exp"    :
                           mat_type == 3 ? "rbf_kernel" : "poly_kernel";
    const char* sketch_str = sketch_type == 1 ? "SASO" : "Gaussian";

    std::vector<int64_t> n_sizes;
    if (is_mtx) {
        int64_t mtx_n = 0;
        {
            std::ifstream peek(mat_file_path);
            std::string ln;
            while (std::getline(peek, ln))
                if (!ln.empty() && ln[0] != '%') break;
            std::istringstream ss(ln);
            int64_t nr, nc, nz; ss >> nr >> nc >> nz;
            mtx_n = nr;
        }
        n_sizes = {mtx_n};
    } else if (from_file) {
        int64_t fm = 0, fn = 0;
        int wq = 1;
        RandLAPACK::gen::process_input_mat<double>(
            fm, fn, nullptr,
            const_cast<char*>(mat_file_path.c_str()), wq);
        if (fm != fn)
            throw std::runtime_error("File matrix must be square (got "
                + std::to_string(fm) + "x" + std::to_string(fn) + ")");
        n_sizes = {fm};
    } else {
        if (argc < 16) {
            std::cerr << "Error: at least one matrix size (n_1) required for generated types.\n";
            return 1;
        }
        for (int i = 15; i < argc; ++i)
            n_sizes.push_back(std::stol(argv[i]));
    }

    std::ostringstream oss;
    for (auto v : n_sizes) oss << v << ", ";
    std::string sizes_str = oss.str();

    auto state          = RandBLAS::RNGState<RNG>();
    auto state_constant = state;

    std::string output_filename =
        "_FunNystromPP_benchmark_" + precision_str + "_" + std::string(lfa_str) + "_num_info_lines_8.txt";
    std::string path;
    if (std::string(argv[1]) != ".")
        path = std::string(argv[1]) + output_filename;
    else
        path = output_filename;

    const std::string mat_construction_str =
        is_mtx   ? "loaded from .mtx (sparse; both triangles expanded)" :
        from_file ? "loaded from .txt (dense)" :
        mat_type >= 3 ? "gen_kernel_matrix (random data points x_i~N(0,I_d); cost not timed)" :
                        "gen_sym_psd_lowrank (Haar-random eigenvectors; cost not timed)";

    std::ofstream file(path, std::ios::out | std::ios::app);
    file << "Description: FunNystromPP vs Hutchinson+LanczosFA benchmark for tr(f(A))."
            "\nFile format: 32 columns: n, k, k_mat, s, d, matvec_pp, matvec_hutch,"
            " time_pp_total (us), time_pp_phase1 (us), time_pp_phase2 (us), time_hutch (us),"
            " nystrom_alloc, nystrom_syrf, nystrom_matvec, nystrom_gram,"
            " nystrom_potrf, nystrom_trsm, nystrom_svd, nystrom_post_svd,"
            " nystrom_error_est, nystrom_rest, nystrom_total,"
            " lfa_matvec, lfa_run_lanczos, lfa_apply_f, lfa_rest, lfa_total (all us),"
            " est_fun_pp, est_hutch, true_tr (-1 if unavailable), err_fun_pp, err_hutch;"
            " rows = numruns repetitions per matrix size."
            "\nNum OMP threads: " + std::to_string(RandLAPACK::util::get_omp_threads()) +
            "\nInput sizes: " + sizes_str +
            "\nParameters: k=" + std::string(argv[3]) +
            " k_mat=" + std::string(argv[4]) +
            " s=" + std::string(argv[5]) +
            " d=" + std::to_string(d_steps) +
            " mat=" + std::string(mat_str) +
            " func=" + func_str +
            (func_type == 2 ? (" poly_lambda=" + std::to_string(poly_lambda)) : std::string("")) +
            " sketch=" + sketch_str +
            (sketch_type == 1 ? (" vec_nnz=" + std::to_string(vec_nnz)) : std::string("")) +
            " precision=" + precision_str +
            " lfa_type=" + std::string(lfa_str) +
            "\nNum runs per size: " + std::to_string(numruns) +
            "\nMatrix construction: " + mat_construction_str +
            "\n";
    file.flush();

    auto start_all = steady_clock::now();

    using ScalarLFA_d = RandLAPACK::LanczosFA<double, RNG>;
    using BlockLFA_d  = RandLAPACK::BlockLanczosFA<double, RNG>;
    using ScalarLFA_f = RandLAPACK::LanczosFA<float, RNG>;
    using BlockLFA_f  = RandLAPACK::BlockLanczosFA<float, RNG>;

    if (is_mtx) {
        // Sparse path: always double.
        if (use_float)
            printf("Note: sparse .mtx path runs in double regardless of precision flag.\n");
        printf("Loading sparse matrix from %s ...\n", mat_file_path.c_str());
        int64_t n = n_sizes[0];
        auto csr = FunNystromPP_bench::load_mtx(mat_file_path, n);
        printf("  Loaded: n=%lld  nnz=%lld\n", (long long)n, (long long)csr.nnz);
        if (lfa_type == 0)
            call_all_algs_sparse<double, ScalarLFA_d>(numruns, n, k_const, s_const, d_steps,
                                                      csr, state_constant,
                                                      func_type, poly_lambda, sketch_type, vec_nnz, path);
        else
            call_all_algs_sparse<double, BlockLFA_d>(numruns, n, k_const, s_const, d_steps,
                                                     csr, state_constant,
                                                     func_type, poly_lambda, sketch_type, vec_nnz, path);
    } else {
        // Dense path: dispatch on precision and lfa_type.
        auto run_dense = [&]<typename T, typename LFA_t>() {
            int64_t n_max     = *std::max_element(n_sizes.begin(), n_sizes.end());
            int64_t k_mat_max = from_file ? 0 : k_mat_const;
            FunNystromPP_benchmark_data<T> all_data(
                n_max, k_const, k_mat_max, s_const, d_steps, from_file);

            for (int64_t n : n_sizes) {
                all_data.n     = n;
                all_data.k     = k_const;
                all_data.k_mat = from_file ? 0 : std::min(k_mat_const, n);
                all_data.s     = s_const;
                all_data.d     = d_steps;

                auto state_gen = state_constant;
                data_regen(all_data, state_gen, mat_type, func_type, mat_file_path);

                call_all_algs<T, LFA_t>(numruns, all_data, state_constant, mat_type,
                                        func_type, poly_lambda, compute_ref, sketch_type, vec_nnz, path);
            }
        };

        if (!use_float && lfa_type == 0)
            run_dense.template operator()<double, ScalarLFA_d>();
        else if (!use_float && lfa_type == 1)
            run_dense.template operator()<double, BlockLFA_d>();
        else if (use_float && lfa_type == 0)
            run_dense.template operator()<float, ScalarLFA_f>();
        else
            run_dense.template operator()<float, BlockLFA_f>();
    }

    auto stop_all = steady_clock::now();
    long dur_all = duration_cast<microseconds>(stop_all - start_all).count();
    file << "Total benchmark execution time: " + std::to_string(dur_all) + "\n";
    file.flush();

    return 0;
}
#endif
