/*
ABRIK per-triplet accuracy analysis: ABRIK's singular triplets against GESDD's.

Runs ABRIK once per run at one (b_sz, num_matmuls) and GESDD once (deterministic, reused
across runs), then writes one row per triplet i = 1..k. Each row carries, for ABRIK's
triplet (_abrik) and for GESDD's own triplet (_gesdd):

  res_err   sqrt(||A v_i - s_i u_i||^2 + ||A' u_i - s_i v_i||^2) / s_i: the two-sided
            residual normalized per triplet, the computable estimate ABRIK's adaptive mode
            certifies against. It can be driven to machine precision and its level reads
            as a number of correct digits.
  res_sw    sqrt(||A v_i - s_i u_i||^2 + ||A' u_i - s_i v_i||^2): the two-sided absolute
            residual of Tropp and Webber (arXiv:2306.12418, eq. 6.1). It scales with s_i,
            so its level cannot be read as digits of accuracy.
  res_1s    ||A v_i - s_i u_i|| / s_i: the one-sided normalized residual of Tomas,
            Quintana-Orti and Anzt (doi:10.1177/10943420231179699, Section 4.1.1). ABRIK
            forms u_i from A v_i, so this side is accurate by construction and the metric
            can read machine precision on a triplet whose v_i is wrong.

and against GESDD's triplet:

  sval_diff |s_abrik_i - s_gesdd_i| / s_gesdd_i, the relative error of each singular value.
  svec_diff sqrt((sin^2 angle(u_gesdd_i, u_abrik_i) + sin^2 angle(v_gesdd_i, v_abrik_i)) / 2).
            The sine of the angle between two unit vectors is |R(2,2)| of the Householder
            QR factorization of [x1, x2], exact to O(eps); sqrt(1 - (x1'x2)^2) floors at
            sqrt(eps) by cancellation once the vectors nearly coincide.

The GESDD residual columns do not depend on the run; they are a baseline for the metrics
themselves, of order eps * s_1 / s_i for a backward-stable SVD.

Usage:
  ABRIK_accuracy_analysis <precision> <output_dir> <input_matrix_path> <m> <n> <b_sz>
                          <num_matmuls> <num_runs>

  precision   = double | float
  input       = .bin, whitespace-delimited text, or a dense-array .mtx; sparse input is
                rejected because GESDD needs the dense matrix
  m, n        = expected dimensions, checked against the file
  b_sz        = Krylov block size
  num_matmuls = Krylov iteration budget (max_krylov_iters). The matvec count in the header
                is b_sz * num_matmuls, the initial block A*Omega not counted.
  num_runs    = independent ABRIK runs, RNG seeds 0..num_runs-1

ABRIK returns at most ceil(num_matmuls / 2) * b_sz triplets per run, fewer when its rank
criterion narrows a block.

Output CSV: '#' metadata lines, the column header
  run, i, res_err_abrik, res_err_gesdd, sval_diff, svec_diff, res_sw_abrik, res_sw_gesdd,
  res_1s_abrik, res_1s_gesdd
then one row per (run, triplet). Each run also adds a '#' line with its ABRIK time and
triplet count.

Peak memory is about 3 m n + n^2 numbers (A, GESDD's working copy, U, and V^T then V) plus
GESDD's workspace and ABRIK's outputs: about 3.2 GB in double at m = n = 10000.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "ext_matrix_io.hh"
#include "abrik_bench_common.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

static const char* kUsage =
    "<precision> <output_dir> <input_matrix_path> <m> <n> <b_sz> <num_matmuls> <num_runs>";

// The three residuals of one triplet (u, v, s): returns the two-sided normalized one and
// writes the two-sided absolute (res_sw) and one-sided normalized (res_1s) ones.
// scratch_m and scratch_n are work buffers of length m and n.
template <typename T>
static T per_triplet_residuals(const T* A, int64_t m, int64_t n, const T* u, const T* v, T s,
                               T* scratch_m, T* scratch_n, T& res_sw, T& res_1s) {
    // scratch_m = A v - s u
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, (T) 1, A, m, v, 1, (T) 0, scratch_m, 1);
    blas::axpy(m, -s, u, 1, scratch_m, 1);
    T nrm_left = blas::nrm2(m, scratch_m, 1);
    // scratch_n = A' u - s v
    blas::gemv(Layout::ColMajor, Op::Trans, m, n, (T) 1, A, m, u, 1, (T) 0, scratch_n, 1);
    blas::axpy(n, -s, v, 1, scratch_n, 1);
    T nrm_right = blas::nrm2(n, scratch_n, 1);

    res_sw = std::hypot(nrm_left, nrm_right);
    res_1s = nrm_left / s;
    return res_sw / s;
}

// sin(angle(x1, x2)) for unit vectors of length len, as |R(2,2)| of the Householder QR
// factorization of [x1, x2]. work_buf holds 2*len entries, tau_buf two; both are overwritten.
template <typename T>
static T sin_angle_via_qr(const T* x1, const T* x2, int64_t len, T* work_buf, T* tau_buf) {
    blas::copy(len, x1, 1, &work_buf[0],   1);
    blas::copy(len, x2, 1, &work_buf[len], 1);
    lapack::geqrf(len, 2, work_buf, len, tau_buf);
    return std::abs(work_buf[len + 1]);
}

template <typename T>
static int run_analysis(int argc, char* argv[]) {
    if (argc < 9) return abrik_usage(argv[0], kUsage);

    std::string output_dir = argv[2];
    std::string input_path = argv[3];
    int64_t m_expected     = std::stoll(argv[4]);
    int64_t n_expected     = std::stoll(argv[5]);
    int64_t b_sz           = std::stoll(argv[6]);
    int64_t num_matmuls    = std::stoll(argv[7]);
    int     num_runs       = std::stoi(argv[8]);
    if (num_runs < 1 || b_sz < 1 || num_matmuls < 1 || num_matmuls > INT_MAX) {
        std::cerr << "Error: need num_runs >= 1, b_sz >= 1 and 1 <= num_matmuls <= " << INT_MAX << "\n";
        return 2;
    }
    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T) 0.85);

    auto mat = BenchIO::load_matrix<T>(input_path);
    if (mat.is_sparse) {
        std::cerr << "Error: ABRIK_accuracy_analysis needs dense input; '" << input_path
                  << "' is sparse\n";
        return 2;
    }
    int64_t m = mat.m;
    int64_t n = mat.n;
    if (m_expected != m || n_expected != n) {
        std::cerr << "Error: expected " << m_expected << " x " << n_expected << " but the file is "
                  << m << " x " << n << "\n";
        return 2;
    }
    T* A = mat.data();   // owned by mat
    printf("Matrix loaded: %ld x %ld\n", (long) m, (long) n);

    // Open the output before the expensive GESDD, so a bad path fails fast.
    std::ofstream file;
    std::string path = abrik_open_csv(output_dir, "ABRIK_accuracy_analysis", file);
    if (!file) return 1;
    file << std::setprecision(15);

    // GESDD once, on a copy since it destroys its input. With Job::SomeVec, U_g is m x n,
    // S_g has min(m, n) entries and VT_g is n x n.
    printf("Running GESDD (once; deterministic)...\n");
    T* U_g  = new T[m * n];
    T* S_g  = new T[std::min(m, n)];
    T* VT_g = new T[n * n];
    T* A_copy = new T[m * n];
    lapack::lacpy(MatrixType::General, m, n, A, m, A_copy, m);

    auto t0 = steady_clock::now();
    int64_t info = lapack::gesdd(Job::SomeVec, m, n, A_copy, m, S_g, U_g, m, VT_g, n);
    long dur_gesdd = duration_cast<microseconds>(steady_clock::now() - t0).count();
    delete[] A_copy;
    if (info != 0) {
        std::cerr << "Error: GESDD failed with info " << info << "\n";
        delete[] U_g; delete[] S_g; delete[] VT_g;
        return 1;
    }
    printf("GESDD: %.2f s\n", dur_gesdd / 1e6);

    // GESDD returns V^T (column-major, right singular vectors as rows); the metrics want V
    // with one vector per column.
    T* V_g = new T[n * n];
    RandLAPACK::util::transposition(n, n, VT_g, n, V_g, n, 0);
    delete[] VT_g;

    file << "# ABRIK per-triplet accuracy analysis\n"
         << "# RANDLAPACK_GIT_COMMIT=" << abrik_build_commit() << "\n"
         << "# Precision: " << argv[1] << "\n"
         << "# Input matrix: " << input_path << "\n"
         << "# Input size: " << m << " x " << n << "\n"
         << "# b_sz: " << b_sz << "\n"
         << "# num_matmuls: " << num_matmuls << "\n"
         << "# Total matvecs per run: " << b_sz * num_matmuls << " (initial block A*Omega not counted)\n"
         << "# Num runs: " << num_runs << " (distinct RNG seeds 0..num_runs-1)\n"
         << "# GESDD time (us): " << dur_gesdd << " (run once, reused across runs)\n"
         << "# res_err = sqrt(||Av-su||^2 + ||A'u-sv||^2) / s (two-sided, normalized per triplet)\n"
         << "# res_sw = sqrt(||Av-su||^2 + ||A'u-sv||^2) (two-sided absolute, Tropp and Webber eq. 6.1)\n"
         << "# res_1s = ||Av-su|| / s (one-sided normalized, Tomas, Quintana-Orti and Anzt Sec. 4.1.1)\n"
         << "# sval_diff = |s_abrik - s_gesdd| / s_gesdd; svec_diff = sqrt((sin^2 u-angle + sin^2 v-angle) / 2), sines via Householder QR\n"
         << "run, i, res_err_abrik, res_err_gesdd, sval_diff, svec_diff, "
            "res_sw_abrik, res_sw_gesdd, res_1s_abrik, res_1s_gesdd\n";
    file.flush();

    // Work buffers reused across runs.
    T* scratch_m = new T[m];
    T* scratch_n = new T[n];
    T* qr_buf_u  = new T[2 * m];
    T* qr_buf_v  = new T[2 * n];
    T  tau_u[2], tau_v[2];

    for (int run = 0; run < num_runs; ++run) {
        printf("\n########## Run %d/%d ##########\n", run + 1, num_runs);
        auto state_run = RandBLAS::RNGState<r123::Philox4x32>(static_cast<uint32_t>(run));

        printf("Running ABRIK (b_sz=%ld, num_matmuls=%ld)...\n", (long) b_sz, (long) num_matmuls);
        RandLAPACK::ABRIK<T, r123::Philox4x32> abrik(false, false, tol);
        abrik.max_krylov_iters = (int) num_matmuls;
        T* U_a = nullptr;
        T* V_a = nullptr;
        T* S_a = nullptr;

        auto t0a = steady_clock::now();
        int status = abrik.call(m, n, A, m, b_sz, U_a, V_a, S_a, state_run);
        long dur_abrik = duration_cast<microseconds>(steady_clock::now() - t0a).count();
        int64_t k_a = (status == 0) ? abrik.singular_triplets_found : 0;
        printf("ABRIK: %ld singular triplets, %.2f s\n", (long) k_a, dur_abrik / 1e6);
        if (status != 0)
            fprintf(stderr, "ABRIK failed with status %d in run %d; no rows written\n", status, run);

        // Per-run metadata as a comment line, which the readers skip.
        file << "# Run " << run << " ABRIK time (us): " << dur_abrik
             << ", triplets: " << k_a << ", status: " << status << "\n";

        for (int64_t i = 0; i < k_a; ++i) {
            const T* u_a = &U_a[m * i];
            const T* v_a = &V_a[n * i];
            const T* u_g = &U_g[m * i];
            const T* v_g = &V_g[n * i];

            T sw_abrik, os_abrik, sw_gesdd, os_gesdd;
            T res_abrik = per_triplet_residuals(A, m, n, u_a, v_a, S_a[i], scratch_m, scratch_n, sw_abrik, os_abrik);
            T res_gesdd = per_triplet_residuals(A, m, n, u_g, v_g, S_g[i], scratch_m, scratch_n, sw_gesdd, os_gesdd);
            T sval_diff = std::abs(S_a[i] - S_g[i]) / S_g[i];
            T sin_u = sin_angle_via_qr(u_g, u_a, m, qr_buf_u, tau_u);
            T sin_v = sin_angle_via_qr(v_g, v_a, n, qr_buf_v, tau_v);
            T svec_diff = std::sqrt((sin_u * sin_u + sin_v * sin_v) / 2);

            file << run << ", " << (i + 1) << ", "
                 << res_abrik << ", " << res_gesdd << ", "
                 << sval_diff << ", " << svec_diff << ", "
                 << sw_abrik << ", " << sw_gesdd << ", "
                 << os_abrik << ", " << os_gesdd << "\n";
            file.flush();
            if ((i + 1) % 50 == 0)
                printf("  Processed triplet %ld / %ld\n", (long) (i + 1), (long) k_a);
        }

        delete[] U_a;
        delete[] V_a;
        delete[] S_a;
    }

    file.close();
    if (file.fail())
        std::cerr << "Error: writing " << path << " failed\n";
    else
        printf("Results written to: %s\n", path.c_str());

    delete[] U_g;
    delete[] S_g;
    delete[] V_g;
    delete[] scratch_m;
    delete[] scratch_n;
    delete[] qr_buf_u;
    delete[] qr_buf_v;
    return file.fail() ? 1 : 0;
}

int main(int argc, char* argv[]) {
    return abrik_bench_main(argc, argv, kUsage, run_analysis<double>, run_analysis<float>);
}
