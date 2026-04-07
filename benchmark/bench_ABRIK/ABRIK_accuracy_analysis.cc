/*
ABRIK per-triplet accuracy analysis benchmark.

Runs ABRIK and GESDD (full SVD) for a single (b_sz, num_matmuls) configuration,
then computes per-triplet accuracy metrics comparing ABRIK's approximate singular
triplets against GESDD's (essentially exact) triplets.

=== Metrics ===

Four metrics are computed for each singular triplet i = 1, ..., k:

1. res_err_abrik — Per-triplet SVD residual for ABRIK.
   Formula: sqrt(||E_left_i||^2 + ||E_right_i||^2)
   where    E_left_i  = sigma_i^{-1} A v_i - u_i      (the "left residual")
            E_right_i = v_i - A^T u_i sigma_i^{-1}     (the "right residual")

   Equivalently: sqrt(||A v_i - u_i sigma_i||^2 + ||A^T u_i - v_i sigma_i||^2) / sigma_i.

   This is the "correct" a-posteriori error estimator per collaborator guidance:
   it uses inv(Sigma) * A * V - U and V - A^T * U * inv(Sigma), which CAN be driven
   to machine precision with enough Krylov iterations. This is in contrast to the
   Sigma-scaled variant (||A V - U Sigma|| stacked with ||A^T U - V Sigma||) used
   in some prior work (e.g., Rob/Hartwig's paper), which CANNOT be driven to machine
   precision in general because the scaling by Sigma amplifies errors in later triplets.

2. res_err_gesdd — Same formula as (1) but using GESDD's factors.
   This serves as a baseline: GESDD computes the SVD to backward-stable accuracy,
   so this line should sit at O(eps) ≈ 1e-15 for all triplets, confirming the metric
   and matrix are behaving correctly.

3. sval_diff — Relative singular value difference.
   Formula: |sigma_abrik_i - sigma_gesdd_i| / sigma_gesdd_1
   Normalized by the largest singular value (sigma_gesdd_1) so the metric is
   scale-invariant and directly interpretable as relative accuracy.

4. svec_diff — Singular vector angular difference via QR-based sin(angle).
   Formula: sqrt((sin^2(angle(u_g, u_a)) + sin^2(angle(v_g, v_a))) / 2)

   The sin(angle) between two unit vectors x1, x2 is computed as |R(2,2)| from
   the Householder QR factorization of the m-by-2 matrix [x1, x2]. This is
   numerically stable to O(eps) because Householder QR uses orthogonal
   transformations.

   WHY NOT sqrt(1 - cos^2(theta))?
   The naive approach computes cos(theta) = x1 · x2 (a dot product), then
   sin(theta) = sqrt(1 - cos^2(theta)). When cos(theta) ≈ 1 (well-converged
   triplets), this suffers catastrophic cancellation: 1 - cos^2 ≈ 1 - 1 loses
   all significant digits. The result floors at sqrt(eps) ≈ 1e-8 regardless of
   the true angular accuracy. The QR approach avoids this entirely by computing
   sin(theta) directly, reaching O(eps) ≈ 1e-16 for well-converged triplets.

   MATHEMATICAL RELATIONSHIP:
   The combined metric sqrt((sin_u^2 + sin_v^2)/2) is mathematically equivalent
   to the old formula sqrt(1 - cos_u^2/2 - cos_v^2/2), since
     1 - cos_u^2/2 - cos_v^2/2
       = (1 - cos_u^2)/2 + (1 - cos_v^2)/2
       = sin_u^2/2 + sin_v^2/2.
   The only difference is numerical: the QR computation evaluates this quantity
   to full machine precision.

=== Usage ===

  ABRIK_accuracy_analysis <precision> <output_dir> <input_matrix_path> <m> <n> <b_sz> <num_matmuls>

  precision        : "double" or "float"
  output_dir       : directory for the output CSV (use "." for current directory)
  input_matrix_path: path to input matrix file (text format, read via mat_gen)
  m, n             : expected matrix dimensions (verified against file)
  b_sz             : Krylov block size
  num_matmuls      : number of block matrix-vector products (= max_krylov_iters)

  Total matvecs = b_sz * num_matmuls.
  ABRIK produces b_sz * num_matmuls / 2 singular triplets.

=== Output CSV format ===

  Metadata lines prefixed with '#', then header, then data rows:
  i, res_err_abrik, res_err_gesdd, sval_diff, svec_diff

=== Memory requirements ===

  For an m-by-n matrix (m >= n), peak memory is approximately:
    A + A_copy + U_gesdd + VT_gesdd + V_gesdd + ABRIK outputs + scratch
    = m*n + m*n + m*n + n*n + n*n + O(m*k) + O(m)
  For m = n = 10000: approximately 4.8 GB.
  A_copy and VT_gesdd are freed before the metric computation loop.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>
#include <algorithm>
#include <cmath>

// Per-triplet SVD residual: sqrt(||E_left_i||^2 + ||E_right_i||^2).
//
// Computed as sqrt(||Av_i - u_i*s_i||^2 + ||A'u_i - v_i*s_i||^2) / s_i,
// which is algebraically equivalent to the E_left/E_right form (see header).
//
// scratch_m (length m) and scratch_n (length n) are pre-allocated work buffers,
// overwritten on each call.
template <typename T>
static T per_triplet_residual(T* A, int64_t m, int64_t n,
                              T* u_i, T* v_i, T s_i,
                              T* scratch_m, T* scratch_n) {
    // scratch_m = A * v_i  (m-by-n times n-by-1 → m-by-1)
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, (T)1, A, m, v_i, 1, (T)0, scratch_m, 1);
    // scratch_m -= u_i * s_i  →  scratch_m = A*v_i - u_i*s_i
    blas::axpy(m, -s_i, u_i, 1, scratch_m, 1);
    T nrm_left = blas::nrm2(m, scratch_m, 1);

    // scratch_n = A' * u_i  (n-by-m times m-by-1 → n-by-1)
    blas::gemv(Layout::ColMajor, Op::Trans, m, n, (T)1, A, m, u_i, 1, (T)0, scratch_n, 1);
    // scratch_n -= v_i * s_i  →  scratch_n = A'*u_i - v_i*s_i
    blas::axpy(n, -s_i, v_i, 1, scratch_n, 1);
    T nrm_right = blas::nrm2(n, scratch_n, 1);

    // sqrt(nrm_left^2 + nrm_right^2) / s_i = sqrt(||E_left||^2 + ||E_right||^2)
    return std::hypot(nrm_left, nrm_right) / s_i;
}

// Compute sin(angle) between two unit vectors x1 and x2 of length len,
// using Householder QR factorization of the len-by-2 matrix [x1, x2].
//
// After QR: [x1, x2] = Q * R, where R is 2x2 upper triangular.
//   R(1,1) = ||x1|| = 1  (since x1 is unit)
//   R(1,2) = x1 · x2 = cos(angle)
//   R(2,2) = sin(angle)
//
// This is numerically stable to O(eps) because Householder reflections are
// orthogonal transformations. In contrast, the naive formula
//   sin(theta) = sqrt(1 - (x1·x2)^2)
// suffers catastrophic cancellation when x1·x2 ≈ 1, flooring at sqrt(eps).
//
// work_buf must have length >= 2*len. tau_buf must have length >= 2.
// Both are overwritten.
template <typename T>
static T sin_angle_via_qr(T* x1, T* x2, int64_t len, T* work_buf, T* tau_buf) {
    // Form M = [x1, x2] in column-major: col 0 at work_buf[0..len-1],
    //                                     col 1 at work_buf[len..2*len-1].
    blas::copy(len, x1, 1, &work_buf[0],   1);
    blas::copy(len, x2, 1, &work_buf[len], 1);

    // QR factorization: M = Q * R. R is stored in upper triangle of M.
    lapack::geqrf(len, 2, work_buf, len, tau_buf);

    // R(2,2) = work_buf[len + 1] in 0-indexed column-major (row 1, col 1).
    // |R(2,2)| = sin(angle(x1, x2)).
    return std::abs(work_buf[len + 1]);
}

template <typename T>
static void run_analysis(int argc, char *argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input_matrix_path> <m> <n> <b_sz> <num_matmuls>"
                  << std::endl;
        return;
    }

    std::string output_dir    = argv[2];
    int64_t m_expected        = std::stol(argv[4]);
    int64_t n_expected        = std::stol(argv[5]);
    int64_t b_sz              = std::stol(argv[6]);
    int64_t num_matmuls       = std::stol(argv[7]);
    T tol                     = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state                = RandBLAS::RNGState();
    int64_t m = 0, n = 0;

    // Load input matrix via RandLAPACK's file reader.
    // First call with NULL queries dimensions; second reads data.
    RandLAPACK::gen::mat_gen_info<T> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[3];
    m_info.workspace_query_mod = 1;
    RandLAPACK::gen::mat_gen<T>(m_info, NULL, state);
    m = m_info.rows;
    n = m_info.cols;

    if (m_expected != m || n_expected != n) {
        std::cerr << "Expected (" << m_expected << ", " << n_expected
                  << ") but got (" << m << ", " << n << "). Aborting." << std::endl;
        return;
    }

    T* A = new T[m * n]();
    RandLAPACK::gen::mat_gen(m_info, A, state);
    printf("Matrix loaded: %ld x %ld\n", m, n);

    // ======================================================================
    // Phase 1: Run ABRIK
    // ======================================================================
    // ABRIK does not modify A (unlike RSVD which deflates in-place).
    // It allocates U, V, Sigma internally with new[].
    printf("Running ABRIK (b_sz=%ld, num_matmuls=%ld, total_matvecs=%ld)...\n",
           b_sz, num_matmuls, b_sz * num_matmuls);

    RandLAPACK::ABRIK<T, r123::Philox4x32> abrik(true, false, tol);
    abrik.max_krylov_iters = (int) num_matmuls;

    T* U_a = nullptr;
    T* V_a = nullptr;
    T* S_a = nullptr;
    auto state_alg = state;

    auto t0 = std::chrono::steady_clock::now();
    abrik.call(m, n, A, m, b_sz, U_a, V_a, S_a, state_alg);
    auto t1 = std::chrono::steady_clock::now();
    long dur_abrik = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    int64_t k_a = abrik.singular_triplets_found;
    printf("ABRIK: %ld singular triplets, %.2f s\n", k_a, dur_abrik / 1e6);

    // ======================================================================
    // Phase 2: Run GESDD (full SVD via divide-and-conquer)
    // ======================================================================
    // GESDD with Job::SomeVec computes A = U_g * diag(S_g) * VT_g.
    // It destroys its input matrix, so we work on a copy.
    // U_g is m-by-n, S_g is length min(m,n), VT_g is n-by-n.
    printf("Running GESDD...\n");
    T* U_g  = new T[m * n]();
    T* S_g  = new T[std::min(m, n)]();
    T* VT_g = new T[n * n]();

    T* A_copy = new T[m * n];
    lapack::lacpy(MatrixType::General, m, n, A, m, A_copy, m);

    t0 = std::chrono::steady_clock::now();
    lapack::gesdd(Job::SomeVec, m, n, A_copy, m, S_g, U_g, m, VT_g, n);
    t1 = std::chrono::steady_clock::now();
    long dur_gesdd = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("GESDD: %.2f s\n", dur_gesdd / 1e6);

    // Free A_copy immediately — no longer needed.
    delete[] A_copy;

    // GESDD returns right singular vectors as VT (row-major V^T).
    // Transpose to column-major V for per-column access.
    T* V_g = new T[n * n]();
    RandLAPACK::util::transposition(n, n, VT_g, n, V_g, n, 0);
    delete[] VT_g;  // Free VT_g — only V_g needed from here.

    // ======================================================================
    // Phase 3: Compute per-triplet accuracy metrics
    // ======================================================================
    printf("Computing per-triplet metrics for %ld triplets...\n", k_a);

    // Pre-allocate all scratch buffers used in the metric computation loop.
    T* scratch_m = new T[m];           // For per_triplet_residual (GEMV result, length m)
    T* scratch_n = new T[n];           // For per_triplet_residual (GEMV result, length n)
    T* qr_buf_u  = new T[2 * m];      // For sin_angle_via_qr on u vectors (m-by-2 matrix)
    T* qr_buf_v  = new T[2 * n];      // For sin_angle_via_qr on v vectors (n-by-2 matrix)
    T  tau_u[2], tau_v[2];             // Householder scalars for 2-column QR

    // Generate output file with date/time prefix.
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    std::string output_filename = std::string(date_prefix) + "ABRIK_accuracy_analysis.csv";
    std::string path = (output_dir != ".") ? output_dir + "/" + output_filename : output_filename;
    std::ofstream file(path);

    // Write metadata header (lines prefixed with # for easy parsing).
    file << "# ABRIK Per-Triplet Accuracy Analysis\n"
         << "# Precision: " << argv[1] << "\n"
         << "# Input matrix: " << argv[3] << "\n"
         << "# Input size: " << m << " x " << n << "\n"
         << "# b_sz: " << b_sz << "\n"
         << "# num_matmuls: " << num_matmuls << "\n"
         << "# Total matvecs: " << b_sz * num_matmuls << "\n"
         << "# Singular triplets: " << k_a << "\n"
         << "# ABRIK time (us): " << dur_abrik << "\n"
         << "# GESDD time (us): " << dur_gesdd << "\n"
         << "# Residual metric: sqrt(||E_left||^2 + ||E_right||^2) where E_left = inv(s)*Av - u, E_right = v - A'u*inv(s)\n"
         << "# svec_diff metric: sqrt((sin^2(angle(u_g,u_a)) + sin^2(angle(v_g,v_a)))/2), sin via Householder QR\n";
    file << "i, res_err_abrik, res_err_gesdd, sval_diff, svec_diff\n";

    for (int64_t i = 0; i < k_a; ++i) {
        T* u_a = &U_a[m * i];   // ABRIK left singular vector i
        T* v_a = &V_a[n * i];   // ABRIK right singular vector i
        T  s_a = S_a[i];        // ABRIK singular value i

        T* u_g = &U_g[m * i];   // GESDD left singular vector i
        T* v_g = &V_g[n * i];   // GESDD right singular vector i
        T  s_g = S_g[i];        // GESDD singular value i

        // --- Metric 1 & 2: Per-triplet SVD residuals ---
        T res_abrik = per_triplet_residual(A, m, n, u_a, v_a, s_a, scratch_m, scratch_n);
        T res_gesdd = per_triplet_residual(A, m, n, u_g, v_g, s_g, scratch_m, scratch_n);

        // --- Metric 3: Relative singular value difference ---
        T sval_diff = std::abs(s_a - s_g) / s_g;

        // --- Metric 4: Singular vector angle via QR ---
        // sin(angle) between GESDD and ABRIK singular vectors, computed via
        // Householder QR of the 2-column matrix [x_gesdd, x_abrik].
        // This avoids the sqrt(1-cos^2) catastrophic cancellation.
        T sin_u = sin_angle_via_qr(u_g, u_a, m, qr_buf_u, tau_u);
        T sin_v = sin_angle_via_qr(v_g, v_a, n, qr_buf_v, tau_v);
        T svec_diff = std::sqrt((sin_u * sin_u + sin_v * sin_v) / 2.0);

        file << std::setprecision(15)
             << (i + 1) << ", " << res_abrik << ", " << res_gesdd << ", "
             << sval_diff << ", " << svec_diff << "\n";

        if ((i + 1) % 50 == 0)
            printf("  Processed triplet %ld / %ld\n", i + 1, k_a);
    }

    file.flush();
    file.close();
    printf("Results written to: %s\n", path.c_str());

    // Cleanup all allocations.
    delete[] A;
    delete[] U_a;       // Allocated by ABRIK with new[]
    delete[] V_a;
    delete[] S_a;
    delete[] U_g;       // Allocated in this function with new[]
    delete[] S_g;
    delete[] V_g;
    delete[] scratch_m;
    delete[] scratch_n;
    delete[] qr_buf_u;
    delete[] qr_buf_v;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision: double|float> <output_dir> <input_matrix_path>"
                  << " <m> <n> <b_sz> <num_matmuls>" << std::endl;
        return 1;
    }

    std::string precision = argv[1];
    if (precision == "double") {
        run_analysis<double>(argc, argv);
    } else if (precision == "float") {
        run_analysis<float>(argc, argv);
    } else {
        std::cerr << "Error: precision must be 'double' or 'float', got '"
                  << precision << "'" << std::endl;
        return 1;
    }
    return 0;
}
