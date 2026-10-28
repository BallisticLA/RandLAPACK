// Utilities shared by the benchmark programs in this directory. Part of the
// benchmark project, not of RandLAPACK, so a RandLAPACK install leaves it out.
// External plotting and campaign scripts parse the file names and row formats
// written here, so any change to them changes every benchmark's output.

#pragma once

#include <RandLAPACK.hh>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace RandLAPACK {
namespace bench {

// ============================================================================
// Command line and output files
// ============================================================================

/// Prints the usage line and returns the exit status of an argument error.
inline int usage_error(const char* argv0, const char* usage) {
    std::cerr << "Usage: " << argv0 << " " << usage << "\n";
    return 2;
}

/// A main() that dispatches on argv[1], "double" or "float" ("single" is a synonym), to run
/// functions that return the exit status. An exception (an unparsable number, a bad input
/// file, a library precondition) is reported and exits with 1 instead of aborting.
inline int precision_main(int argc, char* argv[], const char* usage,
                          int (*run_double)(int, char**), int (*run_float)(int, char**)) {
    if (argc < 2) return usage_error(argv[0], usage);
    std::string precision = argv[1];
    try {
        if (precision == "double") return run_double(argc, argv);
        if (precision == "float" || precision == "single") return run_float(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "Error: precision must be 'double' or 'float', got '" << precision << "'\n";
    return usage_error(argv[0], usage);
}

/// Reads argv[first], ..., argv[argc - 1] as sizes into the argc - first entries of sizes.
inline void parse_sizes(int argc, char* argv[], int first, int64_t* sizes) {
    for (int i = 0; i < argc - first; ++i)
        sizes[i] = std::stoi(argv[i + first]);
}

/// Formats a size list for a file header as "s1, s2, ..., sk, ".
inline std::string sizes_string(const int64_t* sizes, int64_t count) {
    std::ostringstream oss;
    for (int64_t i = 0; i < count; ++i)
        oss << sizes[i] << ", ";
    return oss.str();
}

/// Grows buf to at least needed entries, keeping its contents and zero-filling the rest.
template <typename T>
void grow(T*& buf, int64_t& buf_sz, int64_t needed) {
    if (needed <= buf_sz)
        return;
    T* new_buf = new T[needed]();
    std::copy(buf, buf + buf_sz, new_buf);
    delete[] buf;
    buf    = new_buf;
    buf_sz = needed;
}

/// Output path from a prefix given on the command line, prepended verbatim rather than
/// joined as a directory: campaign scripts pass a directory and the start of a file name,
/// so <dir>/<date> gives <dir>/<date><filename>. A bare directory must end in a
/// separator; "." means no prefix.
inline std::string output_path(const std::string& prefix, const std::string& filename) {
    return (prefix != ".") ? prefix + filename : filename;
}

/// Local time as "YYYYmmdd_HHMMSS_", the prefix of timestamped result files.
inline std::string timestamp_prefix() {
    std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char prefix[20];
    std::strftime(prefix, sizeof(prefix), "%Y%m%d_%H%M%S_", &local);
    return std::string(prefix);
}

/// Opens <dir>/<YYYYmmdd_HHMMSS>_<name>.csv for writing and returns its path. Unlike
/// output_path, dir is a directory: "/" is added unless it already ends in one, and "."
/// means no prefix. The caller checks the stream: a missing directory leaves it closed.
inline std::string open_csv(const std::string& dir, const std::string& name, std::ofstream& out) {
    bool ends_in_separator = !dir.empty() && dir.back() == '/';
    std::string prefix = (dir == "." || ends_in_separator) ? dir : dir + "/";
    std::string path   = output_path(prefix, timestamp_prefix() + name + ".csv");
    out.open(path);
    if (!out) std::cerr << "Error: cannot open " << path << " for writing\n";
    return path;
}

/// The commit the binary was built from, read from the environment variable
/// RANDLAPACK_GIT_COMMIT that the launching script exports; "unknown" when it is unset or
/// empty. Read at run time rather than baked in at configure time, so a broken chain
/// shows "unknown" instead of a stale SHA.
inline std::string build_commit() {
    const char* v = std::getenv("RANDLAPACK_GIT_COMMIT");
    return (v && *v) ? std::string(v) : std::string("unknown");
}

/// Writes one result row: values separated by ",  ", terminated by ",\n".
template <typename First, typename... Rest>
void write_row(std::ostream& out, const First& first, const Rest&... rest) {
    out << first;
    ((out << ",  " << rest), ...);
    out << ",\n";
}

/// Writes one runtime-breakdown row: each value followed by ", ", then "\n".
template <typename It>
void write_breakdown_row(std::ostream& out, It first, It last) {
    for (; first != last; ++first)
        out << *first << ", ";
    out << "\n";
}

/// Writes the closing "Total benchmark execution time:<us>" line, timed from start.
inline void write_total_time(std::ostream& out, std::chrono::steady_clock::time_point start) {
    long dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    out << "Total benchmark execution time:" + std::to_string(dur) + "\n";
    out.flush();
}

// ============================================================================
// QR and QRCP benchmark data
// ============================================================================

/// Scalars and optional buffers of a QR_benchmark_data; A, tau and J always exist.
template <typename T>
struct QR_benchmark_options {
    T    d_factor = 1.0;
    T    tol      = 0.0;
    bool R        = false;  // n x n R factor
    bool S        = false;  // n singular values
    bool copies   = false;  // two m x n copies of A and an n x n identity, for error checks
};

/// Input matrix and outputs of a QR or QRCP driver, allocated zeroed at construction for the
/// largest size; grow() can enlarge a buffer later. Unrequested buffers stay nullptr, size 0.
template <typename T>
struct QR_benchmark_data {
    int64_t row;
    int64_t col;
    int64_t rank;
    T tolerance;
    T sampling_factor;
    T*       A      = nullptr; int64_t A_sz      = 0;  // m x n
    T*       R      = nullptr; int64_t R_sz      = 0;  // n x n
    T*       tau    = nullptr; int64_t tau_sz    = 0;  // n
    int64_t* J      = nullptr; int64_t J_sz      = 0;  // n
    T*       S      = nullptr; int64_t S_sz      = 0;  // n
    T*       A_cpy1 = nullptr; int64_t A_cpy1_sz = 0;  // m x n
    T*       A_cpy2 = nullptr; int64_t A_cpy2_sz = 0;  // m x n
    T*       I_ref  = nullptr; int64_t I_ref_sz  = 0;  // n x n

    QR_benchmark_data(int64_t m, int64_t n, const QR_benchmark_options<T>& opts = {}) :
    row(m),
    col(n),
    rank(n),
    tolerance(opts.tol),
    sampling_factor(opts.d_factor)
    {
        alloc(A, A_sz, m * n);
        if (opts.R) alloc(R, R_sz, n * n);
        alloc(tau, tau_sz, n);
        alloc(J, J_sz, n);
        if (opts.S) alloc(S, S_sz, n);
        if (opts.copies) {
            alloc(A_cpy1, A_cpy1_sz, m * n);
            alloc(A_cpy2, A_cpy2_sz, m * n);
            alloc(I_ref, I_ref_sz, n * n);
        }
    }

    QR_benchmark_data(const QR_benchmark_data&) = delete;
    QR_benchmark_data& operator=(const QR_benchmark_data&) = delete;

    ~QR_benchmark_data() {
        delete[] A;
        delete[] R;
        delete[] tau;
        delete[] J;
        delete[] S;
        delete[] A_cpy1;
        delete[] A_cpy2;
        delete[] I_ref;
    }

    private:
    template <typename U>
    static void alloc(U*& buf, int64_t& buf_sz, int64_t sz) {
        buf    = new U[sz]();
        buf_sz = sz;
    }
};

/// Regenerates A from state and clears the outputs; S is kept. zero_A clears A
/// before generating, which generators that accumulate into A (Kahan) rely on.
template <typename T, typename RNG>
void data_regen(gen::mat_gen_info<T> m_info,
                QR_benchmark_data<T> &all_data,
                RandBLAS::RNGState<RNG> &state,
                bool zero_A = true) {

    if (zero_A)
        std::fill(all_data.A, all_data.A + all_data.A_sz, 0.0);
    gen::mat_gen(m_info, all_data.A, state);
    if (all_data.A_cpy1 != nullptr) {
        lapack::lacpy(lapack::MatrixType::General, all_data.row, all_data.col, all_data.A, all_data.row, all_data.A_cpy1, all_data.row);
        lapack::lacpy(lapack::MatrixType::General, all_data.row, all_data.col, all_data.A, all_data.row, all_data.A_cpy2, all_data.row);
    }
    std::fill(all_data.R, all_data.R + all_data.R_sz, 0.0);
    std::fill(all_data.tau, all_data.tau + all_data.tau_sz, 0.0);
    std::fill(all_data.J, all_data.J + all_data.J_sz, 0);
    std::fill(all_data.I_ref, all_data.I_ref + all_data.I_ref_sz, 0.0);
}

/// The m x n inputs of the QR error analyses: polynomial, staircase, spiked and Kahan.
template <typename T>
std::array<gen::mat_gen_info<T>, 4> qr_test_matrices(int64_t m, int64_t n) {
    gen::mat_gen_info<T> m_info_poly(m, n, gen::polynomial);
    m_info_poly.cond_num = std::pow(10, 10);
    m_info_poly.exponent = 2.0;
    gen::mat_gen_info<T> m_info_stair(m, n, gen::step);
    m_info_stair.cond_num = std::pow(10, 10);
    gen::mat_gen_info<T> m_info_spiked(m, n, gen::spiked);
    m_info_spiked.scaling = std::pow(10, 10);
    gen::mat_gen_info<T> m_info_kahan(m, n, gen::kahan);
    m_info_kahan.theta   = 1.2;
    m_info_kahan.perturb = 1e3;
    return {m_info_poly, m_info_stair, m_info_spiked, m_info_kahan};
}

/// Returns {||AP - QR||_F / ||AP||_F, ||Q'Q - I||_F / sqrt(n)} of one run and folds them into
/// error_output[0:4] = {sum, max, sum, max}. Expects AP in A_cpy1 (overwritten), Q in A, and
/// an n x n R (ld n) in R; uses the first all_data.rank columns of Q.
template <typename T>
std::pair<T, T> qr_error_check(QR_benchmark_data<T> &all_data, int64_t n, T* error_output) {

    auto m = all_data.row;
    auto k = all_data.rank;

    grow(all_data.I_ref, all_data.I_ref_sz, k * k);
    util::eye(k, k, all_data.I_ref);

    T* A_dat       = all_data.A_cpy1;
    T const* Q_dat = all_data.A;
    T const* R_dat = all_data.R;
    T* I_ref_dat   = all_data.I_ref;

    T norm_A = lapack::lange(lapack::Norm::Fro, m, n, A_dat, m);

    // Q' * Q - I
    blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans, k, m, (T) 1.0, Q_dat, m, (T) -1.0, I_ref_dat, k);
    T norm_0 = lapack::lansy(lapack::Norm::Fro, blas::Uplo::Upper, k, I_ref_dat, k);

    // AP - QR
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k, (T) 1.0, Q_dat, m, R_dat, n, (T) -1.0, A_dat, m);
    T norm_AQR = lapack::lange(lapack::Norm::Fro, m, n, A_dat, m);

    T reconstruction_error = norm_AQR / norm_A;
    T orth_loss            = norm_0 / std::sqrt((T) n);

    error_output[0] += reconstruction_error;
    if (reconstruction_error > error_output[1]) { error_output[1] = reconstruction_error;}
    error_output[2] += orth_loss;
    if (orth_loss > error_output[3]) { error_output[3] = orth_loss;}

    return {reconstruction_error, orth_loss};
}

/// Writes the error-analysis row: mean and (max - mean) of each of the two errors.
template <typename T>
void write_error_row(std::ostream& out, const T* error_output, int num_runs) {
    T avg_reconstruction_error = error_output[0] / num_runs;
    T avg_orthogonality_loss   = error_output[2] / num_runs;
    write_row(out, avg_reconstruction_error, error_output[1] - avg_reconstruction_error, avg_orthogonality_loss, error_output[3] - avg_orthogonality_loss);
}

/// Writes to R_norms[0:n] the Frobenius norms of the trailing blocks R[i:, i:] of an n x n
/// upper-triangular R; echoes the first ten to stdout.
template <typename T>
void trailing_R_norms(int64_t n, const T* R, int64_t ldr, T* R_norms) {
    for (int64_t i = 0; i < n; ++i) {
        R_norms[i] = lapack::lantr(lapack::Norm::Fro, blas::Uplo::Upper, blas::Diag::NonUnit, n - i, n - i, &R[(ldr + 1) * i], ldr);
        if (i < 10)
            std::cout << std::scientific << R_norms[i] << "\n";
    }
}

/// Raw binding to LAPACK's dgejsv (Jacobi SVD), the reference spectrum of the pivot-quality runs.
inline void gejsv(
    char joba, char jobu, char jobv, char jobr,
    char jobt, char jobp,
    int64_t m, int64_t n,
    double *A, int64_t lda,
    double *S,
    double *U, int64_t ldu,
    double *V, int64_t ldv,
    double* work, int64_t* lwork,
    int64_t* iwork,
    int64_t* info
){

    char joba_ = joba;
    char jobu_ = jobu;
    char jobv_ = jobv;
    char jobr_ = jobr;
    char jobt_ = jobt;
    char jobp_ = jobp;

    lapack_int m_   = (lapack_int) m;
    lapack_int n_   = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldv_ = (lapack_int) ldv;

    lapack_int *lwork_ = (lapack_int *) lwork;
    lapack_int *iwork_ = (lapack_int *) iwork;
    lapack_int *info_  = (lapack_int *) info;

    LAPACK_dgejsv( & joba_, & jobu_, & jobv_, & jobr_,
        & jobt_, & jobp_,
        & m_, & n_,
        A, & lda_,
        S,
        U, & ldu_,
        V, & ldv_,
        work, lwork_,
        iwork_,
        info_
        );
}

/// Selects BQRRP's tall QR and its Q^T application by name: "geqrt", "cholqr", else geqrf.
template <typename BQRRP_t>
void set_bqrrp_qr_tall(BQRRP_t &bqrrp, const std::string &qr_tall) {
    using Subroutines = BQRRPSubroutines;
    if(qr_tall == "geqrt") {
        bqrrp.qr_tall       = Subroutines::QRTall::geqrt;
        bqrrp.apply_trans_q = Subroutines::ApplyTransQ::gemqrt;
    } else if (qr_tall == "cholqr") {
        bqrrp.qr_tall       = Subroutines::QRTall::cholqr;
        bqrrp.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
    } else {
        bqrrp.qr_tall       = Subroutines::QRTall::geqrf;
        bqrrp.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
    }
}

} // namespace bench
} // namespace RandLAPACK
