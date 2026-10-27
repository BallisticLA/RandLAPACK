/*
CQRRPT_GPU benchmark: speed of CQRRPT_GPU with each QRCP of the sketch (luqr, and bqrrp at each block size given)
against CPU CQRRPT, with A in pageable and in pinned host memory, in double and float; CQRRPT_GPU's step times;
and the pivot quality of each QRCP against CPU CQRRPT, whose QRCP of the sketch is GEQP3, as in
CQRRPT_pivot_quality.cc. With the default block size, bqrrp runs BQRRP_GPU as one block for n <= 2000, which
gives luqr's pivots; smaller blocks take it through several iterations.

Every timed run starts from the same A and RNG state. A pinned A is allocated with cudaMallocHost, outside the
timed region; CQRRPT_GPU then copies it to the device while the host sketches it.
*/
#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <RandBLAS.hh>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// The GPU drivers and kernels come in with RandLAPACK.hh, under its __CUDACC__ guard.

using RNG = r123::Philox4x32;
using QRCP = RandLAPACK::CQRRPT_GPU_Subroutines::QRCP;
using namespace std::chrono;

// One method: CPU CQRRPT, or CQRRPT_GPU with one QRCP of the sketch (and, for bqrrp, BQRRP_GPU's block size,
// 0 for the default) and A pageable or pinned.
struct Method {
    std::string name;
    bool gpu;
    QRCP qrcp;
    int64_t bqrrp_block_size;
    bool pinned;
};

// CPU CQRRPT, then luqr, then bqrrp at each block size, each with A pageable and pinned.
static std::vector<Method> make_methods(const std::vector<int64_t>& bqrrp_block_sizes) {
    std::vector<Method> methods = {
        {"CQRRPT_cpu",          false, QRCP::luqr, 0, false},
        {"CQRRPT_GPU_luqr",     true,  QRCP::luqr, 0, false},
        {"CQRRPT_GPU_luqr_pin", true,  QRCP::luqr, 0, true}};
    for (int64_t b : bqrrp_block_sizes) {
        std::string name = "CQRRPT_GPU_bqrrp_b" + (b > 0 ? std::to_string(b) : std::string("default"));
        methods.push_back({name, true, QRCP::bqrrp, b, false});
        methods.push_back({name + "_pin", true, QRCP::bqrrp, b, true});
    }
    return methods;
}

// Output of one call: Q in A's buffer, R, J, rank, the wall time and, for CQRRPT_GPU with timing on, its times.
template <typename T>
struct RunResult {
    std::vector<T> Q;
    std::vector<T> R;
    std::vector<int64_t> J;
    int64_t rank;
    long wall_us;
    std::vector<long> times;
};

// Runs one method on the m-by-n A0 from the RNG state seed_state. A_pin is a pinned buffer of m * n entries.
template <typename T>
static RunResult<T> run_method(const Method& meth, int64_t m, int64_t n, const std::vector<T>& A0, T* A_pin,
                               T d_factor, T tol, bool timing, RandBLAS::RNGState<RNG> seed_state) {
    RunResult<T> res;
    res.R.assign(n * n, (T) 0);
    res.J.assign(n, 0);
    T* A = A_pin;
    if (!meth.pinned) {
        res.Q = A0;
        A = res.Q.data();
    } else {
        std::memcpy(A_pin, A0.data(), m * n * sizeof(T));
    }
    auto state = seed_state;
    auto start = steady_clock::now();
    if (meth.gpu) {
        RandLAPACK::CQRRPT_GPU<T, RNG> alg(false, timing, tol);
        alg.qrcp = meth.qrcp;
        alg.bqrrp_block_size = meth.bqrrp_block_size;
        alg.call(m, n, A, m, res.R.data(), n, res.J.data(), d_factor, state);
        res.rank = alg.rank;
        res.times = alg.times;
    } else {
        RandLAPACK::CQRRPT<T, RNG> alg(false, tol);
        alg.call(m, n, A, m, res.R.data(), n, res.J.data(), d_factor, state);
        res.rank = alg.rank;
    }
    res.wall_us = (long) duration_cast<microseconds>(steady_clock::now() - start).count();
    if (meth.pinned)
        res.Q.assign(A_pin, A_pin + m * n);
    return res;
}

// ||A[:, J] - QR||_F / ||A||_F and ||Q'Q - I||_F / sqrt(rank) of a result.
template <typename T>
static void errors(int64_t m, int64_t n, const std::vector<T>& A0, RunResult<T>& res, T& err_qr, T& err_orth) {
    int64_t r = res.rank;
    err_qr = err_orth = -1;
    if (r <= 0)
        return;
    std::vector<T> AP(A0), G(r * r, (T) 0);
    std::vector<int64_t> J(res.J);
    RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, (T) -1, res.Q.data(), m, res.R.data(), n, (T) 1, AP.data(), m);
    err_qr = lapack::lange(Norm::Fro, m, n, AP.data(), m) / lapack::lange(Norm::Fro, m, n, A0.data(), m);
    RandLAPACK::util::eye(r, r, G.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, r, m, (T) 1, res.Q.data(), m, (T) -1, G.data(), r);
    err_orth = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, r, G.data(), r) / std::sqrt((T) r);
}

// Speed: the best wall time of num_runs calls of each method, with the errors of the last call, and the
// CQRRPT_GPU step times of a separate call with timing on. Returns false if the pinned buffer cannot be allocated.
template <typename T>
static bool bench_speed(const std::vector<Method>& methods, int64_t m, int64_t n, int64_t num_runs, T d_factor, const std::string& tname) {
    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T) 0.85);
    std::vector<T> A0(m * n, (T) 0);
    RandLAPACK::gen::mat_gen_info<T> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = sizeof(T) == 8 ? (T) 1e10 : (T) 1e4;
    m_info.rank = n;
    m_info.exponent = 2.0;
    auto gen_state = RandBLAS::RNGState<RNG>(1);
    RandLAPACK::gen::mat_gen(m_info, A0.data(), gen_state);
    T* A_pin = nullptr;
    cudaError_t ierr = cudaMallocHost((void**) &A_pin, m * n * sizeof(T));
    if (ierr != cudaSuccess) {
        std::cerr << "cudaMallocHost of " << m * n * sizeof(T) << " bytes for the pinned A failed: " << cudaGetErrorString(ierr) << "\n";
        return false;
    }
    auto seed_state = RandBLAS::RNGState<RNG>(2);

    std::string file_name = RandLAPACK::util::get_current_date_time<T>() + "_CQRRPT_GPU_speed_" + tname
                          + "_" + std::to_string(m) + "_" + std::to_string(n) + ".txt";
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    file << "Description: CQRRPT_GPU benchmark, speed. One line per method: name, best wall time (us) of " << num_runs
         << " calls, rank, ||A[:, J] - QR|| / ||A||, ||Q'Q - I|| / sqrt(rank); for CQRRPT_GPU, then the 8 step times (us) of"
            " a call with timing on: sketch, qrcp, rank_est, pivot, cholqr, trsm, rest, total."
         << "\nInput: polynomial spectrum, condition number " << m_info.cond_num << ", " << m << " by " << n
         << "\nd_factor: " << d_factor << ", precision: " << tname << "\n";
    for (const auto& meth : methods) {
        // A warm-up call keeps one-time initialization out of the timings.
        run_method<T>(meth, m, n, A0, A_pin, d_factor, tol, false, seed_state);
        long best = std::numeric_limits<long>::max();
        RunResult<T> res;
        for (int64_t i = 0; i < num_runs; ++i) {
            res = run_method<T>(meth, m, n, A0, A_pin, d_factor, tol, false, seed_state);
            best = std::min(best, res.wall_us);
        }
        T err_qr, err_orth;
        errors<T>(m, n, A0, res, err_qr, err_orth);
        file << meth.name << ", " << best << ", " << res.rank << ", " << err_qr << ", " << err_orth;
        std::cout << tname << " " << meth.name << ": best " << best << " us, rank " << res.rank << ", ||AP - QR||/||A|| = "
                  << err_qr << ", ||Q'Q - I||/sqrt(r) = " << err_orth;
        if (meth.gpu) {
            auto timed = run_method<T>(meth, m, n, A0, A_pin, d_factor, tol, true, seed_state);
            std::cout << ", steps (us):";
            for (auto t : timed.times) {
                file << ", " << t;
                std::cout << " " << t;
            }
        }
        file << "\n";
        std::cout << "\n";
    }
    cudaFreeHost(A_pin);
    std::cout << "Wrote " << file_name << "\n";
    return true;
}

// Pivot quality, in double, as in CQRRPT_pivot_quality.cc, against CPU CQRRPT (GEQP3 on the sketch):
// metric 1, ||R(i:n, i:n)||_F of each CQRRPT_GPU QRCP over that of CPU CQRRPT; metric 2, |R(i, i)| / sigma_i(A).
static void bench_pivot_quality(const std::vector<Method>& methods, int64_t m, int64_t n, double d_factor) {
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A0(m * n, 0.0), S(n, 0.0);
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::spiked);
    m_info.cond_num = std::pow(10, 10);
    m_info.rank = n;
    m_info.exponent = 2.0;
    m_info.scaling = std::pow(10, 10);
    auto gen_state = RandBLAS::RNGState<RNG>(3);
    RandLAPACK::gen::mat_gen(m_info, A0.data(), gen_state);
    std::vector<double> A_svd(A0);
    lapack::gesdd(Job::NoVec, m, n, A_svd.data(), m, S.data(), (double*) nullptr, m, (double*) nullptr, n);
    auto seed_state = RandBLAS::RNGState<RNG>(4);

    auto tail_norms = [&](const std::vector<double>& R) {
        std::vector<double> norms(n);
        for (int64_t i = 0; i < n; ++i)
            norms[i] = lapack::lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, n - i, n - i, &R[(n + 1) * i], n);
        return norms;
    };
    auto ref = run_method<double>(methods[0], m, n, A0, nullptr, d_factor, tol, false, seed_state);
    auto ref_norms = tail_norms(ref.R);

    std::string file_name = RandLAPACK::util::get_current_date_time<double>() + "_CQRRPT_GPU_pivot_quality_"
                          + std::to_string(m) + "_" + std::to_string(n) + ".txt";
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    file << "Description: CQRRPT_GPU benchmark, pivot quality. For each method, two lines: metric 1, "
            "||R(i:n, i:n)||_F over that of CPU CQRRPT (GEQP3 on the sketch), for i = 0..n-1; metric 2, |R(i, i)| / sigma_i(A)."
         << "\nInput: spiked, condition number 1e10, scaling 1e10, " << m << " by " << n << "\nd_factor: " << d_factor << "\n";
    for (const auto& meth : methods) {
        if (meth.pinned)
            continue;
        auto res = (meth.gpu) ? run_method<double>(meth, m, n, A0, nullptr, d_factor, tol, false, seed_state) : ref;
        auto norms = tail_norms(res.R);
        double worst1 = 0, worst2 = 0;
        file << meth.name << ", metric 1";
        for (int64_t i = 0; i < n; ++i) {
            double r1 = norms[i] / ref_norms[i];
            worst1 = std::max(worst1, r1);
            file << ", " << r1;
        }
        file << "\n" << meth.name << ", metric 2";
        for (int64_t i = 0; i < n; ++i) {
            double r2 = std::abs(res.R[(n + 1) * i]) / S[i];
            worst2 = std::max(worst2, std::max(r2, 1 / r2));
            file << ", " << r2;
        }
        file << "\n";
        std::cout << "pivot quality " << meth.name << ": rank " << res.rank << ", largest metric 1 " << worst1
                  << ", largest of |R(i, i)| / sigma_i and its inverse " << worst2 << "\n";
    }
    std::cout << "Wrote " << file_name << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <num_rows> <num_columns> [num_runs] [d_factor] [bqrrp_block_sizes]\n"
                  << "  Times CPU CQRRPT and CQRRPT_GPU (luqr, and bqrrp at each block size; A pageable and pinned) in\n"
                  << "  double and float, and writes the pivot quality of each against CPU CQRRPT, in double.\n"
                  << "  num_runs >= 1 (default 3); d_factor default 2.0; bqrrp_block_sizes, a comma-separated list of\n"
                  << "  BQRRP_GPU block sizes, 0 for CQRRPT's rule (n for n <= 2000), default 64,256,0. Sizes that give\n"
                  << "  BQRRP_GPU the block size of an earlier one are dropped, and so are all of them when the sketch has\n"
                  << "  more than 524,280 rows (d = d_factor * n), which CQRRPT_GPU rejects with bqrrp.\n"
                  << "  Example: " << argv[0] << " 100000 1000 5 2.0 64,256,0\n";
        return 1;
    }
    int64_t m = std::stoll(argv[1]);
    int64_t n = std::stoll(argv[2]);
    int64_t num_runs = argc > 3 ? std::stoll(argv[3]) : 3;
    double d_factor = argc > 4 ? std::stod(argv[4]) : 2.0;
    std::string sizes = argc > 5 ? argv[5] : "64,256,0";
    if (num_runs < 1) {
        std::cerr << "num_runs = " << num_runs << " must be at least 1\n";
        return 1;
    }
    // The bqrrp block sizes: nonnegative integers separated by single commas. CQRRPT_GPU clamps a size to 1..n,
    // and 0 takes CQRRPT's rule; a size that comes to the block size of an earlier one is dropped.
    std::vector<int64_t> bqrrp_block_sizes, block_sizes_used;
    for (size_t start = 0;;) {
        size_t end = sizes.find(',', start);
        std::string item = sizes.substr(start, end == std::string::npos ? std::string::npos : end - start);
        int64_t b = 0;
        bool digits = !item.empty() && item.find_first_not_of("0123456789") == std::string::npos;
        if (!digits || std::from_chars(item.data(), item.data() + item.size(), b).ec != std::errc()) {
            std::cerr << "bqrrp_block_sizes \"" << sizes << "\": \"" << item << "\" is not a nonnegative 64-bit integer\n";
            return 1;
        }
        int64_t b_used = b > 0 ? b : (n <= 2000 ? n : (n <= 8000 ? n / 2 : n / 32));
        b_used = std::max((int64_t) 1, std::min(b_used, n));
        if (std::find(block_sizes_used.begin(), block_sizes_used.end(), b_used) == block_sizes_used.end()) {
            bqrrp_block_sizes.push_back(b);
            block_sizes_used.push_back(b_used);
        } else {
            std::cout << "Dropping bqrrp block size " << b << ": BQRRP_GPU would run with block size " << b_used
                      << ", as for an earlier size\n";
        }
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    // CQRRPT_GPU rejects bqrrp when the sketch has more rows than one launch of BQRRP_GPU's gather covers. It
    // computes d = d_factor * n in the precision of the call, so take the larger of the double and float values.
    int64_t d = std::max((int64_t) (d_factor * n), (int64_t) ((float) d_factor * n));
    if (d > 8 * 65535 && !bqrrp_block_sizes.empty()) {
        std::cout << "Skipping the bqrrp methods: the sketch has d = d_factor * n = " << d
                  << " rows, more than the 524,280 that CQRRPT_GPU takes with bqrrp\n";
        bqrrp_block_sizes.clear();
    }
    auto methods = make_methods(bqrrp_block_sizes);
    if (!bench_speed<double>(methods, m, n, num_runs, d_factor, "double") || !bench_speed<float>(methods, m, n, num_runs, (float) d_factor, "float"))
        return 1;
    bench_pivot_quality(methods, m, n, d_factor);
    return 0;
}
