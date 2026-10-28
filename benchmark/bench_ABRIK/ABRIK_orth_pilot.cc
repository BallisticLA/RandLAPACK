/*
ABRIK orthogonalization pilot: how the tall-and-skinny QR candidates for BK's block
factorization lose orthogonality as the block's condition number grows.

Each input is an l-by-b block with singular values geometrically spaced from 1 down to 1/kappa,
a graded stand-in for the conditioning BK's block QR sees once a block is nearly dependent on
the accumulated basis. The quantities are numerical, not performance, so the pilot is meaningful on
any machine; it records no timings.

Candidates:
  hqr        Householder QR (GEQRF followed by ORGQR), BK's default (QR_explicit::geqrf_ungqr).
  cholqr     Single-pass CholeskyQR.
  cholqr2    CholeskyQR applied twice.
  cqrrt      Sketch-preconditioned CholeskyQR, BK's alternative (QR_explicit::cqrrt).

Output CSV: '#' metadata lines, the column header
  run, candidate, b_sz, kappa, orth_err, res_err, status
then one row per (run, block width, kappa, candidate):

  orth_err = ||Q'Q - I||_F
  res_err  = ||A - Q (Q'A)||_F / ||A||_F, with Q'A standing in for each candidate's own R
             (HQRQ and CholQRQ return Q only, so all four are scored alike)
  status   = the candidate's return code; nonzero is a breakdown, recorded with both
             metrics at -1

Usage:
  ABRIK_orth_pilot <precision> <output_dir> <l> <num_runs> <num_block_sizes> <b...>
                   <num_conds> <log10(kappa)...>

  precision     = double | float
  l             = rows of every block
  num_runs      = repetitions of the whole sweep; the generator state carries over
  b             = block widths, each in [1, l]
  log10(kappa)  = condition numbers of the blocks, as powers of ten

Example (block widths 8 and 32, kappa from 1e2 to 1e12):
  ABRIK_orth_pilot double . 20000 1  2 8 32  6 2 4 6 8 10 12
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_bench_utils.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static const char* kUsage =
    "<precision> <output_dir> <l> <num_runs> <num_block_sizes> <b...> <num_conds> <log10(kappa)...>";

// An l-by-b block with singular values 1 = sigma_1 >= ... >= sigma_b = 1/kappa, geometric.
template <typename T, typename RNG>
static void gen_block(int64_t l, int64_t b, T kappa, T* A, RandBLAS::RNGState<RNG> &state) {
    std::vector<T> sigma(b), S(b * b, (T) 0);
    for (int64_t j = 0; j < b; ++j)
        sigma[j] = (b == 1) ? (T) 1 : std::pow(kappa, -(T) j / (T) (b - 1));
    RandLAPACK::util::diag(b, b, sigma.data(), b, S.data());
    RandLAPACK::gen::gen_singvec<T>(l, b, A, b, S.data(), state);
}

// Runs one candidate on a copy of A. On success writes ||Q'Q - I||_F and the relative
// residual of Q (Q'A); returns the candidate's status.
template <typename T, typename RNG>
static int run_candidate(const std::string &name, int64_t l, int64_t b, const T* A, T* Q,
                         T* R, RandBLAS::RNGState<RNG> &state, T &orth, T &res) {
    std::copy(A, A + l * b, Q);
    int status = 0;
    if (name == "hqr") {
        status = RandLAPACK::HQRQ<T>(false, false).call(l, b, Q);
    } else if (name == "cholqr" || name == "cholqr2") {
        RandLAPACK::CholQRQ<T> chol(false, false);
        status = chol.call(l, b, Q);
        if (status == 0 && name == "cholqr2")
            status = chol.call(l, b, Q);
    } else if (name == "cqrrt") {
        RandLAPACK::CQRRT<T, RNG> cqrrt(false, std::numeric_limits<T>::epsilon());
        std::fill(R, R + b * b, (T) 0);
        status = cqrrt.call(l, b, Q, l, R, b, (T) 1.25, state);
    } else {
        throw std::invalid_argument("unknown candidate '" + name + "'");
    }
    if (status != 0) {
        orth = -1;
        res  = -1;
        return status;
    }
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, b, b, l, (T) 1, Q, l, A, l, (T) 0, R, b);
    // verify_qr reports ||Q'Q - I||_F / sqrt(b); the CSV keeps the unnormalized norm.
    auto [fact_err, orth_normalized] = RandLAPACK::testing::verify_qr<T>(A, Q, R, l, b, b);
    orth = orth_normalized * std::sqrt((T) b);
    res  = fact_err;
    return 0;
}

template <typename T>
static int run_pilot(int argc, char* argv[]) {
    if (argc < 7) return RandLAPACK::bench::usage_error(argv[0], kUsage);

    std::string output_dir = argv[2];
    int64_t l              = std::stoll(argv[3]);
    int     num_runs       = std::stoi(argv[4]);
    int     num_b          = std::stoi(argv[5]);
    if (l < 1 || num_runs < 1 || num_b < 1 || argc < 7 + num_b) {
        std::cerr << "Error: l, num_runs and num_block_sizes must be >= 1, with every block width given\n";
        return RandLAPACK::bench::usage_error(argv[0], kUsage);
    }
    std::vector<int64_t> blocks;
    for (int i = 0; i < num_b; ++i)
        blocks.push_back(std::stoll(argv[6 + i]));
    int num_conds = std::stoi(argv[6 + num_b]);
    if (num_conds < 1 || argc != 7 + num_b + num_conds) {
        std::cerr << "Error: num_conds must be >= 1, with exactly that many log10(kappa) values after it\n";
        return RandLAPACK::bench::usage_error(argv[0], kUsage);
    }
    std::vector<T> kappas;
    for (int i = 0; i < num_conds; ++i)
        kappas.push_back(std::pow((T) 10, (T) std::stod(argv[7 + num_b + i])));
    if (*std::min_element(blocks.begin(), blocks.end()) < 1
        || *std::max_element(blocks.begin(), blocks.end()) > l
        || *std::min_element(kappas.begin(), kappas.end()) < (T) 1) {
        std::cerr << "Error: block widths must be in [1, l] and log10(kappa) >= 0\n";
        return 2;
    }

    std::ofstream file;
    std::string path = RandLAPACK::bench::open_csv(output_dir, "ABRIK_orth_pilot", file);
    if (!file) return 1;
    file << "# ABRIK orthogonalization pilot\n"
         << "# RANDLAPACK_GIT_COMMIT=" << RandLAPACK::bench::build_commit() << "\n"
         << "# Precision: " << argv[1] << "\n"
         << "# rows: " << l << "\n"
         << "# runs: " << num_runs << "\n"
         << "# orth_err = ||Q'Q - I||_F; res_err = ||A - Q (Q'A)||_F / ||A||_F; -1 when status != 0\n"
         << "run, candidate, b_sz, kappa, orth_err, res_err, status\n";

    const std::vector<std::string> candidates = {"hqr", "cholqr", "cholqr2", "cqrrt"};
    RandBLAS::RNGState<r123::Philox4x32> state(0);

    for (int run = 0; run < num_runs; ++run) {
        for (int64_t b : blocks) {
            std::vector<T> A(l * b), Q(l * b), R(b * b);
            for (T kappa : kappas) {
                gen_block<T>(l, b, kappa, A.data(), state);
                for (const std::string &cand : candidates) {
                    T orth = 0, res = 0;
                    int status = run_candidate<T>(cand, l, b, A.data(), Q.data(), R.data(), state, orth, res);
                    file << run << ", " << cand << ", " << b << ", "
                         << std::scientific << std::setprecision(6) << kappa << ", "
                         << orth << ", " << res << ", " << status << "\n";
                    printf("run %d b %ld kappa %.1e %-8s orth %.3e res %.3e status %d\n",
                           run, (long) b, (double) kappa, cand.c_str(), (double) orth, (double) res, status);
                }
                file.flush();
            }
        }
    }

    file.close();
    if (file.fail()) {
        std::cerr << "Error: writing " << path << " failed\n";
        return 1;
    }
    printf("Results written to: %s\n", path.c_str());
    return 0;
}

int main(int argc, char* argv[]) {
    return RandLAPACK::bench::precision_main(argc, argv, kUsage, run_pilot<double>, run_pilot<float>);
}
