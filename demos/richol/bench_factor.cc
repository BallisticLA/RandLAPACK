

#define FINE_GRAINED
#include "richol_core.hh"
#include "richol_linops.hh"

#include <iomanip>
#include <iostream>
#include <chrono>

using RandBLAS::CSRMatrix;
using RandBLAS::sparse_data::reserve_csr;
using RandBLAS::sparse_data::reserve_coo;
using RandBLAS::CSCMatrix;
using RandBLAS::COOMatrix;
using RandBLAS::sparse_data::conversions::coo_to_csr;

using namespace richol::linops;
using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;


auto parse_args(int argc, char** argv) {
    // std::string mat{"/home/rjmurr/laps/RandLAPACK/demos/sparse-data-matrices/chesapeake/chesapeake.mtx"};
    // int threads = 1;
    // if (argc > 1)
    //     threads = atoi(argv[1]);
    // if (argc > 2)
    //     mat = argv[2];
    // return std::make_tuple(threads, mat);
    int threads = 1;
    if (argc > 1)
        threads = atoi(argv[1]);
    return threads;
}


template <typename LPINV_t>
double run_nys_approx(
    int k, std::vector<double> &V, std::vector<double> &eigvals,
    LPINV_t &Lpinv,
    RandLAPACK::REVD2<
        RandLAPACK::SYRF<
            RandLAPACK::SYPS<double, DefaultRNG>,
            RandLAPACK::HQRQ<double>
        >
    > &NystromAlg
) {
    int64_t n = Lpinv.dim;
    V.resize(n*k); eigvals.resize(k);
    for (int64_t i = 0; i < n*k; ++i)
        V[i] = 0.0;
    for (int64_t i = 0; i < k; ++i)
        eigvals[i] = 0.0;

    int64_t k_ = k;
    auto _tp0 = std_clock::now();
    double dummy_tol = 1e10;
    RandBLAS::RNGState state(8675309);
    NystromAlg.call(Lpinv, k_, dummy_tol, V, eigvals, state);
    auto _tp1 = std_clock::now();
    double dtime = seconds_elapsed(_tp0, _tp1);
    return dtime;
}


template <typename T>
std::vector<double> richol_pipeline(CSRMatrix<T,int64_t> &L,  CSRMatrix<T,int64_t> &Lperm, CSRMatrix<T,int64_t> &G, bool use_amd_perm) {
    int64_t n = L.n_rows;
    std::vector<double> out{};
    timepoint_t tp0, tp1;
    std::vector<int64_t> perm(n);
    for (int64_t i = 0; i < n; ++i)
        perm[i] = i;
    tp0 = std_clock::now();
    TIMED_LINE(
    if (use_amd_perm) richol::amd_permutation(L, perm);
    richol::permuted(L, perm, Lperm);, "AMD reordering      : ");
    tp1 = std_clock::now();
    out.push_back(seconds_elapsed(tp0, tp1));

    using spvec = richol::SparseVec<T, int64_t>;
    std::vector<spvec> sym;
    std::vector<spvec> C;
    RandBLAS::RNGState state(0);
    int64_t rank;
    tp0 = std_clock::now();
    TIMED_LINE(
    richol::csrlike_from_csr(Lperm.n_rows, Lperm.rowptr, Lperm.colidxs, Lperm.vals, sym, blas::Uplo::Upper);,
    "convert to sym-as-upper-tri  : ");
    TIMED_LINE(
    rank = richol::clb21_rand_cholesky(sym, C, state, false, (T)0.0), "SparseCholesky: ");
    std::cout << "Exited with C of rank k = " << rank << std::endl;
    int64_t nnz_G = 0;
    for (const auto &row : C) nnz_G += static_cast<int64_t>(row.size());
    reserve_csr(nnz_G, G);
    TIMED_LINE(
    csr_from_csrlike(C, G.rowptr, G.colidxs, G.vals), "From csrlike to CSR    : ");
    tp1 = std_clock::now();
    out.push_back(seconds_elapsed(tp0, tp1));
    out.push_back((double) nnz_G);
    return out;
}


int main(int argc, char** argv) {
    using T = double;
    std::cout << std::setprecision(8);
    auto threads = parse_args(argc, argv);
    omp_set_num_threads(threads);
    bool use_amd_perm = false;

    std::vector<std::string> filenames = {
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/G2/sG2.mtx",
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/sG3.mtx",
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/sG4.mtx",
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/sG5.mtx",
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/sG6.mtx",
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/sG7.mtx",
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/sG8.mtx",
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/sG9.mtx",
        "/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/sG10.mtx"
    };

    std::stringstream logfilename;
    logfilename << "/home/rjmurr/laps2/RLPside/demos/richol/EY_logs/";
    logfilename << "sG2tosG10" << "_threads_" << threads << "_amd_" << use_amd_perm << ".txt";
    std::ofstream logfile(logfilename.str());
    logfile << "n, m, reorder_time, sparse_chol_time, nnz_pre\n";
    logfile.flush();

    for (auto fn : filenames) {
        auto L = richol::laplacian_from_matrix_market(fn, (T)0.0);
        int64_t n = L.n_rows;
        int64_t m = (L.nnz - n) / 2;
        CSRMatrix<T> Lperm(n, n);
        CSRMatrix<T> G(n, n);
        logfile << std::left << std::setw(10) << n << ", ";
        logfile << std::left << std::setw(10) << m << ", ";
        auto factimes = richol_pipeline(L, Lperm, G, use_amd_perm);
        logfile << std::left << std::setw(10) << factimes[0] << ", ";
        logfile << std::left << std::setw(10) << factimes[1] << ", ";
        logfile << std::left << std::setw(10) << (int64_t) factimes[2] << ", ";
        logfile.flush();
        logfile << "\n";
        logfile.flush();
    }

    return 0;
}
