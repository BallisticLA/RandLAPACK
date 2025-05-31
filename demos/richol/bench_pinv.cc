

#define FINE_GRAINED
#include "richol_core.hh"
#include "richol_mkl.hh"

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


double run_nys_approx(
    int k, std::vector<double> &V, std::vector<double> &eigvals,
    LaplacianPinv &Lpinv,
    RandLAPACK::REVD2<double, DefaultRNG> &NystromAlg) {
    int64_t n = Lpinv.m;
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
std::vector<double> richol_pipeline(CSRMatrix<T,int64_t> &L,  sparse_matrix_t &Lperm_mkl, sparse_matrix_t &G_mkl, bool use_amd_perm) {
    int64_t n = L.n_rows;
    std::vector<double> out{};
    timepoint_t tp0, tp1;
    std::vector<int64_t> perm(n);
    for (int64_t i = 0; i < n; ++i)
        perm[i] = i;
    CSRMatrix<T, int64_t> Lperm(n, n);
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
    richol::sym_as_upper_tri_from_csr(Lperm.n_rows, Lperm.rowptr, Lperm.colidxs, Lperm.vals, sym);
    rank = richol::clb21_rand_cholesky(sym, C, state, false, (T)0.0), "SparseCholesky: ");
    std::cout << "Exited with C of rank k = " << rank << std::endl;
    int64_t nnz_G = 0;
    for (const auto &row : C) 
        nnz_G += static_cast<int64_t>(row.size());
    CSRMatrix<T,int64_t> G(n, n);
    reserve_csr(nnz_G ,G);
    csr_from_csrlike(C, G.rowptr, G.colidxs, G.vals);

    sparse_matrix_t_from_randblas_csr(Lperm, Lperm_mkl);
    sparse_matrix_t_from_randblas_csr(G, G_mkl);
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
    bool use_amd_perm = true;
    int64_t k = 8;
    int64_t syps_passes = 3;
    RandLAPACK::SYPS<T, DefaultRNG>  SYPS(syps_passes, 1, false, false);
    RandLAPACK::HQRQ<T>              Orth(false, false); 
    RandLAPACK::SYRF<T, DefaultRNG>  SYRF(SYPS, Orth, false, false);
    RandLAPACK::REVD2<T, DefaultRNG> NystromAlg(SYRF, 1, false);

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
    logfilename << "/home/rjmurr/laps2/RandLAPACK/demos/richol/EY_logs/round2";
    logfilename << "sG2tosG10_k_" << k << "_threads_" << threads << "_amd_" << use_amd_perm << "_sypspasses_" << syps_passes << ".txt";
    std::ofstream logfile(logfilename.str());
    logfile << "n         , m         , perm_time , chol_time , nnz_pre   , spmm_time , trsm_time , pcg_time  , pcg_iters , nys_time\n";
    logfile.flush();

    for (auto fn : filenames) {
        auto L = richol::laplacian_from_matrix_market(fn, (T)0.0);
        sparse_matrix_t Lperm_mkl, G_mkl;
        int64_t n = L.n_rows;
        int64_t m = (L.nnz - n) / 2;
        logfile << std::left << std::setw(10) << n << ", ";
        logfile << std::left << std::setw(10) << m << ", ";
        auto factimes = richol_pipeline(L, Lperm_mkl, G_mkl, use_amd_perm);
        logfile << std::left << std::setw(10) << factimes[0] << ", ";
        logfile << std::left << std::setw(10) << factimes[1] << ", ";
        logfile << std::left << std::setw(10) << (int64_t) factimes[2] << ", ";
        logfile.flush();
        CallableSpMat Aperm_callable{Lperm_mkl, n};
        CallableChoSolve N_callable{G_mkl, n};
        LaplacianPinv Lpinv(Aperm_callable, N_callable, 1e-8, 200, true);
        std::vector<T> V(n*k, 0.0);
        std::vector<T> eigvals(k, 0.0);
        double nys_time = run_nys_approx(k, V, eigvals, Lpinv, NystromAlg);
        logfile << std::left << std::setw(10) << Lpinv.times[0] << ", ";
        logfile << std::left << std::setw(10) << Lpinv.times[1] << ", ";
        logfile << std::left << std::setw(10) << Lpinv.times[2] << ", ";
        logfile << std::left << std::setw(10) << (int64_t) Lpinv.times[3] << ", ";
        logfile << std::left << std::setw(10) << nys_time;
        mkl_sparse_destroy(Lperm_mkl);
        mkl_sparse_destroy(G_mkl);
        logfile << "\n";
        logfile.flush();
    }

    return 0;
}
