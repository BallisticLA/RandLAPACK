#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <omp.h>
#include <iomanip>
#include <filesystem>
#include <algorithm>

#include "sparse.hpp"
#include "rchol.hpp"
#include "rchol_parallel.hpp"
#include "util.hpp"
#include "pcg.hpp"
// ^ includes mkl_spblas.h

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

using RandLAPACK::linops::SymmetricLinearOperator;
using RandBLAS::DefaultRNG;
using RandBLAS::CSRMatrix;
using RandBLAS::CSCMatrix;
using RandBLAS::COOMatrix;

using RandBLAS::sparse_data::right_spmm;
using RandBLAS::sparse_data::left_spmm;

using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;

#define DOUT(_d) std::setprecision(8) << _d
#define FINE_GRAINED
#ifdef FINE_GRAINED
#define TIMED_LINE(_op, _name) { \
        auto _tp0 = std_clock::now(); \
        _op; \
        auto _tp1 = std_clock::now(); \
        double dtime = (double) duration_cast<microseconds>(_tp1 - _tp0).count(); \
        std::cout << _name << DOUT(dtime / 1e6) << std::endl; \
        }
#else
#define TIMED_LINE(_op, _name) _op;
#endif

#define SparseCSR_RC SparseCSR

// /* status of the routines */
// typedef enum
// {
//     SPARSE_STATUS_SUCCESS           = 0,    /* the operation was successful */
//     SPARSE_STATUS_NOT_INITIALIZED   = 1,    /* empty handle or matrix arrays */
//     SPARSE_STATUS_ALLOC_FAILED      = 2,    /* internal error: memory allocation failed */
//     SPARSE_STATUS_INVALID_VALUE     = 3,    /* invalid input value */
//     SPARSE_STATUS_EXECUTION_FAILED  = 4,    /* e.g. 0-diagonal element for triangular solver, etc. */
//     SPARSE_STATUS_INTERNAL_ERROR    = 5,    /* internal error */
//     SPARSE_STATUS_NOT_SUPPORTED     = 6     /* e.g. operation for double precision doesn't support other types */
// } sparse_status_t;


void sparse_matrix_t_from_SparseCSR_RC(const SparseCSR_RC &A, sparse_matrix_t &mat) {
    auto N = A.N;
    auto cpt = A.rowPtr;
    auto rpt = A.colIdx;
    auto datapt = A.val;
    size_t *pointerB = new size_t[N + 1]();
    size_t *pointerE = new size_t[N + 1]();
    size_t *update_intend_rpt = new size_t[cpt[N]]();
    double *update_intend_datapt = new double[cpt[N]]();
    for(size_t i = 0; i < N; i++) {
        size_t start = cpt[i];
        size_t last = cpt[i + 1];
        pointerB[i] = start;
        for (size_t j = start; j < last; j++) {
            update_intend_datapt[j] = datapt[j];
            update_intend_rpt[j] = rpt[j];
        }
        pointerE[i] = last;    
    }
    // auto status = 
    mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, N, N, pointerB, pointerE, update_intend_rpt, update_intend_datapt);
    // The AddressSanitizer says that this function has a memory leak.
    // An obvious potential cause is that mkl_sparse_d_create_csr might not take ownership of the buffers we pass it.
    //
    // However, we get a segfault later on if we delete pointerB, pointerE, update_intend_rpt, and update_intend_datapt before returning.
    //
    // See here for a potentially useful thread:
    //   community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/mkl-sparse-d-create-csr-possibility-of-memory-leak/m-p/1313882#M32031
    return;
}


struct CallableSpMat {
    sparse_matrix_t A;
    int64_t m;
    int64_t nnz;
    double* work = nullptr;
    std::vector<double> work_stdvec{};
    matrix_descr des{SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    // ^ The latter two entries of des are not actually used. I'm only specifying them 
    //   to avoid compiler warings.
    std::vector<double> regs{0.0};

    /*  C =: alpha * A * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta,  double* C, int64_t ldc
    ) {
        work_stdvec.resize(m*n);
        work = work_stdvec.data();
        randblas_require(ldb == m);
        blas::copy(m*n, B, 1, work, 1);
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        omp_set_dynamic(1);
        //TIMED_LINE(
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, des, mkl_layout, work, n, ldb, beta, C, ldc
        );//, "SPMM A   : ");
        omp_set_dynamic(0);
    }
};

struct CallableChoSolve {
    sparse_matrix_t G;
    int64_t m;
    matrix_descr des = {SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    double* work = nullptr;
    std::vector<double> work_stdvec{};

    /*  C =: alpha * inv(G G') * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta, double* C, int64_t ldc
    ) {
        work_stdvec.resize(2*m*n);
        work = work_stdvec.data();    
        randblas_require(beta == (double) 0.0);
        blas::copy(m*n, B, 1, work+m*n, 1);
        randblas_require(ldb == m);
        // TRSM, then transposed TRSM.
        sparse_status_t status;
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        omp_set_dynamic(1);
        //TIMED_LINE(
        status = mkl_sparse_d_trsm(SPARSE_OPERATION_TRANSPOSE    , alpha, G, des, mkl_layout, work+m*n, n, ldb, work, m); //, "TRSM G   : ");
        if (status != SPARSE_STATUS_SUCCESS) {
            std::cout << "TRSM failed with error code " << status << std::endl;
            throw std::runtime_error("TRSM failure.");
        }
        //TIMED_LINE(
        status = mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE,   1.0, G, des, mkl_layout, work, n, m, C, ldc); //, "TRSM G^T : ");
        if (status != SPARSE_STATUS_SUCCESS) {
            std::cout << "TRSM failed with error code " << status << std::endl;
            throw std::runtime_error("TRSM failure.");
        }
        omp_set_dynamic(0);
    }
};

enum PCGMode : char {
    RChol    = 'R',
    Lockstep = 'L',
    Block    = 'B'
};

// NOTE: below probably needs to conform to the SymmetricLinearOperator API.
struct LaplacianPinv : public SymmetricLinearOperator<double> {
    public:
    // inherited --> const int64_t m;
    CallableSpMat    L_callable;
    CallableChoSolve N_callable;
    bool verbose_pcg;
    pcg rchol_pcg_object;
    std::vector<double> work_B{};
    std::vector<double> work_C{};
    std::vector<double> work_seminorm{};
    std::vector<double> rchol_pcg_work{};
    double call_pcg_tol = 1e-10;
    int64_t max_iters = 100;
    PCGMode mode = PCGMode::Block;
    bool verbose_outer_iters = false;

    LaplacianPinv(CallableSpMat &L, CallableChoSolve &N,
        std::vector<int> &S, int num_threads, double pcg_tol, int maxit, SparseCSR_RC &G_rchol,
        bool verbose = false
    ) :
        SymmetricLinearOperator<double>(L.m),
        L_callable{L.A, L.m, L.nnz},
        N_callable{N.G, N.m},
        verbose_pcg(verbose),
        rchol_pcg_object(L.A, L.m, S, num_threads, pcg_tol, maxit, G_rchol),
        call_pcg_tol(pcg_tol),
        max_iters((int64_t) maxit)
    {
        mkl_sparse_optimize(L.A);
        mkl_sparse_optimize(N.G);
    }; 

    void prep(int64_t n) {
        int64_t mn = m*n;
        work_B.resize(mn);
        work_C.resize(mn);
        rchol_pcg_work.resize(4*m);
        L_callable.work_stdvec.resize(mn);
        N_callable.work_stdvec.resize(2*mn);
    }

    void _rchol_pcg_solve_with_work(int64_t n) {
        rchol_pcg_work.resize(4*m);
        double *rchol_pcg_work_ = rchol_pcg_work.data();
        if (verbose_outer_iters) {
            std::cout << std::left << std::setw(10) << "index"  << std::setw(10) << "iters"  << std::setw(15) << "relres" << std::endl;
        }
        for (int64_t i = 0; i < n; ++i) {
            int iter = 0;
            double relres = 0.0;
            double* curr_b = work_B.data() + m*i;
            double* curr_c = work_C.data() + m*i;
            rchol_pcg_object.iteration(&L_callable.A, curr_b, &N_callable.G, curr_c, relres, iter, rchol_pcg_work_);
            if (verbose_outer_iters) {
                std::cout << std::left << std::setw(10) << i << std::setw(10) << iter << std::setw(15) << relres << std::endl;
            }
        }
        return;
    }

    void _lockorblock_solve_with_work(int64_t n) {
        std::vector<double> sn_log_R{};
        std::vector<double> sn_log_NR{};
        int64_t count = 0;
        auto seminorm = [&count, &sn_log_R, &sn_log_NR](int64_t __n, int64_t __s, double* NR) {
            double out = blas::nrm2(__n*__s, NR, 1);
            if (count % 2 == 0) {
                sn_log_R.push_back(out);
            } else {
                sn_log_NR.push_back(out);
            }
            count++;
            return out;
        };
        // logging
        if (verbose_outer_iters) {
            std::cout << std::left << std::setw(10) << "iters" << std::setw(15) << "relres" << std::endl;
        }
        // work
        RandBLAS::util::safe_scal(m*n, 0.0, work_C.data(), 1);
        RandLAPACK::lockorblock_pcg(L_callable, work_B, call_pcg_tol, max_iters, N_callable, seminorm, work_C, verbose_pcg);
        // logging
        double denominator;
        if (sn_log_NR[0] > 1e3 * sn_log_R[0]) {
            denominator = sn_log_R[0];
        } else {
            denominator = sn_log_NR[0];
        }
        int64_t numiter = sn_log_NR.size() - 1;
        double relres =  sn_log_NR.back() / denominator;
        if (verbose_outer_iters) {
            std::cout << std::left  << std::setw(10) << numiter << std::setw(15) << relres << std::endl;
        }
    }

    /*  C =: alpha * pinv(L) * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, double* const B, int64_t ldb,
        double beta, double* C, int64_t ldc
    ) {
        randblas_require(layout == Layout::ColMajor);
        randblas_require(beta == (double) 0.0);
        randblas_require(ldb == m);
        randblas_require(ldc == m);
        int64_t mn = m*n;
        work_B.resize(mn);
        work_C.resize(mn);

        for (int64_t i = 0; i < mn; ++i)
            work_B[i] = alpha * B[i];

        if (mode == PCGMode::RChol)  {
            _rchol_pcg_solve_with_work(n);
        } else if (mode == PCGMode::Lockstep) {
            L_callable.regs.resize(n, 0.0);
            _lockorblock_solve_with_work(n);
        } else {
            L_callable.regs.resize(1, 0.0);
            _lockorblock_solve_with_work(n);
        }
        blas::copy(mn, work_C.data(), 1, C, 1);
        return;
    }

    double operator()(int64_t i, int64_t j) {
        UNUSED(i); UNUSED(j);
        randblas_require(false);
        return 0.0;
    }
};

void laplacian_from_matrix_market(std::string fn, SparseCSR_RC &A, double reg) {
    int64_t n, n_ = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<double> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n, n_, rows,  cols, vals
    );
    randblas_require(n == n_); // we must be square.

    // Convert adjacency matrix to COO format Laplacian
    int64_t m = vals.size();
    int64_t nnz = m + n;
    COOMatrix<double> coo(n, n);
    RandBLAS::sparse_data::reserve_coo(nnz, coo);
    std::vector<double> diagvec(n, reg);
    auto diag = diagvec.data();
    for (int64_t i = 0; i < m; ++i) {
        coo.rows[i] = rows[i];
        coo.cols[i] = cols[i];
        double v = vals[i];
        randblas_require(v >= 0);
        coo.vals[i] = -v;
        diag[rows[i]] += v;
    }
    for (int64_t i = 0; i < n; ++i) {
        coo.vals[m+i] = diag[i];
        coo.rows[m+i] = i;
        coo.cols[m+i] = i;
    }
    // convert COO format Laplacian to CSR format, using RandBLAS.
    CSRMatrix<double> csr(n, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);

    // Convert RandBLAS::CSRMatrix to SparseCSR_RC
    std::vector<size_t> rowPtrA(csr.n_rows+1);
    std::vector<size_t> colIdxA(csr.nnz);
    for (int64_t i = 0; i < csr.n_rows + 1; ++i)
        rowPtrA[i] = static_cast<size_t>(csr.rowptr[i]);
    rowPtrA[csr.n_rows] = csr.nnz;
    vals.resize(csr.nnz);
    for (int64_t i = 0; i < csr.nnz; ++i) {
        colIdxA[i] = static_cast<size_t>(csr.colidxs[i]);
        vals[i] = csr.vals[i];
    }

    A.init(rowPtrA, colIdxA, vals, true);
    return;
}


auto parse_args(int argc, char** argv) {
    std::string mat{"/home/rjmurr/laps/RandLAPACK/demos/sparse-data-matrices/chesapeake/chesapeake.mtx"};
    int threads = 1;
    if (argc > 1)
        threads = atoi(argv[1]);
    if (argc > 2)
        mat = argv[2];
    return std::make_tuple(threads, mat);
}

double run_nys_approx(
    int k, std::vector<double> &V, std::vector<double> &eigvals,
    LaplacianPinv &Lpinv,
    RandLAPACK::REVD2<double, DefaultRNG> &NystromAlg
) {
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
    double dtime = (double) duration_cast<microseconds>(_tp1 - _tp0).count() / 1e6;
    return dtime;
}


int main(int argc, char *argv[]) {
    std::cout << std::setprecision(16);


    auto [threads, fn] = parse_args(argc, argv);  
    double reg = 1e-4;
    SparseCSR_RC G, L_reg, Lreg_perm;

    laplacian_from_matrix_market(fn, L_reg, reg);
    int64_t n = L_reg.size();
    std::vector<size_t> P;
    std::vector<int> S;
    std::string filename = "orders/order_n" + std::to_string(n) + "_t" + std::to_string(threads) + ".txt";

    // std::cout << "n = " << n << "  # L is n-by-n\n";
    // std::cout << "m = " << (Lpinv.L_callable.nnz - n)/2 << "  # L has 2m+n nonzeros\n";

    auto _tp0 = std_clock::now();
    rchol(L_reg, G, P, S, threads, filename);
    reorder(L_reg, P, Lreg_perm);
    auto _tp1 = std_clock::now();
    double dtime = (double) duration_cast<microseconds>(_tp1 - _tp0).count() / 1e6;
    std::cout << "\n====================================================================================\n";
    std::cout << "Running with OPENMP_NUM_THREADS = " <<  omp_get_max_threads() << "\n\n";
    std::cout << "Graph information\n";
    std::cout << "  Adjacency matrix file : " << fn << "\n";
    std::cout << "  Number of nodes, n : " << n << "\n";
    std::cout << "  Number of edges, m : " << ( (int64_t) L_reg.nnz() - n) / 2 << "\n";
    std::cout << "  The Laplacian, L, has 2m + n = " << L_reg.nnz() << " nonzeros\n\n"; 
    std::cout << "SparseCholesky information\n";
    std::cout << "  Number of parallel threads : " << threads << "\n";
    std::cout << "  Time to compute factorization : " << dtime << "\n";
    std::cout << "  Number of nonzeros : " << G.nnz() << "\n\n";
    std::cout << "Parameters used in Nystrom approximation of the Laplacian pseudo-inverse, L^+\n";
    std::cout << "  k = approximation rank. We require k > 1.\n";
    std::cout << "  p = total number of black-box (matrix-matrix product) accesses of L^+.\n\n";
    std::cout << "Nystrom running in shift-invert mode with regularization parameter = " << reg << ".\n";
    std::cout << "Logging results in rows of comma-separated values: k, p, seconds in Nystrom.\n\n";

    sparse_matrix_t Lreg_perm_mkl, G_mkl;
    sparse_matrix_t_from_SparseCSR_RC(Lreg_perm, Lreg_perm_mkl);
    sparse_matrix_t_from_SparseCSR_RC(G, G_mkl);
    CallableSpMat Lreg_perm_callable{Lreg_perm_mkl, n, (int64_t) L_reg.nnz()};
    CallableChoSolve N_callable{G_mkl, n};
    LaplacianPinv Lpinv(Lreg_perm_callable, N_callable, S, threads, 1e-10, 200, G, false);   
    
    // low-rank approx time!
    RandLAPACK::SYPS<double, DefaultRNG>  SYPS(3, 1, false, false);
    RandLAPACK::HQRQ<double>              Orth(false, false); 
    RandLAPACK::SYRF<double, DefaultRNG>  SYRF(SYPS, Orth, false, false);
    RandLAPACK::REVD2<double, DefaultRNG> NystromAlg(SYRF, 1, false);

    std::vector<double> V{};
    std::vector<double> eigvals{};

    std::filesystem::path path(fn);
    std::stringstream ss0;
    ss0 << path.parent_path();
    std::string matrix_folder{ss0.str()};
    matrix_folder.erase(
        std::remove( matrix_folder.begin(), matrix_folder.end(), '\"' ),
        matrix_folder.end()
    );

    std::cout << "*******\n"; 
    std::cout << matrix_folder << std::endl;
    std::cout << "*******\n"; 


    Lpinv.mode = PCGMode::Block;
    int64_t num_reps = 3;
    std::vector<int64_t> ps{1,2,3,4,5};
    std::vector<int64_t> ks{4,16};
    for (auto k : ks) {
        Lpinv.prep(k);
        for (auto p : ps) {
            for (int64_t r = 0; r < num_reps; ++r) {
                SYPS.passes_over_data = p;
                double dt_iter = run_nys_approx(k, V, eigvals, Lpinv, NystromAlg);
                std::cout << std::setw(4) << k << ",  " << p << ",  " << dt_iter << ",\n";
            }
            // write the output ...
            std::stringstream ss;
            ss << matrix_folder << "/V_" << k << "_" << p << ".csv";
            std::string temp = ss.str();
            std::ofstream file_stream(temp);
            file_stream << std::setprecision(16);
            for (int64_t i = 0; i < n; ++i) {
                file_stream << P[i] << ", ";
                int64_t j;
                for (j = 0; j < k-1; ++j) {
                    file_stream << V[i + n*j] << ", ";
                }
                file_stream << V[i + n*j] << "\n";
            }
        }
    }
    std::cout << "====================================================================================\n\n";

    mkl_sparse_destroy(Lreg_perm_mkl);
    mkl_sparse_destroy(G_mkl);

    return 0;
}
