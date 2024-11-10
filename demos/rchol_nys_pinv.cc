#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <omp.h>
#include <iomanip>

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
    // this implementation is lifted from /home/rjmurr/laps/rchol-repo/c++/util/pcg.cpp
    //
    // Expected mode of calling:
    //
    //      sparse_matrix_t mat;
    //      sparse_matrix_t_from_SparseCSR_RC(A, mat);
    //      /* do stuff */
    //      mkl_sparse_destroy(mat);
    //
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

void project_out_vec(int64_t m, int64_t n, double* X, int64_t ldx, double* v, double* work_n) {
    // X = (I - vv') X
    //  --> Y = X' v
    //  --> X = X - v Y'
    blas::gemv(Layout::ColMajor, blas::Op::Trans, m, n, (double) 1.0, X, ldx, v, 1, (double) 0.0, work_n, 1);
    blas::ger(Layout::ColMajor, m, n,  (double)  -1.0, v, 1, work_n, 1, X, ldx);
    return;
}

struct CallableSpMat {
    sparse_matrix_t A;
    int64_t m;
    double* work = nullptr;
    std::vector<double> work_stdvec{};
    int64_t n_work = 0;
    matrix_descr des{SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    // ^ The latter two entries of des are not actually used. I'm only specifying them 
    //   to avoid compiler warings.
    std::vector<double> regs{0.0};
    double* unit_ones = nullptr;
    std::vector<double> unit_ones_stdvec{};
    double* work_n = nullptr;
    std::vector<double> work_n_stdvec{};

    /*  C =: alpha * A * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta,  double* C, int64_t ldc
    ) {
        if (work == nullptr) {
            work_stdvec.resize(m*n);
            unit_ones_stdvec.resize(m);
            work_n_stdvec.resize(n);
            work = work_stdvec.data();
            unit_ones = unit_ones_stdvec.data();
            double val = std::pow((double)m, -0.5);
            for (int64_t i = 0; i < m; ++i)
                unit_ones[i] = val;
            work_n = work_n_stdvec.data();
            n_work = n;
        } else {
            randblas_require(n_work >= n);
        }
        randblas_require(ldb == m);
        blas::copy(m*n, B, 1, work, 1);
        project_out_vec(m, n, work, m, unit_ones, work_n);
        //int t = omp_get_max_threads();
        //omp_set_num_threads(1);
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        //TIMED_LINE(
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, des, mkl_layout, work, n, ldb, beta, C, ldc
        );//, "SPMM A   : ");
        //omp_set_num_threads(t);
        project_out_vec(m, n, C, ldc, unit_ones, work_n);
    }
};

struct CallableChoSolve {
    sparse_matrix_t G;
    int64_t m;
    matrix_descr des = {SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    double* work = nullptr;
    std::vector<double> work_stdvec{};
    int64_t n_work = 0;
    double* unit_ones = nullptr;
    std::vector<double> unit_ones_stdvec{};
    double* work_n = nullptr;
    std::vector<double> work_n_stdvec{};

    /*  C =: alpha * inv(G G') * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta, double* C, int64_t ldc
    ) {
        if (work == nullptr) {
            work_stdvec.resize(2*m*n);
            work = work_stdvec.data();    
            unit_ones_stdvec.resize(m);
            unit_ones = unit_ones_stdvec.data();
            double val = std::pow((double)m, -0.5);
            for (int64_t i = 0; i < m; ++i)
               unit_ones[i] = val;
            work_n_stdvec.resize(n);
            work_n = work_n_stdvec.data();
            n_work = n;
        } else {
            randblas_require(n_work >= n);
        }
        randblas_require(beta == (double) 0.0);
        blas::copy(m*n, B, 1, work+m*n, 1);
        project_out_vec(m, n, work+m*n, m, unit_ones, work_n);
        randblas_require(ldb == m);
        // TRSM, then transposed TRSM.
        //int t = omp_get_max_threads();
        //omp_set_num_threads(1);
        sparse_status_t status;
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
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
        //omp_set_num_threads(t);
        project_out_vec(m, n, C, ldc, unit_ones, work_n);
    }

    // ~CallableChoSolve() {}
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
    std::vector<double> unit_ones{};
    std::vector<double> proj_work_n{};
    std::vector<double> rchol_pcg_work{};
    double call_pcg_tol = 1e-10;
    int64_t max_iters = 100;
    bool use_rchol_pcg = true;

    LaplacianPinv(CallableSpMat &L, CallableChoSolve &N,
        std::vector<int> &S, int num_threads, double pcg_tol, int maxit, SparseCSR_RC &G_rchol,
        bool verbose = false
    ) :
        SymmetricLinearOperator<double>(L.m),
        L_callable{L.A, L.m},
        N_callable{N.G, N.m},
        verbose_pcg(verbose),
        rchol_pcg_object(L.A, L.m, S, num_threads, pcg_tol, maxit, G_rchol),
        call_pcg_tol(pcg_tol),
        max_iters((int64_t) maxit)
    {
        mkl_sparse_optimize(L.A);
        mkl_sparse_optimize(N.G);
    }; 

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
        work_seminorm.resize(mn);
        unit_ones.resize(mn, std::pow((double)m, -0.5));
        proj_work_n.resize(n);
        rchol_pcg_work.resize(4*m);
        double *ones = unit_ones.data();
        double *work_n = proj_work_n.data();
        double *work_seminorm_ = work_seminorm.data();
        double *rchol_pcg_work_ = rchol_pcg_work.data();
        if (n < (int64_t) L_callable.regs.size()) {
            L_callable.regs.resize(n, 0.0);
        }

        std::vector<double> sn_log{};
        auto seminorm = [work_seminorm_, ones, work_n, &sn_log](int64_t __n, int64_t __s, double* NR) {
            blas::copy(__n*__s, NR, 1, work_seminorm_, 1);
            project_out_vec(__n, __s,  work_seminorm_, __n, ones, work_n);
            double out = blas::nrm2(__n*__s, work_seminorm_, 1);
            sn_log.push_back(out);
            return out;
        };

        for (int64_t i = 0; i < mn; ++i)
            work_B[i] = alpha * B[i];
        //std::cout << "n = " << n << std::endl;
        project_out_vec(m, n, work_B.data(), m, ones, work_n);
        if (use_rchol_pcg) {
            std::cout << std::left << std::setw(10) << "index" 
                    << std::setw(10) << "iters" 
                    << std::setw(15) << "relres" << std::endl;
            for (int64_t i = 0; i < n; ++i) {
                int iter = 0;
                double relres = 0.0;
                double* curr_b = work_B.data() + m*i;
                double* curr_c = work_C.data() + m*i;
                rchol_pcg_object.iteration(&L_callable.A, curr_b, &N_callable.G, curr_c, relres, iter, rchol_pcg_work_);
                std::cout << std::left << std::setw(10) << i 
                    << std::setw(10) << iter 
                    << std::setw(15) << relres << std::endl;
            }
        } else {
            // logging
            std::cout << std::left << std::setw(10) << "iters" << std::setw(15) << "relres" << std::endl;
            // work
            RandBLAS::util::safe_scal(mn, 0.0, work_C.data(), 1);
            RandLAPACK::lockorblock_pcg(L_callable, work_B, call_pcg_tol, max_iters, N_callable, seminorm, work_C, verbose_pcg);
            // logging
            std::cout << std::left 
            << std::setw(10) << sn_log.size()
            << std::setw(15) << sn_log[sn_log.size()-1] / sn_log[0] << std::endl;
        }
        project_out_vec(m, n, work_C.data(), m, ones, work_n);
        blas::copy(mn, work_C.data(), 1, C, 1);
    }

    double operator()(int64_t i, int64_t j) {
        UNUSED(i); UNUSED(j);
        randblas_require(false);
        return 0.0;
    }
};


void laplacian_from_matrix_market(std::string fn, SparseCSR_RC &A, double reg=0.0) {
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
    int approx_rank = 4;
    int threads = 1;
    if (argc > 1)
        approx_rank = atoi(argv[1]);
    if (argc > 2)
        threads = atoi(argv[2]);
    if (argc > 3)
        mat = argv[3];
    return std::make_tuple(mat, approx_rank, threads);
}



template <typename index_t>
void handle_onezero_diag_ut(int64_t n, index_t* &rowptr, index_t* &colidxs, double* &vals) {
    int64_t nnz = static_cast<int64_t>(rowptr[n]);
    double Gii;
    for (int64_t i = 0; i < n-1; ++i) {
        randblas_require(rowptr[i] < rowptr[i+1]);
        // ^ then we have a structural nonzero.
        Gii = vals[rowptr[i]];
        randblas_require(Gii > 0);
    }
    double TRAILING_ENTRY = 1.0;
    if (rowptr[n-1] < rowptr[n]) {
        // then we have a structural nonzero.
        Gii = vals[rowptr[n-1]];
        if (std::abs(Gii) < 1e-16) {
            vals[rowptr[n-1]] = TRAILING_ENTRY;
        }
        return;
    } 
    // else, we have to add a single nonzero element to G.
    index_t* extended_colidxs = new index_t[nnz+1];
    double*  extended_vals    = new double[nnz+1];
    for (int64_t ell = 0; ell < nnz; ++ell) {
        extended_colidxs[ell] = colidxs[ell];
        extended_vals[ell] = vals[ell];
    }
    extended_colidxs[nnz] = static_cast<size_t>(n-1);
    extended_vals[nnz] = TRAILING_ENTRY;
    rowptr[n] = nnz + 1;
    colidxs = extended_colidxs;
    vals = extended_vals;
    return;
}


int main(int argc, char *argv[]) {
    std::cout << std::setprecision(16);

    auto [fn, k, threads] = parse_args(argc, argv);
    SparseCSR_RC A;
    laplacian_from_matrix_market(fn, A);
    // auto A = laplace_3d(3); // n x n x n grid
    int64_t n = A.size();
    SparseCSR_RC G, Aperm;
    std::vector<size_t> P;
    std::vector<int> S;
    std::string filename = "orders/order_n" + std::to_string(n) + "_t" + std::to_string(threads) + ".txt";
    
    double reg = 1e-8;
    SparseCSR_RC A_reg;
    laplacian_from_matrix_market(fn, A_reg, reg);
    rchol(A_reg, G, P, S, threads, filename);
    // Never reference A_reg again.
    reorder(A, P, Aperm);
    if (reg == 0.0) {
        handle_onezero_diag_ut(n, G.rowPtr, G.colIdx, G.val);
    } // if reg > 0, then there's no need for the logic above.

    sparse_matrix_t Aperm_mkl, G_mkl;
    sparse_matrix_t_from_SparseCSR_RC(Aperm, Aperm_mkl);
    sparse_matrix_t_from_SparseCSR_RC(G, G_mkl);
    CallableSpMat Aperm_callable{Aperm_mkl, n};
    CallableChoSolve N_callable{G_mkl, n};
    LaplacianPinv Lpinv(Aperm_callable, N_callable, S, threads, 1e-8, 200, G, false);
    //
    //  IDEA: right now we have "regs" that's size one and contains 0.0.
    //        if we instead have regs be size n and contain all 0.0 then
    //        the PCG solves will be performed in parallel. This would
    //        make a fairer comparison of block vs non-block PCG.
    //
    //  VERDICT: Lockstep and block have basically the same performance.
    //           To the extent that there's a difference at all, lockstep
    //           is marginally faster.
    //
    //           One key difference is that we compute the error
    //           with a black-box seminorm, while the rchol_pcg function
    //           uses the 2-norm (sensible for nonsingular systems!).
    //           But that wouldn't explain why the issue only shows up
    //           when we increase the number of blocks in the matrix 
    //           partitioning.
    //
    //  NOTE: It's a little dubious to just stick a +1 in the bottom right 
    //        corner of G. Let's switch to regularizing the Laplacian a bit more. 
    //        --> This doesn't address what might happen if the kernel has dimension > 1.
    //        
    
    // low-rank approx time!
    //      NOTE: REVD2 isn't quite like QB2; it doesn't have a block size.
    RandLAPACK::SYPS<double, DefaultRNG>  SYPS(5, 1, false, false);
    RandLAPACK::HQRQ<double>              Orth(false, false); 
    RandLAPACK::SYRF<double, DefaultRNG>  SYRF(SYPS, Orth, false, false);
    RandLAPACK::REVD2<double, DefaultRNG> NystromAlg(SYRF, 2, false);
    double silly_tol = 1e4;
    // ^ ensures we break after one iteration
    std::vector<double> V(n*k, 0.0);
    std::vector<double> eigvals(k, 0.0);
    RandBLAS::RNGState state{};
    int64_t k_ = k;
    // TIMED_LINE(
    // NystromAlg.call(Lpinv, k_, silly_tol, V, eigvals, state), "NystromAlg.call -  rchol_pcg : ");
    std::vector<double> V_next(n*k, 0.0);
    std::vector<double> eigvals_next(k, 0.0);
    Lpinv.use_rchol_pcg = false;
    k_ = k;
    TIMED_LINE(
    NystromAlg.call(Lpinv, k_, silly_tol, V_next, eigvals_next, state), "NystromAlg.call - block PCG : ");
    V.clear();
    V.resize(n*k);
    eigvals.clear();
    eigvals.resize(k);
    Aperm_callable.regs.resize(k, 0.0);
    k_ = k;
    TIMED_LINE(
    NystromAlg.call(Lpinv, k_, silly_tol, V, eigvals, state), "NystromAlg.call - lockstep : ");
    // std::ofstream file_stream("V.txt");
    // RandBLAS::print_buff_to_stream(
    //     file_stream, Layout::ColMajor, n, k_, V.data(), n, "V", 16
    // );

    mkl_sparse_destroy(Aperm_mkl);
    mkl_sparse_destroy(G_mkl);

    return 0;
}
