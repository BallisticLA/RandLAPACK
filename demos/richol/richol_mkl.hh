#pragma once

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#define MKL_INT int64_t
#include <fstream>
#include <omp.h>
#include <iomanip>
#include <mkl_spblas.h>


using RandLAPACK::linops::SymmetricLinearOperator;
using RandBLAS::DefaultRNG;
using RandBLAS::CSRMatrix;
using RandBLAS::CSCMatrix;
using RandBLAS::COOMatrix;

using RandBLAS::sparse_data::right_spmm;
using RandBLAS::sparse_data::left_spmm;

//#define FINE_GRAINED

#ifndef TIMED_LINE
using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;
#define DOUT(_d) std::setprecision(8) << _d
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
#endif

double seconds_elapsed(timepoint_t tp0, timepoint_t tp1) {
    return (double) duration_cast<microseconds>(tp1 - tp0).count() / 1e6;
}


namespace richol::linops {

template <typename T>
void project_out_vec(int64_t dim, int64_t n, T* X, int64_t ldx, T* v, T* work_n) {
    // X = (I - vv') X
    //  --> Y = X' v
    //  --> X = X - v Y'
    blas::gemv(Layout::ColMajor, blas::Op::Trans, dim, n, (T) 1.0, X, ldx, v, 1, (T) 0.0, work_n, 1);
    blas::ger(Layout::ColMajor, dim, n,  (T)  -1.0, v, 1, work_n, 1, X, ldx);
    return;
}

// /* status from MKL sparse matrix routines */
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

template <RandBLAS::SignedInteger sint_t>
void sparse_matrix_t_from_randblas_csr(const CSRMatrix<double,sint_t> &A, sparse_matrix_t &mat) {
    // Expected mode of calling:
    //
    //      sparse_matrix_t mat;
    //      sparse_matrix_t_from_randblas_csr(A, mat);
    //      /* do stuff */
    //      mkl_sparse_destroy(mat);
    //
    auto N = A.n_rows;
    auto cpt = A.rowptr;
    auto rpt = A.colidxs;
    auto datapt = A.vals;
    sint_t *pointerB = new sint_t[N + 1]();
    //sint_t *pointerB = (sint_t *)calloc(N+1, sizeof(sint_t));
    sint_t *pointerE = new sint_t[N + 1]();
    //sint_t *pointerE = (sint_t *)calloc(N+1, sizeof(sint_t));
    sint_t *update_intend_rpt = new sint_t[cpt[N]]();
    //sint_t *update_intend_rpt = (sint_t *)calloc(cpt[N], sizeof(sint_t));
    double *update_intend_datapt = new double[cpt[N]]();
    //double *update_intend_datapt = (double *)calloc(cpt[N], sizeof(double));
    for(sint_t i = 0; i < N; i++) {
        auto start = static_cast<sint_t>(cpt[i]);
        auto last  = static_cast<sint_t>(cpt[i + 1]);
        pointerB[i] = start;
        for (auto j = start; j < last; j++) {
            update_intend_datapt[j] = datapt[j];
            update_intend_rpt[j]    = rpt[j];
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
    //   community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/mkl-sparse-d-create-csr-possibility-of-memory-leak/dim-p/1313882#M32031
    return;
}


struct CallableSpMat {
    sparse_matrix_t A;
    int64_t dim;
    double* work = nullptr;
    std::vector<double> work_stdvec{};
    int64_t n_work = 0;
    matrix_descr des{SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    // ^ The latter two entries of des are not actually used. I'dim only specifying them 
    //   to avoid compiler warings.
    std::vector<double> regs{0.0};
    double* unit_ones = nullptr;
    std::vector<double> unit_ones_stdvec{};
    double* work_n = nullptr;
    std::vector<double> work_n_stdvec{};
    std::vector<double> times{};
    const int64_t num_ops = 1;

    /*  C =: alpha * A * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta,  double* C, int64_t ldc
    ) {
        if (work == nullptr) {
            work_stdvec.resize(dim*n);
            unit_ones_stdvec.resize(dim);
            work_n_stdvec.resize(n);
            work = work_stdvec.data();
            unit_ones = unit_ones_stdvec.data();
            double val = std::pow((double)dim, -0.5);
            for (int64_t i = 0; i < dim; ++i)
                unit_ones[i] = val;
            work_n = work_n_stdvec.data();
            n_work = n;
        } else {
            randblas_require(n_work >= n);
        }
        randblas_require(ldb == dim);
        blas::copy(dim*n, B, 1, work, 1);
        project_out_vec(dim, n, work, dim, unit_ones, work_n);
        //int t = omp_get_max_threads();
        //omp_set_num_threads(1);
        auto t0 = std_clock::now();
        omp_set_dynamic(1);
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        //TIMED_LINE(
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, des, mkl_layout, work, n, ldb, beta, C, ldc
        );//, "SPMM A   : ");
        auto t1 = std_clock::now();
        times.push_back(seconds_elapsed(t0, t1));
        //omp_set_num_threads(t);
        omp_set_dynamic(0);
        project_out_vec(dim, n, C, ldc, unit_ones, work_n);
    }
};


struct CallableChoSolve {
    sparse_matrix_t G;
    int64_t dim;
    matrix_descr des = {SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT};
    double* work = nullptr;
    std::vector<double> work_stdvec{};
    int64_t n_work = 0;
    double* unit_ones = nullptr;
    std::vector<double> unit_ones_stdvec{};
    double* work_n = nullptr;
    std::vector<double> work_n_stdvec{};
    std::vector<double> times{};
    const int64_t num_ops = 1;

    /*  C =: alpha * inv(G G') * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta, double* C, int64_t ldc
    ) {
        if (work == nullptr) {
            work_stdvec.resize(2*dim*n);
            work = work_stdvec.data();    
            unit_ones_stdvec.resize(dim);
            unit_ones = unit_ones_stdvec.data();
            double val = std::pow((double)dim, -0.5);
            for (int64_t i = 0; i < dim; ++i)
               unit_ones[i] = val;
            work_n_stdvec.resize(n);
            work_n = work_n_stdvec.data();
            n_work = n;
        } else {
            randblas_require(n_work >= n);
        }
        randblas_require(beta == (double) 0.0);
        blas::copy(dim*n, B, 1, work+dim*n, 1);
        project_out_vec(dim, n, work+dim*n, dim, unit_ones, work_n);
        randblas_require(ldb == dim);
        // TRSM, then transposed TRSM.
        //int t = omp_get_max_threads();
        //omp_set_num_threads(1);
        sparse_status_t status;
        omp_set_dynamic(1);
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        auto t0 = std_clock::now();
        //TIMED_LINE(
        status = mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE    , alpha, G, des, mkl_layout, work+dim*n, n, ldb, work, dim); //, "TRSM G   : ");
        if (status != SPARSE_STATUS_SUCCESS) {
            std::cout << "TRSM failed with error code " << status << std::endl;
            throw std::runtime_error("TRSM failure.");
        }
        //TIMED_LINE(
        status = mkl_sparse_d_trsm(SPARSE_OPERATION_TRANSPOSE,   1.0, G, des, mkl_layout, work, n, dim, C, ldc); //, "TRSM G^T : ");
        if (status != SPARSE_STATUS_SUCCESS) {
            std::cout << "TRSM failed with error code " << status << std::endl;
            throw std::runtime_error("TRSM failure.");
        }
        auto t1 = std_clock::now();
        times.push_back(seconds_elapsed(t0, t1));
        //omp_set_num_threads(t);
        omp_set_dynamic(0);
        project_out_vec(dim, n, C, ldc, unit_ones, work_n);
    }

};


struct LaplacianPinv {
    public:
    using scalar_t = double;
    const int64_t dim;
    CallableSpMat    L_callable;
    CallableChoSolve N_callable;
    bool verbose_pcg;
    std::vector<double> work_B{};
    std::vector<double> work_C{};
    std::vector<double> work_seminorm{};
    std::vector<double> unit_ones{};
    std::vector<double> proj_work_n{};
    std::vector<double> times{};
    double call_pcg_tol = 1e-10;
    int64_t max_iters = 100;

    LaplacianPinv(CallableSpMat &L, CallableChoSolve &N, double pcg_tol, int maxit,
        bool verbose = false
    ) :
        dim(L.dim),
        L_callable{L.A, L.dim},
        N_callable{N.G, N.dim},
        verbose_pcg(verbose),
        times(4, 0.0),
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
        randblas_require(ldb == dim);
        randblas_require(ldc == dim);
        int64_t mn = dim*n;
        work_B.resize(mn);
        work_C.resize(mn);
        work_seminorm.resize(mn);
        unit_ones.resize(mn, std::pow((double)dim, -0.5));
        proj_work_n.resize(n);
        double *ones = unit_ones.data();
        double *work_n = proj_work_n.data();
        double *work_seminorm_ = work_seminorm.data();
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
        project_out_vec(dim, n, work_B.data(), dim, ones, work_n);
        // logging
        std::cout << std::left << std::setw(10) << "iters" << std::setw(15) << "relres" << std::endl;
        // work
        RandBLAS::util::safe_scal(mn, 0.0, work_C.data(), 1);
        auto t0 = std_clock::now();
        RandLAPACK::pcg(L_callable, work_B.data(), n, seminorm, call_pcg_tol, max_iters, N_callable, work_C.data(), verbose_pcg);
        auto t1 = std_clock::now();
        auto total_spmm   = std::reduce(L_callable.times.begin(), L_callable.times.end());
        auto total_sptrsm = std::reduce(N_callable.times.begin(), N_callable.times.end());
        times[0] += total_spmm;
        times[1] += total_sptrsm;
        L_callable.times.clear();
        N_callable.times.clear();
        times[2] += seconds_elapsed(t0, t1);
        times[3] += (double) sn_log.size()/2;

        // logging
        std::cout << std::left 
        << std::setw(10) << sn_log.size()
        << std::setw(15) << sn_log[sn_log.size()-1] / sn_log[0] << std::endl;
        project_out_vec(dim, n, work_C.data(), dim, ones, work_n);
        blas::copy(mn, work_C.data(), 1, C, 1);
    }

    double operator()(int64_t i, int64_t j) {
        UNUSED(i); UNUSED(j);
        randblas_require(false);
        return 0.0;
    }
};


} // end namespace richol::linops
