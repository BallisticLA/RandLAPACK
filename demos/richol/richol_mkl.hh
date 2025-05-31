#pragma once

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <fstream>
#include <omp.h>
#include <iomanip>


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
void project_out_vec(int64_t m, int64_t n, T* X, int64_t ldx, T* v, T* work_n) {
    // X = (I - vv') X
    //  --> Y = X' v
    //  --> X = X - v Y'
    blas::gemv(Layout::ColMajor, blas::Op::Trans, m, n, (T) 1.0, X, ldx, v, 1, (T) 0.0, work_n, 1);
    blas::ger(Layout::ColMajor, m, n,  (T)  -1.0, v, 1, work_n, 1, X, ldx);
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
    std::vector<double> times{};

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
        project_out_vec(m, n, C, ldc, unit_ones, work_n);
    }
};


struct CallableChoSolve {
    sparse_matrix_t G;
    int64_t m;
    matrix_descr des = {SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT};
    double* work = nullptr;
    std::vector<double> work_stdvec{};
    int64_t n_work = 0;
    double* unit_ones = nullptr;
    std::vector<double> unit_ones_stdvec{};
    double* work_n = nullptr;
    std::vector<double> work_n_stdvec{};
    std::vector<double> times{};

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
        omp_set_dynamic(1);
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        auto t0 = std_clock::now();
        //TIMED_LINE(
        status = mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE    , alpha, G, des, mkl_layout, work+m*n, n, ldb, work, m); //, "TRSM G   : ");
        if (status != SPARSE_STATUS_SUCCESS) {
            std::cout << "TRSM failed with error code " << status << std::endl;
            throw std::runtime_error("TRSM failure.");
        }
        //TIMED_LINE(
        status = mkl_sparse_d_trsm(SPARSE_OPERATION_TRANSPOSE,   1.0, G, des, mkl_layout, work, n, m, C, ldc); //, "TRSM G^T : ");
        if (status != SPARSE_STATUS_SUCCESS) {
            std::cout << "TRSM failed with error code " << status << std::endl;
            throw std::runtime_error("TRSM failure.");
        }
        auto t1 = std_clock::now();
        times.push_back(seconds_elapsed(t0, t1));
        //omp_set_num_threads(t);
        omp_set_dynamic(0);
        project_out_vec(m, n, C, ldc, unit_ones, work_n);
    }

};


struct LaplacianPinv : public SymmetricLinearOperator<double> {
    public:
    // inherited --> const int64_t m;
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
        SymmetricLinearOperator<double>(L.m),
        L_callable{L.A, L.m},
        N_callable{N.G, N.m},
        verbose_pcg(verbose),
        times(4, 0.0),
        call_pcg_tol(pcg_tol),
        max_iters((int64_t) maxit)
    { }; 

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
        project_out_vec(m, n, work_B.data(), m, ones, work_n);
        // logging
        std::cout << std::left << std::setw(10) << "iters" << std::setw(15) << "relres" << std::endl;
        // work
        RandBLAS::util::safe_scal(mn, 0.0, work_C.data(), 1);
        auto t0 = std_clock::now();
        RandLAPACK::lockorblock_pcg(L_callable, work_B, call_pcg_tol, max_iters, N_callable, seminorm, work_C, verbose_pcg);
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
        project_out_vec(m, n, work_C.data(), m, ones, work_n);
        blas::copy(mn, work_C.data(), 1, C, 1);
    }

    double operator()(int64_t i, int64_t j) {
        UNUSED(i); UNUSED(j);
        randblas_require(false);
        return 0.0;
    }
};


} // end namespace richol::linops
