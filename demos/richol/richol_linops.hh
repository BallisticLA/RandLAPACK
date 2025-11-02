#pragma once

#include "RandLAPACK.hh"
#include "RandBLAS/sparse_data/trsm_dispatch.hh"
#include "RandLAPACK/comps/rl_determiter.hh"
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

using RandBLAS::spmm;
using RandBLAS::sparse_data::trsm;
// void trsm(
//      blas::Layout layout, blas::Op opA, T alpha,
//      const SpMat &A, blas::Uplo uplo, blas::Diag diag,
//      int64_t n, T *B, int64_t ldb,
//      int validation_mode = 1
// )
using RandBLAS::sparse_data::trsm_matrix_validation;
// inline void trsm_matrix_validation(
//      const SpMat &A, blas::Uplo uplo,
//      blas::Diag diag, int mode 
// )
using RandBLAS::sparse_data::SparseMatrix;

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

template <SparseMatrix SpMat>
struct CallableSpMat {
    SpMat *A;
    int64_t dim;
    double* work = nullptr;
    std::vector<double> work_stdvec{};
    int64_t n_work = 0;
    std::vector<double> regs{0.0};
    double* unit_ones = nullptr;
    std::vector<double> unit_ones_stdvec{};
    double* work_n = nullptr;
    std::vector<double> work_n_stdvec{};
    std::vector<double> times{};
    const int64_t num_ops = 1;
    bool project_out = true;

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
        if (project_out)
            project_out_vec(dim, n, work, dim, unit_ones, work_n);
        //int t = omp_get_max_threads();
        //omp_set_num_threads(1);
        auto t0 = std_clock::now();
        omp_set_dynamic(1);
        //TIMED_LINE(
        spmm(layout, Op::NoTrans, Op::NoTrans, dim, n, dim, alpha, *A, work, dim, beta, C, ldc);
        //, "SPMM A   : ");
        auto t1 = std_clock::now();
        times.push_back(seconds_elapsed(t0, t1));
        //omp_set_num_threads(t);
        omp_set_dynamic(0);
        if (project_out)
            project_out_vec(dim, n, C, ldc, unit_ones, work_n);
    }
};


template <SparseMatrix SpMat>
struct CallableChoSolve {
    SpMat *G;
    int64_t dim;
    int64_t trsm_validation = 0;
    int64_t n_work = 0;
    double* unit_ones = nullptr;
    std::vector<double> unit_ones_stdvec{};
    double* work_n = nullptr;
    std::vector<double> work_n_stdvec{};
    std::vector<double> times{};
    bool project_out = true;

    void validate() {
        trsm_matrix_validation(*G, Uplo::Lower, Diag::NonUnit, 3);
        return;
    }

    /*  C =: alpha * inv(G G') * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta, double* C, int64_t ldc
    ) {
        if (work_n == nullptr) {
            unit_ones_stdvec.resize(dim);
            unit_ones = unit_ones_stdvec.data();
            double val = std::pow((double)dim, -0.5);
            for (int64_t i = 0; i < dim; ++i)
               unit_ones[i] = val;
            // for (int64_t i = = 0; )
            work_n_stdvec.resize(n);
            work_n = work_n_stdvec.data();
            n_work = n;
        } else {
            randblas_require(n_work >= n);
        }
        randblas_require(beta == (double) 0.0);
        randblas_require(ldb == dim);
        randblas_require(ldc == dim);
        blas::copy(dim*n, B, 1, C, 1);
        if (project_out)
            project_out_vec(dim, n, C, ldc, unit_ones, work_n);
        // TRSM, then transposed TRSM.
        //int t = omp_get_max_threads();
        //omp_set_num_threads(1);
        omp_set_dynamic(1);
        auto t0 = std_clock::now();
        //TIMED_LINE(
        trsm(layout, Op::NoTrans,       alpha, *G, Uplo::Lower, Diag::NonUnit, n, C, ldc, trsm_validation); //, "TRSM G : ");
        //TIMED_LINE(
        trsm(layout,   Op::Trans, (double)1.0, *G, Uplo::Lower, Diag::NonUnit, n, C, ldc, trsm_validation); //, "TRSM G^T : ");
        auto t1 = std_clock::now();
        times.push_back(seconds_elapsed(t0, t1));
        //omp_set_num_threads(t);
        omp_set_dynamic(0);
        if (project_out)
            project_out_vec(dim, n, C, ldc, unit_ones, work_n);
    }

};


template <SparseMatrix SpMat>
struct LaplacianPinv {
    public:
    const int64_t dim;
    CallableSpMat<SpMat>    L_callable;
    CallableChoSolve<SpMat> N_callable;
    bool verbose_pcg;
    std::vector<double> work_B{};
    std::vector<double> work_C{};
    std::vector<double> work_seminorm{};
    std::vector<double> unit_ones{};
    std::vector<double> proj_work_n{};
    std::vector<double> times{};
    double call_pcg_tol = 1e-10;
    int64_t max_iters = 100;
    using scalar_t = typename SpMat::scalar_t;
    const int64_t num_ops = 1;

    LaplacianPinv(CallableSpMat<SpMat> &L, CallableChoSolve<SpMat> &N, double pcg_tol, int maxit,
        bool verbose = false
    ) :
        dim(L.dim),
        L_callable{L.A, L.dim},
        N_callable{N.G, N.dim},
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
        randblas_require(ldb == dim);
        randblas_require(ldc == dim);
        int64_t n_x_dim = dim*n;
        work_B.resize(n_x_dim);
        work_C.resize(n_x_dim);
        work_seminorm.resize(n_x_dim);
        unit_ones.resize(n_x_dim, std::pow((double)dim, -0.5));
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

        for (int64_t i = 0; i < n_x_dim; ++i)
            work_B[i] = alpha * B[i];
        //std::cout << "n = " << n << std::endl;
        project_out_vec(dim, n, work_B.data(), dim, ones, work_n);
        // logging
        std::cout << std::left << std::setw(10) << "iters" << std::setw(15) << "relres" << std::endl;
        // work
        RandBLAS::util::safe_scal(n_x_dim, 0.0, work_C.data(), 1);
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
        << std::setw(10) << sn_log.size()/2
        << std::setw(15) << sn_log[sn_log.size()-1] / sn_log[0] << std::endl;
        project_out_vec(dim, n, work_C.data(), dim, ones, work_n);
        blas::copy(n_x_dim, work_C.data(), 1, C, 1);
    }

    double operator()(int64_t i, int64_t j) {
        UNUSED(i); UNUSED(j);
        randblas_require(false);
        return 0.0;
    }
};


template <typename T>
struct IdentityMatrix {
    int64_t num_ops = 1;
    const int64_t dim;
    std::vector<double> times{};
    IdentityMatrix(int64_t _n) : dim(_n), times(4,(T)0.0) { }
    void operator()(blas::Layout ell, int64_t _n, T alpha, T* const _B, int64_t _ldb, T beta, T* _C, int64_t _ldc) {
        randblas_require(ell == blas::Layout::ColMajor);
        UNUSED(_ldb); UNUSED(_ldc);
        blas::scal(dim*_n, beta, _C, 1);
        blas::axpy(dim*_n, alpha, _B, 1, _C, 1);
    };
};


} // end namespace richol::linops
