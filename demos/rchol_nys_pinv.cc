#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <omp.h>

#include "sparse.hpp"
// ^ defines CSR sparse matrix type
// #include "rchol.hpp"
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


void sparse_matrix_t_from_SparseCSR(const SparseCSR &A, sparse_matrix_t &mat) {
    // this implementation is lifted from /home/rjmurr/laps/rchol-repo/c++/util/pcg.cpp
    //
    // Expected mode of calling:
    //
    //      sparse_matrix_t mat;
    //      sparse_matrix_t_from_SparseCSR(A, mat);
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
    auto status = mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, N, N, pointerB, pointerE, update_intend_rpt, update_intend_datapt);
    return;
}


template <typename RBSpMat>
struct CallableSpMat {
    sparse_matrix_t A;
    int64_t m;
    RBSpMat *A_rb;
    bool use_mkl = false;
    matrix_descr des{SPARSE_MATRIX_TYPE_GENERAL};
    std::vector<double> regs{0.0};

    /*  C =: alpha * A * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta,  double* C, int64_t ldc
    ) {
        int t = omp_get_max_threads();
        omp_set_num_threads(1);
        if (use_mkl) {
            auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
            TIMED_LINE(
            mkl_sparse_d_mm(
                SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, des, mkl_layout, B, n, ldb, beta, C, ldc
            ), "SPMM A   : ");
        } else {
            TIMED_LINE(
            RandBLAS::sparse_data::left_spmm(
                Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m, alpha, *A_rb, 0, 0, B, ldb, beta, C, ldc
            ), "SPMM A   : ");
        }
        omp_set_num_threads(t);
    }
};

struct CallableChoSolve {
    sparse_matrix_t G;
    int64_t m;
    matrix_descr des{SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    double* work = nullptr;
    int64_t n_work = 0;
    /*  C =: alpha * inv(G G') * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, const double* B, int64_t ldb,
        double beta, double* C, int64_t ldc
    ) {
        if (work == nullptr) {
            work = new double[m*n];
            n_work = n;
        } else {
            randblas_require(n_work >= n);
        }
        randblas_require(beta == (double) 0.0);
        // TRSM, then transposed TRSM.
        int t = omp_get_max_threads();
        omp_set_num_threads(1);
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        //TIMED_LINE(
        auto status = mkl_sparse_d_trsm(SPARSE_OPERATION_TRANSPOSE    , alpha, G, des, mkl_layout, B, n, ldb, work, m); //, "TRSM G   : ");
        //TIMED_LINE(
        status = mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE,   1.0, G, des, mkl_layout, work, n, m, C, ldc); //, "TRSM G^T : ");
        omp_set_num_threads(t);
    }

    ~CallableChoSolve() {
        if (work != nullptr) 
            delete [] work;
    }
};

// NOTE: below probably needs to conform to the SymmetricLinearOperator API.
template <typename RBSpMat>
struct LaplacianPinv : public SymmetricLinearOperator<double> {
    public:
    // inherited --> const int64_t m;
    CallableSpMat<RBSpMat>    L_callable;
    CallableChoSolve          N_callable;
    std::vector<double> work_B{};
    std::vector<double> work_C{};
    double call_pcg_tol = 1e-10;
    int64_t max_iters = 100;
    static const auto constexpr seminorm = [](int64_t n, int64_t s, const double* NR){return blas::nrm2(n*s, NR, 1);};

    LaplacianPinv(CallableSpMat<RBSpMat> &L, CallableChoSolve &N) :
        SymmetricLinearOperator<double>(L.m), L_callable{L.A, L.m, L.A_rb}, N_callable{N.G, N.m} {}; 

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
        RandBLAS::util::safe_scal(mn, 0.0, work_C.data(), 1);
        RandLAPACK::lockorblock_pcg(L_callable, work_B, call_pcg_tol, max_iters, N_callable, seminorm, work_C, true);
        blas::copy(mn, work_C.data(), 1, C, 1);
    }

    double operator()(int64_t i, int64_t j) {
        UNUSED(i); UNUSED(j);
        randblas_require(false);
        return 0.0;
    }


};


void laplacian_from_matrix_market(std::string fn, SparseCSR &A) {

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
    double* diag = new double[n]{0.0};
    for (int64_t i = 0; i < m; ++i) {
        coo.rows[i] = rows[i];
        coo.cols[i] = cols[i];
        double v = vals[i];
        randblas_require(v >= 0);
        coo.vals[i] = -v;
        diag[cols[i]] += v;
    }
    for (int64_t i = m; i < nnz; ++i) {
        coo.rows[i] = i - m;
        coo.cols[i] = i - m;
        coo.vals[i] = diag[i - m];
    }
    delete [] diag;
    // convert COO format Laplacian to CSR format, using RandBLAS.
    CSRMatrix<double> csr(n, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);

    // Convert RandBLAS::CSRMatrix to SparseCSR
    std::vector<size_t> rowPtrA(csr.n_rows+1);
    std::vector<size_t> colIdxA(csr.nnz);
    for (int64_t i = 0; i < csr.n_rows + 1; ++i)
        rowPtrA[i] = (size_t) csr.rowptr[i];
    vals.resize(csr.nnz);
    for (int64_t i = 0; i < csr.nnz; ++i) {
        colIdxA[i] = (size_t) csr.colidxs[i];
        vals[i] = csr.vals[i];
    }

    A.init(rowPtrA, colIdxA, vals, true);

    return;
}


auto parse_args(int argc, char** argv) {
    std::string mat{"../sparse-data-matrices/fl2010/fl2010.mtx"};
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



int main(int argc, char *argv[]) {
    std::cout << std::setprecision(3);

    auto [fn, k, threads] = parse_args(argc, argv);
    SparseCSR A, G, Aperm;
    laplacian_from_matrix_market(fn, A);
    std::vector<size_t> P;
    rchol(A, G, P, threads);
    reorder(A, P, Aperm);
    int64_t n = Aperm.size();
    int64_t nnz = Aperm.nnz();
    std::vector<int64_t> ApermRowPtr(nnz);
    std::vector<int64_t> ApermColIdx(nnz);

    for (int64_t i = 0; i < n+1; ++i)
        ApermRowPtr[i] = (int64_t) Aperm.rowPtr[i];
    for (int64_t i = 0; i < nnz; ++i) 
        ApermColIdx[i] = (int64_t) Aperm.colIdx[i];

    CSRMatrix A_perm_rb_csr(n, n, nnz, Aperm.val, ApermRowPtr.data(), ApermColIdx.data());
    CSCMatrix A_perm_rb_csc(n, n, nnz, Aperm.val, ApermColIdx.data(), ApermRowPtr.data());
    // ^ since the matrix is symmetric, CSC and CSR are interchangable.
    // COOMatrix<double> A_perm_rb_coo(n, n);
    // CSCMatrix<double> A_perm_rb_csc(n, n);
    // RandBLAS::sparse_data::conversions::csr_to_coo(A_perm_rb_csr, A_perm_rb_coo);
    // RandBLAS::sparse_data::conversions::coo_to_csc(A_perm_rb_coo, A_perm_rb_csc);

    sparse_matrix_t Aperm_mkl, G_mkl;
    sparse_matrix_t_from_SparseCSR(Aperm, Aperm_mkl);
    sparse_matrix_t_from_SparseCSR(G, G_mkl);
    CallableSpMat<CSCMatrix<double>> Aperm_callable{Aperm_mkl, n, &A_perm_rb_csc};
    CallableChoSolve N_callable{G_mkl, n};
    LaplacianPinv<CSCMatrix<double>> Lpinv(Aperm_callable, N_callable);
    
    // low-rank approx time!
    //      NOTE: REVD2 isn't quite like QB2; it doesn't have a block size.
    RandLAPACK::SYPS<double, DefaultRNG>  SYPS(1, 1, false, false);
    RandLAPACK::HQRQ<double>              Orth(false, false); 
    RandLAPACK::SYRF<double, DefaultRNG>  SYRF(SYPS, Orth, false, false);
    RandLAPACK::REVD2<double, DefaultRNG> NystromAlg(SYRF, 3, false);
    double silly_tol = 1e4;
    // ^ ensures we break after one iteration
    std::vector<double> V(n*k, 0.0);
    std::vector<double> eigvals(k, 0.0);
    RandBLAS::RNGState state{};
    int64_t k_ = k;
    NystromAlg.call(Lpinv, k_, silly_tol, V, eigvals, state);
    return 0;
}
