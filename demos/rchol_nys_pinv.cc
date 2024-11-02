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


COOMatrix<double> laplacian_from_matrix_market(std::string fn, SparseCSR &A, std::vector<double> &A_densevec) {

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
    std::vector<double> diagvec(n, 0.0);
    auto diag = diagvec.data();
    // Initialize the dense matrix
    A_densevec.resize(n * n);
    auto A_dense = A_densevec.data();
    RandBLAS::util::safe_scal(n * n, 0.0, A_dense, 1);
    // Fill the COO matrix and dense matrix
    for (int64_t i = 0; i < m; ++i) {
        coo.rows[i] = rows[i];
        coo.cols[i] = cols[i];
        double v = vals[i];
        randblas_require(v >= 0);
        coo.vals[i] = -v;
        diag[rows[i]] += v;
        A_dense[rows[i] * n + cols[i]] = -v;
    }
    for (int64_t i = 0; i < n; ++i) {
        A_dense[i * n + i] += diag[i];
    }
    // Debugging: Print diagonal values for verification
    for (int64_t i = 0; i < n; ++i) {
        std::cout << "Diagonal A_dense[" << i << "] = " << A_dense[i * n + i] 
                << ", diag[" << i << "] = " << diag[i] << std::endl;
        coo.vals[m+i] = diag[i];
        coo.rows[m+i] = i;
        coo.cols[m+i] = i;
    }
    // convert COO format Laplacian to CSR format, using RandBLAS.
    CSRMatrix<double> csr(n, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);

    // Convert RandBLAS::CSRMatrix to SparseCSR
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
    std::cout << "Hello!\n";
    return coo;
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
std::vector<double> extract_diagonal(int64_t n, index_t* rowptr, index_t* colidxs, double* vals) {
    /***
    
    This is messed up. For an upper-triangular matrix in CSR format the diagonal entries should 
    just be vals[rowptr[i]] for i = 0,...,n-1.

    This function is saying that we get zeros after every element other than the first, even
    on the tiny matrix
     /home/rjmurr/laps/RandLAPACK/demos/sparse-data-matrices/chesapeake/chesapeake.mtx
    which I'm pretty sure was running fine just a little bit ago.
    
     */
    index_t n_ = static_cast<index_t>(n);
    int64_t nnz = static_cast<int64_t>(rowptr[n]);
    std::vector<double> out(n_, 0.0);
    for (index_t i = 0; i < n_; ++i) {
        for (index_t ell = rowptr[i]; ell < rowptr[i+1]; ++ell) {
            if (colidxs[ell] == i) {
                out[i] = vals[ell];
            } // else, don't care.
        }
        if (std::abs(out[i]) < 1e-16) {
            std::cout << "G[i,i] = 0 at index i = " << i << std::endl;
        }
    }
    return out;
}


int main(int argc, char *argv[]) {
    std::cout << std::setprecision(3);

    /**
    
    TODO: verify that for the default graph we're running on, the 
    "Cholesky factor" ends up having a zero on the diagonal. Seems
    that not being strongly connected is creating problems.(?)

     */

    auto [fn, k, threads] = parse_args(argc, argv);
    SparseCSR G, Aperm;
    std::vector<double> Adensevec{};
    SparseCSR A;
    auto A_rb_coo = laplacian_from_matrix_market(fn, A, Adensevec);
    int64_t n = A.size();
    RandBLAS::print_buff_to_stream(
        std::cout, Layout::ColMajor, n, n, Adensevec.data(), n, "A_dense_pre", 2
    );
    // auto A = laplace_3d(10); // n x n x n grid
    std::vector<size_t> P;
    //rchol(A, G, P, threads);
    //reorder(A, P, Aperm);
    rchol(A, G);
    Aperm = A;
    int64_t nnz = Aperm.nnz();
    int64_t nnz_G = G.nnz();
    std::vector<int64_t> ApermRowPtr(n+1);
    std::vector<int64_t> ApermColIdx(nnz);
    std::vector<int64_t> G_rowptr(n+1);
    std::vector<int64_t> G_colidxs(nnz_G);

    for (int64_t i = 0; i < nnz; ++i) 
        ApermColIdx[i] = static_cast<size_t>(Aperm.colIdx[i]);
    for (int64_t i = 0; i < nnz_G; ++i)
        G_colidxs[i] = static_cast<size_t>(G.colIdx[i]);
    for (int64_t i = 0; i < n+1; ++i) {
        ApermRowPtr[i] = static_cast<size_t>(Aperm.rowPtr[i]);
        G_rowptr[i]    = static_cast<size_t>(G.rowPtr[i]);
    }

    CSRMatrix A_perm_rb_csr(n, n, nnz,   Aperm.val, ApermRowPtr.data(), ApermColIdx.data());
    CSCMatrix A_perm_rb_csc(n, n, nnz,   Aperm.val, ApermColIdx.data(), ApermRowPtr.data());
    COOMatrix<double> A_perm_rb_coo(n, n);
    RandBLAS::sparse_data::conversions::csr_to_coo(A_perm_rb_csr, A_perm_rb_coo);
    std::vector<double> A_dense(n*n, 0.0);
    RandBLAS::sparse_data::coo::coo_to_dense(A_rb_coo, Layout::ColMajor, A_dense.data());
    RandBLAS::print_buff_to_stream(
        std::cout, Layout::ColMajor, n, n, A_dense.data(), n, "A_dense_post", 2
    );


    auto diag_G = extract_diagonal(n, G.rowPtr, G.colIdx, G.val);
    double min_diag = *std::min_element(diag_G.begin(), diag_G.end()); 
    std::cout << "Min element of diag(G): " << min_diag << std::endl;

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
