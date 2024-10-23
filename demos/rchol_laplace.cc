#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>

#include "sparse.hpp"
// ^ defines CSR sparse matrix type
#include "rchol.hpp"
#include "util.hpp"
#include "pcg.hpp"
// ^ includes mkl_spblas.h

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"


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
    mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, N, N, pointerB, pointerE, update_intend_rpt, update_intend_datapt);
}


struct CallableSpMat {
    sparse_matrix_t A;
    int64_t m;
    matrix_descr des{SPARSE_MATRIX_TYPE_GENERAL};
    std::vector<double> regs{0.0};

    /*  C =: alpha * A * B + beta * C, where C and B have "n" columns. */
    void operator()(
        Layout layout, int64_t n, 
        double alpha, double* const B, int64_t ldb,
        double beta,  double* const C, int64_t ldc
    ) {
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, des,
            mkl_layout, B, n, ldb, beta, C, ldc
        );
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
        double alpha, double* const B, int64_t ldb,
        double beta, double* const C, int64_t ldc
    ) {
        if (work == nullptr) {
            work = new double[m*n];
            n_work = n;
        } else {
            randblas_require(n_work >= n);
        }
        randblas_require(beta == (double) 0.0);
        // TRSM, then transposed TRSM.
        auto mkl_layout = (layout == Layout::ColMajor) ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR;
        mkl_sparse_d_trsm(SPARSE_OPERATION_TRANSPOSE    , alpha, G, des, mkl_layout, B, n, ldb, work, m);
        mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE,   1.0, G, des, mkl_layout, work, n, m, C, ldc);
    }

    ~CallableChoSolve() {
        if (work != nullptr) 
            delete [] work;
    }
};



int main(int argc, char *argv[]) {
    int n = 3; // DoF in every dimension
    int threads = 1;
    for (int i=0; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
        n = atoi(argv[i+1]);
    if (!strcmp(argv[i], "-t"))
        threads = atoi(argv[i+1]);
    }
    std::cout<<std::setprecision(3);

    // SDDM matrix from 3D constant Poisson equation
    SparseCSR A;
    A = laplace_3d(n); // n x n x n grid

    // random RHS
    int N = A.size();
    std::vector<double> b(N); 
    rand(b);

    // compute preconditioner (single thread) and solve 
    SparseCSR G;
    rchol(A, G);
    std::cout << "Fill-in ratio: " << 2.*G.nnz()/A.nnz() << std::endl;

    // solve with PCG
    double tol = 1e-6;
    int maxit = 200;
    double relres;
    int itr;
    std::vector<double> x;
    pcg(A, b, tol, maxit, G, x, relres, itr);
    std::cout << "# CG iterations: " << itr << std::endl;
    std::cout << "Relative residual: " << relres << std::endl;

    std::cout << "----------- start RandLAPACK PCG -----------------" << std::endl;
    sparse_matrix_t A_mkl, G_mkl;
    sparse_matrix_t_from_SparseCSR(A, A_mkl);
    sparse_matrix_t_from_SparseCSR(G, G_mkl);
    CallableSpMat    A_callable{A_mkl, N};
    CallableChoSolve N_callable{G_mkl, N};
    auto seminorm = [](int64_t n, int64_t s, const double* NR){return blas::nrm2(n*s, NR, 1);};
    std::vector<double> x_randlapack(N, 0.0);
    RandLAPACK::lockorblock_pcg(A_callable, b, 1e-8, 2*itr, N_callable, seminorm, x_randlapack, true);
    std::cout << "------------- end RandLAPACK PCG -----------------" << std::endl;    
    return 0;
}
