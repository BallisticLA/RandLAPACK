

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


template <typename T>
std::vector<T> vector_from_matrix_market(std::string fn) {
    int64_t n_rows, n_cols = 0;
    std::vector<T> vals{};
    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_array(
        file_stream, n_rows, n_cols, vals, fast_matrix_market::col_major
    );
    return vals;
}


template <SparseMatrix SpMat, typename PrecondCallable>
struct LaplacianPinv2 {
    public:
    const int64_t dim;
    CallableSpMat<SpMat>    L_callable;
    PrecondCallable &N_callable;
    bool verbose_pcg;
    std::vector<double> work_B{};
    std::vector<double> work_C{};
    std::vector<double> work_seminorm{};
    std::vector<double> times{};
    double call_pcg_tol = 1e-10;
    int64_t max_iters = 100;
    using scalar_t = typename SpMat::scalar_t;
    const int64_t num_ops = 1;

    LaplacianPinv2(CallableSpMat<SpMat> &L, PrecondCallable &N, double pcg_tol, int maxit,
        bool verbose = false
    ) :
        dim(L.dim),
        L_callable{L.A, L.dim},
        N_callable(N),
        verbose_pcg(verbose),
        times(4, 0.0),
        call_pcg_tol(pcg_tol),
        max_iters((int64_t) maxit)
    {
        L_callable.project_out = L.project_out;
    }; 

    //  C =: alpha * inv(L) * B, where C and B have "n" columns,
    //  and PCG for applying inv(L) is initialized at C.
    //
    //  This has the same function signature as RandLAPACK's LinearOperator 
    //  interface, but the role of C is different.
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
        double *work_seminorm_ = work_seminorm.data();
        if (n < (int64_t) L_callable.regs.size()) {
            L_callable.regs.resize(n, 0.0);
        }
        std::vector<double> sn_log{};
        auto seminorm = [work_seminorm_, &sn_log](int64_t __n, int64_t __s, double* NR) {
            blas::copy(__n*__s, NR, 1, work_seminorm_, 1);
            double out = blas::nrm2(__n*__s, work_seminorm_, 1);
            sn_log.push_back(out);
            return out;
        };
        for (int64_t i = 0; i < n_x_dim; ++i)
            work_C[i] = C[i]; // don't multiply by beta!
        for (int64_t i = 0; i < n_x_dim; ++i)
            work_B[i] = alpha * B[i];
        // logging
        std::cout << std::left << std::setw(10) << "iters" << std::setw(15) << "relres" << std::endl;
        // work
        auto t0 = std_clock::now();
        RandLAPACK::pcg(L_callable, work_B.data(), n, seminorm, call_pcg_tol, max_iters, N_callable, work_C.data(), verbose_pcg);
        auto t1 = std_clock::now();
        // logging
        auto total_spmm   = std::reduce(L_callable.times.begin(), L_callable.times.end());
        auto total_sptrsm = std::reduce(N_callable.times.begin(), N_callable.times.end());
        times[0] += total_spmm;
        times[1] += total_sptrsm;
        L_callable.times.clear();
        N_callable.times.clear();
        times[2] += seconds_elapsed(t0, t1);
        times[3] += (double) sn_log.size()/2;
        std::cout << std::left 
        << std::setw(10) << static_cast<int64_t>(sn_log.size()/2) - 1
        << std::setw(15) << sn_log[sn_log.size()-2] / sn_log[0] << std::endl;
        blas::copy(n_x_dim, work_C.data(), 1, C, 1);
    }

    double operator()(int64_t i, int64_t j) {
        UNUSED(i); UNUSED(j);
        randblas_require(false);
        return 0.0;
    }
};



int main(int argc, char** argv) {
    using T = double;
    using spvec = richol::SparseVec<T, int64_t>;

    // Step 1. Read in the SDD matrix "A"
    std::string datadir = "./";
    if (argc > 1) {
        datadir = argv[1];
    }
    auto A_unperm_coo = richol::from_matrix_market<T>(datadir + "/p_rgh_matrix_A.mtx");
    blas::scal(A_unperm_coo.nnz, -1, A_unperm_coo.vals, 1);
    T reg = 0.0;
    bool offdiag_of_A_is_nonpos = true;
    for (int64_t i = 0; i < A_unperm_coo.nnz; ++i) {
        if (A_unperm_coo.rows[i] == A_unperm_coo.cols[i]) {
            A_unperm_coo.vals[i] += reg;
        } else if (offdiag_of_A_is_nonpos) {
            offdiag_of_A_is_nonpos = A_unperm_coo.vals[i] <= 0;
        }
    }
    int64_t n0 = A_unperm_coo.n_rows;

    // Step 2. Compute AMD ordering
    CSRMatrix<T> A_unperm_csr(n0, n0);
    coo_to_csr(A_unperm_coo, A_unperm_csr);
    std::vector<int64_t> perm(n0, 0);
    std::iota(perm.data(), perm.data() + n0, 0);
    //richol::amd_permutation(A_unperm_csr, perm);
    CSRMatrix<T> A_csr(n0, n0);
    richol::permuted(A_unperm_csr, perm, A_csr);

    std::cout << std::endl << "SDD is M-matrix : " << offdiag_of_A_is_nonpos << std::endl;

    // Step 3. Compute the Gremban expansion
    std::vector<spvec> A_csrlike_triu(n0);
    richol::sym_as_upper_tri_from_csr(n0, A_csr.rowptr, A_csr.colidxs, A_csr.vals, A_csrlike_triu);
    std::vector<spvec> G_csrlike_triu{};
    if (offdiag_of_A_is_nonpos) {
        G_csrlike_triu = A_csrlike_triu;
    } else {
        G_csrlike_triu = richol::lift_sdd2sddm(A_csrlike_triu, false);
        //
        //  sym(G_csrlike_triu) = [  M_csrlike, -P_csrlike ]
        //                        [ -P_csrlike,  M_csrlike ]
        //
    }
    int64_t n = static_cast<int64_t>(G_csrlike_triu.size());

    // Step 4. Compute the incomplete Cholesky decomposition
    RandBLAS::RNGState s_richol(0);
    std::vector<spvec> C_csrlike_lower{};
    bool diag_adjust = true;
    int64_t rank = richol::clb21_rand_cholesky(G_csrlike_triu, C_csrlike_lower, s_richol, diag_adjust, (T)0.0);
    std::cout << "dim : " << n << std::endl;
    std::cout << "rank : " << rank << std::endl << std::endl;

    // Step 5. Get a callable linear operator representation of inv(CC').
    std::ofstream C_filename("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/build/C.mtx");
    richol::write_square_matrix_market(C_csrlike_lower, C_filename, fast_matrix_market::general, "Approximate Cholesky factor");
    int64_t nnz_C = 0;
    for (const auto &row : C_csrlike_lower)
        nnz_C += static_cast<int64_t>(row.size());
    CSRMatrix<T> C_lower(n, n);
    reserve_csr(nnz_C, C_lower);
    csr_from_csrlike(C_csrlike_lower, C_lower.rowptr, C_lower.colidxs, C_lower.vals);
    trsm_matrix_validation(C_lower, Uplo::Lower, Diag::NonUnit, 3);
    CallableChoSolve<decltype(C_lower)> invCCt_callable{&C_lower, n};
    invCCt_callable.project_out = false;


    // Step 6. Get a callable linear operator representation of G.
    //
    //  Right now this is implemented by calling lift_sdd2sddm with different argument
    //  so the output is explicitly symmetric, then passing that to csr_from_csrlike
    //  to get a RandBLAS sparse matrix, then using that to make a CallableSpMat.
    //
    //  It would be preferable to define a new callable linear operator that only needed
    //  the terms M and P from the Gremban expansion. We can get the csrlike representations
    //  of these with the line
    //
    //      auto [M_csrlike, P_csrlike] = richol::split_and_sym_sdd(A_csrlike_triu);
    //
    std::vector<spvec> G_csrlike;
    if (offdiag_of_A_is_nonpos) {
        G_csrlike = G_csrlike_triu;
        int64_t row_ind = 0;
        for (const auto &row : G_csrlike_triu) {
            for (const auto &[vk, k] : row.data) {
                if (k != row_ind) {
                    G_csrlike[k].data.insert(G_csrlike[k].data.begin(), {vk, row_ind});
                }
            }
            row_ind += 1;
        }
    } else {
        G_csrlike = richol::lift_sdd2sddm(A_csrlike_triu, true);
    }
    int64_t nnz_G = 0;
    for (const auto &row : G_csrlike)
        nnz_G += static_cast<int64_t>(row.size());
    CSRMatrix<T> G(n, n);
    reserve_csr(nnz_G, G);
    csr_from_csrlike(G_csrlike, G.rowptr, G.colidxs, G.vals);
    CallableSpMat<decltype(G)> G_callable{&G, n};
    G_callable.project_out = false;

    std::ofstream G_filename("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/build/G_explicit.mtx");
    richol::write_square_matrix_market(G_csrlike, G_filename, fast_matrix_market::general, "I get fed into PCG.");


    // Step 7. Setup PCG.
    // std::vector<T> b(n);
    // RandBLAS::RNGState state2(1997);
    // RandBLAS::DenseDist D(n, 1);
    // RandBLAS::fill_dense(D, b.data(), state2);
    int64_t block_size = 1;
    std::vector<T> b0 = vector_from_matrix_market<T>(datadir + "/p_rgh_source_b.mtx");
    std::vector<T> b(n);
    for (int64_t i = 0; i < n; ++i) {
        b[perm[i]] = -b0[i];
    }
    // b.resize(block_size * n);
    // RandBLAS::RNGState state2(1997);
    // RandBLAS::DenseDist D(n, block_size);
    // RandBLAS::fill_dense(D, b.data(), state2);
    // blas::scal(n*block_size, (T)1.0 / blas::nrm2(n, b.data(), 1), b.data(), 1);
    // std::vector<T> x(block_size * n, (T)0.0);
    std::vector<T> x0 = vector_from_matrix_market<T>(datadir + "/p_rgh_psi_initial_x^{n}.mtx");
    std::vector<T> x(n);
    for (int64_t i = 0; i < n; ++i) {
        x[perm[i]] = x0[i];
    }

    struct trivial_precond {
        int64_t num_ops = 1;
        const int64_t dim;
        std::vector<double> times{};
        trivial_precond(int64_t _n) : dim(_n), times(4,(T)0.0) { }
        void operator()(blas::Layout ell, int64_t _n, T alpha, T* const _B, int64_t _ldb, T beta, T* _C, int64_t _ldc) {
            randblas_require(ell == blas::Layout::ColMajor);
            UNUSED(_ldb); UNUSED(_ldc);
            blas::scal(dim*_n, beta, _C, 1);
            blas::axpy(dim*_n, alpha, _B, 1, _C, 1);
        };
    };
    trivial_precond tp(n);
    int64_t max_iters = 264;
    LaplacianPinv2 Lpinv(G_callable, invCCt_callable, 1e-10, max_iters, true);

    // Step 8. Run PCG.
    TIMED_LINE(
    Lpinv(blas::Layout::ColMajor, block_size, 1.0, b.data(), n, 0.0, x.data(), n), "Linear solve: ");

    return 0;
}
