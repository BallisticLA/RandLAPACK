

#define FINE_GRAINED
#include "richol_core.hh"
#include "richol_linops.hh"

#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>

using RandBLAS::RNGState;
using RandBLAS::CSRMatrix;
using RandBLAS::CSCMatrix;
using RandBLAS::COOMatrix;

using std::vector;
using namespace richol::linops;


template <typename T, typename RNG = RandBLAS::DefaultRNG>
CSRMatrix<T> richol_pipeline(CSRMatrix<T> &A_csr, RNGState<RNG> state ) {
    int64_t n = A_csr.n_rows;
    
    using spvec = richol::SparseVec<T, int64_t>;
    vector<spvec> A_csrlike_triu(n);
    richol::sym_as_upper_tri_from_csr(n, A_csr.rowptr, A_csr.colidxs, A_csr.vals, A_csrlike_triu);

    vector<spvec> C_csrlike_lower{};
    bool diag_adjust = true;
    int64_t rank = richol::clb21_rand_cholesky(A_csrlike_triu, C_csrlike_lower, state, diag_adjust, (T)0.0);
    randblas_require(n == rank);

    CSRMatrix<T> C_lower(n, n);
    C_lower.reserve(nnz(C_csrlike_lower));
    csr_from_csrlike(C_csrlike_lower, C_lower.rowptr, C_lower.colidxs, C_lower.vals);
    return C_lower;
}

template <typename T>
CSRMatrix<T> identity_as_csr(int64_t n) {
    COOMatrix<T> coo(n, n);
    coo.reserve(n);
    std::fill(coo.vals, coo.vals + n, (T)1.0);
    std::iota(coo.rows, coo.rows + n, 0);
    std::iota(coo.cols, coo.cols + n, 0);
    return coo.as_owning_csr();
}

template <typename T>
vector<T> vector_from_matrix_market(std::string fn) {
    int64_t n_rows, n_cols = 0;
    vector<T> vals{};
    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_array(
        file_stream, n_rows, n_cols, vals, fast_matrix_market::col_major
    );
    return vals;
}

template <typename T>
std::pair<vector<T>,vector<T>> setup_pcg_vecs(std::string &datadir, vector<int64_t> &perm) {
    vector<T> b0 = vector_from_matrix_market<T>(datadir + "/p_rgh_source_b.mtx");
    int64_t n = static_cast<int64_t>(b0.size());
    vector<T> b(n);
    for (int64_t i = 0; i < n; ++i) { b[perm[i]] = -b0[i]; }
    vector<T> x0 = vector_from_matrix_market<T>(datadir + "/p_rgh_psi_initial_x^{n}.mtx");
    vector<T> x(n);
    for (int64_t i = 0; i < n; ++i) { x[perm[i]] = x0[i]; }
    return {x, b};
}

template <typename T>
COOMatrix<T> read_negative_M_matrix(std::string &datadir) {
    auto A_coo = richol::from_matrix_market<T>(datadir + "/p_rgh_matrix_A.mtx");
    blas::scal(A_coo.nnz, -1, A_coo.vals, 1);
    T reg = 0.0;
    bool offdiag_of_A_is_nonpos = true;
    for (int64_t i = 0; i < A_coo.nnz; ++i) {
        if (A_coo.rows[i] == A_coo.cols[i]) {
            A_coo.vals[i] += reg;
        } else if (offdiag_of_A_is_nonpos) {
            offdiag_of_A_is_nonpos = A_coo.vals[i] <= 0;
        }
    }
    randblas_require(offdiag_of_A_is_nonpos);
    return A_coo;
}

template <typename T>
auto callable_chosolve( CSRMatrix<T> &C_lower ) {
    CallableChoSolve<CSRMatrix<T>> invCCt_callable{ &C_lower, C_lower.n_rows };
    invCCt_callable.validate();
    invCCt_callable.project_out = false;
    return invCCt_callable;
}

template <typename LPINV_t>
void log_residual_info(LPINV_t &Lpinv, std::ostream &stream, const std::string &pc_name) {
    auto res_norms = Lpinv.pcg_res_norms;
    auto pre_norms = Lpinv.pcg_prec_res_norms;
    auto openfoams = Lpinv.openfoam_norms;
    randblas_require(res_norms.size() == pre_norms.size());
    auto actual_iters = static_cast<int64_t>(res_norms.size());
    std::string name_R  = "norm_R_"  + pc_name;
    std::string name_NR = "norm_NR_" + pc_name;
    std::string name_OF = "openfoam_" + pc_name;
    // RandBLAS::print_buff_to_stream(
    //     stream, blas::Layout::RowMajor, 1, actual_iters, res_norms.data(), actual_iters,
    //     name_R, 8, RandBLAS::ArrayStyle::Python
    // );
    // RandBLAS::print_buff_to_stream(
    //     stream, blas::Layout::RowMajor, 1, actual_iters, pre_norms.data(), actual_iters,
    //     name_NR, 8, RandBLAS::ArrayStyle::Python
    // );
    RandBLAS::print_buff_to_stream(
        stream, blas::Layout::RowMajor, 1, actual_iters, openfoams.data(), actual_iters,
        name_OF, 8, RandBLAS::ArrayStyle::Python
    );
}


int main(int argc, char** argv) {
    using T = double;
    
    std::string datadir = "./";
    if (argc > 1) {
        datadir = argv[1];
    }
    int use_amd = 0;
    if (argc > 2) {
        use_amd = atoi(argv[2]);
    }

    auto A_coo = read_negative_M_matrix<T>(datadir);
    auto A_unperm_csr = A_coo.as_owning_csr();
    int64_t n = A_coo.n_rows;
    vector<int64_t> perm( n, 0 );
    std::iota( perm.data(), perm.data() + n, 0 );
    if (use_amd) {
        richol::amd_permutation(A_unperm_csr, perm);
        A_coo.symperm_inplace(perm.data());
    };
    auto A_csr = A_coo.as_owning_csr();
    CallableSpMat A_callable{ &A_csr, A_csr.n_rows };

    auto richol_C_lower = richol_pipeline(A_csr, {0});
    auto dichol_C_lower = richol::dichol<CSRMatrix>(A_coo, blas::Uplo::Lower);
    auto eyemat_C_lower = identity_as_csr<T>(n);
    auto inv_richol    = callable_chosolve( richol_C_lower );
    inv_richol.validate();
    auto inv_dichol    = callable_chosolve( dichol_C_lower );
    inv_dichol.validate();
    auto inv_identity  = callable_chosolve( eyemat_C_lower );
    inv_identity.validate();


    int64_t max_iters = 400;

    auto [x0, b] = setup_pcg_vecs<T>(datadir, perm);
    T pcg_tol = 0.0;

    {
        auto x = x0;
        std::cout << "\n=== Preconditioner: richol ===\n";
        LaplacianPinv Lpinv(A_callable, inv_richol, pcg_tol, max_iters, false);
        
        TIMED_LINE(
        Lpinv(blas::Layout::ColMajor, 1, 1.0, b.data(), n, 0.0, x.data(), n), "Linear solve: ");
        std::cout << std::endl;
        log_residual_info(Lpinv, std::cout, "richol");
    }

    {
        auto x = x0;
        std::cout << "\n=== Preconditioner: dichol ===\n";
        LaplacianPinv Lpinv(A_callable, inv_dichol, pcg_tol, max_iters, false);
        
        TIMED_LINE(
        Lpinv(blas::Layout::ColMajor, 1, 1.0, b.data(), n, 0.0, x.data(), n), "Linear solve: ");
        std::cout << std::endl;
        log_residual_info(Lpinv, std::cout, "dichol");
    }

    {
        auto x = x0;
        std::cout << "\n=== Preconditioner: identity ===\n";
        LaplacianPinv Lpinv(A_callable, inv_identity, pcg_tol, max_iters, false);
        
        TIMED_LINE(
        Lpinv(blas::Layout::ColMajor, 1, 1.0, b.data(), n, 0.0, x.data(), n), "Linear solve: ");
        std::cout << std::endl;
        log_residual_info(Lpinv, std::cout, "identity");
    }

    return 0;
}

