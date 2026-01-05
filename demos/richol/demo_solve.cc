

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
CSRMatrix<T> richol_pipeline(CSRMatrix<T> &A_csr, RNGState<RNG> state, bool diag_adjust ) {
    using spvec = richol::SparseVec<T, int64_t>;
    int64_t n = A_csr.n_rows;
    
    vector<spvec> A_csrlike_triu(n);
    richol::csrlike_from_csr(n, A_csr.rowptr, A_csr.colidxs, A_csr.vals, A_csrlike_triu, blas::Uplo::Upper);

    vector<spvec> C_csrlike_lower{};
    int64_t rank = richol::clb21_rand_cholesky(A_csrlike_triu, C_csrlike_lower, state, diag_adjust, (T)0.0);
    randblas_require(rank + 1 >= n);

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
std::pair<vector<T>,vector<T>> setup_pcg_vecs(std::string &datadir, vector<int64_t> &perm) {
    vector<T> b0 = richol::vector_from_matrix_market<T>(datadir + "/p_rgh_source_b.mtx");
    int64_t n = static_cast<int64_t>(b0.size());
    vector<T> b(n);
    for (int64_t i = 0; i < n; ++i) { b[perm[i]] = -b0[i]; }
    vector<T> x0 = richol::vector_from_matrix_market<T>(datadir + "/p_rgh_psi_initial_x^{n}.mtx");
    vector<T> x(n);
    for (int64_t i = 0; i < n; ++i) { x[perm[i]] = x0[i]; }
    return {x, b};
}

template <typename T>
COOMatrix<T> read_negative_M_matrix(std::string &datadir) {
    auto A_coo = richol::coo_from_matrix_market<T>(datadir + "/p_rgh_matrix_A.mtx");
    std::for_each(A_coo.vals, A_coo.vals + A_coo.nnz, [](T &a) { a*= -1; return; } );
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

template <typename spvec_t>
void lift_sddm2lap(std::vector<spvec_t> &S) {
    // S is an explicitly symmetric SDDM matrix of order n.
    // We overwrite it with the Laplacian lift of order n+1.
    using scalar_t  = typename spvec_t::scalar_t;
    using ordinal_t = typename spvec_t::ordinal_t;
    auto n = static_cast<ordinal_t>(S.size());
    std::vector<scalar_t> d(n, 0);
    S.resize(n+1);
    scalar_t sum_d = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (const auto &[val, ind] : S[i].data ) {
            d[i] += val;
        }
        if (d[i] <= 0) {
            // < 0 should only happen due to rounding errors.
            continue;
        }
        S[i].push_back(n, -d[i]);
        S[n].push_back(i, -d[i]);
        sum_d += d[i];
    }
    S[n].push_back(n, sum_d);
    return;
}


int main(int argc, char** argv) {
    using T = double; // richol::dd;
    using spvec = richol::SparseVec<T, int64_t>;
    
    std::string datadir = "./";
    if (argc > 1) {
        datadir = argv[1];
    }
    int use_amd = 0;
    if (argc > 2) {
        use_amd = atoi(argv[2]);
    }

    bool use_lift = false;

    auto A_coo = read_negative_M_matrix<T>(datadir);
    auto A_unperm_csr = A_coo.as_owning_csr();
    int64_t n0 = A_coo.n_rows;
    vector<int64_t> perm( n0, 0 );
    std::iota( perm.data(), perm.data() + n0, 0 );
    if (use_amd) {
        richol::amd_permutation(A_unperm_csr, perm);
        A_coo.symperm_inplace(perm.data());
    };
    auto A_csr0 = A_coo.as_owning_csr();
    using csrlike_t = std::vector<spvec>;
    csrlike_t A0_csrlike_triu{};
    richol::csrlike_from_csr(A_csr0, A0_csrlike_triu, blas::Uplo::General);
    int64_t n = n0;
    if (use_lift) {
        lift_sddm2lap(A0_csrlike_triu);
        n += 1;
    }
    CSRMatrix<T> A_csr(n, n);
    richol::csr_from_csrlike(A0_csrlike_triu, A_csr);


    CallableSpMat A_callable{ &A_csr, A_csr.n_rows };


    int64_t max_iters = std::min((int64_t)500, n0);
    std::vector<T> x0{};
    std::vector<T> b{};
    try {
        auto [x0_temp, b_temp] = setup_pcg_vecs<T>(datadir, perm);
        x0 = x0_temp;
        b  = b_temp;
    } catch (std::exception e) {
        std::cerr << std::endl << e.what() << std::endl;
        std::cerr << "Generating dummy test vectors." << std::endl;
        for (int64_t i = 0; i < n0; ++i) {
            x0.push_back( (T) std::sqrt<T>(i)      );
            b.push_back(  (T)1 + (T)1 / (T)(1 + i) );
        }
    }
    
    if (int64_t(perm.size()) == n-1) {
        T sum_x0 = std::accumulate(x0.begin(), x0.end(), (T)0.0);
        T sum_b  = std::accumulate(b.begin(),   b.end(), (T)0.0);
        x0.push_back( -sum_x0 );
        b.push_back(  -sum_b  );
    }
    T pcg_tol = 0.0;

    {
        auto x = x0;
        std::cout << "\n=== Preconditioner: richol ===\n";

        auto richol_C_lower = richol_pipeline(A_csr, {0}, !use_lift);
        auto eyemat_C_lower = identity_as_csr<T>(n);
        auto inv_richol     = callable_chosolve( richol_C_lower );
        inv_richol.validate();
        inv_richol.project_out = use_lift;

        LaplacianPinv Lpinv(A_callable, inv_richol, pcg_tol, max_iters, false);
        Lpinv.L_callable.project_out = use_lift;
        
        TIMED_LINE(
        Lpinv(blas::Layout::ColMajor, 1, (T)1.0, b.data(), n, (T)0.0, x.data(), n), "Linear solve: ");
        std::cout << std::endl;
        log_residual_info(Lpinv, std::cout, "richol");
    }

    {
        auto x = x0;
        std::cout << "\n=== Preconditioner: ssor ===\n";

        auto A_coo_lifted = A_csr.as_owning_coo();
        auto ssor_C_lower = richol::ssor<CSRMatrix>(A_coo_lifted, blas::Uplo::Lower);
        auto inv_ssor    = callable_chosolve( ssor_C_lower );
        inv_ssor.validate();
        inv_ssor.project_out = use_lift;

        LaplacianPinv Lpinv(A_callable, inv_ssor, pcg_tol, max_iters, false);
        Lpinv.L_callable.project_out = use_lift;
        
        TIMED_LINE(
        Lpinv(blas::Layout::ColMajor, 1, (T)1.0, b.data(), n, (T)0.0, x.data(), n), "Linear solve: ");
        std::cout << std::endl;
        log_residual_info(Lpinv, std::cout, "ssor");
    }

    {
        auto x = x0;
        std::cout << "\n=== Preconditioner: identity ===\n";

        auto eyemat_C_lower = identity_as_csr<T>(n);
        auto inv_identity  = callable_chosolve( eyemat_C_lower );
        inv_identity.validate();
        inv_identity.project_out = use_lift;

        LaplacianPinv Lpinv(A_callable, inv_identity, pcg_tol, max_iters, false);
        Lpinv.L_callable.project_out = use_lift;
    
        TIMED_LINE(
        Lpinv(blas::Layout::ColMajor, 1, (T)1.0, b.data(), n, (T)0.0, x.data(), n), "Linear solve: ");
        std::cout << std::endl;
        log_residual_info(Lpinv, std::cout, "identity");
    }

    return 0;
}

