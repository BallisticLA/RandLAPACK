

#define FINE_GRAINED
#include "richol_core.hh"
#include "richol_linops.hh"

#include <iomanip>
#include <iostream>
#include <chrono>

using RandBLAS::CSRMatrix;
using RandBLAS::CSCMatrix;
using RandBLAS::COOMatrix;

using std::vector;
using namespace richol::linops;


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


int main(int argc, char** argv) {
    using T = double;
    using spvec = richol::SparseVec<T, int64_t>;

    // Step 1. Read in the SDD matrix "A"
    std::string datadir = "./";
    if (argc > 1) {
        datadir = argv[1];
    }
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
    int64_t n0 = A_coo.n_rows;

    // Step 2. Compute AMD ordering
    auto A_unperm_csr = A_coo.as_owning_csr();
    vector<int64_t> perm(n0, 0);
    std::iota(perm.data(), perm.data() + n0, 0);
    richol::amd_permutation(A_unperm_csr, perm);
    A_coo.symperm_inplace(perm.data());
    auto A_csr = A_coo.as_owning_csr();

    std::cout << std::endl << "SDD is M-matrix : " << offdiag_of_A_is_nonpos << std::endl;

    // Step 3. Compute the Gremban expansion
    vector<spvec> A_csrlike_triu(n0);
    richol::sym_as_upper_tri_from_csr(n0, A_csr.rowptr, A_csr.colidxs, A_csr.vals, A_csrlike_triu);
    vector<spvec> G_csrlike_triu{};
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
    vector<spvec> C_csrlike_lower{};
    bool diag_adjust = true;
    int64_t rank = richol::clb21_rand_cholesky(G_csrlike_triu, C_csrlike_lower, s_richol, diag_adjust, (T)0.0);
    std::cout << "dim : " << n << std::endl;
    std::cout << "rank : " << rank << std::endl << std::endl;

    // Step 5. Get a callable linear operator representation of inv(CC').
    std::ofstream C_filename("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/build/C.mtx");
    richol::write_square_matrix_market(C_csrlike_lower, C_filename, fast_matrix_market::general, "Approximate Cholesky factor");
    CSRMatrix<T> C_lower(n, n);
    C_lower.reserve(nnz(C_csrlike_lower));
    csr_from_csrlike(C_csrlike_lower, C_lower.rowptr, C_lower.colidxs, C_lower.vals);

    auto dichol_C_lower = richol::dichol<CSRMatrix>(A_coo, blas::Uplo::Lower);

    CallableChoSolve<decltype(C_lower)> invCCt_callable{&dichol_C_lower, n};
    invCCt_callable.validate();
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
    vector<spvec> G_csrlike;
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
    CSRMatrix<T> G(n, n);
    G.reserve(nnz(G_csrlike));
    csr_from_csrlike(G_csrlike, G.rowptr, G.colidxs, G.vals);
    CallableSpMat<decltype(G)> G_callable{&G, n};
    G_callable.project_out = false;

    std::ofstream G_filename("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/build/G_explicit.mtx");
    richol::write_square_matrix_market(G_csrlike, G_filename, fast_matrix_market::general, "I get fed into PCG.");


    // Step 7. Setup PCG.
    // vector<T> b(n);
    // RandBLAS::RNGState state2(1997);
    // RandBLAS::DenseDist D(n, 1);
    // RandBLAS::fill_dense(D, b.data(), state2);
    int64_t block_size = 1;
    vector<T> b0 = vector_from_matrix_market<T>(datadir + "/p_rgh_source_b.mtx");
    vector<T> b(n);
    for (int64_t i = 0; i < n; ++i) {
        b[perm[i]] = -b0[i];
    }
    // b.resize(block_size * n);
    // RandBLAS::RNGState state2(1997);
    // RandBLAS::DenseDist D(n, block_size);
    // RandBLAS::fill_dense(D, b.data(), state2);
    // blas::scal(n*block_size, (T)1.0 / blas::nrm2(n, b.data(), 1), b.data(), 1);
    // vector<T> x(block_size * n, (T)0.0);
    vector<T> x0 = vector_from_matrix_market<T>(datadir + "/p_rgh_psi_initial_x^{n}.mtx");
    vector<T> x(n);
    for (int64_t i = 0; i < n; ++i) {
        x[perm[i]] = x0[i];
    }

    IdentityMatrix<T> I(n);
    int64_t max_iters = 264;
    LaplacianPinv Lpinv(G_callable, invCCt_callable, 1e-10, max_iters, true);

    // Step 8. Run PCG.
    TIMED_LINE(
    Lpinv(blas::Layout::ColMajor, block_size, 1.0, b.data(), n, 0.0, x.data(), n), "Linear solve: ");

    return 0;
}
