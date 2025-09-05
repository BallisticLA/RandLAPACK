

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


int main(int argc, char** argv) {
    using T = double;
    using spvec = richol::SparseVec<T, int64_t>;

    // This example has eight steps.

    // Step 1. Read in the SDD matrix "A"
    std::string fn_smaller("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/tiny10x10_sdd.mtx");
    std::string fn_bigger("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/A_CG_schur.mm");
    auto A_unperm_coo = richol::from_matrix_market<T>(fn_smaller);
    int64_t n0 = A_unperm_coo.n_rows;

    // Step 2. Compute AMD ordering
    CSRMatrix<T> A_unperm_csr(n0, n0);
    coo_to_csr(A_unperm_coo, A_unperm_csr);
    std::vector<int64_t> perm(n0, 0);
    std::iota(perm.data(), perm.data() + n0, 1);
    richol::amd_permutation(A_unperm_csr, perm);
    CSRMatrix<T> A_csr(n0, n0);
    richol::permuted(A_unperm_csr, perm, A_csr);

    // Step 3. Compute the Gremban expansion
    std::vector<spvec> A_csrlike_triu(n0);
    richol::sym_as_upper_tri_from_csr(n0, A_csr.rowptr, A_csr.colidxs, A_csr.vals, A_csrlike_triu);
    auto G_csrlike_triu = richol::lift_sdd2sddm(A_csrlike_triu, false);
    //
    //  sym(G_csrlike_triu) = [  M_csrlike, -P_csrlike ]
    //                        [ -P_csrlike,  M_csrlike ]
    //

    // Step 4. Compute the incomplete Cholesky decomposition
    RandBLAS::RNGState s_richol(1997);
    std::vector<spvec> C_csrlike_lower;
    bool diag_adjust = true;
    int64_t rank = richol::clb21_rand_cholesky(G_csrlike_triu, C_csrlike_lower, s_richol, diag_adjust);

    // Step 5. Get a callable linear operator representation of inv(CC').
    int64_t n = 2*n0;
    int64_t nnz_C = 0;
    for (const auto &row : C_csrlike_lower)
        nnz_C += static_cast<int64_t>(row.size());
    CSRMatrix<T> C_lower(n, n);
    reserve_csr(nnz_C, C_lower);
    csr_from_csrlike(C_csrlike_lower, C_lower.rowptr, C_lower.colidxs, C_lower.vals);
    trsm_matrix_validation(C_lower, Uplo::Lower, Diag::NonUnit, 3);
    CallableChoSolve<decltype(C_lower)> invCCt_callable{&C_lower, n};

    std::ofstream C_filename("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/build/C.mtx");
    richol::write_square_matrix_market(C_csrlike_lower, C_filename, fast_matrix_market::general, "Approximate Cholesky factor");



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
    auto G_csrlike = richol::lift_sdd2sddm(A_csrlike_triu, true);
    int64_t nnz_G = 0;
    for (const auto &row : G_csrlike)
        nnz_G += static_cast<int64_t>(row.size());
    CSRMatrix<T> G(n, n);
    reserve_csr(nnz_G, G);
    csr_from_csrlike(G_csrlike, G.rowptr, G.colidxs, G.vals);
    CallableSpMat<decltype(G)> G_callable{&G, n};

    std::ofstream G_filename("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/build/G_explicit.mtx");
    richol::write_square_matrix_market(G_csrlike, G_filename, fast_matrix_market::symmetric, "Explicit Gremban expansion of A");


    // Step 7. Setup PCG.
    std::vector<T> b(n);
    RandBLAS::RNGState state2(1997);
    RandBLAS::DenseDist D(n, 1);
    RandBLAS::fill_dense(D, b.data(), state2);
    std::vector<T> x(n, 0.0);
    LaplacianPinv Lpinv(G_callable, invCCt_callable, 1e-10, 200, true);

    // Step 8. Run PCG.
    TIMED_LINE(
    Lpinv(blas::Layout::ColMajor, 1, 1.0, b.data(), n, 0.0, x.data(), n), "Linear solve: ");

    return 0;
}