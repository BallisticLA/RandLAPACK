

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

    std::string fn("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/EY/G_10k_2.mtx");
    auto L = richol::laplacian_from_matrix_market(fn, (T)1e-6);
    int64_t n = L.n_rows;

    bool use_amd_perm = true;
    std::vector<int64_t> perm(n);
    for (int64_t i = 0; i < n; ++i)
        perm[i] = i;
    CSRMatrix<T, int64_t> Lperm(n, n);
    TIMED_LINE(
    if (use_amd_perm) richol::amd_permutation(L, perm);
    richol::permuted(L, perm, Lperm);, "AMD reordering      : ");
    using spvec = richol::SparseVec<T, int64_t>;
    std::vector<spvec> sym;
    std::vector<spvec> C;
    RandBLAS::RNGState state(0);
    int64_t rank;
    TIMED_LINE(
    richol::csrlike_from_csr(Lperm.n_rows, Lperm.rowptr, Lperm.colidxs, Lperm.vals, sym, blas::Uplo::Upper);
    bool allow_strict_sdd = true;
    rank = richol::clb21_rand_cholesky(sym, C, state, allow_strict_sdd, (T)0.0), "SparseCholesky: ");
    std::cout << "Exited with C of rank k = " << rank << std::endl;
    int64_t nnz_G = 0;
    for (const auto &row : C) 
        nnz_G += static_cast<int64_t>(row.size());
    CSRMatrix<T,int64_t> G(n, n);
    reserve_csr(nnz_G ,G);
    csr_from_csrlike(C, G.rowptr, G.colidxs, G.vals);

    CallableSpMat<decltype(Lperm)> Aperm_callable{&Lperm, n};
    Aperm_callable.project_out = true;
    trsm_matrix_validation(G, Uplo::Lower, Diag::NonUnit, 3);
    CallableChoSolve<decltype(G)>  N_callable{&G, n};
    N_callable.project_out = true;
    LaplacianPinv Lpinv(Aperm_callable, N_callable, 1e-10, 200, true);

    //      NOTE: REVD2 isn't quite like QB2; it doesn't have a block size.
    RandLAPACK::SYPS<T, DefaultRNG>  SYPS(3, 1, false, false);
    RandLAPACK::HQRQ<T> Orth(false, false); 
    RandLAPACK::SYRF  SYRF(SYPS, Orth, false, false);
    RandLAPACK::REVD2 NystromAlg(SYRF, 1, false);
    T silly_tol = 1e4;
    int64_t k = 8;
    // ^ ensures we break after one iteration
    std::vector<T> V(n*k, 0.0);
    std::vector<T> eigvals(k, 0.0);
    RandBLAS::RNGState state_nys(1997);
    int64_t k_ = k;
    TIMED_LINE(
    NystromAlg.call(Lpinv, k_, silly_tol, V, eigvals, state_nys), "NystromAlg.call -  rchol_pcg : ");
    return 0;
}
