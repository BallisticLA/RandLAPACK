

#define FINE_GRAINED
#include "richol_core.hh"
#include "richol_mkl.hh"

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

template <typename CSR_t = CSRMatrix<double, int64_t>>
void amd_permutation(const CSR_t &A, std::vector<int64_t> &perm) {
    perm.resize(A.n_rows);
    int64_t result;
    double Control [AMD_CONTROL], Info [AMD_INFO];
    amd_l_defaults (Control) ;
    amd_l_control  (Control) ;
    result = amd_l_order (A.n_rows, A.rowptr, A.colidxs, perm.data(), Control, Info) ;
    printf ("return value from amd_order: %ld (should be %d)\n", result, AMD_OK) ;
    if (result != AMD_OK) {
        printf ("AMD failed\n") ;
        exit (1) ;
    }
    return;
}


int main(int argc, char** argv) {
    using T = double;

    std::string fn("/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/uk/uk.mtx");
    auto L = richol::laplacian_from_matrix_market(fn, (T)0.0);
    int64_t n = L.n_rows;

    bool use_amd_perm = true;
    std::vector<int64_t> perm(n);
    for (int64_t i = 0; i < n; ++i)
        perm[i] = i;
    CSRMatrix<T, int64_t> Lperm(n, n);
    TIMED_LINE(
    if (use_amd_perm) amd_permutation(L, perm);
    richol::permuted(L, perm, Lperm);, "AMD reordering      : ");
    using spvec = richol::SparseVec<T, int64_t>;
    std::vector<spvec> sym;
    TIMED_LINE(
    richol::sym_as_upper_tri_from_csr(Lperm.n_rows, Lperm.rowptr, Lperm.colidxs, Lperm.vals, sym), "sym_as_upper_tri    : ");

    std::vector<spvec> C;
    RandBLAS::RNGState state(0);
    int64_t rank;
    TIMED_LINE(
    rank = richol::clb21_rand_cholesky(sym, C, state, true, (T)0.0), "SparseCholesky: ");
    std::cout << "Exited with C of rank k = " << rank << std::endl;
    int64_t nnz_G = 0;
    for (const auto &row : C) 
        nnz_G += static_cast<int64_t>(row.size());
    CSRMatrix<T,int64_t> G(n, n);
    reserve_csr(nnz_G ,G);
    csr_from_csrlike(C, G.rowptr, G.colidxs, G.vals);

    sparse_matrix_t Lperm_mkl, G_mkl;
    sparse_matrix_t_from_randblas_csr(Lperm, Lperm_mkl);
    sparse_matrix_t_from_randblas_csr(G, G_mkl);
    CallableSpMat Aperm_callable{Lperm_mkl, n};
    CallableChoSolve N_callable{G_mkl, n};
    LaplacianPinv Lpinv(Aperm_callable, N_callable, 1e-10, 200, true);

    //      NOTE: REVD2 isn't quite like QB2; it doesn't have a block size.
    RandLAPACK::SYPS<T, DefaultRNG>  SYPS(3, 1, false, false);
    RandLAPACK::HQRQ<T>              Orth(false, false); 
    RandLAPACK::SYRF<T, DefaultRNG>  SYRF(SYPS, Orth, false, false);
    RandLAPACK::REVD2<T, DefaultRNG> NystromAlg(SYRF, 1, false);
    T silly_tol = 1e4;
    int64_t k = 8;
    // ^ ensures we break after one iteration
    std::vector<T> V(n*k, 0.0);
    std::vector<T> eigvals(k, 0.0);
    RandBLAS::RNGState state_nys(1997);
    int64_t k_ = k;
    TIMED_LINE(
    NystromAlg.call(Lpinv, k_, silly_tol, V, eigvals, state_nys), "NystromAlg.call -  rchol_pcg : ");


    mkl_sparse_destroy(Lperm_mkl);
    mkl_sparse_destroy(G_mkl);
    return 0;
}
