
#include "richol_core.hh"

#include <iomanip>
#include <iostream>
#include <chrono>

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

using RandBLAS::CSRMatrix;
using RandBLAS::sparse_data::reserve_csr;
using RandBLAS::sparse_data::reserve_coo;
using RandBLAS::CSCMatrix;
using RandBLAS::COOMatrix;
using RandBLAS::sparse_data::conversions::coo_to_csr;


void small_3dlap() {
    using T = double;
    int64_t n = 8;
    std::vector<T> vals{6, -1, -1, -1, -1, 6, -1, -1, -1, 6, -1, -1, -1, -1, 6, -1, -1, 6, -1, -1, -1, -1, 6, -1, -1, -1, 6, -1, -1, -1, -1, 6};
    std::vector<int64_t> rowptr{0, 4, 8, 12, 16, 20, 24, 28, 32};
    std::vector<int64_t> colidxs{0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 2, 4, 6, 7, 3, 5, 6, 7};

    using spvec = richol::SparseVec<T, int64_t>;
    std::vector<spvec> sym;
    richol::sym_as_upper_tri_from_csr(n, rowptr.data(), colidxs.data(), vals.data(), sym);
    std::ofstream fl("L.mtx");
    richol::write_square_matrix_market(sym, fl, richol::symmetry_type::symmetric, "Laplacian");

    std::vector<spvec> C;
    auto k = richol::full_cholesky(sym, C);
    std::cout << "Exited with C of rank k = " << k << std::endl;
    std::ofstream fc("C.mtx");
    richol::write_square_matrix_market(C, fc, richol::symmetry_type::general, "Exact Cholesky factor");

    std::vector<spvec> C_approx;
    RandBLAS::RNGState state(0);
    auto backup_state = state;
    k = richol::clb21_rand_cholesky(sym, C_approx, state, true);
    std::cout << "Exited with C_approx of rank k = " << k << std::endl;
    std::ofstream fa("C_approx.mtx");
    richol::write_square_matrix_market(C_approx, fa, richol::symmetry_type::general, "Approximate Cholesky factor");

    /*
    Question: how well does RChol's preconditioner (as defined by the lift to n+1 nodes) compare
    to the version computed above with Rob's diagonal trick?
    */
    state = backup_state;
    auto sym_lift = richol::lift_sddm2lap(sym);
    k = richol::clb21_rand_cholesky(sym_lift, C_approx, state, false);
    std::cout << "Exited with C_approx of rank k = " << k << std::endl;
    std::ofstream flc("C_lifted.mtx");
    richol::write_square_matrix_market(C_approx, flc, richol::symmetry_type::general, "Approximate Cholesky factor of lifted matrix.");
    return;
}



int main(int argc, char** argv) {
    using T = double;

    std::string fn("/Users/rjmurr/Documents/randnla/RandLAPACK/demos/sparse-data-matrices/EY/G_10k_4.mtx");
    auto csr = richol::laplacian_from_matrix_market(fn, (T)1e-8);
    int64_t n = csr.n_rows;
    std::vector<int64_t> perm(n);
    for (int64_t i = 0; i < n; ++i)
        perm[i] = i;
    CSRMatrix<double, int64_t> L(n, n);
    TIMED_LINE(
    //richol::amd_permutation(csr, perm);
    richol::permuted(csr, perm, L);, "AMD reordering      : "
    );

    using spvec = richol::SparseVec<T, int64_t>;
    std::vector<spvec> sym;

    TIMED_LINE(
    richol::sym_as_upper_tri_from_csr(L.n_rows, L.rowptr, L.colidxs, L.vals, sym), "sym_as_upper_tri    : "
    );

    std::vector<spvec> C;
    RandBLAS::RNGState state(0);
    int64_t k;
    TIMED_LINE(
    k = richol::clb21_rand_cholesky(sym, C, state, true, (T)0.0), "SparseCholesky: "
    );
    std::cout << "Exited with C of rank k = " << k << std::endl;

    // template <typename spvec_t, typename tol_t = typename spvec_t::scalar_t>
    // void write_square_matrix_market(
    //     std::vector<spvec_t> &csr_like, std::ostream &os, symmetry_type symtype, std::string comment = {}, tol_t tol = 0.0
    // ) {
    std::ofstream fnc("C.mtx");
    richol::write_square_matrix_market(C, fnc, fast_matrix_market::general, "approximate Cholesky factor");


    return 0;
}
