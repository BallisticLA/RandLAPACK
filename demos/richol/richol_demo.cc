
#include "richol.hh"

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

template <typename spvec_t>
std::vector<spvec_t> lift2lap(const std::vector<spvec_t> &sym) {
    std::vector<spvec_t> sym_lift(sym);
    using scalar_t  = typename spvec_t::scalar_t;
    using ordinal_t = typename spvec_t::ordinal_t;
    auto n = static_cast<ordinal_t>(sym.size());
    // set d = sym * vector-of-ones.
    std::vector<scalar_t> d(n, 0);
    for (ordinal_t i = 0; i < n; ++i) {
        auto &vec = sym_lift[i];
        vec.coallesce((scalar_t) 0);
        for (const auto &[val, ind] : vec.data) {
            d[i] += val;
            if (ind != i) {
                d[ind] += val;
            }
        }
    }
    // extend sym_lift = [sym  ,   -d ]
    //                   [ -d' , sum(d) ]
    for (ordinal_t i = 0; i < n; ++i) {
        auto &vec = sym_lift[i];
        vec.push_back(n, -d[i]);
    }
    scalar_t sum_d = std::reduce(d.begin(), d.end());
    sym_lift.resize(n+1);
    sym_lift[n].push_back(sum_d, n);
    return sym_lift;
}

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
    auto sym_lift = lift2lap(sym);
    k = richol::clb21_rand_cholesky(sym_lift, C_approx, state, false);
    std::cout << "Exited with C_approx of rank k = " << k << std::endl;
    std::ofstream flc("C_lifted.mtx");
    richol::write_square_matrix_market(C_approx, flc, richol::symmetry_type::general, "Approximate Cholesky factor of lifted matrix.");
    return;
}


template <typename scalar_t, RandBLAS::SignedInteger sint_t = int64_t>
CSRMatrix<scalar_t, sint_t> laplacian_from_matrix_market(std::string fn, scalar_t reg) {
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
    std::vector<double> diagvec(n, reg);
    auto diag = diagvec.data();
    for (int64_t i = 0; i < m; ++i) {
        coo.rows[i] = rows[i];
        coo.cols[i] = cols[i];
        double v = vals[i];
        randblas_require(v >= 0);
        coo.vals[i] = -v;
        diag[rows[i]] += v;
    }
    for (int64_t i = 0; i < n; ++i) {
        coo.vals[m+i] = diag[i];
        coo.rows[m+i] = i;
        coo.cols[m+i] = i;
    }
    // convert COO format Laplacian to CSR format, using RandBLAS.
    CSRMatrix<double> csr(n, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
    return csr;
}

template <typename CSR_t = CSRMatrix<double, int64_t>>
void amd_permutation(const CSR_t &A, std::vector<int64_t> &perm) {
    perm.resize(A.n_rows);
    int64_t result;
    double Control [AMD_CONTROL], Info [AMD_INFO];
    amd_l_defaults (Control) ;
    amd_l_control  (Control) ;
    result = amd_l_order (A.n_rows, A.rowptr, A.colidxs, perm.data(), Control, Info) ;
    printf ("return value from amd_order: %ld (should be %d)\n", result, AMD_OK) ;
    if (result != AMD_OK)
    {
	printf ("AMD failed\n") ;
	exit (1) ;
    }
    // /* print a character plot of the permuted matrix. */
    // printf ("\nPlot of permuted matrix pattern:\n") ;
    // for (jnew = 0 ; jnew < n ; jnew++)
    // {
	// j = P [jnew] ;
	// for (inew = 0 ; inew < n ; inew++) A [inew][jnew] = '.' ;
	// for (p = Ap [j] ; p < Ap [j+1] ; p++)
	// {
	//     inew = Pinv [Ai [p]] ;
	//     A [inew][jnew] = 'X' ;
	// }
    // }
    return;
}

template <typename scalar_t, RandBLAS::SignedInteger sint_t = int64_t>
void permuted(const CSRMatrix<scalar_t, sint_t> &A, const std::vector<sint_t> &perm, CSRMatrix<scalar_t, sint_t> &out) {
    auto  n   = A.n_rows;
    COOMatrix<scalar_t, sint_t> coo(n, n);
    reserve_coo(A.nnz, coo);
    std::vector<sint_t> invperm(n);
    for (sint_t k = 0; k < n; ++k) {
        invperm[perm[k]] = k;
    }
    sint_t inew, jnew, p, ctr = 0;
    for (jnew = 0; jnew < n; ++jnew) {
        auto j = perm[jnew];
        for (p = A.rowptr[j]; p < A.rowptr[j+1]; ++p) {
            inew = invperm[A.colidxs[p]];
            coo.rows[ctr] = inew;
            coo.cols[ctr] = jnew;
            coo.vals[ctr] = A.vals[p];
            ++ctr;
        }
    }
    coo_to_csr(coo, out);
    return;
}


int main(int argc, char** argv) {
    using T = double;

    std::string fn("/home/rjmurr/laps2/RandLAPACK/demos/sparse_data_matrices/EY/smaller/G1/sG1.mtx");
    auto csr = laplacian_from_matrix_market(fn, (T)0.0);
    int64_t n = csr.n_rows;
    std::vector<int64_t> perm(n);
    for (int64_t i = 0; i < n; ++i)
        perm[i] = i;
    CSRMatrix<double, int64_t> L(n, n);
    TIMED_LINE(
    //amd_permutation(csr, perm);
    permuted(csr, perm, L);, "AMD reordering      : "
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
    k = richol::clb21_rand_cholesky(sym, C, state, false, (T)0.0), "SparseCholesky: "
    );
    std::cout << "Exited with C of rank k = " << k << std::endl;

    return 0;
}
