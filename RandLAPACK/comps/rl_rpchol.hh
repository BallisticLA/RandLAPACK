#pragma once

#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <vector>
#include <set>

namespace RandLAPACK {

namespace _rpchol_impl {

using std::vector;
using blas::Layout;

template <typename T, typename FUNC_T>
void compute_columns(
    Layout layout, int64_t N, FUNC_T &K_stateless, vector<int64_t> &col_indices, T* buff
) {
    randblas_require(layout == Layout::ColMajor);
    int64_t num_cols = col_indices.size();
    #pragma omp parallel for collapse(2)
    for (int64_t ell = 0; ell < num_cols; ++ell) {
        for (int64_t i = 0; i < N; ++i) {
            int64_t j = col_indices[ell];
            buff[i + ell*N] = K_stateless(i, j);
        }
    }
    return;
}

template <typename T>
void pack_selected_rows(
    Layout layout, int64_t rows_mat, int64_t cols_mat, T* mat, vector<int64_t> &row_indices, T* submat
) {
    randblas_require(layout == Layout::ColMajor);
    int64_t num_rows = row_indices.size();
    for (int64_t i = 0; i < num_rows; ++i) {
        blas::copy(cols_mat, mat + row_indices[i], rows_mat, submat + i, num_rows);
    }
    return;
}

template <typename T>
void downdate_d_and_cdf(Layout layout, int64_t N, vector<int64_t> &indices, T* F_panel, vector<T> &d, vector<T> &cdf) {
    randblas_require(layout == Layout::ColMajor);
    int64_t cols_F_panel = indices.size();
    for (int64_t j = 0; j < cols_F_panel; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            T val = F_panel[i + j*N];
            d[i] -= val*val;
        }
    }
    // Then, to accound for the possibility of rounding errors, manually zero-out everything in "indices."
    for (auto i : indices)
        d[i] = 0.0;
    cdf = d;
    RandBLAS::util::weights_to_cdf(cdf.data(), N);
    return;
}

} // end namespace RandLAPACK::_rpchol_impl

/***
 * Computes a rank-k approximation of an implicit n-by-n matrix whose (i,j)^{th}
 * entry is A_stateless(i,j), where A_stateless is a stateless function. We build
 * the approximation iteratively and increase the rank by at most "b" at each iteration.
 * 
 * Implements Algorithm 4 from https://arxiv.org/abs/2304.12465.
 * 
 * Here's example code where the implict matrix is given by a squared exponential kernel:
 * 
 *      // Assume we've already defined ...
 *      //         X  : a rows_x by cols_x double-precision matrix (suitably standardized)
 *      //              where each column defines a datapoint.
 *      //  bandwidth : scale for the squared exponential kernel    
 * 
 *      auto A = [X, rows_x, cols_x, bandwidth](int64_t i, int64_t j) {
 *          double out = 0;
 *          double* Xi = X + i*rows_x;
 *          double* Xj = X + j*rows_x;
 *          for (int64_t ell = 0; ell < rows_x) {
 *              double val = (Xi[ell] - Xj[ell]) / (std::sqrt(2)*bandwidth);
 *              out += val*val;
 *          }
 *          out = std::exp(out);
 *          return out;
 *      };
 *      std::vector<double> F(rows_x*k, 0.0);
 *      std::vector<int64_t> selection(k);
 *      RandBLAS::RNGState state_in(0);
 *      auto state_out = rp_cholesky(cols_x, A, k, selection.data(), F.data(), 64, state_in);
 * 
 * Notes
 * -----
 * Compare to 
 * https://github.com/eepperly/Robust-randomized-preconditioning-for-kernel-ridge-regression/blob/main/code/choleskybase.m
 * 
 */
template <typename T, typename FUNC_T, typename STATE>
STATE rp_cholesky(int64_t n, FUNC_T &A_stateless, int64_t &k, int64_t* S,  T* F, int64_t b, STATE state) {
    // TODO: make this function robust to rank-deficient matrices. 
    using RandBLAS::util::sample_indices_iid;
    using RandBLAS::util::weights_to_cdf;
    using blas::Op;
    using blas::Uplo;
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;

    std::vector<T> work_mat(b*k, 0.0);
    std::vector<T> d(n, 0.0);
    std::vector<T> cdf(n);

    std::vector<int64_t> Sprime{};
    
    for (int64_t i = 0; i < n; ++i)
        d[i] = A_stateless(i,i);
    cdf = d;
    weights_to_cdf(cdf.data(), n);

    int64_t ell = 0;
    while (ell < k) {
        //
        //  1. Compute the next block of column indices
        //
        int64_t curr_B = std::min(b, k - ell);
        Sprime.resize(curr_B);
        state = sample_indices_iid(cdf.data(), n, Sprime.data(), curr_B, state);
        std::sort( Sprime.begin(), Sprime.end() );
        Sprime.erase( unique( Sprime.begin(), Sprime.end() ), Sprime.end() );
        int64_t ell_incr = Sprime.size();

        //
        //  2. Compute F_panel: the next block of ell_incr columns in F.
        //
        T* F_panel = F + ell*n;
        //
        //      2.1. Overwrite F_panel with the matrix "G" from Line 5 of [arXiv:2304.12465, Algorithm 4].
        //
        //           First we compute a submatrix of columns of A and then we downdate with GEMM.
        //           The downdate is delicate since the output matrix shares a buffer with one of the
        //           input matrices, but it's okay since they're non-overlapping regions of that buffer.
        //
        _rpchol_impl::compute_columns(layout, n, A_stateless, Sprime, F_panel);
        //           ^ F_panel = A(:, Sprime).
        _rpchol_impl::pack_selected_rows(layout, n, ell, F, Sprime, work_mat.data());
        //           ^ work_mat is a copy of F(Sprime, 1:ell).
        blas::gemm(
            layout, Op::NoTrans, Op::Trans, n, ell_incr, ell,
            -1.0, F, n, work_mat.data(), ell_incr, 1.0, F_panel, n
        );
        //
        //      2.2. Execute Lines 6 and 7 of [arXiv:2304.12465, Algorithm 4].     
        //
        _rpchol_impl::pack_selected_rows(layout, n, ell_incr, F_panel, Sprime, work_mat.data());
        int status = lapack::potrf(uplo, ell_incr, work_mat.data(), ell_incr);
        if (status) {
            std::cout << "Cholesky failed with exit code " << status << ".\n";
            std::cout << "Returning early, with approximation rank = " << ell << "\n\n";
            k = ell;
            return state;
        }
        blas::trsm(
            layout, blas::Side::Right, uplo, Op::NoTrans, blas::Diag::NonUnit,
            n, ell_incr, 1.0, work_mat.data(), ell_incr, F_panel, n
        );

        //
        // 3. Update S, d, cdf and ell.
        //
        std::copy(Sprime.begin(), Sprime.end(), S + ell);
        _rpchol_impl::downdate_d_and_cdf(layout, n, Sprime, F_panel, d, cdf);
        ell = ell + ell_incr;
    }
    return state;
}


}
