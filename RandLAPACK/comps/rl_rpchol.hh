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
template <typename T>
using Kernel = std::function<T(int64_t, int64_t)>;

template <typename T>
void compute_columns(
    Layout layout, int64_t N, Kernel<T> &K_stateless, vector<int64_t> &col_indices, T* buff
) {
    randblas_require(layout == Layout::ColMajor);
    int64_t num_cols = col_indices.size();
    for (int64_t ell = 0; ell < num_cols; ++ell) {
        T* buffj = buff + ell*N;
        int64_t j = col_indices[ell];
        for (int64_t i = 0; i < N; ++i) {
            buffj[i] = K_stateless(i, j);
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
void downdate_d_and_cdf(Layout layout, int64_t N, int64_t cols_F_panel, T* F_panel, vector<T> &d, vector<T> &cdf) {
    randblas_require(layout == Layout::ColMajor);
    for (int64_t j = 0; j < cols_F_panel; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            T val = F_panel[i + j*N];
            d[i] -= val*val;
        }
    }
    cdf = d;
    weights_to_cdf(cdf.data(), N);
    return;
}

} // end namespace RandLAPACK::_rpchol_impl

/***
 * Computes a rank-r approximation of an implicit N-by-N matrix whose (i,j)^{th}
 * entry is K_stateless(i,j), where K_stateless is a stateless function. We build
 * the approximation iteratively and increase the rank by at most "B" at each iteration.
 * 
 * Implements Algorithm 4 from https://arxiv.org/abs/2304.12465.
 * 
 * Example code for using the squared exponential kernel:
 * 
 *      // Assume we've already defined ...
 *      //         X  : a rows_x by cols_x double-precision matrix (suitably standardized)
 *      //              where each column defines a datapoint.
 *      //  bandwidth : scale for the squared exponential kernel    
 * 
 *      auto K_stateless = [X, rows_x, cols_x, bandwidth](int64_t i, int64_t j) {
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
 *      std::vector<double> Fvec(rows_x*r, 0.0);
 *      std::vector<int64_t> selected_cols(r);
 *      RandBLAS::RNGState state_in(0);
 *      auto state_out = rp_cholesky(cols_x, K, r, selected_cols.data(), Fvec.data(), 64, state_in);
 * 
 */
template <typename T, typename STATE>
STATE rp_cholesky(int64_t N, std::function<T(int64_t, int64_t)> &K_stateless, int64_t r, int64_t* selected_columns,  T* F, int64_t B, STATE s) {
    using RandBLAS::util::sample_indices_iid;
    using RandBLAS::util::weights_to_cdf;
    using blas::Op;
    using blas::Uplo;
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;

    std::vector<T> work_mat(B*r, 0.0);
    std::vector<T> work_d(N, 0.0);
    std::vector<T> work_cdf(N);

    std::vector<int64_t> bcols{};
    // ^ Analogous to the symbol $S'$ in the arXiv paper. 
    
    for (int64_t i = 0; i < N; ++i)
        work_d[i] = K_stateless(i,i);
    work_cdf = work_d;
    weights_to_cdf(work_cdf.data(), N);

    int64_t ell = 0;
    while (ell < r) {
        //
        //  1. Compute the next block of column indices
        //
        int64_t curr_B = std::min(B, r - ell);
        bcols.resize(curr_B);
        s = sample_indices_iid(work_cdf.data(), N, bcols.data(), curr_B, s);
        std::sort( bcols.begin(), bcols.end() );
        bcols.erase( unique( bcols.begin(), bcols.end() ), bcols.end() );
        int64_t ell_incr = bcols.size();
        int64_t ell_next = ell + ell_incr;
        std::fill(selected_columns + ell, selected_columns + ell_next, bcols.begin());

        //
        //  2. Compute F_panel: the next block of ell_incr columns in F.
        //
        T* F_panel = F + ell*N;
        //
        //      2.1. Overwrite F_panel with the matrix "G" from Line 5 of [arXiv:2304.12465, Algorithm 4].
        //
        //           First we compute a submatrix of columns of K and then we downdate with GEMM.
        //           The downdate is delicate since the output matrix shares a buffer with one of the
        //           input matrices, but it's okay since they're non-overlapping regions of that buffer.
        //
        _rpchol_impl::compute_columns(layout, N, K_stateless, bcols, F_panel);
        //           ^ F_panel = K(:, bcols).
        _rpchol_impl::pack_selected_rows(layout, N, ell, F, bcols, work_mat.data());
        //           ^ work_mat is a copy of F(bcols, 1:ell).
        blas::gemm(
            layout, Op::NoTrans, Op::Trans, N, ell_incr, ell,
            -1.0, F, N, work_mat.data(), ell_incr, 1.0, F_panel, N
        );
        //
        //      2.2. Execute Lines 6 and 7 of [arXiv:2304.12465, Algorithm 4].     
        //
        _rpchol_impl::pack_selected_rows(layout, N, ell_incr, F_panel, bcols, work_mat.data());
        int status = lapack::potrf(uplo, ell_incr, work_mat.data(), ell_incr);
        randblas_require(status == 0);
        blas::trsm(
            layout, blas::Side::Right, uplo, Op::NoTrans, blas::Diag::NonUnit,
            N, ell_incr, 1.0, work_mat.data(), ell_incr, F_panel, N
        );

        //
        // 3. Update d and ell. The update formula references F. We define an alias pointer
        //    to the relevant submatrix even though we could use G's pointer. 
        //
        _rpchol_impl::downdate_d_and_cdf(layout, N, ell_incr, F_panel, work_d, work_cdf);
        ell = ell_next;
    }
    return s;
}


}
