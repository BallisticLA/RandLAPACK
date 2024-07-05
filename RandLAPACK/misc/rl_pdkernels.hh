#ifndef randlapack_misc_pdkernels_h
#define randlapack_misc_pdkernels_h

#include "rl_blaspp.hh"
#include "rl_linops.hh"
#include <RandBLAS.hh>

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <execution>
#include <cmath>

namespace RandLAPACK {

/*** 
 * X is a rows_x by cols_x matrix stored in column major format with
 * leading dimension equal to rows_x. Each column of X is interpreted
 * as a datapoint in "rows_x" dimensional space. mu and sigma are
 * buffers of length cols_x that this function overwrites as follows:
 * 
 *     mu(i) = [the sample mean of X(i,1), ..., X(i, end) ].
 * 
 *     sigma(i) = [the sample standard deviation of X(i,1), ..., X(i, end) ].
 * 
 * This function subtracts off a copy of "mu" from each column of X and 
 * divides each row of X by the corresponding entry of sigma.
 * On exit, each row of X has mean 0.0 and sample standard deviation 1.0.
 * 
 */
template <typename T>
void standardize_dataset(
    int64_t rows_x, int64_t cols_x, T* X, T* mu, T* sigma
) {
    randblas_require(cols_x >= 2);
    std::fill(mu, mu + rows_x, 0.0);
    std::fill(sigma, sigma + rows_x, 0.0);
    T* ones_cols_x = new T[cols_x]{1.0};
    blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, rows_x, cols_x, 1.0/ (T)rows_x, X, rows_x, ones_cols_x, 1, 0.0, mu, 1);
    // ^ Computes the mean
    blas::ger(blas::Layout::ColMajor, rows_x, cols_x, -1, mu, 1, ones_cols_x, 1, X, rows_x);
    // ^ Performs a rank-1 update to subtract off the mean.
    delete [] ones_cols_x;
    // Up next: compute the sample standard deviations and rescale each row to have sample stddev = 1.
    T stddev_scale = std::sqrt((T) (cols_x - 1));
    for (int64_t i = 0; i < rows_x; ++i) {
        sigma[i] = blas::nrm2(cols_x, X + i, rows_x);
        sigma[i] /= stddev_scale;
        blas::scal(cols_x, 1.0 / sigma[i], X + i, rows_x);
    }
    return;
}

/***
 * X is a rows_x by cols_x matrix stored in column major format with
 * leading dimension equal to rows_x; sq_colnorms_x is a buffer of 
 * length "cols_x" whose j-th entry is ||X(:,j)||_2^2.
 * 
 * The Euclidean distance matrix induced by X has entries
 * 
 *      E(i,j) = ||X(:,i) - X(:, J)||_2^2
 * 
 * This function computes the contiguous submatrix of E of dimensions
 * rows_eds by cols_eds, whose upper-left corner is offset by
 * (ro_eds, co_eds) from the upper-left corner of the full matrix E.
 * 
 * On exit, Eds contains that computed submatrix.
 */
template <typename T>
void euclidean_distance_submatrix(
    int64_t rows_x, int64_t cols_x, T* X, T* sq_colnorms_x,
    int64_t rows_eds, int64_t cols_eds, T* Eds, int64_t ro_eds, int64_t co_eds
) {
    randblas_require((0 <= co_eds) && ((co_eds + cols_eds) <= cols_x));
    randblas_require((0 <= ro_eds) && ((ro_eds + rows_eds) <= cols_x));
    for (int64_t i = 0; i < rows_eds; ++i) {
        T a = sq_colnorms_x[i + ro_eds];
        for (int64_t j = 0; j < cols_eds; ++j) {
            T b = sq_colnorms_x[j + co_eds];
            Eds[i + rows_eds * j] = a + b;
        }
    }
    T* X_subros = X + rows_x * ro_eds;
    T* X_subcos = X + rows_x * co_eds;
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
        rows_eds, cols_eds, rows_x,
        -2.0, X_subros, rows_x, X_subcos, rows_x, 1.0, Eds, rows_eds
    );
    return;
}

/***
 * X is a rows_x by cols_x matrix stored in column major format with
 * leading dimension equal to rows_x; sq_colnorms_x is a buffer of 
 * length "cols_x" whose j-th entry is ||X(:,j)||_2^2.
 * 
 * The squared exponential kernel with scale given by "bandwidth" is
 * a matrix of the form
 * 
 *      K(i, j) = exp(- ||X(:,i) - X(:, J)||_2^2 / (2*bandwidth^2))
 * 
 * That is -- each column of X defines a datapoint, and K is the induced
 * positive (semi)definite kernel matrix.
 * 
 * This function computes the contiguous submatrix of K of dimensions
 * rows_ksub by cols_ksub, whose upper-left corner is offset by
 * (ro_ksub, co_ksub) from the upper-left corner of the full matrix K.
 * 
 * The result is stored in "Ksub", which is interpreted in column-major
 * order with leading dimension equal to rows_ksub.
 */
template <typename T>
void squared_exp_kernel_submatrix(
    int64_t rows_x, int64_t cols_x, T* X, T* sq_colnorms_x,
    int64_t rows_ksub, int64_t cols_ksub,  T* Ksub, int64_t ro_ksub, int64_t co_ksub,
    T bandwidth
) {
    randblas_require(bandwidth > 0);
    // First, compute the relevant submatrix of the Euclidean distance matrix
    euclidean_distance_submatrix(rows_x, cols_x, X, sq_colnorms_x, rows_ksub, cols_ksub, Ksub, ro_ksub, co_ksub);
    // Next, scale by -1/(2*bandwidth^2).
    T scale = -1.0 / (2.0 * bandwidth * bandwidth);
    int64_t size_Ksub = rows_ksub * cols_ksub;
    blas::scal(size_Ksub, scale, Ksub, 1);
    // Finally, apply an elementwise exponential function
    auto inplace_exp = [](T &val) { val = std::exp(val); };
    // TODO: look at using std::execution for parallelism.
    for (int64_t i = 0; i < size_Ksub; ++i) 
        inplace_exp(Ksub[i]);
    return;
}

template <typename T>
T squared_exp_kernel(int64_t dim, T* x, T* y, T bandwidth) {
    T sq_nrm = 0.0;
    T scale = std::sqrt(2.0)*bandwidth;
    for (int64_t i = 0; i < dim; ++i) {
        T diff = (x[i] - y[i])/scale;
        sq_nrm += diff*diff;
    }
    return std::exp(-sq_nrm);
}

}
#endif