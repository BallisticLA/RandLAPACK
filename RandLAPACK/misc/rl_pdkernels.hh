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
 * buffers of length rows_x. If use_input_mu_sigma is false then this
 * function overwrites them as follows:
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
    int64_t rows_x, int64_t cols_x, T* X, T* mu, T* sigma, bool use_input_mu_sigma = false
) {
    randblas_require(cols_x >= 2);
    if (! use_input_mu_sigma) {
        std::fill(mu, mu + rows_x, (T) 0.0);
        std::fill(sigma, sigma + rows_x, (T) 0.0);
    }
    T* ones_cols_x = new T[cols_x]{1.0};
    blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, rows_x, cols_x, 1.0/ (T)rows_x, X, rows_x, ones_cols_x, 1, (T) 0.0, mu, 1);
    // ^ Computes the mean
    blas::ger(blas::Layout::ColMajor, rows_x, cols_x, -1, mu, 1, ones_cols_x, 1, X, rows_x);
    // ^ Performs a rank-1 update to subtract off the mean.
    delete [] ones_cols_x;
    // Up next: compute the sample standard deviations and rescale each row to have sample stddev = 1.
    T stddev_scale = std::sqrt((T) (cols_x - 1));
    for (int64_t i = 0; i < rows_x; ++i) {
        sigma[i] = blas::nrm2(cols_x, X + i, rows_x);
        sigma[i] /= stddev_scale;
        blas::scal(cols_x, (T) 1.0 / sigma[i], X + i, rows_x);
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
    int64_t rows_x, int64_t cols_x, const T* X, const T* sq_colnorms_x,
    int64_t rows_eds, int64_t cols_eds, T* Eds, int64_t ro_eds, int64_t co_eds
) {
    randblas_require((0 <= co_eds) && ((co_eds + cols_eds) <= cols_x));
    randblas_require((0 <= ro_eds) && ((ro_eds + rows_eds) <= cols_x));
    std::vector<T> ones(rows_eds, 1.0);
    T* ones_d = ones.data();
    for (int64_t j = 0; j < cols_eds; ++j) {
        T* Eds_col = Eds + rows_eds*j;
        blas::copy(rows_eds, sq_colnorms_for_rows, 1, Eds_col, 1);
        blas::axpy(rows_eds, sq_colnorms_for_cols[j], ones_d, 1, Eds_col, 1);
    }
    const T* X_subros = X + rows_x * ro_eds;
    const T* X_subcos = X + rows_x * co_eds;
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
    int64_t rows_x, int64_t cols_x, const T* X, T* sq_colnorms_x,
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
T squared_exp_kernel(int64_t dim, const T* x, const T* y, T bandwidth) {
    T sq_nrm = 0.0;
    T scale = std::sqrt(2.0)*bandwidth;
    for (int64_t i = 0; i < dim; ++i) {
        T diff = (x[i] - y[i])/scale;
        sq_nrm += diff*diff;
    }
    return std::exp(-sq_nrm);
}

namespace linops {

/***
 * It might be practical to have one class that handles several different kinds of kernels.
 */
template <typename T>
struct SEKLO : public SymmetricLinearOperator<T> {
    // squared exp kernel linear operator
    const T* X;
    const int64_t rows_x;
    T bandwidth;
    vector<T> regs;

    vector<T> _sq_colnorms_x;
    vector<T> _eval_work;
    bool      _eval_includes_reg;
    int64_t   _eval_block_size;

    using scalar_t = T;

    SEKLO(
        int64_t m, const T* X, int64_t rows_x, T bandwidth, vector<T> &regs
    ) : SymmetricLinearOperator<T>(m), X(X), rows_x(rows_x), bandwidth(bandwidth),  regs(regs), _sq_colnorms_x(m), _eval_work{} {
        for (int64_t i = 0; i < m; ++i) {
            _sq_colnorms_x[i] = std::pow(blas::nrm2(rows_x, X + i*rows_x, 1), 2);
        }
        _eval_block_size = std::min(m / ((int64_t) 4), (int64_t) 512);
        _eval_work.resize(_eval_block_size * m);
        _eval_includes_reg = false;
        return;
    }

    void _prep_eval_work(
        int64_t rows_ksub, int64_t cols_ksub, int64_t ro_ksub, int64_t co_ksub
    ) {
        randblas_require(rows_ksub * cols_ksub <= (int64_t) _eval_work.size());
        squared_exp_kernel_submatrix(
            rows_x, this->m, X, _sq_colnorms_x.data(),
            rows_ksub, cols_ksub, _eval_work.data(), ro_ksub, co_ksub, bandwidth
        );
    }

    void set_eval_includes_reg(bool eir) {
        _eval_includes_reg = eir;
    }

    void operator()(blas::Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        randblas_require(layout == blas::Layout::ColMajor);
        randblas_require(ldb >= this->m);
        randblas_require(ldc >= this->m);
        int64_t row_start = 0;
        while (true) {
            int64_t row_end  = std::min(this->m, row_start + _eval_block_size);
            int64_t num_rows = row_end - row_start;
            if (num_rows <= 0)
                break;
            _prep_eval_work(num_rows, this->m, row_start, 0);
            T* A = _eval_work.data();
            T* C_submat = C + row_start;
            blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, num_rows, n, this->m, alpha, A, num_rows, B, ldb, beta, C_submat, ldc);
            row_start += num_rows;
        }
        if (_eval_includes_reg) {
            int64_t num_regs = this->regs.size();
            if (num_regs != 1)
                randblas_require(n == num_regs);
            T* regsp = regs.data();
            for (int64_t i = 0; i < n; ++i) {
                T coeff =  alpha * regsp[std::min(i, num_regs - 1)];
                blas::axpy(this->m, coeff, B + i*ldb, 1, C +  i*ldc, 1);
            }
        }
        return;
    }

    inline T operator()(int64_t i, int64_t j) {
        T val = squared_exp_kernel(rows_x, X + i*rows_x, X + j*rows_x, bandwidth);
        if (_eval_includes_reg) {
            randblas_require(regs.size() == 1);
            val += regs[0];
        }
        return val;
    }
};

} // end namespace RandLAPACK::linops

}
#endif