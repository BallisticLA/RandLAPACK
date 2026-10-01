#pragma once

// Public API: ExplicitSymLinOp, DiagSymLinOp, SparseSymLinOp, RegExplicitSymLinOp, SpectralPrecond,
// symmetric linear operators for use with SymmetricLinearOperator-templated algorithms.

#include "rl_exceptions.hh"
#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

// Opt-in performance switch, off by default: runtime switches read on every call, all off by default.
//   RANDLAPACK_PERF_GEMM=1  ExplicitSymLinOp dense products with 2..256 columns use gemm on the full buffer, not symm.
//   RANDLAPACK_PERF_SPMM=1  sparse sketches use right_spmm instead of the one-triangle sketch_symmetric.
// Both read the WHOLE buffer, so they apply only to an ExplicitSymLinOp whose owner sets both_triangles.
inline bool rl_perf_switch(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}


namespace RandLAPACK::linops {

// Symmetric linear operators for use with algorithms templated on
// SymmetricLinearOperator. Five concrete types are provided:
//
//   ExplicitSymLinOp      — wraps a dense symmetric matrix (upper or lower
//                           triangle, any layout). Applies C := alpha*A*B + beta*C
//                           via blas::symm, handling layout mismatches between A
//                           and (B, C) by flipping the Uplo parameter.
//
//   DiagSymLinOp          - diag(lambda) from a pointer to the n diagonal entries;
//                           the n x n matrix is never formed. Same two apply
//                           overloads as ExplicitSymLinOp (dense right operand and
//                           RandBLAS sketching operator), so the two are
//                           interchangeable in the templated drivers.
//
//   SparseSymLinOp        - wraps a RandBLAS sparse matrix holding both triangles;
//                           dense products by RandBLAS::spmm, sparse sketches by
//                           RandBLAS::sketch_sparse (MKL spgemm). Same two overloads.
//
//   RegExplicitSymLinOp   — a container that implicitly holds num_ops >= 1 symmetric
//                           linear operators, all of which differ from one another
//                           by some shift of the identity matrix. See class
//                           documentation for details.
//
//   SpectralPrecond       — spectral preconditioner for systems (G + mu*I)x = b.
//                           Represents P = V*diag(D)*V' + I, where V holds
//                           approximate top eigenvectors of G and D is derived from
//                           the corresponding eigenvalues and regularization parameter.
//                           Used by preconditioned conjugate gradient (pcg).

/*********************************************************/
/*                                                       */
/*                  ExplicitSymLinOp                     */
/*                                                       */
/*********************************************************/

/// Dense symmetric linear operator satisfying SymmetricLinearOperator.
///
/// Wraps a pointer to a symmetric matrix stored in either upper or lower
/// triangle, in either row-major or column-major layout. The SYMM-like
/// callable computes C := alpha * A * B + beta * C via blas::symm.
///
/// If the layout requested for (B, C) differs from buff_layout, the Uplo
/// parameter is flipped so that the same physical triangle is read correctly.
///
/// Also provides element access via operator()(i, j), which currently
/// requires upper-triangle, column-major storage.
template <typename T>
struct ExplicitSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const Uplo uplo;
    const T* A_buff;
    const int64_t lda;
    const Layout buff_layout;
    /// Set true only when A_buff holds the full symmetric matrix (both triangles). The opt-in
    /// RANDLAPACK_PERF_GEMM and RANDLAPACK_PERF_SPMM switches read the whole buffer, so they
    /// take effect only for operators that declare this; the default honours `uplo` alone.
    bool both_triangles = false;

    ExplicitSymLinOp(
        int64_t dim,
        Uplo uplo,
        const T* A_buff,
        int64_t lda,
        Layout buff_layout
    ) : m(dim), dim(dim), uplo(uplo), A_buff(A_buff), lda(lda), buff_layout(buff_layout) {}

    // Note: the "layout" parameter here is interpreted for (B and C).
    // If layout conflicts with this->buff_layout then we manipulate
    // parameters to blas::symm to reconcile the different layouts of
    // A vs (B, C).
    void operator()(
        Layout layout,
        int64_t n,
        T alpha,
        T* const B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        randlapack_require(ldb >= dim) << "ldb=" << ldb << " < dim=" << dim << " (ldb must be >= operator dimension)";
        randlapack_require(ldc >= dim) << "ldc=" << ldc << " < dim=" << dim << " (ldc must be >= operator dimension)";
        auto blas_call_uplo = this->uplo;
        if (layout != this->buff_layout)
            blas_call_uplo = (this->uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
        // Reading the "blas_call_uplo" triangle of "this->A_buff" in "layout" order is the same
        // as reading the "this->uplo" triangle of "this->A_buff" in "this->buff_layout" order.
        // gemm only for 2..256 columns: at n = 50,000 MKL's gemm beats symm up to 1.7x there, but falls off a cliff
        // above 257 columns (2.0 s against symm's 1.27 s at 269) and loses slightly at one column.
        if (this->both_triangles && rl_perf_switch("RANDLAPACK_PERF_GEMM") && n >= 2 && n <= 256) {
            // A is symmetric with both triangles stored, so reading it in either layout gives A.
            blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, dim, n, dim, alpha,
                this->A_buff, this->lda, B, ldb, beta, C, ldc);
            return;
        }
        blas::symm(
            layout, Side::Left, blas_call_uplo, dim, n, alpha,
            this->A_buff, this->lda, B, ldb, beta, C, ldc
        );
    }

    inline T operator()(int64_t i, int64_t j) {
        randlapack_require(this->uplo == Uplo::Upper && this->buff_layout == Layout::ColMajor) << "element access operator()(i,j) requires upper-triangle + ColMajor storage";
        if (i > j) {
            return A_buff[j + i*lda];
        } else {
            return A_buff[i + j*lda];
        }
    }

    /// Apply a dense or sparse sketch. RANDLAPACK_SYMMETRIC_SKETCH enables
    /// a triangle-aware sparse product with a supporting RandBLAS installation.
    /// Otherwise the sparse path uses right_spmm and requires both triangles.
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Layout layout,
        int64_t n_vecs,
        T alpha,
        SkOp& S,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if constexpr (requires { S.buff; S.layout; S.dist; }) {
            // Dense sketch — extract buffer, dispatch to the dense matvec path.
            if (S.buff == nullptr) RandBLAS::fill_dense(S);
            int64_t ldS = S.dist.dim_major;
            randblas_require(S.layout == layout);
            (*this)(layout, n_vecs, alpha, S.buff, ldS, beta, C, ldc);
        } else {
            // Fill once before choosing the triangle-aware or legacy product.
            if (S.nnz < 0) RandBLAS::fill_sparse(S);
#ifdef RANDLAPACK_SYMMETRIC_SKETCH
            if (!(this->both_triangles && rl_perf_switch("RANDLAPACK_PERF_SPMM"))) {
                auto apply_uplo = this->uplo;
                if (layout != this->buff_layout)
                    apply_uplo = (apply_uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
                RandBLAS::sketch_symmetric(layout, apply_uplo, dim, n_vecs,
                    alpha, this->A_buff, this->lda, S, 0, 0, beta, C, ldc);
                return;
            }
#endif
            {
            auto S_coo = RandBLAS::coo_view_of_skop(S);
            RandBLAS::sparse_data::right_spmm(
                layout, blas::Op::NoTrans, blas::Op::NoTrans,
                dim, n_vecs, dim,
                alpha, this->A_buff, this->lda,
                S_coo, 0, 0,
                beta, C, ldc
            );
            }
        }
    }
};

/*********************************************************/
/*                                                       */
/*                    DiagSymLinOp                       */
/*                                                       */
/*********************************************************/

/// Diagonal symmetric linear operator satisfying SymmetricLinearOperator.
///
/// Represents A = diag(lambda) from a pointer to the dim entries of lambda; the
/// dim x dim matrix is never formed. The dense apply is a row scaling of B. The
/// sketching-operator apply walks a sparse operator's COO triples directly and
/// dispatches a dense operator through the dense apply, mirroring
/// ExplicitSymLinOp so the two types are interchangeable in the drivers.
///
/// The caller owns lambda and keeps it alive for the operator's lifetime.
template <typename T>
struct DiagSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const T* lambda;

    DiagSymLinOp(
        int64_t dim,
        const T* lambda
    ) : m(dim), dim(dim), lambda(lambda) {}

    // C := alpha * diag(lambda) * B + beta * C. C is not read when beta == 0.
    // Strides: ColMajor needs ldb, ldc >= dim; RowMajor needs ldb, ldc >= n.
    void operator()(
        Layout layout,
        int64_t n,
        T alpha,
        T* const B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        const int64_t min_ld = (layout == Layout::ColMajor) ? dim : n;
        randlapack_require(ldb >= min_ld) << "ldb=" << ldb << " < " << min_ld << " (stride must cover the operator dimension in ColMajor or n in RowMajor)";
        randlapack_require(ldc >= min_ld) << "ldc=" << ldc << " < " << min_ld << " (stride must cover the operator dimension in ColMajor or n in RowMajor)";
        if (layout == Layout::ColMajor) {
            for (int64_t j = 0; j < n; ++j) {
                const T* bj = B + j * ldb;
                T* cj = C + j * ldc;
                if (beta == (T)0) {
                    for (int64_t i = 0; i < dim; ++i) cj[i] = alpha * lambda[i] * bj[i];
                } else {
                    for (int64_t i = 0; i < dim; ++i) cj[i] = alpha * lambda[i] * bj[i] + beta * cj[i];
                }
            }
        } else {
            for (int64_t i = 0; i < dim; ++i) {
                const T* bi = B + i * ldb;
                T* ci = C + i * ldc;
                const T a = alpha * lambda[i];
                if (beta == (T)0) {
                    for (int64_t j = 0; j < n; ++j) ci[j] = a * bi[j];
                } else {
                    for (int64_t j = 0; j < n; ++j) ci[j] = a * bi[j] + beta * ci[j];
                }
            }
        }
    }

    inline T operator()(int64_t i, int64_t j) {
        return (i == j) ? lambda[i] : (T)0;
    }

    /// SkOp overload, same contract as ExplicitSymLinOp's. A dense operator is
    /// filled and applied through the dense path. A sparse operator is filled if
    /// needed and its COO triples (i, j, v) are accumulated as
    /// C[i, j] += alpha * lambda[i] * v after C is scaled by beta; the sketch is
    /// never densified. ColMajor only, which is all the drivers use.
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Layout layout,
        int64_t n_vecs,
        T alpha,
        SkOp& S,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if constexpr (requires { S.buff; S.layout; S.dist; }) {
            if (S.buff == nullptr) RandBLAS::fill_dense(S);
            int64_t ldS = S.dist.dim_major;
            randblas_require(S.layout == layout);
            (*this)(layout, n_vecs, alpha, S.buff, ldS, beta, C, ldc);
        } else {
            randlapack_require(layout == Layout::ColMajor) << "DiagSymLinOp sparse-sketch apply supports ColMajor only";
            randlapack_require(ldc >= dim) << "ldc=" << ldc << " < dim=" << dim << " (ldc must be >= operator dimension)";
            if (S.nnz < 0) RandBLAS::fill_sparse(S);
            auto S_coo = RandBLAS::coo_view_of_skop(S);
            randlapack_require(S_coo.index_base == RandBLAS::sparse_data::IndexBase::Zero) << "sparse sketch view must be zero-based";
            randlapack_require(S_coo.n_rows == dim && S_coo.n_cols == n_vecs) << "sparse sketch is " << S_coo.n_rows << " x " << S_coo.n_cols << ", expected " << dim << " x " << n_vecs;
            for (int64_t j = 0; j < n_vecs; ++j) {
                T* cj = C + j * ldc;
                if (beta == (T)0) {
                    std::fill(cj, cj + dim, (T)0);
                } else if (beta != (T)1) {
                    for (int64_t i = 0; i < dim; ++i) cj[i] *= beta;
                }
            }
            for (int64_t t = 0; t < S_coo.nnz; ++t) {
                const int64_t i = (int64_t)S_coo.rows[t];
                const int64_t j = (int64_t)S_coo.cols[t];
                C[i + j * ldc] += alpha * lambda[i] * S_coo.vals[t];
            }
        }
    }
};

/*********************************************************/
/*                                                       */
/*                   SparseSymLinOp                      */
/*                                                       */
/*********************************************************/

/// Sparse symmetric linear operator satisfying SymmetricLinearOperator.
///
/// Wraps a RandBLAS sparse matrix (COO, CSR or CSC) that holds BOTH triangles of a
/// symmetric matrix; no triangle is inferred. The dense apply is RandBLAS::spmm.
/// A dense sketching operator goes through the dense apply; a sparse one is applied
/// by RandBLAS::sketch_sparse (sparse times sparse, MKL spgemm), so neither the
/// matrix nor the sketch is densified. Same two apply overloads as
/// ExplicitSymLinOp, so the types are interchangeable in the templated drivers.
///
/// The caller owns the sparse matrix and keeps it alive for the operator's lifetime.
template <typename T, RandBLAS::sparse_data::SparseMatrix SpMat>
struct SparseSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const SpMat& A_sp;

    SparseSymLinOp(
        const SpMat& A_sp
    ) : m(A_sp.n_rows), dim(A_sp.n_rows), A_sp(A_sp) {
        randlapack_require(A_sp.n_rows == A_sp.n_cols) << "sparse operator is " << A_sp.n_rows << " x " << A_sp.n_cols << ", expected square";
        randlapack_require(A_sp.index_base == RandBLAS::sparse_data::IndexBase::Zero) << "sparse operator must be zero-based";
    }

    // C := alpha * A * B + beta * C. Strides: ColMajor needs ldb, ldc >= dim; RowMajor needs ldb, ldc >= n.
    void operator()(
        Layout layout,
        int64_t n,
        T alpha,
        T* const B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        const int64_t min_ld = (layout == Layout::ColMajor) ? dim : n;
        randlapack_require(ldb >= min_ld) << "ldb=" << ldb << " < " << min_ld << " (stride must cover the operator dimension in ColMajor or n in RowMajor)";
        randlapack_require(ldc >= min_ld) << "ldc=" << ldc << " < " << min_ld << " (stride must cover the operator dimension in ColMajor or n in RowMajor)";
        RandBLAS::spmm(layout, blas::Op::NoTrans, blas::Op::NoTrans, dim, n, dim, alpha, A_sp, B, ldb, beta, C, ldc);
    }

    /// Element access for CSC storage with sorted row indices (binary search in column j).
    inline T operator()(int64_t i, int64_t j) {
        if constexpr (requires { A_sp.colptr; A_sp.rowidxs; }) {
            const auto* first = A_sp.rowidxs + A_sp.colptr[j];
            const auto* last  = A_sp.rowidxs + A_sp.colptr[j + 1];
            const auto* hit = std::lower_bound(first, last, i);
            return (hit != last && (int64_t)*hit == i) ? A_sp.vals[hit - A_sp.rowidxs] : (T)0;
        } else {
            throw RandLAPACK::Error("SparseSymLinOp element access requires CSC storage");
        }
    }

    /// SkOp overload, same contract as ExplicitSymLinOp's.
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Layout layout,
        int64_t n_vecs,
        T alpha,
        SkOp& S,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if constexpr (requires { S.buff; S.layout; S.dist; }) {
            if (S.buff == nullptr) RandBLAS::fill_dense(S);
            int64_t ldS = S.dist.dim_major;
            randblas_require(S.layout == layout);
            (*this)(layout, n_vecs, alpha, S.buff, ldS, beta, C, ldc);
        } else {
#if defined(RandBLAS_HAS_MKL)
            if (S.nnz < 0) RandBLAS::fill_sparse(S);
            RandBLAS::sketch_sparse(layout, blas::Op::NoTrans, blas::Op::NoTrans, dim, n_vecs, dim,
                alpha, A_sp, S, 0, 0, beta, C, ldc);
#else
            (void)layout; (void)n_vecs; (void)alpha; (void)S; (void)beta; (void)C; (void)ldc;
            throw RandLAPACK::Error("SparseSymLinOp with a sparse sketch needs RandBLAS built with MKL (sparse times sparse); use a dense sketch instead");
#endif
        }
    }
};

/*********************************************************/
/*                                                       */
/*               RegExplicitSymLinOp                     */
/*                                                       */
/*********************************************************/

/// Regularized dense symmetric linear operator satisfying SymmetricLinearOperator.
///
/// A container that implicitly holds num_ops >= 1 symmetric linear operators,
/// all of which differ from one another by some shift of the identity matrix.
/// The underlying matrix A is stored in upper triangle, column-major layout.
///
/// If num_ops == 1, then operator()(...) has the usual behavior. If num_ops > 1,
/// then operator()(...) can only be invoked for matrix-matrix products where the
/// right operand (a dense matrix) has exactly num_ops columns. In this latter
/// case the i-th column of the input matrix will be acted on by
///     A + regs[i] * I
/// where regs is the array of distinct regularization parameters.
///
/// When _eval_includes_reg is false, only the unregularized A*B is computed
/// regardless of num_ops.
///
/// Element access operator()(i, j) returns A(i,j) + regs[0]*delta(i,j)
/// when regularization is enabled and num_ops == 1.
template <typename T>
struct RegExplicitSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const T* A_buff;
    const int64_t lda;
    int64_t num_ops = 1;
    T* regs = nullptr;
    bool _eval_includes_reg;

    static const Uplo uplo = Uplo::Upper;
    static const Layout buff_layout = Layout::ColMajor;

    RegExplicitSymLinOp(
        int64_t dim, const T* A_buff, int64_t lda, T* arg_regs, int64_t arg_num_ops
    ) : m(dim), dim(dim), A_buff(A_buff), lda(lda) {
        randlapack_require(lda >= dim) << "lda=" << lda << " < dim=" << dim << " (lda must be >= operator dimension)";
        _eval_includes_reg = false;
        num_ops = arg_num_ops;
        num_ops = std::max(num_ops, (int64_t) 1);
        regs = new T[num_ops]{};
        std::copy(arg_regs, arg_regs + arg_num_ops, regs);
    }

    RegExplicitSymLinOp(
        int64_t dim, const T* A_buff, int64_t lda, std::vector<T> &arg_regs
    ) : RegExplicitSymLinOp<T>(dim, A_buff, lda, arg_regs.data(), static_cast<int64_t>(arg_regs.size())) {}

    ~RegExplicitSymLinOp() {
        if (regs != nullptr) delete [] regs;
    }

    void set_eval_includes_reg(bool eir) {
        _eval_includes_reg = eir;
    }

    void operator()(Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        randlapack_require(layout == this->buff_layout) << "operation layout must match the operator storage layout (buff_layout)";
        randlapack_require(ldb >= dim) << "ldb=" << ldb << " < dim=" << dim << " (ldb must be >= operator dimension)";
        randlapack_require(ldc >= dim) << "ldc=" << ldc << " < dim=" << dim << " (ldc must be >= operator dimension)";
        blas::symm(layout, blas::Side::Left, this->uplo, dim, n, alpha, this->A_buff, this->lda, B, ldb, beta, C, ldc);

        if (_eval_includes_reg) {
            if (num_ops != 1) randlapack_require(n == num_ops) << "with num_ops>1, n=" << n << " must equal num_ops=" << num_ops << " so each column gets its own regularization";
            for (int64_t i = 0; i < n; ++i) {
                T coeff =  alpha * regs[std::min(i, num_ops - 1)];
                blas::axpy(dim, coeff, B + i*ldb, 1, C +  i*ldc, 1);
            }
        }
        return;
    }

    inline T operator()(int64_t i, int64_t j) {
        T val;
        if (i > j) {
            val = A_buff[j + i*lda];
        } else {
            val = A_buff[i + j*lda];
        }
        if (_eval_includes_reg && i == j) {
            randlapack_require(num_ops == 1) << "this operation requires num_ops=1; got num_ops=" << num_ops;
            val += regs[0];
        }
        return val;
    }

};

/*********************************************************/
/*                                                       */
/*                  SpectralPrecond                      */
/*                                                       */
/*********************************************************/

/// Spectral preconditioner satisfying SymmetricLinearOperator.
///
/// Represents the linear operator P = V * diag(D) * V' + I, where V holds
/// approximate top eigenvectors of a positive semidefinite matrix G, and
/// D = (lambda_min + mu) / (lambda + mu) - 1 for each approximate eigenvalue
/// lambda. This is used by preconditioned conjugate gradient (pcg) to
/// accelerate solves of (G + mu*I)x = b.
///
/// The SYMM-like callable computes C := alpha * P * B + beta * C in four steps:
///   1. W = V' * B
///   2. W = diag(D) * W   (apply as row-scaling)
///   3. C = beta * C + alpha * B
///   4. C = alpha * V * W + C
///
/// Supports multiple regularization parameters (one per right-hand side)
/// via set_D_from_eigs_and_regs(). If num_ops == 1, the single D vector
/// is broadcast across all columns.
template<typename T>
struct SpectralPrecond {

    using scalar_t = T;
    const int64_t m; ///< Alias for dim (backward compatibility).
    const int64_t dim;
    int64_t dim_pre;
    int64_t num_rhs;
    T* V = nullptr;
    T* D = nullptr;
    T* W = nullptr;
    int64_t num_ops = 0;

    /* Suppose we want to precondition a positive semidefinite matrix G_mu = G + mu*I.
     *
     * Once properly preparred, this preconditioner represents a linear operator of the form
     *      P = V diag(D) V' + I.
     * The columns of V approximate the top dim_pre eigenvectors of G, while the
     * entries of D are *functions of* the corresponding approximate eigenvalues.
     *
     * The specific form of the entries of D are as follows. Suppose we start with
     * (V, lambda) as approximations of the top dim_pre eigenpairs of G, and define the vector
     *      D0 = (min(lambda) + mu) / (lambda + mu).
     * From a mathematical perspective, this preconditioner represents the linear operator
     *      P = V diag(D0) V' + (I - VV').
     * The action of this linear operator can be computed with two calls to GEMM
     * instead of three if we store D = D0 - 1 instead of D0 itself.
     */

    SpectralPrecond(int64_t dim) : m(dim), dim(dim), dim_pre(0), num_rhs(0) {}

    // Move constructor
    // Call as SpectralPrecond<T> spc(std::move(other)) when we want to transfer the
    // contents of "other" to "this".
    SpectralPrecond(SpectralPrecond &&other) noexcept
        : m(other.dim), dim(other.dim), dim_pre(other.dim_pre), num_rhs(other.num_rhs), num_ops(other.num_ops)
    {
        std::swap(V, other.V);
        std::swap(D, other.D);
        std::swap(W, other.W);
    }

    // Copy constructor
    // Call as SpectralPrecond<T> spc(other) when we want to copy "other".
    SpectralPrecond(const SpectralPrecond &other)
        : m(other.dim), dim(other.dim), dim_pre(other.dim_pre), num_rhs(other.num_rhs),  num_ops(other.num_ops)
     {
        reset_owned_buffers(dim_pre, num_rhs, num_ops);
        std::copy(other.V, other.V + dim * dim_pre,        V);
        std::copy(other.D, other.D + dim_pre * num_ops, D);
     }

    ~SpectralPrecond() {
        if (D != nullptr) delete [] D;
        if (V != nullptr) delete [] V;
        if (W != nullptr) delete [] W;
    }

    void reset_owned_buffers(int64_t arg_dim_pre, int64_t arg_num_rhs, int64_t arg_num_ops) {
        randlapack_require(arg_num_rhs == arg_num_ops || arg_num_ops == 1) << "arg_num_rhs=" << arg_num_rhs << " must equal arg_num_ops=" << arg_num_ops << ", or arg_num_ops must be 1";

        if (arg_dim_pre * arg_num_ops > dim_pre * num_ops) {
            if (D != nullptr) delete [] D;
            D = new T[arg_dim_pre * arg_num_ops]{};
        }
        if (arg_dim_pre > dim_pre) {
            if (V != nullptr) delete [] V;
            V = new T[dim * arg_dim_pre];
        }
        if (arg_dim_pre * arg_num_rhs > dim_pre * num_rhs) {
            if (W != nullptr) delete [] W;
            W = new T[arg_dim_pre * arg_num_rhs];
        }

        dim_pre = arg_dim_pre;
        num_rhs = arg_num_rhs;
        num_ops = arg_num_ops;
    }

    void set_D_from_eigs_and_regs(T* eigvals, T* mus) {
        for (int64_t r = 0; r < num_ops; ++r) {
            T  mu_r = mus[r];
            T* D_r  = D + r*dim_pre;
            T  numerator = eigvals[dim_pre-1] + mu_r;
            for (int i = 0; i < dim_pre; ++i) {
                D_r[i] = (numerator / (eigvals[i] + mu_r)) - 1.0;
            }
        }
        return;
    }

    void prep(std::vector<T> &eigvecs, std::vector<T> &eigvals, std::vector<T> &mus, int64_t arg_num_rhs) {
        // assume eigvals are positive numbers sorted in decreasing order.
        int64_t arg_num_ops = mus.size();
        int64_t arg_dim_pre  = eigvals.size();
        reset_owned_buffers(arg_dim_pre, arg_num_rhs, arg_num_ops);
        set_D_from_eigs_and_regs(eigvals.data(), mus.data());
        std::copy(eigvecs.begin(), eigvecs.end(), V);
        return;
    }

    void operator()(
        Layout layout, int64_t n, T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc
    ) {
        randlapack_require(layout == Layout::ColMajor) << "this operator only supports ColMajor layout";
        randlapack_require(ldb >= dim) << "ldb=" << ldb << " < dim=" << dim << " (ldb must be >= operator dimension)";
        randlapack_require(ldc >= dim) << "ldc=" << ldc << " < dim=" << dim << " (ldc must be >= operator dimension)";
        if (this->num_ops != 1) {
            randlapack_require(n == num_ops) << "with num_ops>1, n=" << n << " must equal num_ops=" << num_ops << " so each column gets its own regularization";
        } else {
            randlapack_require(this->num_rhs >= n) << "this->num_rhs=" << this->num_rhs << " must be >= n=" << n;
        }
        // update C = alpha*(V diag(D) V' + I)B + beta*C
        //      Step 1: w = V'B                    with blas::gemm
        //      Step 2: w = D w                    with our own kernel
        //      Step 3: C = beta * C + alpha * B   with blas::copy or blas::scal + blas::axpy
        //      Step 4: C = alpha * V w + C        with blas::gemm
        blas::gemm(layout, blas::Op::Trans, blas::Op::NoTrans, dim_pre, n, dim, (T) 1.0, V, dim, B, ldb, (T) 0.0, W, dim_pre);

        // -----> start step 2
        #define mat_D(_i, _j)  ((num_ops == 1) ? D[(_i)] : D[(_i) + dim_pre*(_j)])
        #define mat_W(_i, _j)  W[(_i) + dim_pre*(_j)]
        for (int64_t j = 0; j < n; j++) {
            for (int64_t i = 0; i < dim_pre; i++) {
                mat_W(i, j) = mat_D(i, j) * mat_W(i, j);
            }
        }
        #undef mat_D
        #undef mat_W
        // <----- end step 2

        // -----> start step 3
        int64_t i;
        #define colB(_i) &B[(_i)*ldb]
        #define colC(_i) &C[(_i)*ldb]
        if (beta == (T) 0.0 && alpha == (T) 1.0) {
            for (i = 0; i < n; ++i)
                blas::copy(dim, colB(i), 1, colC(i), 1);
        } else {
            for (i = 0; i < n; ++i) {
                T* Ci = colC(i);
                blas::scal(dim, beta, Ci, 1);
                blas::axpy(dim, alpha, colB(i), 1, Ci, 1);
            }
        }
        #undef colB
        #undef colC
        // <----- end step 3

        blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, dim, n, dim_pre, (T) 1.0, V, dim, W, dim_pre, 1.0, C, ldc);
        return;
    }
};

} // end namespace RandLAPACK::linops
