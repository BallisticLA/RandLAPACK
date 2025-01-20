#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace RandLAPACK::linops {

using std::vector;

template <typename T>
struct SymmetricLinearOperator {

    const int64_t m;

    SymmetricLinearOperator(int64_t m) : m(m) {};

    /* The semantics of this function are similar to blas::symm.
        * We compute
        *      C = alpha * A * B + beta * C
        * where this SymmetricLinearOperator object represents "A"
        * and "B" has "n" columns.
        * 
        * Note: The parameter "layout" refers to the storage
        * order of B and C. There's no universal notion of "layout"
        * for A since A is an abstract linear operator.
    */
    virtual void operator()(
        Layout layout,
        int64_t n,
        T alpha,
        T* const B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) = 0;

    virtual T operator()(int64_t i, int64_t j) = 0;
 
    virtual ~SymmetricLinearOperator() {}
};

template <typename T>
struct ExplicitSymLinOp : public SymmetricLinearOperator<T> {

    const blas::Uplo uplo;
    const T* A_buff;
    const int64_t lda;
    const blas::Layout buff_layout;

    ExplicitSymLinOp(
        int64_t m,
        blas::Uplo uplo,
        const T* A_buff,
        int64_t lda,
        blas::Layout buff_layout
    ) : SymmetricLinearOperator<T>(m), uplo(uplo), A_buff(A_buff), lda(lda), buff_layout(buff_layout) {}

    // Note: the "layout" parameter here is interpreted for (B and C).
    // If layout conflicts with this->buff_layout then we manipulate
    // parameters to blas::symm to reconcile the different layouts of
    // A vs (B, C).
    void operator()(
        blas::Layout layout,
        int64_t n,
        T alpha,
        T* const B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        randblas_require(ldb >= this->m);
        randblas_require(ldc >= this->m);
        auto blas_call_uplo = this->uplo;
        if (layout != this->buff_layout)
            blas_call_uplo = (this->uplo == blas::Uplo::Upper) ? blas::Uplo::Lower : blas::Uplo::Upper;
        // Reading the "blas_call_uplo" triangle of "this->A_buff" in "layout" order is the same
        // as reading the "this->uplo" triangle of "this->A_buff" in "this->buff_layout" order.
        blas::symm(
            layout, Side::Left, blas_call_uplo, this->m, n, alpha,
            this->A_buff, this->lda, B, ldb, beta, C, ldc
        );
    }

    inline T operator()(int64_t i, int64_t j) {
        randblas_require(this->uplo == blas::Uplo::Upper && this->buff_layout == blas::Layout::ColMajor);
        if (i > j) {
            return A_buff[j + i*lda];
        } else {
            return A_buff[i + j*lda];
        }
    }
};

template <typename T>
struct RegExplicitSymLinOp : public SymmetricLinearOperator<T> {

    const T* A_buff;
    const int64_t lda;
    int64_t num_regs = 1;
    T* regs = nullptr;
    bool _eval_includes_reg;

    static const blas::Uplo uplo = blas::Uplo::Upper;
    static const blas::Layout buff_layout = blas::Layout::ColMajor;
    using scalar_t = T;

    RegExplicitSymLinOp(
        int64_t m, const T* A_buff, int64_t lda, vector<T> &arg_regs
    ) : SymmetricLinearOperator<T>(m), A_buff(A_buff), lda(lda) {
        randblas_require(lda >= m);
        _eval_includes_reg = false;
        num_regs = arg_regs.size();
        num_regs = std::max(num_regs, (int64_t) 1);
        regs = new T[num_regs]{};
        std::copy(arg_regs.begin(), arg_regs.end(), regs);
    }

    ~RegExplicitSymLinOp() {
        if (regs != nullptr) delete [] regs;
    }

    void set_eval_includes_reg(bool eir) {
        _eval_includes_reg = eir;
    }

    void operator()(blas::Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        randblas_require(layout == this->buff_layout);
        randblas_require(ldb >= this->m);
        randblas_require(ldc >= this->m);
        blas::symm(layout, blas::Side::Left, this->uplo, this->m, n, alpha, this->A_buff, this->lda, B, ldb, beta, C, ldc);

        if (_eval_includes_reg) {
            if (num_regs != 1) randblas_require(n == num_regs);
            for (int64_t i = 0; i < n; ++i) {
                T coeff =  alpha * regs[std::min(i, num_regs - 1)];
                blas::axpy(this->m, coeff, B + i*ldb, 1, C +  i*ldc, 1);
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
            randblas_require(num_regs == 1);
            val += regs[0];
        }
        return val;
    }

};

template<typename T>
struct SpectralPrecond {

    public:
    using scalar_t = T; 
    const int64_t m;
    int64_t dim_pre;
    int64_t num_rhs;
    T* V = nullptr;
    T* D = nullptr;
    T* W = nullptr;
    int64_t num_regs = 0;

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

    SpectralPrecond(int64_t m) : m(m), dim_pre(0), num_rhs(0) {}

    // Move constructor
    // Call as SpectralPrecond<T> spc(std::move(other)) when we want to transfer the
    // contents of "other" to "this". 
    SpectralPrecond(SpectralPrecond &&other) noexcept
        : m(other.m), dim_pre(other.dim_pre), num_rhs(other.num_rhs), num_regs(other.num_regs)
    {
        std::swap(V, other.V);
        std::swap(D, other.D);
        std::swap(W, other.W);
    }

    // Copy constructor
    // Call as SpectralPrecond<T> spc(other) when we want to copy "other".
    SpectralPrecond(const SpectralPrecond &other)
        : m(other.m), dim_pre(other.dim_pre), num_rhs(other.num_rhs),  num_regs(other.num_regs)
     {
        reset_owned_buffers(dim_pre, num_rhs, num_regs);
        std::copy(other.V, other.V + m * dim_pre,        V);
        std::copy(other.D, other.D + dim_pre * num_regs, D);
     } 

    ~SpectralPrecond() {
        if (D != nullptr) delete [] D;
        if (V != nullptr) delete [] V;
        if (W != nullptr) delete [] W;
    }

    void reset_owned_buffers(int64_t arg_dim_pre, int64_t arg_num_rhs, int64_t arg_num_regs) {
        randblas_require(arg_num_rhs == arg_num_regs || arg_num_regs == 1);

        if (arg_dim_pre * arg_num_regs > dim_pre * num_regs) {
            if (D != nullptr) delete [] D;
            D = new T[arg_dim_pre * arg_num_regs]{};
        } 
        if (arg_dim_pre > dim_pre) {
            if (V != nullptr) delete [] V;
            V = new T[m * arg_dim_pre];
        }
        if (arg_dim_pre * arg_num_rhs > dim_pre * num_rhs) {
            if (W != nullptr) delete [] W;
            W = new T[arg_dim_pre * arg_num_rhs];
        }

        dim_pre  = arg_dim_pre;
        num_rhs  = arg_num_rhs;
        num_regs = arg_num_regs;
    }

    void prep(vector<T> &eigvecs, vector<T> &eigvals, vector<T> &mus, int64_t arg_num_rhs) {
        // assume eigvals are positive numbers sorted in decreasing order.
        int64_t arg_num_regs = mus.size();
        int64_t arg_dim_pre  = eigvals.size();
        reset_owned_buffers(arg_dim_pre, arg_num_rhs, arg_num_regs);
        for (int64_t r = 0; r < num_regs; ++r) {
            T  mu_r = mus[r];
            T* D_r  = D + r*dim_pre;
            T  numerator = eigvals[dim_pre-1] + mu_r;
            for (int i = 0; i < dim_pre; ++i) {
                D_r[i] = (numerator / (eigvals[i] + mu_r)) - 1.0;
            }
        }
        std::copy(eigvecs.begin(), eigvecs.end(), V);
        return;
    }

    void operator()(
        blas::Layout layout, int64_t n, T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc
    ) {
        randblas_require(layout == blas::Layout::ColMajor);
        randblas_require(ldb >= this->m);
        randblas_require(ldc >= this->m);
        if (this->num_regs != 1) {
            randblas_require(n == num_regs);
        } else {
            randblas_require(this->num_rhs >= n);
        }
        // update C = alpha*(V diag(D) V' + I)B + beta*C
        //      Step 1: w = V'B                    with blas::gemm
        //      Step 2: w = D w                    with our own kernel
        //      Step 3: C = beta * C + alpha * B   with blas::copy or blas::scal + blas::axpy
        //      Step 4: C = alpha * V w + C        with blas::gemm
        blas::gemm(layout, blas::Op::Trans, blas::Op::NoTrans, dim_pre, n, m, (T) 1.0, V, m, B, ldb, (T) 0.0, W, dim_pre);
 
        // -----> start step 2
        #define mat_D(_i, _j)  ((num_regs == 1) ? D[(_i)] : D[(_i) + dim_pre*(_j)])
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
                blas::copy(m, colB(i), 1, colC(i), 1);
        } else {
            for (i = 0; i < n; ++i) {
                T* Ci = colC(i);
                blas::scal(m, beta, Ci, 1);
                blas::axpy(m, alpha, colB(i), 1, Ci, 1);
            }
        }
        #undef colB
        #undef colC
        // <----- end step 3
    
        blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, m, n, dim_pre, (T) 1.0, V, m, W, dim_pre, 1.0, C, ldc);
        return;
    }
};

} // end namespace RandLAPACK::linops
