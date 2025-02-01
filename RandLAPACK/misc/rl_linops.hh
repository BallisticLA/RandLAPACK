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
#include <concepts>


namespace RandLAPACK::linops {


template<typename LinOp, typename T = LinOp::scalar_t>
concept SymmetricLinearOperator = requires(LinOp A) {
    { A.dim }  -> std::same_as<const int64_t&>;
    // It's recommended that A also have const int64_t members n_rows and n_cols,
    // both equal to A.dim.
} && requires(LinOp A, Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
    // A SYMM-like function that updates C := alpha A*B + beta C, where
    // B and C have n columns and are stored in layout order with strides (ldb, ldc).
    //
    // If layout is ColMajor then an error will be thrown if min(ldb, ldc) < A.dim.
    //
    { A(layout, n, alpha, B, ldb, beta, C, ldc) } -> std::same_as<void>;
};


template <typename T>
struct ExplicitSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const Uplo uplo;
    const T* A_buff;
    const int64_t lda;
    const Layout buff_layout;

    ExplicitSymLinOp(
        int64_t m,
        Uplo uplo,
        const T* A_buff,
        int64_t lda,
        Layout buff_layout
    ) : m(m), dim(dim), uplo(uplo), A_buff(A_buff), lda(lda), buff_layout(buff_layout) {}

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
        randblas_require(ldb >= this->m);
        randblas_require(ldc >= this->m);
        auto blas_call_uplo = this->uplo;
        if (layout != this->buff_layout)
            blas_call_uplo = (this->uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
        // Reading the "blas_call_uplo" triangle of "this->A_buff" in "layout" order is the same
        // as reading the "this->uplo" triangle of "this->A_buff" in "this->buff_layout" order.
        blas::symm(
            layout, Side::Left, blas_call_uplo, this->m, n, alpha,
            this->A_buff, this->lda, B, ldb, beta, C, ldc
        );
    }

    inline T operator()(int64_t i, int64_t j) {
        randblas_require(this->uplo == Uplo::Upper && this->buff_layout == Layout::ColMajor);
        if (i > j) {
            return A_buff[j + i*lda];
        } else {
            return A_buff[i + j*lda];
        }
    }
};

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
        int64_t m, const T* A_buff, int64_t lda, T* arg_regs, int64_t arg_num_ops
    ) : m(m), dim(dim), A_buff(A_buff), lda(lda) {
        randblas_require(lda >= m);
        _eval_includes_reg = false;
        num_ops = arg_num_ops;
        num_ops = std::max(num_ops, (int64_t) 1);
        regs = new T[num_ops]{};
        std::copy(arg_regs, arg_regs, regs);
    }

    RegExplicitSymLinOp(
        int64_t m, const T* A_buff, int64_t lda, std::vector<T> &arg_regs
    ) : RegExplicitSymLinOp<T>(m, A_buff, lda, arg_regs.data(), static_cast<int64_t>(arg_regs.size())) {}

    ~RegExplicitSymLinOp() {
        if (regs != nullptr) delete [] regs;
    }

    void set_eval_includes_reg(bool eir) {
        _eval_includes_reg = eir;
    }

    void operator()(Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        randblas_require(layout == this->buff_layout);
        randblas_require(ldb >= this->m);
        randblas_require(ldc >= this->m);
        blas::symm(layout, blas::Side::Left, this->uplo, this->m, n, alpha, this->A_buff, this->lda, B, ldb, beta, C, ldc);

        if (_eval_includes_reg) {
            if (num_ops != 1) randblas_require(n == num_ops);
            for (int64_t i = 0; i < n; ++i) {
                T coeff =  alpha * regs[std::min(i, num_ops - 1)];
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
            randblas_require(num_ops == 1);
            val += regs[0];
        }
        return val;
    }

};

template<typename T>
struct SpectralPrecond {

    using scalar_t = T; 
    const int64_t m;
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

    SpectralPrecond(int64_t m) : m(m), dim_pre(0), num_rhs(0) {}

    // Move constructor
    // Call as SpectralPrecond<T> spc(std::move(other)) when we want to transfer the
    // contents of "other" to "this". 
    SpectralPrecond(SpectralPrecond &&other) noexcept
        : m(other.m), dim_pre(other.dim_pre), num_rhs(other.num_rhs), num_ops(other.num_ops)
    {
        std::swap(V, other.V);
        std::swap(D, other.D);
        std::swap(W, other.W);
    }

    // Copy constructor
    // Call as SpectralPrecond<T> spc(other) when we want to copy "other".
    SpectralPrecond(const SpectralPrecond &other)
        : m(other.m), dim_pre(other.dim_pre), num_rhs(other.num_rhs),  num_ops(other.num_ops)
     {
        reset_owned_buffers(dim_pre, num_rhs, num_ops);
        std::copy(other.V, other.V + m * dim_pre,        V);
        std::copy(other.D, other.D + dim_pre * num_ops, D);
     } 

    ~SpectralPrecond() {
        if (D != nullptr) delete [] D;
        if (V != nullptr) delete [] V;
        if (W != nullptr) delete [] W;
    }

    void reset_owned_buffers(int64_t arg_dim_pre, int64_t arg_num_rhs, int64_t arg_num_ops) {
        randblas_require(arg_num_rhs == arg_num_ops || arg_num_ops == 1);

        if (arg_dim_pre * arg_num_ops > dim_pre * num_ops) {
            if (D != nullptr) delete [] D;
            D = new T[arg_dim_pre * arg_num_ops]{};
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
        randblas_require(layout == Layout::ColMajor);
        randblas_require(ldb >= this->m);
        randblas_require(ldc >= this->m);
        if (this->num_ops != 1) {
            randblas_require(n == num_ops);
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
