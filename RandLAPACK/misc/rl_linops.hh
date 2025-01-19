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
    vector<T> regs;
    bool      _eval_includes_reg;

    static const blas::Uplo uplo = blas::Uplo::Upper;
    static const blas::Layout buff_layout = blas::Layout::ColMajor;
    using scalar_t = T;

    RegExplicitSymLinOp(
        int64_t m, const T* A_buff, int64_t lda, vector<T> &regs
    ) : SymmetricLinearOperator<T>(m), A_buff(A_buff), lda(lda), regs(regs) {
        randblas_require(lda >= m);
        _eval_includes_reg = false;
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
        T val;
        if (i > j) {
            val = A_buff[j + i*lda];
        } else {
            val = A_buff[i + j*lda];
        }
        if (_eval_includes_reg) {
            randblas_require(regs.size() == 1);
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
    int64_t k;
    int64_t s;
    vector<T> V;
    T* V_ptr;
    vector<T> D;
    T* D_ptr;
    vector<T> work;
    T* work_ptr;
    int64_t num_regs = 1;

    /* Suppose we want to precondition a positive semidefinite matrix G_mu = G + mu*I.
     *
     * Once properly preparred, this preconditioner represents a linear operator of the form
     *      P = V diag(D) V' + I.
     * The columns of V approximate the top k eigenvectors of G, while the 
     * entries of D are *functions of* the corresponding approximate eigenvalues.
     * 
     * The specific form of the entries of D are as follows. Suppose we start with
     * (V, lambda) as approximations of the top k eigenpairs of G, and define the vector
     *      D0 = (min(lambda) + mu) / (lambda + mu).
     * From a mathematical perspective, this preconditioner represents the linear operator
     *      P = V diag(D0) V' + (I - VV').
     * The action of this linear operator can be computed with two calls to GEMM
     * instead of three if we store D = D0 - 1 instead of D0 itself.
     */

    SpectralPrecond(int64_t m)
        : m(m), k(1), s(1),
          V(k * m), V_ptr(V.data()),
          D(k), D_ptr(D.data()),
          work(k * s), work_ptr(work.data()) {}

    // Move constructor
    // Call as SpectralPrecond<T> spc(std::move(other)) when we want to transfer the
    // contents of "other" to "this". 
    SpectralPrecond(SpectralPrecond &&other) noexcept
        : m(other.m), k(other.k), s(other.s),
          V(std::move(other.V)), V_ptr(V.data()),
          D(std::move(other.D)), D_ptr(D.data()),
          work(std::move(other.work)), work_ptr(work.data()),
          num_regs(other.num_regs) {}

    // Copy constructor
    // Call as SpectralPrecond<T> spc(other) when we want to copy "other".
    SpectralPrecond(const SpectralPrecond &other)
        : m(other.m), k(other.k), s(other.s),
          V(other.V), V_ptr(V.data()),
          D(other.D), D_ptr(D.data()),
          work(other.work), work_ptr(work.data()),
          num_regs(other.num_regs) {} 

    void prep(vector<T> &eigvecs, vector<T> &eigvals, vector<T> &mus, int64_t arg_s) {
        // assume eigvals are positive numbers sorted in decreasing order.
        num_regs = mus.size();
        randblas_require(num_regs == 1 || num_regs == arg_s);
        k = eigvals.size();
        D.resize(k * num_regs);

        s = arg_s;
        V = eigvecs;
        V_ptr = V.data();
        work.resize(k * s);
        work_ptr = work.data();

        D_ptr = D.data();
        for (int64_t r = 0; r < num_regs; ++r) {
            T  mu_r = mus[r];
            T* D_r  = &D_ptr[r*k];
            T  numerator = eigvals[k-1] + mu_r;
            for (int i = 0; i < k; ++i)
                D_r[i] = (numerator / (eigvals[i] + mu_r)) - 1.0;
        }
        return;
    }

    void evaluate(int64_t s, const T *x, T *dest) {
        operator()(blas::Layout::ColMajor, s, (T) 1.0, x, m, (T) 0.0, dest, m);
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
            randblas_require(this->s >= n);
        }
        // update C = alpha*(V diag(D) V' + I)B + beta*C
        //      Step 1: w = V'B                    with blas::gemm
        //      Step 2: w = D w                    with our own kernel
        //      Step 3: C = beta * C + alpha * B   with blas::copy or blas::scal + blas::axpy
        //      Step 4: C = alpha * V w + C        with blas::gemm
        blas::gemm(layout, blas::Op::Trans, blas::Op::NoTrans, k, n, m, (T) 1.0, V_ptr, m, B, ldb, (T) 0.0, work_ptr, k);
 
        // -----> start step 2
        #define mat_D(_i, _j)    ((num_regs == 1) ? D_ptr[(_i)] : D_ptr[(_i) + k*(_j)])
        #define mat_work(_i, _j) work_ptr[(_i) + k*(_j)]
        for (int64_t j = 0; j < n; j++) {
            for (int64_t i = 0; i < k; i++) {
                mat_work(i, j) = mat_D(i, j) * mat_work(i, j);
            }
        }
        #undef mat_D
        #undef mat_work
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
    
        blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k, (T) 1.0, V_ptr, m, work_ptr, k, 1.0, C, ldc);
        return;
    }
};

} // end namespace RandLAPACK::linops
