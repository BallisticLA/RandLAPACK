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

namespace RandLAPACK {

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
    ) : SymmetricLinearOperator<T>(m), uplo(uplo), A_buff(A_buff), lda(lda), buff_layout(buff_layout) {};

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
    };
};


// template <typename T>
// struct RegularizedSymLinOp  {
//
//     const int64_t m;
//     std::vector<T> regs;
//
//     RegularizedSymLinOp(int64_t m, std::vector<T> &regs) : m(m), regs(regs) {};
//
//     virtual void operator()(T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) = 0;
//
// };

template <typename T>
struct RegExplicitSymLinOp : public SymmetricLinearOperator<T> {

    const T* A_buff;
    const int64_t lda;
    std::vector<T> regs;
    static const blas::Uplo uplo = blas::Uplo::Upper;
    static const blas::Layout buff_layout = blas::Layout::ColMajor;

    RegExplicitSymLinOp(
        int64_t m, const T* A_buff, int64_t lda, std::vector<T> &regs
    ) : SymmetricLinearOperator<T>(m), A_buff(A_buff), lda(lda), regs(regs) {
        randblas_require(lda >= m);
    };

    void operator()(blas::Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        randblas_require(layout == this->buff_layout);
        randblas_require(ldb >= this->m);
        randblas_require(ldc >= this->m);
        blas::symm(layout, blas::Side::Left, this->uplo, this->m, n, alpha, this->A_buff, this->lda, B, ldb, beta, C, ldc);
        int64_t num_regs = this->regs.size();
        if (num_regs != 1)
            randblas_require(n == num_regs);
        T* regsp = regs.data();
        for (int64_t i = 0; i < n; ++i) {
            T coeff =  alpha * regsp[std::min(i, num_regs - 1)];
            blas::axpy(this->m, coeff, B + i*ldb, 1, C +  i*ldc, 1);
        }
        return;
    };

};

} // end namespace RandLAPACK
