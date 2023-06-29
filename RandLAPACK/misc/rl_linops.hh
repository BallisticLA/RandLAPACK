#ifndef randlapack_linops_h
#define randlapack_linops_h

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
        blas::Layout layout,
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
            blas_call_uplo = (this->uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
        // Reading the "blas_call_uplo" triangle of "this->A_buff" in "layout" order is the same
        // as reading the "this->uplo" triangle of "this->A_buff" in "this->buff_layout" order.
        blas::symm(
            layout, blas::Side::Left, blas_call_uplo, this->m, n, alpha,
            this->A_buff, this->lda, B, ldb, beta, C, ldc
        );
    };
};


} // end namespace RandLAPACK
#endif
