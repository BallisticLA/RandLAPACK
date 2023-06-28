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

    ExplicitSymLinOp(
        int64_t m,
        blas::Uplo uplo,
        const T* A_buff,
        int64_t lda
    ) : SymmetricLinearOperator<T>(m), uplo(uplo), A_buff(A_buff), lda(lda) {};

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
        blas::symm(
            layout, blas::Side::Left, this->uplo, this->m, n, alpha,
            this->A_buff, this->lda, B, ldb, beta, C, ldc
        );
    };
};


} // end namespace RandLAPACK
#endif