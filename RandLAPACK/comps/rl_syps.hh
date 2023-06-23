#ifndef randlapack_comps_syps_h
#define randlapack_comps_syps_h

#include "rl_orth.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstdio>

namespace RandLAPACK {

template <typename T, typename RNG>
class SymmetricPowerSketch {
    public:
        virtual ~SymmetricPowerSketch() {}

        virtual RandBLAS::DenseSkOp<T,RNG> call(
            blas::Uplo uplo,
            int64_t m,
            const std::vector<T>& A,
            int64_t lda,
            int64_t k,
            RandBLAS::RNGState<RNG> state,
            T* skop_buff = nullptr,
            T* work_buff = nullptr
        ) { // TODO: try to remove this implementation and set = 0.
            // I don't remember why we needed a concrete implementation.
            UNUSED(uplo); UNUSED(m); UNUSED(A); UNUSED(lda);
            UNUSED(k); UNUSED(state); UNUSED(skop_buff); UNUSED(work_buff);
            throw std::logic_error("Abstract method called.");
        };

        virtual RandBLAS::DenseSkOp<T,RNG> call(
            blas::Uplo uplo,
            int64_t m,
            const T* A,
            int64_t lda,
            int64_t k,
            RandBLAS::RNGState<RNG> state,
            T* skop_buff = nullptr,
            T* work_buff = nullptr
        ) {
            UNUSED(uplo); UNUSED(m); UNUSED(A); UNUSED(lda);
            UNUSED(k); UNUSED(state); UNUSED(skop_buff); UNUSED(work_buff);
            throw std::logic_error("Abstract method called.");
        };

};


template <typename T, typename RNG>
class SYPS : public SymmetricPowerSketch<T, RNG> {
    public:

        SYPS(
            int64_t p, // number of passes
            int64_t q, // passes per stabilization
            bool verb,
            bool cond
        ) {
            verbose = verb;
            cond_check = cond;
            passes_over_data = p;
            passes_per_stab = q;
        }

        /// Return an m-by-k matrix Y for use in sketching a symmetric matrix mat(A).
        /// The qualitative goal is that the range of Y should be well-aligned with
        /// the dominant k eigenvectors of A. This function takes "passes_over_data" steps
        /// of a power method that starts with a random Gaussian matrix.
        /// We stabilize the power method with a user-defined function.
        ///
        /// @param[in] uplo
        ///     blas::Uplo::Upper or blas::Uplo::Lower.
        ///     The triangular part of mat(A) that we can read from A.
        ///
        /// @param[in] m
        ///     The matrix mat(A) is m-by-m.
        ///
        /// @param[in] A
        ///     Buffer that partially defines the matrix mat(A).
        ///
        /// @param[in] lda
        ///     Leading dimension of mat(A) when reading from A.
        ///
        /// @param[in] k
        ///     The number of columns in the sketch.
        ///
        /// @param[in] state
        ///     The RNGState used to define the data-oblivious sketching operator at the
        ///     start of the power iteration.a
        ///
        /// @param[out] skop_buff
        ///     Optional. If provided, must have length >= m*k, and upon exit its contents
        ///     will underpin the returned sketching operator. If not provided, we will
        ///     allocate this memory and assign ownership to the returned sketching operator.
        ///
        /// @param[in,out] work_buff
        ///     Optional. If provided, must have length >= m*k. If not provided, we
        ///     will allocate a buffer of size m*k and deallocate before we return.
        ///
        /// @returns
        ///     An RNGState that the calling function should use the next
        ///     time it needs an RNGState.
        ///
        RandBLAS::DenseSkOp<T,RNG> call(
            blas::Uplo uplo,
            int64_t m,
            const std::vector<T>& A,
            int64_t lda,
            int64_t k,
            RandBLAS::RNGState<RNG> state,
            T* skop_buff,
            T* work_buff
        );

        RandBLAS::DenseSkOp<T,RNG> call(
            blas::Uplo uplo,
            int64_t m,
            const T* A,
            int64_t lda,
            int64_t k,
            RandBLAS::RNGState<RNG> state,
            T* skop_buff,
            T* work_buff
        );
    
        int64_t passes_over_data;
        int64_t passes_per_stab;
        bool verbose;
        bool cond_check;
        std::vector<T> cond_nums;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
RandBLAS::DenseSkOp<T,RNG> SYPS<T, RNG>::call(
    blas::Uplo uplo,
    int64_t m,
    const T* A,
    int64_t lda,
    int64_t k,
    RandBLAS::RNGState<RNG> state,
    T* skop_buff,
    T* work_buff
){
    int64_t p = this->passes_over_data;
    int64_t q = this->passes_per_stab;
    int64_t p_done = 0;

     bool callers_skop_buff = skop_buff != nullptr;
     if (!callers_skop_buff)
         skop_buff = new T[m * k];
    RandBLAS::DenseDist D{m, k};
    auto next_state = RandBLAS::fill_dense(D, skop_buff, state);

     bool callers_work_buff = work_buff != nullptr;
     if (!callers_work_buff)
         work_buff = new T[m * k];
    RandBLAS::util::safe_scal(m * k, 0.0, work_buff, 1);

    T *symm_out = work_buff;
    T *symm_in  = skop_buff;
    int64_t* ipiv = new int64_t[m]{};
    while (p - p_done > 0) {
        blas::symm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, k, 1.0, A, lda, symm_in, m, 0.0, symm_out, m);
        ++p_done;
        if (p_done % q == 0) {
                if(lapack::getrf(m, k, symm_out, m, ipiv))
                    throw std::runtime_error("Sketch did not have an LU decomposition.");
                util::get_L(m, k, symm_out, 1);
                lapack::laswp(k, symm_out, m, 1, k, ipiv, 1);
        }
        symm_out = (p_done % 2 == 1) ? skop_buff : work_buff;
        symm_in  = (p_done % 2 == 1) ? work_buff : skop_buff; 
    }
    delete[] ipiv;
    if (p % 2 == 1)
        blas::copy(m * k, work_buff, 1, skop_buff, 1);

    auto S = RandBLAS::DenseSkOp<T>(D, state, skop_buff);
    S.next_state = next_state;

    if (!callers_work_buff)
        delete[] work_buff;

   return S;
}

template <typename T, typename RNG>
RandBLAS::DenseSkOp<T,RNG> SYPS<T, RNG>::call(
    blas::Uplo uplo,
    int64_t m,
    const std::vector<T>& A,
    int64_t lda,
    int64_t k,
    RandBLAS::RNGState<RNG> state,
    T* skop_buff,
    T* work_buff
) {
    this->call(uplo, m, A.data(), lda, k, state, skop_buff, work_buff);
}


} // end namespace RandLAPACK
#endif
