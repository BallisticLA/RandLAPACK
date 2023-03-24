#ifndef randlapack_drivers_rsvd_h
#define randlapack_drivers_rsvd_h

#include "rl_qb.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>

namespace RandLAPACK {

template <typename T>
class RSVDalg {
    public:

        virtual ~RSVDalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t& k,
            T tol,
            std::vector<T>& U,
            std::vector<T>& S,
            std::vector<T>& VT
        ) = 0;
};

template <typename T>
class RSVD : public RSVDalg<T> {
    public:

        // Constructor
        RSVD(
            // Requires a QB algorithm object.
            RandLAPACK::QBalg<T>& qb_obj,
            bool verb,
            int64_t b_sz
        ) : QB_Obj(qb_obj) {
            verbosity = verb;
            block_sz = b_sz;
        }

        /// Computes an economy Singular Value Decomposition:
        ///     A = U \Sigma \transpose{V},
        /// where U is m-by-k, V is n-by-k are orthogonal and \Sigma is k-by-k diagonal. 
        /// Relies on the randomized QB factorization algorithm
        ///
        /// Templated for `float` and `double` types.
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] n
        ///     The number of columns in the matrix A.
        ///
        /// @param[in] A
        ///     The m-by-n matrix A, stored in a column-major format.
        ///
        /// @param[in] k
        ///     Expected rank of the matrix A. If unknown, set k=min(m,n).
        ///
        /// @param[in] block_sz
        ///     Block size parameter for randomized QB, block_sz <= k.
        ///
        /// @param[in] tol
        ///     Error tolerance parameter for ||A-QB||_Fro.
        ///
        /// @param[in] U
        ///     Buffer for the U-factor.
        ///     Initially, may not have any space allocated for it.
        ///
        /// @param[in] S
        ///     Buffer for the \Sigma-factor.
        ///     Initially, may not have any space allocated for it.
        ///
        /// @param[in] VT
        ///     Buffer for the \transpose{V}-factor.
        ///     Initially, may not have any space allocated for it.
        ///
        /// @param[out] U
        ///     Stores m-by-k factor U.
        ///
        /// @param[out] S
        ///     Stores k-by-k factor \Sigma.
        ///
        /// @param[out] VT
        ///     Stores k-by-n factor \transpose{V}.
        ///
        /// @returns 0 if successful

        int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t& k,
            T tol,
            std::vector<T>& U,
            std::vector<T>& S,
            std::vector<T>& VT
        ) override;

    public:
        RandLAPACK::QBalg<T>& QB_Obj;
        bool verbosity;
        int64_t block_sz;

        std::vector<T> Q;
        std::vector<T> B;
        std::vector<T> U_buf;
};

// -----------------------------------------------------------------------------
template <typename T>
int RSVD<T>::call(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t& k,
    T tol,
    std::vector<T>& U,
    std::vector<T>& S,
    std::vector<T>& VT
){
    // Q and B sizes will be adjusted automatically
    this->QB_Obj.call(m, n, A, k, this->block_sz, tol, this->Q, this->B);

    // Making sure all vectors are large enough
    util::upsize(m * k, U);
    util::upsize(k * k, this->U_buf);
    util::upsize(k * 1, S);
    util::upsize(k * n, VT);

    // SVD of B
    lapack::gesdd(Job::SomeVec, k, n, this->B.data(), k, S.data(), this->U_buf.data(), k, VT.data(), k);
    // Adjusting U
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, this->Q.data(), m, this->U_buf.data(), k, 0.0, U.data(), m);

    return 0;
}

} // end namespace RandLAPACK
#endif
