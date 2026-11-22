// Explicit float and double instantiations: a template contributes no lines to a coverage report
// until something instantiates it, so this file covers class templates and member and function
// templates that the tests miss. It also compiles the float versions no test builds.

#include "RandLAPACK.hh"

using RNG = RandBLAS::DefaultRNG;
template <typename T> using CSR = RandBLAS::sparse_data::CSRMatrix<T>;
template <typename T> using DenseComposite =
    RandLAPACK::linops::CompositeOperator<RandLAPACK::linops::DenseLinOp<T>, RandLAPACK::linops::DenseLinOp<T>>;

// operator() overloads whose other operand is a sparse matrix or a sketching operator.
#define RL_INSTANTIATE_APPLY(T, Op_t, B_t) \
    template void Op_t::operator()<B_t>(Layout, blas::Op, blas::Op, int64_t, int64_t, int64_t, \
                                        T, B_t&, T, T*, int64_t);

#define RL_INSTANTIATE(T) \
    template class RandLAPACK::CholQRQ<T>; \
    template class RandLAPACK::HQRQ<T>; \
    template class RandLAPACK::PLUL<T>; \
    template class RandLAPACK::RS<T, RNG>; \
    template class RandLAPACK::RF<T, RNG>; \
    template class RandLAPACK::QB<T, RNG>; \
    template class RandLAPACK::RSVD<T, RNG>; \
    template class RandLAPACK::SYPS<T, RNG>; \
    template class RandLAPACK::SYRF<RandLAPACK::SYPS<T, RNG>, RandLAPACK::HQRQ<T>>; \
    template class RandLAPACK::BK<T, RNG>; \
    template class RandLAPACK::ABRIK<T, RNG>; \
    template class RandLAPACK::BQRRP<T, RNG>; \
    template class RandLAPACK::CQRRPT<T, RNG>; \
    template struct RandLAPACK::StatefulFrobeniusNorm<T>; \
    template struct RandLAPACK::linops::RBFKernelMatrix<T>; \
    template struct RandLAPACK::linops::CSRRowBlockView<T>; \
    template struct RandLAPACK::linops::CSRColBlock<T>; \
    template struct RandLAPACK::linops::CSCColBlockView<T>; \
    template struct RandLAPACK::linops::CSCRowBlock<T>; \
    template struct RandLAPACK::linops::SparseLinOp<CSR<T>>; \
    template struct RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>>; \
    template class RandLAPACK::linops::DowndatableLinOp<T, RandLAPACK::linops::DenseLinOp<T>>; \
    template struct RandLAPACK::linops::CompositeOperator< \
        RandLAPACK::linops::DenseLinOp<T>, RandLAPACK::linops::DenseLinOp<T>>; \
    template int RandLAPACK::BK<T, RNG>::call<CSR<T>>( \
        int64_t, int64_t, CSR<T>&, int64_t, int64_t, \
        T*&, T*&, T*&, T*&, int64_t&, int64_t&, bool&, RandBLAS::RNGState<RNG>&); \
    RL_INSTANTIATE_APPLY(T, RandLAPACK::linops::SparseLinOp<CSR<T>>, CSR<T>) \
    RL_INSTANTIATE_APPLY(T, RandLAPACK::linops::SparseLinOp<CSR<T>>, RandBLAS::DenseSkOp<T>) \
    RL_INSTANTIATE_APPLY(T, DenseComposite<T>, CSR<T>) \
    RL_INSTANTIATE_APPLY(T, DenseComposite<T>, RandBLAS::DenseSkOp<T>) \
    template void RandLAPACK::standardize_dataset<T>(int64_t, int64_t, T*, T*, T*, bool);

RL_INSTANTIATE(float)
RL_INSTANTIATE(double)
