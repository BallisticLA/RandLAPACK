// Explicit instantiations of the GPU drivers for float and double, the counterpart of
// test/misc/test_instantiations.cc for the CUDA test executable.

#include "RandLAPACK.hh"

using RNG = RandBLAS::DefaultRNG;

template class RandLAPACK::BQRRP_GPU<float, RNG>;
template class RandLAPACK::BQRRP_GPU<double, RNG>;
template class RandLAPACK::CQRRPT_GPU<float, RNG>;
template class RandLAPACK::CQRRPT_GPU<double, RNG>;
