// The umbrella header: RandLAPACK's single public entry point, installed to
// <prefix>/include. RandLAPACK is header only (an INTERFACE CMake target), so there is
// nothing to link and this is what tests, benchmarks and downstream projects include.
// Sections below follow the source tree in dependency order. The testing utilities are
// public on purpose: the benchmark project consumes them like the tests do. HQRRP is
// not listed; it arrives transitively through rl_cqrrpt.hh and rl_bqrrp.hh.

#ifndef RANDLAPACK_HH
#define RANDLAPACK_HH

// config and dependencies
#include "RandLAPACK/rl_blaspp.hh"
#include "RandLAPACK/rl_lapackpp.hh"
#include "RandLAPACK/rl_exceptions.hh"
#include "RandBLAS.hh"

// misc
#include "RandLAPACK/misc/rl_util.hh"
#include "RandLAPACK/misc/rl_pdkernels.hh"

// linear operator infrastructure
#include "RandLAPACK/linops/rl_linops.hh"

// testing utilities (used by benchmarks, so included in umbrella header)
#include "RandLAPACK/testing/rl_gen.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"

// Computational routines
#include "RandLAPACK/comps/rl_determiter.hh"
#include "RandLAPACK/comps/rl_preconditioners.hh"
#include "RandLAPACK/comps/rl_qb.hh"
#include "RandLAPACK/comps/rl_rf.hh"
#include "RandLAPACK/comps/rl_rs.hh"
#include "RandLAPACK/comps/rl_syps.hh"
#include "RandLAPACK/comps/rl_syrf.hh"
#include "RandLAPACK/comps/rl_orth.hh"
#include "RandLAPACK/comps/rl_rpchol.hh"

// Drivers
#include "RandLAPACK/drivers/rl_rsvd.hh"
#include "RandLAPACK/drivers/rl_cqrrt.hh"
#include "RandLAPACK/drivers/rl_cholqr_linops.hh"
#include "RandLAPACK/drivers/rl_cqrrt_linops.hh"
#include "RandLAPACK/drivers/rl_scholqr3_linops.hh"
#include "RandLAPACK/drivers/rl_cqrrpt.hh"
#include "RandLAPACK/drivers/rl_bqrrp.hh"
#include "RandLAPACK/drivers/rl_revd2.hh"
#include "RandLAPACK/comps/rl_lanczos_fa.hh"
#include "RandLAPACK/comps/rl_lanczos_fa_block.hh"
#include "RandLAPACK/comps/rl_lanczos_qfa.hh"
#include "RandLAPACK/comps/rl_lanczos_qfa_block.hh"
#include "RandLAPACK/drivers/rl_nystrom_evd.hh"
#include "RandLAPACK/drivers/rl_fun_nystrom_pp.hh"
#include "RandLAPACK/drivers/rl_abrik.hh"
#include "RandLAPACK/drivers/rl_krill.hh"

// GPU layer. __CUDACC__ is set only while a CUDA compiler is processing this file, so
// .cu translation units get the GPU drivers and host .cc files build with no CUDA
// toolkit on the include path. rl_cuda_kernels.cuh must come first: it decides whether
// the kernels exist, and #pragma once means the first inclusion is the one that counts.
#if defined(__CUDACC__)
#include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"
#include "RandLAPACK/drivers/rl_cqrrpt_gpu.hh"
#include "RandLAPACK/drivers/rl_bqrrp_gpu.hh"
#endif

#endif
