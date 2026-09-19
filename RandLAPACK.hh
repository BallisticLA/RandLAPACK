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
#include "RandLAPACK/comps/rl_cholqr.hh"

// Drivers
#include "RandLAPACK/drivers/rl_rsvd.hh"
#include "RandLAPACK/drivers/rl_cqrrt.hh"      // dense CQRRT and the operator-based CQRRTO_linops
#include "RandLAPACK/drivers/rl_cholqr_linops.hh"
#include "RandLAPACK/drivers/rl_iter_refine_lsq.hh"
// Both of these declare themselves "Public API" in their headers but were reachable
// only by including them directly, which is part of why neither had any test coverage.
#include "RandLAPACK/drivers/rl_lsqr.hh"
#include "RandLAPACK/drivers/rl_restarted_pcg_ne.hh"
#include "RandLAPACK/drivers/rl_blendenpik.hh"
#include "RandLAPACK/drivers/rl_scholqr3_linops.hh"
#include "RandLAPACK/drivers/rl_cqrrpt.hh"
#include "RandLAPACK/drivers/rl_bqrrp.hh"
#include "RandLAPACK/drivers/rl_revd2.hh"
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
