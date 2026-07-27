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
#include "RandLAPACK/drivers/rl_cqrrt.hh"     // holds both dense CQRRT and CQRRT_linops
#include "RandLAPACK/drivers/rl_cholqr_linops.hh"
#include "RandLAPACK/drivers/rl_iter_refine_lsq.hh"
// Both of these declare themselves "Public API" in their headers but were reachable
// only by including them directly, which is part of why neither had any test coverage.
#include "RandLAPACK/drivers/rl_lsqr.hh"
#include "RandLAPACK/drivers/rl_blendenpik.hh"
#include "RandLAPACK/drivers/rl_scholqr3_linops.hh"
#include "RandLAPACK/drivers/rl_cqrrpt.hh"
#include "RandLAPACK/drivers/rl_bqrrp.hh"
#include "RandLAPACK/drivers/rl_revd2.hh"
#include "RandLAPACK/drivers/rl_abrik.hh"
#include "RandLAPACK/drivers/rl_krill.hh"

// Cuda functions - issues with linking/visibility when present if the below is uncommented.
// A temporary fix is to add the below directly in the test/benchmark files.
// Ideally, we would like below to be uncommented so that we could simply include RandLAPACK.hh everywhere.
//#include "RandLAPACK/drivers/rl_cqrrpt_gpu.hh"
//#include "RandLAPACK/drivers/rl_cqrrp_gpu.hh"
//#include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"

#endif
