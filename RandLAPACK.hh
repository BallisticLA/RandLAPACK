#ifndef RANDLAPACK_HH
#define RANDLAPACK_HH

// config and dependencies
#include "RandLAPACK/rl_blaspp.hh"
#include "RandLAPACK/rl_lapackpp.hh"
#include "RandBLAS.hh"

// misc
#include "RandLAPACK/misc/rl_util.hh"
#include "RandLAPACK/misc/rl_linops.hh"
#include "RandLAPACK/misc/rl_gen.hh"

// Computational routines
#include "RandLAPACK/comps/rl_determiter.hh"
#include "RandLAPACK/comps/rl_preconditioners.hh"
#include "RandLAPACK/comps/rl_qb.hh"
#include "RandLAPACK/comps/rl_rf.hh"
#include "RandLAPACK/comps/rl_rs.hh"
#include "RandLAPACK/comps/rl_syps.hh"
#include "RandLAPACK/comps/rl_syrf.hh"
#include "RandLAPACK/comps/rl_orth.hh"

// Drivers
#include "RandLAPACK/drivers/rl_rsvd.hh"
#include "RandLAPACK/drivers/rl_cqrrpt.hh"
#include "RandLAPACK/drivers/rl_bqrrp.hh"
#include "RandLAPACK/drivers/rl_revd2.hh"
#include "RandLAPACK/drivers/rl_rbki.hh"

// Cuda functions - issues with linking/visibility when present if the below is uncommented.
// A temporary fix is to add the below directly in the test/benchmark files.
// Ideally, we would like below to be uncommented so that we could simply include RandLAPACK.hh everywhere.
//#include "RandLAPACK/drivers/rl_cqrrpt_gpu.hh"
//#include "RandLAPACK/drivers/rl_cqrrp_gpu.hh"
//#include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"

#endif
