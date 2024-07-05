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
#include "RandLAPACK/misc/rl_pdkernels.hh"

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
#include "RandLAPACK/drivers/rl_cqrrpt.hh"
#include "RandLAPACK/drivers/rl_cqrrp.hh"
#include "RandLAPACK/drivers/rl_revd2.hh"
#include "RandLAPACK/drivers/rl_rbki.hh"

#endif
