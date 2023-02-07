#ifndef RANDLAPACK_HH
#define RANDLAPACK_HH

/// For bothe computational routine and driver classes,
/// we adapt a "one algorithm per class" paradigm. 
/// We then use a virtual class + anonimous "call"
/// function interface for universal access of an algroith mype.

// Computational routines
#include <RandLAPACK/comps/determiter.hh>
#include <RandLAPACK/comps/qb.hh>
#include <RandLAPACK/comps/rf.hh>
#include <RandLAPACK/comps/rs.hh>
#include <RandLAPACK/comps/util.hh>
#include <RandLAPACK/comps/orth.hh>

// Drivers
#include <RandLAPACK/drivers/rsvd.hh>
#include <RandLAPACK/drivers/cholqrcp.hh>
#include <RandLAPACK/drivers/hqrrp.hh>
#endif
