#pragma once

// Convenience header — includes all linear operator components.
//
// Functions templated on LinearOperator or SymmetricLinearOperator concepts
// must work with any conforming linop type. Since the caller — not the
// function — decides which concrete linop to pass, these functions cannot
// know in advance which linop headers they need. Including this single
// header gives them access to every linop type without coupling to specific
// implementations.

#include "rl_concepts.hh"
#include "rl_sparse_views.hh"
#include "rl_dense_linop.hh"
#include "rl_sparse_linop.hh"
#include "rl_composite_linop.hh"
#include "rl_sym_linops.hh"
#include "rl_materialize.hh"
#include "rl_matfun_linops.hh"
