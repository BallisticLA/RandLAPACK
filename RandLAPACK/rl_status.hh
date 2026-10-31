#pragma once

// Numerical outcomes (docs/CONVENTIONS.md, "Errors"). A routine returns a Status for what the
// numerics did; it throws RandLAPACK::Error for a violated contract (bad arguments or options).
// Details beyond the category (rank reached, iterations, the failing stage, a LAPACK info value)
// live in the driver's info().

namespace RandLAPACK {

enum class [[nodiscard]] Status : int {
    Success = 0,
    CholeskyFailure,          ///< a Cholesky factorization met a nonpositive or non-finite pivot
    OrthogonalizationFailure, ///< a basis could not be orthonormalized to working accuracy
    NotConverged,             ///< an iteration reached its limit without meeting its test
    Breakdown,                ///< a Krylov or Lanczos recurrence broke down with no recovery
    NotPositiveSemidefinite,  ///< an input required to be positive semidefinite is not
    NonFiniteInput,           ///< an input entry the routine reads is NaN or Inf
    NonFiniteResult,          ///< a NaN or Inf arose during the computation
};

inline const char* to_string(Status s) noexcept {
    switch (s) {
        case Status::Success:                  return "Success";
        case Status::CholeskyFailure:          return "CholeskyFailure";
        case Status::OrthogonalizationFailure: return "OrthogonalizationFailure";
        case Status::NotConverged:             return "NotConverged";
        case Status::Breakdown:                return "Breakdown";
        case Status::NotPositiveSemidefinite:  return "NotPositiveSemidefinite";
        case Status::NonFiniteInput:           return "NonFiniteInput";
        case Status::NonFiniteResult:          return "NonFiniteResult";
    }
    return "unknown Status";
}

} // namespace RandLAPACK

// Test-only fault injection for failure paths that no input triggers deterministically. Library code
// writes `if (RL_TEST_FAULT("qb.block_orth")) ...`; it is the constant false unless the program
// defines RandLAPACK_ENABLE_TEST_HOOKS, as the test suite does (for every translation unit or none).
#ifdef RandLAPACK_ENABLE_TEST_HOOKS
#include <functional>
namespace RandLAPACK::test_hooks {
/// Set by a test; RL_TEST_FAULT(name) returns its answer for the named fault point.
inline std::function<bool(const char*)> fault;
} // namespace RandLAPACK::test_hooks
#define RL_TEST_FAULT(name) (RandLAPACK::test_hooks::fault && RandLAPACK::test_hooks::fault(name))
#else
#define RL_TEST_FAULT(name) false
#endif
