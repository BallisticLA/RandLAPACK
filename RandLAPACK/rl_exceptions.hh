#pragma once

#include <exception>
#include <sstream>
#include <string>

// RandLAPACK error infrastructure. Throws a distinct RandLAPACK::Error so
// callers can catch RandLAPACK failures independently of RandBLAS failures.
//
// Usage at call sites uses iostream-style chaining:
//
//   randlapack_require(b_sz >= 1)
//       << "BQRRP::call: b_sz=" << b_sz << " must be positive";
//
// On the happy path (cond is true) the macro short-circuits with zero cost;
// no temporary is constructed and no stream is touched. On failure, a
// StreamThrower temporary is built, the operator<< chain populates its
// stringstream, and the temporary's destructor throws RandLAPACK::Error at
// the end of the full expression. The message can be omitted entirely
// (`randlapack_require(cond);`) — the destructor falls back to the
// stringified condition.

namespace RandLAPACK {

// -----------------------------------------------------------------------------
/// Minimalist exception class for RandLAPACK errors.
///
/// @verbatim embed:rst:leading-slashes
///  Typically triggered by a statement of the form
///  ``randlapack_require(cond) << "message"`` where ``cond`` was false.
///  See ``RandLAPACK/rl_exceptions.hh`` for the macro definition.
///
///  RandLAPACK errors are distinct from ``RandBLAS::Error``; catch each type
///  separately if a caller cares which library raised the problem.
/// @endverbatim
///
class Error : public std::exception {
private:
    std::string msg_;
public:
    Error() : std::exception() {}
    Error(std::string const &msg) : Error() { msg_ = msg; }

    // Constructs RandLAPACK error with message: "msg, in function func"
    Error(std::string const &msg, const char* func)
        : Error(msg + ", in function " + func) {}

    // ---------------------------------------------------------------
    /// Returns a C-string representation of this error's message.
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }
};

} // namespace RandLAPACK

namespace RandLAPACK::exceptions::internal {

// RAII helper used by randlapack_require. Constructed only when the required
// condition fails; destructor throws RandLAPACK::Error at end of full
// expression after operator<< populates the stream.
class StreamThrower {
    const char* condstr_;
    const char* func_;
    std::ostringstream ss_;
public:
    StreamThrower(const char* condstr, const char* func)
        : condstr_(condstr), func_(func) {}

    StreamThrower(const StreamThrower&) = delete;
    StreamThrower& operator=(const StreamThrower&) = delete;

    template <typename T>
    StreamThrower& operator<<(const T& v) { ss_ << v; return *this; }

    [[noreturn]] ~StreamThrower() noexcept(false) {
        std::string msg = ss_.str();
        if (msg.empty()) msg = condstr_;
        throw Error(msg, func_);
    }
};

} // namespace RandLAPACK::exceptions::internal


// Macro: throws RandLAPACK::Error if cond is false. Supports an optional
// stream-style message appended via operator<<.
// Examples:
//   randlapack_require(m >= 0);
//   randlapack_require(m >= n) << "BQRRP requires m >= n; got m=" << m
//                              << ", n=" << n;
//
// The `for` form (instead of `if (cond) ; else ...`) is deliberate: it
// shields the macro from -Wdangling-else when used inside an unbraced outer
// `if`, while preserving the zero-cost happy path (the loop body is not
// entered when cond holds) and the throw-on-destruct semantics of the
// StreamThrower temporary.
#define randlapack_require(cond) \
    for (; !(cond); ) ::RandLAPACK::exceptions::internal::StreamThrower(#cond, __func__)
