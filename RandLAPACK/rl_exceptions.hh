#pragma once

#include <exception>
#include <cstdarg>
#include <cstdio>
#include <string>

// Most code in this file is structurally adapted from RandBLAS/exceptions.hh
// (which itself was adapted from BLAS++). The macros and helpers here throw a
// distinct RandLAPACK::Error so that callers can catch RandLAPACK failures
// independently of RandBLAS failures.

namespace RandLAPACK {

// -----------------------------------------------------------------------------
/// Minimalist exception class for RandLAPACK errors.
///
/// @verbatim embed:rst:leading-slashes
///  These are typically triggered by statements of the form
///  ``randlapack_require(cond)`` where ``cond`` was false, or
///  ``randlapack_error_if_msg(cond, fmt, ...)`` where ``cond`` was true.
///  See ``RandLAPACK/rl_exceptions.hh`` for the macro definitions.
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
    Error(const char* msg, const char* func)
        : Error(std::string(msg) + ", in function " + func) {}

    // ---------------------------------------------------------------
    /// Returns a C-string representation of this error's message.
    /// It's common to wrap this function's return value with std::string.
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }
};

} // namespace RandLAPACK

namespace RandLAPACK::exceptions {

namespace internal {

// -----------------------------------------------------------------------------
// internal helper: throws RandLAPACK::Error if cond is true.
// Called by the randlapack_error_if macro.
inline void throw_if(bool cond, const char* condstr, const char* func) {
    if (cond) {
        throw Error(condstr, func);
    }
}

#if defined(_MSC_VER)
    #define RandLAPACK_ATTR_FORMAT(I, F)
#else
    #define RandLAPACK_ATTR_FORMAT(I, F) __attribute__((format(printf, I, F)))
#endif

#define RandLAPACK_ERROR_MESSAGE_SIZE 256

// -----------------------------------------------------------------------------
// internal helper: printf-style; throws RandLAPACK::Error if cond is true.
// Called by the randlapack_error_if_msg macro.
// condstr is captured for symmetry with the non-message overload but ignored
// here; the formatted buffer carries the user-supplied message instead.
inline void throw_if(bool cond, const char* condstr, const char* func, const char* format, ...)
    RandLAPACK_ATTR_FORMAT(4, 5);

inline void throw_if(bool cond, const char* condstr, const char* func, const char* format, ...) {
    (void) condstr;
    if (cond) {
        char buf[RandLAPACK_ERROR_MESSAGE_SIZE];
        va_list va;
        va_start(va, format);
        vsnprintf(buf, sizeof(buf), format, va);
        va_end(va);
        throw Error(buf, func);
    }
}

#undef RandLAPACK_ATTR_FORMAT

} // namespace internal


// Macro: throws RandLAPACK::Error if cond is true.
// Example: randlapack_error_if(m < 0);
#define randlapack_error_if(cond) \
    RandLAPACK::exceptions::internal::throw_if((cond), #cond, __func__)

// Macro: printf-style; throws RandLAPACK::Error if cond is true.
// Example: randlapack_error_if_msg(m < 0, "m must be >= 0; got m=%ld", m);
#define randlapack_error_if_msg(cond, ...) \
    RandLAPACK::exceptions::internal::throw_if((cond), #cond, __func__, __VA_ARGS__)

// Macro: throws RandLAPACK::Error if cond is false (i.e. the required
// invariant did not hold).
// Example: randlapack_require(m >= 0);
#define randlapack_require(cond) \
    RandLAPACK::exceptions::internal::throw_if(!(cond), "(" #cond ") was required, but did not hold", __func__)


} // namespace RandLAPACK::exceptions
