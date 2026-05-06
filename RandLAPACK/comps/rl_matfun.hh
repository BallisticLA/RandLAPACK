#pragma once

#include <cmath>
#include <algorithm>

namespace RandLAPACK {

/// SqrtFun: f(x) = sqrt(max(x, 0)).
/// Clamps to zero to tolerate floating-point roundoff near zero eigenvalues.
template <typename T>
struct SqrtFun {
    T operator()(T x) const { return std::sqrt(std::max(x, (T)0)); }
};

/// LogFun: f(x) = log(x).
/// Undefined for x <= 0. The caller must ensure the matrix is strictly PD
/// (all eigenvalues > 0) before using this function.
template <typename T>
struct LogFun {
    T operator()(T x) const { return std::log(x); }
};

/// PolyFun: f(x) = x*(x + lambda) = x^2 + lambda*x.
///
/// Models tr(K(K + λI)) = tr(K² + λK) for a symmetric PSD kernel matrix K.
/// LanczosFA is exact for this function in d=2 Lanczos steps, since Gauss
/// quadrature integrates polynomials of degree ≤ 2d-1 exactly (d=2 covers
/// degree 3, which contains all degree-2 polynomials).
template <typename T>
struct PolyFun {
    T lam;
    explicit PolyFun(T lambda) : lam(lambda) {}
    T operator()(T x) const { return x * (x + lam); }
};

} // end namespace RandLAPACK
