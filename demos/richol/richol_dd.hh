#pragma once

#include <cmath> 
#include <ostream>
#include <iomanip>
#include <limits>
#include <cmath>

namespace richol {

struct dd {
    double hi, lo;  

    // constructors
    dd()           : hi(0.0),         lo(0.0) {}
    dd( double a ) : hi(a),           lo(0.0) {}
    dd( double a, double b ) : hi(a), lo(b) {}
    template <typename T>
    dd( T a ) : hi((double)a), lo(0.0) {}

    // Error‐free transform: twoSum
    //   computes s = a + b exactly split into hi‐part and lo‐part
    static inline dd twoSum(double a, double b) {
        dd res;
        res.hi = a + b;
        double bv = res.hi - a;
        res.lo = (a - (res.hi - bv)) + (b - bv);
        return res;
    }

    // Error‐free transform: fastTwoSum
    //   assumes |a| >= |b|, computes s = a + b exactly
    static inline dd fastTwoSum(double a, double b) {
        dd res;
        res.hi = a + b;
        res.lo = b - (res.hi - a);
        return res;
    }

    // Dekker’s multiplication to get hi, lo of a*b
    static inline dd twoProd(double a, double b) {
        dd res;
        res.hi = a * b;
        // you can replace the next lines with
        //   res.lo = std::fma(a, b, -res.hi);
        // if your platform has a correct hardware/software fma.
        const double SPLIT = 134217729.0;  // 2^27 + 1
        double a_hi, a_lo, b_hi, b_lo;
        double c = SPLIT * a;
        a_hi = c - (c - a);
        a_lo = a - a_hi;
        c    = SPLIT * b;
        b_hi = c - (c - b);
        b_lo = b - b_hi;
        res.lo = ((a_hi * b_hi - res.hi) 
                  + a_hi * b_lo 
                  + a_lo * b_hi) 
                 + a_lo * b_lo;
        return res;
    }

    // construction from sum of hi‐ and lo-parts
    static inline dd makeFromSum(double hi_part, double lo_part) {
        // first do a fast two‐sum of hi_part + lo_part
        return fastTwoSum(hi_part, lo_part);
    }

    // addition
    dd operator+(const dd &b) const {
        dd s = twoSum(hi, b.hi);     // s.hi = hi + b.hi, s.lo = error
        s.lo += lo + b.lo;           // accumulate the small parts
        return fastTwoSum(s.hi, s.lo);
    }

    // subtraction
    dd operator-(const dd &b) const {
        dd s = twoSum(hi, -b.hi);
        s.lo += lo - b.lo;
        return fastTwoSum(s.hi, s.lo);
    }

    // multiplication
    dd operator*(const dd &b) const {
        dd p = twoProd(hi, b.hi);
        // cross terms
        p.lo += hi * b.lo + lo * b.hi;
        return fastTwoSum(p.hi, p.lo);
    }

    // division via one Newton step on 1/b
    dd operator/(const dd &b) const {
        double q1 = hi / b.hi;             // approximate quotient
        dd prod = b * dd(q1);              // b*q1
        dd   r  = *this - prod;            // remainder
        double q2 = (r.hi + r.lo) / b.hi;  // correction term
        return twoSum(q1, q2);
    }

    // compound‐assignment
    dd &operator+=(const dd &b) { *this = *this + b; return *this; }
    dd &operator-=(const dd &b) { *this = *this - b; return *this; }
    dd &operator*=(const dd &b) { *this = *this * b; return *this; }
    dd &operator/=(const dd &b) { *this = *this / b; return *this; }

    // unary minus
    dd operator-() const { return dd(-hi, -lo); }

    // convert back to a normal double (with rounding)
    double to_double() const { return hi + lo; }
};

//---------------------------------------------------------------------
// Comparison operators for double-double (dd)
//---------------------------------------------------------------------

// equality
inline bool operator==(const dd &a, const dd &b) noexcept {
  return (a.hi == b.hi) && (a.lo == b.lo);
}

// inequality
inline bool operator!=(const dd &a, const dd &b) noexcept {
  return !(a == b);
}

// less-than
inline bool operator<(const dd &a, const dd &b) noexcept {
  if      (a.hi <  b.hi) return true;
  else if (a.hi >  b.hi) return false;
  else                   return a.lo < b.lo;
}

// greater-than
inline bool operator>(const dd &a, const dd &b) noexcept {
  return b < a;
}

// less-than or equal
inline bool operator<=(const dd &a, const dd &b) noexcept {
  return !(b < a);
}

// greater-than or equal
inline bool operator>=(const dd &a, const dd &b) noexcept {
  return !(a < b);
}

/// free‐function sqrt for dd (found by ADL)
inline dd sqrt(const dd &a) {
    // domain checks
    if (a.hi < 0.0) {
        return dd(std::nan(""));  
    }
    if (a.hi == 0.0 && a.lo == 0.0) {
        return dd(0.0);
    }
    double x0 = std::sqrt(a.hi);
    dd x(x0);
    x = (x + a/x) * dd(0.5);
    x = (x + a/x) * dd(0.5);
    return x;
}

inline dd abs( const dd &a ) noexcept {
    if (a.hi < 0.0 || (a.hi == 0.0 && a.lo < 0.0)) {
        return -a;
    }
    return a;
}

inline dd imag( const dd &a ) { return 0.0; }
inline dd real( const dd &a ) { return a;   }

inline std::ostream & operator<<(std::ostream &os, dd const &a) {
    // save state
    std::ios_base::fmtflags  f = os.flags();
    std::streamsize          p = os.precision();

    // we print hi and lo each with max_digits10 so that
    // a double→string→double round-trip is exact
    constexpr int D = std::numeric_limits<double>::max_digits10;
    os << "(" 
       << std::setprecision(D) << a.hi 
       << ", " 
       << std::setprecision(D) << a.lo 
       << ")";

    // restore state
    os.flags(f);
    os.precision(p);
    return os;
}

}


// inline bool operator==(const richol::dd &a, double b) noexcept { return (a.hi == b) && (a.lo == 0.0); }
// inline bool operator==(double a, const richol::dd &b) noexcept { return (a == b.hi) && (b.lo == 0.0); }
// inline bool operator!=(const richol::dd &a, double b) noexcept { return !(a == b); }
// inline bool operator!=(double a, const richol::dd &b) noexcept { return !(a == b); }
// inline bool operator<(const richol::dd &a, double b) noexcept { 
//   if (a.hi < b) return true; 
//   if (a.hi > b) return false; 
//   return a.lo < 0.0; 
// }
// inline bool operator<(double a, const richol::dd &b) noexcept {
//   if (a < b.hi) return true;
//   if (a > b.hi) return false;
//   return 0.0 < b.lo;
// }
// inline bool operator>(const richol::dd &a, double b) noexcept { return b < a; }
// inline bool operator>(double a, const richol::dd &b) noexcept { return b < a; }
// inline bool operator<=(const richol::dd &a, double b) noexcept { return !(b < a); }
// inline bool operator<=(double a, const richol::dd &b) noexcept { return !(b < a); }
// inline bool operator>=(const richol::dd &a, double b) noexcept { return !(a < b); }
// inline bool operator>=(double a, const richol::dd &b) noexcept { return !(a < b); }

using richol::dd;
using richol::sqrt;
using richol::abs;
using richol::real;
using richol::imag;
using std::abs;
using std::sqrt;

namespace std {

template<>
class numeric_limits<dd> {
public:
  // required to tell <limits> you really specialize it
  static constexpr bool is_specialized = true;

  // number of radix-digits in the mantissa
  //    hi has 53 bits, lo up to 52 → p ≈ 105
  static constexpr int digits = 105;
  static constexpr int digits10 = 31;    // floor(digits * log10(2))

  // double-double is signed, not an integer
  static constexpr bool is_signed    = true;
  static constexpr bool is_integer   = false;
  static constexpr bool is_exact     = false;

  // do we have infinities and NaNs?
  static constexpr bool has_infinity       = std::numeric_limits<double>::has_infinity;
  static constexpr bool has_quiet_NaN      = std::numeric_limits<double>::has_quiet_NaN;
  static constexpr bool has_signaling_NaN  = std::numeric_limits<double>::has_signaling_NaN;

  // the usual goodies
  static constexpr float_denorm_style has_denorm = std::numeric_limits<double>::has_denorm;
  static constexpr bool                has_denorm_loss = false;

  // smallest positive normalized dd
  static dd min() noexcept {
    // same exponent range as double, but twice the precision
    return dd(std::numeric_limits<double>::min());
  }

  // largest finite dd
  static dd max() noexcept {
    return dd(std::numeric_limits<double>::max(),
              std::numeric_limits<double>::max() * std::numeric_limits<double>::epsilon());
  }

  // machine‐epsilon for dd:
  //   the difference between 1.0 and the next representable dd > 1.0
  //
  //   in a normalized double‐double the smallest increment at 1.0
  //   lives entirely in the low word, and is exactly
  //      std::numeric_limits<double>::denorm_min()
  static dd epsilon() noexcept {
    return dd(0.0, std::numeric_limits<double>::denorm_min());
  }

//   // you can fill in more if you need them:
//   static dd round_error() noexcept { return dd(0.5); }
//   static int   min_exponent       = std::numeric_limits<double>::min_exponent;
//   static int   max_exponent       = std::numeric_limits<double>::max_exponent;
//   static int   min_exponent10     = std::numeric_limits<double>::min_exponent10;
//   static int   max_exponent10     = std::numeric_limits<double>::max_exponent10;

};

} // namespace std
