#pragma once

#include <cmath> 
#include <ostream>
#include <iomanip>
#include <limits>


namespace richol {

using std::abs;
using std::sqrt;

struct dd {
    double hi, lo;  

    // constructors
    dd()           : hi(0.0),         lo(0.0) {}
    dd( double a ) : hi(a),           lo(0.0) {}
    dd( double a, double b ) {
        if (abs(a) >= abs(b)) {
          hi = a; lo = b;
        } else {
          hi = b; lo = a;
        }
    }
    dd( const dd &other ) : hi(other.hi), lo(other.lo) {}
    dd& operator=( const dd &other ) {
      if (this != &other) {
        hi = other.hi;
        lo = other.lo;
      }
      return *this;
    }
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
    os << a.hi;
    return os;
}

}

using richol::dd;
using richol::sqrt;
using richol::abs;
using richol::real;
using richol::imag;

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

  static dd epsilon() noexcept {
    return dd(0.0, std::numeric_limits<double>::denorm_min());
  }

};

} // namespace std
