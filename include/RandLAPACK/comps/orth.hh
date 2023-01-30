#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::orth {

#ifndef ORTH_CLASS
#define ORTH_CLASS

template <typename T>
class Stabilization {
    public:
        virtual int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ) = 0;
};

template <typename T>
class CholQRQ : public Stabilization<T> {
    public:
        bool chol_fail;
        bool cond_check;
        bool verbosity;

        // CholQR-specific
        std::vector<T> Q_gram;
        std::vector<T> Q_gram_cpy;
        std::vector<T> s;

        // Constructor
        CholQRQ(bool c_check, bool verb) {
            cond_check = c_check;
            verbosity = verb;
            chol_fail = false; 
        };

        int cholqrq(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        );

        int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ){
            return cholqrq(m, k, Q);
        }
};


template <typename T>
class HQRQ : public Stabilization<T> {
    public:
        std::vector<T> tau;
        bool cond_check;
        bool verbosity;

        // Constructor
        HQRQ(bool c_check, bool verb) {
            cond_check = c_check;
            verbosity = verb;
        };

        int hqrq(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            std::vector<T>& tau
        );

        int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ){
            return hqrq(m, k, Q, this->tau);;
        }
};

#if !defined(__APPLE__)
template <typename T>
class GEQR : public Stabilization<T> {
    public:
        std::vector<T> tvec;
        bool cond_check;
        bool verbosity;

        // Constructor
        GEQR(bool c_check, bool verb) {
            cond_check = c_check;
            verbosity = verb;
        };

        int geqrq(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            std::vector<T>& tvec
        );

        int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ){
            return geqrq(m, k, Q, this->tvec);
        }
};
#endif

template <typename T>
class PLUL : public CholQRQ<T>
{
    public:
        std::vector<int64_t> ipiv;
        
        // Constructor
        PLUL(bool c_check, bool verb) : CholQRQ<T>::CholQRQ(c_check, verb) {
            this->cond_check = c_check;
            this->verbosity = verb;
            this->chol_fail = false; 
        };

        int plul(
            int64_t m,
            int64_t k,
            std::vector<T>& Q,
            std::vector<int64_t>& ipiv
        );

        // Control of Stab types calls.
        virtual int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ){
            return plul(m, k, Q, this->ipiv);
        }
};

#endif
} // end namespace RandLAPACK::comps::rs
