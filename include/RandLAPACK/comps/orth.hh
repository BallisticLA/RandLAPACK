#ifndef randlapack_comps_orth_h
#define randlapack_comps_orth_h

#include "blaspp.h"

namespace RandLAPACK::comps::orth {

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

template <typename T>
class PLUL : public Stabilization<T>
{
    public:
        std::vector<int64_t> ipiv;
        bool cond_check;
        bool verbosity;
        
        // Constructor
        PLUL(bool c_check, bool verb) {
            this->cond_check = c_check;
            this->verbosity = verb;
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

} // end namespace RandLAPACK::comps::rs
#endif
