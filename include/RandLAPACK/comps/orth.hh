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

/*
enum decision_orth_stab {use_CholQRQ, use_HQRQ, use_PLUL, use_GEQR};

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
class Orth : public Stabilization<T> {
    public:
        std::vector<T> tvec;
        std::vector<T> tau;
        bool chol_fail;
        bool cond_check;
        bool verbosity;
        decision_orth_stab decision_orth;

        // CholQR-specific
        std::vector<T> Q_gram;
        std::vector<T> Q_gram_cpy;
        std::vector<T> s;

        // Constructor
        Orth(decision_orth_stab dec, bool c_check, bool verb) {
            cond_check = c_check;
            verbosity = verb;
            decision_orth = dec;
            chol_fail = false; 
        };

        int CholQRQ(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        );

        int HQRQ(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            std::vector<T>& tau
        );

        int GEQR(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            std::vector<T>& tvec
        );

        // Control of Orth types calls.
        int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ){
            // Default
            int termination = 0;
            switch(this->decision_orth) 
            {
                case use_CholQRQ:
                    if(this->chol_fail) {
                        termination = HQRQ(m, k, Q, this->tau);
                    } else {
                        //Conventionally, we may call it twice for better orthogonality. Practically, the 2nd call makes no difference.
                        if(CholQRQ(m, k, Q)) {
                                termination = HQRQ(m, k, Q, this->tau);
                        }
                        //termination = CholQRQ(m, k, Q);
                    }
                    break;
                case use_HQRQ:
                    termination = HQRQ(m, k, Q, this->tau);
                    break;
                case use_GEQR: 
#if !defined(__APPLE__)
                    termination = GEQR(m, k, Q, this->tvec);
#else
                    throw(1);  // GEQR not available on macOS; must use HQR.
#endif
            }
            return termination;
        }
};

template <typename T>
class Stab : public Orth<T>
{
    public:
        std::vector<int64_t> ipiv;
        decision_orth_stab decision_stab;
        
        // Constructor
        Stab(decision_orth_stab dec, bool c_check, bool verb) : Orth<T>::Orth(dec, c_check, verb) {
            this->cond_check = c_check;
            this->verbosity = verb;
            decision_stab = dec;
            this->chol_fail = false; 
        };

        int PLUL(
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
            int termination = 0;
            switch(this->decision_stab) 
            {
                case use_CholQRQ:
                    if(this->chol_fail) {
                        termination = PLUL(m, k, Q, this->ipiv);
                    } else {
                        // Only call once
                        termination = this->CholQRQ(m, k, Q);
                        if(termination) {
                            termination = PLUL(m, k, Q, this->ipiv);
                        }
                    }
                    break;
                case use_PLUL:
                    termination = PLUL(m, k, Q, this->ipiv);
                    break;
            }
            return termination;
        }
};
*/
#endif
} // end namespace RandLAPACK::comps::rs
