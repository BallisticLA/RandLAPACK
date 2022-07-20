#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::orth {

#ifndef ORTH_CLASS
#define ORTH_CLASS

template <typename T>
class Stabilization
{
        public:
                virtual int call(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q
                ) = 0;
};

template <typename T>
class Orth : public Stabilization<T>
{
	public:
                std::vector<T> tvec;
                std::vector<T> tau;
                bool chol_fail;
                bool cond_check;
                bool verbosity;
                int decision_orth;

                // CholQR-specific
                std::vector<T> Q_gram;
                std::vector<T> Q_gram_cpy;
                std::vector<T> s;

                // Constructor
                Orth(int decision, bool c_check, bool verb) 
                {
                        cond_check = c_check;
                        verbosity = verb;
                        decision_orth = decision;
                        chol_fail = false; 
                };

                int CholQR(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q
                );

                int HQR(
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
                                case 0:
                                        if(this->chol_fail)
                                        {
                                                termination = HQR(m, k, Q, this->tau);
                                        }
                                        else{
                                                //Call it twice for better orthogonality
                                                if(CholQR(m, k, Q))
                                                {
                                                        termination = HQR(m, k, Q, this->tau);
                                                }
                                                termination = CholQR(m, k, Q);
                                        }
                                        break;
                                case 1:
                                        termination = HQR(m, k, Q, this->tau);
                                        break;
                                case 2: 
                                        termination = GEQR(m, k, Q, this->tvec);
                        }
                        return termination;
                }
};

template <typename T>
class Stab : public Orth<T>
{
	public:
                std::vector<int64_t> ipiv;
                int decision_stab;
                
                // Constructor
                Stab(int decision, bool c_check, bool verb) : Orth<T>::Orth(decision, c_check, verb) 
                {
                        this->cond_check = c_check;
                        this->verbosity = verb;
                        decision_stab = decision;
                        this->chol_fail = false; 
                };

                int PLU(
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
                                case 0:
                                        if(this->chol_fail)
                                        {
                                                termination = PLU(m, k, Q, this->ipiv);
                                        }
                                        else{
                                                // Only call once
                                                termination = this->CholQR(m, k, Q);
                                                if(termination)
                                                {
                                                        termination = PLU(m, k, Q, this->ipiv);
                                                }
                                        }
                                        break;
                                case 1:
                                        termination = PLU(m, k, Q, this->ipiv);
                                        break;
                        }
                        return termination;
                }
};
#endif
} // end namespace RandLAPACK::comps::rs
