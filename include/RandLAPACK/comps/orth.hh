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
class Orth : public Stabilization<T> // TODO #1
{
	public:
                std::vector<T> tvec;
                std::vector<T> tau;
                bool chol_fail;
                int decision_orth;

                // CholQR-specific
                std::vector<T> Q_gram;

                // Constructor
                Orth(int decision = 0) : decision_orth(decision) {chol_fail = false;};

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
                int decision_stab;
                
                // Constructor
                Stab(int decision = 0) : Orth<T>::Orth(decision), decision_stab(decision) {this->chol_fail = false;};

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
