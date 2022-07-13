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
                virtual void call(
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
                Orth(int decision = 0) : decision_orth(decision) {};

                void CholQR(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q
                );

                void HQR(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        std::vector<T>& tau
                );

                void GEQR(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        std::vector<T>& tvec
                );

                // Control of Orth types calls.
                void call(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q
                ){
                        switch(this->decision_orth)
                        {
                                case 0:
                                        if(this->chol_fail)
                                        {
                                                HQR(m, k, Q, this->tau);
                                        }
                                        else{
                                                //Call it twice for better orthogonality
                                                CholQR(m, k, Q);
                                                CholQR(m, k, Q);
                                        }
                                        break;
                                case 1:
                                        HQR(m, k, Q, this->tau);
                                        break;
                                case 2: 
                                        GEQR(m, k, Q, this->tvec);
                        }
                }
};

template <typename T>
class Stab : public Orth<T>
{
	public:
                std::vector<int64_t> ipiv;
                int decision_stab;
                
                // Constructor
                Stab(int decision = 0) : Orth<T>::Orth(decision), decision_stab(decision) {};

                void PLU(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q,
                        std::vector<int64_t>& ipiv
                );

                // Control of Stab types calls.
                virtual void call(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q
                ){
                        switch(this->decision_stab)
                        {
                                case 0:
                                        if(this->chol_fail)
                                        {
                                                PLU(m, k, Q, this->ipiv);
                                        }
                                        else{
                                                // Only call once
                                                this->CholQR(m, k, Q);
                                        }
                                        break;
                                case 1:
                                        PLU(m, k, Q, this->ipiv);
                                        break;
                        }
                }
};
#endif
} // end namespace RandLAPACK::comps::rs
