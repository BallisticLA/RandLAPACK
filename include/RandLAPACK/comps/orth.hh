#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::orth {

#ifndef ORTH_CLASS
#define ORTH_CLASS

template <typename T>
class Orth
{
	public:
                std::vector<T> tau;
                bool chol_fail;
                int decision_orth;
                // Constructor
                Orth(int decision)
                {
                        decision_orth = decision;
                }

                void CholQR(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q // pointer to the beginning
                );

                void HQR(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        std::vector<T>& tau
                );

                // Control of Orth types calls.
                void call(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q
                ){
                        switch(Orth::decision_orth)
                        {
                                case 0:
                                        CholQR(m, k, Q);
                                        break;

                                case 1:
                                        HQR(m, k, Q, Orth::tau);
                                        break;
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
                Stab(int decision) : Orth<T>::Orth(decision)
                {
                        decision_stab = decision;
                }      

                void PLU(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q,
                        std::vector<int64_t>& ipiv
                );

                // Control of Stab types calls.
                void call(
                        int64_t m,
                        int64_t k,
                        std::vector<T>& Q
                ){
                        switch(Stab::decision_stab)
                        {
                                case 0:
                                        this -> CholQR(m, k, Q);
                                        break;

                                case 1:
                                        PLU(m, k, Q, Stab::ipiv);
                                        break;
                        }
                }
};

#endif
} // end namespace RandLAPACK::comps::rs