/*
TODO: 
	1. Figure out how to use RangeFinder istead of RF everywhere.
*/
#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "rs.hh"
#include "orth.hh"

namespace RandLAPACK::comps::rf {

#ifndef RF_CLASS
#define RF_CLASS

template <typename T>
class RangeFinder
{
        public:
                virtual void call(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q
                ) = 0;    
};

template <typename T>
class RF : public RangeFinder<T>
{
	public:
                // Instantiated in the constructor
                RandLAPACK::comps::rs::RS<T>& RS_Obj;
                RandLAPACK::comps::orth::Orth<T>& Orth_Obj;
                bool verbosity;
                bool cond_check;
                std::vector<T> Omega;
                // Avoiding reallocation
                std::vector<T> Q_cpy;
                std::vector<T> s;

                // Controls RF version to be used
                int decision_RF;

                // Implementation-specific vars
                T cond_num; // Condition nuber of a sketch
                bool use_qr; // Use QR if CholQR fails - technically, can just alter the decifion variable

		// Constructor
		RF(
                        RandLAPACK::comps::rs::RS<T>& rs_obj,
                        RandLAPACK::comps::orth::Orth<T>& orth_obj,
                        bool verb,
                        bool cond,
                        int decision
		) : RS_Obj(rs_obj), Orth_Obj(orth_obj)
                {
                        verbosity = verb;
                        cond_check = cond;
                        decision_RF = decision;
		}

                void rf1(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q,
                        bool use_qr
                );

                void rf1_test_mode(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q,
                        T& cond_num
                );

                // Control of RF types calls.
                virtual void call(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q
                ){
                        switch(this->decision_RF)
                        {
                                case 0:
                                        rf1(m, n, A, k, Q, this->use_qr);
                                        break;
                                case 1:
                                        rf1_test_mode(m, n, A, k, Q, this->cond_num);
                                        break;
                        }
                }
};
#endif
} // end namespace RandLAPACK::comps::rs
