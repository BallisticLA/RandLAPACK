#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "rs.hh"
#include "orth.hh"

namespace RandLAPACK::comps::rf {

#ifndef RF_CLASS
#define RF_CLASS

/*
template <typename T>
class RangeFinder
{
        public:
                virtual void call<T>(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q
                ) = 0;    
};
*/
template <typename T>
class RF //: public RangeFinder<T>
{
	public:
                // Instantiated in the constructor
                RandLAPACK::comps::rs::RS<T>& RS_Obj;
                RandLAPACK::comps::orth::Orth<T>& Orth_Obj;
                bool verbosity;
                bool cond_check;
                
                // Controls RF version to be used
                int decision_RF;

                // Implementation-specific vars
                T cond_num;
                bool use_qr;

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
                void call(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q
                ){
                        switch(RF::decision_RF)
                        {
                                case 0:
                                        rf1(m, n, A, k, Q, RF::use_qr);
                                        break;

                                case 1:
                                        rf1_test_mode(m, n, A, k, Q, RF::cond_num);
                                        break;
                        }
                }
};
#endif
} // end namespace RandLAPACK::comps::rs
