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
                virtual int call(
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
                RandLAPACK::comps::rs::RowSketcher<T>& RS_Obj;
                RandLAPACK::comps::orth::Stabilization<T>& Orth_Obj;
                bool verbosity;
                bool cond_check;
                std::vector<T> Omega;

                std::vector<T> Q_cpy;
                std::vector<T> s;

                // Controls RF version to be used
                int decision_RF;

                // Implementation-specific vars
                std::vector<T> cond_nums; // Condition nubers of sketches

		// Constructor
		RF(
                        RandLAPACK::comps::rs::RowSketcher<T>& rs_obj,
                        RandLAPACK::comps::orth::Stabilization<T>& orth_obj,
                        bool verb,
                        bool cond,
                        int decision
		) : RS_Obj(rs_obj), Orth_Obj(orth_obj)
                {
                        verbosity = verb;
                        cond_check = cond;
                        decision_RF = decision;
		}

                int rf1(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q
                );

                // Control of RF types calls.
                virtual int call(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q
                ){
                        int termination = 0;
                        switch(this->decision_RF)
                        {
                                case 0:
                                        termination = rf1(m, n, A, k, Q);
                                        break;
                        }

                        if(this->verbosity)
                        {
                                switch(termination)
                                {
                                case 0:
                                        printf("\nRF TERMINATED VIA: Normal termination.\n");
                                        break;
                                case 1:
                                        printf("\nRF TERMINATED VIA: RowSketcher failed.\n");
                                        break;
                                case 2:
                                        printf("\nRF TERMINATED VIA: Orthogonalization failed.\n");
                                        break;
                                }
                        }
                        return termination;
                }
};
#endif
} // end namespace RandLAPACK::comps::rs
