/*
TODO: 
	1. Figure out how to use QBalg istead of QB everywhere.
*/
#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "orth.hh"
#include "rf.hh"

namespace RandLAPACK::comps::qb {

#ifndef QB_CLASS
#define QB_CLASS

template <typename T>
class QBalg
{
        virtual void call(
                int64_t m,
                int64_t n,
                std::vector<T>& A,
                int64_t& k,
                int64_t block_sz,
                T tol,
                std::vector<T>& Q,
                std::vector<T>& B
        ) = 0;
};

template <typename T>
class QB : public QBalg<T>
{
	public:
                RandLAPACK::comps::rf::RF<T>& RF_Obj;
                RandLAPACK::comps::orth::Orth<T>& Orth_Obj;
                bool verbosity;
                bool orth_check;

                // Avoiding preallocations
                std::vector<T> Q_gram;
                std::vector<T> Q_i_gram;

                std::vector<T> QtQi; 
                std::vector<T> Q_i;
                std::vector<T> B_i;

                // Controls QB version to be used
                int decision_QB;

                // Implementation-specific vars
                std::vector<T> cond_nums;

                // Output
                int termination;

		// Constructor
		QB(
                        RandLAPACK::comps::rf::RF<T>& rf_obj,
                        RandLAPACK::comps::orth::Orth<T>& orth_obj,
                        bool verb,
                        bool orth,
                        int decision
		) : RF_Obj(rf_obj), Orth_Obj(orth_obj)
                {
                        verbosity = verb;
                        orth_check = orth;
                        decision_QB = decision;
		}

		void QB2(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        int64_t& k,
                        int64_t block_sz,
                        T tol,
                        std::vector<T>& Q,
                        std::vector<T>& B
                );

                void QB2_test_mode(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        int64_t& k,
                        int64_t block_sz,
                        T tol,
                        std::vector<T>& Q,
                        std::vector<T>& B
                );

                virtual void call(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        int64_t& k,
                        int64_t block_sz,
                        T tol,
                        std::vector<T>& Q,
                        std::vector<T>& B
                )
                {
                        switch(this->decision_QB)
                        {
                                case 0:
                                        QB2(m, n, A, k, block_sz, tol, Q, B);
                                        break;
                                case 1:
                                        QB2_test_mode(m, n, A, k, block_sz, tol, Q, B);
                                        break;
                        }
                }
};
#endif
} // end namespace RandLAPACK::comps::rs
