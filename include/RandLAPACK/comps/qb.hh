#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "rf.hh"

namespace RandLAPACK::comps::qb {

#ifndef QB_CLASS
#define QB_CLASS

template <typename T>
class QB
{
	public:
                RandLAPACK::comps::rf::RangeFinder<T>& RF_Obj;
                // Only used in QB2_test_mode
                void(*Orthogonalization)(int64_t, int64_t, std::vector<T>&);
                bool verbosity;
                bool orth_check;
                bool cond_check;

		// Constructor
		QB(
                        RandLAPACK::comps::rf::RangeFinder<T>& rf_obj,
                        void(*Orth)(int64_t, int64_t, std::vector<T>&),
                        bool verb,
                        bool orth,
                        bool cond
		) : RF_Obj(rf_obj)
                {
                        Orthogonalization = Orth;
                        verbosity = verb;
                        orth_check = orth;
                        cond_check = cond;
		}

		void QB1(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q,
                        std::vector<T>& B 
                );

                int QB2_test_mode(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        int64_t& k,
                        int64_t block_sz,
                        T tol,
                        std::vector<T>& Q,
                        std::vector<T>& B,
                        std::vector<T>& cond_nums
                );

                int QB2(
                        int64_t m,
                        int64_t n,
                        std::vector<T>& A,
                        int64_t& k,
                        int64_t block_sz,
                        T tol,
                        std::vector<T>& Q,
                        std::vector<T>& B
                );
};
#endif

} // end namespace RandLAPACK::comps::rs