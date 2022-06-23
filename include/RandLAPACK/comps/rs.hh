#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::rs {

#ifndef RS_CLASS
#define RS_CLASS

template <typename T>
class RowSketcher
{
	public:
		int32_t seed;
		int64_t passes_over_data;
		int64_t passes_per_stab;
		void(*stabilizer)(int64_t, int64_t, std::vector<T>&);
		//void(*sketch_gen)(int64_t, int64_t, T*, int32_t);

		// Constructor
		RowSketcher(
			int32_t s, 
			int64_t p, 
			int64_t q, 
			void (*stab)(int64_t, int64_t, std::vector<T>&)//,
			//void (*sk_gen)(int64_t, int64_t, T*, int32_t)
		){
			seed = s;
			passes_over_data = p;
			passes_per_stab = q;
			stabilizer = stab;
			//sketch_gen = sk_gen;
		}

		// Run
		void RS1(
			int64_t m,
			int64_t n,
			const std::vector<T>& A,
			int64_t k,
			std::vector<T>& Omega
		);
};
#endif
} // end namespace RandLAPACK::comps::rs
