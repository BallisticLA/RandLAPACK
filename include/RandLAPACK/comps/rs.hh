/*
TODO: 
	1. Figure out how to use RowSketcher istead of RS everywhere.
*/
#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "orth.hh"

namespace RandLAPACK::comps::rs {

#ifndef RS_CLASS
#define RS_CLASS

template <typename T>
class RowSketcher
{
	virtual void call(
		int64_t m,
		int64_t n,
		const std::vector<T>& A,
		int64_t k,
		std::vector<T>& Omega 
	) = 0;
};

template <typename T>
class RS : public RowSketcher<T>
{
	public:
		RandLAPACK::comps::orth::Stab<T>& Stab_Obj;
		//void(& SketchOpGen)(int64_t, int64_t, T*, int32_t);
		int32_t seed;
		int64_t passes_over_data;
		int64_t passes_per_stab;
		int decision_RS;

		RS(
			RandLAPACK::comps::orth::Stab<T>& stab_obj,
			//void(& sk_gen)(int64_t, int64_t, T*, int32_t),
			int32_t s, 
			int64_t p, 
			int64_t q,
			int decision
		) : Stab_Obj(stab_obj)//, SketchOpGen(sk_gen)
		{
			seed = s;
			passes_over_data = p;
			passes_per_stab = q;
			decision_RS = decision;
		}

		void rs1(
			int64_t m,
			int64_t n,
			const std::vector<T>& A,
			int64_t k,
			std::vector<T>& Omega 
		);

		virtual void call(
			int64_t m,
			int64_t n,
			const std::vector<T>& A,
			int64_t k,
			std::vector<T>& Omega 
		){
			switch(RS::decision_RS)
			{
				case 0:
					rs1(m, n, A, k, Omega);
					break;
			}
		}
};
#endif
} // end namespace RandLAPACK::comps::rs
