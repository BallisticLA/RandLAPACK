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
	public:
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
		RandLAPACK::comps::orth::Stabilization<T>& Stab_Obj;
		//void(& SketchOpGen)(int64_t, int64_t, T*, int32_t);
		int32_t seed;
		int64_t passes_over_data;
		int64_t passes_per_stab;
		bool verbosity;
		bool cond_check;
		int decision_RS;
		std::vector<T> Omega_1;
		std::vector<T> cond_nums;

		std::vector<T> Omega_cpy;
		std::vector<T> Omega_1_cpy;
		std::vector<T> s;

		RS(
			RandLAPACK::comps::orth::Stabilization<T>& stab_obj,
			//void(& sk_gen)(int64_t, int64_t, T*, int32_t),
			int32_t s, 
			int64_t p, 
			int64_t q,
			bool verb,
			bool cond,
			int decision
		) : Stab_Obj(stab_obj)//, SketchOpGen(sk_gen)
		{
			verbosity = verb;
			cond_check = cond;
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
			switch(this->decision_RS)
			{
				case 0:
					rs1(m, n, A, k, Omega);
					break;
			}
		}
};
#endif
} // end namespace RandLAPACK::comps::rs
