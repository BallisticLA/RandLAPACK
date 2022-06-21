#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "rs.hh"

namespace RandLAPACK::comps::rf {

template <typename T>
void rf1(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // n by k
	uint32_t seed,
        T& cond_num // For testing purposes
);

template <typename T>
bool rf1_safe(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // n by k
        bool use_qr,
	uint32_t seed
);

#ifndef RF_CLASS
#define RF_CLASS

template <typename T>
class RangeFinder
{
	public:
                RandLAPACK::comps::rs::RowSketcher<T>* RS_Obj;
                void(*Orthogonalization)(int64_t, int64_t, std::vector<T>&);
                bool verbosity;
                bool cond_check;

		// Constructor
		RangeFinder(
                        RandLAPACK::comps::rs::RowSketcher<T> rs_obj,
                        void(*Orth)(int64_t, int64_t, std::vector<T>&),
                        bool verb,
                        bool cond
		){
                        RS_Obj = &rs_obj;
                        Orthogonalization = Orth;
                        verbosity = verb;
                        cond_check = cond;
		}

		void RF1_test_mode(
			int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q, // n by k
                        T& cond_num // For testing purposes
		);

                bool RF1(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q, // n by k
                        bool use_qr
                );
};

/*
template <typename T>
class Two
{
    public:
        RandLAPACK::comps::rs::One<T>* Obj;

        // Constructor
        int J;

        Two(int j, RandLAPACK::comps::rs::One<T> One_Obj)
        {
            J = j;
            Obj = &One_Obj;
        }

        void do_more_stuff(T more_var);
};
*/
#endif
} // end namespace RandLAPACK::comps::rs
