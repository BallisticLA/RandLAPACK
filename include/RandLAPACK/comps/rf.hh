#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "rs.hh"

namespace RandLAPACK::comps::rf {

#ifndef RF_CLASS
#define RF_CLASS

template <typename T>
class RangeFinder
{
        public:
                // RS1 call
                virtual bool call(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q,
                        bool use_qr
                ) = 0;
                
                // RS1_test_mode call
                virtual void call(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q, // n by k
                        T& cond_num // For testing purposes
                ) = 0;
};

template <typename T>
class RF1 : public RangeFinder<T>
{
	public:
                RandLAPACK::comps::rs::RowSketcher<T>& RS_Obj;
                void(*Orthogonalization)(int64_t, int64_t, std::vector<T>&);
                bool verbosity;
                bool cond_check;

		// Constructor
		RF1(
                        RandLAPACK::comps::rs::RowSketcher<T>& rs_obj,
                        void(*Orth)(int64_t, int64_t, std::vector<T>&),
                        bool verb,
                        bool cond
		) : RS_Obj(rs_obj)
                {
                        Orthogonalization = Orth;
                        verbosity = verb;
                        cond_check = cond;
		}

                virtual bool call(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q,
                        bool use_qr
                );

                virtual void call(
                        int64_t m,
                        int64_t n,
                        const std::vector<T>& A,
                        int64_t k,
                        std::vector<T>& Q, // n by k
                        T& cond_num // For testing purposes
                );
};
#endif
} // end namespace RandLAPACK::comps::rs
