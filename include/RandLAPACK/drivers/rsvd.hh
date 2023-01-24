#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "../comps/qb.hh"

namespace RandLAPACK::drivers::rsvd {

#ifndef RSVD_CLASS
#define RSVD_CLASS

enum decision_RSVD {use_rsvd1};

template <typename T>
class RSVDalg
{
    public:
        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t& k,
            int64_t block_sz,
            T tol,
            std::vector<T>& U,
            std::vector<T>& S,
            std::vector<T>& VT
        ) = 0;
};

template <typename T>
class RSVD : public RSVDalg<T>
{
	public:
        RandLAPACK::comps::qb::QBalg<T>& QB_Obj;
        bool verbosity;

        std::vector<T> Q; 
        std::vector<T> B;
        std::vector<T> U_buf;

        // Controls QB version to be used
        decision_RSVD decision_rsvd;

    // Constructor
    RSVD(
        RandLAPACK::comps::qb::QBalg<T>& qb_obj,
        bool verb,
        decision_RSVD decision
    ) : QB_Obj(qb_obj) {
        verbosity = verb;
        decision_rsvd = decision;
    }

    int RSVD1(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t& k,
        int64_t block_sz,
        T tol,
        std::vector<T>& U,
        std::vector<T>& S,
        std::vector<T>& VT
    );

    virtual int call(
		int64_t m,
		int64_t n,
		std::vector<T>& A,
		int64_t& k,
		int64_t block_sz,
		T tol,
		std::vector<T>& U,
		std::vector<T>& S,
		std::vector<T>& VT
    ) {
		int termination = 0;
		switch(this->decision_rsvd) {
			case use_rsvd1:
				termination = RSVD1(m, n, A, k, block_sz, tol, U, S, VT);
				break;
		}

		if(this->verbosity) {
			switch(termination)
			{
			case 1:
				printf("\nQB TERMINATED VIA: QB failed.\n");
				break;
			case 0:
				printf("\nQB TERMINATED VIA: normal termination.\n");
				break;
			}
		}
		return termination;
    }
};
#endif
} // end namespace RandLAPACK::comps::rsvd