#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include "../comps/qb.hh"

namespace RandLAPACK::drivers::rsvd {

#ifndef RSVD_CLASS
#define RSVD_CLASS

template <typename T>
class RSVDalg {
    public:
        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t& k,
            T tol,
            std::vector<T>& U,
            std::vector<T>& S,
            std::vector<T>& VT
        ) = 0;
};

template <typename T>
class RSVD : public RSVDalg<T> {
	public:
        RandLAPACK::comps::qb::QBalg<T>& QB_Obj;
        bool verbosity;
        int64_t block_sz;

        std::vector<T> Q; 
        std::vector<T> B;
        std::vector<T> U_buf;

    // Constructor
    RSVD(
        // Requires a QB algorithm object.
        RandLAPACK::comps::qb::QBalg<T>& qb_obj,
        bool verb,
        int64_t b_sz
    ) : QB_Obj(qb_obj) {
        verbosity = verb;
        block_sz = b_sz;
    }

    int RSVD1(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t& k,
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
        T tol,
        std::vector<T>& U,
        std::vector<T>& S,
        std::vector<T>& VT
    ) {
        int termination = RSVD1(m, n, A, k, tol, U, S, VT);

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
