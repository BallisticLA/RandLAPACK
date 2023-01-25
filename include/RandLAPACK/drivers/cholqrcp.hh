#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::drivers::cholqrcp {

#ifndef CholQRCP_CLASS
#define CholQRCP_CLASS

enum decision_CholQRCP {use_cholqrcp1};

template <typename T>
class CholQRCPalg {
    public:
        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t d,
            std::vector<T>& R,
            std::vector<int64_t>& J
        ) = 0;
};

template <typename T>
class CholQRCP : public CholQRCPalg<T> {
	public:
        bool verbosity;
        bool timing;
        uint32_t seed;
        T eps;
        int64_t rank;
        int64_t b_sz;

        // 10 entries
        std::vector<long> times;

        // tuning SASOS
        int num_threads;
        int64_t nnz;

        // Buffers
        std::vector<T> A_cpy;
        std::vector<T> Omega;
        std::vector<T> A_hat;
        std::vector<T> tau;
        std::vector<T> R_sp;
        std::vector<T> R_buf;

        std::vector<T> Q_cpy;

        // Controls QB version to be used
        decision_CholQRCP decision_cholqrcp;

    // Constructor
    CholQRCP(
        bool verb,
        bool t,
        uint32_t sd,
        T ep,
        decision_CholQRCP decision
    ) {
        verbosity = verb;
        timing = t;
        seed = sd;
        eps = ep;
        decision_cholqrcp = decision;
    }

    int CholQRCP1(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t d,
        std::vector<T>& R,
        std::vector<int64_t>& J
    );

    virtual int call(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t d,
        std::vector<T>& R,
        std::vector<int64_t>& J
    ) {
        int termination = 0;
        switch(this->decision_cholqrcp) {
            case use_cholqrcp1:
                termination = CholQRCP1(m, n, A, d, R, J);
                break;
        }

        if(this->verbosity) {
            switch(termination) {
            case 1:
                printf("\nCholQRCP TERMINATED VIA: 1.\n");
                break;
            case 0:
                printf("\nCholQRCP TERMINATED VIA: normal termination.\n");
                break;
            }
        }
        return termination;
    }
};
#endif
} // end namespace RandLAPACK::comps::rsvd