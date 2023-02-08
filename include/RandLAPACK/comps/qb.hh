#ifndef randlapack_comps_qb_h
#define randlapack_comps_qb_h

#include "blaspp.h"
#include "orth.hh"
#include "rf.hh"

namespace RandLAPACK::comps::qb {

template <typename T>
class QBalg {
    public:
        virtual int call(
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
class QB : public QBalg<T> {
    public:
            RandLAPACK::comps::rf::RangeFinder<T>& RF_Obj;
            RandLAPACK::comps::orth::Stabilization<T>& Orth_Obj;
            bool verbosity;
            bool orth_check;

            std::vector<T> Q_gram;
            std::vector<T> Q_i_gram;

            std::vector<T> QtQi; 
            std::vector<T> Q_i;
            std::vector<T> B_i;

            //This represents how much space is currently allocated for cols of Q and rows of B.
            //This is <= k. We are assuming that the user may not have given "enough"
            //space when allocating Q, B initially.
            
            int64_t curr_lim;

            // By how much are we increasing the dimension when we've reached curr_lim
            int dim_growth_factor;

        // Constructor
        QB(
            // Requires a RangeFinder scheme object.
            RandLAPACK::comps::rf::RangeFinder<T>& rf_obj,
            // Requires a stabilization algorithm object.
            RandLAPACK::comps::orth::Stabilization<T>& orth_obj,
            bool verb,
            bool orth
        ) : RF_Obj(rf_obj), Orth_Obj(orth_obj) {
            verbosity = verb;
            orth_check = orth;
            dim_growth_factor = 4;
        }

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

        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t& k,
            int64_t block_sz,
            T tol,
            std::vector<T>& Q,
            std::vector<T>& B
        ) {
            int termination = QB2(m, n, A, k, block_sz, tol, Q, B);

            if(this->verbosity) {
                switch(termination)
                {
                case 1:
                    printf("\nQB TERMINATED VIA: Input matrix of zero entries.\n");
                    break;
                case 2:
                    printf("\nQB TERMINATED VIA: Early termination due to unexpected error accumulation.\n");
                    break;
                case 3:
                    printf("\nQB TERMINATED VIA: Reached the expected rank without achieving the specified tolerance.\n");
                    break;
                case 4:
                    printf("\nQB TERMINATED VIA: Lost orthonormality of Q_i.\n");
                    break;
                case 5:
                    printf("\nQB TERMINATED VIA: Lost orthonormality of Q.\n");
                    break;
                case 6:
                    printf("\nQB TERMINATED VIA: RangeFinder failed.\n");
                    break;
                case 0:
                    printf("\nQB TERMINATED VIA: Normal termination; Expected tolerance reached.\n");
                    break;
                }
            }
            return termination;
        }
};

} // end namespace RandLAPACK::comps::qb
#endif
