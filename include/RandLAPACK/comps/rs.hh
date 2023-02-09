#ifndef randlapack_comps_rs_h
#define randlapack_comps_rs_h

#include "orth.hh"

#include <vector>
#include <cstdint>
#include <cstdio>

namespace RandLAPACK::comps::rs {

template <typename T>
class RowSketcher
{
    public:
        virtual int call(
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
        int32_t seed;
        int64_t passes_over_data;
        int64_t passes_per_stab;
        bool verbosity;
        bool cond_check;
        std::vector<T> Omega_1;
        std::vector<T> cond_nums;

        std::vector<T> Omega_cpy;
        std::vector<T> Omega_1_cpy;
        std::vector<T> s;

        RS(
            // Requires a stabilization algorithm object.
            RandLAPACK::comps::orth::Stabilization<T>& stab_obj,
            int32_t s, 
            int64_t p, 
            int64_t q,
            bool verb,
            bool cond
        ) : Stab_Obj(stab_obj) {
            verbosity = verb;
            cond_check = cond;
            seed = s;
            passes_over_data = p;
            passes_per_stab = q;
        }

        int rs1(
            int64_t m,
            int64_t n,
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Omega 
        );

        virtual int call(
            int64_t m,
            int64_t n,
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Omega 
        ){
            // Default
            int termination = rs1(m, n, A, k, Omega);

            if(this->verbosity) {
                switch(termination) {
                case 0:
                        printf("\nRS TERMINATED VIA: Normal termination.\n");
                        break;
                case 1:
                        printf("\nRS TERMINATED VIA: Stabilization failed.\n");
                        break;
                }
            }
            return termination;
        }
};

} // end namespace RandLAPACK::comps::rs
#endif
