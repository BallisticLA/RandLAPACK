#include <cstdint>
#include <vector>

#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::rs {

template <typename T>
int RS<T>::rs1(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    int64_t k,
    std::vector<T>& Omega 
){
    using namespace blas;
    using namespace lapack;

    int64_t p = this->passes_over_data;
    int64_t q = this->passes_per_stab;
    int32_t seed = this->seed;
    int64_t p_done= 0;

    const T* A_dat = A.data();
    T* Omega_dat = Omega.data();
    T* Omega_1_dat = upsize<T>(m * k, this->Omega_1);
    auto state = RandBLAS::base::RNGState(seed, 0);
    if (p % 2 == 0) {
        // Fill n by k Omega
        RandBLAS::dense::DenseDist  D{.n_rows = n, .n_cols = k};
        RandBLAS::dense::fill_buff<T>(Omega_dat, D, state);
    } else {
        // Fill m by k Omega_1
        RandBLAS::dense::DenseDist D{.n_rows = m, .n_cols = k};
        RandBLAS::dense::fill_buff<T>(Omega_1_dat, D, state);

        // multiply A' by Omega results in n by k omega
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);

        ++ p_done;
        if (p_done % q == 0) {
            if(this->Stab_Obj.call(n, k, Omega))
                return 1; // Scheme failure
        }
    }

    while (p - p_done > 0) {
        // Omega = A * Omega
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Omega_1_dat, m);
        ++ p_done;
        
        if(this->cond_check)
            this->cond_nums.push_back(cond_num_check<T>(m, k, Omega_1, this->Omega_1_cpy, this->s, this->verbosity));

        if (p_done % q == 0) {
            if(this->Stab_Obj.call(m, k, Omega_1))
                return 1;
        }

        // Omega = A' * Omega
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);
        ++ p_done;
        
        if (this->cond_check)
            this->cond_nums.push_back(cond_num_check<T>(n, k, Omega, this->Omega_cpy, this->s, this->verbosity));
        
        if (p_done % q == 0) {
            if(this->Stab_Obj.call(n, k, Omega))
                return 1;
        }
    }
    // Increment seed upon termination
    this->seed += m * n;
    //successful termination
    return 0;
}

template int RS<float>::rs1(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Omega);
template int RS<double>::rs1(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Omega);
} // end namespace RandLAPACK::comps::rs
