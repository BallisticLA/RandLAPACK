#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

namespace RandLAPACK::comps::rs {


// Add check for zero matrix

//  Version where workspace gets allocated inside the function.
template <typename T>
void rs1(
        int64_t m,
        int64_t n,
        T* const A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        T* Omega, // n by k
		uint64_t seed
){
	using namespace blas;
	using namespace lapack;

	int64_t p_done= 0;

	// Needs preallocated - will be used either way.
	std::vector<T> Omega_1(m * k, 0.0);

	if (p % 2 == 0) {
		// Fill n by k omega
		RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega, seed);
	}
	else{
		// Fill m by k omega_1
		RandBLAS::dense_op::gen_rmat_norm<T>(m, k, Omega_1.data(), seed);

		// multiply A' by Omega results in n by k omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, Omega_1.data(), m, 0.0, Omega, n);
		
		++ p_done;
		if (p_done % passes_per_stab == 0) 
		{
#ifdef USE_LU
			// Pivot vectors
			std::vector<int64_t> ipiv(k, 0);
			// Stores L, U into Omega
			getrf(n, k, Omega, n, ipiv.data());
			// Addresses pivoting
			RandLAPACK::comps::util::pivot_swap<T>(n, k, Omega, ipiv.data());
			// Extracting L
			RandLAPACK::comps::util::get_L<T>(1, n, k, Omega);
	
#elseif USE_QR
			//[Omega, ~] = tsqr(Omega)
			// use geqrf
			// tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
			// tau needs to be a vector of all 2's by default
			std::vector<T> tau(k, 2.0);
			geqrf(n, k, Omega, n, tau.data());
			// use ungqr to get the Q factor
			ungqr(n, k, k, Omega, n, tau.data());
#else
			RandLAPACK::comps::util::chol_QR<T>(n, k, Omega);
			// Performing the alg twice for better orthogonality	
			RandLAPACK::comps::util::chol_QR<T>(n, k, Omega);
#endif
		}
	}

	while (p - p_done > 0) 
	{
		// Omega = A * Omega
		gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Omega, n, 0.0, Omega_1.data(), m);
		++ p_done;
		if (p_done % passes_per_stab == 0) 
		{
#ifdef USE_LU
			getrf(m, k, Omega_1.data(), m, ipiv.data());
			RandLAPACK::comps::util::pivot_swap<T>(m, k, Omega_1.data(), ipiv.data());
			RandLAPACK::comps::util::get_L<T>(1, n, k, Omega);

#elseif USE_QR
			geqrf(m, k, Omega_1.data(), m, tau.data());
			ungqr(m, k, k, Omega_1.data(), m, tau.data());
#else
			RandLAPACK::comps::util::chol_QR<T>(m, k, Omega_1.data());
			RandLAPACK::comps::util::chol_QR<T>(m, k, Omega_1.data());
#endif
		}

		// Omega = A' * Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, Omega_1.data(), m, 0.0, Omega, n);
		++ p_done;
		if (p_done % passes_per_stab == 0) 
		{
#ifdef USE_LU
			getrf(n, k, Omega, n, ipiv.data());
			RandLAPACK::comps::util::pivot_swap<T>(n, k, Omega, ipiv.data());
			RandLAPACK::comps::util::get_L<T>(1, n, k, Omega);			

#elseif USE_QR
			geqrf(n, k, Omega, n, tau.data());
			ungqr(n, k, k, Omega, n, tau.data());
#else
			RandLAPACK::comps::util::chol_QR<T>(n, k, Omega);	
			RandLAPACK::comps::util::chol_QR<T>(n, k, Omega);
#endif
		}
	}
}


template void rs1<float>(int64_t m, int64_t n, float* const A, int64_t k, int64_t p, int64_t passes_per_stab, float* Omega, uint64_t seed);
template void rs1<double>(int64_t m, int64_t n, double* const A, int64_t k, int64_t p, int64_t passes_per_stab, double* Omega, uint64_t seed);
} // end namespace RandLAPACK::comps::rs
