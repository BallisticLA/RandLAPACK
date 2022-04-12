#include <RandLAPACK/comps/rs.hh>
#include <iostream>


namespace RandLAPACK::comps::rs {


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
		bool use_lu,
		uint64_t seed
){
	using namespace blas;
	//using namespace lapack;

	int64_t p_done= 0;

	// Needs preallocated - will be used either way.
	std::vector<T> Omega_1(m * k, 0.0);
	//if (use_lu) {
		// Pivot vectors
		std::vector<T> ipiv_1(m, 0.0);
		std::vector<T> ipiv(n, 0.0);
	//}
	//else{
		// tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
		// tau needs to be a vector of all 2's by default
		std::vector<T> tau(k, 2.0);
	//}

	if (p % 2 == 0) {
		// Fill n by k omega
		gen_rmat_normal(n, k, Omega, seed);
	}
	else{
		// Fill m by k omega_1
		gen_rmat_normal(m, k, Omega, seed);
		// multiply A' by Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, m, k, 1.0, A, m, Omega_1.data(), m, 0.0, Omega, n);
		++ p_done;
		if (p_done % passes_per_stab == 0) {
			if (use_lu) {
				// Stores L, U into Omega
				getrf(n, k, Omega, n, ipiv.data());
				// Extracts L - pivoting is not addressed, but should it matter?
				get_L(1, n, k, Omega);
			}
			else{
				//[Omega, ~] = tsqr(Omega)
				// use geqrf
				// Does LAPACK have function that just does orth
				geqrf(n, k, Omega, n, tau.data());
				// use ungqr to get the Q factor
				ungqr(n, k, k, Omega, n, tau.data());
			}
		}
	}

	while (p - p_done > 0) {
		// Omega = A * Omega
		gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A, m, Omega, n, 0.0, Omega_1.data(), m);
		++ p_done;
		if (p_done % passes_per_stab == 0) {
			if (use_lu) {
				getrf(m, k, Omega_1.data(), m, ipiv_1.data());
				get_L(1, n, k, Omega);
			}
			else{
				geqrf(m, k, Omega_1.data(), m, tau.data());
				ungqr(m, k, k, Omega_1.data(), m, tau.data());
			}
		}

		// Omega = A' * Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, m, k, 1.0, A, m, Omega_1.data(), m, 0.0, Omega, n);
		++ p_done;
		if (p_done % passes_per_stab == 0) {
			if (use_lu) {
				getrf(n, k, Omega, n, ipiv.data());
				get_L(1, n, k, Omega);			
			}
			else{
				geqrf(n, k, Omega, n, tau.data());
				ungqr(n, k, k, Omega, n, tau.data());
			}
		}
	}
}
} // end namespace RandLAPACK::comps::rs
