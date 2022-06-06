//#include <RandLAPACK/comps/rs.hh>
//#include <RandLAPACK/comps/util.hh>
//#include <iostream>

#include <lapack.hh>

#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <typeinfo>

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
		bool use_lu,
		uint64_t seed
){
	using namespace blas;
	using namespace lapack;

	int64_t p_done= 0;

	// Needs preallocated - will be used either way.
	std::vector<T> Omega_1(m * k, 0.0);

	//if (use_lu) {
		// Pivot vectors
		std::vector<int64_t> ipiv(k, 0);
	//}
	//else{
		// tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
		// tau needs to be a vector of all 2's by default
		std::vector<T> tau(k, 2.0);
	//}

	if (p % 2 == 0) {
		// Fill n by k omega
		RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega, seed);

		//printf("IS DOUBLE %d\n", typeid(*Omega) == typeid(double));

		//char name1[] = "Omega upon generation";
		//RandBLAS::util::print_colmaj(n, k, Omega, name1);
	}
	else{
		// Fill m by k omega_1
		RandBLAS::dense_op::gen_rmat_norm<T>(m, k, Omega_1.data(), seed);

		// multiply A' by Omega results in n by k omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, Omega_1.data(), m, 0.0, Omega, n);
		
		
		//char name2[] = "A' * Omega";
		//RandBLAS::util::print_colmaj(n, k, Omega, name2);
		
		++ p_done;
		if (p_done % passes_per_stab == 0) {
			if (use_lu) {
				// Stores L, U into Omega
				getrf(n, k, Omega, n, ipiv.data());
				// Addresses pivoting
				RandLAPACK::comps::util::pivot_swap<T>(n, k, Omega, ipiv.data());
				// Extracting L
				RandLAPACK::comps::util::get_L<T>(1, n, k, Omega);
			}
			else{
				//[Omega, ~] = tsqr(Omega)
				// use geqrf
				// Does LAPACK have function that just does orth
				geqrf(n, k, Omega, n, tau.data());
				// use ungqr to get the Q factor
				ungqr(n, k, k, Omega, n, tau.data());
			
				//RandLAPACK::comps::util::chol_QR<T>(n, k, Omega);
    			// Performing the alg twice for better orthogonality	
    			//RandLAPACK::comps::util::chol_QR<T>(n, k, Omega);
			}
		}
	}

	//char name[] = "Omega Before loop";
	//RandBLAS::util::print_colmaj(n, k, Omega, name);
	while (p - p_done > 0) 
	{
		// Omega = A * Omega
		gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Omega, n, 0.0, Omega_1.data(), m);
		++ p_done;
		if (p_done % passes_per_stab == 0) {
			if (use_lu) {
				getrf(m, k, Omega_1.data(), m, ipiv.data());
				RandLAPACK::comps::util::pivot_swap<T>(m, k, Omega_1.data(), ipiv.data());
				RandLAPACK::comps::util::get_L<T>(1, n, k, Omega);
			}
			else{
				geqrf(m, k, Omega_1.data(), m, tau.data());
				ungqr(m, k, k, Omega_1.data(), m, tau.data());
			
				//RandLAPACK::comps::util::chol_QR<T>(m, k, Omega_1.data());
    			//RandLAPACK::comps::util::chol_QR<T>(m, k, Omega_1.data());
			}
		}

		// Omega = A' * Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, Omega_1.data(), m, 0.0, Omega, n);
		++ p_done;
		if (p_done % passes_per_stab == 0) {
			if (use_lu) {
				getrf(n, k, Omega, n, ipiv.data());
				RandLAPACK::comps::util::pivot_swap<T>(n, k, Omega, ipiv.data());
				RandLAPACK::comps::util::get_L<T>(1, n, k, Omega);			
			}
			else{
				geqrf(n, k, Omega, n, tau.data());
				ungqr(n, k, k, Omega, n, tau.data());

				//RandLAPACK::comps::util::chol_QR<T>(n, k, Omega);	
    			//RandLAPACK::comps::util::chol_QR<T>(n, k, Omega);
			}
		}
	}
	//char name_final[] = "Omega after RS";
	//RandBLAS::util::print_colmaj(n, k, Omega, name_final);
}


template void rs1<float>(int64_t m, int64_t n, float* const A, int64_t k, int64_t p, int64_t passes_per_stab, float* Omega, bool use_lu, uint64_t seed);
template void rs1<double>(int64_t m, int64_t n, double* const A, int64_t k, int64_t p, int64_t passes_per_stab, double* Omega, bool use_lu, uint64_t seed);
} // end namespace RandLAPACK::comps::rs
