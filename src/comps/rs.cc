#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

//#define USE_QR
//#define USE_LU
#define USE_CHOL

namespace RandLAPACK::comps::rs {

template <typename T>
void rs1(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Omega, // n by k
		uint32_t seed
){
	using namespace blas;
	using namespace lapack;

	int64_t p_done= 0;

	std::vector<T> Omega_1(m * k, 0.0);

	const T* A_dat = A.data();
	T* Omega_dat = Omega.data();
	T* Omega_1_dat = Omega_1.data();

#ifdef USE_LU
	// Pivot vectors
	std::vector<int64_t> ipiv(k, 0);
	int64_t* ipiv_dat = ipiv.data();
#endif
#ifdef USE_QR
	// tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
	// tau needs to be a vector of all 2's by default
	std::vector<T> tau(k, 2.0);
	T* tau_dat = tau.data();
#endif

	if (p % 2 == 0) {
		// Fill n by k omega
		RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega_dat, seed);
	}
	else{
		// Fill m by k omega_1
		RandBLAS::dense_op::gen_rmat_norm<T>(m, k, Omega_1_dat, seed);

		// multiply A' by Omega results in n by k omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);

		++ p_done;
		if (p_done % passes_per_stab == 0) 
		{
#ifdef USE_LU
			// Stores L, U into Omega
			getrf(n, k, Omega_dat, n, ipiv_dat);
			// Addresses pivoting
			RandLAPACK::comps::util::row_swap<T>(n, k, Omega, ipiv);
			// Extracting L
			RandLAPACK::comps::util::get_L<T>(n, k, Omega);
#endif
#ifdef USE_QR
			//[Omega, ~] = tsqr(Omega)
			// use geqrf
			geqrf(n, k, Omega_dat, n, tau_dat);
			// use ungqr to get the Q factor
			ungqr(n, k, k, Omega_dat, n, tau_dat);
#endif
#ifdef USE_CHOL
			// No need to refine orthogonality here
			RandLAPACK::comps::orth::chol_QR<T>(n, k, Omega);
#endif
		}
	}

	while (p - p_done > 0) 
	{
		// Omega = A * Omega
		gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Omega_1_dat, m);
		++ p_done;
		if (p_done % passes_per_stab == 0) 
		{
#ifdef USE_LU
			getrf(m, k, Omega_1_dat, m, ipiv_dat);
			RandLAPACK::comps::util::row_swap<T>(m, k, Omega_1, ipiv);
			RandLAPACK::comps::util::get_L<T>(m, k, Omega_1);
#endif
#ifdef USE_QR
			geqrf(m, k, Omega_1_dat, m, tau_dat);
			ungqr(m, k, k, Omega_1_dat, m, tau_dat);
#endif
#ifdef USE_CHOL
			RandLAPACK::comps::orth::chol_QR<T>(m, k, Omega_1);
#endif
		}

		// Omega = A' * Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);
		++ p_done;
		if (p_done % passes_per_stab == 0) 
		{
#ifdef USE_LU
			getrf(n, k, Omega_dat, n, ipiv_dat);
			RandLAPACK::comps::util::row_swap<T>(n, k, Omega, ipiv);
			RandLAPACK::comps::util::get_L<T>(n, k, Omega);			
#endif
#ifdef USE_QR
			geqrf(n, k, Omega_dat, n, tau_dat);
			ungqr(n, k, k, Omega_dat, n, tau_dat);
#endif
#ifdef USE_CHOL
			RandLAPACK::comps::orth::chol_QR<T>(n, k, Omega);	
#endif
		}
	}
}


template void rs1<float>(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<float>& Omega, uint32_t seed);
template void rs1<double>(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<double>& Omega, uint32_t seed);

template <typename T>
void RowSketcher<T>::RS1(
	int64_t m,
	int64_t n,
	const std::vector<T>& A,
	int64_t k,
	std::vector<T>& Omega // n by k
){
	using namespace blas;
	using namespace lapack;
	
	int64_t p = RowSketcher::passes_over_data;
	int64_t q = RowSketcher::passes_per_stab;
	int32_t seed = RowSketcher::seed;
	int64_t p_done= 0;

	std::vector<T> Omega_1(m * k, 0.0);

	const T* A_dat = A.data();
	T* Omega_dat = Omega.data();
	T* Omega_1_dat = Omega_1.data();

	if (p % 2 == 0) {
		// Fill n by k omega
		RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega_dat, seed);
	}
	
	else{
		// Fill m by k omega_1
		RandBLAS::dense_op::gen_rmat_norm<T>(m, k, Omega_1_dat, seed);

		// multiply A' by Omega results in n by k omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);

		++ p_done;
		if (p_done % q == 0) 
		{
			RowSketcher::stabilizer(n, k, Omega);
		}
	}
	
	while (p - p_done > 0) 
	{
		// Omega = A * Omega
		gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Omega_1_dat, m);
		++ p_done;
		if (p_done % q == 0) 
		{
			RowSketcher::stabilizer(m, k, Omega_1);
		}

		// Omega = A' * Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);
		++ p_done;
		if (p_done % q == 0) 
		{
			RowSketcher::stabilizer(n, k, Omega);
		}
	}
}

template void RowSketcher<float>::RS1(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Omega);
template void RowSketcher<double>::RS1(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Omega);
} // end namespace RandLAPACK::comps::rs
