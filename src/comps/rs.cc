#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

//#define USE_QR
//#define USE_LU
#define USE_CHOL

namespace RandLAPACK::comps::rs {

template <typename T>
void RS1<T>::call(
	int64_t m,
	int64_t n,
	const std::vector<T>& A,
	int64_t k,
	std::vector<T>& Omega // n by k
){
	using namespace blas;
	using namespace lapack;
	
	int64_t p = RS1::passes_over_data;
	int64_t q = RS1::passes_per_stab;
	int32_t seed = RS1::seed;
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
			RS1::stabilizer(n, k, Omega);
		}
	}
	
	while (p - p_done > 0) 
	{
		// Omega = A * Omega
		gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Omega_1_dat, m);
		++ p_done;
		if (p_done % q == 0) 
		{
			RS1::stabilizer(m, k, Omega_1);
		}

		// Omega = A' * Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);
		++ p_done;
		if (p_done % q == 0) 
		{
			RS1::stabilizer(n, k, Omega);
		}
	}
}

template void RS1<float>::call(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Omega);
template void RS1<double>::call(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Omega);
} // end namespace RandLAPACK::comps::rs
