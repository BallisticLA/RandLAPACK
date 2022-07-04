#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

/*
    References
    ----------
    This implementation is inspired by [ZM:2020, Algorithm 3.3]. The most
    significant difference is that this function stops one step "early",
    so that it returns a matrix S for use in sketching Y = A @ S, rather than
    returning an orthonormal basis for a sketched matrix Y. Here are the
    differences between this implementation and [ZM:2020, Algorithm 3.3],
    assuming the latter algorithm was modified to stop "one step early" like
    this algorithm:
        (1) We make no assumptions on the distribution of the initial
            (oblivious) sketching matrix. [ZM:2020, Algorithm 3.3] uses
            a Gaussian distribution.
        (2) We allow any number of passes over A, including zero passes.
            [ZM2020: Algorithm 3.3] requires at least one pass over A.
        (3) We let the user provide the stabilization method. [ZM:2020,
            Algorithm 3.3] uses LU for stabilization.
        (4) We let the user decide how many applications of A or A.T
            can be made between calls to the stabilizer.
*/

namespace RandLAPACK::comps::rs {

template <typename T>
void RS<T>::rs1(
	int64_t m,
	int64_t n,
	const std::vector<T>& A,
	int64_t k,
	std::vector<T>& Omega 
){
	using namespace blas;
	using namespace lapack;
	
	int64_t p = RS::passes_over_data;
	int64_t q = RS::passes_per_stab;
	int32_t seed = RS::seed;
	int64_t p_done= 0;

	// Preallocations
	std::vector<T> Omega_1(m * k, 0.0);
	// Setting stabilization parameters
	RS::Stab_Obj.tau.resize(k);
	RS::Stab_Obj.ipiv.resize(k);

	const T* A_dat = A.data();
	T* Omega_dat = Omega.data();
	T* Omega_1_dat = Omega_1.data();

	if (p % 2 == 0) 
	{
		// Fill n by k Omega
		RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega_dat, seed);
		return;
	}
	else
	{
		// Fill m by k Omega_1
		RandBLAS::dense_op::gen_rmat_norm<T>(m, k, Omega_1_dat, seed);

		// multiply A' by Omega results in n by k omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);

		++ p_done;
		if (p_done % q == 0) 
		{
			// Use the specified stabilization routine
			switch(RS::Stab_Obj.decision_stab)
			{
				// CholQR stabilization
				case 0:
					// Try CholQR stabilization
					RS::Stab_Obj.call(n, k, Omega);
					if(RS::Stab_Obj.chol_fail)
					{
						// If CholQR fails, fall back on PLU
						RS::Stab_Obj.decision_stab = 1;
						RS::Stab_Obj.call(n, k, Omega);
					}
					break;
				// PLU stabilization
				case 1:
					RS::Stab_Obj.call(n, k, Omega);
					break;
			}
		}
	}
	
	while (p - p_done > 0) 
	{
		// Omega = A * Omega
		gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Omega_1_dat, m);
		++ p_done;
		if (p_done % q == 0) 
		{
			switch(RS::Stab_Obj.decision_stab)
			{
				case 0:
					RS::Stab_Obj.call(m, k, Omega_1);
					if(RS::Stab_Obj.chol_fail)
					{
						RS::Stab_Obj.decision_stab = 1;
						RS::Stab_Obj.call(m, k, Omega_1);
					}
					break;
				case 1:
					RS::Stab_Obj.call(m, k, Omega_1);
					break;
			}
		}

		// Omega = A' * Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);
		++ p_done;
		if (p_done % q == 0) 
		{
			switch(RS::Stab_Obj.decision_stab)
			{
				case 0:
					RS::Stab_Obj.call(n, k, Omega);
					if(RS::Stab_Obj.chol_fail)
					{
						RS::Stab_Obj.decision_stab = 1;
						RS::Stab_Obj.call(n, k, Omega);
					}
					break;
				case 1:
					RS::Stab_Obj.call(n, k, Omega);
					break;
			}
		}
	}
}

template void RS<float>::rs1(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Omega);
template void RS<double>::rs1(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Omega);
} // end namespace RandLAPACK::comps::rs
