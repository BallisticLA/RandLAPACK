#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

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
	
	int64_t p = this->passes_over_data;
	int64_t q = this->passes_per_stab;
	int32_t seed = this->seed;
	int64_t p_done= 0;

	const T* A_dat = A.data();
	T* Omega_dat = Omega.data();
	T* Omega_1_dat = RandLAPACK::comps::util::resize(m * k, this->Omega_1);

	if (p % 2 == 0) 
	{
		// Fill n by k Omega
		RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega_dat, seed);
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
			switch(this->Stab_Obj.decision_stab)
			{
				// CholQR stabilization
				case 0:
					// Try CholQR stabilization
					this->Stab_Obj.call(n, k, Omega);
					if(this->Stab_Obj.chol_fail)
					{
						// If CholQR fails, fall back on PLU
						this->Stab_Obj.decision_stab = 1;
						this->Stab_Obj.call(n, k, Omega);
					}
					break;
				// PLU stabilization
				case 1:
					this->Stab_Obj.call(n, k, Omega);
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
			switch(this->Stab_Obj.decision_stab)
			{
				case 0:
					this->Stab_Obj.call(m, k, Omega_1);
					if(this->Stab_Obj.chol_fail)
					{
						this->Stab_Obj.decision_stab = 1;
						this->Stab_Obj.call(m, k, Omega_1);
					}
					break;
				case 1:
					this->Stab_Obj.call(m, k, Omega_1);
					break;
			}
		}

		// Condition number check
		if (this->cond_check)
		{
			// Copy to avoid any changes
			T* Omega_1_cpy_dat = RandLAPACK::comps::util::resize(m * k, this->Omega_1_cpy);
			T* s_dat = RandLAPACK::comps::util::resize(k, this->s);

			lacpy(MatrixType::General, m, k, Omega_1_dat, m, Omega_1_cpy_dat, m);
			gesdd(Job::NoVec, m, k, Omega_1_cpy_dat, m, s_dat, NULL, m, NULL, k);
			T cond_num = s_dat[0] / s_dat[k - 1];

			if (this->verbosity)
				printf("CONDITION NUMBER OF OMEGA IS: %f\n", cond_num);
			
			this->cond_nums.push_back(cond_num);
		}

		// Omega = A' * Omega
		gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_dat, m, Omega_1_dat, m, 0.0, Omega_dat, n);
		++ p_done;
		if (p_done % q == 0) 
		{
			switch(this->Stab_Obj.decision_stab)
			{
				case 0:
					this->Stab_Obj.call(n, k, Omega);
					if(this->Stab_Obj.chol_fail)
					{
						this->Stab_Obj.decision_stab = 1;
						this->Stab_Obj.call(n, k, Omega);
					}
					break;
				case 1:
					this->Stab_Obj.call(n, k, Omega);
					break;
			}
		}

		// Condition number check
		if (this->cond_check)
		{
			// Copy to avoid any changes
			T* Omega_cpy_dat = RandLAPACK::comps::util::resize(n * k, this->Omega_cpy);
			T* s_dat = RandLAPACK::comps::util::resize(k, this->s);

			lacpy(MatrixType::General, n, k, Omega_dat, n, Omega_cpy_dat, n);
			gesdd(Job::NoVec, n, k, Omega_cpy_dat, n, s_dat, NULL, n, NULL, k);
			T cond_num = s_dat[0] / s_dat[k - 1];

			if (this->verbosity)
				printf("CONDITION NUMBER OF OMEGA IS: %f\n", cond_num);
			
			this->cond_nums.push_back(cond_num);
		}
	}
}

template void RS<float>::rs1(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Omega);
template void RS<double>::rs1(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Omega);
} // end namespace RandLAPACK::comps::rs
