#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>

using namespace RandLAPACK;

template <typename T, typename RNG>
static void 
chol_check(int64_t m, int64_t k, RandBLAS::RNGState<RNG> state) {

    std::vector<T> A(m * m, 0.0);
    std::vector<T> A_leading_submat_symm(k * k, 0.0);
    std::vector<T> R_submat(k * k, 0.0);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, m, A, m, state, std::make_tuple(0, std::pow(10, 8), false));

    T* A_dat = A.data();
    T* A_leading_submat_symm_dat = A_leading_submat_symm.data();
    T* R_submat_dat = R_submat.data();

    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, k, 1.0, A.data(), k, 0.0, A_leading_submat_symm.data(), k);

    A[0] = A_leading_submat_symm[0];
    for(int i = 1; i < k; ++i) {
        // Fill the upper-triangular part
        blas::copy(i + 1, &A_leading_submat_symm_dat[(i * k)], 1, &A_dat[i * m], 1);
        // Fill the lower-triangular part
        blas::copy(k - i, &A_leading_submat_symm_dat[(i - 1) + (i * k)], k, &A_dat[i + ((i-1) * m)], 1);
        // Also fill the full symmetric matrix
        blas::copy(k - i, &A_leading_submat_symm_dat[(i - 1) + (i * k)], k, &A_leading_submat_symm_dat[i + ((i-1) * k)], 1);
    }

    // Now, the k by k portion of A is summetric and the rest is random

    if(lapack::potrf(Uplo::Upper, m, A_dat, m)) 
        printf("Cholesky failed as expected.\n");

    // Copy the k by k portion of the R factor
    lapack::lacpy(MatrixType::Upper, k, k, A_dat, m, R_submat_dat, k);

    // Doing this through GEMM to perform subtraction immediately
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, k, 1.0, R_submat_dat, k, R_submat_dat, k, -1.0, A_leading_submat_symm_dat, k);

    T norm = lapack::lange(Norm::Fro, k, k, A_leading_submat_symm_dat, k);
    printf("||R[:k, :k]'*R[:k, :k] - A[:k, :k]||_F:  %e\n", norm);
}

int main() {
    for(int i = 0; i < 10; ++i) {
        auto state = RandBLAS::RNGState(i);
        chol_check<double, r123::Philox4x32>(1000, 500, state);
    }
    return 0;
}