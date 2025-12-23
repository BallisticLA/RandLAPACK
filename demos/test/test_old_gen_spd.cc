#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <iostream>
#include <algorithm>
#include <cmath>

// OLD BROKEN VERSION of gen_spd_mat
template <typename T, typename RNG>
void gen_spd_mat_OLD_BROKEN(
    int64_t n,
    T cond_num,
    T* A,
    RandBLAS::RNGState<RNG> &state
) {
    // OLD BROKEN IMPLEMENTATION
    ::std::vector<T> A_sym(n * n);
    RandBLAS::DenseDist D(n, n);
    state = RandBLAS::fill_dense(D, A_sym.data(), state);

    // Make positive definite: A = A_sym^T * A_sym + n*I
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n,
               (T)1.0, A_sym.data(), n, (T)0.0, A, n);

    // Add diagonal regularization: A += n*I  // BUG: n dominates!
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] += n;  // For n=1138, adds 1138 to every eigenvalue
    }

    // Copy upper triangle to lower triangle
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            A[i + j * n] = A[j + i * n];
        }
    }
}

int main() {
    int64_t n = 1138;
    double cond_num = 10.0;
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    std::vector<double> A(n * n);
    gen_spd_mat_OLD_BROKEN(n, cond_num, A.data(), state);

    // Compute eigenvalues via symmetric eigenvalue solver
    std::vector<double> eigenvalues(n);

    // Make a copy since syev destroys the input
    std::vector<double> A_copy = A;

    int info = lapack::syev(Job::NoVec, Uplo::Upper, n, A_copy.data(), n, eigenvalues.data());

    if (info != 0) {
        std::cerr << "syev failed with info=" << info << std::endl;
        return 1;
    }

    double min_eig = *std::min_element(eigenvalues.begin(), eigenvalues.end());
    double max_eig = *std::max_element(eigenvalues.begin(), eigenvalues.end());
    double actual_cond = max_eig / min_eig;

    std::cout << "\n=== OLD BROKEN gen_spd_mat ===" << std::endl;
    std::cout << "Requested: n=" << n << ", cond_num=" << cond_num << std::endl;
    std::cout << "Min eigenvalue: " << min_eig << std::endl;
    std::cout << "Max eigenvalue: " << max_eig << std::endl;
    std::cout << "Actual condition number: " << actual_cond << std::endl;
    std::cout << "Eigenvalues concentrated around: " << n << " (the bug!)" << std::endl;

    // Now test the NEW fixed version
    std::cout << "\n=== NEW FIXED gen_spd_mat ===" << std::endl;
    state = RandBLAS::RNGState<r123::Philox4x32>();  // Reset state
    std::vector<double> A_new(n * n);
    RandLAPACK::gen::gen_spd_mat(n, cond_num, A_new.data(), state);

    A_copy = A_new;
    info = lapack::syev(Job::NoVec, Uplo::Upper, n, A_copy.data(), n, eigenvalues.data());

    if (info != 0) {
        std::cerr << "syev failed with info=" << info << std::endl;
        return 1;
    }

    min_eig = *std::min_element(eigenvalues.begin(), eigenvalues.end());
    max_eig = *std::max_element(eigenvalues.begin(), eigenvalues.end());
    actual_cond = max_eig / min_eig;

    std::cout << "Requested: n=" << n << ", cond_num=" << cond_num << std::endl;
    std::cout << "Min eigenvalue: " << min_eig << std::endl;
    std::cout << "Max eigenvalue: " << max_eig << std::endl;
    std::cout << "Actual condition number: " << actual_cond << std::endl;

    return 0;
}
