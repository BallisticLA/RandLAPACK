#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <vector>
#include <numeric>
#include <utility>



void set_J(int64_t* J, int64_t len_J, int64_t b, int64_t chosen_block) {
    std::iota(J, J + len_J, 1);
    // ^ J = {1, 2, ..., len_J}
    std::swap_ranges(J, J + b, J + b*chosen_block);
    return;
}

template <typename T>
void swap_cols(int64_t m, int64_t n, T* A, int64_t lda, int64_t b, int64_t chosen_block) {
    for (int64_t i = 0; i < b; ++i) {
        T* ai_current = A + lda*i;
        T* ai_desired = A + lda*(chosen_block*b + i);
        std::swap_ranges(ai_current, ai_current + m, ai_desired);
    }
}


int main(int argc, char** argv) {

    using T = float;
    int64_t m = 3;
    int64_t n = 12;
    std::vector<T> M0(m*n);
    for (int64_t i = 0; i < n; ++i) {
        M0[i*m + (i%m)] = (T) (i+1);
    }
    RandBLAS::print_buff_to_stream(std::cout, blas::Layout::ColMajor, m, n, M0.data(), m, "M0", 2);
    std::vector<T> M2(M0);
    swap_cols(m, n, M2.data(), m, 2, 2);
    RandBLAS::print_buff_to_stream(std::cout, blas::Layout::ColMajor, m, n, M2.data(), m, "M2_b2", 2);
    M2 = M0;
    swap_cols(m, n, M2.data(), m, 3, 2);
    RandBLAS::print_buff_to_stream(std::cout, blas::Layout::ColMajor, m, n, M2.data(), m, "M2_b3", 2);
}