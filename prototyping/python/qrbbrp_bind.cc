#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include <RandBLAS.hh>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <numeric>
#include <vector>
#include <cmath>

namespace py = pybind11;

// Helpers and setup_work copied from qrbbrp_qp3_demo.cc

void set_J(int64_t* J, int64_t len_J, int64_t b, int64_t chosen_block) {
    std::iota(J, J + len_J, 1);
    if (chosen_block == 0)
        return;
    std::swap_ranges(J, J + b, J + b*chosen_block);
}

template <typename T>
void swap_cols(int64_t m, T* A, int64_t lda, int64_t b, int64_t chosen_block) {
    for (int64_t i = 0; i < b; ++i) {
        T* ai_current = A + lda*i;
        T* ai_desired = A + lda*(chosen_block*b + i);
        std::swap_ranges(ai_current, ai_current + m, ai_desired);
    }
}

template <typename T>
struct setup_work {
    std::vector<T> work_vec;
    int64_t num_blocks;
    int64_t block_size;

    void reserve(int64_t rows, int64_t cols) {
        work_vec.resize((rows + block_size)*cols + num_blocks + 2*block_size + cols, 0.0);
    }
    void free() { return; }
    void operator()(int64_t rows, int64_t cols, T* A, int64_t lda, int64_t* J) {
        num_blocks = cols / block_size;
        T* diagsR    = work_vec.data();
        T* v         = diagsR + block_size;
        T* tau       = v + num_blocks;
        T* extra_arr = tau + block_size;
        T* work_mat  = extra_arr + cols;

        int64_t rows_w = rows + block_size;
        for (int i = 0; i < cols*rows_w; i++)
            work_mat[i] = 0.0;

        int indx_work, indx_A;
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                indx_A    = j + i*lda;
                indx_work = indx_A + block_size*i;
                work_mat[indx_work] = A[indx_A];
            }
            indx_work += (i % block_size) + 1;
            work_mat[indx_work] = 1.0;
        }

        for (int i = 0; i < num_blocks; ++i) {
            T* W_block = &work_mat[i * block_size * rows_w];
            lapack::geqrf(rows_w, block_size, W_block, rows_w, tau);
            for (int j = 0; j < block_size; ++j)
                diagsR[j] = W_block[j * (rows_w + 1)];
            for (int j = 0; j < block_size; ++j)
                v[i] += std::log(std::abs(diagsR[j]));
        }

        int64_t idx_of_max = blas::iamax(num_blocks, v, 1);
        swap_cols(rows, A, rows, block_size, idx_of_max);
        set_J(J, cols, block_size, idx_of_max);
        lapack::geqrf(rows, cols, A, lda, extra_arr);
    }
};


std::pair<py::array_t<int64_t>, py::array_t<double>>
qrbbrp_binding(
    py::array_t<double, py::array::f_style | py::array::forcecast> A,
    int64_t block_size,
    double  d_factor,
    int     seed)
{
    py::buffer_info buf = A.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("A must be a 2-D array.");
    int64_t m = buf.shape[0];
    int64_t n = buf.shape[1];
    if (n % block_size != 0)
        throw std::invalid_argument("n must be divisible by block_size.");

    setup_work<double> qrcp_wide{};
    qrcp_wide.block_size = block_size;
    qrcp_wide.num_blocks = n / block_size;

    RandLAPACK::QRBBRP<double, setup_work<double>> alg(
        qrcp_wide, /*timing=*/false, block_size, d_factor);

    std::vector<int64_t> J(n, 0);
    std::vector<double>  tau(n, 0.0);

    RandBLAS::RNGState state(seed);
    alg.call(m, n, static_cast<double*>(buf.ptr), m, J.data(), tau.data(), state);

    // Convert from LAPACK 1-based to Python 0-based
    for (auto& j : J)
        j -= 1;

    py::array_t<int64_t> J_out(n, J.data());
    py::array_t<double>  tau_out(n, tau.data());
    return {J_out, tau_out};
}


PYBIND11_MODULE(qrbbrp, m) {
    m.def("qrbbrp", &qrbbrp_binding,
          py::arg("A"),
          py::arg("block_size"),
          py::arg("d_factor") = 2.0,
          py::arg("seed") = 99,
          "In-place QRBBRP on a column-major float64 array.\n"
          "Overwrites A with implicit Q and R. Returns (J, tau).\n"
          "J is 0-based. tau has the same meaning as in GEQP3.");
}
