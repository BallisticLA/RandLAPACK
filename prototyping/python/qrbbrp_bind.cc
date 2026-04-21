#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include <RandBLAS.hh>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <limits>
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

namespace qrcp_wide_algs {

template <typename T>
struct basic_greedy_logdet {
    std::vector<T> work_vec;
    int64_t block_size;
    int64_t num_blocks;          // initial value; set before reserve()
    int64_t rows_reserve = 0;
    int64_t num_blocks_reserve = 0;

    void reserve(int64_t rows, int64_t cols) {
        rows_reserve       = rows;
        num_blocks_reserve = num_blocks;
        // Layout: [block_copy] [diagsR] [v] [tau_block] [extra_arr]
        work_vec.assign(rows*block_size + block_size + num_blocks + block_size + cols, T(0));
    }

    void free() {}

    void operator()(int64_t rows, int64_t cols, T* A, int64_t lda, int64_t* J) {
        num_blocks = cols / block_size;

        T* block_copy = work_vec.data();
        T* diagsR     = block_copy + rows_reserve * block_size;
        T* v          = diagsR + block_size;
        T* tau_block  = v + num_blocks_reserve;
        T* extra_arr  = tau_block + block_size;

        std::fill(v, v + num_blocks, T(0));

        // Score each block: QR a copy of the block, sum log|diag(R)|
        constexpr T floor = std::numeric_limits<T>::min();  // smallest normal > 0
        for (int64_t i = 0; i < num_blocks; ++i) {
            // Copy block i into block_copy (compact column-major, lda=rows)
            for (int64_t j = 0; j < block_size; ++j)
                std::copy(A + lda*(i*block_size + j),
                          A + lda*(i*block_size + j) + rows,
                          block_copy + rows*j);
            lapack::geqrf(rows, block_size, block_copy, rows, tau_block);
            for (int64_t j = 0; j < block_size; ++j) {
                diagsR[j] = block_copy[j*(rows + 1)];  // (j,j) in col-major, lda=rows
                v[i] += std::log(std::max(std::abs(diagsR[j]), floor));
            }
        }

        int64_t idx_of_max = std::max_element(v, v + num_blocks) - v;
        swap_cols(rows, A, lda, block_size, idx_of_max);
        set_J(J, cols, block_size, idx_of_max);
        lapack::geqrf(rows, cols, A, lda, extra_arr);
    }
};

} // end namespace qrcp_wide


template <typename T>
py::tuple run_basic_impl(
    py::array A_in,
    int64_t   block_size,
    int64_t   num_blocks,
    T         d_factor,
    bool      timing,
    int       seed,
    bool      overwrite_a)
{
    py::buffer_info buf_in = A_in.request();
    int64_t m = buf_in.shape[0];
    int64_t n = buf_in.shape[1];
    if (n % block_size != 0)
        throw std::invalid_argument("n must be divisible by block_size.");

    py::array_t<T> A_work;
    if (overwrite_a) {
        if (buf_in.strides[0] != sizeof(T))
            throw std::invalid_argument(
                "A must be column-major (Fortran-contiguous). "
                "Pass np.asfortranarray(A) if needed.");
        A_work = py::reinterpret_borrow<py::array_t<T>>(A_in);
    } else {
        // forcecast handles both dtype conversion and layout reordering.
        // If pybind11 returned the same buffer (input was already T + F-contiguous),
        // call .copy() to ensure independence.
        auto A_f = py::array_t<T, py::array::f_style | py::array::forcecast>(A_in);
        if (A_f.request().ptr == buf_in.ptr)
            A_work = py::array_t<T, py::array::f_style>(A_f.attr("copy")());
        else
            A_work = std::move(A_f);
    }

    py::buffer_info buf = A_work.request();

    using subroutine_t = qrcp_wide_algs::basic_greedy_logdet<T>;
    subroutine_t subroutine{};
    subroutine.block_size = block_size;
    subroutine.num_blocks = n / block_size;

    RandLAPACK::QRBBRP<T, subroutine_t> alg(subroutine, timing, block_size, d_factor, num_blocks);

    std::vector<int64_t> J(n, 0);
    std::vector<T>       tau(n, T(0));

    RandBLAS::RNGState state(seed);
    alg.call(m, n, static_cast<T*>(buf.ptr), m, J.data(), tau.data(), state);

    for (auto& j : J) j -= 1;

    py::array_t<int64_t> J_out(n, J.data());
    py::array_t<T>       tau_out(n, tau.data());
    return py::make_tuple(J_out, tau_out, A_work);
}

py::tuple run_basic(
    py::array A,
    int64_t   block_size,
    int64_t   num_blocks,
    double    d_factor,
    bool      overwrite_a,
    bool      timing,
    int       seed)
{
    if (A.ndim() != 2)
        throw std::invalid_argument("A must be a 2-D array.");
    if (py::dtype::of<float>().is(A.dtype()))
        return run_basic_impl<float>(A, block_size, num_blocks, (float)d_factor, timing, seed, overwrite_a);
    if (py::dtype::of<double>().is(A.dtype()))
        return run_basic_impl<double>(A, block_size, num_blocks, (double)d_factor, timing, seed, overwrite_a);
    throw std::invalid_argument("A must have dtype float32 or float64.");
}


PYBIND11_MODULE(qrbbrp, m) {
    m.def("run_basic", run_basic,
          py::arg("A"),
          py::arg("block_size"),
          py::arg("num_blocks") = -1,
          py::arg("d_factor") = 2.0,
          py::arg("overwrite_a") = false,
          py::arg("timing") = false,
          py::arg("seed") = 99,
          "QRBBRP on a float32 or float64 array.\n"
          "Returns (J, tau, A_decomposed). J is 0-based.\n"
          "If overwrite_a=True, A must be column-major and A_decomposed shares its memory.\n"
          "If overwrite_a=False (default), a column-major copy is made and returned.");
    
}
