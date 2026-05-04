#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include <RandBLAS.hh>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>


namespace py = pybind11;



template <typename T>
void block_linear_dopt(
    blas::Layout layout,
    int64_t m,
    int64_t n,
    T* A,
    int64_t b_sz,
    int64_t max_blocks,
    int64_t* block_pivs
) {
    static_assert(std::is_floating_point<T>::value,
                  "block_linear_dopt requires floating-point T.");

    if (A == nullptr) {
        throw std::invalid_argument("block_linear_dopt: A must be non-null.");
    }
    if (block_pivs == nullptr) {
        throw std::invalid_argument("block_linear_dopt: block_pivs must be non-null.");
    }
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("block_linear_dopt: m and n must be positive.");
    }
    if (b_sz <= 0) {
        throw std::invalid_argument("block_linear_dopt: b_sz must be positive.");
    }
    if ((n / b_sz) * b_sz != n) {
        throw std::invalid_argument("block_linear_dopt: require n divisible by b_sz.");
    }
    if (max_blocks < 0) {
        throw std::invalid_argument("block_linear_dopt: max_blocks must be nonnegative.");
    }

    const int64_t ldaA = (layout == blas::Layout::ColMajor) ? m : n;
    const int64_t n_candidates = n / b_sz;
    const int64_t num_blocks = std::min<int64_t>(n_candidates, max_blocks);

    if (num_blocks == 0) {
        return;
    }

    // Each W_i is a column-major (m + b_sz)-by-m matrix:
    //
    //     W_i = [ A_i^T ]
    //           [  I_m  ]
    //
    // where A_i is the i-th m-by-b_sz column block of A.
    //
    // On later iterations, after QR, the upper part of W_i contains the
    // candidate-specific triangular state, and the bottom b_sz rows are
    // overwritten by the transpose of the newly selected block.
    //
    const int64_t W_rows = m + b_sz;
    const int64_t W_cols = m;
    const int64_t ldW    = W_rows;
    const int64_t W_size = ldW * W_cols;

    // Monolithic storage for all candidate workspaces and tau arrays.
    std::vector<T> W_all(
        static_cast<size_t>(n_candidates) * static_cast<size_t>(W_size),
        T(0)
    );
    std::vector<T> tau_all(
        static_cast<size_t>(n_candidates) * static_cast<size_t>(m),
        T(0)
    );

    // selected[i] == 1 iff block i has already been chosen.
    std::vector<unsigned char> selected(static_cast<size_t>(n_candidates), 0);

    auto W_ptr = [&](int64_t blk) -> T* {
        return W_all.data() + static_cast<size_t>(blk) * static_cast<size_t>(W_size);
    };

    auto tau_ptr = [&](int64_t blk) -> T* {
        return tau_all.data() + static_cast<size_t>(blk) * static_cast<size_t>(m);
    };

    auto A_at = [&](int64_t row, int64_t col) -> T {
        if (layout == blas::Layout::ColMajor) {
            return A[col * ldaA + row];
        } else {
            return A[row * ldaA + col];
        }
    };

    // -------------------------------------------------------------------------
    // Initialize all W_i:
    //
    // Top b_sz rows    <- transpose of i-th column block of A
    // Bottom m rows    <- identity matrix
    //
    // Since W_i is column-major (m+b_sz)-by-m, for each column j of W_i:
    //   W_i(0:b_sz-1, j)      = A(j, blk*b_sz : blk*b_sz+b_sz-1)
    //   W_i(b_sz + j, j)      = 1
    // -------------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (int64_t blk = 0; blk < n_candidates; ++blk) {
        T* W = W_ptr(blk);
        std::fill(W, W + W_size, T(0));

        const int64_t col0 = blk * b_sz;
        for (int64_t j = 0; j < m; ++j) {
            for (int64_t r = 0; r < b_sz; ++r) {
                W[j * ldW + r] = A_at(j, col0 + r);
            }
            W[j * ldW + (b_sz + j)] = T(1);
        }
    }

    // -------------------------------------------------------------------------
    // Main greedy selection loop.
    // -------------------------------------------------------------------------
    for (int64_t iter = 0; iter < num_blocks; ++iter) {
        int64_t i_star = -1;

        #pragma omp parallel
        {
            // -------------------------------------------------------------
            // Step 1: QR-factor all remaining candidates in parallel.
            // -------------------------------------------------------------
            #pragma omp for schedule(dynamic)
            for (int64_t blk = 0; blk < n_candidates; ++blk) {
                if (!selected[static_cast<size_t>(blk)]) {
                    lapack::geqrf(W_rows, W_cols, W_ptr(blk), ldW, tau_ptr(blk));
                }
            }

            // -------------------------------------------------------------
            // Step 2: One thread chooses the candidate with largest
            //         sum_j log(abs(R_jj)).
            //
            // After GEQRF, R is in the upper-triangular part of the first
            // m rows/columns of W_i. Since W_i is column-major with ldW=W_rows,
            // diagonal entry (j,j) is W[j*ldW + j].
            // -------------------------------------------------------------
            #pragma omp single
            {
                T best_score = -std::numeric_limits<T>::infinity();

                for (int64_t blk = 0; blk < n_candidates; ++blk) {
                    if (selected[static_cast<size_t>(blk)]) {
                        continue;
                    }

                    T* W = W_ptr(blk);
                    T score = T(0);
                    bool singular = false;

                    for (int64_t j = 0; j < m; ++j) {
                        const T d = std::abs(W[j * ldW + j]);
                        if (!(d > T(0))) {
                            singular = true;
                            score = -std::numeric_limits<T>::infinity();
                            break;
                        }
                        score += std::log(d);
                    }

                    if (!singular && (i_star < 0 || score > best_score)) {
                        best_score = score;
                        i_star = blk;
                    }
                }

                if (i_star < 0) {
                    throw std::runtime_error("block_linear_dopt: failed to identify a valid candidate block.");
                }

                block_pivs[iter] = i_star;
                selected[static_cast<size_t>(i_star)] = 1;
            }

            // Ensure i_star and selected[] are visible before updates.
            #pragma omp barrier

            // -------------------------------------------------------------
            // Step 3: For all remaining candidates:
            //   - zero everything below the diagonal,
            //   - overwrite the bottom b_sz rows with the transpose of the
            //     selected block of A.
            //
            // "Below the diagonal" means rows i > j in column j, across the
            // full (m+b_sz)-by-m matrix.
            //
            // The "lower b_sz rows" are rows m, ..., m+b_sz-1.
            // -------------------------------------------------------------
            #pragma omp for schedule(static)
            for (int64_t blk = 0; blk < n_candidates; ++blk) {
                if (selected[static_cast<size_t>(blk)]) {
                    continue;
                }

                T* W = W_ptr(blk);

                // Zero everything strictly below the diagonal.
                RandBLAS::overwrite_triangle(blas::Layout::ColMajor, blas::Uplo::Lower, m, 1, W, ldW);
                lapack::laset(lapack::MatrixType::General, b_sz, m, (T)0.0, (T)0.0, W + m, ldW);

                // Overwrite the bottom b_sz rows with A_{i_star}^T.
                const int64_t sel_col0 = i_star * b_sz;
                for (int64_t j = 0; j < m; ++j) {
                    for (int64_t r = 0; r < b_sz; ++r) {
                        W[j * ldW + (m + r)] = A_at(j, sel_col0 + r);
                    }
                }
            }
        }
    }
}

template <typename T>
py::array_t<int64_t> run_block_linear_dopt_impl(
    py::array A_in,
    int64_t block_size,
    int64_t max_blocks)
{
    py::buffer_info buf_in = A_in.request();

    if (buf_in.ndim != 2) {
        throw std::invalid_argument("A must be a 2-D array.");
    }

    const int64_t m = buf_in.shape[0];
    const int64_t n = buf_in.shape[1];

    if (block_size <= 0) {
        throw std::invalid_argument("block_size must be positive.");
    }
    if (n % block_size != 0) {
        throw std::invalid_argument("n must be divisible by block_size.");
    }
    if (max_blocks < 0) {
        throw std::invalid_argument("max_blocks must be nonnegative.");
    }

    const int64_t n_candidates = n / block_size;
    const int64_t num_blocks = std::min<int64_t>(n_candidates, max_blocks);

    const bool is_f_contig = py::cast<bool>(A_in.attr("flags").attr("f_contiguous"));
    const bool is_c_contig = py::cast<bool>(A_in.attr("flags").attr("c_contiguous"));

    py::array_t<T> A_work;
    blas::Layout layout;

    if (is_f_contig) {
        A_work = py::array_t<T, py::array::f_style | py::array::forcecast>(A_in);
        layout = blas::Layout::ColMajor;
    } else if (is_c_contig) {
        A_work = py::array_t<T, py::array::c_style | py::array::forcecast>(A_in);
        layout = blas::Layout::RowMajor;
    } else {
        A_work = py::array_t<T, py::array::f_style | py::array::forcecast>(A_in);
        layout = blas::Layout::ColMajor;
    }

    py::buffer_info buf = A_work.request();
    T* A_ptr = static_cast<T*>(buf.ptr);

    std::vector<int64_t> block_pivs(num_blocks, 0);
    block_linear_dopt<T>(layout, m, n, A_ptr, block_size, max_blocks, block_pivs.data());

    return py::array_t<int64_t>(num_blocks, block_pivs.data());
}

py::array_t<int64_t> run_block_linear_dopt(
    py::array A,
    int64_t block_size,
    int64_t max_blocks)
{
    if (A.ndim() != 2) {
        throw std::invalid_argument("A must be a 2-D array.");
    }
    if (py::dtype::of<float>().is(A.dtype())) {
        return run_block_linear_dopt_impl<float>(A, block_size, max_blocks);
    }
    if (py::dtype::of<double>().is(A.dtype())) {
        return run_block_linear_dopt_impl<double>(A, block_size, max_blocks);
    }
    throw std::invalid_argument("A must have dtype float32 or float64.");
}


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

    m.def("run_block_linear_dopt", run_block_linear_dopt,
        py::arg("A"),
        py::arg("block_size"),
        py::arg("max_blocks"),
        "Greedy block selection via block_linear_dopt on a float32 or float64 array.\n"
        "Returns block_pivs as 0-based block indices.\n"
        "A may be C-contiguous or F-contiguous; if neither, a contiguous copy is made.");
}
