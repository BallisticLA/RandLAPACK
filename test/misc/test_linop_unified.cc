// Unified templated tests for LinearOperator implementations
// Tests DenseLinOp, SparseLinOp, CompositeOperator, and any future operator types
//
// The test framework uses template parameters to test any LinearOperator implementation:
//   - OpTag: Tag type that specifies the operator type and its configuration
//   - T: Scalar type (double, float)
//
// Helper function templates are specialized for each operator type:
//   - make_operator<OpTag>: Creates simple operators (dense, sparse) from generated data
//   - make_composite_operator<LeftTag, RightTag>: Creates composite operators from two sub-operators
//   - densify_operator<OpTag>: Converts any operator to dense format for reference computation

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <type_traits>
#include <optional>
#include "../RandLAPACK/RandBLAS/test/comparison.hh"

using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::RNGState;
using namespace RandLAPACK::util; // for MatrixDimensions, calculate_dimensions, compute_gemm_reference

// ============================================================================
// Operator type tags for template specialization
// ============================================================================

// Tag for DenseLinOp
template <typename T>
struct DenseOpTag {
    using scalar_t = T;
    using op_type = RandLAPACK::linops::DenseLinOp<T>;
    static constexpr bool is_composite = false;
};

// Tag for SparseLinOp with CSC storage
template <typename T>
struct SparseCSCOpTag {
    using scalar_t = T;
    using sparse_mat_t = RandBLAS::sparse_data::csc::CSCMatrix<T>;
    using op_type = RandLAPACK::linops::SparseLinOp<sparse_mat_t>;
    static constexpr bool is_composite = false;
};

// Tag for CompositeOperator - parameterized by left and right operator tags
template <typename LeftTag, typename RightTag>
struct CompositeOpTag {
    using scalar_t = typename LeftTag::scalar_t;
    using left_tag = LeftTag;
    using right_tag = RightTag;
    static constexpr bool is_composite = true;
};

// Convenience aliases for composite operator tags
template <typename T>
using DenseDenseCompositeTag = CompositeOpTag<DenseOpTag<T>, DenseOpTag<T>>;

template <typename T>
using DenseSparseCompositeTag = CompositeOpTag<DenseOpTag<T>, SparseCSCOpTag<T>>;

template <typename T>
using SparseDenseCompositeTag = CompositeOpTag<SparseCSCOpTag<T>, DenseOpTag<T>>;

template <typename T>
using SparseSparseCompositeTag = CompositeOpTag<SparseCSCOpTag<T>, SparseCSCOpTag<T>>;

// ============================================================================
// Operator data holder - stores the underlying matrix data
// ============================================================================

// Base template (not used directly)
template <typename OpTag>
struct OperatorData;

// Specialization for DenseLinOp
template <typename T>
struct OperatorData<DenseOpTag<T>> {
    T* buffer = nullptr;
    int64_t rows, cols, ld;
    Layout layout;

    ~OperatorData() {
        if (buffer) delete[] buffer;
    }
};

// Specialization for SparseLinOp<CSCMatrix>
// Uses std::optional because CSCMatrix has deleted copy assignment (const members)
template <typename T>
struct OperatorData<SparseCSCOpTag<T>> {
    std::optional<RandBLAS::sparse_data::csc::CSCMatrix<T>> csc_mat;
    int64_t rows, cols;
};

// Specialization for CompositeOperator - holds data for both left and right operators
template <typename LeftTag, typename RightTag>
struct OperatorData<CompositeOpTag<LeftTag, RightTag>> {
    OperatorData<LeftTag> left_data;
    OperatorData<RightTag> right_data;
    // Store the actual operator objects (CompositeOperator stores references)
    std::optional<typename LeftTag::op_type> left_op;
    std::optional<typename RightTag::op_type> right_op;
    int64_t rows_left, cols_left;    // Dimensions of left operator
    int64_t rows_right, cols_right;  // Dimensions of right operator
    int64_t intermediate_dim;         // cols_left == rows_right
};

// ============================================================================
// make_operator: Creates operator and populates OperatorData
// ============================================================================

// Dense operator creation
template <typename T>
RandLAPACK::linops::DenseLinOp<T> make_operator(
    OperatorData<DenseOpTag<T>>& data,
    int64_t rows,
    int64_t cols,
    Layout layout,
    T density,  // unused for dense
    RNGState<r123::Philox4x32_R<10>>& state
) {
    data.rows = rows;
    data.cols = cols;
    data.layout = layout;
    data.ld = (layout == Layout::ColMajor) ? rows : cols;
    data.buffer = new T[rows * cols];

    RandLAPACK::gen::gen_random_dense(rows, cols, data.buffer, layout, state);

    return RandLAPACK::linops::DenseLinOp<T>(rows, cols, data.buffer, data.ld, layout);
}

// Sparse CSC operator creation
template <typename T>
RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>> make_operator(
    OperatorData<SparseCSCOpTag<T>>& data,
    int64_t rows,
    int64_t cols,
    Layout layout,  // layout doesn't affect CSC storage
    T density,
    RNGState<r123::Philox4x32_R<10>>& state
) {
    data.rows = rows;
    data.cols = cols;
    // Use emplace since gen_sparse_csc returns by value (already an rvalue)
    data.csc_mat.emplace(RandLAPACK::gen::gen_sparse_csc<T>(rows, cols, density, state));

    return RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>>(
        rows, cols, *data.csc_mat);
}

// ============================================================================
// densify_operator: Converts operator data to dense format
// ============================================================================

// Dense operator densification (just copy or convert layout)
template <typename T>
void densify_operator(
    const OperatorData<DenseOpTag<T>>& data,
    Layout target_layout,
    T* dense_out
) {
    if (data.layout == target_layout) {
        std::copy(data.buffer, data.buffer + data.rows * data.cols, dense_out);
    } else {
        // Convert layout using omatcopy
        if (data.layout == Layout::ColMajor) {
            // ColMajor to RowMajor: irs=1, ics=rows -> irs=cols, ics=1
            RandBLAS::util::omatcopy(data.rows, data.cols,
                                     data.buffer, 1, data.rows,
                                     dense_out, data.cols, 1);
        } else {
            // RowMajor to ColMajor: irs=cols, ics=1 -> irs=1, ics=rows
            RandBLAS::util::omatcopy(data.rows, data.cols,
                                     data.buffer, data.cols, 1,
                                     dense_out, 1, data.rows);
        }
    }
}

// Sparse CSC operator densification
template <typename T>
void densify_operator(
    const OperatorData<SparseCSCOpTag<T>>& data,
    Layout target_layout,
    T* dense_out
) {
    // Zero-initialize output
    std::fill(dense_out, dense_out + data.rows * data.cols, T(0));
    RandBLAS::sparse_data::csc::csc_to_dense(*data.csc_mat, target_layout, dense_out);
}

// ============================================================================
// Composite operator creation and densification
// ============================================================================

// Composite operator creation
// IMPORTANT: The operators are stored in data.left_op and data.right_op because
// CompositeOperator stores references, so we need the operators to outlive the CompositeOperator.
template <typename LeftTag, typename RightTag, typename T = typename LeftTag::scalar_t>
auto make_composite_operator(
    OperatorData<CompositeOpTag<LeftTag, RightTag>>& data,
    int64_t rows, int64_t cols,
    int64_t intermediate_dim,
    Layout layout,
    T density_left,
    T density_right,
    RNGState<r123::Philox4x32_R<10>>& state
) {
    // Store dimensions
    data.rows_left = rows;
    data.cols_left = intermediate_dim;
    data.rows_right = intermediate_dim;
    data.cols_right = cols;
    data.intermediate_dim = intermediate_dim;

    // Create left and right operators and store them in the data structure
    // This ensures they outlive the CompositeOperator which stores references
    data.left_op.emplace(make_operator<T>(data.left_data, rows, intermediate_dim, layout, density_left, state));
    data.right_op.emplace(make_operator<T>(data.right_data, intermediate_dim, cols, layout, density_right, state));

    // Return composite operator with references to the stored operators
    return RandLAPACK::linops::CompositeOperator(rows, cols, *data.left_op, *data.right_op);
}

// Composite operator densification - materializes (left * right) as dense matrix
template <typename LeftTag, typename RightTag, typename T = typename LeftTag::scalar_t>
void densify_operator(
    const OperatorData<CompositeOpTag<LeftTag, RightTag>>& data,
    Layout target_layout,
    T* dense_out
) {
    int64_t rows = data.rows_left;
    int64_t cols = data.cols_right;
    int64_t inter = data.intermediate_dim;

    // Densify left operator to ColMajor for intermediate computation
    T* left_dense = new T[rows * inter]();
    densify_operator(data.left_data, Layout::ColMajor, left_dense);

    // Densify right operator to ColMajor
    T* right_dense = new T[inter * cols]();
    densify_operator(data.right_data, Layout::ColMajor, right_dense);

    // Compute composite = left * right in ColMajor
    T* composite_colmajor = new T[rows * cols]();
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               rows, cols, inter,
               1.0, left_dense, rows,
               right_dense, inter,
               0.0, composite_colmajor, rows);

    // Convert to target layout if needed
    if (target_layout == Layout::ColMajor) {
        std::copy(composite_colmajor, composite_colmajor + rows * cols, dense_out);
    } else {
        // ColMajor to RowMajor
        RandBLAS::util::omatcopy(rows, cols,
                                 composite_colmajor, 1, rows,
                                 dense_out, cols, 1);
    }

    delete[] left_dense;
    delete[] right_dense;
    delete[] composite_colmajor;
}

// ============================================================================
// Unified test class
// ============================================================================

class TestLinearOperator : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Unified test function for any LinearOperator implementation (simple or composite).
    ///
    /// This test verifies that a LinearOperator correctly computes matrix products of the form
    /// C := alpha * op(A) * op(B) + beta * C (side=Left) or
    /// C := alpha * op(B) * op(A) + beta * C (side=Right), where A is wrapped as a LinearOperator.
    ///
    /// For composite operators, A = L * R where L and R are the left and right factor operators.
    ///
    /// Test structure:
    ///
    /// 1. MATRIX GENERATION
    ///    - For simple operators: Generate operator matrix A with dimensions from side/transpose.
    ///    - For composite operators: Generate L and R with intermediate_dim linking them.
    ///    - Generate random matrix B, either dense or sparse (CSC format) based on sparse_B flag.
    ///    - Initialize output matrix C with random values (to test beta != 0 case).
    ///
    /// 2. LINEAR OPERATOR CONSTRUCTION
    ///    - For simple operators: Create via make_operator<OpTag>.
    ///    - For composite operators: Create via make_composite_operator.
    ///
    /// 3. COMPUTATION VIA LINEAR OPERATOR
    ///    - Apply the LinearOperator to compute:
    ///        Side::Left:  C_op := alpha * op(A) * op(B) + beta * C_op
    ///        Side::Right: C_op := alpha * op(B) * op(A) + beta * C_op
    ///
    /// 4. REFERENCE COMPUTATION
    ///    - Convert A to dense format using densify_operator<OpTag>.
    ///    - If B is sparse, convert it to dense format as well.
    ///    - Compute C_reference using BLAS gemm directly on dense matrices.
    ///
    /// 5. VERIFICATION
    ///    - Compare C_op and C_reference entry-wise using componentwise error bounds.
    ///    - Error bound: |C_op - C_ref| <= |alpha| * k * 2 * eps * |A| * |B| + |beta| * eps * |C_old|
    ///    - This bound accounts for rounding errors in GEMM proportional to the inner dimension
    ///      and the magnitudes of the operands.
    ///
    /// Parameters:
    ///    test_linear_operator<OpTag>(side, sparse_B, layout, trans_A, trans_B, m, n, k, params)
    ///
    /// where params is a TestParams<T> struct with optional fields for densities and intermediate_dim.
    ///

    // Parameter struct for test configuration
    template <typename T>
    struct TestParams {
        T density_A = 0.3;           // For simple operators
        T density_left = 0.3;        // For composite operators (left factor)
        T density_right = 0.3;       // For composite operators (right factor)
        T density_B = 0.3;           // For input matrix B
        int64_t intermediate_dim = 0; // For composite operators (required if composite)
    };

    // Unified test function for any LinearOperator implementation
    template <typename OpTag>
    void test_linear_operator(
        Side side,
        bool sparse_B,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        TestParams<typename OpTag::scalar_t> params = {}
    ) {
        using T = typename OpTag::scalar_t;
        RNGState state(0);

        // Calculate dimensions using utility function
        auto dims = calculate_dimensions<T>(side, layout, trans_A, trans_B, m, n, k);

        // Create operator data holder and operator
        OperatorData<OpTag> op_data;

        if constexpr (OpTag::is_composite) {
            using LeftTag = typename OpTag::left_tag;
            using RightTag = typename OpTag::right_tag;
            auto A_op = make_composite_operator<LeftTag, RightTag>(
                op_data, dims.rows_A, dims.cols_A, params.intermediate_dim,
                layout, params.density_left, params.density_right, state);
            run_test_logic(op_data, A_op, side, sparse_B, layout, trans_A, trans_B,
                           m, n, k, dims, params.density_B, state);
        } else {
            auto A_op = make_operator<T>(op_data, dims.rows_A, dims.cols_A, layout, params.density_A, state);
            run_test_logic(op_data, A_op, side, sparse_B, layout, trans_A, trans_B,
                           m, n, k, dims, params.density_B, state);
        }
    }

private:
    // Common test logic for both simple and composite operators
    template <typename OpTag, typename OpType, typename T>
    void run_test_logic(
        OperatorData<OpTag>& op_data,
        OpType& A_op,
        Side side,
        bool sparse_B,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        const MatrixDimensions& dims,
        T density_B,
        RNGState<r123::Philox4x32_R<10>>& state
    ) {
        T alpha = 1.5;
        T beta = 0.5;

        // Create and initialize output buffers with random data for beta testing
        T* C_op = new T[m * n];
        RandLAPACK::gen::gen_random_dense(m, n, C_op, Layout::ColMajor, state);
        T* C_reference = new T[m * n];
        std::copy(C_op, C_op + m * n, C_reference);

        // Store |C_old| for error bound computation (beta term)
        T* E = new T[m * n];  // Error bound matrix
        for (int64_t i = 0; i < m * n; ++i) {
            E[i] = std::abs(C_op[i]);
        }

        // Allocate dense matrices for reference computation
        T* A_dense = new T[dims.rows_A * dims.cols_A]();
        T* B_dense = new T[dims.rows_B * dims.cols_B]();

        // Densify operator A
        densify_operator(op_data, layout, A_dense);

        if (sparse_B) {
            // Generate sparse matrix B
            auto B_csc = RandLAPACK::gen::gen_sparse_csc<T>(dims.rows_B, dims.cols_B, density_B, state);

            // Compute using LinearOperator with sparse B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_op, dims.ldc);

            // Densify B for reference
            RandBLAS::sparse_data::csc::csc_to_dense(B_csc, layout, B_dense);
        } else {
            // Generate dense matrix B
            RandLAPACK::gen::gen_random_dense(dims.rows_B, dims.cols_B, B_dense, layout, state);

            // Compute using LinearOperator with dense B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense, dims.ldb, beta, C_op, dims.ldc);
        }

        // Compute reference using BLAS GEMM
        compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, alpha,
                               A_dense, dims.lda, B_dense, dims.ldb,
                               beta, C_reference, dims.ldc);

        // Compute componentwise error bound matrix E
        // Error bound: |C_computed - C_exact| <= |alpha| * inner_dim * 2 * eps * |A| * |B| + |beta| * eps * |C_old|
        // We use GEMM on absolute values to compute this bound
        T eps = std::numeric_limits<T>::epsilon();
        T err_alpha = std::abs(alpha) * k * 2 * eps;
        T err_beta = std::abs(beta) * eps;

        // Compute |A| and |B|
        T* A_abs = new T[dims.rows_A * dims.cols_A];
        T* B_abs = new T[dims.rows_B * dims.cols_B];
        for (int64_t i = 0; i < dims.rows_A * dims.cols_A; ++i) {
            A_abs[i] = std::abs(A_dense[i]);
        }
        for (int64_t i = 0; i < dims.rows_B * dims.cols_B; ++i) {
            B_abs[i] = std::abs(B_dense[i]);
        }

        // E = err_alpha * |A| * |B| + err_beta * |C_old|
        // (E was initialized with |C_old|)
        compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, err_alpha,
                               A_abs, dims.lda, B_abs, dims.ldb,
                               err_beta, E, dims.ldc);

        // Compare results using componentwise bounds
        test::comparison::buffs_approx_equal(
            C_op, C_reference, E, m * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        // Clean up
        delete[] A_dense;
        delete[] B_dense;
        delete[] A_abs;
        delete[] B_abs;
        delete[] C_op;
        delete[] C_reference;
        delete[] E;
    }
};

// ============================================================================
// Static assertions for LinearOperator concept
// ============================================================================

using DenseOp = RandLAPACK::linops::DenseLinOp<double>;
using SparseOp = RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>>;

static_assert(RandLAPACK::linops::LinearOperator<DenseOp, double>,
              "DenseLinOp must satisfy LinearOperator concept");
static_assert(RandLAPACK::linops::LinearOperator<SparseOp, double>,
              "SparseLinOp must satisfy LinearOperator concept");
static_assert(RandLAPACK::linops::LinearOperator<RandLAPACK::linops::CompositeOperator<DenseOp, DenseOp>, double>,
              "CompositeOperator<DenseOp, DenseOp> must satisfy LinearOperator concept");
static_assert(RandLAPACK::linops::LinearOperator<RandLAPACK::linops::CompositeOperator<SparseOp, SparseOp>, double>,
              "CompositeOperator<SparseOp, SparseOp> must satisfy LinearOperator concept");
static_assert(RandLAPACK::linops::LinearOperator<RandLAPACK::linops::CompositeOperator<DenseOp, SparseOp>, double>,
              "CompositeOperator<DenseOp, SparseOp> must satisfy LinearOperator concept");
static_assert(RandLAPACK::linops::LinearOperator<RandLAPACK::linops::CompositeOperator<SparseOp, DenseOp>, double>,
              "CompositeOperator<SparseOp, DenseOp> must satisfy LinearOperator concept");

// ============================================================================
// DenseLinOp tests - Side::Left with dense B, ColMajor
// ============================================================================

TEST_F(TestLinearOperator, dense_left_dense_colmajor_notrans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_dense_colmajor_notrans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_dense_colmajor_trans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_dense_colmajor_trans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// DenseLinOp tests - Side::Left with dense B, RowMajor
// ============================================================================

TEST_F(TestLinearOperator, dense_left_dense_rowmajor_notrans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_dense_rowmajor_notrans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_dense_rowmajor_trans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_dense_rowmajor_trans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// DenseLinOp tests - Side::Right with dense B, ColMajor
// ============================================================================

TEST_F(TestLinearOperator, dense_right_dense_colmajor_notrans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_dense_colmajor_notrans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_dense_colmajor_trans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_dense_colmajor_trans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// DenseLinOp tests - Side::Right with dense B, RowMajor
// ============================================================================

TEST_F(TestLinearOperator, dense_right_dense_rowmajor_notrans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_dense_rowmajor_notrans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_dense_rowmajor_trans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_dense_rowmajor_trans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// DenseLinOp tests - Side::Left with sparse B, ColMajor
// ============================================================================

TEST_F(TestLinearOperator, dense_left_sparse_colmajor_notrans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_sparse_colmajor_notrans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_sparse_colmajor_trans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_sparse_colmajor_trans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// DenseLinOp tests - Side::Left with sparse B, RowMajor
// ============================================================================

TEST_F(TestLinearOperator, dense_left_sparse_rowmajor_notrans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_sparse_rowmajor_notrans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_sparse_rowmajor_trans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_left_sparse_rowmajor_trans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// DenseLinOp tests - Side::Right with sparse B, ColMajor
// ============================================================================

TEST_F(TestLinearOperator, dense_right_sparse_colmajor_notrans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_sparse_colmajor_notrans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_sparse_colmajor_trans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_sparse_colmajor_trans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// DenseLinOp tests - Side::Right with sparse B, RowMajor
// ============================================================================

TEST_F(TestLinearOperator, dense_right_sparse_rowmajor_notrans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_sparse_rowmajor_notrans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_sparse_rowmajor_trans_notrans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, dense_right_sparse_rowmajor_trans_trans) {
    test_linear_operator<DenseOpTag<double>>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// SparseLinOp tests - Side::Left with dense B, ColMajor
// ============================================================================

TEST_F(TestLinearOperator, sparse_left_dense_colmajor_notrans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_left_dense_colmajor_notrans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_left_dense_colmajor_trans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_left_dense_colmajor_trans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// SparseLinOp tests - Side::Left with dense B, RowMajor
// ============================================================================

TEST_F(TestLinearOperator, sparse_left_dense_rowmajor_notrans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_left_dense_rowmajor_notrans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_left_dense_rowmajor_trans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_left_dense_rowmajor_trans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// SparseLinOp tests - Side::Right with dense B, ColMajor
// ============================================================================

TEST_F(TestLinearOperator, sparse_right_dense_colmajor_notrans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_right_dense_colmajor_notrans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_right_dense_colmajor_trans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_right_dense_colmajor_trans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// SparseLinOp tests - Side::Right with dense B, RowMajor
// ============================================================================

TEST_F(TestLinearOperator, sparse_right_dense_rowmajor_notrans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_right_dense_rowmajor_notrans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_right_dense_rowmajor_trans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestLinearOperator, sparse_right_dense_rowmajor_trans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// SparseLinOp tests - Side::Left with sparse B, ColMajor
// ============================================================================

TEST_F(TestLinearOperator, sparse_left_sparse_colmajor_notrans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_left_sparse_colmajor_notrans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_left_sparse_colmajor_trans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_left_sparse_colmajor_trans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, {.density_B = 0.25});
}

// ============================================================================
// SparseLinOp tests - Side::Left with sparse B, RowMajor
// ============================================================================

TEST_F(TestLinearOperator, sparse_left_sparse_rowmajor_notrans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_left_sparse_rowmajor_notrans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_left_sparse_rowmajor_trans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_left_sparse_rowmajor_trans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, {.density_B = 0.25});
}

// ============================================================================
// SparseLinOp tests - Side::Right with sparse B, ColMajor
// ============================================================================

TEST_F(TestLinearOperator, sparse_right_sparse_colmajor_notrans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_right_sparse_colmajor_notrans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_right_sparse_colmajor_trans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_right_sparse_colmajor_trans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, {.density_B = 0.25});
}

// ============================================================================
// SparseLinOp tests - Side::Right with sparse B, RowMajor
// ============================================================================

TEST_F(TestLinearOperator, sparse_right_sparse_rowmajor_notrans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_right_sparse_rowmajor_notrans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_right_sparse_rowmajor_trans_notrans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, {.density_B = 0.25});
}

TEST_F(TestLinearOperator, sparse_right_sparse_rowmajor_trans_trans) {
    test_linear_operator<SparseCSCOpTag<double>>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, {.density_B = 0.25});
}

// ============================================================================
// CompositeOperator tests - Dense-Dense composition
// ============================================================================

TEST_F(TestLinearOperator, composite_dd_left_dense_colmajor) {
    test_linear_operator<DenseDenseCompositeTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_dd_left_dense_rowmajor) {
    test_linear_operator<DenseDenseCompositeTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_dd_right_dense_colmajor) {
    test_linear_operator<DenseDenseCompositeTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_dd_right_dense_rowmajor) {
    test_linear_operator<DenseDenseCompositeTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_dd_left_sparse_colmajor) {
    test_linear_operator<DenseDenseCompositeTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_dd_left_sparse_rowmajor) {
    test_linear_operator<DenseDenseCompositeTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_dd_right_sparse_colmajor) {
    test_linear_operator<DenseDenseCompositeTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_dd_right_sparse_rowmajor) {
    test_linear_operator<DenseDenseCompositeTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

// ============================================================================
// CompositeOperator tests - Sparse-Sparse composition
// ============================================================================

TEST_F(TestLinearOperator, composite_ss_left_dense_colmajor) {
    test_linear_operator<SparseSparseCompositeTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ss_left_dense_rowmajor) {
    test_linear_operator<SparseSparseCompositeTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ss_right_dense_colmajor) {
    test_linear_operator<SparseSparseCompositeTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ss_right_dense_rowmajor) {
    test_linear_operator<SparseSparseCompositeTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ss_left_sparse_colmajor) {
    test_linear_operator<SparseSparseCompositeTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ss_left_sparse_rowmajor) {
    test_linear_operator<SparseSparseCompositeTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ss_right_sparse_colmajor) {
    test_linear_operator<SparseSparseCompositeTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ss_right_sparse_rowmajor) {
    test_linear_operator<SparseSparseCompositeTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

// ============================================================================
// CompositeOperator tests - Dense-Sparse composition
// ============================================================================

TEST_F(TestLinearOperator, composite_ds_left_dense_colmajor) {
    test_linear_operator<DenseSparseCompositeTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ds_left_dense_rowmajor) {
    test_linear_operator<DenseSparseCompositeTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ds_right_dense_colmajor) {
    test_linear_operator<DenseSparseCompositeTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ds_right_dense_rowmajor) {
    test_linear_operator<DenseSparseCompositeTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ds_left_sparse_colmajor) {
    test_linear_operator<DenseSparseCompositeTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ds_left_sparse_rowmajor) {
    test_linear_operator<DenseSparseCompositeTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ds_right_sparse_colmajor) {
    test_linear_operator<DenseSparseCompositeTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_ds_right_sparse_rowmajor) {
    test_linear_operator<DenseSparseCompositeTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

// ============================================================================
// CompositeOperator tests - Sparse-Dense composition
// ============================================================================

TEST_F(TestLinearOperator, composite_sd_left_dense_colmajor) {
    test_linear_operator<SparseDenseCompositeTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_sd_left_dense_rowmajor) {
    test_linear_operator<SparseDenseCompositeTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_sd_right_dense_colmajor) {
    test_linear_operator<SparseDenseCompositeTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_sd_right_dense_rowmajor) {
    test_linear_operator<SparseDenseCompositeTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_sd_left_sparse_colmajor) {
    test_linear_operator<SparseDenseCompositeTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_sd_left_sparse_rowmajor) {
    test_linear_operator<SparseDenseCompositeTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_sd_right_sparse_colmajor) {
    test_linear_operator<SparseDenseCompositeTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}

TEST_F(TestLinearOperator, composite_sd_right_sparse_rowmajor) {
    test_linear_operator<SparseDenseCompositeTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, {.intermediate_dim = 6});
}
