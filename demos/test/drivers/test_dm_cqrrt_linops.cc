// Test suite for CQRRT_linops driver with composite linear operators.
// Tests the algorithm on C = A^{-1}B where A^{-1} is a CholSolverLinOp (sparse Cholesky-based inverse)
// and B is a SparseLinOp, without explicitly forming the dense composite matrix. Verifies Q orthogonality
// and factorization accuracy using test mode to compute the Q-factor.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include "../../functions/drivers/dm_cqrrt_linops.hh"
#include "../../functions/linops_external/dm_cholsolver_linop.hh"
#include "../../functions/misc/dm_util.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

class TestDmCQRRTLinops : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct CQRRTLinopsTestData {
        int64_t row;
        int64_t col;
        int64_t rank; // has to be modifiable
        std::vector<T> R;
        std::vector<T> A_dense; // Dense representation of composite operator for verification
        std::vector<T> A_cpy;
        std::vector<T> I_ref;

        CQRRTLinopsTestData(int64_t m, int64_t n, int64_t k) :
        R(n * n, 0.0),
        A_dense(m * n, 0.0),
        A_cpy(m * n, 0.0),
        I_ref(k * k, 0.0)
        {
            row = m;
            col = n;
            rank = k;
        }
    };

    /// Compute dense representation of A_inv_linop * B where A_inv_linop is CholSolverLinOp and B is sparse
    template <typename T>
    static void compute_composite_dense(
        RandLAPACK_demos::CholSolverLinOp<T>& A_inv_linop,
        RandBLAS::sparse_data::csc::CSCMatrix<T>& B_csc,
        T* result,
        int64_t m,
        int64_t n
    ) {
        // Convert B to dense
        std::vector<T> B_dense(A_inv_linop.n_cols * n, 0.0);
        RandLAPACK::util::sparse_to_dense(B_csc, Layout::ColMajor, B_dense.data());

        // Compute A_inv_linop * B_dense -> result
        A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, A_inv_linop.n_cols,
              1.0, B_dense.data(), A_inv_linop.n_cols, 0.0, result, m);
    }

    template <typename T>
    static void norm_and_copy_computational_helper(T &norm_A, CQRRTLinopsTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A_dense.data(), m, all_data.A_cpy.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A_dense.data(), m);
    }

    /// Error checking routine for CQRRT with linear operators
    template <typename T>
    static void
    error_check(T &norm_A, CQRRTLinopsTestData<T> &all_data, T* Q, int64_t Q_rows, int64_t Q_cols) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = n;

        RandLAPACK::util::upsize(k * k, all_data.I_ref);
        RandLAPACK::util::eye(k, k, all_data.I_ref);

        T* A_dat         = all_data.A_cpy.data();
        T const* A_dense_dat = all_data.A_dense.data();
        T const* Q_dat   = Q;
        T const* R_dat   = all_data.R.data();
        T* I_ref_dat     = all_data.I_ref.data();

        // Check orthogonality of Q
        // Q' * Q - I = 0
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, I_ref_dat, k);
        T norm_0 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);

        // A - QR
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, R_dat, n, -1.0, A_dat, m);

        // Implementing max col norm metric
        T max_col_norm = 0.0;
        T col_norm = 0.0;
        int max_idx = 0;
        for(int i = 0; i < n; ++i) {
            col_norm = blas::nrm2(m, &A_dat[m * i], 1);
            if(max_col_norm < col_norm) {
                max_col_norm = col_norm;
                max_idx = i;
            }
        }
        T col_norm_A = blas::nrm2(n, &A_dense_dat[m * max_idx], 1);
        T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, m);

        printf("REL NORM OF A - QR:    %15e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC:    %15e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF (Q'Q - I)/sqrt(n): %2e\n\n", norm_0 / std::sqrt((T) n));

        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_LE(norm_AQR, atol * norm_A);
        ASSERT_LE(max_col_norm, atol * col_norm_A);
        ASSERT_LE(norm_0, atol * std::sqrt((T) n));
    }

    /// General test for CQRRT_linops with composite operator
    template <typename T, typename RNG, typename CompositeLinOp>
    static void test_CQRRT_linops_general(
        T d_factor,
        T norm_A,
        CQRRTLinopsTestData<T> &all_data,
        RandLAPACK_demos::CQRRT_linops<T, RNG> &CQRRT_linops_alg,
        CompositeLinOp &A_composite,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRT_linops_alg.call(A_composite, all_data.R.data(), n, d_factor, state);

        // Access Q-factor from test mode
        ASSERT_NE(CQRRT_linops_alg.Q, nullptr);
        error_check(norm_A, all_data, CQRRT_linops_alg.Q, CQRRT_linops_alg.Q_rows, CQRRT_linops_alg.Q_cols);
    }
};

// Test CQRRT_linops with composite operator: A_inv_linop * B_sparse
// where A_inv_linop comes from CholSolverLinOp and B_sparse is a SparseLinOp
TEST_F(TestDmCQRRTLinops, CQRRTLinops_composite_cholsolver_sparse) {
    // Dimensions
    int64_t n_spd = 50;   // Size of SPD matrix (A_inv_linop will be n_spd x n_spd)
    int64_t n_sparse_cols = 20;  // Columns in sparse matrix B
    int64_t m = n_spd;    // Rows in composite = rows in A_inv_linop
    int64_t n = n_sparse_cols;  // Cols in composite = cols in B
    int64_t k = n;        // Expected rank
    double d_factor = 2.0;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate and save SPD matrix to temporary file
    std::string spd_filename = "/tmp/test_spd_matrix.mtx";
    RandLAPACK_demos::generate_spd_matrix_file<double>(spd_filename, n_spd, 10.0, state);

    // Create CholSolverLinOp
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(spd_filename);

    // Generate sparse matrix B in COO format
    double density = 0.2;
    auto B_coo = RandLAPACK::gen::gen_sparse_mat<double>(n_spd, n_sparse_cols, density, state);

    // Convert to CSC for SparseLinOp
    RandBLAS::sparse_data::csc::CSCMatrix<double> B_csc(n_spd, n_sparse_cols);
    RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

    // Create SparseLinOp
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> B_sp_linop(n_spd, n_sparse_cols, B_csc);

    // Create CompositeOperator
    RandLAPACK::linops::CompositeOperator A_composite(m, n, A_inv_linop, B_sp_linop);

    // Compute dense representation for verification
    CQRRTLinopsTestData<double> all_data(m, n, k);
    compute_composite_dense(A_inv_linop, B_csc, all_data.A_dense.data(), m, n);

    norm_and_copy_computational_helper(norm_A, all_data);

    // Create CQRRT_linops driver with test mode enabled
    RandLAPACK_demos::CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 2;

    test_CQRRT_linops_general(d_factor, norm_A, all_data, CQRRT_linops_alg, A_composite, state);

    // Clean up
    std::remove(spd_filename.c_str());
}
