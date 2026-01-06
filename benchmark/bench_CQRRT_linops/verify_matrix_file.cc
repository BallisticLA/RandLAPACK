#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <string>

// Quick utility to verify what's actually in a matrix file

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    std::string filename = argv[1];
    printf("Loading matrix from: %s\n", filename.c_str());

    // Load as sparse
    Eigen::SparseMatrix<double, Eigen::ColMajor> A_sparse;
    Eigen::loadMarket(A_sparse, filename);

    printf("Matrix dimensions: %ld x %ld\n", A_sparse.rows(), A_sparse.cols());
    printf("Non-zeros: %ld (%.2f%% dense)\n", A_sparse.nonZeros(),
           100.0 * A_sparse.nonZeros() / (A_sparse.rows() * A_sparse.cols()));

    // Convert to dense for eigenvalue computation
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse);

    // Check symmetry
    double sym_error = (A_dense - A_dense.transpose()).norm() / A_dense.norm();
    printf("Symmetry error: %.6e\n", sym_error);

    // Compute eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_dense);
    if (es.info() != Eigen::Success) {
        printf("ERROR: Eigenvalue computation failed!\n");
        return 1;
    }

    auto eigenvalues = es.eigenvalues();
    double lambda_min = eigenvalues(0);
    double lambda_max = eigenvalues(A_dense.rows()-1);
    double actual_cond = lambda_max / lambda_min;

    printf("λ_min = %.6e\n", lambda_min);
    printf("λ_max = %.6e\n", lambda_max);
    printf("Actual condition number: %.6e\n", actual_cond);

    // Try sparse Cholesky
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> sparse_chol;
    sparse_chol.compute(A_sparse);
    printf("Sparse Cholesky (NaturalOrdering): %s\n",
           sparse_chol.info() == Eigen::Success ? "SUCCESS" : "FAILED");

    // Try dense Cholesky
    Eigen::LLT<Eigen::MatrixXd> dense_chol(A_dense);
    printf("Dense Cholesky: %s\n",
           dense_chol.info() == Eigen::Success ? "SUCCESS" : "FAILED");

    return 0;
}
