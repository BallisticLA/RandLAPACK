/*
BudgetedSVDSolver — a wrapper around Spectra's PartialSVDSolver that exposes
a "max_restarts" parameter and returns ALL Ritz approximations (not just
converged ones).

This allows benchmarking SVDS with a controlled computational budget,
making it directly comparable to ABRIK's (b_sz, num_matmuls) parameterization.

Algorithm: Implicitly Restarted Lanczos (single-vector) for eigenvalues of A'A.
  - nev eigenvalues requested (= target_rank)
  - ncv Krylov subspace dimension (= 2*nev + 1)
  - max_restarts controls the number of restart iterations
  - Total A'A operations ≈ ncv + max_restarts * (ncv - nev)
  - Total matvecs with A ≈ 2 * (total A'A operations)
*/

#ifndef BUDGETED_SVD_SOLVER_HH
#define BUDGETED_SVD_SOLVER_HH

#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/contrib/PartialSVDSolver.h>
#include <Spectra/LinAlg/TridiagEigen.h>
#include <algorithm>
#include <cmath>

namespace BenchmarkUtil {

// Subclass of SymEigsSolver that can extract ALL Ritz pairs (even unconverged)
// by recomputing eigenvectors from the tridiagonal matrix H.
template <typename OpType>
class AllRitzSymEigsSolver : public Spectra::SymEigsSolver<OpType>
{
public:
    using Base = Spectra::SymEigsSolver<OpType>;
    using Scalar = typename OpType::Scalar;
    using Index = Eigen::Index;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealMatrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealVector = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    AllRitzSymEigsSolver(OpType& op, Index nev, Index ncv) :
        Base(op, nev, ncv) {}

    // Return ALL nev Ritz values (eigenvalue approximations), sorted largest first.
    // These are available after compute() regardless of convergence status.
    RealVector all_eigenvalues() const
    {
        // m_ritz_val is protected, contains ncv Ritz values sorted by selection rule
        // First nev entries are the "wanted" ones (largest for SVD).
        return this->m_ritz_val.head(this->m_nev);
    }

    // Return ALL nev Ritz vectors in the original space, sorted to match all_eigenvalues().
    // Recomputes eigenvectors of H from scratch (O(ncv^3), negligible vs matvecs).
    Matrix all_eigenvectors() const
    {
        // Get the tridiagonal matrix H from the Lanczos factorization
        Spectra::TridiagEigen<RealScalar> decomp(this->m_fac.matrix_H().real());
        const RealVector& evals = decomp.eigenvalues();
        const RealMatrix& evecs = decomp.eigenvectors();

        // Sort eigenvalues largest first (matching all_eigenvalues() order)
        std::vector<Index> ind(evals.size());
        std::iota(ind.begin(), ind.end(), 0);
        std::sort(ind.begin(), ind.end(),
                  [&evals](Index a, Index b) { return evals[a] > evals[b]; });

        // Extract first nev eigenvectors in sorted order
        Index nev = this->m_nev;
        Index ncv = this->m_ncv;
        RealMatrix ritz_vec(ncv, nev);
        for (Index i = 0; i < nev; i++)
            ritz_vec.col(i) = evecs.col(ind[i]);

        // Transform to original space: V * ritz_vec
        // V is the Krylov basis (n x ncv), ritz_vec is (ncv x nev)
        return this->m_fac.matrix_V() * ritz_vec;
    }
};

// Budgeted Partial SVD Solver.
// Like Spectra::PartialSVDSolver but with explicit max_restarts control
// and access to ALL Ritz approximations.
template <typename MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
class BudgetedPartialSVDSolver
{
public:
    using Scalar = typename MatrixType::Scalar;
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ConstGenericMatrix = const Eigen::Ref<const MatrixType>;

private:
    ConstGenericMatrix m_mat;
    const Index m_m;
    const Index m_n;
    Spectra::SVDMatOp<Scalar>* m_op;
    AllRitzSymEigsSolver<Spectra::SVDMatOp<Scalar>>* m_eigs;
    Index m_nconv;
    Index m_nev;

public:
    BudgetedPartialSVDSolver(ConstGenericMatrix& mat, Index ncomp, Index ncv) :
        m_mat(mat), m_m(mat.rows()), m_n(mat.cols()), m_nev(ncomp)
    {
        if (m_m > m_n)
            m_op = new Spectra::SVDTallMatOp<Scalar, MatrixType>(mat);
        else
            m_op = new Spectra::SVDWideMatOp<Scalar, MatrixType>(mat);

        m_eigs = new AllRitzSymEigsSolver<Spectra::SVDMatOp<Scalar>>(*m_op, ncomp, ncv);
    }

    ~BudgetedPartialSVDSolver()
    {
        delete m_eigs;
        delete m_op;
    }

    // Run with explicit max_restarts.
    // Uses tiny tolerance to force all restart iterations.
    // Returns number of (formally) converged eigenvalues.
    Index compute(Index max_restarts)
    {
        m_eigs->init();
        // Use extremely small tol to prevent early stopping
        m_nconv = m_eigs->compute(Spectra::SortRule::LargestAlge, max_restarts, 1e-100);
        return m_nconv;
    }

    // Number of matrix operations performed (A'A applications)
    Index num_operations() const { return m_eigs->num_operations(); }

    // ALL singular value approximations (nev values, even unconverged)
    Vector singular_values() const
    {
        Vector evals = m_eigs->all_eigenvalues();
        // Clamp negative eigenvalues to small positive (numerical noise)
        for (Index i = 0; i < evals.size(); i++)
            evals[i] = std::sqrt(std::max(evals[i], Scalar(0)));
        return evals;
    }

    // ALL left singular vector approximations
    Matrix matrix_U(Index nu)
    {
        nu = (std::min)(nu, m_nev);
        Matrix evecs = m_eigs->all_eigenvectors().leftCols(nu);
        Vector evals = m_eigs->all_eigenvalues().head(nu);

        if (m_m <= m_n)
            return evecs;

        // U = A * V * diag(1/sigma)
        Matrix result = m_mat * evecs;
        for (Index i = 0; i < nu; i++) {
            Scalar sigma = std::sqrt(std::max(evals[i], Scalar(0)));
            if (sigma > 0)
                result.col(i) /= sigma;
        }
        return result;
    }

    // ALL right singular vector approximations
    Matrix matrix_V(Index nv)
    {
        nv = (std::min)(nv, m_nev);
        Matrix evecs = m_eigs->all_eigenvectors().leftCols(nv);
        Vector evals = m_eigs->all_eigenvalues().head(nv);

        if (m_m > m_n)
            return evecs;

        // V = A' * U * diag(1/sigma)
        Matrix result = m_mat.transpose() * evecs;
        for (Index i = 0; i < nv; i++) {
            Scalar sigma = std::sqrt(std::max(evals[i], Scalar(0)));
            if (sigma > 0)
                result.col(i) /= sigma;
        }
        return result;
    }
};

// Convert a matvec budget to max_restarts for the Lanczos solver.
// budget = total matvecs with A (comparable to ABRIK's b_sz * num_matmuls)
// Each A'A application = 2 matvecs with A.
// Initial Lanczos: ncv A'A ops. Each restart: ~(ncv - nev) A'A ops.
inline int64_t budget_to_restarts(int64_t budget, int64_t nev, int64_t ncv)
{
    int64_t ata_ops = budget / 2;  // A'A operations from matvec budget
    if (ata_ops <= ncv)
        return 1;  // minimum 1 restart
    return std::max((int64_t)1, (ata_ops - ncv) / (ncv - nev));
}

}  // namespace BenchmarkUtil

#endif  // BUDGETED_SVD_SOLVER_HH
