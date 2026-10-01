/*
BudgetedPartialSVDSolver: Spectra's partial SVD with a restart budget, returning every
Ritz approximation rather than only the converged ones. This is the baseline the ABRIK
speed comparisons run against at each matvec checkpoint.

Spectra runs implicitly restarted Lanczos on A'A (AA' when m <= n):
  - nev singular triplets are requested (the target rank);
  - ncv is the Krylov subspace dimension, the caller's choice (the speed comparisons use
    min(2 nev + 1, min(m, n) - 1), reduced by effective_ncv for small budgets);
  - max_restarts bounds the restarts;
  - the Lanczos iteration applies A'A ncv + 1 times for the initial factorization and about
    ncv - nev times per restart, each application two matvecs with A; num_operations()
    reports the count actually made.
*/

#ifndef BUDGETED_SVD_SOLVER_HH
#define BUDGETED_SVD_SOLVER_HH

#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/contrib/PartialSVDSolver.h>
#include <Spectra/LinAlg/TridiagEigen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace BenchmarkUtil {

// SymEigsSolver that exposes all nev Ritz pairs after compute(), converged or not, by
// recomputing the eigenvectors of the tridiagonal matrix H.
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

    AllRitzSymEigsSolver(OpType& op, Index nev, Index ncv) : Base(op, nev, ncv) {}

    // The nev wanted Ritz values, largest first: the head of the ncv values Spectra keeps
    // sorted by its selection rule.
    RealVector all_eigenvalues() const
    {
        return this->m_ritz_val.head(this->m_nev);
    }

    // The nev Ritz vectors in the original space, in the order of all_eigenvalues().
    // Recomputes the eigenvectors of H, O(ncv^3), negligible next to the matvecs.
    Matrix all_eigenvectors() const
    {
        Spectra::TridiagEigen<RealScalar> decomp(this->m_fac.matrix_H().real());
        const RealVector& evals = decomp.eigenvalues();
        const RealMatrix& evecs = decomp.eigenvectors();

        std::vector<Index> ind(evals.size());
        std::iota(ind.begin(), ind.end(), 0);
        std::sort(ind.begin(), ind.end(),
                  [&evals](Index a, Index b) { return evals[a] > evals[b]; });

        Index nev = this->m_nev;
        Index ncv = this->m_ncv;
        RealMatrix ritz_vec(ncv, nev);
        for (Index i = 0; i < nev; i++)
            ritz_vec.col(i) = evecs.col(ind[i]);

        // V is the Krylov basis (n x ncv); ritz_vec is ncv x nev.
        return this->m_fac.matrix_V() * ritz_vec;
    }
};

// Spectra's PartialSVDSolver with an explicit restart budget and access to every Ritz
// approximation. MatrixType is dense or sparse Eigen; the matrix is held by reference.
template <typename MatrixType>
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
    std::unique_ptr<Spectra::SVDMatOp<Scalar>> m_op;
    std::unique_ptr<AllRitzSymEigsSolver<Spectra::SVDMatOp<Scalar>>> m_eigs;
    Index m_nev;
    Matrix m_ritz_vecs;   // cached by ritz_vectors(); matrix_U and matrix_V both need them

    const Matrix& ritz_vectors()
    {
        if (m_ritz_vecs.size() == 0)
            m_ritz_vecs = m_eigs->all_eigenvectors();
        return m_ritz_vecs;
    }

    // Singular values from the Ritz values of A'A; a negative Ritz value is rounding
    // noise and maps to zero.
    Vector sigmas(Index count) const
    {
        Vector evals = m_eigs->all_eigenvalues().head(count);
        for (Index i = 0; i < evals.size(); i++)
            evals[i] = std::sqrt(std::max(evals[i], Scalar(0)));
        return evals;
    }

    // Applies A (or A') to the eigenvectors and divides by sigma: the other singular
    // vectors. A zero sigma leaves the column unscaled.
    template <typename Product>
    Matrix scaled(const Product& prod, const Vector& sigma) const
    {
        Matrix result = prod;
        for (Index i = 0; i < sigma.size(); i++)
            if (sigma[i] > 0)
                result.col(i) /= sigma[i];
        return result;
    }

public:
    BudgetedPartialSVDSolver(ConstGenericMatrix& mat, Index ncomp, Index ncv) :
        m_mat(mat), m_m(mat.rows()), m_n(mat.cols()), m_nev(ncomp)
    {
        if (m_m > m_n)
            m_op.reset(new Spectra::SVDTallMatOp<Scalar, MatrixType>(mat));
        else
            m_op.reset(new Spectra::SVDWideMatOp<Scalar, MatrixType>(mat));
        m_eigs.reset(new AllRitzSymEigsSolver<Spectra::SVDMatOp<Scalar>>(*m_op, ncomp, ncv));
    }

    BudgetedPartialSVDSolver(const BudgetedPartialSVDSolver&) = delete;
    BudgetedPartialSVDSolver& operator=(const BudgetedPartialSVDSolver&) = delete;

    // Runs max_restarts restarts: the tolerance is the smallest positive value of the
    // scalar type, so Spectra stops earlier only on an exact Lanczos breakdown (an input
    // of rank below ncv). Returns the number of formally converged values.
    Index compute(Index max_restarts)
    {
        m_ritz_vecs.resize(0, 0);
        m_eigs->init();
        return m_eigs->compute(Spectra::SortRule::LargestAlge, max_restarts,
                               std::numeric_limits<Scalar>::denorm_min());
    }

    // A'A (or AA') applications made by the Lanczos iteration.
    Index num_operations() const { return m_eigs->num_operations(); }

    // All nev singular value approximations, converged or not.
    Vector singular_values() const { return sigmas(m_nev); }

    // The leading nu left singular vector approximations.
    Matrix matrix_U(Index nu)
    {
        nu = std::min(nu, m_nev);
        Matrix evecs = ritz_vectors().leftCols(nu);
        if (m_m <= m_n)
            return evecs;
        return scaled(m_mat * evecs, sigmas(nu));
    }

    // The leading nv right singular vector approximations.
    Matrix matrix_V(Index nv)
    {
        nv = std::min(nv, m_nev);
        Matrix evecs = ritz_vectors().leftCols(nv);
        if (m_m > m_n)
            return evecs;
        return scaled(m_mat.transpose() * evecs, sigmas(nv));
    }
};

// The Krylov dimension a matvec budget can afford: ncv_default when the budget covers the
// initial factorization of ncv + 1 A'A applications, otherwise as large as the budget
// buys, but never below nev + 1, which Spectra requires. Below that the initial
// factorization alone exceeds the budget.
inline int64_t effective_ncv(int64_t budget, int64_t nev, int64_t ncv_default)
{
    int64_t ata_ops = budget / 2;
    if (ata_ops >= ncv_default + 1)
        return ncv_default;
    return std::max(nev + 1, ata_ops - 1);
}

// The restarts a matvec budget buys after the initial factorization of ncv + 1 A'A
// applications, at ncv - nev applications per restart; zero when the budget does not
// reach one restart. The divisor is guarded against a degenerate ncv.
inline int64_t budget_to_restarts(int64_t budget, int64_t nev, int64_t ncv)
{
    int64_t ata_ops = budget / 2;
    if (ata_ops <= ncv + 1 || ncv <= nev)
        return 0;
    return (ata_ops - ncv - 1) / (ncv - nev);
}

}  // namespace BenchmarkUtil

#endif  // BUDGETED_SVD_SOLVER_HH
