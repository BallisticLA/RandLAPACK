
#include "kernelbench_common.hh"
#include <RandLAPACK.hh>
#include <RandBLAS.hh>
#include <lapack.hh>
#include <blas.hh>


#ifndef DOUT
#define DOUT(_d) std::setprecision(8) << _d
#endif

using RandLAPACK::rp_cholesky;
using blas::Layout;
using lapack::gesdd;
using lapack::Job;
using std::vector;



template <typename T>
int cholsvd_square(int64_t m, int64_t n, T* A, int64_t lda, T* singvals_squared, T* work) {
    auto layout = Layout::ColMajor;
    auto uplo = blas::Uplo::Lower;
    blas::syrk(layout, uplo, blas::Op::Trans, n, m, (T)1.0, A, lda, 0.0, work, n);
    lapack::syevd(Job::Vec, uplo, n, work, n, singvals_squared);
    // The first n*n entries in work hold the right singular vectors of A.
    // But they're sorted in the wrong order!
    for (int64_t j = 0; j < n/2; ++j) {
        auto lead_off  = j;
        auto trail_off = n-j-1;
        T* colj       = work +   lead_off * n;
        T* coljtrail  = work +  trail_off * n;
        for (int64_t i = 0; i < n; ++i) {
            std::swap(colj[i], coljtrail[i]);
        }
        std::swap(singvals_squared[lead_off], singvals_squared[trail_off]);
    }
    T* trailing_work = work + n*n;
    lapack::lacpy(lapack::MatrixType::General, m, n, A, m, trailing_work, m);
    // trailing_work is a copy of A.
    blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, m, n, n, (T)1.0, trailing_work, m, work, n, (T)0.0, A, lda);
    // invert the scale on each column of A.
    for (int64_t i = 0; i < n; ++i)
        blas::scal(m, (T) std::pow(singvals_squared[i], -0.5), A + i*lda, 1);
    return 0;
}

enum TSSVD : char {
    GESDD    = 'G',
    CholSVD  = 'C',
    RandPrecondCholSVD = 'R'
};

template <typename T>
std::pair<timepoint_t,timepoint_t> convert_svd(int64_t m, int64_t rank, vector<T> &U, vector<T> &kevals, TSSVD cs = TSSVD::GESDD) {
    auto _tp0 = std_clock::now();
    if (cs == TSSVD::GESDD) {
        vector<T> work(rank*rank, 0.0);
        gesdd(Job::OverwriteVec, m, rank, U.data(), m, kevals.data(), nullptr, 1, work.data(), rank);
        for (int64_t i = 0; i < rank; ++i)
            kevals[i] = std::pow(kevals[i], 2);
    } else if (cs == TSSVD::CholSVD) {
        vector<T> work((rank + m)*rank, 0.0);
        cholsvd_square(m, rank, U.data(), m, kevals.data(), work.data());
    }
    auto _tp1 = std_clock::now();
    return {_tp0, _tp1};
}


int main() {
    //std::string dn{"/Users/rjmurr/Documents/open-data/kernel-ridge-regression/sensit_vehicle"};
    std::string dn{"/Users/rjmurr/Documents/open-data/kernel-ridge-regression/cod-rna"};
    auto krrd = mmread_krr_data_dir(dn);
    using T = double;
    int64_t m = krrd.X_train.ncols;
    int64_t d = krrd.X_train.nrows;
    std::cout << "\nDataset\n " << dn << std::endl;
    std::cout << " cols : " << m << std::endl; 
    std::cout << " rows : " << d << "\n\n";
    vector<T> mus{0.0};
    RandLAPACK::linops::RBFKernelMatrix K_reg(m, krrd.X_train.vals.data(), d, 3.0, mus);
    K_reg.set_eval_includes_reg(false);

    // Variables for RPCholesky
    int64_t rpchol_block_size = 64;
    int64_t rank = (int64_t) std::sqrt(m);
    vector<T> U(m * rank, 0.0);
    RNGState state(0);
    vector<int64_t> selection(rank, -1);

    std::cout << "RPCholesky (RPC)\n";
    std::cout << " block size   : " << rpchol_block_size << std::endl;
    std::cout << " rank limit   : " << rank << std::endl;
    auto _tp0 = std_clock::now();
    state = rp_cholesky(m, K_reg, rank, selection.data(), U.data(), rpchol_block_size, state);
    auto _tp1 = std_clock::now();
    std::cout << " exit rank    : " << rank << std::endl;
    std::cout << " RPC time (s) : " << DOUT(sec_elapsed(_tp0, _tp1)) << std::endl;

    // Variables for SVD conversion
    //      We don't allocate these earlier, since "rank" might have decreased
    //      in the call to rp_cholesky.
    vector<T> kevals(rank, 0.0);

{
    auto [tp0, tp1] = convert_svd(m, rank, U, kevals, TSSVD::CholSVD);
    std::cout << " SVD time (s) : " << DOUT(sec_elapsed(tp0, tp1)) << "\n\n";
}
    // Now check || K_reg @ U[:, 0:num_pc] - U[:,0:num_pc] @ diag(eivals[0:num_pc]) ||,
    //        or || K_reg @ U[:, 0:num_pc] @ inv(diag(eigvals[0:num_pc])) - U[:,0:num_pc]||
    int64_t num_pc = 2;
    vector<T> V(m*num_pc, 0.0);
    T onef = 1.0;
    K_reg(blas::Layout::ColMajor, num_pc, onef, U.data(), m, (T)0.0, V.data(), m);
    for (int64_t i = 0; i < num_pc; ++i)
        blas::scal(m, onef/kevals[i], V.data() + i*m, 1);
    // ^ Now, V = K_reg @ U[:, 0:num_pc] @ inv(diag(eigvals[0:num_pc]))
    vector<T> W(V);
    // subtract off U
    for (int64_t i = 0; i < m*num_pc; ++i)
        W[i] -= U[i];
    // compute column norms of W.
    std::cout << "Error in KPCA components " << std::endl;
    for (int64_t i = 0; i < num_pc; ++i) {
        std::cout << " component " << i << " : " << DOUT(blas::nrm2(m, W.data()+i*m, 1)) << std::endl;
    }
    std::cout << std::endl;
    return 0;
}
