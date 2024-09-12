
#include "kernelbench_common.hh"

#ifndef DOUT
#define DOUT(_d) std::setprecision(8) << _d
#endif

using RandLAPACK::rp_cholesky;
using lapack::gesdd;
using lapack::Job;
using std::vector;


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
    vector<T> work(rank*rank, 0.0);
    vector<T> ksvals(rank, 0.0);
    vector<T> kevals(rank, 0.0);

    _tp0 = std_clock::now();
    gesdd(Job::OverwriteVec, m, rank, U.data(), m, ksvals.data(), nullptr, 1, work.data(), rank);
    _tp1 = std_clock::now();
    std::cout << " SVD time (s) : " << DOUT(sec_elapsed(_tp0, _tp1)) << "\n\n";
    for (int64_t i = 0; i < rank; ++i)
        kevals[i] = std::pow(ksvals[i], 2);

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
