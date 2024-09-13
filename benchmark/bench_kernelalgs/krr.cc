
#include "kernelbench_common.hh"

#ifndef DOUT
#define DOUT(_d) std::setprecision(std::numeric_limits<double>::max_digits10) << _d
#endif

#ifndef TIMED_LINE
#define TIMED_LINE(_op, _name) { \
        auto _tp0 = std_clock::now(); \
        _op; \
        auto _tp1 = std_clock::now(); \
        auto dtime = sec_elapsed(_tp0, _tp1); \
        std::cout << _name << DOUT(dtime) << std::endl; \
        }
#endif

int main() {
    //std::string dn{"/Users/rjmurr/Documents/open-data/kernel-ridge-regression/sensit_vehicle"};
    std::string dn{"/Users/rjmurr/Documents/open-data/kernel-ridge-regression/cod-rna"};
    auto krrd = mmread_krr_data_dir(dn);
    using T = double;
    int64_t m = krrd.X_train.ncols;
    int64_t d = krrd.X_train.nrows;
    std::cout << "cols  : " << m << std::endl; 
    std::cout << "rows  : " << d << std::endl;
    T mu_min = m * 1e-7;
    vector<T> mus{mu_min};
    RandLAPACK::linops::RBFKernelMatrix A_linop(m, krrd.X_train.vals.data(), d, 3.0, mus);
    for (int64_t s = 1; s <= 8; s*=2) {
        vector<T> H(m*s, 0.0);

        T* Hd = H.data();
        T* hd = krrd.Y_train.vals.data();
        blas::copy(m, hd, 1, Hd, 1);
        if (s > 1) {
            RNGState state_H(1);
            RandBLAS::DenseDist D(m, s - 1, RandBLAS::ScalarDist::Gaussian);
            RandBLAS::fill_dense(D, Hd + m, state_H);
            T nrm_h = blas::nrm2(m, hd, 1);
            for (int i = 1; i < s; ++i) {
                // T nrm_Hi = blas::nrm2(m, Hd + i*m, 1);
                // T scale = std::pow(2.0*nrm_Hi, -1); 
                // blas::scal(m, scale, Hd + i*m, 1);
                // blas::axpy(m, 1.0, hd, 1, Hd + i*m, 1);
                T nrm_Hi = blas::nrm2(m, Hd + i*m, 1);
                T scale = nrm_h / nrm_Hi;
                blas::scal(m, scale, Hd + i*m, 1);
            }
        }

        vector<T> X(m*s, 0.0);
        // solve A_linop X == H
        RNGState state(0);
        auto seminorm = [](int64_t n, int64_t s, const T* NR){return blas::nrm2(n, NR, 1);};
        int64_t k = 2*1024;
        int64_t rpc_b = 64;
        int64_t eval_block_size = 1024;
        std::cout << "k     : " << k << std::endl;
        std::cout << "s     : " << s << std::endl;
        std::cout << "mu0   : " << mu_min << std::endl;
        std::cout << "rpc_b : " << rpc_b << std::endl << std::endl;
        T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        int64_t max_iters = 25;
        A_linop._eval_block_size = eval_block_size;
        A_linop._eval_work1.resize(A_linop._eval_block_size * m);
        TIMED_LINE(
        RandLAPACK::krill_full_rpchol(
            m, A_linop, H, X, tol, state, seminorm, rpc_b, max_iters, k
        );, "\nKrill : ")
        std::cout << std::endl;
    }
    return 0;
}
