#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <vector>


int main(int argc, char** argv) {

    struct qrcp_wide_qp3 {
        std::vector<float> tau_vec;
        void reserve(int64_t _m, int64_t _n) { tau_vec.resize(_n, 0.0); }
        void free() { return; }
        void operator()(int64_t _m, int64_t _n, float* A, int64_t _lda, int64_t* J) {
            randblas_require( static_cast<int64_t>(tau_vec.size()) >= _n);
            float* tau = tau_vec.data();
            lapack::geqp3(_m, _n, A, _lda, J, tau);
        }
    };

    int64_t M = 500;
    int64_t n = 50;
    int64_t b = 20;
    int64_t N = n*b;

    qrcp_wide_qp3 qp3{};
    float d_factor = 1.2;
    RandLAPACK::QRBBRP<float, qrcp_wide_qp3> QRBBRP(qp3, true, b, d_factor);

    RandLAPACK::gen::mat_gen_info<float> m_info(M, N, RandLAPACK::gen::polynomial);
    m_info.exponent = 0.5;
    m_info.cond_num = 1e6;

    std::vector<float>   Avec(M*N, 0.0);
    std::vector<int64_t> Jvec(N,0);
    std::vector<float>   tauvec(N, 0.0);

    float*   A   = Avec.data();
    int64_t* J   = Jvec.data();
    float*   tau = tauvec.data();

    RandBLAS::RNGState data_gen_state(0);
    RandLAPACK::gen::mat_gen(m_info, A, data_gen_state);

    int64_t lda = M;

    RandBLAS::RNGState alg_state(99);
    auto out_alg_state = QRBBRP.call(M, N, A, lda, J, tau, alg_state);
}