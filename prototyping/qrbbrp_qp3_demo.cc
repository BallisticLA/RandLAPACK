#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <vector>
#include <fast_matrix_market/fast_matrix_market.hpp>


//Riley's function
void set_J(int64_t* J, int64_t len_J, int64_t b, int64_t chosen_block) {
    std::iota(J, J + len_J, 1);
    // ^ J = {1, 2, ..., len_J}
    if (chosen_block == 0)
        return;
    std::swap_ranges(J, J + b, J + b*chosen_block);
    return;
}

//Riley's swap function
template <typename T>
void swap_cols(int64_t m, T* A, int64_t lda, int64_t b, int64_t chosen_block) {
    for (int64_t i = 0; i < b; ++i) {
        T* ai_current = A + lda*i;
        T* ai_desired = A + lda*(chosen_block*b + i);
        std::swap_ranges(ai_current, ai_current + m, ai_desired);
    }
}

struct qrcp_wide_qp3 {
    std::vector<float> tau_vec;
    void reserve(int64_t rows, int64_t cols) { tau_vec.resize(cols, 0.0); }
    void free() { return; }
    void operator()(int64_t rows, int64_t cols, float* A, int64_t _lda, int64_t* J) {
        randblas_require( static_cast<int64_t>(tau_vec.size()) >= cols);
        float* tau = tau_vec.data();
        lapack::geqp3(rows, cols, A, _lda, J, tau);
    }
};

//struc for setting up work matrix
template<typename T> 
struct setup_work {

std::vector<T> work_vec;
int64_t num_blocks;
int64_t block_size;

void reserve(int64_t rows, int64_t cols) { work_vec.resize((rows+block_size)*cols + num_blocks + 2*block_size + cols, 0.0); }
void free() { return; }
void operator()(int64_t rows, int64_t cols, T* A, int64_t lda, int64_t* J) {
    num_blocks = cols / block_size;     // decreases from one iteration to the next. 
    T* diagsR    = work_vec.data();     // new T[block_size]();
    T* v         = diagsR + block_size; // new T[num_blocks]{0.0};
    T* tau       = v + num_blocks;      // new T[block_size]();
    T* extra_arr = tau + block_size;    // new T[cols]();
    T* work_mat  = extra_arr + cols;    

    // 1. initialize work_mat to 0
    int64_t rows_w = rows + block_size;
    for (int i = 0; i < cols*rows_w; i++) {
        work_mat[i] = 0.0;
    }

    // 2. set up augmented work matrix
    int indx_work, indx_A;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++){
            indx_A    = j + i*lda;
            indx_work = indx_A + block_size*i;
            work_mat[indx_work] = A[indx_A];
        }
        // then set the bottom idenity
        // rem+1
        indx_work += (i % block_size) + 1;
        work_mat[indx_work] = 1.0;
    }	

    // 3. use mitchell's code to compute pivot scores of blocks
    for (int i = 0; i < num_blocks; ++i) {
        // Call the QR function for each block
        T* W_block = &work_mat[i * block_size * rows_w];
        lapack::geqrf(rows_w, block_size, W_block, rows_w, tau);
        // Extract the diagonal elements of R
        for (int j = 0; j < block_size; ++j) {
            diagsR[j] = W_block[j  * (rows_w + 1)];
        } 
        // Store the sum log |diagsR[j] | for each block in v[j]
        for (int j = 0; j < block_size; ++j) {
            v[i] += std::log(std::abs(diagsR[j]));
        }
    }
    // 4. find max of v
    int64_t idx_of_max = blas::iamax(num_blocks, v, 1);
    // 5. swap blocks of original matrix A
    swap_cols(rows, A, rows, block_size, idx_of_max);
    set_J(J, cols, block_size, idx_of_max);
    // 6. unpivoted QR on A matrix (original matrix, not the augmented)
    lapack::geqrf(rows, cols, A, lda, extra_arr);

} //end operator
};


void qp3_demo() {
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

void logdet_demo() {
    using T = float;
    int64_t mm = 4;
    int64_t nn = 8;
    std::vector<T> M0(mm*nn);
    for (int64_t i = 0; i < nn; ++i) {
        M0[i*mm + (i%mm)] = (T) (i+1);
    }

    RandBLAS::print_buff_to_stream(std::cout, blas::Layout::ColMajor, mm, nn, M0.data(), mm, "M0", 2);
    
    //tau = vec of floats, same to gqref init to 0, len cols
    T* tau = new T[nn];
    for (int i = 0; i < nn; i++) { tau[i] = 0.0;}

    //call the setup_work function
    setup_work<float> qrcp_wide{};
    qrcp_wide.block_size = 2;
    qrcp_wide.num_blocks = nn/qrcp_wide.block_size;

    RandLAPACK::QRBBRP<float, setup_work<float>> alg(qrcp_wide, true, qrcp_wide.block_size, 2.0); 

    ///J is just array int 64 of all 0s num cols
    int64_t* J = new int64_t[nn];
    for (int i = 0; i < nn; i++) { J[i] = 0;}

    RandBLAS::RNGState alg_state(99);
    auto out_alg_state = alg.call(mm, nn, M0.data(), mm, J, tau, alg_state);
    RandBLAS::print_buff_to_stream(std::cout, 1, nn, J, nn, 1, "J", 2);
    return;
}


int main(int argc, char** argv) {

    using T = float;
    if (argc < 2)
        throw std::invalid_argument("Must specify a filename.");
    int64_t mm;
    int64_t nn;
    std::vector<T> Avec(0);
    // example file name: ipam2025/data/Jt_cptp_2qxyi_1920x45396.mtx
    std::string filename = argv[1];
    std::ifstream file(filename);
    fast_matrix_market::read_matrix_market_array(
        file, mm, nn, Avec, fast_matrix_market::col_major
    );

    // call the setup_work function
    setup_work<float> qrcp_wide{};
    qrcp_wide.block_size = 4;
    qrcp_wide.num_blocks = nn/qrcp_wide.block_size;

    RandLAPACK::QRBBRP<float, setup_work<float>> alg(qrcp_wide, true, qrcp_wide.block_size, 2.0); 

    // J is just array int 64 of all 0s num cols
    T* tau = new T[nn]{};
    int64_t* J = new int64_t[nn]{};
    RandBLAS::RNGState alg_state(99);
    auto out_alg_state = alg.call(mm, nn, Avec.data(), mm, J, tau, alg_state);
    RandBLAS::print_buff_to_stream(std::cout, 1, 64, J, nn, 1, "J", 2);
}
