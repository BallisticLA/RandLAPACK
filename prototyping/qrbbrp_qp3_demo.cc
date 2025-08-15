#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <vector>

//Riley's function
void set_J(int64_t* J, int64_t len_J, int64_t b, int64_t chosen_block) {
    std::iota(J, J + len_J, 1);
    // ^ J = {1, 2, ..., len_J}
    std::swap_ranges(J, J + b, J + b*chosen_block);
    return;
}

//Riley's swap function
template <typename T>
void swap_cols(int64_t m, int64_t n, T* A, int64_t lda, int64_t b, int64_t chosen_block) {
    for (int64_t i = 0; i < b; ++i) {
        T* ai_current = A + lda*i;
        T* ai_desired = A + lda*(chosen_block*b + i);
        std::swap_ranges(ai_current, ai_current + m, ai_desired);
    }
}

//struc for setting up work matrix
template<typename T> 
struct setup_work {

    std::vector<T> work_matrix;
    int64_t num_blocks;
    int64_t block_size;

    void reserve(int64_t rows, int64_t cols) { work_matrix.resize((rows+block_size)*cols, 0.0); }
    void free() { return; }
    void operator()(int64_t rows, int64_t cols, T* A, int64_t _lda, int64_t* J) {
	   

    T* diagsR = new T[block_size]();
    T* v      = new T[num_blocks]{0.0};
    T* tau    = new T[block_size]();

    T* extra_arr = new T[cols](); //for geqrf

    //1. initialize work_matrix to 0
    for (int i = 0; i < cols*(rows+block_size); i++) {
       work_matrix[i] = 0.0;
    }

    //2. set up augmented work matrix
    int indx_work, indx_A;
    for (int i = 0; i < cols; i++) {
       for (int j = 0; j < rows; j++){
          indx_A    = j + i*rows;
          indx_work = indx_A + block_size*i;
          
	  work_matrix[indx_work] = A[indx_A];
        }

       //then set the bottom idenity
       //rem+1
       indx_work += (i % block_size) + 1;
       work_matrix[indx_work] = 1.0;
    }	
          

    //3. use mitchell's code to compute pivot scores of blocks
    for (int i = 0; i < num_blocks; ++i) {
       // Call the QR function for each block
       T* A_block = &A[i * block_size * rows];
       lapack::geqrf(rows, block_size, A_block, rows, tau);

       // Extract the diagonal elements of R
       for (int j = 0; j < block_size; ++j) {
         diagsR[j] = A_block[j  * (rows + 1)];
       } 

       // Store the sum log |diagsR[j] | for each block in v[j]
       for (int j = 0; j < block_size; ++j) {
           v[i] += std::log(std::abs(diagsR[j]));
       }

     }

    //4. swap blocks of original matrix A
    //find max of v

    float max_score = v[0];
    int idx_of_max  = 0;
    for (int i = 1; i < num_blocks; i++){
       if (v[i] > max_score){
	  max_score = v[i];
	  idx_of_max = i;
       }
    }

    swap_cols(rows, cols, A, rows, block_size, idx_of_max);

    //6. unpivoted QR on A matrix
    //original matrix, not the augmented
    lapack::geqrf(rows, cols, A, rows, extra_arr);

    } //end operator
};


int main(int argc, char** argv) {

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

//
// Riley's test problem
    using T    = float;
    int64_t mm = 8;
    int64_t nn = 12;
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
    qrcp_wide.num_blocks = 3; //setting some variables just to test M0
    qrcp_wide.block_size = 4;

    RandLAPACK::QRBBRP<float, setup_work<float>> alg(qrcp_wide, true, 2,1.2); 

    ///J is just array int 64 of all 0s num cols
    int64_t* J = new int64_t[nn];
    for (int i = 0; i < nn; i++) { J[i] = 0.0;}

    RandBLAS::RNGState alg_state(99);
    auto out_alg_state = alg.call(mm, nn, M0.data(), mm, J, tau, alg_state);

//-----------------------------------------

//----RILEY'S OG--------------------------
/*
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
*/
    
    
}
