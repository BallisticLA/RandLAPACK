#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <chrono>
#include <climits> 
/*
Auxillary benchmark routine, computes flops using GEQRF for a given system
*/

using namespace std::chrono;
using namespace RandLAPACK;
using namespace std;


template <typename T>
T* batched_qr_demo(
    T* matrix, 
    int64_t rows,
    int64_t cols,
    int64_t n_blocks) {

    T* v = new T[n_blocks]{0.0};

    if (cols % n_blocks != 0) {
        throw std::runtime_error("Number of matrix columns must be divisible by the number of blocks.");
    }
       
    int cols_per_block = cols / n_blocks;
    T* diagsR = new T[cols_per_block]();
    T* tau = new T[cols_per_block](); 


    for (int i = 0; i < n_blocks; ++i) {
        // Call the QR function for each block
        T* A_block = &matrix[i * cols_per_block * rows];
        lapack::geqrf(rows, cols_per_block, A_block, rows, tau);

        // Extract the diagonal elements of R
        for (int j = 0; j < cols_per_block; ++j) {
            diagsR[j] = A_block[j  * (rows + 1)];
        }

        // Store the sum log |diagsR[j] | for each block in v[j]
        for (int j = 0; j < cols_per_block; ++j) {
            v[i] += std::log(std::abs(diagsR[j]));
        }

        std::cout << "Block " << i << ": " << v[i] << std::endl;      
    }
    delete[] tau;
    delete[] diagsR;
    return v;
}


int main(int argc, char *argv[]) {

    auto m         = 3;//std::stol(argv[3]);
    auto n         = 12;//std::stol(argv[4]);
    auto n_blocks  = n/m;//std::stol(argv[5]);


    std::vector<float> M0(m*n);
    for (int64_t i = 0; i < n; ++i) {
        M0[i*m + (i%m)] = (float) (i+1);
    }


    batched_qr_demo(M0.data(), m, n, n_blocks);


    return 0;
}
