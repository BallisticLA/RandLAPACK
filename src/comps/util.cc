#include <RandLAPACK/comps/util.hh>

#include <iostream>
#include <cmath>
#include <lapack.hh>
#include <RandBLAS.hh>

/*
UTILITY ROUTINES
QUESTION: some of these very well can be separate namespace for their degree of seriousness.
However, those routines are not necessarily randomized. What do we do with them?
*/
namespace RandLAPACK::comps::util {


// Generate Identity
// Assuming col-maj
template <typename T>
void eye(
        int64_t m,
        int64_t n,
        T* A 
){
        int64_t min = std::min(m, n);
        for(int j = 0; j < min; ++j)
        {
                A[(m * j) + j] = 1.0;
        }
}

// Diagonalization - turns a vector into a diagonal matrix
template <typename T> 
void diag(
        int64_t m,
        int64_t n,
        T* s, 
        T* S // Assuming S is m by n
) {     
        using namespace blas;
        // size of s
        int64_t k = std::min(m, n);
        copy<T, T>(k, s, 1, S, m + 1);
}

// Helper routine for retrieving the lower triangular portion of a matrix
// Puts 1's on the main diagonal
template <typename T> 
void get_L(
        int64_t m,
        int64_t n,
        T* L
) {
	// Vector end pointer
	int size = m * n;
        // The unit diagonal elements of L were not stored.
        L[0] = 1;
    
        for(int i = m, j = 1; i < size && j < m; i += m, ++j) 
        {             
                std::for_each(L + i, L + i + j,
                        // Lambda expression begins
                        [](T& entry)
                        {
                                entry = 0.0;
                        }
                ); 
                // The unit diagonal elements of L were not stored.
                L[i + j] = 1;
                
        }
}

// Addressing Pivoting
template <typename T> 
void row_swap(
        int64_t m,
        int64_t n,
        T* A, // pointer to the beginning
        int64_t* p // Pivot vector
) {     
        using namespace blas;

        std::vector<T> row_buf(n, 0.0);

        for (int i = 0, j = 0; i < n; ++i)
        {
                j = *(p + i) - 1;
                swap<T, T>(n, A + i, m, A + j, m);
        }
}

// GENERATING MATRICES OF VARYING SPECTRAL DECAY
template <typename T> 
void gen_exp_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // vector length
        T t, // controls the decay. The higher the value, the slower the decay
        int32_t seed
) {   
        std::vector<T> s(k, 0.0);
        std::vector<T> S(k * k, 0.0);
        
        T cnt = 0.0;
        // apply lambda function to every entry of s
        std::for_each(s.begin(), s.end(),
                // Lambda expression begins
                [&t, &cnt](T& entry)
                {
                        entry = (std::exp(++cnt / -t));
                }
        );
        
        // form a diagonal S
        diag<T>(k, k, s.data(), S.data());
        gen_mat<T>(m, n, A, k, S.data(), seed);
}

template <typename T> 
void gen_s_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // <= min(m, n)
        int32_t seed
) {   
        std::vector<T> s(k, 0.0);
        std::vector<T> S(k * k, 0.0);

        T cnt = 0.0;
        // apply lambda function to every entry of s
        std::for_each(s.begin(), s.end(),
                // Lambda expression begins
                [&cnt](T& entry)
                {
                        entry = 0.0001 + (1 / (1 + std::exp(++cnt - 30)));
                }
        );

        // form a diagonal S
        diag<T>(k, k, s.data(), S.data());
        gen_mat<T>(m, n, A, k, S.data(), seed);
}

template <typename T> 
void gen_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // vector length
        T* S,
        int32_t seed
) {   
        using namespace blas;
        using namespace lapack;
        
        std::vector<T> U(m * k, 0.0);
        std::vector<T> V(n * k, 0.0);
        std::vector<T> tau(k, 2.0);
        std::vector<T> Gemm_buf(m * k, 0.0);
        
        RandBLAS::dense_op::gen_rmat_norm<T>(m, k, U.data(), seed);
        RandBLAS::dense_op::gen_rmat_norm<T>(n, k, V.data(), ++seed);

        geqrf(m, k, U.data(), m, tau.data());
        ungqr(m, k, k, U.data(), m, tau.data());

        geqrf(n, k, V.data(), n, tau.data());
        ungqr(n, k, k, V.data(), n, tau.data());

        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U.data(), m, S, k, 0.0, Gemm_buf.data(), m);
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, k, 1.0, Gemm_buf.data(), m, V.data(), n, 0.0, A, m);
}

// Explicit instantiation of template functions - workaround to avoid header implementations
template void eye<float>(int64_t m, int64_t n, float* A );
template void eye<double>(int64_t m, int64_t n, double* A );

template void get_L<float>(int64_t m, int64_t n, float* L);
template void get_L<double>(int64_t m, int64_t n, double* L);

template void diag(int64_t m, int64_t n, float* s, float* S);
template void diag(int64_t m, int64_t n, double* s, double* S);

template void row_swap( int64_t m, int64_t n, float* A, int64_t* p);
template void row_swap( int64_t m, int64_t n, double* A, int64_t* p);

template void gen_s_mat(int64_t m, int64_t n, float* A, int64_t k, int32_t seed);
template void gen_s_mat(int64_t m, int64_t n, double* A, int64_t k, int32_t seed);

template void gen_exp_mat(int64_t m, int64_t n, float* A, int64_t k, float t, int32_t seed); 
template void gen_exp_mat(int64_t m, int64_t n, double* A, int64_t k, double t, int32_t seed);

template void gen_mat(int64_t m, int64_t n, float* A, int64_t k, float* S, int32_t seed); 
template void gen_mat(int64_t m, int64_t n, double* A, int64_t k, double* S, int32_t seed);
} // end namespace util
