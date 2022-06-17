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
        std::vector<T>& A 
){
        T* A_dat = A.data();
        int64_t min = std::min(m, n);
        for(int j = 0; j < min; ++j)
        {
                A_dat[(m * j) + j] = 1.0;
        }
}

// Diagonalization - turns a vector into a diagonal matrix
template <typename T> 
void diag(
        int64_t m,
        int64_t n,
        const std::vector<T>& s, 
        int64_t k, // size of s, < min(m, n)
        std::vector<T>& S // Assuming S is m by n
) {     
        using namespace blas;
        // size of s
        copy<T, T>(k, s.data(), 1, S.data(), m + 1);
}

template <typename T> 
void disp_diag(
        int64_t m,
        int64_t n,
        int64_t k, 
        std::vector<T>& A 
) {     
        T* A_dat = A.data();
        if (k == 0)
        {
                k = std::min(m, n);
        }
        printf("DISPLAYING THE MAIN DIAGONAL OF A GIVEN MATRIX: \n");
        for(int i = 0; i < k; ++i)
        {
                printf("ELEMENT %d: %f\n", i, *(A_dat + (i * m) + i));
        }
}


// Helper routine for retrieving the lower triangular portion of a matrix
// Puts 1's on the main diagonal
template <typename T> 
void get_L(
        int64_t m,
        int64_t n,
        std::vector<T>& L
) {
	// Vector end pointer
	int size = m * n;
        // The unit diagonal elements of L were not stored.
        T* L_dat = L.data();
        L_dat[0] = 1;
    
        for(int i = m, j = 1; i < size && j < m; i += m, ++j) 
        {             
                std::for_each(L_dat + i, L_dat + i + j,
                        // Lambda expression begins
                        [](T& entry)
                        {
                                entry = 0.0;
                        }
                ); 
                // The unit diagonal elements of L were not stored.
                L_dat[i + j] = 1;
                
        }
}

// Addressing Pivoting
template <typename T> 
void row_swap(
        int64_t m,
        int64_t n,
        std::vector<T>& A, // pointer to the beginning
        const std::vector<int64_t>& p // Pivot vector
) {     
        using namespace blas;
        const int64_t* p_dat = p.data();
        T* A_dat = A.data();

        std::vector<T> row_buf(n, 0.0);

        for (int i = 0, j = 0; i < n; ++i)
        {
                j = *(p_dat + i) - 1;
                swap<T, T>(n, A_dat + i, m, A_dat + j, m);
        }
}

// Resulting array is to be k by n
template <typename T> 
void row_resize(
        int64_t m,
        int64_t n,
        std::vector<T>& A, // pointer to the beginning
        int64_t k
) {     
        using namespace blas;
        T* A_dat = A.data();
        uint64_t end = k;

        for (int i = 1; i < n; ++i)
        {
                // Place jth column (of k entries) after the (j - 1)st column
                copy(k, A_dat + (m * i), 1, A_dat + end, 1);
                end += k;
        }
        // Cut off the end
        A.resize(k * n);
}

template <typename T> 
void qb_resize(
        int64_t m,
        int64_t n,
        std::vector<T>& Q,
        std::vector<T>& B,
        int64_t& k,
        int64_t curr_sz
) { 
        Q.resize(m * curr_sz);
        row_resize<T>(k, n, B, curr_sz);
        k = curr_sz;
}

template <typename T> 
void gen_mat_type(
        int64_t& m, // These may change
        int64_t& n,
        std::vector<T>& A,
        int64_t k, 
        int32_t seed,
        std::tuple<int, T, bool> type
) {  
        T* A_dat = A.data();

        switch(std::get<0>(type)) 
        {
                case 0:
                        // Generating matrix with polynomially decaying singular values
                        printf("TEST MATRIX: POLYNOMIAL DECAY sigma_i = 1 / (i + 1)^pow (first k * 0.2 sigmas = 1)\n");
                        RandLAPACK::comps::util::gen_poly_mat<T>(m, n, A, k, std::get<1>(type), std::get<2>(type), seed); 
                        break;
                case 1:
                        // Generating matrix with exponentially decaying singular values
                        printf("TEST MATRIX: EXPONENTIAL DECAY sigma_i = e^((i + 1) / -pow) (first k * 0.2 sigmas = 1)\n");
                        RandLAPACK::comps::util::gen_exp_mat<T>(m, n, A, k, std::get<1>(type), seed); 
                        break;
                case 2:
                        // Generating matrix with s-shaped singular values plot
                        printf("TEST MATRIX: S-SHAPED DECAY (first k * 0.2 sigmas = 1)\n");
                        RandLAPACK::comps::util::gen_s_mat<T>(m, n, A, k, seed); 
                        break;
                case 3:
                        // A = [A A]
                        printf("TEST MATRIX: A = [A A]\n");
                        RandBLAS::dense_op::gen_rmat_norm<T>(m, k, A_dat, seed);
                        if (2 * k <= n)
                        {
                        
                        std::copy(A_dat, A_dat + (n / 2) * m, A_dat + (n / 2) * m);
                        }
                        break;
                case 4:
                // Zero matrix
                printf("TEST MATRIX: ZERO\n");
                break;
                case 5:
                {
                        // Random diagonal A of rank k
                        printf("TEST MATRIX: RANDOM DIAGONAL\n");
                        std::vector<T> buf(k, 0.0);
                        RandBLAS::dense_op::gen_rmat_norm<T>(k, 1, buf.data(), seed);
                        // Fills the first k diagonal elements
                        diag<T>(m, n, buf, k, A);
                        break;
                }
                case 6:
                {
                        // A = diag(sigma), where sigma_1 = ... = sigma_l > sigma_{l + 1} = ... = sigma_n
                        // In the case below, sigma = 1 | 0.5
                        printf("TEST MATRIX: A = diag(sigma), where sigma_1 = ... = sigma_l > sigma_{l + 1} = ... = sigma_n\n");
                        std::vector<T> buf(n, 1.0);
                        T* buf_dat = buf.data();
                        std::for_each(buf.begin() + (n / 2), buf.end(),
                                // Lambda expression begins
                                [](T& entry)
                                {
                                        entry -= 0.5;
                                }
                        );
                        diag<T>(m, n, buf, n, A);
                }
                break;
        }
}

// Merge 3 functions below
template <typename T> 
void gen_poly_mat(
        int64_t& m,
        int64_t& n,
        std::vector<T>& A,
        int64_t k, // vector length
        T t, // controls the decay. The higher the value, the slower the decay
        bool diagon,
        int32_t seed
) {   
        using namespace lapack;

        // Predeclare to all nonzero constants, start decay where needed 
        std::vector<T> s(k, 1.0);
        std::vector<T> S(k * k, 0.0);
        
        T cnt = 0.0;
        // apply lambda function to every entry of s       
        std::for_each(s.begin() + k * 0.2, s.end(),
                // Lambda expression begins
                [&t, &cnt](T& entry)
                {
                        entry = 1 / std::pow(++cnt, t);
                }
        );
        
        // form a diagonal S
        diag<T>(k, k, s, k, S);
        if (diagon)
        {
                if (!(m == k || n == k))
                {
                        m = k;
                        n = k;
                        A.resize(k * k);
                }
                lacpy(MatrixType::General, k, k, S.data(), k, A.data(), k);
        }
        else
        {
                gen_mat<T>(m, n, A, k, S, seed);
        }
}

template <typename T> 
void gen_exp_mat(
        int64_t& m,
        int64_t& n,
        std::vector<T>& A,
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
        diag<T>(k, k, s, k, S);
        gen_mat<T>(m, n, A, k, S, seed);
}

template <typename T> 
void gen_s_mat(
        int64_t& m,
        int64_t& n,
        std::vector<T>& A,
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
        diag<T>(k, k, s, k, S);
        gen_mat<T>(m, n, A, k, S, seed);
}

template <typename T> 
void gen_mat(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t k, // vector length
        const std::vector<T>& S,
        int32_t seed
) {   
        using namespace blas;
        using namespace lapack;
        
        std::vector<T> U(m * k, 0.0);
        std::vector<T> V(n * k, 0.0);
        std::vector<T> tau(k, 2.0);
        std::vector<T> Gemm_buf(m * k, 0.0);

        // Data pointer predeclarations for whatever is accessed more than once
        T* U_dat = U.data();
        T* V_dat = V.data();
        T* tau_dat = tau.data();
        T* Gemm_buf_dat = Gemm_buf.data();

        RandBLAS::dense_op::gen_rmat_norm<T>(m, k, U_dat, seed);
        RandBLAS::dense_op::gen_rmat_norm<T>(n, k, V_dat, ++seed);

        geqrf(m, k, U_dat, m, tau_dat);
        ungqr(m, k, k, U_dat, m, tau_dat);

        geqrf(n, k, V_dat, n, tau_dat);
        ungqr(n, k, k, V_dat, n, tau_dat);

        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U_dat, m, S.data(), k, 0.0, Gemm_buf_dat, m);
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, k, 1.0, Gemm_buf_dat, m, V_dat, n, 0.0, A.data(), m);
}

template void eye<float>(int64_t m, int64_t n, std::vector<float>& A );
template void eye<double>(int64_t m, int64_t n, std::vector<double>& A );

template void get_L<float>(int64_t m, int64_t n, std::vector<float>& L);
template void get_L<double>(int64_t m, int64_t n, std::vector<double>& L);

template void diag(int64_t m, int64_t n, const std::vector<float>& s, int64_t k, std::vector<float>& S);
template void diag(int64_t m, int64_t n, const std::vector<double>& s, int64_t k, std::vector<double>& S);

template void disp_diag(int64_t m, int64_t n, int64_t k, std::vector<float>& A);
template void disp_diag(int64_t m, int64_t n, int64_t k, std::vector<double>& A);

template void row_swap(int64_t m, int64_t n, std::vector<float>& A, const std::vector<int64_t>& p);
template void row_swap(int64_t m, int64_t n, std::vector<double>& A, const std::vector<int64_t>& p);

template void row_resize(int64_t m, int64_t n, std::vector<float>& A, int64_t k);
template void row_resize(int64_t m, int64_t n, std::vector<double>& A, int64_t k);

template void qb_resize(int64_t m, int64_t n, std::vector<float>& Q, std::vector<float>& B, int64_t& k, int64_t curr_sz); 
template void qb_resize(int64_t m, int64_t n, std::vector<double>& Q, std::vector<double>& B, int64_t& k, int64_t curr_sz); 

template void gen_mat_type(int64_t& m, int64_t& n, std::vector<float>& A, int64_t k, int32_t seed, std::tuple<int, float, bool> type);
template void gen_mat_type(int64_t& m, int64_t& n, std::vector<double>& A, int64_t k, int32_t seed, std::tuple<int, double, bool> type);

template void gen_poly_mat(int64_t& m, int64_t& n, std::vector<float>& A, int64_t k, float t, bool diagon, int32_t seed); 
template void gen_poly_mat(int64_t& m, int64_t& n, std::vector<double>& A, int64_t k, double t, bool diagon, int32_t seed);

template void gen_exp_mat(int64_t& m, int64_t& n, std::vector<float>& A, int64_t k, float t, int32_t seed); 
template void gen_exp_mat(int64_t& m, int64_t& n, std::vector<double>& A, int64_t k, double t, int32_t seed);

template void gen_s_mat(int64_t& m, int64_t& n, std::vector<float>& A, int64_t k, int32_t seed);
template void gen_s_mat(int64_t& m, int64_t& n, std::vector<double>& A, int64_t k, int32_t seed);

template void gen_mat(int64_t m, int64_t n, std::vector<float>& A, int64_t k, const std::vector<float>& S, int32_t seed); 
template void gen_mat(int64_t m, int64_t n, std::vector<double>& A, int64_t k, const std::vector<double>& S, int32_t seed);
} // end namespace util
