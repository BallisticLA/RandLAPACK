/*
TODO #1: Test get_L.

TODO: Use laswap istead of pivot restoration
*/

#include <RandLAPACK/comps/util.hh>

#include <iostream>
#include <cmath>
#include <lapack.hh>
#include <RandBLAS.hh>

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
                std::for_each(&L_dat[i], &L_dat[i + j],
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
void swap_rows(
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
                j = p_dat[i] - 1;
                swap<T, T>(n, &A_dat[i], m, &A_dat[j], m);
        }
}

// "intellegent reisze"
template <typename T> 
T* upsize(
        int64_t target_sz,
        std::vector<T>& A
) {     
        if (A.size() < target_sz)
                A.resize(target_sz, 0);

        return A.data();
}


// Resulting array is to be k by n - THIS IS SIZING DOWN
template <typename T> 
T* row_resize(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t k
) {     
        using namespace blas;

        T* A_dat = A.data();

        // SIZING DOWN - just moving data
        if(m > k)
        {
                uint64_t end = k;

                for (int i = 1; i < n; ++i)
                {
                        // Place ith column (of k entries) after the (i - 1)st column
                        copy(k, &A_dat[m * i], 1, &A_dat[end], 1);
                        end += k;
                }
        }
        //SIZING UP
        else{
                // How many rows are being added: k - m
                A_dat = upsize(k * n, A);

                int64_t end = k * (n - 1);
                for(int i = n - 1; i > 0; --i)
                {
                        // Copy in reverse order to avoid overwriting
                        copy(m, &A_dat[m * i], -1, &A_dat[end], -1);
                        std::fill(&A_dat[m * i], &A_dat[end], 0.0);
                        end -= k;
                }
        }

        return A_dat;
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
        using namespace blas;
        T* A_dat = A.data();

        switch(std::get<0>(type)) 
        {
                /*
                First 3 cases are identical, varying ony in the entry generation function.
                is there a way to propagate the lambda expression or somehowe pass several parameners into a undary function in foreach?
                */
                case 0:
                        // Generating matrix with polynomially decaying singular values
                        printf("TEST MATRIX: POLYNOMIAL DECAY sigma_i = 1 / (i + 1)^pow (first k * 0.2 sigmas = 1)\n");
                        RandLAPACK::comps::util::gen_poly_mat<T>(m, n, A, k, std::get<1>(type), std::get<2>(type), seed); 
                        break;
                case 1:
                        // Generating matrix with exponentially decaying singular values
                        printf("TEST MATRIX: EXPONENTIAL DECAY sigma_i = e^((i + 1) * -pow) (first k * 0.2 sigmas = 1)\n");
                        RandLAPACK::comps::util::gen_exp_mat<T>(m, n, A, k, std::get<1>(type), std::get<2>(type), seed); 
                        break;
                case 2:
                        // Generating matrix with s-shaped singular values plot
                        printf("TEST MATRIX: S-SHAPED DECAY (first k * 0.2 sigmas = 1)\n");
                        RandLAPACK::comps::util::gen_s_mat<T>(m, n, A, k, std::get<2>(type), seed); 
                        break;
                case 3:
                        // A = [A A]
                        printf("TEST MATRIX: A = [A A]\n");
                        RandBLAS::dense_op::gen_rmat_norm<T>(m, k, A_dat, seed);
                        if (2 * k <= n)
                        {
                        copy(m * (n / 2), &A_dat[0], 1, &A_dat[(n / 2) * m], 1);
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
        int64_t k,
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
        int64_t k,
        T t, // controls the decay. The higher the value, the slower the decay
        bool diagon,
        int32_t seed
) {   
        using namespace lapack;

        std::vector<T> s(k, 0.0);
        std::vector<T> S(k * k, 0.0);
        
        T cnt = 0.0;
        // apply lambda function to every entry of s
        std::for_each(s.begin(), s.end(),
                // Lambda expression begins
                [&t, &cnt](T& entry)
                {
                        entry = (std::exp(++cnt * -t));
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
void gen_s_mat(
        int64_t& m,
        int64_t& n,
        std::vector<T>& A,
        int64_t k,
        bool diagon,
        int32_t seed
) {   
        using namespace lapack;

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
void gen_mat(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t k,
        std::vector<T>& S,
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

        copy(m * k, U_dat, 1, Gemm_buf_dat, 1);
        for(int i = 0; i < k; ++i)
        {
                scal(m, S[i + k * i], &Gemm_buf_dat[i * m], 1);
        }
        
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, k, 1.0, Gemm_buf_dat, m, V_dat, n, 0.0, A.data(), m);
}

template <typename T> 
T cond_num_check(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        std::vector<T>& A_cpy,
        std::vector<T>& s,
        bool verbosity
) {   
        using namespace lapack;
        
        // Copy to avoid any changes
        T* A_cpy_dat = upsize(m * n, A_cpy);
        T* s_dat = upsize(n, s);

        lacpy(MatrixType::General, m, n, A.data(), m, A_cpy_dat, m);
        gesdd(Job::NoVec, m, n, A_cpy_dat, m, s_dat, NULL, m, NULL, n);
        T cond_num = s_dat[0] / s_dat[n - 1];

        if (verbosity)
                printf("CONDITION NUMBER: %f\n", cond_num);

        return cond_num;
}

// Bool=1 indicates failure
template <typename T> 
bool orthogonality_check(
        int64_t m,
        int64_t n,
        int64_t k,
        const std::vector<T>& A,
        std::vector<T>& A_gram,
        bool verbosity
) {
        using namespace lapack;

        const T* A_dat = A.data();
        T* A_gram_dat = A_gram.data();
        
        gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A_dat, m, A_dat, m, 0.0, A_gram_dat, n);
        for (int oi = 0; oi < k; ++oi) 
        {
                A_gram_dat[oi * n + oi] -= 1.0;
        }
        T orth_err = lange(Norm::Fro, n, n, A_gram_dat, k);
    
        if(verbosity)
        {
                printf("Q ERROR:   %e\n\n", orth_err);
        }

        if (orth_err > 1.0e-10)
                return true;

        return false;
}

template void eye<float>(int64_t m, int64_t n, std::vector<float>& A );
template void eye<double>(int64_t m, int64_t n, std::vector<double>& A );

template void get_L<float>(int64_t m, int64_t n, std::vector<float>& L);
template void get_L<double>(int64_t m, int64_t n, std::vector<double>& L);

template void diag(int64_t m, int64_t n, const std::vector<float>& s, int64_t k, std::vector<float>& S);
template void diag(int64_t m, int64_t n, const std::vector<double>& s, int64_t k, std::vector<double>& S);

template void disp_diag(int64_t m, int64_t n, int64_t k, std::vector<float>& A);
template void disp_diag(int64_t m, int64_t n, int64_t k, std::vector<double>& A);

template void swap_rows(int64_t m, int64_t n, std::vector<float>& A, const std::vector<int64_t>& p);
template void swap_rows(int64_t m, int64_t n, std::vector<double>& A, const std::vector<int64_t>& p);

template float* upsize(int64_t target_sz, std::vector<float>& A);
template double* upsize(int64_t target_sz, std::vector<double>& A);

template float* row_resize(int64_t m, int64_t n, std::vector<float>& A, int64_t k);
template double* row_resize(int64_t m, int64_t n, std::vector<double>& A, int64_t k);

template void gen_mat_type(int64_t& m, int64_t& n, std::vector<float>& A, int64_t k, int32_t seed, std::tuple<int, float, bool> type);
template void gen_mat_type(int64_t& m, int64_t& n, std::vector<double>& A, int64_t k, int32_t seed, std::tuple<int, double, bool> type);

template void gen_poly_mat(int64_t& m, int64_t& n, std::vector<float>& A, int64_t k, float t, bool diagon, int32_t seed); 
template void gen_poly_mat(int64_t& m, int64_t& n, std::vector<double>& A, int64_t k, double t, bool diagon, int32_t seed);

template void gen_exp_mat(int64_t& m, int64_t& n, std::vector<float>& A, int64_t k, float t, bool diagon, int32_t seed); 
template void gen_exp_mat(int64_t& m, int64_t& n, std::vector<double>& A, int64_t k, double t, bool diagon, int32_t seed);

template void gen_s_mat(int64_t& m, int64_t& n, std::vector<float>& A, int64_t k, bool diagon, int32_t seed);
template void gen_s_mat(int64_t& m, int64_t& n, std::vector<double>& A, int64_t k, bool diagon, int32_t seed);

template void gen_mat(int64_t m, int64_t n, std::vector<float>& A, int64_t k, std::vector<float>& S, int32_t seed); 
template void gen_mat(int64_t m, int64_t n, std::vector<double>& A, int64_t k, std::vector<double>& S, int32_t seed);

template float cond_num_check(int64_t m, int64_t n, const std::vector<float>& A, std::vector<float>& A_cpy, std::vector<float>& s, bool verbosity);
template double cond_num_check(int64_t m, int64_t n, const std::vector<double>& A, std::vector<double>& A_cpy, std::vector<double>& s, bool verbosity);

template bool orthogonality_check(int64_t m, int64_t n, int64_t k, const std::vector<float>& A, std::vector<float>& A_gram, bool verbosity);
template bool orthogonality_check(int64_t m, int64_t n, int64_t k, const std::vector<double>& A, std::vector<double>& A_gram, bool verbosity);
} // end namespace util
