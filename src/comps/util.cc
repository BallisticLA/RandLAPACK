#include <RandLAPACK/comps/util.hh>

#include <iostream>
#include <cmath>
#include <lapack.hh>
#include <RandBLAS.hh>

/*
UTILITY ROUTINES
QUESTION: some of these very well can be separate namespace for their degree of seriousness.
However, those routines are not necessarily randomized. What do we do with them?

TODO: (maybe) substitute std copies for lapack copy functions.
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
    // Generate an identity A - kinda ugly, think of a better way
    for (int j = 0; j < n; ++ j)
        {
                A[(m * j) + j] = 1;
        }
}


// Helper routine for retrieving the lower triangular portion of a matrix
// Puts 1's on the main diagonal
template <typename T> 
void get_L(
        bool col_maj,
        int64_t m,
        int64_t n,
        T* L // pointer to the beginning
) {
	// Vector end pointer
	int size = m * n;
        // Buffer zero vector
        std::vector<T> z_buf(m, 0.0);
        T* z_begin = z_buf.data();

        // The unit diagonal elements of L were not stored.
        L[0] = 1;
    
        if (col_maj) {
                for(int i = m, j = 1; i < size && j < m; i += m, ++j) 
                {
                        // Copy zeros into elements above the diagonal
                        std::copy(z_begin, z_begin + j, L + i);
                        // The unit diagonal elements of L were not stored.
                        L[i + j] = 1;
                }
	}
	else {
		// This should be fine if matrices are stored by rows (row1 followed by row2, etc.) 
		for (int  i = n, j = 1; i < size && j < n; i += n, ++j) 
                {
                        // Copy zeros into elements above the diagonal
			std::copy(z_begin, z_begin + j, L + i);
			// The unit diagonal elements of L were not stored.
			L[i + j] = 1;
		}
	}
}

// Helper routine for retrieving the upper triangular portion of a matrix
// Maintains the diagonal entries
template <typename T> 
void get_U(
        bool col_maj,
        int64_t m,
        int64_t n,
        T* U // pointer to the beginning
) {
	// Vector end pointer
	int size = m * n;
        // Buffer zero vector
        std::vector<T> z_buf(m, 0.0);
        T* z_begin = z_buf.data();
    
        if (col_maj) {
                for(int i = 1, j = (m - 1); i < (size - m) && j > 0; i += (m + 1), --j) 
                {
                        // Copy zeros into elements above the diagonal
                        std::copy(z_begin, z_begin + j, U + i);
                        // The unit diagonal elements of L were not stored.
                }
	}
	else {
		// This should be fine if matrices are stored by rows (row1 followed by row2, etc.) 
		for(int i = 1, j = (n - 1); i < (size - n) && j > 0; i += (n + 1), --j) 
                {
                        // Copy zeros into elements above the diagonal
                        std::copy(z_begin, z_begin + j, U + i);
                        // The unit diagonal elements of L were not stored.
                }
	}
}

// Scale the diagonal of a matrix by some constant factor
template <typename T> 
void scale_diag(
        int64_t m,
        int64_t n,
        T* U, // pointer to the beginning
        T c //scaling factor 
) {
	for (int i = 0; i < m; ++i)
        {
                for(int j = 0; j < n; ++j)
                {
                        U[i + j] = c * U [i + j];
                }
        }
}

// Given an upper triangular matrix, produces a symmetric one
template <typename T> 
void get_sym(
        bool col_maj,
        int64_t m,
        int64_t n,
        T* U // pointer to the beginning
) {
	// Vector end pointer
	int size = m * n;
        // Buffer zero vector
    
        if (col_maj) {
                for(int i = m, j = 1; i < size && j < m; i += m, ++j) 
                {
                        for(int k = 0; k < j; ++k)
                        {
                                U[(m * (k)) + j] = U[i + k];
                        }
                }
	}
	else {
                // TODO
	}
}


// Perfoms a Cholesky QR factorization
template <typename T> 
void chol_QR(
        int64_t m,
        int64_t k,
        T* Q // pointer to the beginning
) {
        using namespace blas;
        using namespace lapack;

        std::vector<T> Q_buf(k * k, 0.0);
        // Find normal equation Q'Q - Just the upper triangular portion
        syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q, m, 1.0, Q_buf.data(), k);

        // Positive definite cholesky factorization
        potrf(Uplo::Upper, k, Q_buf.data(), k);

        // Inverse of an upper-triangular matrix
        trtri(Uplo::Upper, Diag::NonUnit, k, Q_buf.data(), k);
        // Q = Q * R^(-1)
        std::vector<T> Q_chol(m * k, 0.0);
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, Q, m, Q_buf.data(), k, 0.0, Q_chol.data(), m);

        // Copy the result into Q
        lacpy(MatrixType::General, m, k, Q_chol.data(), m, Q, m);
}

// Diagonalization - turns a vector into a diagonal matrix
template <typename T> 
void diag(
        int64_t m,
        int64_t n,
        T* s, // pointer to the beginning
        T* S
) {     
        int64_t size = m * n;
        for (int i = 0, j = 0; i < size && j < n; i += m, ++j)
        {
                S[i + j] = s[j];
        }
}

// Addressing Pivoting
template <typename T> 
void pivot_swap(
        int64_t m,
        int64_t n,
        T* A, // pointer to the beginning
        int64_t* p // Pivot vector
) {     
        using namespace blas;
        using namespace lapack;

        std::vector<T> P(m * m, 0.0);
        std::vector<T> A_cpy(m * n, 0.0);
        std::vector<T> col_buf(m, 0.0);

        for (int j = 0; j < m; ++ j)
        {
                P[(m * j) + j] = 1;
        }

        for (int i = 0, j = 0; i < n; ++i)
        {
                j = *(p + i) - 1;
                if (j != 0)
                {  
                        // Swap rows
                        // Store ith column into the buffer
                        copy(m, P.data() + (m * i), 1, col_buf.data(), 1);

                        // copy jth column into the ith position
                        copy(m, P.data() + (m * j), 1, P.data() + (m * i), 1);

                        // copy the column from the buffer into jth position
                        copy(m, col_buf.data(), 1, P.data() + (m * j), 1);
                }
        }

        lacpy(MatrixType::General, m, n, A, m, A_cpy.data(), m);
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m, 1.0, P.data(), m, A_cpy.data(), m, 0.0, A, m);
}

// GENERATING MATRICES OF VARYING SPECTRAL DECAY
template <typename T> 
void gen_exp_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // vector length
        int64_t t, // controls the decay. The higher the value, the slower the decay
        int64_t seed
) {   
        std::vector<T> s(k, 0.0);
        std::vector<T> S(k * k, 0.0);
        
        // apply lambda function to every entry of s
        std::for_each(s.begin(), s.end(),
                // Lambda expression begins
                [&t](T& entry)
                {
                        static T cnt = 0.0;
                        entry = std::exp(++cnt / -t);
                }
        );

        // form a diagonal S
        diag<T>(m, n, s.data(), S.data());
        gen_mat<T>(m, n, A, k, S.data(), seed);

}

template <typename T> 
void gen_s_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // vector length
        int64_t seed
) {   
        std::vector<T> s(k, 0.0);
        std::vector<T> S(k * k, 0.0);

        // apply lambda function to every entry of s
        std::for_each(s.begin(), s.end(),
                // Lambda expression begins
                [](T& entry)
                {
                        static T cnt = 0.0;
                        entry = 0.0001 + (1 / (1 + std::exp(++cnt - 30)));
                }
        );

        // form a diagonal S
        diag<T>(m, n, s.data(), S.data());
        gen_mat<T>(m, n, A, k, S.data(), seed);
}

template <typename T> 
void gen_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // vector length
        T* S,
        int64_t seed
) {   
        using namespace blas;
        using namespace lapack;

        std::vector<T> V(m * k, 0.0);
        std::vector<T> U(n * k, 0.0);
        std::vector<T> tau(k, 2.0);
        std::vector<T> Gemm_buf(m * k, 0.0);
        RandBLAS::dense_op::gen_rmat_norm<T>(m, k, U.data(), seed);
        RandBLAS::dense_op::gen_rmat_norm<T>(n, k, V.data(), ++seed);

        geqrf(m, k, U.data(), m, tau.data());
        ungqr(m, k, k, U.data(), m, tau.data());

        geqrf(m, k, V.data(), m, tau.data());
        ungqr(n, k, k, V.data(), n, tau.data());

        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U.data(), m, S, k, 0.0, Gemm_buf.data(), m);
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, k, 1.0, Gemm_buf.data(), m, V.data(), k, 0.0, A, m);
}

// Explicit instantiation of template functions - workaround to avoid header implementations
template void eye<float>(int64_t m, int64_t n, float* A );
template void eye<double>(int64_t m, int64_t n, double* A );

template void get_L<float>(bool col_maj, int64_t m, int64_t n, float* L);
template void get_L<double>(bool col_maj, int64_t m, int64_t n, double* L);

template void get_U<float>(bool col_maj, int64_t m, int64_t n, float* U);
template void get_U<double>(bool col_maj, int64_t m, int64_t n, double* U);

template void scale_diag(int64_t m, int64_t n, float* U, float c);
template void scale_diag(int64_t m, int64_t n, double* U, double c);

template void get_sym(bool col_maj, int64_t m, int64_t n, float* U);
template void get_sym(bool col_maj, int64_t m, int64_t n, double* U);

template void chol_QR(int64_t m, int64_t k, float* Q);
template void chol_QR(int64_t m, int64_t k, double* Q);

template void diag(int64_t m, int64_t n, float* s, float* S);
template void diag(int64_t m, int64_t n, double* s, double* S);

template void pivot_swap( int64_t m, int64_t n, float* A, int64_t* p);
template void pivot_swap( int64_t m, int64_t n, double* A, int64_t* p);

template void gen_s_mat(int64_t m, int64_t n, float* A, int64_t k, int64_t seed);
template void gen_s_mat(int64_t m, int64_t n, double* A, int64_t k, int64_t seed);

template void gen_exp_mat(int64_t m, int64_t n, float* A, int64_t k, int64_t t, int64_t seed); 
template void gen_exp_mat(int64_t m, int64_t n, double* A, int64_t k, int64_t t, int64_t seed);

template void gen_mat(int64_t m, int64_t n, float* A, int64_t k, float* S, int64_t seed); 
template void gen_mat(int64_t m, int64_t n, double* A, int64_t k, double* S, int64_t seed);
} // end namespace util
