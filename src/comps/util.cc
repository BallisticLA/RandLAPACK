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
    // Generate an identity A - kinda ugly, think of a better way
    //std::vector<T> I (size, 0.0);
    int64_t size = m * n;
    for (int i = 0, j = 0; i < size && j < m; i += m, ++j)
    {
        A[i + j] = 1;
    }
}


// Householder reflector-based orthogonalization
// Assuming column-major storage - need row-major case
// Not sure how non-square cases should work here
template <typename T>
void householder_ref_gen(
        int64_t m,
        int64_t n,
        T* const A,
        T* Q 
)
{
        using namespace blas;
        using namespace lapack;
        
        int size = m * n;
        eye<T>(m, n, Q);

        // Grab columns of input matrix, get reflector vector
        for(int i = m, j = 0; i <= size && j < m; i += m, ++j) 
        {
                std::vector<T> col(m, 0.0);
                std::vector<T> buf_1(m, 0.0);
                std::vector<T> buf_2(m, 0.0);

                // Grab a column of an input matrix
                std::copy(A + (i - m) + j, A + i, col.data() + j); 

                // Get an l-2 norm of a vector
                T norm = nrm2(m, col.data(), 1);

                // Using axpy (vector sum) only to perform const * vec - what's a better way to do it?
                axpy<T>(m, 1.0 / norm, col.data(), 1, buf_1.data(), 1);

                T* first = &buf_1[j];
                if(*first >= 0) {
                        *first += 1;
                }
                else {
                        *first -= 1;
                }
                // Scale the vector by this
                T alpha = 1 / sqrt(abs(*first));

                // Using axpy (vector sum) only to perform const * vec - what's a better way to do it?
                axpy<T>(m, alpha, buf_1.data(), 1, buf_2.data(), 1);
                // Householder reflection constant
                T tau = 1.0; // or 2?

                // Q * (I - tau * v * v')
                larf(Side::Right, m, n, buf_2.data(), 1, tau, Q, m);
        }
}

// Helper routine for retrieving the proper L factor of LU decomposition.
/*
Concern - not sure how the row major vs col major ordering works here & how matrices are stored.
*/
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
                for(int i = m, j = 0; i < size && j < m; i += m, ++j) 
                {
                        // Copy zeros into elements above the diagonal
                        std::copy(z_begin, z_begin + j, L + i);
                        // The unit diagonal elements of L were not stored.
                        L[i + 1 + j] = 1;
                }
	}
	else {
		// This should be fine if matrices are stored by rows (row1 followed by row2, etc.) 
		for (int  i = n, j = 0; i < size && j < n; i += n, ++j) 
                {
                        // Copy zeros into elements above the diagonal
			std::copy(z_begin, z_begin + j, L + i);
			// The unit diagonal elements of L were not stored.
			L[i + 1 + j] = 1;
		}
	}
}

// Explicit instantiation of template functions - workaround to avoid header implementations
template void eye<float>(int64_t m, int64_t n, float* A );
template void eye<double>(int64_t m, int64_t n, double* A );

template void householder_ref_gen<float>(int64_t m, int64_t n, float* const A, float* Q );
template void householder_ref_gen<double>(int64_t m, int64_t n, double* const A, double* Q );

template void get_L<float>(bool col_maj, int64_t m, int64_t n, float* L);
template void get_L<double>(bool col_maj, int64_t m, int64_t n, double* L);
} // end namespace util
