#include <RandLAPACK/comps/util.hh>

#include <iostream>
#include <cmath>
#include <lapack.hh>
namespace RandLAPACK::comps::util {





/*

// Generate Identity
// Assuming col-maj
template <typename T>
void eye(
        int64_t m,
        int64_t n,
        T* I 
){
    // Generate an identity Q - kinda ugly, think of a better way
    //std::vector<T> I (size, 0.0);
    int64_t size = m * n;
    for (int i = 0; i < size; i += m)
    {
        //I[i] = 1;
    }
}

// Householder reflector-based orthogonalization
// Assuming column-major storage
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
        for (int i = m; i < size; i += m)
        {
                // Grab a column of an input matrix
                std::vector<T> col(&A[i - m], &A[i]); 

                // Get an l-2 norm of a vector
                T norm = nrm2(m, col.data(), 1);
                T first = col[1];

                if(first >= 0) {
                        first += 1;
                }
                else {
                        first -= 1;
                }
                // Scale the vector by this
                T alpha = 1 / (norm * sqrt(abs(first)));

                // Using axpy (vector sum) only to perform const * vec - what's a better way to do it?
                // Dummy zero vector
                std::vector<T> buf(m);
                axpy<T>(m, alpha, col.data(), 1, buf.data(), 1);
                // Householder reflection constant
                T tau = 1; // or 2?

                larf(Side::Right,  m, n, col.data(), 1, tau, Q, 1);	
        }
}

// Helper routine for retrieving the proper L factor of LU decomposition.
/*
Concern - not sure how the row major vs col major ordering works here & how matrices are stored.
*/
/*
template <typename T> 
void get_L(
		bool col_maj,
		int64_t m,
        int64_t n,
        T* L
) {
	// Grab the reference 
	std::vector<T>& ref_L = *L;
	int64_t size = m * n;

	if (col_maj) {
		// Case if matrices are stored by columns
		ref_L[0] = 1;
		for(int i = m, j = 0; i < size && j < m; i += m, ++j) {
			typename std::vector<T>::const_iterator first = i;
			typename std::vector<T>::const_iterator last = i + j;
			ref_L.erase(first, last);
			ref_L[i + 1] = 1;
		}
	}
	else {
		// This should ve fine if matrices are stored by rows (row1 followed by row2, etc.) 
		for (int i = n, j = 1; i < size && j < n; i += n, ++j) {
			typename std::vector<T>::const_iterator first = i - n + j;
			typename std::vector<T>::const_iterator last = i;
			ref_L.erase(first, last);
			// The unit diagonal elements of L are not stored.
			ref_L[i] = 1;
		}
	}
}
*/

} // end namespace util