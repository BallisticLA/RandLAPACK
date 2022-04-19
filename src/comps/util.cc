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



/*
// assuming column - major storage
template <typename T>
static void orth_dcgs2_main(
        int64_t m,
        int64_t n,
        T* Q,
        T* R 
)
{
        using namespace blas;

        std::vector<T> q(m * 2, 0.0);
        std::vector<T> r((n - 1) * 2, 0.0);

        // Buffer array
        std::vector<T> q_1(m, 0.0); 

        if (n == 2)
        {
                //  r(1,1) = Q(:,1)' * Q(:,1);
                r[0] = dot(m, Q, 1, Q, 1);	
                // r(1,2) = Q(:,1)' * Q(:,2);
                r[n] = dot(m, Q, 1, Q + m, 1);
                printf("%f\n", r[n]);
                r[0] = sqrt(r[0]);
                r[n] = r[n] / r[0];

                // Using axpy (vector sum) only to perform const * vec - what's a better way to do it?
                // q(:,1) = Q(:,1) / r(1,1);
                axpy<T>(m, 1.0 / r[0], Q, 1, q_1.data(), 1);
                // Copy over to Q from a buffer
                std::copy(q_1.data(), q_1.data() + m, Q);

                // q(:,2) = Q(:,2) - q(:,1) * r(1,2);
                axpy<T>(m,  -r[m], q_1.data(), 1, Q + m, 1);


                char label1[] = "q";
                RandBLAS::util::print_colmaj<T>( m, n, Q, label1);

                char label2[] = "r";
                RandBLAS::util::print_colmaj<T>( n, 2, r.data(), label2);
        }
        else if (n >= 3)
        {
                //tmp(1:j-1,1:2) = Q(:,1:j-1)' * Q(:,j-1:j);
                std::vector<T> tmp((n - 1) * 2, 0.0);
                gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n - 1, 2, m, 1.0, Q, m, Q, m, 0.0, tmp.data(), n - 1);

                //work(1:j-2,1) = tmp(1:j-2,1);
                std::vector<T> work(n - 2, 0.0);
                std::copy(tmp.data(), tmp.data() + n - 2, work.data());

                //r(1:j-1,2)    = tmp(1:j-1,2);
                std::copy(tmp.data() + n - 1, tmp.data() + 2 *(n - 2), r.data() + m);

                //r(j-1,1)      = tmp(j-1,1);
                r[n - 1] = tmp[n - 1];

                //r(j-1,2) = r(j-1,2) - work(1:j-2,1)' * r(1:j-2,2);
                r[2 * (n - 1)] = r[2 * (n - 1)] - dot(n - 2, work.data(), 1, r.data() + n - 2, 1);

                //r(j-1,1) = r(j-1,1) - work(1:j-2,1)' * work(1:j-2,1); 
                r[n - 1] = r[n - 1] - dot(n - 2, work.data(), 1, work.data() + n - 2, 1);

                //r(1:j-2,1) = R(1:j-2,1) + work(1:j-2,1);
                axpy<T>(n - 2, 1.0, work.data(), 1, R, 1);

                //r(j-1,1) = sqrt( r(j-1,1) );
                R[n - 1] = sqrt(R[n - 1]);

                //r(j-1,2) = r(j-1,2) / r(j-1,1);
                R[2 * (n - 1)] = R[2 * (n - 1)] / R[n - 1];

                //Q(:,j-1) = Q(:,j-1) - Q(:,1:j-2) * work(1:j-2,1);
                std::vector<T> buf(m, 0.0);
                gemv<T>(Layout::ColMajor, Op::NoTrans, m, n - 2, 1.0, Q, m, work.data(), 1, 1.0, buf.data(), 1);	
                axpy<T>(m,  1.0, buf.data(), 1, Q + m * (n - 1), 1);

                //Q(:,j) = Q(:,j) - Q(:,1:j-2) * r(1:j-2,2);
                std::vector<T> buf_1(m, 0.0);
                gemv<T>(Layout::ColMajor, Op::NoTrans, m, n - 2, 1.0, Q, m, r.data() + n - 1, 1, 1.0, buf_1.data(), 1);	
                axpy<T>(m,  1.0, buf_1.data(), 1, Q + m * n, 1);
        
                //q(:,1) = Q(:,j-1) / r(j-1,1);
                axpy<T>(m, 1.0 / r[n - 1], Q + m * (n - 1), 1, q_1.data(), 1);
                std::copy(q_1.data(), q_1.data() + m, Q + m * (n - 1));

                //q(:,2) = Q(:,j) - q(:,1) * r(j-1,2);
                axpy<T>(m,  -r[2 * (n - 1)], q_1.data(), 1, Q + m, 1);
        }
}
*/

// Advanced orthogonalization algorithm
// Prototype does not work for m < n
template <typename T>
void orth_dcgs2(
        int64_t m,
        int64_t n,
        T* const A,
        T* Q 
)
{
        using namespace blas;
        std::vector<T> R(n * n, 0.0);
        std::copy(A, A + m * n, Q);
        
        for (int j = 2; j <= n; ++j)
        {
                std::vector<T> q(m * 2, 0.0);

                // SWITCH TO J HERE _ CAREFUL
                std::vector<T> r((j - 1) * 2, 0.0);

                // Buffer array
                std::vector<T> q_1(m, 0.0); 

                if (j == 2)
                {
                        //  r(1,1) = Q(:,1)' * Q(:,1);
                        R[0] = dot(m, Q, 1, Q, 1);	
                        // r(1,2) = Q(:,1)' * Q(:,2);
                        R[n] = dot(m, Q, 1, Q + m, 1);
                        R[0] = sqrt(R[0]);
                        R[n] = R[n] / R[0];

                        // Using axpy (vector sum) only to perform const * vec - what's a better way to do it?
                        // q(:,1) = Q(:,1) / r(1,1);
                        axpy<T>(m, 1.0 / R[0], Q, 1, q_1.data(), 1);
                        // Copy over to Q from a buffer
                        std::copy(q_1.data(), q_1.data() + m, Q);

                        // q(:,2) = Q(:,2) - q(:,1) * r(1,2);
                        axpy<T>(m, -R[n], q_1.data(), 1, Q + m, 1);
                }
                if (j >= 3)
                {       
                        //tmp(1:j-1,1:2) = Q(:,1:j-1)' * Q(:,j-1:j);
                        std::vector<T> tmp((n - 1) * 2, 0.0);
                        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, j - 1, 2, m, 1.0, Q, m, Q + m * (j - 2), m, 0.0, tmp.data(), j - 1);
                
                        //work(1:j-2,1) = tmp(1:j-2,1);
                        std::vector<T> work(j - 2, 0.0);
                        std::copy(tmp.data(), tmp.data() + j - 2, work.data());

                        //r(1:j-1,2) = tmp(1:j-1,2);
                        std::copy(tmp.data() + j - 1, tmp.data() + 2 * (j - 1), r.data() + j - 1);

                        //r(j-1,1)  = tmp(j-1,1);
                        r[j - 2] = tmp[j - 2];
                        
                        //r(j-1,2) = r(j-1,2) - work(1:j-2,1)' * r(1:j-2,2);
                        r[2 * (j - 1) - 1] = r[2 * (j - 1) - 1] - dot(j - 2, work.data(), 1, r.data() + j - 2, 1);

                        //r(j-1,1) = r(j-1,1) - work(1:j-2,1)' * work(1:j-2,1); 
                        r[j - 1] = r[j - 1] - dot(j - 2, work.data(), 1, work.data() + j - 2, 1);

                        // Very questionable moment here
                        //r(1:j-2,1) = R(1:j-2,1) + work(1:j-2,1);
                        // BUT ALSO R = R(1:j-2,j-1)
                        std::vector<T> buffer(j - 2, 0.0);
                        std::copy(R.data() + n * (j - 2), R.data() + (n + 1) * (j - 2), buffer.data());
                        axpy<T>(j - 2, 1.0, work.data(), 1, buffer.data(), 1);
                        std::copy(buffer.data(), buffer.data() + j - 2, r.data());
                        
                        //r(j-1,1) = sqrt( r(j-1,1) );
                        r[j - 2] = sqrt(r[j - 2]);
                
                        //r(j-1,2) = r(j-1,2) / r(j-1,1);
                        r[2 * (j - 1) - 1] = r[2 * (j - 1) - 1] / r[j - 2];
                        
                        //Q(:,j-1) = Q(:,j-1) - Q(:,1:j-2) * work(1:j-2,1);
                        std::vector<T> buf(m, 0.0);
                        gemv<T>(Layout::ColMajor, Op::NoTrans, m, j - 2, 1.0, Q, m, work.data(), 1, 1.0, buf.data(), 1);	
                        axpy<T>(m,  1.0, buf.data(), 1, Q + m * (j - 1), 1);
                
                        //Q(:,j) = Q(:,j) - Q(:,1:j-2) * r(1:j-2,2);
                        std::vector<T> buf_1(m, 0.0);
                        gemv<T>(Layout::ColMajor, Op::NoTrans, m, j - 2, 1.0, Q, m, r.data() + j - 1, 1, 1.0, buf_1.data(), 1);	
                        axpy<T>(m,  -1.0, buf_1.data(), 1, Q + m * (j - 1), 1);
                
                        //q(:,1) = Q(:,j-1) / r(j-1,1);
                        axpy<T>(m, 1.0 / r[j - 2], Q + m * (j - 2), 1, q_1.data(), 1);
                        std::copy(q_1.data(), q_1.data() + m, Q + m * (j - 2));

                        //q(:,2) = Q(:,j) - q(:,1) * r(j-1,2);
                        std::vector<T> aaaaa(m, 0.0);
                        axpy<T>(m,  -r[2 * (j - 1) - 1], q_1.data(), 1, Q + m * (j - 1), 1);    
                }
        } 
        // "Cleanup stage"
        //r(1:n,1) = Q(:,1:n)' * Q(:,n); 
        std::vector<T> buf(n, 0.0);
        gemv<T>(Layout::ColMajor, Op::Trans, m, n, 1.0, Q, m, Q + m * (n - 1), 1, 0.0, buf.data(), 1);
        
        //r(n,1) = r(n,1) - r(1:n-1,1)' * r(1:n-1,1);
        buf[n - 1] =  buf[n - 1] - dot(n - 1, buf.data(), 1, buf.data(), 1);
        //r(n,1) = sqrt( r(n,1) );  
        buf[n - 1] = sqrt(buf[n - 1]);

        //Q(:,n) = Q(:,n) - Q(:,1:n-1) * r(1:n-1,1);  
        std::vector<T> buf_1(m, 0.0);
        gemv<T>(Layout::ColMajor, Op::NoTrans, m, n - 1, 1.0, Q, m, buf.data(), 1, 0.0, buf_1.data(), 1);
        axpy<T>(m, -1.0, buf_1.data(), 1, Q + m * (n - 1), 1);

        // But also, R = R(1:n-1,n)
        //r(1:n-1,1) = R(1:n-1,1) + r(1:n-1,1);  
        axpy<T>(n - 1, 1.0, R.data() + (n - 1) * n, 1, buf.data(), 1);

        //q(:,1) = Q(:,n) / r(n,1);
        std::vector<T> buf_2(m, 0.0);
        axpy<T>(m, 1.0 / buf[n - 1], Q + m * (n - 1), 1, buf_2.data(), 1);
        std::copy(buf_2.data(), buf_2.data() + m, Q + m * (n - 1));
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

//template void orth_dcgs2<float>(int64_t m, int64_t n, float* const A, float* Q);
template void orth_dcgs2<double>(int64_t m, int64_t n, double* const A, double* Q);

} // end namespace util
