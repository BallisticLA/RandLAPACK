#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif
#include <lapack.hh>
// Issue : after including lapack, anything that is called I is considered something related to complex vals. gtest has some objects with such name
#include <typeinfo>

namespace RandLAPACK::comps::util {

template <typename T>
void eye(
        int64_t m,
        int64_t n,
        T* A
){
    // Generate an identity Q - kinda ugly, think of a better way
    //std::vector<T> I (size, 0.0);
    int64_t size = m * n;
    for (int i = 0; i < size; i += m)
    {
        //printf(typeid(I).name());
        //printf(I);
        A[i] = 1;
    }
}



template <typename T>
void householder_ref_gen(
        int64_t m,
        int64_t n,
        T* const A,
        T* Q 
)
{
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
}

/*
template <typename T> 
void get_L(
        bool col_maj,
        int64_t m,
        int64_t n,
        T* L
);
*/
} // end namespace RandLAPACK::comps::rs