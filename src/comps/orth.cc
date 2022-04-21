#include <RandLAPACK/comps/orth.hh>

#include <iostream>
#include <cmath>
#include <lapack.hh>
#include <RandBLAS.hh>

namespace RandLAPACK::comps::orth {

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
                        r[j - 2] = r[j - 2] - dot(j - 2, work.data(), 1, work.data(), 1);
                        //

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


                        // Fill R - no copy repetition
                        std::copy(r.data() + j - 2, r.data() + j - 1, R.data() + n * (j - 2));
                        std::copy(r.data() + j - 1, r.data() + 2 * (j - 1), R.data() + n * (j - 1));
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


//template void orth_dcgs2<float>(int64_t m, int64_t n, float* const A, float* Q);
template void orth_dcgs2<double>(int64_t m, int64_t n, double* const A, double* Q);

} // end namespace orth