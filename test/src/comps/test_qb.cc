#include <gtest/gtest.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestQB : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void test_QB2(int64_t m, int64_t n, int64_t k, int64_t block_sz, T tol, int64_t mat_type, uint32_t seed) {

        using namespace blas;
        using namespace lapack;
        
        int64_t size = m * n;

        // For running QB
        std::vector<T> A(size, 0.0);

        std::vector<T> Q(m * k, 0.0);
        std::vector<T> B(k * n, 0.0);

        // For results comparison
        std::vector<T> A_hat(size, 0.0);
        std::vector<T> A_k(size, 0.0);
        std::vector<T> A_cpy (m * n, 0.0);

        // For low-rank SVD
        std::vector<T> s(n, 0.0);
        std::vector<T> S(n * n, 0.0);
        std::vector<T> U(m * n, 0.0);
        std::vector<T> VT(n * n, 0.0);

        T* A_dat = A.data();
        T* A_hat_dat = A_hat.data();
        T* A_k_dat = A_k.data();
        T* A_cpy_dat = A_cpy.data();
        T* Q_dat = Q.data();
        T* B_dat = B.data();
        T* U_dat = U.data();
        T* s_dat = s.data();
        T* S_dat = S.data();
        T* VT_dat = VT.data();

        switch(mat_type) 
        {
            case 1:
                // Generating matrix with exponentially decaying singular values
                RandLAPACK::comps::util::gen_exp_mat<T>(m, n, A, k, 0.5, seed); 
                break;
            case 2:
                // Generating matrix with s-shaped singular values plot
                RandLAPACK::comps::util::gen_s_mat<T>(m, n, A, k, seed); 
                break;
            case 3:
                // Full-rank random A
                RandBLAS::dense_op::gen_rmat_norm<T>(m, k, A_dat, seed);
                if (2 * k <= n)
                {
                    // Add entries without increasing the rank
                    std::copy(A_dat, A_dat + (n / 2) * m, A_dat + (n / 2) * m);
                }
                break;
        }
        
        //char name1[] = "A";
        //RandBLAS::util::print_colmaj(m, n, A.data(), name1);

        // Create a copy of the original matrix
        std::copy(A_dat, A_dat + size, A_cpy_dat);

        RandLAPACK::comps::qb::qb2_safe<T>(
        m,
        n,
        A,
        k, // Here, serves as a backup termination criteria
        block_sz,
        tol,
        5,
        1,
        Q, // m by k
        B, // k by n
        seed
        );
        
        //char name_3[] = "QB1  output Q";
        //RandBLAS::util::print_colmaj(m, k, Q.data(), name_3);

        //char name_4[] = "QB1  output B";
        //RandBLAS::util::print_colmaj(k, n, B.data(), name_4);

        // A_hat = Q * B
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, B_dat, k, 0.0, A_hat_dat, m);
        // A = A - Q * B
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, -1.0, Q_dat, m, B_dat, k, 1.0, A_dat, m);

        // Get low-rank SVD
        gesdd(Job::SomeVec, m, n, A_cpy_dat, m, s_dat, U_dat, m, VT_dat, n);
        // buffer zero vector
        std::vector<T> z_buf(n, 0.0);
        T* z_buf_dat = z_buf.data();
        // zero out the trailing singular values
        std::copy(z_buf_dat, z_buf_dat + (n - k), s_dat + k);
        RandLAPACK::comps::util::diag(n, n, s, S);

        //char name_u[] = "U";
        //RandBLAS::util::print_colmaj(m, n, U.data(), name_u);

        //char name_s[] = "s";
        //RandBLAS::util::print_colmaj(n, 1, s.data(), name_s);

        //char name_vt[] = "VT";
        //RandBLAS::util::print_colmaj(n, n, VT.data(), name_vt);

        //char name_S[] = "S";
        //RandBLAS::util::print_colmaj(n, n, S.data(), name_S);

        // Below is A_k - A_hat = A_k - QB
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U_dat, m, S_dat, n, 1.0, A_k_dat, m);
        // A_k * VT -  A_hat == 0
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k_dat, m, VT_dat, n, -1.0, A_hat_dat, m);

        T norm_fro = lapack::lange(lapack::Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF A_k - QB: %f\n", norm_fro);
        T norm_A = lange(lapack::Norm::Fro, m, n, A_dat, m);
        printf("FRO NORM OF A - QB:   %f\n", norm_A);
        printf("Inner dimension of QB: %ld\n\n", k);
        // Compare this result with low-rank svd
        //ASSERT_NEAR(norm_fro, 0, 1e-12);
    }
};


TEST_F(TestQB, SimpleTest)
{
    //for (uint32_t seed : {0, 1, 2})
    //{
        // Testing rank-deficient matrices with exp decay, increasing clock size, normal termination
        //test_QB2<double>(1000, 1000, 100, 1, 0.0000000001, 1, 0);
        //test_QB2<double>(1000, 1000, 100, 5, 0.0000000001, 1, 0);
        //test_QB2<double>(1000, 1000, 100, 10, 0.0000000001, 1, 0);
        //test_QB2<double>(1000, 1000, 100, 20, 0.0000000001, 1, 0);
        //test_QB2<double>(1000, 1000, 100, 50, 0.0000000001, 1, 0);
        // 74 is when conditioning becomes an issue
        test_QB2<double>(1000, 1000, 100, 74, 0.0000000001, 1, 0);
        //test_QB2<double>(5000, 1000, 500, 10, 0.0000000001, 1, 0);
    //}
}