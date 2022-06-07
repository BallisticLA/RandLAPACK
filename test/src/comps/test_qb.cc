#include <gtest/gtest.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>


#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestQB : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};



    template <typename T>
    static void check_QB1(int64_t m, int64_t n, int64_t k, int64_t block_sz, T tol, uint32_t seed) {

        using namespace blas;
        using namespace lapack;
        
        int64_t size = m * n;

        // For running QB
        std::vector<T> A(size);
        std::vector<T> Q(m * k);
        std::vector<T> B(k * n);

        // For results comparison
        std::vector<T> A_hat(size);
        std::vector<T> A_k(size);
        std::vector<T> A_cpy (m * n, 0.0);
        lacpy(MatrixType::General, m, n, A.data(), m, A_cpy.data(), m);

        // For low-rank SVD
        std::vector<T> s(n, 0.0);
        std::vector<T> S(n * n, 0.0);
        std::vector<T> U(m * n, 0.0);
        std::vector<T> VT(n * n, 0.0);


        // Generate a random matrix of std normal distribution
        // Rank of this matrix is half of the smaller dimension
        RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);
        std::copy(A.data(), A.data() + (n / 2) * m, A.data() + (n / 2) * m);

        char name1[] = "A";
        RandBLAS::util::print_colmaj(m, n, A.data(), name1);
        
        /*
        RandLAPACK::comps::qb::qb1<T>(
        m,
        n,
        A.data(),
        k, // k - start with full rank
        0, // passes over data - vary
        1, // passes per stab == 1 by default
        Q.data(), // m by k
        B.data(), // k by n
    	seed
        );
        */
        
        RandLAPACK::comps::qb::qb2<T>(
        m,
        n,
        A.data(),
        k, // Here, serves as a backup termination criteria
        block_sz,
        tol,
        5,
        1,
        Q.data(), // m by k
        B.data(), // k by n
        seed
        );
        
        char name_3[] = "QB1  output Q";
        RandBLAS::util::print_colmaj(m, k, Q.data(), name_3);

        char name_4[] = "QB1  output B";
        RandBLAS::util::print_colmaj(k, n, B.data(), name_4);

        // A_hat = Q * B
        //gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q.data(), m, B.data(), k, 0.0, A_hat.data(), m);

        // A = A - Q * B
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, -1.0, Q.data(), m, B.data(), k, 1.0, A.data(), m);

        T norm_A = lange(lapack::Norm::Fro, m, n, A.data(), m);
        printf("FRO NORM OF A - QB %f\n", norm_A);
/*
        // Get low-rank SVD
        gesdd(Job::SomeVec, m, n, A_cpy.data(), m, s.data(), U.data(), m, VT.data(), n);
        // buffer zero vector
        std::vector<T> z_buf(n, 0.0);
        // zero out the trailing singular values
        std::copy(z_buf.data(), z_buf.data() + (n - k), s.data() + k);
        RandLAPACK::comps::util::diag(n, n, s.data(), S.data());

        //char name_u[] = "U";
        //RandBLAS::util::print_colmaj(m, n, U.data(), name_u);

        //char name_s[] = "s";
        //RandBLAS::util::print_colmaj(n, 1, s.data(), name_s);

        //char name_vt[] = "VT";
        //RandBLAS::util::print_colmaj(n, n, VT.data(), name_vt);

        //char name_S[] = "S";
        //RandBLAS::util::print_colmaj(n, n, S.data(), name_S);

        // Below is A_k - A_hat = A_k - QB
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U.data(), m, S.data(), n, 1.0, A_k.data(), m);
        // A_k * VT -  A_hat == 0
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k.data(), m, VT.data(), n, -1.0, A_hat.data(), m);

        T norm_fro = lapack::lange(lapack::Norm::Fro, m, n, A_hat.data(), m);
        printf("FRO NORM OF A_k - QB %f\n", norm_fro);
        // Compare this result with low-rank svd
        //ASSERT_NEAR(norm_fro, 0, 1e-12);
*/
    }
};


TEST_F(TestQB, SimpleTest)
{
    check_QB1<double>(100, 70, 35, 10, 0.0000000001, 0);
}