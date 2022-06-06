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
    static void check_QB1(int64_t m, int64_t n, int64_t k, uint32_t seed) {

        using namespace blas;
        using namespace lapack;

        /*
        std::vector<double> tau(n, 2.0);

        std::vector<double> A(m * n, 0.0);

        geqrf(m, n, A.data(), m, tau.data());
        ungqr(m, n, n, A.data(), m, tau.data());
        */

        
        int64_t size = m * n;


        std::vector<T> A(size);
        std::vector<T> A_hat(size);
        std::vector<T> A_k(size);
        std::vector<T> I_ref(k * k);
        std::vector<T> I_buf(k * k);
        std::vector<T> Omega(n * k, 0.0);
        std::vector<T> Q(m * k);
        std::vector<T> B(k * n);
        std::vector<T> I_test(n * n);
        //RandLAPACK::comps::util::eye<T>(n, n, I_test.data());

        // Generate a random matrix of std normal distribution
        // Rank of this matrix is half of the smaller dimension
        RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);
        std::copy(A.data(), A.data() + (n / 2) * m, A.data() + (n / 2) * m);
        
        // Generate a reference identity
        //RandLAPACK::comps::util::eye<T>(k, k, I_ref.data());

        //char name1[] = "A";
        //RandBLAS::util::print_colmaj(m, n, A.data(), name1);
        
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
	    0, // use_lu - start without
    	seed
        );
        */

        T tol = 0.0000000001;
        int block_sz = 1; 
        
        RandLAPACK::comps::qb::qb2<T>(
        m,
        n,
        A.data(),
        k, // Here, serves as a backup termination criteria
        block_sz,
        tol,
        0,
        1,
        Q.data(), // m by k
        B.data(), // k by n
        0,
        seed
        );
        

        //char name_3[] = "QB1  output Q";
        //RandBLAS::util::print_colmaj(m, k, Q.data(), name_3);

        //char name_4[] = "QB1  output B";
        //RandBLAS::util::print_colmaj(k, n, B.data(), name_4);


        //std::vector<T> A_cpy (m * n, 0.0);
        //lacpy(MatrixType::General, m, n, A.data(), m, A_cpy.data(), m);

        // A = A - Q * B
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, -1.0, Q.data(), m, B.data(), k, 1.0, A.data(), m);

        // A_hat = Q * B
        //gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q.data(), m, B.data(), k, 0.0, A_hat.data(), m);
    
        
        T norm_A = lange(lapack::Norm::Fro, m, n, A.data(), m);
        printf("A - A_hat %f\n", norm_A);

        /*
        GEMM SANITY CHECK
        std::vector<T> Three(3 * 3);
        std::vector<T> One(5 * 3);
        RandBLAS::dense_op::gen_rmat_norm<T>(5, 3, One.data(), seed);

        char name_5[] = "One";
        RandBLAS::util::print_colmaj(5, 3, One.data(), name_5);

        std::vector<T> Two(5 * 3);
        RandBLAS::dense_op::gen_rmat_norm<T>(5, 3, Two.data(), seed);

        char name_6[] = "Two";
        RandBLAS::util::print_colmaj(5, 3, Two.data(), name_6);

        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, 3, 3, 5, 1.0, One.data(), 5, Two.data(), 5, 0.0, Three.data(), 3);

        char name_7[] = "Three";
        RandBLAS::util::print_colmaj(3, 3, Three.data(), name_7);

        //char name_5[] = "QB1  output A_hat";
        //RandBLAS::util::print_colmaj(m, n, A_hat.data(), name_5);
        */


        /*
        // Get low-rank SVD to compare the result with
        std::vector<T> s(n, 0.0);
        std::vector<T> S(n * n, 0.0);
        std::vector<T> U(m * n, 0.0);
        std::vector<T> VT(n * n, 0.0);
        gesdd(Job::SomeVec, m, n, A_cpy.data(), m, s.data(), U.data(), m, VT.data(), n);

        //char name_u[] = "U";
        //RandBLAS::util::print_colmaj(m, n, U.data(), name_u);

        //char name_s[] = "s";
        //RandBLAS::util::print_colmaj(n, 1, s.data(), name_s);

        //char name_vt[] = "VT";
        //RandBLAS::util::print_colmaj(n, n, VT.data(), name_vt);

        // buffer zero vector
        std::vector<T> z_buf(n, 0.0);
        // zero out the trailing singular values
        std::copy(z_buf.data(), z_buf.data() + (n - k), s.data() + k);
        RandLAPACK::comps::util::diag(n, n, s.data(), S.data());

        //char name_S[] = "S";
        //RandBLAS::util::print_colmaj(n, n, S.data(), name_S);

        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U.data(), m, S.data(), n, 1.0, A_k.data(), m);
        // A_k * VT -  A_hat == 0
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k.data(), m, VT.data(), n, -1.0, A_hat.data(), m);

        //char name_7[] = "A - A_hat";
        //RandBLAS::util::print_colmaj(m, n, A_hat.data(), name_7);
        T norm_fro = lapack::lange(lapack::Norm::Fro, m, n, A_hat.data(), m);
        printf("FROBENIUS NORM OF A_k - A_hat %f\n", norm_fro);
        // Compare this result with low-rank svd
        //ASSERT_NEAR(norm_fro, 0, 1e-12);
        */
    }
};


TEST_F(TestQB, SimpleTest)
{
    check_QB1<double>(900, 450, 225, 0);
}