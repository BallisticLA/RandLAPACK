#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>
#include <lapack.hh>


#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestUtil : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};



    // Currently, Householder reorthogonalization scheme works for square matrices of any size, but I'm not sure what it should do in a rectangular case
    //Testing Q' * Q = I.
    template <typename T>
    static void check_orth(int64_t m, int64_t n, uint32_t seed) {
    
        using namespace blas;

        int64_t size = m * n;
        std::vector<T> A(size);
        std::vector<T> Q(size);
        std::vector<T> I_test(size);
        std::vector<T> I_ref(size);
        // Generate a random matrix of std normal distribution
        RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);
        // Generate a reference identity
        RandLAPACK::comps::util::eye<T>(m, n, I_ref.data());  
        
        // Orthonormalize A
        RandLAPACK::comps::util::householder_ref_gen<T>(m, n, A.data(), Q.data());
        
        // Q' * Q = I
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, m, m, 1.0, Q.data(), m, Q.data(), m, 0.0, I_test.data(), n);
        // Q * Q' = I
        //gemm<double>(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, n, 1.0, Q.data(), m, Q.data(), m, 0.0, I_test.data(), m);

        char name_1[] = "I";
        RandBLAS::util::print_colmaj(m, n, I_test.data(), name_1);

        T norm_fro = lapack::lange(lapack::Norm::Fro, m, n, I_test.data(), m);	
        ASSERT_NEAR(norm_fro, sqrt(std::min(m, n)), 1e-12);
    }


    template <typename T>
    static void check_dcgs2(int64_t m, int64_t n, uint32_t seed) {
    
        using namespace blas;

        int64_t size = m * n;
        std::vector<T> A(size);
        std::vector<T> Q(size);
        std::vector<T> I_test(size);
        std::vector<T> I_ref(size);
        // Generate a random matrix of std normal distribution
        RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);
        // Generate a reference identity
        RandLAPACK::comps::util::eye<T>(m, n, I_ref.data());  

        //char label[] = "A";
        //RandBLAS::util::print_colmaj<T>( m, n, A, label);

        // Orthonormalize A
        RandLAPACK::comps::orth::orth_dcgs2<T>(m, n, A.data(), Q.data());

        char name[] = "A";
        RandBLAS::util::print_colmaj(m, n, A.data(), name);
        
        char name_1[] = "Q";
        RandBLAS::util::print_colmaj(m, n, Q.data(), name_1);


        // Q' * Q = I
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, m, m, 1.0, Q.data(), m, Q.data(), m, 0.0, I_test.data(), n);
        // Q * Q' = I
        //gemm<double>(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, n, 1.0, Q.data(), m, Q.data(), m, 0.0, I_test.data(), m);

        char name_2[] = "I";
        RandBLAS::util::print_colmaj(m, n, I_test.data(), name_2);

        T norm_fro = lapack::lange(lapack::Norm::Fro, m, n, I_test.data(), m);	
        ASSERT_NEAR(norm_fro, sqrt(std::min(m, n)), 1e-12);
    }
};

/*
TEST_F(TestUtil, SimpleTest)
{
    for (uint32_t seed : {0, 1, 2})
    {
        //check_orth<double>(500, 500, seed);
        //check_orth<double>(373, 373, seed);
        check_orth<double>(5, 3, seed);
        //check_orth<float>(500, 500, seed);
    }
}
*/

TEST_F(TestUtil, SimpleTest)
{
    //for (uint32_t seed : {0, 1, 2})
    //{
        //check_orth<double>(500, 500, seed);
        //check_orth<double>(373, 373, seed);
        check_dcgs2<double>(373, 373, 0);
        //check_orth<float>(500, 500, seed);
    //}
}