#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>


#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestUtil : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};


    /*
    Notes on using templates:
        I'm having issues separating the header definition and implementation like it is done for the determiter.
        For now, functions will be fully defined in headers.
    */
    virtual void check_orth() {
    
        using namespace blas;

        uint32_t seed = 12;
        int64_t m = 500; // rows
        int64_t n = 500; // cols
        int64_t size = m * n;
        std::vector<double> A(size);
        std::vector<double> Q(size);
        std::vector<double> I_test(size);
        std::vector<double> I_ref(size);
        // Generate a random matrix of std normal distribution
        RandBLAS::dense_op::gen_rmat_norm<double>(m, n, A.data(), seed);
        // Generate a reference identity

        RandLAPACK::comps::util::eye<double>(m, n, I_ref.data());

        
        // Orthonormalize A
        RandLAPACK::comps::util::householder_ref_gen<double>(m, n, A.data(), Q.data());
        /*
        // Test: norm(I_ref - I_test) ~= 0
        // Q' * Q
        gemm<float>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, m, m, 1.0, Q.data(), m, Q.data(), m, 0.0, I_test.data(), n);
        // Q * Q'
        gemm<float>(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, n, 1.0, Q.data(), m, Q.data(), m, 0.0, I_test.data(), m);

        // l-2 norm of Q should be 1

        // fro norm of Q should be sqrt(min(m, n)) - l-2 vector norm acts pretty much like fro matrix norm
        double norm = nrm2(m, Q.data(), 1);
        ASSERT_NEAR(norm, sqrt(std::min(m, n)), 1e-12);
        printf("Frobenius norm: %f\n", norm);
        printf("Estimate: %f\n", sqrt(std::min(m, n)));
        */
    
    }

};


TEST_F(TestUtil, SimpleTest)
{
    check_orth();
}