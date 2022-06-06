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



    // Currently, Householder reorthogonalization scheme works for square matrices of any size, but I'm not sure what it should do in a rectangular case
    //Testing Q' * Q = I.
    template <typename T>
    static void check_L(int64_t m, int64_t n, uint32_t seed) {

        using namespace blas;

        int64_t size = m * n;
        std::vector<T> A(size);

        // Generate a random matrix of std normal distribution
        RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);

        // Try to grab lower-triangular portion
        RandLAPACK::comps::util::get_L<T>(1, m, n, A.data());

        char name[] = "Lower Triangular A";
        RandBLAS::util::print_colmaj(m, n, A.data(), name);

    }
};


TEST_F(TestUtil, SimpleTest)
{
    //check_L<double>(5, 5, 0);
}
