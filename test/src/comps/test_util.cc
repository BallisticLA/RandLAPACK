/*
TODO #1: Resizing tests.

TODO #2: Diagonalization tests.

TODO #4: L & pivotig tests.
*/

#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>

#include <chrono>
using namespace std::chrono;

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestUtil : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void 
    test_1()
    {
    }
};
