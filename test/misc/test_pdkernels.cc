#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include "/Users/rjmurr/Documents/randnla/RandLAPACK/RandBLAS/test/comparison.hh"

#include <math.h>
#include <gtest/gtest.h>


/**
 * Test that if X is rank-1 then the squared exponential kernel
 * gives a matrix of all ones.
 */


/**
 * Test that squared_exp_kernel_submatrix gives the same result
 * as calls to squared_exp_kernel.
 */


/**
 * Test that if the columns of X are orthonormal then the diagonal
 * will be all ones and the off-diagonal will be exp(-bandwidth^{-2});
 * this needs to vary with differ
 */

