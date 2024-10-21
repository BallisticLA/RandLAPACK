#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>

#include "sparse.hpp" // define CSR sparse matrix type
#include "rchol.hpp"
#include "util.hpp"
#include "pcg.hpp"

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

struct InvLaplacian : public RandLAPACK::SymmetricLinearOperator<double> {
    SparseCSR A;
    const SparseCSR G;
    // also has const m
};


int main(int argc, char *argv[]) {
  int n = 3; // DoF in every dimension
  int threads = 1;
  for (int i=0; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i], "-t"))
      threads = atoi(argv[i+1]);
  }
  std::cout<<std::setprecision(3);
 
  // SDDM matrix from 3D constant Poisson equation
  SparseCSR A;
  A = laplace_3d(n); // n x n x n grid

  // random RHS
  int N = A.size();
  std::vector<double> b(N); 
  rand(b);

  // compute preconditioner (single thread) and solve 
  SparseCSR G;
  rchol(A, G);
  std::cout<<"Fill-in ratio: "<<2.*G.nnz()/A.nnz()<<std::endl;

  // solve with PCG
  double tol = 1e-6;
  int maxit = 200;
  double relres;
  int itr;
  std::vector<double> x;
  pcg(A, b, tol, maxit, G, x, relres, itr);
  std::cout<<"# CG iterations: "<<itr<<std::endl;
  std::cout<<"Relative residual: "<<relres<<std::endl;

  return 0;
}
