#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include "sparse.hpp" // define CSR sparse matrix type
#include "rchol_parallel.hpp"
#include "util.hpp"
#include "pcg.hpp"

#define SparseCSR_RC SparseCSR<double>


int main(int argc, char *argv[]) {
  int n = 3; // DoF in every dimension
  int threads = 64;
  for (int i=0; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i], "-t"))
      threads = atoi(argv[i+1]);
  }
  std::cout<<std::setprecision(3);
 
  // SDDM matrix from 3D constant Poisson equation
  SparseCSR_RC A;
  A = laplace_3d(n); // n x n x n grid

  // random RHS
  int N = A.size();
  std::vector<double> b(N); 
  rand(b);

  // compute preconditioner (multithread) and solve
  SparseCSR_RC G;
  std::vector<size_t> P;
  rchol(A, G, P, threads);
  std::cout<<"Fill-in ratio: "<<2.*G.nnz()/A.nnz()<<std::endl;

  // solve the reordered problem with PCG
  SparseCSR_RC Aperm; reorder(A, P, Aperm);
  std::vector<double> bperm; reorder(b, P, bperm);
    
  double tol = 1e-6;
  int maxit = 200;
  double relres;
  int itr;
  std::vector<double> x;
  pcg(Aperm, bperm, tol, maxit, G, x, relres, itr);
  std::cout<<"# CG iterations: "<<itr<<std::endl;
  std::cout<<"Relative residual: "<<relres<<std::endl;


  return 0;
}
