<p align="center">
  <img src=".github/assets/randlapack-banner.png" alt="RandLAPACK" width="100%">
</p>

# RandLAPACK

RandLAPACK is a C++ library of high-performance randomized algorithms for
linear algebra problems such as least squares, (kernel) ridge regression,
low-rank approximation, and matrix factorizations. Its API is not yet stable;
we are actively working on changing that.

Questions, or want to get involved in development? Join
[**our Discord server**](https://discord.gg/R4qj8Er9YW).

## Install

**Linux and macOS:**

```shell
git clone https://github.com/BallisticLA/RandLAPACK.git
cd RandLAPACK
bash install.sh
```

Then check the install by running the test suite:

```shell
ctest --test-dir ../RandNLA-project/build/RandLAPACK-build
```

**Native Windows**, from an "x64 Native Tools Command Prompt for VS 2022"
([INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) explains which prompt to open,
and why):

```bat
git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
cd RandLAPACK
powershell -ExecutionPolicy Bypass -File .\installers\install.ps1
```

**What you need first** (on Linux and macOS):

- a C++20 compiler, CMake 3.21 or later, and Git;
- a BLAS/LAPACK library, the standard low-level linear algebra routines
  (OpenBLAS or Intel oneMKL on Linux; on macOS 13.3 or later the installer
  uses Apple's built-in Accelerate, so there is nothing to install);
- GoogleTest, for the test suite.

**What the installer builds:** RandLAPACK, its dependencies, the extras and
benchmark projects, and the test suite. The dependencies are BLAS++ and
LAPACK++ (C++ wrappers for BLAS and LAPACK, built from the BallisticLA
forks), Random123, and RandBLAS, which comes inside this repository as a
pinned git submodule (another git repository included at one fixed commit). Run `bash install.sh --help`
for the options: GPU support, parallelism, reusing preinstalled dependencies,
and more.

To build by hand, or to use RandLAPACK from your own CMake project, see
[docs/INSTALL.md](docs/INSTALL.md).

## What's in the library

The user-facing algorithms, called "drivers"
([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the library is
organized):

| Driver | Problem it solves | Reference |
|--------|-------------------|-----------|
| `BQRRP` | Blocked QR with randomized column pivoting, any aspect ratio | [arXiv:2507.00976](https://arxiv.org/abs/2507.00976) |
| `CQRRPT` | Sketch-based QR with column pivoting for tall matrices | [arXiv:2311.08316](https://arxiv.org/abs/2311.08316) |
| `CQRRT` | Unpivoted QR of tall, full-rank matrices by sketch-preconditioned Cholesky QR; `CQRRT_linops` is its Q-less variant for linear operators | [arXiv:2111.11148](https://arxiv.org/abs/2111.11148) |
| `RSVD` | Randomized low-rank SVD via the QB decomposition, of a dense matrix or any linear operator (sparse or implicit) | [arXiv:0909.4061](https://arxiv.org/abs/0909.4061) |
| `REVD2` | Randomized low-rank eigendecomposition of symmetric positive semidefinite matrices | [arXiv:2110.02820](https://arxiv.org/abs/2110.02820) |
| `ABRIK` | Block-Krylov iterative SVD for many accurate singular triplets | [arXiv:2306.12418](https://arxiv.org/abs/2306.12418) |
| `hqrrp` | Householder QR with randomized pivoting | [arXiv:1512.02671](https://arxiv.org/abs/1512.02671) |
| `krill_full_rpchol` | Kernel ridge regression by conjugate gradients, preconditioned with randomly pivoted Cholesky | [arXiv:2304.12465](https://arxiv.org/abs/2304.12465) |

- BQRRP and CQRRPT also have CUDA (GPU) implementations, `BQRRP_GPU` and
  `CQRRPT_GPU`.
- The drivers are assembled from smaller randomized building blocks that you
  can also use on their own: rangefinders, QB, block Krylov, row sketching,
  orthogonalization and stabilization routines, and randomly pivoted Cholesky
  (`rp_cholesky`).
- The RandNLA monograph ([arXiv:2302.11474](https://arxiv.org/abs/2302.11474))
  is the best background reference for these algorithm families.

## Related libraries

RandLAPACK depends on [RandBLAS](https://github.com/BallisticLA/RandBLAS),
which we are also developing.

Before RandLAPACK, we implemented several high-level RandNLA algorithms in
MATLAB ([MARLA](https://github.com/BallisticLA/marla)) and Python
([PARLA](https://github.com/BallisticLA/parla)). PARLA treats *algorithms as
objects*: you instantiate an algorithm with its tuning parameters and
subroutines before using it. RandLAPACK currently emphasizes the same
approach.

## Where to go from here

| You want to... | Read |
|----------------|------|
| Install with one command on Linux or macOS | [docs/INSTALL_SCRIPT.md](docs/INSTALL_SCRIPT.md) |
| Install on native Windows (MSVC) | [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) |
| Install by hand, or use RandLAPACK from your CMake project | [docs/INSTALL.md](docs/INSTALL.md) |
| Understand how the library is organized | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Contribute code | [CONTRIBUTING.md](CONTRIBUTING.md) |
| See what CI checks, and why | [docs/CI.md](docs/CI.md) |
| Run or add performance benchmarks | [benchmark/README.md](benchmark/README.md) |
| Use the Eigen/Matrix-Market integrations | [extras/README.md](extras/README.md) |
