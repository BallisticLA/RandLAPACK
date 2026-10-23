<p align="center">
  <img src=".github/assets/randlapack-banner.png" alt="RandLAPACK" width="100%">
</p>

# RandLAPACK

RandLAPACK provides high-performance randomized algorithms for linear algebra
problems such as least squares, (kernel) ridge regression, low-rank
approximation, and matrix factorizations.
RandLAPACK's API is not yet stable; we're actively working on changing that.

Please swing by [**our Discord server**](https://discord.gg/R4qj8Er9YW) if you
have questions about RandLAPACK or would like to get involved in its
development.

## Quickstart

```shell
git clone https://github.com/BallisticLA/RandLAPACK.git
cd RandLAPACK
bash install.sh
```

The installer builds RandLAPACK together with its dependencies (BLAS++ and
LAPACK++ from the BallisticLA forks, and Random123; RandBLAS comes along as a
pinned git submodule), the extras and benchmark projects, and the test
suite. You bring a C++20 compiler, CMake 3.21 or later, Git, a BLAS/LAPACK
library (on macOS 13.3 or later the installer uses Apple's Accelerate, so
there is nothing to install), and GoogleTest for the test suite. Run
`bash install.sh --help` for the options
(GPU support, parallelism, reusing preinstalled dependencies, and more), and
see [docs/INSTALL.md](docs/INSTALL.md) for manual installation and for consuming
RandLAPACK from your own CMake project. A smoke test after installation:

```shell
ctest --test-dir ../RandNLA-project/build/RandLAPACK-build
```

On native Windows, run `installers\install.ps1` from an x64 MSVC developer
prompt instead; [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) gives the
exact command and explains which prompt to open.

## What's in the library

The user-facing algorithms ("drivers"; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the library is organized):

| Driver | Problem it solves | Reference |
|--------|-------------------|-----------|
| `BQRRP` | Blocked QR with randomized column pivoting, any aspect ratio | [arXiv:2507.00976](https://arxiv.org/abs/2507.00976) |
| `CQRRPT` | Sketch-based QR with column pivoting for tall matrices | [arXiv:2311.08316](https://arxiv.org/abs/2311.08316) |
| `CQRRT` | Unpivoted QR of tall, full-rank matrices by sketch-preconditioned Cholesky QR; `CQRRT_linops` is its Q-less variant for linear operators | [arXiv:2111.11148](https://arxiv.org/abs/2111.11148) |
| `RSVD` | Randomized low-rank SVD via the QB decomposition | [arXiv:0909.4061](https://arxiv.org/abs/0909.4061) |
| `REVD2` | Randomized low-rank eigendecomposition of symmetric positive semidefinite matrices | [arXiv:2110.02820](https://arxiv.org/abs/2110.02820) |
| `ABRIK` | Block-Krylov iterative SVD for many accurate singular triplets | [arXiv:2306.12418](https://arxiv.org/abs/2306.12418) |
| `hqrrp` | Householder QR with randomized pivoting | [arXiv:1512.02671](https://arxiv.org/abs/1512.02671) |
| `krill_full_rpchol` | Kernel ridge regression by conjugate gradients, preconditioned with randomly pivoted Cholesky | [arXiv:2304.12465](https://arxiv.org/abs/2304.12465) |

BQRRP and CQRRPT also have CUDA implementations (`BQRRP_GPU`,
`CQRRPT_GPU`). Drivers are assembled from smaller randomized building blocks
(rangefinders, QB, block Krylov, row sketching, orthogonalization and
stabilization routines, and randomly pivoted Cholesky, `rp_cholesky`) that
are usable on their own; the RandNLA monograph
([arXiv:2302.11474](https://arxiv.org/abs/2302.11474)) is the best background
reference for the algorithm families.

## Related libraries

RandLAPACK depends on [RandBLAS](https://github.com/BallisticLA/RandBLAS),
which we are also developing.

Before starting on RandLAPACK we implemented several high-level RandNLA
algorithms in Matlab ([MARLA](https://github.com/BallisticLA/marla)) and
Python ([PARLA](https://github.com/BallisticLA/parla)).
In the latter library we took an approach where *algorithms are objects.*
An algorithm needs to be instantiated with its tuning parameters and
subroutines in order to be used.
RandLAPACK currently emphasizes that "algorithms as objects" approach.

## Where to go from here

Everything beyond this page lives in `docs/` (reference documentation),
`CONTRIBUTING.md` at the root (the development workflow), and the READMEs of
the two standalone projects, `benchmark/` and `extras/`.

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
