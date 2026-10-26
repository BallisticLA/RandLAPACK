# RandLAPACK Extras

Standalone project for extended functionality and tests that depend on
libraries core RandLAPACK does not require (e.g., Eigen, fast_matrix_market).
Not to be confused with `benchmark/`, which measures the performance of
RandLAPACK itself; this one extends its functionality.

## Why this folder is a holding area

Core RandLAPACK deliberately depends on nothing beyond RandBLAS, BLAS++,
LAPACK++ and Random123. Some useful functionality (solver-backed linear
operators, Matrix Market file I/O) currently needs third-party libraries
that we are not willing to make RandLAPACK dependencies. Such code lives
here, behind its own build, until one of two things happens:

- the external dependency is replaced with an in-house implementation, and
  the functionality graduates into RandLAPACK proper; or
- the functionality proves niche enough to stay an optional integration
  permanently.

Nothing in `extras/` is part of RandLAPACK's API, and nothing in RandLAPACK
may include from it. Some benchmarks do borrow headers from here.

## Directory layout

```
extras/
├── linops/      Solver-based linear operators (CholSolverLinOp, LUSolverLinOp)
├── misc/        Matrix Market conversions (ext_util.hh) and the matrix-file
│                loader BenchIO::load_matrix (ext_matrix_io.hh)
└── test/        GTest-based regression tests for the above
```

## Building

This is a standalone CMake project that uses an *installed* RandLAPACK. It
needs CMake 3.24 or later.

- **With the installer.** `bash install.sh` builds extras by default on Linux
  and macOS (`--no-extras` skips it), into
  `RandNLA-project/build/extras-build/`. The Windows installer does not build
  extras.
- **By hand**, against an existing install, from the repository root:
  ```bash
  cmake -S extras -B extras-build -DCMAKE_BUILD_TYPE=Release \
      -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK
  cmake --build extras-build -j 4
  ```
  - Give `RandLAPACK_DIR` as an absolute path. After `install.sh` it is
    `<project-dir>/install/RandLAPACK-install/lib/cmake/RandLAPACK`.
  - With oneMKL, load its environment first (`source <oneapi>/setvars.sh`;
    [INSTALL.md, section 5](../docs/INSTALL.md#5-using-randlapack-in-other-projects)).
  - If the machine has a CUDA Toolkit whose `nvcc` rejects your compiler, add
    `-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE`.
  - On macOS with Homebrew's `libomp`, add the OpenMP settings at the end of
    [INSTALL.md, section 5](../docs/INSTALL.md#5-using-randlapack-in-other-projects).

The tests are built when GoogleTest is found; `-DBUILD_TESTING=OFF` skips
them. Run them with `ctest --test-dir extras-build`, or after `install.sh`
with `ctest --test-dir <project-dir>/build/extras-build`.

**No manual dependency installation is needed.** CMake downloads these
during configuration, with FetchContent:

- **Eigen** (https://eigen.tuxfamily.org/): sparse matrix factorizations
  (SimplicialLLT, SparseLU);
- **fast_matrix_market**: Matrix Market file I/O.

Both are pinned to the source archive of one exact commit, checked against a
SHA256 (`URL` and `URL_HASH` in `extras/CMakeLists.txt`). `benchmark/` pins
the same two archives, so a bump changes both files together; see
[benchmark/README.md](../benchmark/README.md).

## Matrix file input

`misc/ext_matrix_io.hh` provides `BenchIO::load_matrix<T>` for shared
benchmark inputs. It reads:

- Matrix Market array or coordinate files (`.mtx`);
- native binary dense files (`.bin`), in the format documented by
  `RandLAPACK::gen::read_bin_matrix`;
- whitespace-delimited dense text, for other extensions or filenames without
  an extension.

Extension matching ignores case. With the extras directory on the include
path and the dependencies above linked:

```cpp
#include "misc/ext_matrix_io.hh"

auto matrix = BenchIO::load_matrix<double>("matrix.mtx");
auto block = BenchIO::load_matrix<double>("matrix.mtx", 0.5);
```

- **Result.** Dense input gives a compact column-major buffer through
  `matrix.data()`. Sparse input gives both `*matrix.csc` (RandBLAS) and
  `*matrix.eigen_sparse` (Eigen), and `matrix.is_sparse` tells the two cases
  apart. The result owns its storage and can be moved but not copied; keep it
  alive while you use its data or sparse views.
- **Ratio.** The optional second argument selects the top-left submatrix,
  for every format. Each output dimension is rounded down after scaling. The
  ratio must be finite and in `(0, 1]`, and neither output dimension may be
  zero.
- **Errors.** Invalid input or arguments raise `RandLAPACK::Error`.
