# RandLAPACK Extras

Standalone project for extended functionality and tests that depend on libraries not required by core RandLAPACK (e.g., Eigen, fast_matrix_market).

## Directory Layout

```
extras/
├── linops/      Solver-based linear operators (CholSolverLinOp, LUSolverLinOp)
├── misc/        General utilities (format conversions, diagnostics)
├── testing/     Test-specific utilities (SPD generators, etc.)
└── test/        GTest-based regression tests for the above
```

## Dependencies

Extras automatically fetch these external libraries via CMake FetchContent:

- **Eigen** (https://eigen.tuxfamily.org/): Sparse matrix factorizations (SimplicialLLT, SparseLU)
- **fast_matrix_market**: Matrix Market file I/O

**No manual dependency installation required!** CMake will download them during configuration.

## Building

This is a standalone CMake project. It requires RandLAPACK to be installed first:

```bash
# From the RandNLA-project root (after RandLAPACK is installed):
cmake -S lib/RandLAPACK/extras/ -B build/extras-build/ \
    -DCMAKE_BUILD_TYPE=Release \
    -DRandLAPACK_DIR=install/RandLAPACK-install/lib/cmake/RandLAPACK/
make -C build/extras-build/ -j$(nproc)
```

Or use the top-level `install.sh` which handles the full build sequence automatically.

## Matrix file input

`misc/ext_matrix_io.hh` provides `BenchIO::load_matrix<T>` for shared benchmark
inputs. It reads Matrix Market array or coordinate files (`.mtx`), native binary
dense files (`.bin`), and whitespace-delimited dense text for other extensions
or filenames without an extension. Extension matching ignores case. Binary
files use the format documented by `RandLAPACK::gen::read_bin_matrix`.

With the extras directory on the include path and the dependencies above linked:

```cpp
#include "misc/ext_matrix_io.hh"

auto matrix = BenchIO::load_matrix<double>("matrix.mtx");
auto block = BenchIO::load_matrix<double>("matrix.mtx", 0.5);
```

Dense input gives a compact column-major buffer through `matrix.data()`. Sparse
input gives both `*matrix.csc` (RandBLAS) and `*matrix.eigen_sparse` (Eigen), with
`matrix.is_sparse` distinguishing the two cases. The result owns its storage and
can be moved but not copied; keep it alive while using its data or sparse views.

The optional ratio selects the top-left submatrix for every format. Each output
dimension is rounded down after scaling. The ratio must be finite and in
`(0, 1]`, and neither output dimension may be zero. Invalid input or arguments
raise `RandLAPACK::Error`.
