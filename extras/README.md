# RandLAPACK Extras

Standalone project for extended functionality and tests that depend on
libraries not required by core RandLAPACK (e.g., Eigen, fast_matrix_market).

**Why this folder exists, and why it is a holding area.** Core RandLAPACK
deliberately depends on nothing beyond RandBLAS, BLAS++, LAPACK++, and
Random123. Some useful functionality (solver-backed linear operators, Matrix
Market file I/O) currently needs third-party libraries that we are not
willing to make RandLAPACK dependencies. Such code lives here, quarantined
behind its own build, until one of two things happens: the external
dependency is replaced with an in-house implementation and the functionality
graduates into RandLAPACK proper, or the functionality proves niche enough
to stay an optional integration permanently. Nothing in `extras/` is part of
RandLAPACK's API, and nothing in RandLAPACK may include from it.

**Not to be confused with `benchmark/`**: that directory measures the
performance of RandLAPACK itself; this one extends its functionality.

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
