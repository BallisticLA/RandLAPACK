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
