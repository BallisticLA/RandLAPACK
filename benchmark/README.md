# RandLAPACK benchmarks

This directory is a **standalone CMake project**: it consumes an *installed*
RandLAPACK via `find_package`, exactly like user code does. The autoinstaller
builds it automatically into `RandNLA-project/build/benchmark-build/`; to
build it by hand against an existing install:

```shell
cmake -S benchmark -B benchmark-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK
cmake --build benchmark-build -j
```

(If BLAS++/LAPACK++/Random123 are not discoverable from the RandLAPACK
config, pass `-Dblaspp_DIR/-Dlapackpp_DIR/-DRandom123_DIR` too, as printed
by the installer's final summary.)

The executables are grouped by driver: `bench_BQRRP/`, `bench_CQRRPT/`,
`bench_ABRIK/`, `bench_CQRRT_linops/`, and `bench_general/`. Each writes
timing text files into the current working directory. Benchmarks are
compile-checked in CI but never executed there; run them on quiet, pinned
hardware if you intend to quote numbers.

Note for macOS: benchmarks that depend on routines absent from Apple's
default LAPACK build as whole-file stubs that print a message and return 1.

## GPU benchmarks

Prerequisites: RandLAPACK installed with CUDA support (`--gpu` in the
installer, or `-DRequireCUDA=ON` manually), the CUDA Toolkit, and a GPU.

### BQRRP GPU benchmark

Two modes:

**Block size sweep** (default):
```shell
./BQRRP_GPU_benchmark block_size [matrix_size] [profile_runtime] [run_qrf]
```

Examples:
```shell
# Default settings (16384x16384 matrix)
./BQRRP_GPU_benchmark block_size

# 32768x32768 matrix
./BQRRP_GPU_benchmark block_size 32768

# Profiling enabled and QRF comparison
./BQRRP_GPU_benchmark block_size 16384 1 1
```

**Matrix size sweep**:
```shell
./BQRRP_GPU_benchmark mat_size [profile_runtime] [run_qrf]
```

Examples:
```shell
# Default settings
./BQRRP_GPU_benchmark mat_size

# Profiling disabled, QRF comparison enabled
./BQRRP_GPU_benchmark mat_size 0 1
```

### Output files

- `_BQRRP_GPU_speed_comparisons_block_size_*.txt` - block size sweep results
- `BQRRP_GPU_speed_comparisons_mat_size_*.txt` - matrix size sweep results
- `_BQRRP_GPU_runtime_breakdown_qrf_*.txt` - detailed profiling with QRF (profiling enabled)
- `_BQRRP_GPU_runtime_breakdown_cholqr_*.txt` - detailed profiling with CholQR (profiling enabled)
