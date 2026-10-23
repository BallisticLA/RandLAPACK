# RandLAPACK benchmarks

Performance-measurement harnesses for RandLAPACK's drivers and subroutines
(the numbers behind the papers and speed regressions). Not to be confused
with `extras/`, which extends functionality using third-party dependencies.

This directory is a **standalone CMake project**: it consumes an *installed*
RandLAPACK via `find_package`, exactly like user code does. On Linux and
macOS the autoinstaller builds it automatically into
`RandNLA-project/build/benchmark-build/` (the Windows installer does not
build it); to build it by hand against an existing install, from the
repository root:

```shell
cmake -S benchmark -B benchmark-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK
cmake --build benchmark-build -j
```

If CMake cannot find BLAS++, LAPACK++ or Random123 through the RandLAPACK
config, pass `-Dblaspp_DIR`, `-Dlapackpp_DIR` and `-DRandom123_DIR` as well;
the installer builds all three under `RandNLA-project/install/`. Configuring
also fetches Eigen, Spectra and fast_matrix_market with CMake's FetchContent,
so the first configure needs a network connection, and some benchmarks
borrow headers from `extras/`.

The sources are grouped by driver: `bench_BQRRP/`, `bench_CQRRPT/`,
`bench_ABRIK/`, `bench_CQRRT_linops/`, and `bench_general/`; the executables
land flat in the build directory. The driver benchmarks print their usage
when run without arguments, and most write their results as text files into
an output directory given on the command line. Benchmarks are compiled in
CI on Linux and macOS (by the core build jobs and by the default installer
lanes) but never executed there; run them on quiet, pinned hardware if you
intend to quote numbers.

## GPU benchmarks

Prerequisites: RandLAPACK installed with CUDA support (`--gpu` in the
installer, or `-DRequireCUDA=ON` manually), the CUDA Toolkit, and a GPU.
The benchmark project builds the GPU benchmark only when CMake finds the
CUDA Toolkit.

### BQRRP GPU benchmark

The first argument selects one of two modes; without one, the benchmark
prints its usage.

**Block size sweep**:
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

Written to the current working directory:

- `_BQRRP_GPU_speed_comparisons_block_size_*.txt` - block size sweep results
- `BQRRP_GPU_speed_comparisons_mat_size_*.txt` - matrix size sweep results
- `_BQRRP_GPU_runtime_breakdown_qrf_*.txt` - detailed profiling with QRF (profiling enabled)
- `_BQRRP_GPU_runtime_breakdown_cholqr_*.txt` - detailed profiling with CholQR (profiling enabled)
