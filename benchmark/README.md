# RandLAPACK benchmarks

Performance-measurement harnesses for RandLAPACK's drivers and subroutines:
the numbers behind the papers, and a way to catch speed regressions. Not to
be confused with `extras/`, which extends functionality using third-party
dependencies.

## Building

This directory is a **standalone CMake project**: it uses an *installed*
RandLAPACK through `find_package`, exactly as user code does.

- **With the installer.** On Linux and macOS, `install.sh` builds it
  automatically into `RandNLA-project/build/benchmark-build/`. The Windows
  installer does not build it.
- **By hand**, against an existing install, from the repository root:
  ```shell
  cmake -S benchmark -B benchmark-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK
  cmake --build benchmark-build -j 4
  ```
  - If CMake cannot find BLAS++, LAPACK++ or Random123 through the
    RandLAPACK config, also pass `-Dblaspp_DIR`, `-Dlapackpp_DIR` and
    `-DRandom123_DIR`; the installer builds all three under
    `RandNLA-project/install/`.
  - With oneMKL, load its environment first (`source <oneapi>/setvars.sh`;
    [USING.md](../docs/USING.md#building-it)).
  - On macOS with Homebrew's `libomp`, add the OpenMP settings of
    [USING.md](../docs/USING.md#macos-with-homebrews-libomp).

The project needs CMake 3.24 or later. Its first configure needs a network
connection: it fetches Eigen, Spectra and fast_matrix_market with CMake's
FetchContent. Some benchmarks also borrow headers from `extras/`.

## Running

- The sources are grouped by driver: `bench_BQRRP/`, `bench_CQRRPT/`,
  `bench_ABRIK/`, `bench_CQRRTO_linops/` (Q-less QR and least squares on a
  FEM problem), and `bench_general/`. The executables land flat in the build
  directory.
- The driver benchmarks print their usage when run without arguments. Most
  write their results as text files into an output directory given on the
  command line. Create it first, and end its name with `/` (for example
  `results/`): several drivers append the file name to it directly.
- The ABRIK and `bench_CQRRTO_linops/` drivers copy the environment variable
  `RANDLAPACK_GIT_COMMIT` into the header of every CSV file they write
  (`unknown` when it is unset).
  Export the commit you built from, so that each result file names its build.
- CI compiles the benchmarks on Linux and macOS (in the core build jobs and
  the default installer lanes) but never runs them. Run them on quiet,
  pinned hardware if you intend to quote numbers.

## Writing a benchmark

Shared scaffolding lives in `rl_bench_utils.hh` at the root of this directory
(namespace `RandLAPACK::bench`). It is part of the benchmark project and does
not install with the library. It holds the argument and usage helpers, the
`main()` that dispatches on `double` or `float`, the output-path helpers, the
timestamp prefix of result files and the timestamped CSV opener, the
build-commit stamp, the row and footer writers, the `QR_benchmark_data`
struct with its regeneration routine, and the QR error-analysis and
pivot-quality helpers. Use these instead of copying them
from a neighboring driver: external plotting and campaign scripts parse the
file names and row formats they write.

## GPU benchmarks

Prerequisites: RandLAPACK installed with CUDA support (`--gpu` in the
installer, or `-DRequireCUDA=ON` manually), the CUDA Toolkit, and a GPU. The
benchmark project builds the GPU benchmarks only when CMake finds the CUDA
Toolkit and RandLAPACK's BLAS++ and LAPACK++ were built with CUDA; otherwise
it skips them and says why.

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

Output files, written to the current working directory:

| File | Contents |
|---|---|
| `_BQRRP_GPU_speed_comparisons_block_size_*.txt` | block size sweep results |
| `BQRRP_GPU_speed_comparisons_mat_size_*.txt` | matrix size sweep results |
| `_BQRRP_GPU_runtime_breakdown_qrf_*.txt` | detailed profiling with QRF (profiling enabled) |
| `_BQRRP_GPU_runtime_breakdown_cholqr_*.txt` | detailed profiling with CholQR (profiling enabled) |

### CQRRPT GPU benchmark

```shell
./CQRRPT_GPU_benchmark num_rows num_columns [num_runs] [d_factor] [bqrrp_block_sizes]
```

It times CPU CQRRPT and CQRRPT_GPU in double and float, with A in pageable
and in pinned host memory. CQRRPT_GPU runs with each QRCP of the sketch:
`luqr`, and `bqrrp` at each block size in the comma-separated list
`bqrrp_block_sizes` (default `64,256,0`, where 0 is CQRRPT's rule). It also
compares the pivot quality of each method against CPU CQRRPT. The defaults
are 3 runs and `d_factor` 2.0. The `bqrrp` methods are skipped when the
sketch has more than 524,280 rows (`d_factor` times `num_columns`).

```shell
./CQRRPT_GPU_benchmark 100000 1000 5 2.0 64,256,0
```

Output files, written to the current working directory:

| File | Contents |
|---|---|
| `*_CQRRPT_GPU_speed_double_<rows>_<columns>.txt`, `*_CQRRPT_GPU_speed_float_<rows>_<columns>.txt` | best time, rank, errors and CQRRPT_GPU's step times for each method |
| `*_CQRRPT_GPU_pivot_quality_<rows>_<columns>.txt` | pivot quality of each method against CPU CQRRPT, in double |

## Pinned downloads (for maintainers)

Eigen, Spectra and fast_matrix_market are pinned. Each is downloaded as the
source archive of one exact commit and checked against a SHA256 (`URL` and
`URL_HASH` in `benchmark/CMakeLists.txt`), not cloned from a moving branch,
so every build compiles the same sources. Eigen's pin is a commit on its
`master` branch.

- **To bump a pin**, put the new commit in the `URL` and the `sha256sum` of
  the new archive in `URL_HASH`.
- **Keep `extras/` in step.** `extras/CMakeLists.txt` pins Eigen and
  fast_matrix_market too, and the two files must name the same archives: the
  installer configures both projects with one shared FetchContent download
  directory.
