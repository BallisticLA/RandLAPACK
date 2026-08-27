# Using RandLAPACK's Automated Install Script

The installer scripts live in the `install/` directory (`install/install.sh`
for Linux/macOS, `install/install.ps1` for native Windows); a small wrapper is
kept at the repository root so `bash install.sh` keeps working.

This guide explains how to use the `install.sh` script to automatically install
RandLAPACK and all of its dependencies (BLAS++, LAPACK++, Random123) with a
single command.

**When to use this guide:** Use this automated installation method if you want
a quick, streamlined setup process. If you need fine-grained control over
dependency configurations, refer to RandLAPACK's `INSTALL.md` instead.

> **Windows users:** this document describes the Linux/macOS installer
> (`install.sh`). The native Windows companion is `install\install.ps1`,
> documented in [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md) -- prerequisites,
> every option, backend selection, and troubleshooting.

## 0. Software Requirements

Before running the install script, ensure you have the following software
available on your system:

### Essential Requirements
* **C++ Compiler:** GNU GCC 13.3.0 or higher (required for C++20 features)
* **CMake:** Version 3.21 or higher (the project's CMake floor; recent
  releases recommended)
* **BLAS/LAPACK Library:** Intel MKL 2022 or higher recommended
* **GoogleTest:** (Optional but recommended) For running RandLAPACK tests

### GPU Support Requirements (Optional)
* **CUDA Toolkit:** Version 12.4.1 or higher
  - **Recommended:** CUDA 12.9.0 + GCC 13.3.0 (verified working as of 2025-11-26)
  - **IMPORTANT:** CUDA versions have strict GCC compatibility requirements:
    - CUDA 12.9.0: Compatible with GCC 13.x ✓
    - CUDA 12.4.1: Compatible with GCC 13.x ✓
    - CUDA 12.2.1: Requires GCC ≤ 12.x (GCC 13.x will fail with "unsupported GNU version")
  - See `INSTALL.md` Section 0 for full compatibility matrix
  - Ensure compatible NVIDIA driver (v580+ recommended for CUDA 12.9)
* **CUDA Libraries:** cuBLAS and cuSOLVER (included with CUDA Toolkit)

### Installing Requirements with Spack

We strongly recommend using [Spack](https://github.com/spack/spack) to manage
these dependencies. A typical Spack installation would look like:

```shell
# Step 1: Install the compiler FIRST
spack install gcc@13.3.0

# Step 2: Register the new compiler with Spack
spack compiler find

# Step 3: Load the compiler
spack load gcc@13.3.0

# Step 4: Install all other dependencies using the new compiler
spack install cmake@3.31.9
spack install intel-oneapi-mkl
spack install googletest

# For GPU support
spack install cuda@12.9.0
```

**IMPORTANT:** The compiler must be installed, registered with `spack compiler find`,
and loaded *before* installing other dependencies. This ensures all packages are
built with the correct compiler version. Spack will automatically use the loaded
compiler for subsequent package installations.

After installation, load the environment:
```shell
spack load gcc@13.3.0
spack load cmake
spack load intel-oneapi-mkl
spack load googletest
spack load cuda@12.9.0  # If GPU support needed
```

**Pro tip:** Add the spack load commands to your `~/.bashrc` to automatically
load the environment in every shell session. Make sure to load the compiler first
in your `.bashrc`.

## 1. Preparing for Installation

### Directory Structure

The install script expects a specific directory structure:

```
~/RandNLA/
├── RandLAPACK/               # Your clone. The script does NOT move it.
└── RandNLA-project/          # Created automatically
    ├── lib/
    │   ├── blaspp/           # Source, fetched at a pinned commit
    │   ├── lapackpp/         # Source, fetched at a pinned commit
    │   └── RandLAPACK -> ../../RandLAPACK   # Symlink to your clone
    ├── install/
    │   ├── blaspp-<backend>-<cpu|cuda>-install
    │   ├── lapackpp-<backend>-<cpu|cuda>-install
    │   ├── random123/
    │   └── RandLAPACK-install
    └── build/                # One build directory per project above
```

Two things worth noting. **Your clone stays where you put it** — earlier
versions of this script relocated it into `lib/`, which broke git worktrees;
`lib/RandLAPACK` is now a symlink. And the dependency install directories carry
the backend and GPU configuration in their names, so an ILP64 MKL build and an
LP64 OpenBLAS build, or a CUDA and a CPU build, cannot be mistaken for each
other or silently reused for one another.

### Initial Setup

1. Create the base directory:
   ```shell
   mkdir -p ~/RandNLA
   cd ~/RandNLA
   ```

2. Clone RandLAPACK repository:
   ```shell
   git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
   cd RandLAPACK
   ```

3. **(Important)** Switch to the correct development branch if needed:
   ```shell
   git checkout <branch-name>
   ```

   **Note:** Always verify with the development team which branch to use for
   the latest GPU support and stability improvements.

## 2. Running the Install Script

### Basic Usage

From inside the `RandLAPACK` directory:

```shell
bash install.sh
```

The script will:
1. Detect if GPU hardware is available on your system and, on a terminal,
   ask whether to build with CUDA support
2. Automatically clone and build all dependencies (or reuse preinstalled
   ones, see the discovery variables below)
3. Build RandLAPACK with appropriate configuration
4. Build test and benchmark executables

Run `bash install.sh --help` for the full option list. The main flags, each
with an environment-variable equivalent:

```
--blas=BACKEND        auto | openblas | mkl | accelerate | custom
                      (default: auto -- Accelerate on macOS, MKL on Linux when
                      MKLROOT is set, otherwise OpenBLAS)
--blas-int=WIDTH      ilp64 | lp64 (default: ilp64 where the backend can
                      provide it; see section 6)
--blas-libraries=L    link line for --blas=custom, used for BLAS and LAPACK
    --gpu / --no-gpu  decide GPU support without asking
--project-dir=DIR     place/locate RandNLA-project at DIR (default:
                      $RANDNLA_PROJECT_DIR if set, else ../RandNLA-project)
--prefix=DIR          install RandLAPACK itself here instead of
                      <project-dir>/install/RandLAPACK-install
-j, --jobs N          parallel build jobs (default: number of cores)
    --fresh           clear build directories first (default: reuse them,
                      so re-running is an incremental rebuild)
    --no-extras       skip the extras project
    --no-benchmarks   skip the benchmark project
    --no-openmp       configure without OpenMP
-y, --yes             assume "yes" for every prompt
    --modify-rc       append RANDNLA_PROJECT_DIR/RANDNLA_PROJECT_GPU_AVAIL
                      exports to your shell config (default: never touch it;
                      the summary prints the lines to add yourself)
    --no-progress     plain one-line-per-step output, no redrawing
```

Extras and benchmarks are built **by default**; `--no-extras` and
`--no-benchmarks` opt out. They need nothing the script has not already built,
so leaving them on costs only time. (RandBLAS's installer makes its `examples/`
opt-in instead, because those pull in dependencies RandBLAS itself does not.)

### Sharing one dependency tree with RandBLAS

Both installers use the same `RandNLA-project` layout and both honour
`RANDNLA_PROJECT_DIR`, so setting it once keeps everything in one place:

```shell
export RANDNLA_PROJECT_DIR=$HOME/RandNLA-project
```

That shares the *location*, not the artifacts. Each project builds its own
BLAS++ into a separately named directory — `blaspp-mkl-cpu-install` here versus
`blaspp-mkl-install` for RandBLAS — deliberately, so neither can overwrite the
other's dependency while its provenance stamp still describes the original.

To genuinely reuse one BLAS++ across both, name it:

```shell
BLASPP_INSTALL_DIR=$RANDNLA_PROJECT_DIR/install/blaspp-mkl-install bash install.sh
```

The install then verifies that choice by compiling, linking and running against
it rather than trusting it.

### Automated Installation (Non-Interactive)

Prompts appear only when stdin is a terminal. Piped and CI runs are already
non-interactive with safe defaults (NVIDIA detected: GPU build; AMD or no
GPU: CPU build), so no `yes |` piping is needed:

```shell
bash install.sh < /dev/null      # or simply: bash install.sh --yes
```

### Installation Logging

All compiler output goes to `<project-dir>/install.log` automatically; the
console shows one line per step, and any failure prints the log path plus
the last lines of the log. There is no need to tee the output yourself.

## 3. What the Script Does

The `install.sh` script performs the following steps automatically:

1. **Creates the project structure** shown in section 1, and initialises the
   RandBLAS submodule. RandBLAS stays a pinned submodule and is not installed
   separately; its own installer exists for people who want RandBLAS alone.

2. **Checks the toolchain** — compiler, CMake 3.21+, Git — reporting everything
   missing at once rather than failing on the first item.

3. **Builds BLAS++** at a pinned commit, with the selected backend and integer
   width, then **reads back the width it actually built** (section 6).

4. **Builds LAPACK++** at a pinned commit against that BLAS++.

5. **Installs Random123** (header-only) at a pinned tag.

6. **Builds and installs RandLAPACK**, then **verifies the result by running
   it**: a small program is compiled, linked and executed against the finished
   install, checking a BLAS++ `gemm` and a LAPACK++ `gesdd` numerically.

   This step is not ceremony. A configuration that merely *configures* can
   still be broken in ways nothing catches by inspection — most importantly, if
   BLAS++'s headers were built for one integer width while the library actually
   loaded uses another, the guards inside BLAS++ compile out and the symptom is
   an absurd workspace size and a run that never finishes rather than an error.
   Only executing something finds that.

7. **Builds the extras and benchmark projects**, unless `--no-extras` or
   `--no-benchmarks`.

Every dependency is fetched at an immutable ref and stamped with its
provenance, so it is reused only when it came from the same source *in the same
configuration* — backend, integer width and GPU setting all count. Switching
`--no-gpu` to `--gpu` therefore rebuilds BLAS++ rather than silently reusing a
CPU-only one.

## 4. Verifying the Installation

### Running Tests

After installation completes, verify everything works correctly:

```shell
cd ~/RandNLA/RandNLA-project/build/RandLAPACK-build
ctest
```

This runs the complete test suite (456 tests). Expected output:
```
99% tests passed, 1 tests failed out of 456
Total Test time (real) = 124.62 sec
```

**Note:** Some test failures are known and acceptable in development branches.
Consult the development team if you see unexpected failures.

### Running GPU Tests Only

If you enabled GPU support, test GPU functionality specifically:

```shell
./bin/RandLAPACK_tests_gpu
```

Expected output: 13-14 GPU tests should pass within 15-20 seconds.

## 5. Working with the Installed Project

### Key File Locations

After installation:

* **RandLAPACK library:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/libRandLAPACK.a`
* **Headers:** `~/RandNLA/RandNLA-project/lib/RandLAPACK/RandLAPACK/`
* **Tests:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/bin/RandLAPACK_tests*`
* **Benchmarks:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/bin/RandLAPACK_bench*`
* **CMake config:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/RandLAPACKConfig.cmake`

### Recompiling After Code Changes

If you modify RandLAPACK source code:

```shell
cd ~/RandNLA/RandNLA-project/build/RandLAPACK-build
source ~/.bashrc  # Ensures environment is loaded
make -j
```

**Important:** Always source your `.bashrc` (or equivalent environment setup)
before running `make` to ensure CUDA libraries and other dependencies are in
your `LD_LIBRARY_PATH`.

### Using RandLAPACK in Your Own Projects

See Section 4 of `INSTALL.md` for details on linking RandLAPACK to external
CMake projects. You'll need to specify:

```cmake
-Dblaspp_DIR=~/RandNLA/RandNLA-project/lib/blaspp/lib/cmake/blaspp
-Dlapackpp_DIR=~/RandNLA/RandNLA-project/lib/lapackpp/lib/cmake/lapackpp
-DRandBLAS_DIR=~/RandNLA/RandNLA-project/build/RandLAPACK-build/RandBLAS
-DRandLAPACK_DIR=~/RandNLA/RandNLA-project/build/RandLAPACK-build
```

---

## 6. Integer width, backends, and tested configurations

### 6.1 What we test

Every row below corresponds to a CI lane, so this is a statement about what is
exercised on each commit rather than what ought to work. Anything absent may
well work; it is simply untested.

| OS | Compiler | BLAS backend | Integer width | GPU | OpenMP |
|---|---|---|---|---|---|
| Ubuntu (latest) | gcc | OpenBLAS | LP64 | none | yes |
| Ubuntu (latest) | gcc | oneMKL | ILP64 | none | yes |
| Ubuntu (latest) | clang | OpenBLAS | LP64 | none | yes |
| macOS 14/15 | Apple Clang | Accelerate (new interface) | ILP64 | none | no (see below) |
| Windows | MSVC | oneMKL | ILP64 | none | yes (`/openmp:llvm`) |

The installer lanes additionally cover a fresh install, an idempotent re-run,
and dependency discovery through `BLASPP_INSTALL_DIR` and friends.

Compiler floor: **gcc >= 13**, because RandBLAS (vendored as a submodule) uses
C++20 concepts. CMake 3.21 or later on every platform. Apple Clang ships no
OpenMP runtime, so a macOS build is single-threaded unless you install
Homebrew's `libomp`, which the installer will use when present.

CUDA builds are not covered by CI — there is no GPU runner — so `--gpu` is
exercised locally only. CUDA 12.9 with gcc 13.3 is the reference combination.

### 6.2 Which integer width you get, and why

A BLAS comes in one of two flavours: **LP64** uses 32-bit integers for matrix
dimensions, **ILP64** uses 64-bit. The installer prefers ILP64 wherever the
backend can genuinely provide it:

| `--blas=` | Width you get | Why |
|---|---|---|
| `mkl` | ILP64 | `mkl_intel_ilp64` is a separate library, so requesting it actually selects it |
| `openblas` | LP64, with a warning | there is only `-lopenblas`, so the request selects nothing |
| `accelerate` | ILP64 on macOS >= 13.3 | Apple's new interface ships ILP64 in the framework, and BLAS++ selects it (`icl-utk-edu/blaspp#134`); older macOS falls back to LP64 with a warning |
| `custom` | whatever you pass | you named the library, so its width is yours to state |

The OpenBLAS row deserves explaining, because it is counter-intuitive. BLAS++
probes `int32` before `int64`, and `blas_int` only filters which *library names*
to consider. With one candidate name, a plain LP64 OpenBLAS passes the `int32`
probe and is accepted — so a successful `blas_int=int64` configure proves
nothing. The installer therefore reads the width back out of BLAS++'s generated
`blas/defines.h` after building, and reports what was actually produced.

ILP64 OpenBLAS **does** exist (`libopenblas64`, from Debian/Ubuntu's
`libopenblas64-dev` or Fedora's `openblas64`); BLAS++ just never looks for it.
Until that is fixed upstream, name it explicitly:

```shell
bash install.sh --blas=custom --blas-int=ilp64 \
  --blas-libraries=/usr/lib/x86_64-linux-gnu/libopenblas64.so
```

### 6.3 Does LP64 limit RandLAPACK?

Rarely, and it fails loudly rather than silently. RandLAPACK is `int64_t`
throughout, and BLAS++/LAPACK++ **throw** rather than truncate when a value
exceeds the BLAS's range (`to_blas_int`, `to_lapack_int`), naming the offending
argument. The guard is on individual dimensions and leading dimensions, not
element counts — the BLAS never receives `m*n` — so a 100,000 x 100,000 matrix
is fine under LP64. The limit bites only past roughly 2.1 billion in a *single*
dimension, and for sparse work with `nnz > 2^31`.

The case that *is* silent is different, and it is why this installer runs a
program rather than only linking one: if BLAS++'s headers were built for one
width while the library actually loaded uses the other, the guard compiles out
entirely and 64-bit values reach routines reading 32 bits. The symptom is not
wrong numbers but nonsense control values — a misread workspace query becomes an
absurd `lwork`, and the run dies in allocation or never finishes. The
verification step exists to catch that at install time.

### 6.4 macOS: Accelerate, through Apple's new interface only

The macOS default is Accelerate, selected through Apple's **new** interface
(macOS >= 13.3, `ACCELERATE_NEW_LAPACK`, LAPACK 3.12 on current SDKs), which
the pinned BLAS++/LAPACK++ support (`icl-utk-edu/blaspp#134`,
`icl-utk-edu/lapackpp#88`). No Homebrew BLAS is needed for the default build.

Apple's **legacy** interface (LAPACK 3.2.1, from 2009) is a different story: its
divide-and-conquer `gesdd` returns wrong results on Apple Silicon, and RandLAPACK
calls `gesdd` in `rl_rsvd.hh`, `rl_abrik.hh`, `rl_revd2.hh`,
`rl_preconditioners.hh` and `rl_util.hh`. The installer therefore refuses a
build in which BLAS++ silently fell back to the legacy interface (checked
against BLAS++'s generated `defines.h` right after the build, and again
numerically by the `gesdd` conftest). On macOS older than 13.3, use
`--blas=openblas` (after `brew install openblas`).

---

## Building and Running GPU Benchmarks

GPU benchmarks are in the `benchmark/` directory and must be built separately from the main RandLAPACK project.

### Prerequisites

- RandLAPACK must already be built and installed with CUDA support (`-DRequireCUDA=ON`)
- CUDA Toolkit must be available on your system
- GPU hardware must be available

### Building GPU Benchmarks

Navigate to the benchmark directory and build as a standalone project:

```shell
cd ~/RandNLA/RandNLA-project/lib/RandLAPACK/benchmark
mkdir -p build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DRandLAPACK_DIR=~/RandNLA/RandNLA-project/install/RandLAPACK-install/lib/cmake/RandLAPACK \
  ..
make -j
```

**Note:** Adjust the `RandLAPACK_DIR` path to match your installation location.

### Running GPU Benchmarks

#### BQRRP GPU Benchmark

The BQRRP GPU benchmark supports two modes:

**Block size sweep** (default):
```shell
./BQRRP_GPU_benchmark block_size [matrix_size] [profile_runtime] [run_qrf]
```

Examples:
```shell
# Run with default settings (16384x16384 matrix)
./BQRRP_GPU_benchmark block_size

# Run with 32768x32768 matrix
./BQRRP_GPU_benchmark block_size 32768

# Run with profiling enabled and QRF comparison
./BQRRP_GPU_benchmark block_size 16384 1 1
```

**Matrix size sweep**:
```shell
./BQRRP_GPU_benchmark mat_size [profile_runtime] [run_qrf]
```

Examples:
```shell
# Run with default settings
./BQRRP_GPU_benchmark mat_size

# Run with profiling disabled but QRF comparison enabled
./BQRRP_GPU_benchmark mat_size 0 1
```

### Output Files

The benchmarks generate text files with timing results in the current directory:

- `_BQRRP_GPU_speed_comparisons_block_size_*.txt` - Speed comparison results for block size sweep
- `BQRRP_GPU_speed_comparisons_mat_size_*.txt` - Speed comparison results for matrix size sweep
- `_BQRRP_GPU_runtime_breakdown_qrf_*.txt` - Detailed profiling with QRF (if profiling enabled)
- `_BQRRP_GPU_runtime_breakdown_cholqr_*.txt` - Detailed profiling with CholQR (if profiling enabled)

**Last Updated:** 2025-11-26
**Tested With:**
- GCC 13.3.0
- CMake 3.31.9
- CUDA 12.9.0
- Intel MKL 2025.0.3
- Ubuntu 22.04 / WSL2
