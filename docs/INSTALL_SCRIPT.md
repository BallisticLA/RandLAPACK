# Using RandLAPACK's Automated Install Script

This guide explains how to use the `install.sh` script to automatically install
RandLAPACK and all of its dependencies (BLAS++, LAPACK++, Random123) with a
single command.

**When to use this guide:** Use this automated installation method if you want
a quick, streamlined setup process. If you need fine-grained control over
dependency configurations, refer to RandLAPACK's `docs/INSTALL.md` instead.

## 0. Software Requirements

Before running the install script, ensure you have the following software
available on your system:

### Essential Requirements
* **C++ Compiler:** GNU GCC 13.3.0 is the reference version (any C++20
  compiler can work; the script warns on other versions and continues)
* **CMake:** Version 3.21 or higher (recent releases recommended)
* **BLAS/LAPACK Library:** Intel MKL 2022 or higher recommended on Linux;
  on macOS, Homebrew OpenBLAS and libomp are required
  (`brew install openblas libomp`)
* **GoogleTest:** (Optional but recommended) For running RandLAPACK tests

### GPU Support Requirements (Optional)
* **CUDA Toolkit:** Version 12.4.1 or higher
  - **Recommended:** CUDA 12.9.0 + GCC 13.3.0
  - **IMPORTANT:** CUDA versions have strict GCC compatibility requirements:
    - CUDA 12.9.0: Compatible with GCC 13.x
    - CUDA 12.4.1: Compatible with GCC 13.x
    - CUDA 12.2.1: Requires GCC 12.x or older ("unsupported GNU version" otherwise)
  - See `INSTALL.md` Section 0 for the full compatibility matrix
  - Ensure a compatible NVIDIA driver (v580+ recommended for CUDA 12.9)
* **CUDA Libraries:** cuBLAS and cuSOLVER (included with the CUDA Toolkit)

### Installing Requirements with Spack

We recommend [Spack](https://github.com/spack/spack) for managing these
dependencies. A typical installation:

```shell
# Step 1: Install the compiler FIRST
spack install gcc@13.3.0

# Step 2: Register the new compiler with Spack
spack compiler find

# Step 3: Load the compiler
spack load gcc@13.3.0

# Step 4: Install all other dependencies using the new compiler
spack install cmake
spack install intel-oneapi-mkl
spack install googletest

# For GPU support
spack install cuda@12.9.0
```

**IMPORTANT:** The compiler must be installed, registered with
`spack compiler find`, and loaded *before* installing other dependencies, so
every package builds with the intended compiler.

After installation, load the environment (and consider adding these commands
to your shell startup file, compiler first):

```shell
spack load gcc@13.3.0
spack load cmake
spack load intel-oneapi-mkl
spack load googletest
spack load cuda@12.9.0  # If GPU support needed
```

## 1. Preparing for Installation

### Directory Structure

On its first run the script moves your clone into a project layout that it
creates next to the clone:

```
<parent directory>/
|-- RandLAPACK/               # your clone (moved into the layout on first run)
`-- RandNLA-project/          # created by the script
    |-- lib/
    |   |-- RandLAPACK/       # the clone, after the move
    |   |-- blaspp/           # cloned + built by the script
    |   `-- lapackpp/         # cloned + built by the script
    |-- install/
    |   |-- RandLAPACK-install/   # installed headers + CMake config
    |   |-- blaspp-install/
    |   |-- lapackpp-install/
    |   `-- random123/            # header-only clone
    |-- build/                # one build directory per project above
    `-- install.log           # full build log of the latest run
```

Re-running the script from the moved location
(`RandNLA-project/lib/RandLAPACK/install.sh`) detects the layout, reuses the
dependency installs and build directories, and performs an incremental
rebuild.

### Initial Setup

```shell
mkdir -p ~/RandNLA
cd ~/RandNLA
git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
cd RandLAPACK
```

If you need a development branch, check it out before running the script.

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
-y, --yes             assume "yes" for every prompt
    --gpu / --no-gpu  decide GPU support without asking
-j, --jobs <N>        parallel build jobs (default: number of cores)
    --fresh           clear build directories first (default: reuse them,
                      so re-running is an incremental rebuild)
    --modify-rc       append RANDNLA_PROJECT_DIR/RANDNLA_PROJECT_GPU_AVAIL
                      exports to your shell config (default: never touch it;
                      the summary prints the lines to add yourself)
    --project-dir <D> place/locate RandNLA-project at D
```

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

### Reusing Preinstalled Dependencies (Discovery)

If you already have BLAS++, LAPACK++, or Random123 installed, point the
script at them and it will skip those builds:

```shell
BLASPP_INSTALL_DIR=/path/to/blaspp-install \
LAPACKPP_INSTALL_DIR=/path/to/lapackpp-install \
RANDOM123_INSTALL_DIR=/path/to/random123 \
bash install.sh
```

Each variable must point at an install root (the directory containing
`lib*/cmake/<pkg>/` or, for Random123, `include/Random123/`). Dependencies
the script itself installed on a previous run are reused automatically.
RandBLAS is deliberately not covered: it stays a pinned git submodule (see
`INSTALL.md`, section "RandBLAS is a pinned submodule").

## 3. What the Script Does

In order (steps are numbered on the console and logged to `install.log`):

1. **Toolchain and GPU decision.** GCC/NVCC versions are checked (warn and
   continue by default), GPU hardware is detected, and the CUDA choice is
   made from flags, prompts, or non-interactive defaults.
2. **Project layout.** Creates `RandNLA-project/{lib,install,build}` next to
   the clone and moves the clone into `lib/RandLAPACK` (first run only).
3. **Dependency discovery.** Reuses externally provided or previously built
   BLAS++/LAPACK++/Random123 installs; clones what is missing.
4. **BLAS++** configure, build, install (into `install/blaspp-install/`).
5. **LAPACK++** configure, build, install (into `install/lapackpp-install/`),
   against the BLAS++ from the previous step.
6. **RandLAPACK** configure, build, install (headers and CMake config into
   `install/RandLAPACK-install/`), with the test suite enabled.
7. **extras** and **benchmarks**: two standalone downstream projects,
   configured against the *installed* RandLAPACK and built (executables stay
   in their build directories).

## 4. Verifying the Installation

Run the test suite from anywhere:

```shell
ctest --test-dir <parent>/RandNLA-project/build/RandLAPACK-build
```

The suite should pass (the exact test count grows over time; CI runs this
same suite on every change). If you enabled GPU support, the GPU tests run
as part of the same suite; to run only them:

```shell
<parent>/RandNLA-project/build/RandLAPACK-build/bin/RandLAPACK_tests_gpu
```

## 5. Working with the Installed Project

### Key File Locations

* **Installed headers + CMake config:**
  `RandNLA-project/install/RandLAPACK-install/` (the config file lives under
  `lib*/cmake/RandLAPACK/`; RandLAPACK is header-only, so there is no
  library archive)
* **Test executables:** `RandNLA-project/build/RandLAPACK-build/bin/`
* **Extras executables:** `RandNLA-project/build/extras-build/`
* **Benchmark executables:** `RandNLA-project/build/benchmark-build/`
  (see `benchmark/README.md` for how to run them)

### Recompiling After Code Changes

```shell
cd <parent>/RandNLA-project/build/RandLAPACK-build
make -j && make install
```

The `make install` matters: RandLAPACK is header-only, and the extras,
benchmarks, and your own projects consume the *installed* headers, so
changes only reach them after an install. Alternatively just re-run
`bash install.sh` from `lib/RandLAPACK/`; re-runs are incremental.

### Using RandLAPACK in Your Own Projects

See Section 4 of `INSTALL.md`. The paths produced by this script are:

```
-Dblaspp_DIR=<parent>/RandNLA-project/install/blaspp-install/lib/cmake/blaspp
-Dlapackpp_DIR=<parent>/RandNLA-project/install/lapackpp-install/lib/cmake/lapackpp
-DRandLAPACK_DIR=<parent>/RandNLA-project/install/RandLAPACK-install/lib/cmake/RandLAPACK
```

(depending on the platform, `lib` may be `lib64`; the installer's final
summary prints the exact `RandLAPACK_DIR` for your machine).

## 6. Benchmarks

Benchmark executables are built automatically into
`RandNLA-project/build/benchmark-build/`. Usage, including the GPU
benchmarks and their output formats, is documented in
[`benchmark/README.md`](../benchmark/README.md).
