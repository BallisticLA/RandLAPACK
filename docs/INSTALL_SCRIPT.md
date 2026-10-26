# Using RandLAPACK's Automated Install Script

The installer scripts live in the `installers/` directory (`installers/install.sh`
for Linux/macOS, `installers/install.ps1` for native Windows); a small wrapper is
kept at the repository root so `bash install.sh` keeps working.

This guide explains how to use the `install.sh` script to automatically install
RandLAPACK and all of its dependencies (BLAS++, LAPACK++, Random123) with a
single command.

**When to use this guide:** Use this automated installation method if you want
a quick, streamlined setup process. If you need fine-grained control over
dependency configurations, refer to [INSTALL.md](INSTALL.md) instead.

> **Windows users:** this document describes the Linux/macOS installer
> (`install.sh`). The native Windows companion is `installers\install.ps1`,
> documented in [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md): prerequisites,
> every option, backend selection, and troubleshooting.

## 0. Software Requirements

Before running the install script, ensure you have the following software
available on your system:

### Essential Requirements
* **C++ Compiler:** a C++20 compiler; GNU GCC 13.3.0 is the reference
  version (the script notes other versions and continues, and warns below
  GCC 13, which RandBLAS's C++20 concepts need)
* **CMake:** Version 3.21 or higher (the project's CMake floor; recent
  releases recommended)
* **BLAS/LAPACK Library:** on Linux, OpenBLAS or Intel oneMKL (2022 or
  higher; chosen by default when `MKLROOT` is set); on macOS, Apple's
  Accelerate (macOS 13.3 or later), which needs nothing installed. Section 6
  covers the choices.
* **GoogleTest:** (Optional but recommended) For running RandLAPACK tests

### GPU Support Requirements (Optional)
* **CUDA Toolkit:** Version 12.4.1 or higher
  - **Recommended:** CUDA 12.9.0 + GCC 13.3.0
  - **IMPORTANT:** CUDA versions have strict GCC compatibility requirements:
    - CUDA 12.9.0: Compatible with GCC 13.x
    - CUDA 12.4.1: Compatible with GCC 13.x
    - CUDA 12.2.1: Requires GCC 12.x or older ("unsupported GNU version" otherwise)
  - See [INSTALL.md](INSTALL.md#cuda-and-gcc-versions) for the full compatibility matrix
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
spack install cmake@3.31.9
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
    ├── build/                # One build directory per project above
    └── install.log           # Full build log, appended on every run
```

Two things worth noting. **Your clone stays where you put it**: earlier
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

3. If you need a development branch, check it out before running the script:
   ```shell
   git checkout <branch-name>
   ```

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
4. Build the test (if GoogleTest is found), extras and benchmark executables

Run `bash install.sh --help` for the full option list. The main flags, each
with an environment-variable equivalent:

```
--blas=BACKEND        auto | openblas | mkl | accelerate | custom
                      (default: auto, which picks Accelerate on macOS, MKL
                      on Linux when MKLROOT is set, otherwise OpenBLAS)
--blas-int=WIDTH      ilp64 | lp64 (default: ilp64 where the backend can
                      provide it; see section 6)
--blas-libraries=L    link line for --blas=custom, used for BLAS and LAPACK
    --gpu / --no-gpu  decide GPU support without asking
--project-dir=DIR     place/locate RandNLA-project at DIR (default:
                      $RANDNLA_PROJECT_DIR if set, else ../RandNLA-project)
--prefix=DIR          install RandLAPACK itself here instead of
                      <project-dir>/install/RandLAPACK-install
-j, --jobs N          parallel build jobs (default: number of cores)
    --fresh           clear the build directories and rebuild the
                      dependencies the script built itself (default: reuse
                      what matches, so re-running is an incremental rebuild)
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

### Reusing preinstalled dependencies (discovery)

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
[INSTALL.md, "RandBLAS is a pinned submodule"](INSTALL.md#randblas-is-a-pinned-submodule)).

### Sharing one dependency tree with RandBLAS

Both installers use the same `RandNLA-project` layout and both honour
`RANDNLA_PROJECT_DIR`, so setting it once keeps everything in one place:

```shell
export RANDNLA_PROJECT_DIR=$HOME/RandNLA-project
```

That shares the *location*, not the artifacts. Each project builds its own
BLAS++ into a separately named directory (`blaspp-mkl-cpu-install` here versus
`blaspp-mkl-install` for RandBLAS), deliberately, so neither can overwrite the
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

1. **Checks the toolchain** (compiler, CMake 3.21+, Git), reporting everything
   missing at once rather than failing on the first item.

2. **Creates the project structure** shown in section 1, and initialises the
   RandBLAS submodule. RandBLAS stays a pinned submodule and is not installed
   separately; its own installer exists for people who want RandBLAS alone.

3. **Builds BLAS++** at a pinned commit, with the selected backend and integer
   width, then **reads back the width it actually built** (section 6).

4. **Builds LAPACK++** at a pinned commit against that BLAS++.

5. **Installs Random123** (header-only) at a pinned tag.

6. **Builds and installs RandLAPACK**, then **verifies the result by running
   it**: a small program is compiled, linked and executed against the finished
   install, checking a BLAS++ `gemm` and a LAPACK++ `gesdd` numerically.

   This step is not ceremony. A configuration that merely *configures* can
   still be broken in ways nothing catches by inspection. Most importantly, if
   BLAS++'s headers were built for one integer width while the library actually
   loaded uses another, the guards inside BLAS++ compile out and the symptom is
   an absurd workspace size and a run that never finishes rather than an error.
   Only executing something finds that.

7. **Builds the extras and benchmark projects**, unless `--no-extras` or
   `--no-benchmarks`. Their first configure downloads Eigen,
   fast_matrix_market and (for the benchmarks) Spectra, each a pinned,
   checksummed archive, into `<project-dir>/build/fetchcontent-cache/`, which
   the two projects share; `--fresh` deletes it with the rest of `build/`.

Every dependency is fetched at an immutable ref and stamped with its
provenance, so it is reused only when it came from the same source *in the same
configuration*: backend, integer width and GPU setting all count. Switching
`--no-gpu` to `--gpu` therefore rebuilds BLAS++ rather than silently reusing a
CPU-only one.

BLAS++ and LAPACK++ come from the BallisticLA forks
([blaspp](https://github.com/BallisticLA/blaspp),
[lapackpp](https://github.com/BallisticLA/lapackpp)), at the commits pinned in
[`installers/dependency-pins.env`](../installers/dependency-pins.env) and tagged
`ballistic-2026.10.22`. The forks carry fixes that are not yet in an upstream
release; each fork's `BALLISTIC.md` lists them, and issue
[#204](https://github.com/BallisticLA/RandLAPACK/issues/204)
tracks the return to upstream.

## 4. Verifying the Installation

### Running Tests

After installation completes, verify everything works correctly:

```shell
cd ~/RandNLA/RandNLA-project/build/RandLAPACK-build
ctest
```

The suite should pass (the exact test count grows over time; CI runs this
same suite on every pull request).

### Running GPU Tests Only

If you enabled GPU support, the GPU tests run as part of the same suite; to
run only them, from that same build directory:

```shell
./bin/RandLAPACK_tests_gpu
```

## 5. Working with the Installed Project

### Key File Locations

After installation:

* **Installed headers and CMake config:**
  `~/RandNLA/RandNLA-project/install/RandLAPACK-install/` (or the directory
  given to `--prefix`). The config file lives under `lib*/cmake/RandLAPACK/`;
  RandLAPACK is header-only, so there is no library archive.
* **Test executables:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/bin/`
* **Extras test executable:**
  `~/RandNLA/RandNLA-project/build/extras-build/test/RandLAPACK_extras_tests`
* **Benchmark executables:** `~/RandNLA/RandNLA-project/build/benchmark-build/`
  (see section 7)

### Recompiling After Code Changes

If you modify RandLAPACK source code:

```shell
cd ~/RandNLA/RandNLA-project/build/RandLAPACK-build
cmake --build . -j --target install
```

The install step matters: RandLAPACK is header-only, and the extras,
benchmarks, and your own projects consume the *installed* headers, so
changes reach them only after an install. Alternatively, re-run
`bash install.sh` from your clone; re-runs are incremental. Either way, use
a shell with the same environment you installed from (compiler, CMake, and
CUDA if you enabled it).

### Using RandLAPACK in Your Own Projects

See Section 5 of [INSTALL.md](INSTALL.md#5-using-randlapack-in-other-projects) for details on linking RandLAPACK to external
CMake projects. The paths produced by this script are:

```
-Dblaspp_DIR=~/RandNLA/RandNLA-project/install/blaspp-<backend>-<cpu|cuda>-install/lib/cmake/blaspp
-Dlapackpp_DIR=~/RandNLA/RandNLA-project/install/lapackpp-<backend>-<cpu|cuda>-install/lib/cmake/lapackpp
-DRandLAPACK_DIR=~/RandNLA/RandNLA-project/install/RandLAPACK-install/lib/cmake/RandLAPACK
```

Depending on the platform, `lib` may be `lib64`; the installer's final
summary prints the exact `RandLAPACK_DIR` for your machine. RandBLAS is
installed next to RandLAPACK and found through RandLAPACK's own CMake
config, so it needs no flag of its own.

---

## 6. Integer width, backends, and tested configurations

### 6.1 What we test

Every row below corresponds to a CI lane, so this is a statement about what is
exercised on each commit rather than what ought to work. Anything absent may
well work; it is simply untested.

| OS | Compiler | BLAS backend | Integer width | GPU | OpenMP |
|---|---|---|---|---|---|
| Ubuntu (latest) | gcc | OpenBLAS | LP64 | none | yes |
| Ubuntu (latest) | gcc | OpenBLAS (`libopenblas64` only) | ILP64 | none | yes |
| macOS (`macos-latest`) | Apple Clang | Accelerate (new interface) | ILP64 | none | yes, Homebrew `libomp` |
| macOS (`macos-latest`) | Apple Clang | Accelerate (new interface), `--blas-int=lp64` | LP64 | none | yes, Homebrew `libomp`; no in core-macos |
| macOS (`macos-latest`) | Apple Clang | Homebrew OpenBLAS (`--blas=openblas`) | LP64 | none | yes, Homebrew `libomp` |
| Windows | MSVC | oneMKL | ILP64 | none | yes (`/openmp:llvm`), and a serial lane |
| Windows | MSVC | OpenBLAS | LP64 | none | no |

The installer lanes additionally cover a fresh install and an idempotent
re-run, and on Linux dependency discovery through `BLASPP_INSTALL_DIR` and
friends.

Compiler floor: **gcc >= 13**, because RandBLAS (vendored as a submodule) uses
C++20 concepts. CMake 3.21 or later on every platform. Apple Clang ships no
OpenMP runtime, so a macOS build is single-threaded unless you install
Homebrew's `libomp`, which the installer will use when present.

No CI job builds or runs the CUDA code
([#215](https://github.com/BallisticLA/RandLAPACK/issues/215)), so `--gpu` is
exercised locally only. CUDA 12.9 with gcc 13.3 is the reference combination.

### 6.2 Which integer width you get, and why

A BLAS comes in one of two flavours: **LP64** uses 32-bit integers for matrix
dimensions, **ILP64** uses 64-bit. The installer prefers ILP64 wherever the
backend can genuinely provide it:

| `--blas=` | Width you get | Why |
|---|---|---|
| `mkl` | ILP64 | `mkl_intel_ilp64` is a separate library, so requesting it actually selects it |
| `openblas` | LP64 with a warning, or ILP64 when `libopenblas64` is the only OpenBLAS installed | BLAS++ tries plain `-lopenblas` first; see below |
| `accelerate` | ILP64 | Apple's new interface (macOS >= 13.3) ships ILP64 in the framework; BLAS++ selects it (`icl-utk-edu/blaspp#134`) and LAPACK++ compiles against it (`icl-utk-edu/lapackpp#90`). Older macOS lacks the new interface entirely, and the installer refuses the legacy one (see 6.4) |
| `custom` | whatever you pass | you named the library, so its width is yours to state |

The OpenBLAS row deserves explaining, because it is counter-intuitive. The
pinned BLAS++ looks for plain `-lopenblas` first and for the ILP64
`-lopenblas64` second (`icl-utk-edu/blaspp#129`). Its `int64` BLAS probe cannot
tell the widths apart, so an LP64 `libopenblas` passes it by accident; the
LAPACK probe then catches the mismatch and fails the whole configure, before
`-lopenblas64` is ever tried. The installer then falls back to LP64 and says
so. With `libopenblas64` as the only OpenBLAS installed (on Debian/Ubuntu,
`libopenblas64-dev` without `libopenblas-dev`), `--blas=openblas` gives ILP64.
Either way the installer reads the width back out of BLAS++'s generated
`blas/defines.h` after building, and reports what was actually produced.

With both installed, or with an ILP64 OpenBLAS under another name, name it
explicitly:

```shell
bash install.sh --blas=custom --blas-int=ilp64 \
  --blas-libraries=/usr/lib/x86_64-linux-gnu/libopenblas64.so
```

An OpenBLAS whose symbols carry a suffix, such as conda-forge's
`libopenblas64_` (`dgemm_64_`), needs BLAS++'s `blas_symbol_suffix` option,
which the installer does not set. Build BLAS++ and LAPACK++ by hand
([INSTALL.md, "ILP64 OpenBLAS"](INSTALL.md#ilp64-openblas)) and point the
installer at them with `BLASPP_INSTALL_DIR` and `LAPACKPP_INSTALL_DIR`.

### 6.3 Does LP64 limit RandLAPACK?

Rarely, and it fails loudly rather than silently. RandLAPACK is `int64_t`
throughout, and BLAS++/LAPACK++ **throw** rather than truncate when a value
exceeds the BLAS's range (`to_blas_int`, `to_lapack_int`), naming the offending
argument. The guard is on individual dimensions and leading dimensions, not
element counts (the BLAS never receives `m*n`), so a 100,000 x 100,000 matrix
is fine under LP64. The limit bites only past roughly 2.1 billion in a *single*
dimension, and for sparse work with `nnz > 2^31`.

The case that *is* silent is different, and it is why this installer runs a
program rather than only linking one: if BLAS++'s headers were built for one
width while the library actually loaded uses the other, the guard compiles out
entirely and 64-bit values reach routines reading 32 bits. The symptom is not
wrong numbers but nonsense control values: a misread workspace query becomes an
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
`rl_preconditioners.hh`, `rl_util.hh` and `rl_test_utils.hh`. The installer therefore refuses a
build in which BLAS++ silently fell back to the legacy interface (checked
against BLAS++'s generated `defines.h` right after the build, and again
numerically by the `gesdd` conftest). On macOS older than 13.3, use
`--blas=openblas` (after `brew install openblas`).

## 7. Benchmarks

Benchmark executables are built automatically into
`~/RandNLA/RandNLA-project/build/benchmark-build/`. Usage, including the GPU
benchmarks and their output formats, is documented in
[`benchmark/README.md`](../benchmark/README.md).
