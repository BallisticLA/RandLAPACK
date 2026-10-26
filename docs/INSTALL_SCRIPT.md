# Installing with `install.sh` (Linux and macOS)

`install.sh` builds RandLAPACK and everything around it with one command: its
dependencies (BLAS++, LAPACK++, Random123), the `extras/` and `benchmark/`
projects, and, when GoogleTest is found, the test suite. On native Windows use
`installers\install.ps1` instead ([INSTALL_WINDOWS.md](INSTALL_WINDOWS.md)).
To build each piece by hand, see [INSTALL.md](INSTALL.md), which also
explains the [key terms](INSTALL.md#1-key-terms) used here.

The script lives at `installers/install.sh`; a small wrapper at the
repository root keeps `bash install.sh` working.

## 1. Quick start

1. Install the requirements (section 2).
2. Clone RandLAPACK and run the installer from inside the clone:
   ```shell
   mkdir -p ~/RandNLA
   cd ~/RandNLA
   git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
   cd RandLAPACK
   bash install.sh
   ```
   To build a development branch, run `git checkout <branch-name>` before
   the script.
3. Check the install by running the test suite:
   ```shell
   OMP_NUM_THREADS=1 ctest --test-dir ~/RandNLA/RandNLA-project/build/RandLAPACK-build -j 4
   ```
   The suite should pass, typically in a minute or two
   ([INSTALL.md, section 4](INSTALL.md#4-building-and-installing-randlapack)
   explains the settings). CI runs this same suite on every pull request; the
   exact test count grows over time.

Good to know:

- On a terminal the script may ask questions, for example whether to build
  with CUDA support when it detects GPU hardware. Section 3 shows how to skip
  them, and how to reuse dependencies you already installed.
- Everything goes into `~/RandNLA/RandNLA-project/`, next to your clone
  (section 4). Re-running the script is an incremental rebuild.

## 2. Requirements

- **A C++20 compiler.** GCC 13.3.0 is the reference version, and GCC 13 the
  floor: GCC 11 and 12 do not compile the current version. The script notes
  other versions and continues, and warns below GCC 13.
- **CMake** 3.21 or later (the project's floor; recent releases
  recommended).
- **Git.**
- **A backend**, the BLAS/LAPACK library that does the arithmetic:
  - on Linux, OpenBLAS or Intel oneMKL 2022 or later. oneMKL is chosen by
    default when `MKLROOT` is set, which loading oneAPI's environment does
    ([INSTALL.md, "Intel oneMKL"](INSTALL.md#intel-onemkl-recommended));
  - on macOS, Apple's Accelerate (macOS 13.3 or later), which needs nothing
    installed.

  Section 6 covers the choices.
- **GoogleTest** (optional but recommended). Without it, no tests are built.
- **For GPU support (optional):**
  - the CUDA Toolkit, 12.4.1 or later, which includes cuBLAS and cuSOLVER;
    CUDA 12.9.0 with GCC 13.3.0 is recommended;
  - a compatible NVIDIA driver (v580 or later recommended for CUDA 12.9);
  - a GCC version your CUDA release accepts: CUDA has strict GCC
    requirements, listed in the
    [compatibility table](INSTALL.md#cuda-and-gcc-versions).
  - **Known issue:** in the current version the GPU benchmark does not
    compile, so a GPU build also needs `--no-benchmarks`. That covers
    `--gpu`, answering yes to the installer's GPU question, and an unattended
    run on a machine with an NVIDIA GPU, which builds with GPU support by
    default (section 3).

On Ubuntu 24.04 and macOS, the [README](../README.md#install) gives the
package-manager command that installs everything but CUDA.

### Installing the requirements with Spack

Elsewhere, we recommend [Spack](https://github.com/spack/spack). Install,
register and load the compiler **before** anything else, so that every
package builds with it:

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

After installation, load the environment, compiler first. Consider adding
these lines to your shell startup file:

```shell
spack load gcc@13.3.0
spack load cmake
spack load intel-oneapi-mkl
spack load googletest
spack load cuda@12.9.0  # If GPU support needed
```

## 3. Options

Run `bash install.sh --help` for the full list. Each flag has an
environment-variable equivalent. The main flags:

| Flag | Effect | Default |
|---|---|---|
| `--blas=BACKEND` | backend: `auto`, `openblas`, `mkl`, `accelerate` or `custom` | `auto`: Accelerate on macOS; on Linux, MKL when `MKLROOT` is set, otherwise OpenBLAS |
| `--blas-int=WIDTH` | integer width: `ilp64` or `lp64` (section 6) | `ilp64` where the backend can provide it |
| `--blas-libraries=L` | link line for `--blas=custom`, used for both BLAS and LAPACK | |
| `--gpu`, `--no-gpu` | decide GPU support without asking | |
| `--project-dir=DIR` | place or find `RandNLA-project` at `DIR` | `$RANDNLA_PROJECT_DIR` if set, else `../RandNLA-project` |
| `--prefix=DIR` | install RandLAPACK itself in `DIR` | `<project-dir>/install/RandLAPACK-install` |
| `-j`, `--jobs N` | parallel build jobs; lower it if the build runs out of memory | number of cores |
| `--fresh` | clear the build directories and rebuild the dependencies the script built itself | reuse what matches, so a re-run is an incremental rebuild |
| `--no-extras`, `--no-benchmarks` | skip the extras or the benchmark project | both are built |
| `--no-openmp` | configure without OpenMP | |
| `-y`, `--yes` | assume "yes" for every prompt | |
| `--modify-rc` | append an `export RANDNLA_PROJECT_DIR=...` line to your shell config | never touch it; the summary prints the line to add yourself |
| `--no-progress` | plain one-line-per-step output, no redrawing | |

The extras and benchmark projects need nothing the script has not already
built, so leaving them on costs only time.

### Reusing preinstalled dependencies (discovery)

If you already have BLAS++, LAPACK++ or Random123 installed, point the script
at them and it skips those builds:

```shell
BLASPP_INSTALL_DIR=/path/to/blaspp-install \
LAPACKPP_INSTALL_DIR=/path/to/lapackpp-install \
RANDOM123_INSTALL_DIR=/path/to/random123 \
bash install.sh
```

Each variable names an install root: the directory containing
`lib*/cmake/<pkg>/` or, for Random123, `include/Random123/`. Dependencies the
script installed on an earlier run are reused automatically. RandBLAS is not
covered, on purpose: it stays a pinned submodule (see
[INSTALL.md, "RandBLAS is a pinned submodule"](INSTALL.md#randblas-is-a-pinned-submodule)).

### Sharing one dependency tree with RandBLAS

RandLAPACK's and RandBLAS's installers use the same `RandNLA-project` layout,
and both honour `RANDNLA_PROJECT_DIR`, so setting it once keeps everything in
one place:

```shell
export RANDNLA_PROJECT_DIR=$HOME/RandNLA-project
```

That shares the *location*, not the builds. Each project builds its own
BLAS++ into a separately named directory (`blaspp-mkl-cpu-install` here,
`blaspp-mkl-install` for RandBLAS), so that neither can overwrite the other's
copy while that copy's provenance stamp (section 5) still describes the
original. To reuse one BLAS++ for both, name it:

```shell
BLASPP_INSTALL_DIR=$RANDNLA_PROJECT_DIR/install/blaspp-mkl-install bash install.sh
```

The script then checks that choice by compiling, linking and running against
it, rather than trusting it.

### Running without prompts

Prompts appear only when stdin is a terminal. Piped and CI runs are already
non-interactive, with safe defaults (NVIDIA GPU detected: GPU build, but see
the known issue in section 2; AMD GPU or none: CPU build), so there is no
need to pipe `yes` into the script:

```shell
bash install.sh < /dev/null      # or simply: bash install.sh --yes
```

## 4. After the install

### Where everything is

With the quick-start commands, the layout is:

```
~/RandNLA/
├── RandLAPACK/               # Your clone. The script does NOT move it.
└── RandNLA-project/          # Created automatically
    ├── lib/
    │   ├── blaspp/           # Source, fetched at a pinned commit
    │   ├── lapackpp/         # Source, fetched at a pinned commit
    │   └── RandLAPACK -> ~/RandNLA/RandLAPACK   # Symlink to your clone
    ├── install/
    │   ├── blaspp-<backend>-<cpu|cuda>-install
    │   ├── lapackpp-<backend>-<cpu|cuda>-install
    │   ├── random123/
    │   └── RandLAPACK-install
    ├── build/                # Build directories and download caches
    └── install.log           # Full build log, appended on every run
```

- **Your clone stays where you put it.** `lib/RandLAPACK` is only a symlink
  to it, so git worktrees keep working.
- **The dependency install directories are named after their backend and GPU
  setting**, so an MKL and an OpenBLAS build, or a CUDA and a CPU build, sit
  side by side. The integer width is recorded in each directory's provenance
  stamp (section 5), so a build of the other width is rebuilt rather than
  reused.

| What | Where |
|---|---|
| Installed headers and CMake config | `~/RandNLA/RandNLA-project/install/RandLAPACK-install/` (or the `--prefix` directory). The config file is under `lib*/cmake/RandLAPACK/`. RandLAPACK is header-only, so there is no library archive. |
| Test executables | `~/RandNLA/RandNLA-project/build/RandLAPACK-build/bin/` |
| Extras test executable | `~/RandNLA/RandNLA-project/build/extras-build/test/RandLAPACK_extras_tests` |
| Benchmark executables | `~/RandNLA/RandNLA-project/build/benchmark-build/`; [`benchmark/README.md`](../benchmark/README.md) explains how to run them, including the GPU benchmarks and their output formats |

### Running only the GPU tests

With GPU support enabled, the GPU tests run as part of the suite. To run only
them:

```shell
~/RandNLA/RandNLA-project/build/RandLAPACK-build/bin/RandLAPACK_tests_gpu
```

### Rebuilding after code changes

After you modify RandLAPACK's source:

```shell
cmake --build ~/RandNLA/RandNLA-project/build/RandLAPACK-build -j 4 --target install
```

The install step matters. RandLAPACK is header-only, and the extras, the
benchmarks and your own projects use the *installed* headers, so they see a
change only after an install. Re-running `bash install.sh` from your clone
works too, since re-runs are incremental. Either way, use a shell with the
same environment you installed from (compiler, CMake, and CUDA if you enabled
it).

### Using RandLAPACK in your own CMake project

[INSTALL.md, section 5](INSTALL.md#5-using-randlapack-in-other-projects)
shows the `CMakeLists.txt`. With this script's layout, the flags are:

```
-Dblaspp_DIR=$HOME/RandNLA/RandNLA-project/install/blaspp-<backend>-<cpu|cuda>-install/lib/cmake/blaspp
-Dlapackpp_DIR=$HOME/RandNLA/RandNLA-project/install/lapackpp-<backend>-<cpu|cuda>-install/lib/cmake/lapackpp
-DRandLAPACK_DIR=$HOME/RandNLA/RandNLA-project/install/RandLAPACK-install/lib/cmake/RandLAPACK
```

On some platforms `lib` is `lib64`; the installer's final summary prints the
exact `RandLAPACK_DIR` for your machine. RandBLAS is installed next to
RandLAPACK and found through RandLAPACK's own CMake config, so it needs no
flag of its own. On macOS with Homebrew's `libomp`, your project also needs
the OpenMP settings at the end of
[INSTALL.md, section 5](INSTALL.md#5-using-randlapack-in-other-projects).

## 5. What the script does

1. **Checks the toolchain** (compiler, CMake 3.21+, Git), reporting
   everything missing at once rather than failing on the first item.
2. **Creates the layout** of section 4 and initialises the RandBLAS
   submodule. RandBLAS stays a pinned submodule, built and installed together
   with RandLAPACK rather than separately; its own installer exists for
   people who want RandBLAS alone.
3. **Builds BLAS++** at a pinned commit, with the selected backend and integer
   width, then **reads back the width it actually built** (section 6.2).
4. **Builds LAPACK++** at a pinned commit, against that BLAS++.
5. **Installs Random123** (header-only) at a pinned tag.
6. **Builds and installs RandLAPACK, then runs it.** A small test program is
   compiled, linked and executed against the finished install, checking a
   BLAS++ `gemm` and a LAPACK++ `gesdd` numerically. A configuration that
   configures cleanly can still be broken in ways no inspection catches;
   section 6.3 gives the most important case.
7. **Builds the extras and benchmark projects**, unless `--no-extras` or
   `--no-benchmarks` is given. Their first configure downloads Eigen,
   fast_matrix_market and (for the benchmarks) Spectra, each a pinned,
   checksummed archive, into `<project-dir>/build/fetchcontent-cache/`, which
   the two projects share. `--fresh` deletes it with the rest of `build/`.

**When a dependency is reused.** Every dependency is fetched at an immutable
ref and stamped with its provenance (the source and configuration it was
built from). It is reused only when it came from the same source *in the
same configuration*: backend, integer width and GPU setting all count.
Switching from `--no-gpu` to `--gpu` therefore rebuilds BLAS++ rather than
silently reusing a CPU-only one.

**Where BLAS++ and LAPACK++ come from.** The BallisticLA forks, at the
commits (and tag) pinned in
[`installers/dependency-pins.env`](../installers/dependency-pins.env).
[INSTALL.md](INSTALL.md#why-blas-and-lapack-come-from-forks) explains why.

## 6. Integer width, backends, and tested configurations

### 6.1 What we test

Each row is a configuration that CI builds and tests on every pull request
([CI.md](CI.md) lists the jobs), so the table states what is exercised, not
what ought to work. A configuration not listed may well work; it is simply
untested. None of them uses a GPU.

| OS | Compiler | BLAS backend | Integer width | OpenMP |
|---|---|---|---|---|
| Ubuntu (`ubuntu-latest`) | gcc | OpenBLAS | LP64 | yes |
| Ubuntu (`ubuntu-latest`) | gcc | OpenBLAS (`libopenblas64` only) | ILP64 | yes |
| macOS (`macos-latest`) | Apple Clang | Accelerate (new interface) | ILP64 | yes, Homebrew `libomp` |
| macOS (`macos-latest`) | Apple Clang | Accelerate (new interface), `--blas-int=lp64` | LP64 | yes, Homebrew `libomp` |
| macOS (`macos-latest`) | Apple Clang | Accelerate (new interface), built by hand (core-macos) | LP64 | no |
| macOS (`macos-latest`) | Apple Clang | Homebrew OpenBLAS (`--blas=openblas`) | LP64 | yes, Homebrew `libomp` |
| Windows | MSVC | oneMKL | ILP64 | yes (`/openmp:llvm`), and a serial lane |
| Windows | MSVC | OpenBLAS | LP64 | no |
| Windows | MSVC | `-Backend custom`, given oneMKL's import libraries | ILP64 | no |

- The installer lanes also cover a fresh install, an idempotent re-run (the
  installer run again in place must succeed), and, on Linux, dependency
  discovery through `BLASPP_INSTALL_DIR` and friends.
- Apple Clang ships no OpenMP runtime, so a macOS build is single-threaded
  unless you install Homebrew's `libomp`, which the installer uses when
  present.
- No CI job builds or runs the CUDA code
  ([#215](https://github.com/BallisticLA/RandLAPACK/issues/215)), so `--gpu`
  is exercised locally only.

### 6.2 Which integer width you get, and why

A backend uses either 32-bit (**LP64**) or 64-bit (**ILP64**) integers for
matrix dimensions. The installer prefers ILP64 wherever the backend can
genuinely provide it:

| `--blas=` | Width you get | Why |
|---|---|---|
| `mkl` | ILP64 | `mkl_intel_ilp64` is a separate library, so requesting it actually selects it |
| `openblas` | LP64 with a warning, or ILP64 when `libopenblas64` is the only OpenBLAS installed | BLAS++ tries plain `-lopenblas` first; see below |
| `accelerate` | ILP64 | Apple's new interface (macOS 13.3 or later) ships ILP64 in the framework, and the pinned BLAS++ and LAPACK++ support it ([INSTALL.md](INSTALL.md#why-blas-and-lapack-come-from-forks)). Older macOS lacks the new interface entirely, and the installer refuses the legacy one (section 6.4) |
| `custom` | whatever you pass | you named the library, so its width is yours to state |

**Why OpenBLAS usually gives LP64.** The pinned BLAS++ looks for plain
`-lopenblas` first and for the ILP64 `-lopenblas64` second
([blaspp#129](https://github.com/icl-utk-edu/blaspp/pull/129)). Its `int64`
BLAS probe cannot tell the widths apart, so an LP64 `libopenblas` passes it
by accident. The LAPACK probe then
catches the mismatch and fails the whole configure before `-lopenblas64` is
ever tried, and the installer falls back to LP64 and says so. With
`libopenblas64` as the only OpenBLAS installed (on Debian/Ubuntu,
`libopenblas64-dev` without `libopenblas-dev`), `--blas=openblas` gives ILP64.
Either way, after building, the installer reads the width back out of
BLAS++'s generated `blas/defines.h` and reports what it actually produced.

**With both installed**, or with an ILP64 OpenBLAS under another name, name
the library yourself:

```shell
bash install.sh --blas=custom --blas-int=ilp64 \
  --blas-libraries=/usr/lib/x86_64-linux-gnu/libopenblas64.so
```

**An OpenBLAS whose symbols carry a suffix**, such as conda-forge's
`libopenblas64_` (`dgemm_64_`), needs BLAS++'s `blas_symbol_suffix` option,
which the installer does not set. Build BLAS++ and LAPACK++ by hand
([INSTALL.md, "ILP64 OpenBLAS"](INSTALL.md#ilp64-openblas)) and point the
installer at them with `BLASPP_INSTALL_DIR` and `LAPACKPP_INSTALL_DIR`
([section 3](#reusing-preinstalled-dependencies-discovery)).

### 6.3 Does LP64 limit RandLAPACK?

Rarely, and it fails loudly rather than silently. RandLAPACK uses `int64_t`
throughout, and when a value exceeds the backend's range, BLAS++ and LAPACK++
**throw** an error naming the offending argument rather than truncate it
(`to_blas_int`, `to_lapack_int`). The check is on individual dimensions and
leading dimensions, not on element counts (the backend never receives
`m*n`), so a 100,000 x 100,000 matrix is fine under LP64. The limit bites
only past roughly 2.1 billion in a *single* dimension, and for sparse work
with `nnz > 2^31`.

**The silent case is a width mismatch**, and it is why the installer runs a
program rather than only linking one. If BLAS++'s headers were built for one
width while the library actually loaded uses the other, the check compiles
out entirely, and 64-bit values reach routines that read 32 bits. The symptom
is not wrong numbers but nonsense control values: a misread workspace query
becomes an absurd `lwork`, and the run dies in allocation or never finishes.
The verification step (section 5, step 6) catches that at install time.

### 6.4 macOS: Accelerate, through Apple's new interface only

The macOS default is Accelerate, through Apple's **new** interface (macOS
13.3 or later), which the pinned BLAS++ and LAPACK++ support. No Homebrew
BLAS is needed. RandLAPACK does not support Apple's legacy interface, whose
`gesdd` returns wrong results on Apple Silicon;
[INSTALL.md, "Apple Accelerate"](INSTALL.md#apple-accelerate-macos) gives the
details.

- **ILP64 by default.** The installer builds BLAS++ with `-Dblas_int=int64`,
  and falls back to LP64 with a warning only if that configure fails. Pass
  `--blas-int=lp64` for an LP64 build.
- **No silent fallback to the legacy interface.** The installer refuses a
  build in which BLAS++ fell back to it. It checks BLAS++'s generated
  `defines.h` right after the build, and again numerically with the `gesdd`
  check of the verification step (section 5, step 6).
- **On macOS older than 13.3**, use `--blas=openblas` (after
  `brew install openblas`).

## 7. When something goes wrong

- **Read the log.** All compiler output goes to `<project-dir>/install.log`
  automatically, while the console shows one line per step. Any failure
  prints the log path and the last lines of the log, so there is no need to
  `tee` the output yourself.
- **Common messages**, here and in manual builds:

| Message | Cause and fix |
|---|---|
| "BLAS++ requires LAPACK for [cz]rot, [cz]syr, [cz]symv" | BLAS++ was asked for a width the backend does not provide, usually ILP64 (`--blas-int=ilp64`, `-Dblas_int=int64`) with an LP64 OpenBLAS. Drop the request, or install an ILP64 backend (section 6.2). |
| "Could not find a package configuration file provided by "blaspp"" (or `lapackpp`, `RandLAPACK`) | The `<package>_DIR` flag names the wrong directory. It must contain `<package>Config.cmake`, sometimes under `lib64` rather than `lib`. |
| `ctest` reports "No tests were found" | GoogleTest was not found when RandLAPACK was configured. Install it and re-run the installer. |
| "unsupported GNU version" from `nvcc` | Your GCC is too new for your CUDA release; see the [compatibility table](INSTALL.md#cuda-and-gcc-versions). |
| "Could NOT find OpenMP", configuring your own project on macOS | See the macOS note at the end of [INSTALL.md, section 5](INSTALL.md#5-using-randlapack-in-other-projects). |

- **For a manual build**, see also
  [INSTALL.md, section 8](INSTALL.md#8-troubleshooting).
