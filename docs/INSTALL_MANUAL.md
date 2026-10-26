# Installing and using RandLAPACK

**Most people should use an installer.** It builds RandLAPACK and all of its
dependencies with one command:

| Your machine | Command | Guide |
|---|---|---|
| Linux or macOS | `bash install.sh` | [INSTALL_SCRIPT.md](INSTALL_SCRIPT.md) |
| Native Windows (MSVC) | `powershell -ExecutionPolicy Bypass -File .\installers\install.ps1` | [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md) |

This guide covers the rest: building each piece by hand when you want control
over how the dependencies are configured, and using RandLAPACK from your own
CMake project.

- Sections 2 to 4: a basic manual build. Section 5: using it from another
  CMake project.
- Section 6: choosing and configuring the backend. Skip it on your first
  build; its BLAS++ and LAPACK++ recipes can replace those of Section 3.
- Section 7: troubleshooting. Pinned versions, the forks, the RandBLAS
  submodule and CUDA versions are in [DEPENDENCIES.md](DEPENDENCIES.md).

The commands in Sections 3 to 6 are for bash or zsh on Linux and macOS. On
Windows, use `install.ps1`.

## 1. Key terms

| Term | Meaning |
|---|---|
| **BLAS, LAPACK** | Standard interfaces for dense linear algebra. BLAS covers basic operations such as matrix multiplication; LAPACK builds factorizations, linear solvers and the SVD on top of BLAS. |
| **Backend** | The library that implements BLAS and LAPACK on your machine: Intel oneMKL (`mkl` in build flags), OpenBLAS, Apple Accelerate (part of macOS) or AMD AOCL. |
| **BLAS++, LAPACK++** | C++ wrappers around BLAS and LAPACK (upstream: [icl-utk-edu/blaspp](https://github.com/icl-utk-edu/blaspp) and [icl-utk-edu/lapackpp](https://github.com/icl-utk-edu/lapackpp)). They hide the differences between backends and are templated on the numeric type. RandLAPACK does its dense linear algebra through them. |
| **LP64, ILP64** | The integer width a backend uses for matrix dimensions and indices: 32-bit (LP64) or 64-bit (ILP64). BLAS++, LAPACK++ and the backend must agree on it. |
| **Upstream, fork** | Upstream is a project's original repository. A fork is a copy of it under BallisticLA, where we carry fixes that upstream has not released yet. |
| **Pin, tag** | A pin fixes a dependency to one exact git commit, so every build compiles the same source. A tag is a readable name for one commit. |
| **Submodule** | A git repository included in another at one fixed commit. RandBLAS is a submodule of RandLAPACK; `git clone --recursive` fetches it. |
| **`<package>_DIR`** | A CMake flag naming the directory that contains `<package>Config.cmake`, so that CMake can find an installed package. |

## 2. Requirements

- **CMake** 3.21 or later, and 3.24 or later for the `extras/` and
  `benchmark/` projects (3.31 recommended).
- **A C++20 compiler**, one of:
  - GCC 13 or later, the tested floor. GCC 11 and 12 do not compile the
    current version.
  - Clang 14 or later, with libstdc++ 13 or newer, or libc++ (not
    extensively tested).
  - Intel icpx (not extensively tested). Build LAPACK++ from the pinned fork
    of Section 3: an icpx-built upstream LAPACK++ crashes in `gemqrt`
    ([#202](https://github.com/BallisticLA/RandLAPACK/issues/202)), and the
    fork carries the fix.
  - MSVC (Visual Studio 2022) on native Windows, with Ninja as the build tool;
    both come with the "Desktop development with C++" workload. See
    [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md).
- **A backend** (Section 6). On macOS 13.3 or later, Apple Accelerate is
  built in.
- **Optional:**
  - GoogleTest, Google's C++ testing framework. Without it, RandLAPACK's
    regression tests are not built. Any package manager provides it.
  - OpenMP, for multithreaded loops. RandLAPACK and its dependencies detect
    it and use it automatically, except on macOS: Apple Clang finds
    Homebrew's `libomp` only with
    [these settings](USING.md#macos-with-homebrews-libomp). Pass them to
    every configure; without them, RandLAPACK's own loops run on one thread
    (Accelerate still uses all cores).
  - CUDA, for GPU support. It is off by default;
    [DEPENDENCIES.md](DEPENDENCIES.md#cuda-and-gcc-versions) lists what a
    GPU build needs.

On Ubuntu 24.04 and macOS, the [README](../README.md#install) gives one
package-manager command that installs everything but CUDA. Elsewhere, we
recommend [Spack](https://github.com/spack/spack)
([INSTALL_SCRIPT.md](INSTALL_SCRIPT.md#installing-the-requirements-with-spack)
has a recipe). On some systems library directories are called `lib64` rather
than `lib`; adjust the paths in the commands below.

## 3. Building the dependencies

RandLAPACK needs BLAS++, LAPACK++ and Random123 (header-only random number
generators). RandBLAS, which depends on BLAS++ and Random123 too, comes inside
this repository as a submodule.

- BLAS++ and LAPACK++ can be built with GNU make or with CMake.
  **RandLAPACK requires the CMake-installed versions.**
- Build BLAS++ and LAPACK++ from the BallisticLA forks, at the commits
  `install.sh` and CI build. The latest upstream release lacks fixes
  RandLAPACK relies on
  ([DEPENDENCIES.md](DEPENDENCIES.md#why-blas-and-lapack-come-from-forks)).

Run every command in this section and the next from one parent directory.
Start by cloning RandLAPACK there and reading its dependency pins into your
shell:

```shell
git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
. RandLAPACK/installers/dependency-pins.env   # sets BLASPP_URL, BLASPP_REF, ...
```

1. **BLAS++** ([source](https://github.com/BallisticLA/blaspp)):
   ```shell
   git clone $BLASPP_URL blaspp && git -C blaspp checkout $BLASPP_REF
   cmake -S blaspp -B blaspp-build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=$PWD/blaspp-install \
       -Dbuild_tests=OFF -Dgpu_backend=none
   cmake --build blaspp-build -j 4 --target install
   ```
   - BLAS++ chooses the integer width itself, trying LP64 first. To force a
     width, add `-Dblas_int=int32` (LP64) or `-Dblas_int=int64` (ILP64).
     ILP64 needs an ILP64 backend, such as oneMKL or `libopenblas64`
     (Section 6); with an ordinary OpenBLAS, `-Dblas_int=int64` fails with
     "BLAS++ requires LAPACK".
   - BLAS++ picks oneMKL when oneMKL's environment is loaded. Then add
     `-Dblas_int=int64`: against an LP64 oneMKL, RandLAPACK's test suite
     (and any code that multiplies two sparse matrices) does not compile.
     Section 6 has the full oneMKL recipe.
   - `-Dgpu_backend=none` keeps BLAS++ from building CUDA support just
     because it finds a CUDA Toolkit; for GPU support see
     [DEPENDENCIES.md](DEPENDENCIES.md#cuda-and-gcc-versions).
2. **LAPACK++** ([source](https://github.com/BallisticLA/lapackpp)), against
   that BLAS++:
   ```shell
   git clone $LAPACKPP_URL lapackpp && git -C lapackpp checkout $LAPACKPP_REF
   cmake -S lapackpp -B lapackpp-build -DCMAKE_BUILD_TYPE=Release \
       -Dblaspp_DIR=$PWD/blaspp-install/lib/cmake/blaspp \
       -DCMAKE_INSTALL_PREFIX=$PWD/lapackpp-install \
       -Dbuild_tests=OFF -Dgpu_backend=none
   cmake --build lapackpp-build -j 4 --target install
   ```
3. **Random123** ([source](https://github.com/DEShawResearch/random123)) is
   header-only, so a clone at the pinned release is enough:
   ```shell
   git clone --branch v1.14.0 https://github.com/DEShawResearch/Random123.git random123
   ```

## 4. Building and installing RandLAPACK

With the dependencies of Section 3 in place, build and install RandLAPACK from
the same parent directory (for GPU support, see
[DEPENDENCIES.md](DEPENDENCIES.md#cuda-and-gcc-versions)):

```shell
cmake -S RandLAPACK -B RandLAPACK-build -DCMAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=$PWD/blaspp-install/lib/cmake/blaspp \
    -Dlapackpp_DIR=$PWD/lapackpp-install/lib/cmake/lapackpp \
    -DRandom123_DIR=$PWD/random123/include \
    -DCMAKE_INSTALL_PREFIX=$PWD/RandLAPACK-install \
    -DBUILD_TESTS=OFF
cmake --build RandLAPACK-build -j 4 --target install
```

Then run all tests:

```shell
OMP_NUM_THREADS=1 ctest --test-dir RandLAPACK-build -j 4
```

This runs four single-threaded tests at a time, typically in a minute or two.
Plain `ctest` runs one test at a time, each using every core, and can take
far longer, especially on a busy machine. If `ctest` reports "No tests were
found", GoogleTest was not found at configure time. Benchmarks live in
`benchmark/`, and `ctest` does not run them.

What the flags in the recipe mean:

| Flag | Meaning |
|---|---|
| `-Dblaspp_DIR=X` | `X` contains `blasppConfig.cmake`. |
| `-Dlapackpp_DIR=X` | `X` contains `lapackppConfig.cmake`. |
| `-DRandom123_DIR=X` | `X` contains a folder `Random123` with the Random123 headers: `X/Random123/philox.h` must exist. |
| `-DCMAKE_INSTALL_PREFIX=X` | `X/include` receives the headers of RandLAPACK and RandBLAS. `X/lib/cmake/RandLAPACK` (`lib64` on some systems) receives `RandLAPACKConfig.cmake`, which Section 5 needs. |
| `-DBUILD_TESTS=OFF` | Skips the RandBLAS submodule's own tests, as `install.sh` and CI do. |

Other optional flags:

| Flag | Effect |
|---|---|
| `-DRandLAPACK_BUILD_TESTS=OFF` | Skips RandLAPACK's own GoogleTest suite, which is built by default when GoogleTest is found. |
| `-DSANITIZE_ADDRESS=ON` | Adds AddressSanitizer instrumentation (CI uses it with Debug builds). |

The `extras/` project has a third test switch of its own,
`-DBUILD_TESTING=OFF`.

## 5. Using RandLAPACK in other projects

[USING.md](USING.md) shows how to use an installed RandLAPACK from your own
CMake project, with a minimal example program. After the recipe of Section
4, point `RandLAPACK_DIR` at your own install prefix:

```
-DRandLAPACK_DIR=<parent>/RandLAPACK-install/lib/cmake/RandLAPACK
```

Here `<parent>` is the directory you ran Sections 3 and 4 from, as an
absolute path (`lib64` instead of `lib` on some systems). If you built
RandLAPACK with `-DRandLAPACK_EXTERNAL_RandBLAS=ON`, also pass
`-DRandBLAS_DIR`.

## 6. Choosing and configuring the backend

On Linux, use Intel oneMKL (recommended) or OpenBLAS. On macOS, use Apple
Accelerate, or Homebrew OpenBLAS. On Windows, the installer defaults to
oneMKL
([INSTALL_WINDOWS.md, section 4](INSTALL_WINDOWS.md#4-choosing-a-blaslapack-backend)).

RandLAPACK's performance depends heavily on how its dependencies are
configured, and BLAS++ is the most sensitive one. If performance matters,
read what `cmake` prints when you configure BLAS++ (it names the exact BLAS
libraries BLAS++ links) and save it while you set up.

### Intel oneMKL (recommended)

We recommend building BLAS++ and LAPACK++ against Intel oneMKL 2022 or later.
That version ships CMake configuration files, which are very useful if a
project of yours combines RandLAPACK with oneMKL, for example to use oneMKL's
sparse linear algebra.

Getting BLAS++ to link oneMKL as intended can be tricky. This configuration
is known to work. It installs BLAS++ and LAPACK++ into `$HOME/mklpp`; any
directory you can write to works.

1. Install oneMKL ([download](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=webdownload&options=online))
   and load its environment in the shell you build from (adjust the path to
   your installation):
   ```shell
   source /opt/intel/oneapi/setvars.sh    # or only oneMKL's: .../oneapi/mkl/latest/env/vars.sh
   ```
   This sets `MKLROOT` and puts oneMKL's headers and libraries on the search
   paths. `install.sh` also chooses oneMKL by default when `MKLROOT` is set.
2. Clone RandLAPACK, BLAS++ and LAPACK++ as in Section 3. From the same
   parent directory, build BLAS++ with `g++` against oneMKL, saving the
   configure output to `blaspp-configure.log`:
   ```shell
   export CXX=g++
   cmake -S blaspp -B blaspp-mkl-build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=$HOME/mklpp -Dblas=mkl -Dblas_int=int64 \
       -Dbuild_tests=OFF -Dgpu_backend=none | tee blaspp-configure.log
   cmake --build blaspp-mkl-build -j 4 --target install
   ```
3. Check that `$HOME/mklpp/include/blas/defines.h` contains `BLAS_HAVE_MKL`.
   Without it (BLAS++ links oneMKL but did not find `mkl.h`, usually
   because oneMKL's headers were not on the include path), RandBLAS uses its
   portable sparse code instead of oneMKL's, and says so only in its
   configure output.
4. Build LAPACK++ into the same prefix:
   ```shell
   cmake -S lapackpp -B lapackpp-mkl-build -DCMAKE_BUILD_TYPE=Release \
       -Dblaspp_DIR=$HOME/mklpp/lib/cmake/blaspp -DCMAKE_INSTALL_PREFIX=$HOME/mklpp \
       -Dbuild_tests=OFF -Dgpu_backend=none
   cmake --build lapackpp-mkl-build -j 4 --target install
   ```

RandLAPACK (Section 4) then finds both with:

```
-Dblaspp_DIR=$HOME/mklpp/lib/cmake/blaspp -Dlapackpp_DIR=$HOME/mklpp/lib/cmake/lapackpp
```

### Apple Accelerate (macOS)

Accelerate is part of macOS, so there is nothing to install. It contains two
LAPACK implementations, and **RandLAPACK supports only the new one** (macOS
13.3 or later, selected through `ACCELERATE_NEW_LAPACK`; LAPACK 3.12 on
current SDKs).

- The legacy default is LAPACK 3.2.1, from 2009. Its divide-and-conquer SVD
  (`gesdd`) returns wrong results on Apple Silicon, and it lacks routines
  RandLAPACK requires, such as `orhr_col` and `geqrt` (BQRRP and HQRRP use
  them).
- The new interface fixes both, and also provides ILP64.

To build against the new interface:

1. Build BLAS++ with `-Dblas=accelerate`, from a BLAS++/LAPACK++ version that
   supports the new interface (the pinned forks of Section 3 do). Add
   `-Dblas_int=int64` for ILP64 or `-Dblas_int=int32` for LP64. The interface
   carries over to LAPACK++ and RandLAPACK automatically, and LAPACK++ takes
   its integer width from BLAS++.
2. Check that the installed `include/blas/defines.h` contains
   `ACCELERATE_NEW_LAPACK`.

`install.sh` does this on macOS, with ILP64 by default
([INSTALL_SCRIPT.md, section 6.4](INSTALL_SCRIPT.md#64-macos-accelerate-through-apples-new-interface-only)).
On macOS older than 13.3, or if you prefer it, use Homebrew OpenBLAS
(`brew install openblas`, then `install.sh --blas=openblas`).

### ILP64 OpenBLAS

The ILP64 builds of OpenBLAS come under other library names, and BLAS++ has
to be told which kind you have. Both cases build BLAS++ and LAPACK++ from the
pinned forks of Section 3.

- **Unsuffixed, such as Debian/Ubuntu's `libopenblas64`** (package
  `libopenblas64-dev`; the symbols have the usual names, such as `dgemm_`).
  Add `-Dblas=openblas -Dblas_int=int64` to the BLAS++ command. BLAS++ then
  finds `-lopenblas64`, but only when no LP64 `libopenblas` is installed
  beside it, because it tries `-lopenblas` first.
  [INSTALL_SCRIPT.md, section 6.2](INSTALL_SCRIPT.md#62-which-integer-width-you-get-and-why)
  explains why, and how to name the library when both are installed.
- **Symbol-suffixed, such as conda-forge's `libopenblas64_`** (OpenBLAS built
  with `INTERFACE64=1 SYMBOLSUFFIX=64_`, so the symbols are named like
  `dgemm_64_`). Add `-Dblas=openblas -Dblas_int=int64 -Dblas_symbol_suffix=64_`
  to the BLAS++ command, and name the library with
  `-DBLAS_LIBRARIES=<dir>/libopenblas64_.so` unless its directory is on the
  linker's search path. LAPACK++ needs no extra flag: its
  `lapack_symbol_suffix` option, when left empty, takes BLAS++'s suffix. Set
  it only for a LAPACK library whose suffix differs from the BLAS library's.

`install.sh` has no option for a symbol suffix; section 6.2 of
[INSTALL_SCRIPT.md](INSTALL_SCRIPT.md#62-which-integer-width-you-get-and-why)
shows how to use your own BLAS++ and LAPACK++ builds with it. RandLAPACK's CI
does not cover a suffixed OpenBLAS; the forks' own CI builds and tests BLAS++
and LAPACK++ against conda-forge's.

## 7. Troubleshooting

- **Common error messages** and their fixes are in
  [INSTALL_SCRIPT.md, section 7](INSTALL_SCRIPT.md#7-when-something-goes-wrong);
  they apply to manual builds too.
- **Compare with CI.** RandLAPACK's GitHub Actions workflows install it from
  scratch and run its unit tests; [CI.md](CI.md) describes each lane. The
  [workflow files](https://github.com/BallisticLA/RandLAPACK/tree/main/.github/workflows)
  are a working reference: the core-linux and core-macos build steps show a
  directory layout that works.
- **Building BLAS++'s own tester.** The recipes above turn it off
  (`-Dbuild_tests=OFF`). If you want it, set `CPATH` to the backend's
  `include/` folder, so that CMake finds CBLAS.
