# Installing and using RandLAPACK

**Most people should use an installer.** It builds RandLAPACK and all of its
dependencies with one command:

| Your machine | Command | Guide |
|---|---|---|
| Linux or macOS | `bash install.sh` | [INSTALL_SCRIPT.md](INSTALL_SCRIPT.md) |
| Native Windows (MSVC) | `installers\install.ps1` | [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md) |

This guide covers the rest: building each piece by hand when you want control
over how the dependencies are configured, and using RandLAPACK from your own
CMake project.

- Sections 2 to 4: a basic manual build. Section 5: using it from another
  CMake project.
- Section 6: choosing and configuring the backend. Skip it on your first
  build; its BLAS++ and LAPACK++ recipes can replace those of Section 3.
- Section 7: details (forks and pins, the RandBLAS submodule, CUDA
  versions). Section 8: troubleshooting.

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

- **CMake** 3.21 or later (3.31.9 recommended).
- **A C++20 compiler**, one of:
  - GCC 13 or later: the tested floor, because RandBLAS uses C++20 concepts.
  - Clang 14 or later (not extensively tested).
  - Intel icpx (not extensively tested). Build LAPACK++ from the fork tag of
    Section 3: an icpx-built upstream LAPACK++ crashes in `gemqrt`
    ([#202](https://github.com/BallisticLA/RandLAPACK/issues/202)), and the
    tag carries the fix.
  - MSVC (Visual Studio 2022) on native Windows, with Ninja as the build tool;
    both come with the "Desktop development with C++" workload. See
    [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md).
- **A backend** (Section 6). On macOS 13.3 or later, Apple Accelerate is
  built in.
- **Optional:**
  - GoogleTest, Google's C++ testing framework. Without it, RandLAPACK's
    regression tests are not built. Any package manager provides it.
  - OpenMP, for multithreaded loops. RandLAPACK and its dependencies detect
    it and use it automatically.
  - CUDA, for GPU support. It is off by default; `-DRequireCUDA=ON` enables
    it. Section 7 lists the CUDA, GCC and driver versions.

We recommend [Spack](https://github.com/spack/spack) for installing these,
GoogleTest included. On some systems library directories are called `lib64`
rather than `lib`; adjust the paths in the commands below.

## 3. Building the dependencies

RandLAPACK needs BLAS++, LAPACK++ and Random123 (header-only random number
generators). RandBLAS, which depends on BLAS++ and Random123 too, comes inside
this repository as a submodule.

- BLAS++ and LAPACK++ can be built with GNU make or with CMake.
  **RandLAPACK requires the CMake-installed versions.**
- Build BLAS++ and LAPACK++ from the BallisticLA forks at the tag
  `ballistic-2026.10.22`, the commits `install.sh` and CI build. The latest
  upstream release lacks fixes RandLAPACK relies on
  ([Section 7](#why-blas-and-lapack-come-from-forks)).
- Run these recipes, and the one in Section 4, from one parent directory.
  The RandLAPACK clone goes there too.

1. **BLAS++** ([source](https://github.com/BallisticLA/blaspp)).
   `-Dblas_int=int64` asks for ILP64; `-Dblas_int=int32` gives LP64.
   ```shell
   git clone --branch ballistic-2026.10.22 https://github.com/BallisticLA/blaspp.git
   mkdir blaspp-build
   cd blaspp-build
   cmake -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=`pwd`/../blaspp-install \
       -Dbuild_tests=OFF \
       -Dblas_int=int64 \
       ../blaspp
   make -j2 install
   ```
2. **LAPACK++** ([source](https://github.com/BallisticLA/lapackpp)), against
   that BLAS++:
   ```shell
   git clone --branch ballistic-2026.10.22 https://github.com/BallisticLA/lapackpp.git
   mkdir lapackpp-build
   cd lapackpp-build
   cmake -DCMAKE_BUILD_TYPE=Release \
       -Dblaspp_DIR=`pwd`/../blaspp-install/lib/cmake/blaspp \
       -DCMAKE_INSTALL_PREFIX=`pwd`/../lapackpp-install \
       -Dbuild_tests=OFF \
       ../lapackpp
   make -j2 install
   ```
3. **Random123** ([source](https://github.com/DEShawResearch/random123)):
   ```shell
   git clone https://github.com/DEShawResearch/random123.git
   cd random123/
   make prefix=`pwd`/../random123-install install-include
   ```

## 4. Building and installing RandLAPACK

RandLAPACK is configured with CMake and built with GNU make. With the
dependencies of Section 3 in place, download, build and install it (add
`-DRequireCUDA=ON` for GPU support):

```shell
git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
mkdir RandLAPACK-build
cd RandLAPACK-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dlapackpp_DIR=`pwd`/../lapackpp-install/lib/cmake/lapackpp/ \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/cmake/blaspp/ \
    -DRandom123_DIR=`pwd`/../random123-install/include/ \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../RandLAPACK-install \
    ../RandLAPACK/
make -j install
```

Then run all tests from the same directory:

```shell
ctest
```

They complete reasonably quickly. Benchmarks live in `benchmark/`, and
`ctest` does not run them.

What the flags in the recipe mean:

| Flag | Meaning |
|---|---|
| `-Dblaspp_DIR=X` | `X` contains `blasppConfig.cmake`. |
| `-Dlapackpp_DIR=X` | `X` contains `lapackppConfig.cmake`. After the oneMKL recipe of Section 6 it is `/opt/mklpp/lib/cmake/lapackpp`. |
| `-DRandom123_DIR=X` | `X` contains a folder `Random123` with the Random123 headers: `X/Random123/philox.h` must exist. |
| `-DCMAKE_INSTALL_PREFIX=X` | Subdirectories of `X` receive RandLAPACK's headers and the CMake configuration files other projects use. Note the directory that ends up holding `RandLAPACKConfig.cmake`; Section 5 needs it. |

Optional flags the recipe does not use:

| Flag | Effect |
|---|---|
| `-DRandLAPACK_BUILD_TESTS=OFF` | Skips the GoogleTest regression suite, which is built by default when GoogleTest is found. |
| `-DBUILD_TESTS=OFF` | Skips the RandBLAS submodule's own tests, as `install.sh` and the CI recipes do. |
| `-DSANITIZE_ADDRESS=ON` | Adds AddressSanitizer instrumentation (CI uses it with Debug builds). |

## 5. Using RandLAPACK in other projects

An installed RandLAPACK works like any other CMake package. This
`CMakeLists.txt` links an executable to it:

```cmake
cmake_minimum_required(VERSION 3.21)
project(myexec)

find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)
find_package(RandLAPACK REQUIRED)

add_executable(myexec ...)
target_link_libraries(myexec RandLAPACK::RandLAPACK ...)
```

The exported target `RandLAPACK::RandLAPACK` brings RandBLAS, LAPACK++ and
BLAS++ along as link dependencies. Configure the project with:

- `-Dblaspp_DIR=W`, where `W` contains `blasppConfig.cmake`;
- `-Dlapackpp_DIR=X`, where `X` contains `lapackppConfig.cmake`;
- `-DRandLAPACK_DIR=Z`, where `Z` contains `RandLAPACKConfig.cmake`.

RandBLAS needs no flag with the default submodule build: it installs under
the same prefix as RandLAPACK, and RandLAPACK's config finds it there. If
RandLAPACK was built with `-DRandLAPACK_EXTERNAL_RandBLAS=ON`, also pass
`-DRandBLAS_DIR`.

After `install.sh`, the exact paths are in
[INSTALL_SCRIPT.md, section 4](INSTALL_SCRIPT.md#4-after-the-install). On
Windows, also see
[INSTALL_WINDOWS.md, section 7](INSTALL_WINDOWS.md#7-runtime-dlls-what-staging-means)
for the runtime DLLs.

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
is known to work:

1. Install oneMKL ([download](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=webdownload&options=online))
   and make its libraries findable, for example with lines like these in your
   shell startup file (adjust the path to your installation):
   ```
   export MAIN_MKL_LIBS="/path/to/intel/oneapi/mkl/latest/lib/intel64"
   export LD_LIBRARY_PATH="${MAIN_MKL_LIBS}:${LD_LIBRARY_PATH}"
   export LIBRARY_PATH="${MAIN_MKL_LIBS}:${LIBRARY_PATH}"
   ```
2. Get the BLAS++ source (the fork and tag of Section 3), create a folder
   `build` at its top level, and `cd` into it.
3. Run `export CXX=g++`, so that `g++` is the compiler for this shell session.
4. Choose one install prefix for both BLAS++ and LAPACK++. We recommend
   `/opt/mklpp`.
5. Configure BLAS++, and save the output:
   ```
   cmake -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=/opt/mklpp \
       -Dblas=mkl \
       -Dblas_int=int64 \
       -Dbuild_tests=OFF ..
   ```
6. Run `cmake --build .`
7. Run `sudo make install`
8. Get the LAPACK++ source (the fork and tag of Section 3), create a folder
   `build` at its top level, and `cd` into it.
9. Configure, build and install LAPACK++:
   ```
   cmake -DCMAKE_BUILD_TYPE=Release \
      -Dblaspp_DIR=/opt/mklpp/lib/cmake/blaspp \
      -DCMAKE_INSTALL_PREFIX=/opt/mklpp \
      -Dbuild_tests=OFF ..
   make -j2 install
   ```

Other CMake projects then find both through `find_package(blaspp REQUIRED)`
and `find_package(lapackpp REQUIRED)` in their `CMakeLists.txt`, configured
with:

```
-Dblaspp_DIR=/opt/mklpp/lib/cmake/blaspp -Dlapackpp_DIR=/opt/mklpp/lib/cmake/lapackpp
```

### Apple Accelerate (macOS)

Accelerate is part of macOS, so there is nothing to install. It contains two
LAPACK implementations, and **RandLAPACK supports only the new one** (macOS
13.3 or later, selected through `ACCELERATE_NEW_LAPACK`; LAPACK 3.12 on
current SDKs).

- The legacy default is LAPACK 3.2.1, from 2009. Its divide-and-conquer SVD
  (`gesdd`) returns wrong results on Apple Silicon, and it lacks routines
  RandLAPACK requires, such as `orhr_col` and `geqrt` (BQRRP and HQRRP use
  them). RandLAPACK calls `gesdd` in `rl_rsvd.hh`, `rl_abrik.hh`,
  `rl_revd2.hh`, `rl_preconditioners.hh`, `rl_util.hh` and
  `rl_test_utils.hh`.
- The new interface fixes both, and also provides ILP64.

To build against the new interface:

1. Build BLAS++ with `-Dblas=accelerate`, from a BLAS++/LAPACK++ version that
   supports the new interface (the fork tag of Section 3 does). Add
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
fork tag of Section 3.

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

## 7. Details

### Why BLAS++ and LAPACK++ come from forks

RandLAPACK builds BLAS++ and LAPACK++ from the BallisticLA forks
([blaspp](https://github.com/BallisticLA/blaspp),
[lapackpp](https://github.com/BallisticLA/lapackpp)), not from an upstream
release.

- **What is pinned.**
  [`installers/dependency-pins.env`](../installers/dependency-pins.env)
  records each fork's URL and a full commit hash, plus the tag
  `ballistic-2026.10.22` on those commits. `install.sh`, `install.ps1` and
  every CI lane read this one file. If you build BLAS++ and LAPACK++
  yourself, build that tag.
- **Why not upstream.** The latest upstream release, `v2025.05.28`, lacks
  fixes RandLAPACK relies on. Some are already on upstream `master`. The rest
  are seven upstream pull requests that are open but not yet merged or
  released, so only the forks have them.

| Fix | Upstream pull requests | Where it is |
|---|---|---|
| Library paths that contain spaces | icl-utk-edu/blaspp#137, icl-utk-edu/lapackpp#92 | forks only |
| Re-configuring a build directory without losing the settings BLAS++ and LAPACK++ detected | icl-utk-edu/blaspp#141 and #142, icl-utk-edu/lapackpp#94 | forks only |
| `lapack_symbol_suffix`, for symbol-suffixed LAPACK libraries | icl-utk-edu/lapackpp#91 | forks only |
| The `gemqrt` fix, without which an icpx-built LAPACK++ crashes | icl-utk-edu/lapackpp#93 | forks only |
| Apple's new Accelerate interface | icl-utk-edu/blaspp#134, icl-utk-edu/lapackpp#88 | upstream `master` and forks |
| LAPACK++ compiling against Accelerate's ILP64 | icl-utk-edu/lapackpp#90 | upstream `master` and forks |
| Discovery of an ILP64 OpenBLAS (`libopenblas64`), and `blas_symbol_suffix` | icl-utk-edu/blaspp#129 | upstream `master` and forks |
| The MSVC build fixes | icl-utk-edu/blaspp#132, icl-utk-edu/lapackpp#87 | upstream `master` and forks |

Each fork's `BALLISTIC.md`
([blaspp](https://github.com/BallisticLA/blaspp/blob/main/BALLISTIC.md),
[lapackpp](https://github.com/BallisticLA/lapackpp/blob/main/BALLISTIC.md))
lists what it carries. Issue
[#204](https://github.com/BallisticLA/RandLAPACK/issues/204) lists each fix
with its upstream pull request and tracks the return to an upstream release.
[CONTRIBUTING.md](../CONTRIBUTING.md#moving-the-blas-and-lapack-pins)
describes how a pin is moved.

### RandBLAS is a pinned submodule

The RandBLAS copy inside this repository is pinned to an exact commit, and
that pinned copy is the only RandBLAS configuration RandLAPACK is developed
and tested against. Two rules follow:

1. **Do not develop RandBLAS inside the submodule checkout.** To work on
   RandBLAS itself, clone it separately
   (`git clone https://github.com/BallisticLA/RandBLAS.git`) and build and
   install it as its own project. RandLAPACK does not use an installed
   RandBLAS, so nothing you install can leak into RandLAPACK builds: the
   submodule stays authoritative.
2. **Do not point RandLAPACK at your own RandBLAS working copy.** Version
   skew between RandLAPACK and RandBLAS is not a supported configuration.

The one exception is for *package maintainers* (Spack, conda-forge and
similar), whose recipes must build RandBLAS as its own package. Configuring
RandLAPACK with `-DRandLAPACK_EXTERNAL_RandBLAS=ON` uses an installed RandBLAS
instead of the submodule. A configure-time check fails unless that install
was built from exactly the commit the submodule pins, so version skew stays
impossible.

### CUDA and GCC versions

GPU support needs:

- the CUDA Toolkit, 12.4.1 or later (12.9.0 recommended);
- an NVIDIA driver, v580 or later recommended for CUDA 12.9.0;
- a GCC version your CUDA release accepts: CUDA has strict GCC version
  requirements.

| CUDA | GCC | Status | Notes |
|---|---|---|---|
| 13.3.0 | up to 15.x | Supported | Tested with GCC 15.2 |
| **12.9.0** | **13.x** (up to 14.x) | **Recommended** | Tested with GCC 13.3.0 and driver v581.80; `nvcc` rejects GCC above 14 |
| 12.4.1 | 13.x | Supported | Minimum supported version |
| 12.2.1 | 12.x | Limited | GCC 12.3.0 or older only |
| 12.2.1 | 13.x | Incompatible | `nvcc` error: "unsupported GNU version" |

The table is not exhaustive. The Supported Compilers table in NVIDIA's CUDA
Installation Guide for Linux lists the host compilers each CUDA release
supports, and the limits `nvcc` enforces may differ from what is documented.
If you hit compiler version errors, try an older GCC.

We recommend **CUDA 12.9.0 with GCC 13.3.0**, for best compatibility and to
avoid known issues in earlier CUDA versions (CUDA 12.4.x, for example, had
problems parsing Intel AMX intrinsics with GCC 13.x). Verified working as of
2025-11-26: CUDA 12.9.0, GCC 13.3.0, CMake 3.31.9 and driver v581.80.

## 8. Troubleshooting

- **Compare with CI.** RandLAPACK's GitHub Actions workflows install it from
  scratch and run its unit tests; [CI.md](CI.md) describes each lane. The
  [workflow files](https://github.com/BallisticLA/RandLAPACK/tree/main/.github/workflows)
  are a working reference. The core workflows print the working directory
  and list its contents at several points, so you can copy a directory
  layout that works.
- **Building BLAS++'s own tester.** The recipes above turn it off
  (`-Dbuild_tests=OFF`). If you want it, set `CPATH` to the backend's
  `include/` folder, so that CMake finds CBLAS.
- **CUDA rejects your compiler.** See the table in
  [Section 7](#cuda-and-gcc-versions).
