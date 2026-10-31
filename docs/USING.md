# Using RandLAPACK from your own project

An installed RandLAPACK works like any other CMake package. This page shows
a minimal program that uses it, and how to build that program against an
install made by `install.sh`, `install.ps1` or a manual build.

## A minimal example

Put these two files in a folder of their own, say `myexec/`. The program
factors a random matrix with the `CQRRPT` driver (QR with column pivoting of
a tall matrix).

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.21)
project(myexec LANGUAGES CXX)

find_package(RandLAPACK REQUIRED)

add_executable(myexec main.cc)
target_link_libraries(myexec RandLAPACK::RandLAPACK)
randlapack_stage_runtime_dlls(myexec)   # no-op on non-Windows platforms
```

`main.cc`:

```cpp
#include <RandLAPACK.hh>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
    int64_t m = 1000, n = 50;
    std::vector<double> A(m * n), R(n * n);
    std::vector<int64_t> J(n);
    RandBLAS::RNGState<> state(0);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    // Pivoted QR, A[:, J] = Q R, with a sketch of 2n rows. A is overwritten by Q.
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    RandLAPACK::CQRRPT<double> cqrrpt(false, tol);
    cqrrpt.call(m, n, A.data(), m, R.data(), n, J.data(), 2.0, state);
    std::printf("rank %lld\n", (long long) cqrrpt.rank);
}
```

Matrices are column-major arrays. `RandBLAS::fill_dense` fills `A` with
random numbers, and the program prints `rank 50`.

## Building it

On Linux and macOS, from the folder that contains `myexec/`:

```shell
cmake -S myexec -B myexec-build -DCMAKE_BUILD_TYPE=Release \
    -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK
cmake --build myexec-build
./myexec-build/myexec
```

On Windows, from an x64 developer prompt such as "x64 Native Tools Command
Prompt for VS 2022"
([INSTALL_WINDOWS.md](INSTALL_WINDOWS.md#starting-from-the-right-shell)
explains which prompt):

```bat
cmake -S myexec -B myexec-build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    "-DRandLAPACK_DIR=<printed by install.ps1>"
cmake --build myexec-build
myexec-build\myexec.exe
```

`-G Ninja` matters on Windows: the default Visual Studio generator ignores
`CMAKE_BUILD_TYPE` and puts the executable in a `Debug\` or `Release\`
subfolder.

- **`RandLAPACK_DIR`** names the directory that contains
  `RandLAPACKConfig.cmake`; give it as an absolute path. The installers
  print it at the end of their summary. After `install.sh` with the default
  layout, it is `RandNLA-project/install/RandLAPACK-install/lib/cmake/RandLAPACK`
  next to your clone (`lib64` instead of `lib` on some systems).
- **It is the only path you need to give.** The exported target
  `RandLAPACK::RandLAPACK` brings RandBLAS, LAPACK++ and BLAS++ along as
  link dependencies, and RandLAPACK's config records where it found them
  and Random123. Two cases need more: if RandLAPACK was built with
  `-DRandLAPACK_EXTERNAL_RandBLAS=ON`, also pass `-DRandBLAS_DIR`; and on
  macOS with Homebrew's `libomp`, add the settings
  [below](#macos-with-homebrews-libomp).
- **`-DCMAKE_BUILD_TYPE=Release`** turns on the compiler's optimizations.
  RandLAPACK is header-only, so it is compiled as part of your program, and
  without a build type it is compiled without optimization.
- **With oneMKL on Linux or macOS**, load its environment
  (`source <oneapi>/setvars.sh`) in every shell where you build or run a
  program that uses RandLAPACK. BLAS++ records oneMKL by library name only,
  so the linker and the loader find it through that environment. On
  Windows, `install.ps1` links oneMKL by full path and its DLLs are copied
  next to your executable (next item), so no environment script is needed.
- **On Windows**, `randlapack_stage_runtime_dlls(myexec)` copies the runtime
  DLLs your executable needs next to it, so it runs from any shell; on other
  platforms the line does nothing.
  [INSTALL_WINDOWS.md, section 7](INSTALL_WINDOWS.md#7-runtime-dlls-what-staging-means)
  explains this staging, and the one DLL it does not copy (MSVC's OpenMP
  runtime, needed with OpenBLAS and OpenMP).

## macOS with Homebrew's `libomp`

A RandLAPACK built with that OpenMP requires your project to find it too,
and Apple Clang does not find it without help. If your configure stops at
"Could NOT find OpenMP", set what `install.sh` sets for every project it
configures. This is a copy of the installer's settings, not a separately
tested recipe:

```shell
LIBOMP=$(brew --prefix)/opt/libomp
export CFLAGS="-Xpreprocessor -fopenmp -I$LIBOMP/include"
export CXXFLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1 $CFLAGS"
export LDFLAGS="-L$LIBOMP/lib"
cmake ... -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp \
    -DOpenMP_omp_LIBRARY=$LIBOMP/lib/libomp.dylib \
    "-DOpenMP_C_FLAGS=-Xpreprocessor;-fopenmp" \
    "-DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp"
```

## Thread caps in the iterative least-squares solvers

When RandBLAS is built with MKL support (`RandBLAS_HAS_MKL`), the iterative
least-squares solvers (`lsqr`, `Blendenpik_linops`, `restarted_pcg_ne`,
`IterRefineLSQ`) cap MKL's thread count around their single-vector triangular
solves: 8 threads for factors of order up to 4000, 16 above, never more than
the caller's own width. A backward-error oracle passed to `restarted_pcg_ne`
or `IterRefineLSQ` runs under the same cap. Two environment variables, read
once per process, adjust this:

- `RANDLAPACK_BLAS2_THREADS=k` replaces the calibrated cap with `k`; `k <= 0`
  disables it.
- `RANDLAPACK_SOLVE_WIDTH_MATCH=0` stops operators applied inside a solver
  (such as the extras `ToeplitzLinOp`) from narrowing their own threading to
  the solver's width.

Other BLAS libraries need no cap: OpenBLAS runs these solves on one thread,
and Accelerate has no per-call thread control.
