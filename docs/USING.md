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
project(myexec)

find_package(RandLAPACK REQUIRED)

add_executable(myexec main.cc)
target_link_libraries(myexec RandLAPACK::RandLAPACK)
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

From the folder that contains `myexec/`:

```shell
cmake -S myexec -B myexec-build -DCMAKE_BUILD_TYPE=Release \
    -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK
cmake --build myexec-build
./myexec-build/myexec
```

- **`RandLAPACK_DIR`** names the directory that contains
  `RandLAPACKConfig.cmake`; give it as an absolute path. The installers
  print it at the end of their summary. After `install.sh` with the default
  layout, it is `RandNLA-project/install/RandLAPACK-install/lib/cmake/RandLAPACK`
  next to your clone (`lib64` instead of `lib` on some systems).
- **It is the only flag you need.** The exported target
  `RandLAPACK::RandLAPACK` brings RandBLAS, LAPACK++ and BLAS++ along as
  link dependencies, and RandLAPACK's config records where it found them
  and Random123.
- **`-DCMAKE_BUILD_TYPE=Release`** turns on the compiler's optimizations.
  RandLAPACK is header-only, so it is compiled as part of your program, and
  without a build type it is compiled without optimization.
- **With oneMKL**, load its environment (`source <oneapi>/setvars.sh`) in
  every shell where you build or run a program that uses RandLAPACK. BLAS++
  records oneMKL by library name only, so the linker and the loader find it
  through that environment.
- **On Windows**, your executable needs the runtime DLLs next to it;
  [INSTALL_WINDOWS.md, section 7](INSTALL_WINDOWS.md#7-runtime-dlls-what-staging-means)
  shows the one CMake line that copies them.

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
