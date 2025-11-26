# Installing and using RandLAPACK

Sections 1 through 3 of this file describe how to perform a *basic* installation
of RandLAPACK and its dependencies.

Section 4 explains how RandLAPACK can be used in other CMake projects.

Section 5 gives detailed recommendations on configuring BLAS++ and LAPACK++
for use with RandLAPACK.
Its installation instructions for BLAS++ and LAPACK++ can be used in place
of the corresponding instructions in Section 1.

*We recommend that you not bother with Section 5 the first time you build RandLAPACK.*

## 0. Software requirements

### Minimum Requirements
- **CMake**: 3.21 or higher (3.31.9 recommended)
- **C++ Compiler**: C++20 support required
  - GCC 11 or higher
  - Clang 14 or higher (not extensively tested)
  - Intel ICPX (has known issues, see GitHub issue #91)

### GPU Support (Optional)
For GPU/CUDA support (enabled with `-DRequireCUDA=ON`), you need:
- **CUDA Toolkit**: 12.4.1 or higher
- **NVIDIA Driver**: Compatible with your CUDA version
- **GCC Compatibility**: CUDA has strict GCC version requirements

#### CUDA/GCC Compatibility Matrix

| CUDA Version | GCC Support | Status | Notes |
|--------------|-------------|---------|-------|
| 12.9.0 | GCC 13.x ✓ | ✅ Verified | Tested with GCC 13.3.0 (2025-11-24) |
| 12.4.1 | GCC 13.x ✓ | ✅ Recommended | Documented working configuration |
| 12.2.1 | GCC 12.x ✓ | ✅ Works | Use GCC ≤ 12.3.0 |
| 12.2.1 | GCC 13.x ✗ | ❌ Fails | NVCC error: "unsupported GNU version" |

**Important**: CUDA's `nvcc` compiler has strict GCC version limits that may differ from what's documented. Always check your CUDA Toolkit's release notes for compiler compatibility. If you encounter compiler version errors, try using an older GCC version.

**Tested Configuration** (as of 2025-11-26):
- CUDA 12.9.0 + GCC 13.3.0 + CMake 3.31.9 + Driver v581.80 ✅

### Note on Directory Names
On some systems, library directories are called `lib` while on others they're called `lib64`. Adjust paths accordingly in the CMake configuration commands below.

We recomment installing software (including googletest, if desired) using Spack:
https://github.com/spack/spack.git

## 1. Optional dependencies

GoogleTest is Google’s C++ testing and mocking framework.  GTest is an optional
dependency without which RandLAPACK regression tests will not be available. GTest
can be installed with your favorite package manager.

OpemMP is an open standard that enables code to be parallelized as it is
compiled. RandLAPACK (and its dependencies) detects the presence of OpenMP
automatically and makes use of it if it's found.

CUDA support can be enabled using -DRequireCUDA=ON flag.
It is disabled by default.

## 2. Required Dependencies: BLAS++, LAPACK++, and Random123.

BLAS++ and LAPACK++ are C++ wrappers for BLAS and LAPACK libraries.
They provide a portability layer and numeric type templating.
While these libraries can be built by GNU make, they also have a CMake
build system, *and RandLAPACK requires that they be installed via CMake.*

RandLAPACK's git repository includes a C++ project called *RandBLAS* as a git submodule.
RandBLAS has BLAS++ and Random123 as dependencies.

We give recipes for installing BLAS++, LAPACK++, and Random123 below.
Later on, we'll assume these recipes were executed from a directory
that contains (or will contain) the ``RandLAPACK`` project directory as a subdirectory.

One can compile and install BLAS++ from
[source](https://bitbucket.org/icl/blaspp/src/master/) using CMake by running
```shell
git clone https://github.com/icl-utk-edu/blaspp.git
mkdir blaspp-build
cd blaspp-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../blaspp-install \
    -DCMAKE_BINARY_DIR=`pwd` \
    -Dbuild_tests=OFF \
    -Dblas_int=int64 \
    ../blaspp
make -j2 install
```

If you wish for BLAS++ tester to be built, make sure that the CPATH is set properly,
i.e. pointing at the BLAS vendor library's /include/ folder.
This will ensure that CBLAS is properly encountered by CMake.

One can compile and install LAPACK++ from
[source](https://bitbucket.org/icl/lapackpp/src/master/) using CMake by running
```shell
git clone https://github.com/icl-utk-edu/lapackpp.git
mkdir lapackpp-build
cd lapackpp-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/cmake/blaspp \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../lapackpp-install \
    -DCMAKE_BINARY_DIR=`pwd` \
    -Dbuild_tests=OFF \
    ../lapackpp
make -j2 install
```

One can install Random123 from
[source](https://github.com/DEShawResearch/random123) by running
```shell
git clone https://github.com/DEShawResearch/random123.git
cd random123/
make prefix=`pwd`/../random123-install install-include
```

## 3. Building and installing RandLAPACK

RandLAPACK is configured with CMake and built with GNU make.
The configuration and build processes are simple once its dependencies are in place. 

Assuming you used the recipes from Section 2 to get RandLAPACK's dependencies,
you can build download, build, and install RandLAPACK as follows
(add -DRequireCUDA=ON if you need CUDA support):

```shell
git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
mkdir RandLAPACK-build
cd RandLAPACK-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dlapackpp_DIR=`pwd`/../lapackpp-install/lib/cmake/lapackpp/ \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/cmake/blaspp/ \
    -DRandom123_DIR=`pwd`/../random123-install/include/ \
    -DCMAKE_BINARY_DIR=`pwd` \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../RandLAPACK-install \
    ../RandLAPACK/
make -j install
```
You can run all tests with
```shell
ctest
```
All tests should complete reasonably quickly. Benchmarks are kept separate in the `benchmark/` directory and are not run by `ctest`.

Here are the conceptual meanings in the recipe's build flags:

* `-Dlapackpp_DIR=X` means `X` is the directory containing `lapackppConfig.cmake`.
    If you follow BLAS++ installation instructions from Section 5 instead of
    Section 1, then you'd set `-Dlapackpp_DIR=/opt/mklpp/lib/lapackpp`.
  
* `-Dblaspp_DIR=X` means `X` is the directory containing the file `blasppConfig.cmake`.

* `-DRandom123_DIR=X` means `X` is the directory that contains a folder called ``Random123``
  that includes the Random123 header files. For example, ``X/Random123/philox.h`` needs
  to be a file on your system.

* `-DCMAKE_INSTALL_PREFIX=X` means subdirectories within `X` will contain
   the RandLAPACK binaries, header files, and CMake configuration files needed
   for using RandLAPACK in other projects. You should make note of the directory
   that ends up containing the file ``RandLAPACKConfig.cmake``.


## 4. Using RandLAPACK in other projects

Once RandLAPACK has been compiled and installed it can be used like any other CMake project.
For instance, the following CMakeLists.txt demonstrates how an executable can
be linked to the RandLAPACK library:

```cmake
cmake_minimum_required(VERSION 3.0)
project(myexec)

find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)
find_package(RandBLAS REQUIRED)
find_package(RandLAPACK REQUIRED)

add_executable(myexec ...)
target_link_libraries(myexec RandLAPACK RandBLAS lapackpp blaspp ...)
```
In order to build that CMake project you'd need to specify a build flags
 * `-Dblaspp_DIR=W`, where `W` contains the file `blasppConfig.cmake`.
 * `-Dlapackpp_DIR=X`, where `X` contains the file `lapackppConfig.cmake`.
 * `-DRandBLAS_DIR=Y`, where `Y` contains the file `RandBLASConfig.cmake`.
 * `-DRandLAPACK_DIR=Z`, where `Z` contains the file `RandLAPACKConfig.cmake`. 


## 5. Tips

### Pay attention to the BLAS++ configuration

The performance of RandLAPACK depends heavily on how its dependencies are configured.
Its most sensitive dependency is BLAS++.
If performance matters to you then you should inspect the
information that's printed to screen when you run ``cmake`` for the BLAS++ installation.
Save that information somewhere while you're setting up your RandLAPACK
development environment.

### Recommended BLAS++ and LAPACK++ configuration

We recommend you install BLAS++ and LAPACK++ so they link to Intel MKL
version 2022 or higher.
That version of MKL will come with CMake configuration files.
Those configuration files are extremely useful if
you want to make a project that connects RandLAPACK and Intel MKL.
Such a situation might arise if you want to use RandLAPACK together with
MKL's sparse linear algebra functionality.

One of the RandLAPACK developers (Riley) has run into trouble
getting BLAS++ to link to MKL as intended.
Here's how Riley configured his BLAS++ and LAPACK++ installations:

0. Install and configure MKL. You can get MKL [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=webdownload&options=online).
   Once you've installed it you need to edit your `.bashrc` file.
   Riley's bashrc file was updated to contain the line
   ```
   export MAIN_MKL_LIBS="/home/riley/intel/oneapi/mkl/latest/lib/intel64"
   export LD_LIBRARY_PATH="${MAIN_MKL_LIBS}:${LD_LIBRARY_PATH}"
   export LIBRARY_PATH="${MAIN_MKL_LIBS}:${LIBRARY_PATH}"
   ```

1. Download BLAS++ source, create a new folder called ``build``
   at the top level of the BLAS++ project directory, and ``cd`` into that
   folder.

2. Run ``export CXX=gcc`` so that ``gcc`` is the default compiler for
   the current bash session.

3. Decide a common prefix for where you'll put BLAS++ and LAPACK++
   installation files. We recommend ``/opt/mklpp``.

4. Run the following CMake command 
    ```
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/mklpp \
        -Dblas=mkl \
        -Dblas_int=int64 \
        -Dbuild_tests=OFF ..
    ```
   Save the output of that command somewhere. It contains information
   on the precise BLAS libraries linked to BLAS++.

5. Run ``cmake --build .``

6. Run ``sudo make install``

7. Download LAPACK++ source, create a new folder called ``build`` at the top level
   of the LAPACK++ project directory, and ``cd`` into that folder.

8. Run the following CMake command
   ```
    cmake -DCMAKE_BUILD_TYPE=Release \
       -Dblaspp_DIR=/opt/mklpp/lib/blaspp \
       -DCMAKE_INSTALL_PREFIX=/opt/mklpp \
       -DCMAKE_BINARY_DIR=`pwd` \
       -Dbuild_tests=OFF ..
    make -j2 install
    ```

You can then link to BLAS++ and LAPACK++ in other CMake projects
just by including ``find_package(blaspp REQUIRED)`` and ``find_package(lapackpp REQUIRED)``
in your ``CMakeLists.txt`` file, and then passing build flags
```
-Dblaspp_DIR=/opt/mklpp/lib/blaspp -Dlapackpp_DIR=/opt/mklpp/lib/lapackpp
```
when running ``cmake``.

### Installation trouble

RandLAPACK has a GitHub Actions workflow to install it from scratch and run its suite of unit tests.
If you're having trouble installing RandLAPACK, you can always refer to [that workflow file](https://github.com/BallisticLA/RandLAPACK/tree/main/.github/workflows).
The workflow includes statements which print the working directory
and list the contents of that directory at various points in the installation.
We do that so that it's easier to infer a valid choice of directory structure for building RandLAPACK.
