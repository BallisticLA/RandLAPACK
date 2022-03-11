# Installing and using RandLAPACK

Sections 1 through 3 of this file describe how to perform a *basic* installation
of RandLAPACK and its dependencies.

Section 4 explains how RandLAPACK can be used in other CMake projects.

Section 5 gives detailed recommendations on configuring BLAS++ and LAPACK++
for use with RandLAPACK.
Its installation instructions for BLAS++ and LAPACK++ can be used in place
of the corresponding instructions in Section 1.

*We recommend that you not bother with Section 5 the first time you build RandLAPACK.*


## 1. Optional dependencies

GoogleTest is Google’s C++ testing and mocking framework.  GTest is an optional
dependency without which RandLAPACK regression tests will not be available. GTest
can be installed with your favorite package manager.

OpemMP is an open standard that enables code to be parallelized as it is
compiled. RandLAPACK (and its dependencies) detects the presence of OpenMP
automatically and makes use of it if it's found.

## 2. Required Dependencies: BLAS++, LAPACK++, RandBLAS, and Random123.

BLAS++ and LAPACK++ are C++ wrappers for BLAS and LAPACK libraries.
They provide a portability layer and numeric type templating.
While these libraries can be built by GNU make, they also have a CMake
build system, *and RandLAPACK requires that they be installed via CMake.*

RandBLAS is a C++ library for sketching, which is the most basic operation
in randomized numerical linear algebra.
It requires BLAS++ and Random123.
Right now we list RandBLAS and Random123 as separate dependencies, but in time
we'll hide the Random123 dependency entirely within RandBLAS.

We give recipes for installing BLAS++, LAPACK++, and Random123 below.
Later on, we'll assume these recipes were executed from a directory
that contains (or will contain) the ``RandLAPACK`` project directory as a subdirectory.

One can compile and install BLAS++ from
[source](https://bitbucket.org/icl/blaspp/src/master/) using CMake by running
```shell
git clone https://bitbucket.org/icl/blaspp.git
mkdir blaspp-build
cd blaspp-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../blaspp-install \
    -DCMAKE_BINARY_DIR=`pwd` \
    -Dbuild_tests=OFF \
    ../blaspp
make -j2 install
```

One can compile and install LAPACK++ from
[source](https://bitbucket.org/icl/lapackpp/src/master/) using CMake by running
```shell
git clone https://bitbucket.org/icl/lapackpp.git
mkdir lapackpp-build
cd lapackpp-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/blaspp \
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

One can compile and install RandBLAS from [source](https://github.com/BallisticLA/RandBLAS)
by running
```shell
git clone https://github.com/BallisticLA/RandLAPACK.git
mkdir RandBLAS-build
cd RandBLAS-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/blaspp/ \
    -DRandom123_DIR=`pwd`/../random123-install/include/ \
    -DCMAKE_BINARY_DIR=`pwd` \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../RandBLAS-install \
    ../RandBLAS/
make -j install
ctest  # run unit tests (only if GTest was found by CMake)
```

## 3. Building and installing RandLAPACK

RandLAPACK is configured with CMake and built with GNU make.
The configuration and build processes are simple once its dependencies are in place. 

Assuming you used the recipes from Section 2 to get RandLAPACK's dependencies,
you can build download, build, and install RandLAPACK as follows:

```shell
git clone https://github.com/BallisticLA/RandLAPACK.git
mkdir RandLAPACK-build
cd RandLAPACK-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/blaspp/ \
    -Dlapackpp_DIR=`pwd`/../lapackpp-install/lib/lapackpp/ \
    -DRandBLAS_DIR=`pwd`/../RandBLAS-install/lib/cmake/ \
    -DCMAKE_BINARY_DIR=`pwd` \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../RandLAPACK-install \
    ../RandLAPACK/
make -j install
ctest  # run unit tests (only if GTest was found by CMake)
```

Here are the conceptual meanings in the recipe's build flags:

* `-Dblaspp_DIR=W` means `W` is the directory containing the file `blasppConfig.cmake`.
   Similarly, `-Dlapackpp_DIR=X` means `X` is the directory containing `lapackppConfig.cmake`.
   
    If you follow BLAS++ installation instructions from Section 5 instead of
    Section 1, then you'd set ``-Dblaspp_DIR=/opt/mklpp/lib/blaspp`` and
    `-Dlapackpp_DIR=/opt/mklpp/lib/lapackpp`. (You'd also need to use this value
    of `-Dblaspp_DIR` when building LAPACK++ in the first place.)
    Recall that we do not recommend that you follow Section 5 the first time you
    build RandLAPACK.

* `-DRandBLAS_DIR=Y` means `Y` is the directory containing `RandBLASConfig.cmake`.

* `-DCMAKE_INSTALL_PREFIX=Z` means subdirectories within `Z` will contain
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
   The minimal change is to set the environment variable `MKLROOT`  to something like
   `~/intel/oneapi/mkl/latest`. The preferred change is to execute an MKL-provided script
   that changes several environment variables automatically. That script is usually called
   `setvars.sh`. Riley's bashrc file was updated to contain the line
   ```
   source ~/intel/oneapi/setvars.sh
   ```
1. Download BLAS++ source, create a new folder called ``build``
   at the top level of the BLAS++ project directory, and ``cd`` into that
   folder.
2. Run ``export CXX=gcc`` so that ``gcc`` is the default compiler for
   the current bash session.
3. Decide a common prefix for where you'll put BLAS++ and LAPACK++
   installation files. We recommend ``/opt/mklpp``.
4. Run the following CMake command 
    ```shell
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/mklpp \
        -DBLAS_LIBRARIES='-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread' \
        -Dbuild_tests=OFF ..
    ```
    Note how the MKL BLAS and threading libraries are specified explicitly with the ``-DBLAS_LIBRARIES`` flag.
    Using that flag is in contrast with simply setting ``-Dblas=mkl``,
    in which case the BLAS++ CMake recipe tries to configure MKL for you.
5. Run ``cmake --build .``
6. Run ``sudo make -j2 install``
7. Download LAPACK++ source, create a new folder called ``build`` at the top level
   of the LAPACK++ project directory, and ``cd`` into that folder.
8. Run the following CMake command
   ```shell
    cmake -DCMAKE_BUILD_TYPE=Release \
       -Dblaspp_DIR=/opt/mklpp/lib/blaspp \
       -DCMAKE_INSTALL_PREFIX=/opt/mklpp \
       -DCMAKE_BINARY_DIR=`pwd` \
       -Dbuild_tests=OFF ..
    make -j2 install
    ```

You can then link to BLAS++ and LAPACK++ in other CMake projects
just by including ``find_package(blaspp)`` and ``find_package(lapackpp)``
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
