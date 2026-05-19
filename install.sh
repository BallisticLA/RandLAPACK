#!/bin/bash
# Make sure to enable the script via "chmod +x install.sh"
#
# This script automatically installs RandLAPACK library with all of its dependencies, as well as builds the RandLAPACK benchmark files (done separately).
# The project layout will be as such: the directory where the RandLAPACK project was originally located will contain the top-level "RandNLA-project" project direcory with three subdirectories:
# lib: contains library files for RandLAPACK, blaspp, and lapackpp;
# install: will contain the installed RandLAPACK-install, blaspp-install, lapackpp-install and random123;
# build: will contain builds for RandLAPACK-build, benchmark-build, blaspp-build, lapackpp-build.
# Prerequisites for installation can be seen in the INSTALL.md file.
# Stop execution on error
set -e

# Determine the appropriate shell config file:
#   zsh (default on macOS)       → ~/.zshrc
#   bash on macOS (login shell)  → ~/.bash_profile  (macOS terminals don't source ~/.bashrc)
#   bash on Linux                → ~/.bashrc
if [[ "$(basename "${SHELL:-bash}")" == "zsh" ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$(uname)" == "Darwin" ]]; then
    SHELL_RC="$HOME/.bash_profile"
else
    SHELL_RC="$HOME/.bashrc"
fi

# Check for GCC version
PREFERRED_GCC_VERSION="13.3.0"
CURRENT_GCC_VERSION=$(gcc --version 2>/dev/null | head -n 1 | awk '{print $NF}')

if [[ "$CURRENT_GCC_VERSION" != "$PREFERRED_GCC_VERSION" ]]; then
    echo "Warning: GCC $PREFERRED_GCC_VERSION is preferred. Found GCC $CURRENT_GCC_VERSION."
    echo "Consider installing GCC $PREFERRED_GCC_VERSION before running this script."

    # Ask the user if they want to continue or terminate the script
    read -p "Do you want to continue with the current GCC version? (y/n): " user_input
    if [[ "$user_input" != "y" && "$user_input" != "Y" && "$user_input" != "yes" ]]; then
        echo "Terminating script. Please install GCC $PREFERRED_GCC_VERSION and try again."
        exit 1
    fi
fi

RELOAD_SHELL=0
RANDLAPACK_CUDA="OFF"
RANDNLA_PROJECT_GPU_AVAIL="none"
# Detect NVIDIA GPU
echo "Detecting a GPU..."
if command -v nvidia-smi &> /dev/null; then
    # NVIDIA GPU found. Ask user if they want to proceed with GPU support or not.
    read -p "NVIDIA GPU detected. Would you like to build libraries with GPU support? (CUDA-only option available for now) (y/n): " user_input
    if [[ "$user_input" != "y" && "$user_input" != "Y" && "$user_input" != "yes" ]]; then
        echo "Building libraries without GPU support."
        RANDNLA_PROJECT_GPU_AVAIL="none"
    else
        echo "Building libraries with GPU support."
        RANDNLA_PROJECT_GPU_AVAIL="auto"
        RANDLAPACK_CUDA="ON"
        # We need to add the RANDNLA_PROJECT_GPU_AVAIL variable to bashrc so that it can be used in our other scripts
        if ! grep -q "export RANDNLA_PROJECT_GPU_AVAIL=" $SHELL_RC; then
            echo "#Added via RandLAPACK/install.sh" >> $SHELL_RC
            echo "export RANDNLA_PROJECT_GPU_AVAIL=\"auto\"" >> $SHELL_RC
            RELOAD_SHELL=1
        fi
    fi
elif { command -v lspci &>/dev/null && lspci | grep -i "VGA" | grep -qi "AMD"; } || \
     { [[ "$(uname)" == "Darwin" ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -qi "AMD"; }; then
    # AMD GPU found. Ask user if they want to proceed with GPU support or not.
    read -p "AMD GPU detected. Would you like to build libraries with GPU support? (CUDA-only option available for now) (y/n): " user_input
    if [[ "$user_input" != "y" && "$user_input" != "Y" && "$user_input" != "yes" ]]; then
        echo "Building libraries without GPU support."
        RANDNLA_PROJECT_GPU_AVAIL="none"
    else
        echo "Building libraries with GPU support."
        RANDLAPACK_CUDA="ON"
        RANDNLA_PROJECT_GPU_AVAIL="auto"
        # We need to add the RANDNLA_PROJECT_GPU_AVAIL variable to bashrc so that it can be used in our other scripts
        if ! grep -q "export RANDNLA_PROJECT_GPU_AVAIL=" $SHELL_RC; then
            echo "#Added via RandLAPACK/install.sh" >> $SHELL_RC
            echo "export RANDNLA_PROJECT_GPU_AVAIL=\"auto\"" >> $SHELL_RC
            RELOAD_SHELL=1
        fi
    fi
else
    echo "No NVIDIA GPU detected."
fi

if [[ "$RANDNLA_PROJECT_GPU_AVAIL" == "auto" ]]; then
    # Check for NVCC version
    PREFERRED_NVCC_VERSION="12.9"
    CURRENT_NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1)

    if [[ "$CURRENT_NVCC_VERSION" != "$PREFERRED_NVCC_VERSION" ]]; then
        echo "Warning: NVCC $PREFERRED_NVCC_VERSION is preferred. Found NVCC $CURRENT_NVCC_VERSION."
        echo "Consider installing NVCC $PREFERRED_NVCC_VERSION before running this script."

        # Ask the user if they want to continue or terminate the script
        read -p "Do you want to continue with the current NVCC version? (y/n): " user_input
        if [[ "$user_input" != "y" && "$user_input" != "Y" ]]; then
            echo "Terminating script. Please install NVCC $PREFERRED_NVCC_VERSION and try again."
            exit 1
        fi
    fi
fi

# On macOS, OpenBLAS must be installed via Homebrew before running this script:
#   brew install openblas
# On Linux, ensure your BLAS/LAPACK installation (MKL, AOCL, OpenBLAS, etc.) is on PATH.
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ ! -f /opt/homebrew/opt/openblas/lib/libopenblas.dylib ]]; then
        echo "ERROR: OpenBLAS not found. Install it first: brew install openblas"
        exit 1
    fi
    if [[ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]]; then
        echo "ERROR: libomp not found. Install it first: brew install libomp"
        exit 1
    fi
fi

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
# Get the parent directory (one level above the script)
PARENT_DIR=$(dirname "$SCRIPT_DIR")
PARENT_BASE=$(basename "$(dirname "$SCRIPT_DIR")")
# Define the project directory
if [[ "$PARENT_BASE" == "lib" ]]; then
    # Project already exists
    RANDNLA_PROJECT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
else
    # New project library to be created
    RANDNLA_PROJECT_DIR="$PARENT_DIR/RandNLA-project"
    # We want to make sure that RANDNLA_PROJECT_DIR variable is in the
    # user's bashrc so that it can be used by our other bash scripts.
    # $RANDNLA_PROJECT_DIR is already absolute (derived from realpath of script path);
    # avoid calling realpath on a path that may not exist yet (macOS BSD realpath requires existence)
    RANDNLA_PROJECT_DIR_ABSOLUTE_PATH="$RANDNLA_PROJECT_DIR"
    if ! grep -q "export RANDNLA_PROJECT_DIR=" $SHELL_RC; then
        echo "#Added via RandLAPACK/install.sh" >> $SHELL_RC
        echo "export RANDNLA_PROJECT_DIR=\"$RANDNLA_PROJECT_DIR_ABSOLUTE_PATH\"" >> $SHELL_RC
        RELOAD_SHELL=1
    fi
fi

# Create the project directory and its subdirectories
if [[ ! -d "$RANDNLA_PROJECT_DIR" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR"
    echo "Directory created at: $RANDNLA_PROJECT_DIR"
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/install/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/install/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/install/"
else 
    echo "Directory exists at: $RANDNLA_PROJECT_DIR/install/"
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/lib/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/lib/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/lib/"
else 
    echo "Directory exists at: $RANDNLA_PROJECT_DIR/lib/"
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/build/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/build/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/build/"
else 
    echo "Directory exists at: $RANDNLA_PROJECT_DIR/build/"
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/build/blaspp-build/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/build/blaspp-build/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/build/blaspp-build/"
else 
    rm -rf $RANDNLA_PROJECT_DIR/build/blaspp-build/*
    echo "Directory cleared at: $RANDNLA_PROJECT_DIR/build/blaspp-build/"
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/build/lapackpp-build/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/build/lapackpp-build/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/build/lapackpp-build/"
else 
    rm -rf $RANDNLA_PROJECT_DIR/build/lapackpp-build/*
    echo "Directory cleared at: $RANDNLA_PROJECT_DIR/build/lapackpp-build/"
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/build/RandLAPACK-build/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/build/RandLAPACK-build/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/build/RandLAPACK-build/"
else 
    rm -rf $RANDNLA_PROJECT_DIR/build/RandLAPACK-build/*
    echo "Directory cleared at: $RANDNLA_PROJECT_DIR/build/RandLAPACK-build/"
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/build/extras-build/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/build/extras-build/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/build/extras-build/"
else
    rm -rf $RANDNLA_PROJECT_DIR/build/extras-build/*
    echo "Directory cleared at: $RANDNLA_PROJECT_DIR/build/extras-build/"
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/build/benchmark-build/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/build/benchmark-build/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/build/benchmark-build/"
else
    rm -rf $RANDNLA_PROJECT_DIR/build/benchmark-build/*
    echo "Directory cleared at: $RANDNLA_PROJECT_DIR/build/benchmark-build/"
fi

#==============================================================================
# Discovery phase: probe for already-installed dependencies via env vars.
# If <DEP>_INSTALL_DIR is set and points at a valid install, skip clone+build
# and reuse it. Otherwise fall back to the fresh clone+build path below.
# Supported env vars:
#   BLASPP_INSTALL_DIR    -- root of blaspp install (containing lib/cmake/blaspp/ or lib64/...)
#   LAPACKPP_INSTALL_DIR  -- root of lapackpp install
#   RANDOM123_INSTALL_DIR -- root containing include/Random123/
# RandBLAS is intentionally not covered here -- it stays a submodule.
#==============================================================================

# find_cmake_config <install_root> <pkg_name>
# Echoes the directory containing <pkg_name>Config.cmake, or empty if not found.
# Always returns 0; the caller checks for an empty result (avoids tripping set -e).
find_cmake_config() {
    local root="$1"
    local pkg="$2"
    local libdir
    for libdir in lib lib64 lib/x86_64-linux-gnu; do
        if [[ -f "$root/$libdir/cmake/$pkg/${pkg}Config.cmake" ]]; then
            echo "$root/$libdir/cmake/$pkg/"
            return 0
        fi
    done
    return 0
}

USE_EXTERNAL_BLASPP=false
USE_EXTERNAL_LAPACKPP=false
USE_EXTERNAL_RANDOM123=false
BLASPP_CMAKE_DIR=""
LAPACKPP_CMAKE_DIR=""
RANDOM123_DIR=""
BLASPP_LIB_DIR=""
LAPACKPP_LIB_DIR=""

echo "=========================================="
echo "Dependency discovery..."
echo "=========================================="

if [[ -n "${BLASPP_INSTALL_DIR:-}" ]]; then
    BLASPP_CMAKE_DIR=$(find_cmake_config "$BLASPP_INSTALL_DIR" "blaspp")
    if [[ -n "$BLASPP_CMAKE_DIR" ]]; then
        USE_EXTERNAL_BLASPP=true
        BLASPP_LIB_DIR=$(dirname "$(dirname "$BLASPP_CMAKE_DIR")")
        echo "  [blaspp]    Using external install at $BLASPP_INSTALL_DIR"
        echo "              CMake config: $BLASPP_CMAKE_DIR"
    else
        echo "  [blaspp]    BLASPP_INSTALL_DIR=$BLASPP_INSTALL_DIR set but blasppConfig.cmake not found; will build from source."
    fi
else
    echo "  [blaspp]    No BLASPP_INSTALL_DIR set; will build from source."
fi

if [[ -n "${LAPACKPP_INSTALL_DIR:-}" ]]; then
    LAPACKPP_CMAKE_DIR=$(find_cmake_config "$LAPACKPP_INSTALL_DIR" "lapackpp")
    if [[ -n "$LAPACKPP_CMAKE_DIR" ]]; then
        USE_EXTERNAL_LAPACKPP=true
        LAPACKPP_LIB_DIR=$(dirname "$(dirname "$LAPACKPP_CMAKE_DIR")")
        echo "  [lapackpp]  Using external install at $LAPACKPP_INSTALL_DIR"
        echo "              CMake config: $LAPACKPP_CMAKE_DIR"
    else
        echo "  [lapackpp]  LAPACKPP_INSTALL_DIR=$LAPACKPP_INSTALL_DIR set but lapackppConfig.cmake not found; will build from source."
    fi
else
    echo "  [lapackpp]  No LAPACKPP_INSTALL_DIR set; will build from source."
fi

if [[ -n "${RANDOM123_INSTALL_DIR:-}" ]]; then
    if [[ -f "$RANDOM123_INSTALL_DIR/include/Random123/philox.h" ]]; then
        USE_EXTERNAL_RANDOM123=true
        RANDOM123_DIR="$RANDOM123_INSTALL_DIR/include/"
        echo "  [random123] Using external install at $RANDOM123_INSTALL_DIR"
    else
        echo "  [random123] RANDOM123_INSTALL_DIR=$RANDOM123_INSTALL_DIR set but include/Random123/philox.h not found; will clone from source."
    fi
else
    echo "  [random123] No RANDOM123_INSTALL_DIR set; will clone from source."
fi
echo ""

# Initialize and update RandLAPACK submodule -- RandBLAS
git -C $SCRIPT_DIR submodule init; git -C $SCRIPT_DIR submodule update

if [[ ! -d "$RANDNLA_PROJECT_DIR/lib/RandLAPACK" ]]; then
    # Move RandLAPACK in its intended location (use $SCRIPT_DIR to support any clone folder name)
    mv "$SCRIPT_DIR" "$PARENT_DIR/RandNLA-project/lib/RandLAPACK"
fi

# Obtain BLAS++ and LAPACK++ (skip the clones for any dep discovered above)
if [[ "$USE_EXTERNAL_LAPACKPP" != "true" && ! -d "$RANDNLA_PROJECT_DIR/lib/lapackpp" ]]; then
git clone https://github.com/icl-utk-edu/lapackpp         $RANDNLA_PROJECT_DIR/lib/lapackpp
fi
if [[ "$USE_EXTERNAL_BLASPP" != "true" && ! -d "$RANDNLA_PROJECT_DIR/lib/blaspp" ]]; then
git clone https://github.com/icl-utk-edu/blaspp           $RANDNLA_PROJECT_DIR/lib/blaspp
fi
if [[ "$USE_EXTERNAL_RANDOM123" != "true" && ! -d "$RANDNLA_PROJECT_DIR/install/random123" ]]; then
git clone https://github.com/DEShawResearch/random123.git $RANDNLA_PROJECT_DIR/install/random123
fi

echo "All libraries placed in: $RANDNLA_PROJECT_DIR/lib"

# Configure, build, and install BLAS++
# Add "-DBLAS_LIBRARIES='-lflame -lblis'" if using AMD AOCL
# On macOS, Homebrew OpenBLAS is LP64 (int32); Linux typically has ILP64 (int64) BLAS available.
# The macOS CLT does not expose C++ stdlib headers in its default search path — export CXXFLAGS
# so cmake picks them up via CMAKE_CXX_FLAGS_INIT for all subsequent cmake invocations.
BLAS_INT="int64"
MACOS_BLAS_FLAGS=""
MACOS_LAPACK_FLAGS=""
MACOS_OPENMP_FLAGS=""
if [[ "$(uname)" == "Darwin" ]]; then
    BLAS_INT="int32"
    MACOS_SDK_PATH=$(xcrun --show-sdk-path)
    # SDK C++ headers + Apple Clang OpenMP flags (no native OpenMP; use Homebrew libomp).
    # Appending to CXXFLAGS/CFLAGS lets cmake pick them up via CMAKE_CXX_FLAGS_INIT /
    # CMAKE_C_FLAGS_INIT for all try_compile tests, including FindOpenMP.
    export CXXFLAGS="-isystem ${MACOS_SDK_PATH}/usr/include/c++/v1 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
    export CFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
    # Homebrew OpenBLAS is keg-only; pass full path and Fortran mangling directly.
    MACOS_BLAS_FLAGS="-DBLAS_LIBRARIES=/opt/homebrew/opt/openblas/lib/libopenblas.dylib -Dblas_fortran=add"
    # OpenBLAS bundles LAPACK; point lapackpp at the same library.
    MACOS_LAPACK_FLAGS="-DLAPACK_LIBRARIES=/opt/homebrew/opt/openblas/lib/libopenblas.dylib"
    # Explicit hints for cmake's FindOpenMP (C and CXX).
    # Flags use semicolon cmake-list syntax to avoid bash word-splitting on spaces.
    MACOS_OPENMP_FLAGS="-DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib -DOpenMP_C_FLAGS=-Xpreprocessor;-fopenmp -DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp"
fi
if [[ "$USE_EXTERNAL_BLASPP" != "true" ]]; then
    cmake  -S $RANDNLA_PROJECT_DIR/lib/blaspp/ -B $RANDNLA_PROJECT_DIR/build/blaspp-build/ \
        -Dgpu_backend=$RANDNLA_PROJECT_GPU_AVAIL \
        -DCMAKE_BUILD_TYPE=Release \
        -Dblas_int=$BLAS_INT \
        -DCMAKE_INSTALL_PREFIX=$RANDNLA_PROJECT_DIR/install/blaspp-install/ \
        $MACOS_BLAS_FLAGS $MACOS_OPENMP_FLAGS
    make  -C $RANDNLA_PROJECT_DIR/build/blaspp-build/ -j20 install

    BLASPP_CMAKE_DIR=$(find_cmake_config "$RANDNLA_PROJECT_DIR/install/blaspp-install" "blaspp")
    BLASPP_LIB_DIR=$(dirname "$(dirname "$BLASPP_CMAKE_DIR")")
fi

# Configure, build, and install LAPACK++
# Add "-DBLAS_LIBRARIES='-lflame -lblis'" if using AMD AOCL
if [[ "$USE_EXTERNAL_LAPACKPP" != "true" ]]; then
    cmake  -S $RANDNLA_PROJECT_DIR/lib/lapackpp/ -B $RANDNLA_PROJECT_DIR/build/lapackpp-build/ -Dgpu_backend=$RANDNLA_PROJECT_GPU_AVAIL -DCMAKE_BUILD_TYPE=Release  -Dblaspp_DIR=$BLASPP_CMAKE_DIR  -DCMAKE_INSTALL_PREFIX=$RANDNLA_PROJECT_DIR/install/lapackpp-install -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON $MACOS_LAPACK_FLAGS $MACOS_OPENMP_FLAGS
    make  -C $RANDNLA_PROJECT_DIR/build/lapackpp-build/ -j20 install

    LAPACKPP_CMAKE_DIR=$(find_cmake_config "$RANDNLA_PROJECT_DIR/install/lapackpp-install" "lapackpp")
    LAPACKPP_LIB_DIR=$(dirname "$(dirname "$LAPACKPP_CMAKE_DIR")")
fi

# random123 is header-only; pin RANDOM123_DIR to the local clone if we're not using an external install
if [[ "$USE_EXTERNAL_RANDOM123" != "true" ]]; then
    RANDOM123_DIR="$RANDNLA_PROJECT_DIR/install/random123/include/"
fi
# Configure, build, and install RandLAPACK
echo "=========================================="
echo "Configuring and building RandLAPACK..."
echo "=========================================="
cmake  -S $RANDNLA_PROJECT_DIR/lib/RandLAPACK/ -B $RANDNLA_PROJECT_DIR/build/RandLAPACK-build/ -DCMAKE_BUILD_TYPE=Release -DRequireCUDA=$RANDLAPACK_CUDA -Dlapackpp_DIR=$LAPACKPP_CMAKE_DIR -Dblaspp_DIR=$BLASPP_CMAKE_DIR -DRandom123_DIR=$RANDOM123_DIR -DCMAKE_INSTALL_PREFIX=$RANDNLA_PROJECT_DIR/install/RandLAPACK-install -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON $MACOS_OPENMP_FLAGS
if [ $? -ne 0 ]; then
    echo "ERROR: RandLAPACK configuration failed!"
    exit 1
fi
make  -C $RANDNLA_PROJECT_DIR/build/RandLAPACK-build/ -j20 install
if [ $? -ne 0 ]; then
    echo "ERROR: RandLAPACK build failed!"
    exit 1
fi
RANDLAPACK_CMAKE_DIR=$(find_cmake_config "$RANDNLA_PROJECT_DIR/install/RandLAPACK-install" "RandLAPACK")
RANDLAPACK_LIB_DIR=$(dirname "$(dirname "$RANDLAPACK_CMAKE_DIR")")
echo "RandLAPACK configured and built successfully"
echo ""

# If GPU support is disabled AND we're building blaspp from source (not using an
# external install), prevent extras and benchmarks from auto-detecting CUDA.
# When using an external blaspp install, the external install's configuration
# dictates whether CUDAToolkit is required (its config calls find_dependency on it).
DISABLE_CUDA_FLAG=""
if [[ "$RANDLAPACK_CUDA" == "OFF" && "$USE_EXTERNAL_BLASPP" != "true" ]]; then
    DISABLE_CUDA_FLAG="-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE"
fi

# Configure and build RandLAPACK-extras
echo "=========================================="
echo "Configuring and building RandLAPACK extras..."
echo "=========================================="
cmake  -S $RANDNLA_PROJECT_DIR/lib/RandLAPACK/extras/ -B $RANDNLA_PROJECT_DIR/build/extras-build/ -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR=$RANDNLA_PROJECT_DIR/build/fetchcontent-cache/ -DRandLAPACK_DIR=$RANDLAPACK_CMAKE_DIR -Dlapackpp_DIR=$LAPACKPP_CMAKE_DIR -Dblaspp_DIR=$BLASPP_CMAKE_DIR -DRandom123_DIR=$RANDOM123_DIR -DCMAKE_BUILD_RPATH="$BLASPP_LIB_DIR;$LAPACKPP_LIB_DIR;$RANDLAPACK_LIB_DIR" $DISABLE_CUDA_FLAG $MACOS_OPENMP_FLAGS
if [ $? -ne 0 ]; then
    echo "ERROR: RandLAPACK extras configuration failed!"
    exit 1
fi
make  -C $RANDNLA_PROJECT_DIR/build/extras-build/ -j20
if [ $? -ne 0 ]; then
    echo "ERROR: RandLAPACK extras build failed!"
    exit 1
fi
echo "RandLAPACK extras configured and built successfully"
echo ""

# Configure and build RandLAPACK-benchmark
echo "=========================================="
echo "Configuring and building RandLAPACK benchmarks..."
echo "=========================================="
cmake  -S $RANDNLA_PROJECT_DIR/lib/RandLAPACK/benchmark/ -B $RANDNLA_PROJECT_DIR/build/benchmark-build/  -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR=$RANDNLA_PROJECT_DIR/build/fetchcontent-cache/ -DRandLAPACK_DIR=$RANDLAPACK_CMAKE_DIR -Dlapackpp_DIR=$LAPACKPP_CMAKE_DIR -Dblaspp_DIR=$BLASPP_CMAKE_DIR -DRandom123_DIR=$RANDOM123_DIR -DCMAKE_BUILD_RPATH="$BLASPP_LIB_DIR;$LAPACKPP_LIB_DIR;$RANDLAPACK_LIB_DIR" $DISABLE_CUDA_FLAG $MACOS_OPENMP_FLAGS
if [ $? -ne 0 ]; then
    echo "ERROR: RandLAPACK benchmarks configuration failed!"
    exit 1
fi
make  -C $RANDNLA_PROJECT_DIR/build/benchmark-build/ -j20
if [ $? -ne 0 ]; then
    echo "ERROR: RandLAPACK benchmarks build failed!"
    exit 1
fi
echo "RandLAPACK benchmarks configured and built successfully"
echo ""

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo "RandLAPACK, extras, and benchmarks are ready to use."
echo ""
echo "Extras executables: $RANDNLA_PROJECT_DIR/build/extras-build/"
echo "Benchmark executables: $RANDNLA_PROJECT_DIR/build/benchmark-build/"
echo ""

if [ $RELOAD_SHELL -eq 1 ]; then
    # Source the shell config and spawn a new shell so that the variable change takes place
    echo "Writing variables into $SHELL_RC"
    exec "${SHELL:-bash}" -c "source $SHELL_RC && exec ${SHELL:-bash}"
fi