# Make sure to enable the script via "chmod +x install.sh"
#
# This script automatically installs RandLAPACK library with all of its dependencies, as well as builds the RandLAPACK benchmark files (done separately).
# The project layout will be as such: the directory where the RandLAPACK project was originally located will contain the top-level "RandNLA-project" priject direcort with three subdirectories: 
# lib: contains library files for RandLAPACK, blaspp, and lapackpp; 
# install: will contain the installed RandLAPACK-install, blaspp-install, lapackpp-install and random123;
# build: will contain builds for RandLAPACK-build, benchmark-build, blaspp-build, lapackpp-build.
# Prerequisits for installation can be seen in the INSTALL.md file.
#!/bin/bash
# Stop execution on error
set -e

# Check for GCC version
PREFERRED_GCC_VERSION="13.2.0"
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

GPU_AVAIL="none"
# Detect NVIDIA GPU
echo "Detecting a GPU..." | tee -a $LOG_FILE
if command -v nvidia-smi &> /dev/null; then
    # NVIDIA GPU found. Ask user if they want to proceed with GPU support or not.
    read -p "NVIDIA GPU detected. Would you like to build libraries with GPU support? (CUDA-only option available for now) (y/n): " user_input
    if [[ "$user_input" != "y" && "$user_input" != "Y" && "$user_input" != "yes" ]]; then
        echo "Building libraries without GPU support."
        GPU_AVAIL="none"
    else
        echo "Building libraries with GPU support."
        GPU_AVAIL="auto"
    fi
elif lspci | grep -i "VGA" | grep -i "AMD" &> /dev/null; then
    # AND GPU found. Ask user if they want to proceed with GPU support or not.
    read -p "AMD GPU detected. Would you like to build libraries with GPU support? (CUDA-only option available for now) (y/n): " user_input
    if [[ "$user_input" != "y" && "$user_input" != "Y" && "$user_input" != "yes" ]]; then
        echo "Building libraries without GPU support."
        GPU_AVAIL="none"
    else
        echo "Building libraries with GPU support."
        GPU_AVAIL="auto"
    fi
else
    echo "No NVIDIA GPU detected." | tee -a $LOG_FILE
fi

if [[ "$GPU_AVAIL" == "auto" ]]; then
    # Check for NVCC version
    PREFERRED_NVCC_VERSION="12.4.1"
    CURRENT_NVCC_VERSION=$(nvcc --version 2>/dev/null | head -n 1 | awk '{print $NF}')

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

# Consider adding a block that interacts with BLAS and LAPACK library versions
# that are available on the given system.
# For now, before installing the RandLAPACK project, 
# consider adding the proper BLAS and LAPACK libraries to PATH, LDPATH and CPATH like such:
# export BLAS_LAPACK_ROOT=*Point to the top-level library directory*
# export LIBRARY_PATH=$BLAS_LAPACK_ROOT/lib
# export LD_LIBRARY_PATH=$BLAS_LAPACK_ROOT/lib
# export CPATH=$BLAS_LAPACK_ROOT/include

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
    RANDNLA_PROJECT_DIR_ABSOLUTE_PATH=$(realpath "$RANDNLA_PROJECT_DIR")
    echo "Adding variable $RANDNLA_PROJECT_DIR to ~/.bashrc."
    echo "#Added via RandLAPACK/install.sh" >> ~/.bashrc
    echo "export RANDNLA_PROJECT_DIR=\"$RANDNLA_PROJECT_DIR_ABSOLUTE_PATH\"" >> ~/.bashrc
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
if [[ ! -d "$RANDNLA_PROJECT_DIR/build/benchmark-build/" ]]; then
    mkdir -p "$RANDNLA_PROJECT_DIR/build/benchmark-build/"
    echo "Directory created at: $RANDNLA_PROJECT_DIR/build/benchmark-build/"
else 
    rm -rf $RANDNLA_PROJECT_DIR/build/benchmark-build/*
    echo "Directory cleared at: $RANDNLA_PROJECT_DIR/build/benchmark-build/"
fi

# Initialize and update RandLAPACK submodule -- RandBLAS
git -C $SCRIPT_DIR submodule init; git -C $SCRIPT_DIR submodule update

if [[ ! -d "$RANDNLA_PROJECT_DIR/lib/RandLAPACK" ]]; then
    # Move RandLAPACK in its intended location
    mv $PARENT_DIR/RandLAPACK $PARENT_DIR/RandNLA-project/lib/
fi

# Obtain BLAS++ and LAPACK++
if [[ ! -d "$RANDNLA_PROJECT_DIR/lib/lapackpp" ]]; then
git clone https://github.com/icl-utk-edu/lapackpp         $RANDNLA_PROJECT_DIR/lib/lapackpp
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/lib/blaspp" ]]; then
git clone https://github.com/icl-utk-edu/blaspp           $RANDNLA_PROJECT_DIR/lib/blaspp
fi
if [[ ! -d "$RANDNLA_PROJECT_DIR/install/random123" ]]; then
git clone https://github.com/DEShawResearch/random123.git $RANDNLA_PROJECT_DIR/install/random123
fi

echo "All libraries placed in: $RANDNLA_PROJECT_DIR/lib"

# Configure, build, and install BLAS++
# Add "-DBLAS_LIBRARIES='-lflame -lblis'" if using AMD AOCL
cmake  -S $RANDNLA_PROJECT_DIR/lib/blaspp/ -B $RANDNLA_PROJECT_DIR/build/blaspp-build/ -Dgpu_backend=$GPU_AVAIL  -DCMAKE_BUILD_TYPE=Release -Dblas_int=int64 -DCMAKE_INSTALL_PREFIX=$RANDNLA_PROJECT_DIR/install/blaspp-install/ 
make  -C $RANDNLA_PROJECT_DIR/build/blaspp-build/ -j20 install

# Check if lib or lib64 folder name will be in use
LIB_VAR="lib"
if [[ -d "$RANDNLA_PROJECT_DIR/install/blaspp-install/lib" ]]; then
    LIB_VAR="lib"
elif [[ -d "$RANDNLA_PROJECT_DIR/install/blaspp-install/lib64" ]]; then
    LIB_VAR="lib64"
fi

# Configure, build, and install LAPACK++
# Add "-DBLAS_LIBRARIES='-lflame -lblis'" if using AMD AOCL
cmake  -S $RANDNLA_PROJECT_DIR/lib/lapackpp/ -B $RANDNLA_PROJECT_DIR/build/lapackpp-build/ -Dgpu_backend=$GPU_AVAIL -DCMAKE_BUILD_TYPE=Release  -Dblaspp_DIR=$RANDNLA_PROJECT_DIR/install/blaspp-install/$LIB_VAR/cmake/blaspp/  -DCMAKE_INSTALL_PREFIX=$RANDNLA_PROJECT_DIR/install/lapackpp-install
make  -C $RANDNLA_PROJECT_DIR/build/lapackpp-build/ -j20 install
# Configure, build, and install RandLAPACK
echo $LIB_VAR
cmake  -S $RANDNLA_PROJECT_DIR/lib/RandLAPACK/ -B $RANDNLA_PROJECT_DIR/build/RandLAPACK-build/ -DCMAKE_BUILD_TYPE=Release  -Dlapackpp_DIR=$RANDNLA_PROJECT_DIR/install/lapackpp-install/$LIB_VAR/cmake/lapackpp/ -Dblaspp_DIR=$RANDNLA_PROJECT_DIR/install/blaspp-install/$LIB_VAR/cmake/blaspp/  -DRandom123_DIR=$RANDNLA_PROJECT_DIR/install/random123/include/  -DCMAKE_INSTALL_PREFIX=$RANDNLA_PROJECT_DIR/install/RandLAPACK-install
make  -C $RANDNLA_PROJECT_DIR/build/RandLAPACK-build/ -j20 install
# Configure and build RandLAPACK-benchmark
cmake  -S $RANDNLA_PROJECT_DIR/lib/RandLAPACK/benchmark/ -B $RANDNLA_PROJECT_DIR/build/benchmark-build/  -DCMAKE_BUILD_TYPE=Release  -DRandLAPACK_DIR=$RANDNLA_PROJECT_DIR/install/RandLAPACK-install/$LIB_VAR/cmake/ -Dlapackpp_DIR=$RANDNLA_PROJECT_DIR/install/lapackpp-install/$LIB_VAR/cmake/lapackpp/ -Dblaspp_DIR=$RANDNLA_PROJECT_DIR/install/blaspp-install/$LIB_VAR/cmake/blaspp/ -DRandom123_DIR=$RANDNLA_PROJECT_DIR/install/random123/include/
make  -C $RANDNLA_PROJECT_DIR/build/benchmark-build/ -j20

# Source from bash and spawn a new shell so that the variable change takes place
bash -c "source ~/.bashrc && exec bash"