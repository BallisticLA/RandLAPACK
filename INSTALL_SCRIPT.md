# Using RandLAPACK's Automated Install Script

This guide explains how to use the `install.sh` script to automatically install
RandLAPACK and all of its dependencies (BLAS++, LAPACK++, Random123) with a
single command.

**When to use this guide:** Use this automated installation method if you want
a quick, streamlined setup process. If you need fine-grained control over
dependency configurations, refer to RandLAPACK's `INSTALL.md` instead.

## 0. Software Requirements

Before running the install script, ensure you have the following software
available on your system:

### Essential Requirements
* **C++ Compiler:** GNU GCC 13.3.0 or higher (required for C++20 features)
* **CMake:** Version 3.27 or higher
* **BLAS/LAPACK Library:** Intel MKL 2022 or higher recommended
* **GoogleTest:** (Optional but recommended) For running RandLAPACK tests

### GPU Support Requirements (Optional)
* **CUDA Toolkit:** Version 12.4.1 or higher
  - **Recommended:** CUDA 12.9.0 + GCC 13.3.0 (verified working as of 2025-11-26)
  - **IMPORTANT:** CUDA versions have strict GCC compatibility requirements:
    - CUDA 12.9.0: Compatible with GCC 13.x ✓
    - CUDA 12.4.1: Compatible with GCC 13.x ✓
    - CUDA 12.2.1: Requires GCC ≤ 12.x (GCC 13.x will fail with "unsupported GNU version")
  - See `INSTALL.md` Section 0 for full compatibility matrix
  - Ensure compatible NVIDIA driver (v580+ recommended for CUDA 12.9)
* **CUDA Libraries:** cuBLAS and cuSOLVER (included with CUDA Toolkit)

### Installing Requirements with Spack

We strongly recommend using [Spack](https://github.com/spack/spack) to manage
these dependencies. A typical Spack installation would look like:

```shell
# Step 1: Install the compiler FIRST
spack install gcc@13.3.0

# Step 2: Register the new compiler with Spack
spack compiler find

# Step 3: Load the compiler
spack load gcc@13.3.0

# Step 4: Install all other dependencies using the new compiler
spack install cmake@3.27
spack install intel-oneapi-mkl
spack install googletest

# For GPU support
spack install cuda@12.9.0
```

**IMPORTANT:** The compiler must be installed, registered with `spack compiler find`,
and loaded *before* installing other dependencies. This ensures all packages are
built with the correct compiler version. Spack will automatically use the loaded
compiler for subsequent package installations.

After installation, load the environment:
```shell
spack load gcc@13.3.0
spack load cmake
spack load intel-oneapi-mkl
spack load googletest
spack load cuda@12.9.0  # If GPU support needed
```

**Pro tip:** Add the spack load commands to your `~/.bashrc` to automatically
load the environment in every shell session. Make sure to load the compiler first
in your `.bashrc`.

## 1. Preparing for Installation

### Directory Structure

The install script expects a specific directory structure:

```
~/RandNLA/
├── RandLAPACK/          # Clone RandLAPACK here (script will move it)
└── RandNLA-project/     # Created automatically by script
    ├── lib/
    │   ├── blaspp/      # Built by script
    │   ├── lapackpp/    # Built by script
    │   ├── random123/   # Built by script
    │   └── RandLAPACK/  # Moved here by script
    └── build/           # Build artifacts
```

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

3. **(Important)** Switch to the correct development branch if needed:
   ```shell
   git checkout <branch-name>
   ```

   **Note:** Always verify with the development team which branch to use for
   the latest GPU support and stability improvements.

## 2. Running the Install Script

### Basic Usage

From inside the `RandLAPACK` directory:

```shell
bash install.sh
```

The script will:
1. Detect if GPU hardware is available on your system
2. If GPU is detected, prompt: `"GPU detected. Would you like to enable GPU support? (yes/no)"`
3. Automatically clone and build all dependencies
4. Build RandLAPACK with appropriate configuration
5. Build test and benchmark executables

### Automated Installation (Non-Interactive)

If you want to automatically answer "yes" to all prompts (useful for scripts):

```shell
yes | bash install.sh
```

### Installation with Logging

To capture the entire installation process in a log file:

```shell
yes | bash install.sh 2>&1 | tee install_log.txt
```

This creates `install_log.txt` with complete build output, which is invaluable
for troubleshooting if issues arise.

## 3. What the Script Does

The `install.sh` script performs the following steps automatically:

1. **Creates Project Structure**
   - Creates `~/RandNLA/RandNLA-project/` directory tree
   - Sets up subdirectories for libraries and build artifacts

2. **Builds BLAS++**
   - Clones BLAS++ from official repository
   - Configures with appropriate BLAS backend (MKL if available)
   - Builds with GPU support if requested
   - Installs to `~/RandNLA/RandNLA-project/lib/blaspp/`

3. **Builds LAPACK++**
   - Clones LAPACK++ from official repository
   - Configures to use previously built BLAS++
   - Builds with GPU support if requested
   - Installs to `~/RandNLA/RandNLA-project/lib/lapackpp/`

4. **Installs Random123**
   - Clones Random123 header-only library
   - Installs headers to `~/RandNLA/RandNLA-project/lib/random123/`

5. **Moves and Builds RandLAPACK**
   - Moves `RandLAPACK` directory to `~/RandNLA/RandNLA-project/lib/`
   - Configures CMake with all dependency paths
   - Builds RandLAPACK library
   - Builds test suite and benchmarks
   - Creates executables in `~/RandNLA/RandNLA-project/build/RandLAPACK-build/bin/`

## 4. Verifying the Installation

### Running Tests

After installation completes, verify everything works correctly:

```shell
cd ~/RandNLA/RandNLA-project/build/RandLAPACK-build
ctest
```

This runs the complete test suite (456 tests). Expected output:
```
99% tests passed, 1 tests failed out of 456
Total Test time (real) = 124.62 sec
```

**Note:** Some test failures are known and acceptable in development branches.
Consult the development team if you see unexpected failures.

### Running GPU Tests Only

If you enabled GPU support, test GPU functionality specifically:

```shell
./bin/RandLAPACK_tests_gpu
```

Expected output: 13-14 GPU tests should pass within 15-20 seconds.

## 5. Working with the Installed Project

### Key File Locations

After installation:

* **RandLAPACK library:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/libRandLAPACK.a`
* **Headers:** `~/RandNLA/RandNLA-project/lib/RandLAPACK/RandLAPACK/`
* **Tests:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/bin/RandLAPACK_tests*`
* **Benchmarks:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/bin/RandLAPACK_bench*`
* **CMake config:** `~/RandNLA/RandNLA-project/build/RandLAPACK-build/RandLAPACKConfig.cmake`

### Recompiling After Code Changes

If you modify RandLAPACK source code:

```shell
cd ~/RandNLA/RandNLA-project/build/RandLAPACK-build
source ~/.bashrc  # Ensures environment is loaded
make -j
```

**Important:** Always source your `.bashrc` (or equivalent environment setup)
before running `make` to ensure CUDA libraries and other dependencies are in
your `LD_LIBRARY_PATH`.

### Using RandLAPACK in Your Own Projects

See Section 4 of `INSTALL.md` for details on linking RandLAPACK to external
CMake projects. You'll need to specify:

```cmake
-Dblaspp_DIR=~/RandNLA/RandNLA-project/lib/blaspp/lib/cmake/blaspp
-Dlapackpp_DIR=~/RandNLA/RandNLA-project/lib/lapackpp/lib/cmake/lapackpp
-DRandBLAS_DIR=~/RandNLA/RandNLA-project/build/RandLAPACK-build/RandBLAS
-DRandLAPACK_DIR=~/RandNLA/RandNLA-project/build/RandLAPACK-build
```

---
