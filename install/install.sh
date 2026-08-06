#!/bin/bash
# RandLAPACK autoinstaller.
#
# Installs RandLAPACK with all of its dependencies and builds the extras and
# benchmark projects. The directory that contains the RandLAPACK clone ends up
# with a top-level "RandNLA-project" directory:
#   lib:     RandLAPACK, blaspp, lapackpp sources
#   install: RandLAPACK-install, blaspp-install, lapackpp-install, random123
#   build:   one build directory per project above
#
# Usage: bash install.sh [options]
#
#   -y, --yes             Assume "yes" for every prompt (also the behavior when
#                         stdin is not a terminal, e.g. curl | bash or CI).
#       --gpu             Build with CUDA support without asking.
#       --no-gpu          Build without GPU support without asking.
#   -j, --jobs <N>        Parallel build jobs (default: number of cores).
#       --fresh           Clear all build directories first. The default reuses
#                         them, so re-running after a failure or a source
#                         update is an incremental rebuild.
#       --modify-rc       Append RANDNLA_PROJECT_DIR / RANDNLA_PROJECT_GPU_AVAIL
#                         exports to your shell config. The default never
#                         touches your shell config; the final summary prints
#                         the export lines to add yourself if you want them.
#       --project-dir <D> Place/locate RandNLA-project at D instead of next to
#                         this clone.
#   -h, --help            Show this help and exit.
#
# Every option has an environment-variable equivalent (flags win):
#   RANDLAPACK_INSTALL_YES=1, RANDLAPACK_INSTALL_GPU=on|off,
#   RANDLAPACK_INSTALL_JOBS=N, RANDLAPACK_INSTALL_FRESH=1,
#   RANDLAPACK_INSTALL_MODIFY_RC=1, RANDLAPACK_INSTALL_PROJECT_DIR=D
#
# Already-installed dependencies are discovered through:
#   BLASPP_INSTALL_DIR, LAPACKPP_INSTALL_DIR, RANDOM123_INSTALL_DIR
# (RandBLAS is intentionally not covered: it stays a git submodule.)
#
# All compiler output goes to <project-dir>/install.log; the console shows one
# line per step. On failure the log path is printed.
#
# Prerequisites are listed in INSTALL.md.
set -euo pipefail

#==============================================================================
# Option parsing. Environment variables provide defaults; flags override.
#==============================================================================
ASSUME_YES="${RANDLAPACK_INSTALL_YES:-0}"
GPU_CHOICE="${RANDLAPACK_INSTALL_GPU:-ask}"       # ask | on | off
JOBS="${RANDLAPACK_INSTALL_JOBS:-}"
FRESH="${RANDLAPACK_INSTALL_FRESH:-0}"
MODIFY_RC="${RANDLAPACK_INSTALL_MODIFY_RC:-0}"
PROJECT_DIR_OVERRIDE="${RANDLAPACK_INSTALL_PROJECT_DIR:-}"

usage() { sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)        ASSUME_YES=1 ;;
        --gpu)           GPU_CHOICE="on" ;;
        --no-gpu)        GPU_CHOICE="off" ;;
        -j|--jobs)       JOBS="${2:?--jobs requires a number}"; shift ;;
        --jobs=*)        JOBS="${1#*=}" ;;
        --fresh)         FRESH=1 ;;
        --modify-rc)     MODIFY_RC=1 ;;
        --project-dir)   PROJECT_DIR_OVERRIDE="${2:?--project-dir requires a path}"; shift ;;
        --project-dir=*) PROJECT_DIR_OVERRIDE="${1#*=}" ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

# Prompts happen only on a terminal and only without --yes. When stdin is not
# a terminal (piped/CI), every prompt silently takes its default.
INTERACTIVE=0
if [[ -t 0 && "$ASSUME_YES" != "1" ]]; then
    INTERACTIVE=1
fi

# ask <question> <default y|n> -> returns 0 for yes.
ask() {
    local question="$1" default="$2" reply
    if [[ "$INTERACTIVE" != "1" ]]; then
        [[ "$default" == "y" ]]
        return
    fi
    read -r -p "$question [$( [[ $default == y ]] && echo Y/n || echo y/N )]: " reply
    reply="${reply:-$default}"
    [[ "$reply" == "y" || "$reply" == "Y" || "$reply" == "yes" ]]
}

if [[ -z "$JOBS" ]]; then
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
fi

# Plain output when not on a terminal or when NO_COLOR/TERM=dumb ask for it.
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
    C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_ERR=""; C_BOLD=""; C_OFF=""
fi

#==============================================================================
# Toolchain checks. Warn always; abort only if the user says so at a prompt.
#==============================================================================
PREFERRED_GCC_VERSION="13.3.0"
CURRENT_GCC_VERSION=$(gcc --version 2>/dev/null | head -n 1 | awk '{print $NF}')
if [[ "$CURRENT_GCC_VERSION" != "$PREFERRED_GCC_VERSION" ]]; then
    echo "Note: GCC $PREFERRED_GCC_VERSION is the reference version; found ${CURRENT_GCC_VERSION:-none}."
    if ! ask "Continue with the current GCC?" y; then
        echo "Stopping at your request. Install GCC $PREFERRED_GCC_VERSION and re-run."
        exit 1
    fi
fi

#==============================================================================
# GPU decision. --gpu/--no-gpu (or RANDLAPACK_INSTALL_GPU) decide outright;
# otherwise detection + prompt. Non-interactive defaults: NVIDIA detected ->
# GPU on; AMD or nothing detected -> GPU off (the CUDA-only build cannot
# succeed on AMD, so saying yes for the user would guarantee a failure).
#==============================================================================
RANDLAPACK_CUDA="OFF"
RANDNLA_PROJECT_GPU_AVAIL="none"
case "$GPU_CHOICE" in
    on)  RANDLAPACK_CUDA="ON";  RANDNLA_PROJECT_GPU_AVAIL="auto" ;;
    off) ;;
    ask)
        if command -v nvidia-smi &> /dev/null; then
            if ask "NVIDIA GPU detected. Build with CUDA support?" y; then
                RANDLAPACK_CUDA="ON"; RANDNLA_PROJECT_GPU_AVAIL="auto"
            fi
        elif { command -v lspci &>/dev/null && lspci | grep -i "VGA" | grep -qi "AMD"; } || \
             { [[ "$(uname)" == "Darwin" ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -qi "AMD"; }; then
            if ask "AMD GPU detected, but only a CUDA build is available for now. Attempt a CUDA build anyway?" n; then
                RANDLAPACK_CUDA="ON"; RANDNLA_PROJECT_GPU_AVAIL="auto"
            fi
        else
            echo "No GPU detected; building without GPU support."
        fi
        ;;
    *) echo "RANDLAPACK_INSTALL_GPU must be 'on' or 'off' (got '$GPU_CHOICE')" >&2; exit 2 ;;
esac

if [[ "$RANDNLA_PROJECT_GPU_AVAIL" == "auto" ]]; then
    PREFERRED_NVCC_VERSION="12.9"
    CURRENT_NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1)
    if [[ "$CURRENT_NVCC_VERSION" != "$PREFERRED_NVCC_VERSION" ]]; then
        echo "Note: NVCC $PREFERRED_NVCC_VERSION is the reference version; found ${CURRENT_NVCC_VERSION:-none}."
        if ! ask "Continue with the current NVCC?" y; then
            echo "Stopping at your request. Install NVCC $PREFERRED_NVCC_VERSION and re-run."
            exit 1
        fi
    fi
fi

#==============================================================================
# macOS preflight: Homebrew OpenBLAS + libomp, SDK C++ headers, OpenMP hints.
#==============================================================================
BLAS_INT="int64"
MACOS_BLAS_FLAGS=""
MACOS_LAPACK_FLAGS=""
MACOS_OPENMP_FLAGS=""
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ ! -f /opt/homebrew/opt/openblas/lib/libopenblas.dylib ]]; then
        echo "ERROR: OpenBLAS not found. Install it first: brew install openblas" >&2
        exit 1
    fi
    if [[ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]]; then
        echo "ERROR: libomp not found. Install it first: brew install libomp" >&2
        exit 1
    fi
    BLAS_INT="int32"
    MACOS_SDK_PATH=$(xcrun --show-sdk-path)
    # SDK C++ headers + Apple Clang OpenMP flags (no native OpenMP; Homebrew
    # libomp). Appending to CXXFLAGS/CFLAGS lets cmake pick them up via
    # CMAKE_<LANG>_FLAGS_INIT for all try_compile tests, including FindOpenMP.
    export CXXFLAGS="-isystem ${MACOS_SDK_PATH}/usr/include/c++/v1 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
    export CFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
    MACOS_BLAS_FLAGS="-DBLAS_LIBRARIES=/opt/homebrew/opt/openblas/lib/libopenblas.dylib -Dblas_fortran=add"
    MACOS_LAPACK_FLAGS="-DLAPACK_LIBRARIES=/opt/homebrew/opt/openblas/lib/libopenblas.dylib"
    MACOS_OPENMP_FLAGS="-DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib -DOpenMP_C_FLAGS=-Xpreprocessor;-fopenmp -DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp"
fi

#==============================================================================
# Project layout. The clone moves itself into <parent>/RandNLA-project/lib/
# on first run; on re-runs (script already under lib/) the layout is detected.
#==============================================================================
# This script lives in <repo>/install/; REPO_DIR is the RandLAPACK clone.
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
REPO_DIR=$(dirname "$SCRIPT_DIR")
PARENT_DIR=$(dirname "$REPO_DIR")
PARENT_BASE=$(basename "$PARENT_DIR")
if [[ -n "$PROJECT_DIR_OVERRIDE" ]]; then
    RANDNLA_PROJECT_DIR="$PROJECT_DIR_OVERRIDE"
elif [[ "$PARENT_BASE" == "lib" ]]; then
    RANDNLA_PROJECT_DIR=$(dirname "$PARENT_DIR")
else
    RANDNLA_PROJECT_DIR="$PARENT_DIR/RandNLA-project"
fi

mkdir -p "$RANDNLA_PROJECT_DIR"/{install,lib,build}
for d in blaspp-build lapackpp-build RandLAPACK-build extras-build benchmark-build; do
    if [[ "$FRESH" == "1" ]]; then
        rm -rf "$RANDNLA_PROJECT_DIR/build/$d"
    fi
    mkdir -p "$RANDNLA_PROJECT_DIR/build/$d"
done

LOG="$RANDNLA_PROJECT_DIR/install.log"
: > "$LOG"
echo "RandLAPACK install started $(date)" >> "$LOG"

# run_step <label> <command...>: one console line per step, full output in the
# log, log path printed on failure.
STEP=0
TOTAL_STEPS=10
run_step() {
    local label="$1"; shift
    STEP=$((STEP + 1))
    printf "%s[%d/%d]%s %s ... " "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$label"
    local t0 t1
    t0=$(date +%s)
    {
        echo ""
        echo "===== [$STEP/$TOTAL_STEPS] $label ====="
        echo "\$ $*"
    } >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        t1=$(date +%s)
        printf "%sdone%s (%ds)\n" "$C_OK" "$C_OFF" "$((t1 - t0))"
    else
        printf "%sFAILED%s\n" "$C_ERR" "$C_OFF" >&2
        echo "" >&2
        echo "Step '$label' failed. Full output: $LOG" >&2
        echo "The last 20 log lines:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
}

#==============================================================================
# Dependency discovery: reuse preinstalled deps pointed at by env vars.
#==============================================================================
find_cmake_config() {
    local root="$1" pkg="$2" libdir
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

# Idempotent re-runs: a dependency this project already built and installed
# is reused as if it were external, instead of re-driving its build. This is
# faster, and it also sidesteps an upstream blaspp defect where re-running
# cmake over an existing build directory regenerates blas/defines.h WITHOUT
# the Fortran-mangling and BLAS-backend defines (cached detection results
# skip the list-building code), which then breaks every downstream compile
# through LAPACK_GLOBAL. --fresh forces the full rebuild.
if [[ "$FRESH" != "1" ]]; then
    if [[ -z "${BLASPP_INSTALL_DIR:-}" ]]; then
        PRIOR=$(find_cmake_config "$RANDNLA_PROJECT_DIR/install/blaspp-install" "blaspp")
        if [[ -n "$PRIOR" ]]; then
            BLASPP_INSTALL_DIR="$RANDNLA_PROJECT_DIR/install/blaspp-install"
        fi
    fi
    if [[ -z "${LAPACKPP_INSTALL_DIR:-}" ]]; then
        PRIOR=$(find_cmake_config "$RANDNLA_PROJECT_DIR/install/lapackpp-install" "lapackpp")
        if [[ -n "$PRIOR" ]]; then
            LAPACKPP_INSTALL_DIR="$RANDNLA_PROJECT_DIR/install/lapackpp-install"
        fi
    fi
fi

echo "Dependency discovery:"
if [[ -n "${BLASPP_INSTALL_DIR:-}" ]]; then
    BLASPP_CMAKE_DIR=$(find_cmake_config "$BLASPP_INSTALL_DIR" "blaspp")
    if [[ -n "$BLASPP_CMAKE_DIR" ]]; then
        USE_EXTERNAL_BLASPP=true
        BLASPP_LIB_DIR=$(dirname "$(dirname "$BLASPP_CMAKE_DIR")")
        echo "  [blaspp]    external install: $BLASPP_INSTALL_DIR"
    else
        echo "  [blaspp]    BLASPP_INSTALL_DIR set but blasppConfig.cmake not found; building from source."
    fi
else
    echo "  [blaspp]    building from source (set BLASPP_INSTALL_DIR to reuse an install)."
fi
if [[ -n "${LAPACKPP_INSTALL_DIR:-}" ]]; then
    LAPACKPP_CMAKE_DIR=$(find_cmake_config "$LAPACKPP_INSTALL_DIR" "lapackpp")
    if [[ -n "$LAPACKPP_CMAKE_DIR" ]]; then
        USE_EXTERNAL_LAPACKPP=true
        LAPACKPP_LIB_DIR=$(dirname "$(dirname "$LAPACKPP_CMAKE_DIR")")
        echo "  [lapackpp]  external install: $LAPACKPP_INSTALL_DIR"
    else
        echo "  [lapackpp]  LAPACKPP_INSTALL_DIR set but lapackppConfig.cmake not found; building from source."
    fi
else
    echo "  [lapackpp]  building from source (set LAPACKPP_INSTALL_DIR to reuse an install)."
fi
if [[ -n "${RANDOM123_INSTALL_DIR:-}" ]]; then
    if [[ -f "$RANDOM123_INSTALL_DIR/include/Random123/philox.h" ]]; then
        USE_EXTERNAL_RANDOM123=true
        RANDOM123_DIR="$RANDOM123_INSTALL_DIR/include/"
        echo "  [random123] external install: $RANDOM123_INSTALL_DIR"
    else
        echo "  [random123] RANDOM123_INSTALL_DIR set but include/Random123/philox.h not found; cloning."
    fi
else
    echo "  [random123] cloning (set RANDOM123_INSTALL_DIR to reuse an install)."
fi

#==============================================================================
# Sources: submodule, self-move into the layout, dependency clones.
#==============================================================================
git -C "$REPO_DIR" submodule init >> "$LOG" 2>&1
git -C "$REPO_DIR" submodule update >> "$LOG" 2>&1

if [[ ! -d "$RANDNLA_PROJECT_DIR/lib/RandLAPACK" ]]; then
    mv "$REPO_DIR" "$RANDNLA_PROJECT_DIR/lib/RandLAPACK"
    echo "Moved this clone to $RANDNLA_PROJECT_DIR/lib/RandLAPACK"
fi

if [[ "$USE_EXTERNAL_LAPACKPP" != "true" && ! -d "$RANDNLA_PROJECT_DIR/lib/lapackpp" ]]; then
    git clone https://github.com/icl-utk-edu/lapackpp "$RANDNLA_PROJECT_DIR/lib/lapackpp" >> "$LOG" 2>&1
fi
if [[ "$USE_EXTERNAL_BLASPP" != "true" && ! -d "$RANDNLA_PROJECT_DIR/lib/blaspp" ]]; then
    git clone https://github.com/icl-utk-edu/blaspp "$RANDNLA_PROJECT_DIR/lib/blaspp" >> "$LOG" 2>&1
fi
if [[ "$USE_EXTERNAL_RANDOM123" != "true" && ! -d "$RANDNLA_PROJECT_DIR/install/random123" ]]; then
    git clone https://github.com/DEShawResearch/random123.git "$RANDNLA_PROJECT_DIR/install/random123" >> "$LOG" 2>&1
fi

#==============================================================================
# Builds. Each pair below is one configure step + one build step; skipped
# steps still advance the counter so the numbering is stable.
#==============================================================================
if [[ "$USE_EXTERNAL_BLASPP" != "true" ]]; then
    # Add "-DBLAS_LIBRARIES='-lflame -lblis'" here if using AMD AOCL.
    # The MACOS_* variables expand unquoted on purpose: they hold multiple
    # -D words, and some values are CMake lists with semicolons, which must
    # reach cmake verbatim (never re-parse them through a shell string).
    run_step "Configuring BLAS++" \
        cmake -S "$RANDNLA_PROJECT_DIR/lib/blaspp/" -B "$RANDNLA_PROJECT_DIR/build/blaspp-build/" \
            -Dgpu_backend=$RANDNLA_PROJECT_GPU_AVAIL \
            -DCMAKE_BUILD_TYPE=Release \
            -Dblas_int=$BLAS_INT \
            -DCMAKE_INSTALL_PREFIX="$RANDNLA_PROJECT_DIR/install/blaspp-install/" \
            $MACOS_BLAS_FLAGS $MACOS_OPENMP_FLAGS
    run_step "Building + installing BLAS++" \
        cmake --build "$RANDNLA_PROJECT_DIR/build/blaspp-build/" -j "$JOBS" --target install
    BLASPP_CMAKE_DIR=$(find_cmake_config "$RANDNLA_PROJECT_DIR/install/blaspp-install" "blaspp")
    BLASPP_LIB_DIR=$(dirname "$(dirname "$BLASPP_CMAKE_DIR")")
else
    STEP=$((STEP + 2)); echo "[$STEP/$TOTAL_STEPS] BLAS++ ... reused external install"
fi

if [[ "$USE_EXTERNAL_LAPACKPP" != "true" ]]; then
    run_step "Configuring LAPACK++" \
        cmake -S "$RANDNLA_PROJECT_DIR/lib/lapackpp/" -B "$RANDNLA_PROJECT_DIR/build/lapackpp-build/" \
            -Dgpu_backend=$RANDNLA_PROJECT_GPU_AVAIL \
            -DCMAKE_BUILD_TYPE=Release \
            -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
            -DCMAKE_INSTALL_PREFIX="$RANDNLA_PROJECT_DIR/install/lapackpp-install" \
            -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
            $MACOS_LAPACK_FLAGS $MACOS_OPENMP_FLAGS
    run_step "Building + installing LAPACK++" \
        cmake --build "$RANDNLA_PROJECT_DIR/build/lapackpp-build/" -j "$JOBS" --target install
    LAPACKPP_CMAKE_DIR=$(find_cmake_config "$RANDNLA_PROJECT_DIR/install/lapackpp-install" "lapackpp")
    LAPACKPP_LIB_DIR=$(dirname "$(dirname "$LAPACKPP_CMAKE_DIR")")
else
    STEP=$((STEP + 2)); echo "[$STEP/$TOTAL_STEPS] LAPACK++ ... reused external install"
fi

if [[ "$USE_EXTERNAL_RANDOM123" != "true" ]]; then
    RANDOM123_DIR="$RANDNLA_PROJECT_DIR/install/random123/include/"
fi

RL_SRC="$RANDNLA_PROJECT_DIR/lib/RandLAPACK"
run_step "Configuring RandLAPACK" \
    cmake -S "$RL_SRC" -B "$RANDNLA_PROJECT_DIR/build/RandLAPACK-build/" \
        -DCMAKE_BUILD_TYPE=Release \
        -DRequireCUDA=$RANDLAPACK_CUDA \
        -Dlapackpp_DIR="$LAPACKPP_CMAKE_DIR" \
        -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
        -DRandom123_DIR="$RANDOM123_DIR" \
        -DCMAKE_INSTALL_PREFIX="$RANDNLA_PROJECT_DIR/install/RandLAPACK-install" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -DBUILD_TESTS=OFF -DRandLAPACK_BUILD_TESTS=ON $MACOS_OPENMP_FLAGS
run_step "Building + installing RandLAPACK" \
    cmake --build "$RANDNLA_PROJECT_DIR/build/RandLAPACK-build/" -j "$JOBS" --target install
RANDLAPACK_CMAKE_DIR=$(find_cmake_config "$RANDNLA_PROJECT_DIR/install/RandLAPACK-install" "RandLAPACK")
RANDLAPACK_LIB_DIR=$(dirname "$(dirname "$RANDLAPACK_CMAKE_DIR")")

# If GPU support is disabled AND blaspp was built from source, keep extras and
# benchmarks from auto-detecting CUDA. With an external blaspp, its own config
# dictates whether CUDAToolkit is required.
DISABLE_CUDA_FLAG=""
if [[ "$RANDLAPACK_CUDA" == "OFF" && "$USE_EXTERNAL_BLASPP" != "true" ]]; then
    DISABLE_CUDA_FLAG="-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE"
fi

run_step "Configuring extras" \
    cmake -S "$RL_SRC/extras/" -B "$RANDNLA_PROJECT_DIR/build/extras-build/" \
        -DCMAKE_BUILD_TYPE=Release \
        -DFETCHCONTENT_BASE_DIR="$RANDNLA_PROJECT_DIR/build/fetchcontent-cache/" \
        -DRandLAPACK_DIR="$RANDLAPACK_CMAKE_DIR" \
        -Dlapackpp_DIR="$LAPACKPP_CMAKE_DIR" \
        -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
        -DRandom123_DIR="$RANDOM123_DIR" \
        -DCMAKE_BUILD_RPATH="$BLASPP_LIB_DIR;$LAPACKPP_LIB_DIR;$RANDLAPACK_LIB_DIR" \
        $DISABLE_CUDA_FLAG $MACOS_OPENMP_FLAGS
run_step "Building extras" \
    cmake --build "$RANDNLA_PROJECT_DIR/build/extras-build/" -j "$JOBS"

run_step "Configuring benchmarks" \
    cmake -S "$RL_SRC/benchmark/" -B "$RANDNLA_PROJECT_DIR/build/benchmark-build/" \
        -DCMAKE_BUILD_TYPE=Release \
        -DFETCHCONTENT_BASE_DIR="$RANDNLA_PROJECT_DIR/build/fetchcontent-cache/" \
        -DRandLAPACK_DIR="$RANDLAPACK_CMAKE_DIR" \
        -Dlapackpp_DIR="$LAPACKPP_CMAKE_DIR" \
        -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
        -DRandom123_DIR="$RANDOM123_DIR" \
        -DCMAKE_BUILD_RPATH="$BLASPP_LIB_DIR;$LAPACKPP_LIB_DIR;$RANDLAPACK_LIB_DIR" \
        $DISABLE_CUDA_FLAG $MACOS_OPENMP_FLAGS
run_step "Building benchmarks" \
    cmake --build "$RANDNLA_PROJECT_DIR/build/benchmark-build/" -j "$JOBS"

#==============================================================================
# Shell config: opt-in only. The default prints what to add and touches nothing.
#==============================================================================
if [[ "$(basename "${SHELL:-bash}")" == "zsh" ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$(uname)" == "Darwin" ]]; then
    SHELL_RC="$HOME/.bash_profile"
else
    SHELL_RC="$HOME/.bashrc"
fi
EXPORT_DIR="export RANDNLA_PROJECT_DIR=\"$RANDNLA_PROJECT_DIR\""
EXPORT_GPU="export RANDNLA_PROJECT_GPU_AVAIL=\"$RANDNLA_PROJECT_GPU_AVAIL\""
if [[ "$MODIFY_RC" == "1" ]]; then
    if ! grep -q "export RANDNLA_PROJECT_DIR=" "$SHELL_RC" 2>/dev/null; then
        { echo "# Added via RandLAPACK/install.sh"; echo "$EXPORT_DIR"; } >> "$SHELL_RC"
    fi
    if ! grep -q "export RANDNLA_PROJECT_GPU_AVAIL=" "$SHELL_RC" 2>/dev/null; then
        echo "$EXPORT_GPU" >> "$SHELL_RC"
    fi
    echo "Added RANDNLA_PROJECT_DIR and RANDNLA_PROJECT_GPU_AVAIL to $SHELL_RC (open a new shell to pick them up)."
fi

#==============================================================================
# Success summary.
#==============================================================================
echo ""
echo "${C_OK}${C_BOLD}RandLAPACK installed successfully.${C_OFF}"
echo ""
echo "  Project layout:    $RANDNLA_PROJECT_DIR"
echo "  Installed library: $RANDNLA_PROJECT_DIR/install/RandLAPACK-install"
echo "  Extras:            $RANDNLA_PROJECT_DIR/build/extras-build/"
echo "  Benchmarks:        $RANDNLA_PROJECT_DIR/build/benchmark-build/"
echo "  Full build log:    $LOG"
echo ""
echo "  Smoke test:"
echo "    ctest --test-dir $RANDNLA_PROJECT_DIR/build/RandLAPACK-build"
echo ""
echo "  Consume from CMake with:"
echo "    -DRandLAPACK_DIR=$RANDLAPACK_CMAKE_DIR"
if [[ "$MODIFY_RC" != "1" ]]; then
    echo ""
    echo "  The benchmark scripts expect two environment variables. This script"
    echo "  no longer edits your shell config; add them yourself if needed:"
    echo "    $EXPORT_DIR"
    echo "    $EXPORT_GPU"
    echo "  (or re-run with --modify-rc to have them appended to $SHELL_RC)"
fi
