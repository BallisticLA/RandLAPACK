#!/bin/bash
# RandLAPACK autoinstaller for Linux and macOS.
#
# Builds RandLAPACK, its dependencies, the extras and the benchmark projects
# into a self-contained "RandNLA-project" directory laid out as:
#   lib:     RandLAPACK, blaspp, lapackpp sources
#   install: RandLAPACK-install, blaspp-*-install, lapackpp-*-install, random123
#   build:   one build directory per project above
#
# Nothing is installed system-wide, and your shell configuration is untouched
# unless you pass --modify-rc.
#
# You bring a C++20 compiler, CMake 3.21+, Git and a BLAS/LAPACK. This script
# does not install compilers or package managers; when something is missing it
# says so and tells you the usual way to get it.
#
# RandBLAS is intentionally not covered here: RandLAPACK vendors it as a git
# submodule pinned to an exact commit, and that pinned copy stays authoritative.
# RandBLAS's own installer is for people who want RandBLAS on its own.
#
# Prerequisites and tested configurations are listed in INSTALL_SCRIPT.md.

set -euo pipefail

usage() {
    # A heredoc rather than a line-range sed over this file's own comment block:
    # the latter starts printing unrelated code the moment anyone adds a line
    # above it.
    cat <<'USAGE'
Usage: bash install.sh [options]

Backend selection:
  --blas=BACKEND        auto | openblas | mkl | accelerate | custom
                        (default: auto -- Accelerate on macOS, MKL on Linux
                        when MKLROOT is set, otherwise OpenBLAS)
  --blas-int=WIDTH      ilp64 | lp64. Defaults to ilp64 wherever the backend
                        can actually provide it, falling back to lp64 with a
                        warning. Accelerate ILP64 requires macOS 13.3 or newer.
  --blas-libraries=L    Link line for --blas=custom, used for both BLAS and
                        LAPACK, e.g. "/opt/aocl/lib/libflame.so;/opt/aocl/lib/libblis.so"

GPU:
      --gpu             Build with CUDA support without asking
      --no-gpu          Build without GPU support without asking

Locations:
  --project-dir=DIR     Where dependencies, builds and installs go.
                        Default: $RANDNLA_PROJECT_DIR if set, otherwise
                        ../RandNLA-project next to this clone.
  --prefix=DIR          Install RandLAPACK itself here instead of
                        <project-dir>/install/RandLAPACK-install. Dependencies
                        still go in the project directory.

Build:
  -j, --jobs N          Parallel build jobs (default: number of cores)
      --fresh           Clear build directories and rebuild dependencies
      --no-extras       Skip the extras project
      --no-benchmarks   Skip the benchmark project
      --no-openmp       Configure without OpenMP

Output:
  -y, --yes             Assume "yes" at every prompt. Also the behavior when
                        stdin is not a terminal (CI, pipes).
      --modify-rc       Append RANDNLA_PROJECT_DIR / RANDNLA_PROJECT_GPU_AVAIL
                        exports to your shell config. The default touches
                        nothing and prints the lines to add yourself.
      --no-progress     Plain one-line-per-step output, no redrawing
  -h, --help            Show this help and exit

Every option has an environment-variable equivalent (flags win):
  RANDLAPACK_INSTALL_BLAS, RANDLAPACK_INSTALL_BLAS_INT,
  RANDLAPACK_INSTALL_BLAS_LIBRARIES, RANDLAPACK_INSTALL_GPU,
  RANDLAPACK_INSTALL_PROJECT_DIR, RANDLAPACK_INSTALL_PREFIX,
  RANDLAPACK_INSTALL_JOBS, RANDLAPACK_INSTALL_FRESH,
  RANDLAPACK_INSTALL_EXTRAS, RANDLAPACK_INSTALL_BENCHMARKS,
  RANDLAPACK_INSTALL_OPENMP, RANDLAPACK_INSTALL_YES,
  RANDLAPACK_INSTALL_MODIFY_RC, RANDLAPACK_INSTALL_PROGRESS

Already-installed dependencies are reused when pointed at by:
  BLASPP_INSTALL_DIR, LAPACKPP_INSTALL_DIR, RANDOM123_INSTALL_DIR

All compiler output goes to <project-dir>/install.log; the console shows one
line per step. On failure the log path is printed.
USAGE
}

#==============================================================================
# Option parsing. Environment variables provide defaults; flags override.
#==============================================================================
BLAS_BACKEND="${RANDLAPACK_INSTALL_BLAS:-auto}"
BLAS_INT_CHOICE="${RANDLAPACK_INSTALL_BLAS_INT:-auto}"      # auto | ilp64 | lp64
BLAS_LIBRARIES_ARG="${RANDLAPACK_INSTALL_BLAS_LIBRARIES:-}"
GPU_CHOICE="${RANDLAPACK_INSTALL_GPU:-ask}"                 # ask | on | off
PROJECT_DIR_OVERRIDE="${RANDLAPACK_INSTALL_PROJECT_DIR:-}"
PREFIX_OVERRIDE="${RANDLAPACK_INSTALL_PREFIX:-}"
JOBS="${RANDLAPACK_INSTALL_JOBS:-}"
FRESH="${RANDLAPACK_INSTALL_FRESH:-0}"
WANT_EXTRAS="${RANDLAPACK_INSTALL_EXTRAS:-1}"
WANT_BENCHMARKS="${RANDLAPACK_INSTALL_BENCHMARKS:-1}"
WANT_OPENMP="${RANDLAPACK_INSTALL_OPENMP:-1}"
ASSUME_YES="${RANDLAPACK_INSTALL_YES:-0}"
MODIFY_RC="${RANDLAPACK_INSTALL_MODIFY_RC:-0}"
WANT_PROGRESS="${RANDLAPACK_INSTALL_PROGRESS:-1}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --blas)             BLAS_BACKEND="${2:?--blas requires a backend}"; shift ;;
        --blas=*)           BLAS_BACKEND="${1#*=}" ;;
        --blas-int)         BLAS_INT_CHOICE="${2:?--blas-int requires a width}"; shift ;;
        --blas-int=*)       BLAS_INT_CHOICE="${1#*=}" ;;
        --blas-libraries)   BLAS_LIBRARIES_ARG="${2:?--blas-libraries requires a value}"; shift ;;
        --blas-libraries=*) BLAS_LIBRARIES_ARG="${1#*=}" ;;
        --gpu)              GPU_CHOICE="on" ;;
        --no-gpu)           GPU_CHOICE="off" ;;
        --project-dir)      PROJECT_DIR_OVERRIDE="${2:?--project-dir requires a path}"; shift ;;
        --project-dir=*)    PROJECT_DIR_OVERRIDE="${1#*=}" ;;
        --prefix)           PREFIX_OVERRIDE="${2:?--prefix requires a path}"; shift ;;
        --prefix=*)         PREFIX_OVERRIDE="${1#*=}" ;;
        -j|--jobs)          JOBS="${2:?--jobs requires a number}"; shift ;;
        --jobs=*)           JOBS="${1#*=}" ;;
        -j*)                JOBS="${1#-j}" ;;   # attached form, as in -j8
        --fresh)            FRESH=1 ;;
        --no-extras)        WANT_EXTRAS=0 ;;
        --no-benchmarks)    WANT_BENCHMARKS=0 ;;
        --no-openmp)        WANT_OPENMP=0 ;;
        -y|--yes)           ASSUME_YES=1 ;;
        --modify-rc)        MODIFY_RC=1 ;;
        --no-progress)      WANT_PROGRESS=0 ;;
        -h|--help)          usage; exit 0 ;;
        *) printf 'Unknown option: %s (see --help)\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

case "$BLAS_BACKEND" in
    auto|openblas|mkl|accelerate|custom) ;;
    *) die "--blas must be auto, openblas, mkl, accelerate or custom (got '$BLAS_BACKEND')" ;;
esac
case "$BLAS_INT_CHOICE" in
    auto|ilp64|lp64) ;;
    *) die "--blas-int must be ilp64 or lp64 (got '$BLAS_INT_CHOICE')" ;;
esac
case "$GPU_CHOICE" in
    ask|on|off) ;;
    *) die "RANDLAPACK_INSTALL_GPU must be 'on' or 'off' (got '$GPU_CHOICE')" ;;
esac
if [[ "$BLAS_BACKEND" == "custom" && -z "$BLAS_LIBRARIES_ARG" ]]; then
    die "--blas=custom needs --blas-libraries=<semicolon-separated link line>"
fi
if [[ -n "$BLAS_LIBRARIES_ARG" && "$BLAS_BACKEND" != "custom" ]]; then
    die "--blas-libraries only applies to --blas=custom (backend is '$BLAS_BACKEND')"
fi
if [[ -z "$JOBS" ]]; then
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
fi

#==============================================================================
# Interactivity and output style.
#
# Prompts happen only on a terminal and only without --yes. When stdin is not a
# terminal (piped, CI) every prompt silently takes its default, so this script
# can never hang waiting for input nobody is there to give.
#==============================================================================
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

if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" && "$WANT_PROGRESS" == "1" ]]; then
    C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_WARN=$'\033[33m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_ERR=""; C_WARN=""; C_BOLD=""; C_OFF=""
fi

# Progress rendering tier.
#   2  a terminal that can draw: redraw a bar in place, with block characters
#   1  a terminal without colour or UTF-8: same bar, ASCII, still redrawn
#   0  not a terminal: one line per step, no escapes, no carriage returns
#
# Tier 0 is a requirement, not a fallback. Redirected output ends up in
# install.log, in CI transcripts and in bug reports, and control characters make
# all three unreadable.
PROGRESS_TIER=0
if [[ -t 1 && "$WANT_PROGRESS" == "1" && "${TERM:-}" != "dumb" ]]; then
    if [[ -z "${NO_COLOR:-}" && "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" == *[Uu][Tt][Ff]* ]]; then
        PROGRESS_TIER=2
    else
        PROGRESS_TIER=1
    fi
fi
if (( PROGRESS_TIER >= 2 )); then
    BAR_FULL="━"; BAR_EMPTY="─"
else
    BAR_FULL="#"; BAR_EMPTY="-"
fi

note() { printf '%s\n' "$*"; }
warn() { printf '%swarning:%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }

# Collected and reprinted in the final summary. A warning emitted twenty minutes
# and several thousand log lines before the summary is a warning nobody reads.
WARNINGS=()
record_warning() { WARNINGS+=("$1"); warn "$1"; }

#==============================================================================
# Toolchain preflight. Report everything missing at once rather than failing on
# the first one, so a bare machine takes one round trip instead of three.
#==============================================================================
UNAME_S="$(uname -s)"
MISSING=()
command -v cmake >/dev/null 2>&1 || MISSING+=("cmake")
command -v git   >/dev/null 2>&1 || MISSING+=("git")
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && \
   ! command -v clang++ >/dev/null 2>&1; then
    MISSING+=("a C++ compiler")
fi
if (( ${#MISSING[@]} )); then
    printf 'ERROR: missing prerequisites: %s\n\n' "${MISSING[*]}" >&2
    if [[ "$UNAME_S" == "Darwin" ]]; then
        printf '  xcode-select --install    # Apple Clang and git\n' >&2
        printf '  brew install cmake\n\n' >&2
    else
        printf '  sudo apt install g++ gfortran cmake git      # Debian, Ubuntu\n' >&2
        printf '  sudo dnf install gcc-c++ gcc-gfortran cmake git  # Fedora, RHEL\n\n' >&2
    fi
    printf 'See INSTALL_SCRIPT.md for the full prerequisite list.\n' >&2
    exit 1
fi

CMAKE_VERSION="$(cmake --version | head -n1 | awk '{print $3}')"
if [[ "$(printf '%s\n3.21\n' "$CMAKE_VERSION" | sort -V | head -n1)" != "3.21" ]]; then
    die "CMake 3.21 or later is required (found $CMAKE_VERSION). See INSTALL_SCRIPT.md."
fi

# GCC 13.3.0 is the reference version. Warn rather than block: newer usually
# works, and the C++20 concepts RandBLAS uses need at least 13.
PREFERRED_GCC_VERSION="13.3.0"
CURRENT_GCC_VERSION=$(gcc --version 2>/dev/null | head -n 1 | awk '{print $NF}')
if [[ -n "$CURRENT_GCC_VERSION" && "$CURRENT_GCC_VERSION" != "$PREFERRED_GCC_VERSION" ]]; then
    GCC_MAJOR="${CURRENT_GCC_VERSION%%.*}"
    if [[ "$GCC_MAJOR" =~ ^[0-9]+$ ]] && (( GCC_MAJOR < 13 )); then
        record_warning "gcc $CURRENT_GCC_VERSION is older than 13; RandBLAS uses C++20 concepts and may not compile."
    else
        note "Note: gcc $PREFERRED_GCC_VERSION is the reference version; found $CURRENT_GCC_VERSION."
    fi
fi

#==============================================================================
# GPU decision. --gpu/--no-gpu (or RANDLAPACK_INSTALL_GPU) decide outright;
# otherwise detection plus a prompt. Non-interactive defaults: NVIDIA detected
# means GPU on; AMD or nothing detected means GPU off, because the CUDA-only
# build cannot succeed on AMD and saying yes for the user would guarantee a
# failure.
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
             { [[ "$UNAME_S" == "Darwin" ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -qi "AMD"; }; then
            if ask "AMD GPU detected, but only a CUDA build is available for now. Attempt a CUDA build anyway?" n; then
                RANDLAPACK_CUDA="ON"; RANDNLA_PROJECT_GPU_AVAIL="auto"
            fi
        else
            note "No GPU detected; building without GPU support."
        fi
        ;;
esac

if [[ "$RANDNLA_PROJECT_GPU_AVAIL" == "auto" ]]; then
    PREFERRED_NVCC_VERSION="12.9"
    CURRENT_NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1)
    if [[ "$CURRENT_NVCC_VERSION" != "$PREFERRED_NVCC_VERSION" ]]; then
        note "Note: NVCC $PREFERRED_NVCC_VERSION is the reference version; found ${CURRENT_NVCC_VERSION:-none}."
        if ! ask "Continue with the current NVCC?" y; then
            die "Stopping at your request. Install NVCC $PREFERRED_NVCC_VERSION and re-run."
        fi
    fi
fi

#==============================================================================
# Project layout.
#
# Precedence: --project-dir, then RANDNLA_PROJECT_DIR, then a sibling of this
# clone. Honouring the environment variable is what lets this installer and
# RandBLAS's share one dependency tree -- whichever runs second finds the
# first one's BLAS++ and reuses it.
#
# This script no longer moves your clone. The previous version relocated the
# repository into <project>/lib/RandLAPACK on first run, which breaks git
# worktrees and surprises anyone who cloned deliberately. The layout below is
# created regardless of where the clone lives, and lib/RandLAPACK is a symlink
# so the tree still reads as complete.
#==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -n "$PROJECT_DIR_OVERRIDE" ]]; then
    RANDNLA_PROJECT_DIR="$PROJECT_DIR_OVERRIDE"
elif [[ -n "${RANDNLA_PROJECT_DIR:-}" ]]; then
    :   # already set in the environment; use it as-is
else
    RANDNLA_PROJECT_DIR="$(dirname "$REPO_DIR")/RandNLA-project"
fi
mkdir -p "$RANDNLA_PROJECT_DIR"
RANDNLA_PROJECT_DIR="$(cd "$RANDNLA_PROJECT_DIR" && pwd)"
mkdir -p "$RANDNLA_PROJECT_DIR"/{install,lib,build}

# A symlink, not a move: the clone stays where the user put it.
if [[ ! -e "$RANDNLA_PROJECT_DIR/lib/RandLAPACK" ]]; then
    ln -s "$REPO_DIR" "$RANDNLA_PROJECT_DIR/lib/RandLAPACK"
fi
RL_SRC="$REPO_DIR"

RANDLAPACK_INSTALL_DIR="${PREFIX_OVERRIDE:-$RANDNLA_PROJECT_DIR/install/RandLAPACK-install}"

LOG="$RANDNLA_PROJECT_DIR/install.log"
# Appended, not truncated: the previous run's output is exactly what you want
# when the current run fails the same way.
{
    printf '\n===============================================================\n'
    printf 'RandLAPACK install started %s\n' "$(date)"
    printf '===============================================================\n'
} >> "$LOG"

STEP=0
TOTAL_STEPS=0

run_step() {
    local label="$1"; shift
    STEP=$((STEP + 1))
    printf '%s[%d/%d]%s %s ... ' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$label"
    local t0 t1
    t0=$(date +%s)
    {
        printf '\n===== [%d/%d] %s =====\n' "$STEP" "$TOTAL_STEPS" "$label"
        printf '$ %s\n' "$*"
    } >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        t1=$(date +%s)
        printf '%sdone%s (%ds)\n' "$C_OK" "$C_OFF" "$((t1 - t0))"
    else
        printf '%sFAILED%s\n' "$C_ERR" "$C_OFF" >&2
        printf '\nStep "%s" failed. Full output: %s\n' "$label" "$LOG" >&2
        printf 'The last 20 log lines:\n' >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
}

draw_bar() {
    local pct="$1" label="$2" width=28 filled i bar=""
    (( pct > 100 )) && pct=100
    filled=$(( pct * width / 100 ))
    for ((i = 0; i < width; i++)); do
        if (( i < filled )); then bar+="$BAR_FULL"; else bar+="$BAR_EMPTY"; fi
    done
    printf '\r%s[%d/%d]%s %s %s %3d%%\033[K' \
        "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$label" "$bar" "$pct"
}

# run_build_step: like run_step, but drives a real progress bar from the build
# tool's own output. Both tools already report progress on the stream being
# captured anyway -- Ninja writes "[12/34]", Make writes "[ 42%]" -- so the bar
# tracks actual work rather than elapsed time. Falls through to run_step at
# tier 0 so redirected output is byte-identical to the plain form.
run_build_step() {
    local label="$1"; shift
    if (( PROGRESS_TIER == 0 )); then
        run_step "$label" "$@"
        return
    fi
    STEP=$((STEP + 1))
    local t0 t1 status statusfile
    t0=$(date +%s)
    {
        printf '\n===== [%d/%d] %s =====\n' "$STEP" "$TOTAL_STEPS" "$label"
        printf '$ %s\n' "$*"
    } >> "$LOG"
    draw_bar 0 "$label"
    # The while loop runs in a subshell, so the command's exit status cannot
    # come back through a variable; pass it through a file instead.
    statusfile="$(mktemp)"
    {
        "$@" 2>&1
        printf '%d\n' "$?" > "$statusfile"
    } | while IFS= read -r line; do
        printf '%s\n' "$line" >> "$LOG"
        if [[ "$line" =~ ^\[([0-9]+)/([0-9]+)\] ]]; then
            draw_bar $(( 100 * ${BASH_REMATCH[1]} / ${BASH_REMATCH[2]} )) "$label"
        elif [[ "$line" =~ ^\[[[:space:]]*([0-9]+)%\] ]]; then
            draw_bar "${BASH_REMATCH[1]}" "$label"
        fi
    done
    status="$(cat "$statusfile" 2>/dev/null || echo 1)"
    rm -f "$statusfile"
    if [[ "$status" == "0" ]]; then
        t1=$(date +%s)
        printf '\r%s[%d/%d]%s %s ... %sdone%s (%ds)\033[K\n' \
            "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$label" "$C_OK" "$C_OFF" "$((t1 - t0))"
    else
        printf '\r%s[%d/%d]%s %s ... %sFAILED%s\033[K\n' \
            "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$label" "$C_ERR" "$C_OFF" >&2
        printf '\nStep "%s" failed. Full output: %s\n' "$label" "$LOG" >&2
        printf 'The last 20 log lines:\n' >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
}

skip_step() {
    STEP=$((STEP + 1))
    printf '%s[%d/%d]%s %s\n' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$1"
}

#==============================================================================
# Provenance stamps.
#
# A dependency is reused only when it was built from the source we would build
# from now, in the configuration we want now. Reuse keyed on mere presence means
# changing a pin, switching BLAS backend, or toggling --gpu is a silent no-op
# for anyone who already ran this script: they keep the old artifact and the new
# setting never takes effect.
#==============================================================================
stamp_file() { printf '%s/.randlapack-provenance' "$1"; }

stamp_matches() {  # <install-dir> <expected-stamp>
    [[ -f "$(stamp_file "$1")" ]] || return 1
    [[ "$(cat "$(stamp_file "$1")")" == "$2" ]]
}

write_stamp() { printf '%s\n' "$2" > "$(stamp_file "$1")"; }

# Shallow-fetch exactly one commit or tag, so the pin cannot drift the way a
# branch name would. The source tree gets its own stamp because a shallow
# checkout of a tag does not keep the tag ref locally, so git cannot be asked
# afterwards whether a tree is at the pin -- and without it, a source tree left
# from an older pin would be reused as if current.
clone_pinned() {  # <url> <dest> <ref>
    local url="$1" dest="$2" ref="$3"
    rm -rf "$dest"
    mkdir -p "$dest"
    git -C "$dest" init --quiet
    git -C "$dest" remote add origin "$url"
    git -C "$dest" fetch --quiet --depth 1 origin "$ref"
    git -C "$dest" checkout --quiet FETCH_HEAD
    write_stamp "$dest" "$url@$ref"
}

source_is_current() { stamp_matches "$1" "$2@$3"; }

#==============================================================================
# Dependency pins. Immutable refs only: a tag or a full commit hash, never a
# branch name. These match the refs this repository's own Windows provisioner
# already validated, so the two halves of the project stop disagreeing about
# what they build.
#==============================================================================
BLASPP_URL="https://github.com/icl-utk-edu/blaspp.git"
# The commit that merged new-Apple-Accelerate support (blaspp PR #134,
# 2026-08-27); also contains the MSVC portability fix (PR #132). Not in a
# release yet -- the latest tag, v2025.05.28, predates both.
BLASPP_REF="2d8d4e937ac46fffab33d4174a4fc7659726dbda"
LAPACKPP_URL="https://github.com/icl-utk-edu/lapackpp.git"
# The commit that merged the LAPACK++ half of new-Accelerate support
# (lapackpp PR #88, 2026-08-27).
LAPACKPP_REF="b9439cf3c26d1655d88e7f510ae8b4f82fbeb687"
RANDOM123_URL="https://github.com/DEShawResearch/Random123.git"
RANDOM123_REF="v1.14.0"

#==============================================================================
# Backend resolution.
#
# "auto" picks Accelerate on macOS and MKL on Linux when MKLROOT says it is
# installed, OpenBLAS otherwise. The choice is always printed, because a silent
# default is the thing people later cannot explain.
#
# macOS defaults to Accelerate through Apple's NEW interface (macOS >= 13.3,
# ACCELERATE_NEW_LAPACK, LAPACK 3.12 on current SDKs), which the pinned BLAS++
# and LAPACK++ support (icl-utk-edu/blaspp#134, icl-utk-edu/lapackpp#88, issue
# #165). Apple's LEGACY interface (LAPACK 3.2.1) has a broken divide-and-
# conquer gesdd -- RandLAPACK calls gesdd in rl_rsvd, rl_abrik, rl_revd2,
# rl_preconditioners and rl_util -- and lacks routines BQRRP needs, which is
# why the default used to be Homebrew OpenBLAS. A silent fall-back to the
# legacy interface is caught twice: a hard defines.h check right after the
# BLAS++ build, and the numerical gesdd conftest before the summary.
# Homebrew OpenBLAS remains available with --blas=openblas.
#==============================================================================
BREW_PREFIX=""
if [[ "$UNAME_S" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    # Never hardcode /opt/homebrew: that is Apple Silicon only, and breaks both
    # Intel Macs (/usr/local) and any custom HOMEBREW_PREFIX.
    BREW_PREFIX="$(brew --prefix)"
fi

if [[ "$BLAS_BACKEND" == "auto" ]]; then
    if [[ "$UNAME_S" == "Darwin" ]]; then
        BLAS_BACKEND="accelerate"
    elif [[ -n "${MKLROOT:-}" && -d "${MKLROOT:-}" ]]; then
        BLAS_BACKEND="mkl"
    else
        BLAS_BACKEND="openblas"
    fi
    note "Selected BLAS backend: $BLAS_BACKEND (from --blas=auto)"
fi

# Integer width: prefer ILP64 wherever the backend can genuinely provide it.
#
# ILP64 matters because LP64 caps every individual BLAS dimension at 2^31.
# RandLAPACK is int64_t throughout, and BLAS++/LAPACK++ guard the downcast, so
# an oversized dimension throws rather than truncating -- but a header/library
# disagreement is NOT guarded and shows up as an absurd workspace size or a run
# that never finishes. The conftest below is what catches that.
#
# Accelerate: Apple's new interface (macOS >= 13.3) carries ILP64, selected by
# ACCELERATE_LAPACK_ILP64, and the pinned BLAS++ wires blas_int=int64 to it
# (icl-utk-edu/blaspp#134). On macOS older than 13.3 the int64 probe fails, so
# "auto" falls back to LP64 with the usual warning and an explicit
# --blas-int=ilp64 fails the BLAS++ configure rather than silently downgrading.
WIDTH_ORDER=()
case "$BLAS_INT_CHOICE" in
    ilp64) WIDTH_ORDER=(int64) ;;
    lp64)  WIDTH_ORDER=(int32) ;;
    auto)  WIDTH_ORDER=(int64 int32) ;;
esac

# blaspp's own backend selector; its matcher accepts "apple" or "accelerate".
BLASPP_BACKEND_FLAGS=()
LAPACKPP_BACKEND_FLAGS=()
case "$BLAS_BACKEND" in
    mkl)        BLASPP_BACKEND_FLAGS=(-Dblas=mkl) ;;
    accelerate) BLASPP_BACKEND_FLAGS=(-Dblas=apple) ;;
    custom)
        BLASPP_BACKEND_FLAGS=(-DBLAS_LIBRARIES="$BLAS_LIBRARIES_ARG")
        LAPACKPP_BACKEND_FLAGS=(-DLAPACK_LIBRARIES="$BLAS_LIBRARIES_ARG")
        ;;
    openblas)
        BLASPP_BACKEND_FLAGS=(-Dblas=openblas)
        if [[ "$UNAME_S" == "Darwin" ]]; then
            # Homebrew's OpenBLAS is not on the default search path, and needs
            # the gfortran name-mangling convention stated explicitly.
            if [[ -z "$BREW_PREFIX" || ! -f "$BREW_PREFIX/opt/openblas/lib/libopenblas.dylib" ]]; then
                die "OpenBLAS was not found under Homebrew. Install it with 'brew install openblas', or choose another backend with --blas=."
            fi
            BLASPP_BACKEND_FLAGS=(
                "-DBLAS_LIBRARIES=$BREW_PREFIX/opt/openblas/lib/libopenblas.dylib"
                "-Dblas_fortran=add"
            )
            LAPACKPP_BACKEND_FLAGS=("-DLAPACK_LIBRARIES=$BREW_PREFIX/opt/openblas/lib/libopenblas.dylib")
        fi
        ;;
esac

#==============================================================================
# OpenMP. Apple Clang ships no OpenMP runtime; Homebrew's libomp supplies one
# but only via -Xpreprocessor -fopenmp plus explicit paths, so it has to be
# wired up by hand rather than found by FindOpenMP.
#==============================================================================
OPENMP_FLAGS=()
if [[ "$WANT_OPENMP" != "1" ]]; then
    OPENMP_FLAGS=(-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE)
elif [[ "$UNAME_S" == "Darwin" ]]; then
    LIBOMP=""
    if [[ -n "$BREW_PREFIX" && -f "$BREW_PREFIX/opt/libomp/lib/libomp.dylib" ]]; then
        LIBOMP="$BREW_PREFIX/opt/libomp"
    fi
    if [[ -n "$LIBOMP" ]]; then
        MACOS_SDK_PATH="$(xcrun --show-sdk-path 2>/dev/null || true)"
        if [[ -n "$MACOS_SDK_PATH" ]]; then
            export CXXFLAGS="${CXXFLAGS:-} -isystem ${MACOS_SDK_PATH}/usr/include/c++/v1"
        fi
        export CFLAGS="${CFLAGS:-} -Xpreprocessor -fopenmp -I$LIBOMP/include"
        export CXXFLAGS="${CXXFLAGS:-} -Xpreprocessor -fopenmp -I$LIBOMP/include"
        export LDFLAGS="${LDFLAGS:-} -L$LIBOMP/lib"
        # Both C and CXX components must be described: BLAS++'s installed config
        # calls find_dependency(OpenMP) without restricting components, so a
        # consumer resolves OpenMP_C too. With only the CXX variables set that
        # fails with "Could NOT find OpenMP_C" while configuring RandLAPACK,
        # long after BLAS++ itself built cleanly.
        OPENMP_FLAGS=(
            "-DOpenMP_C_LIB_NAMES=omp"
            "-DOpenMP_CXX_LIB_NAMES=omp"
            "-DOpenMP_omp_LIBRARY=$LIBOMP/lib/libomp.dylib"
            "-DOpenMP_C_FLAGS=-Xpreprocessor;-fopenmp"
            "-DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp"
        )
    else
        OPENMP_FLAGS=(-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE)
        record_warning "OpenMP is unavailable, so RandLAPACK will be single-threaded. Apple Clang has no OpenMP runtime; install one with 'brew install libomp' and re-run."
        WANT_OPENMP=0
    fi
fi

#==============================================================================
# Dependency discovery. An install pointed at by an environment variable is
# taken as given: the user knows something we do not, and rebuilding over it
# would be both slow and rude.
#==============================================================================
find_cmake_config() {  # <root> <package> -> prints the config dir, or nothing
    local root="$1" pkg="$2" libdir
    for libdir in lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu; do
        if [[ -f "$root/$libdir/cmake/$pkg/${pkg}Config.cmake" ]]; then
            printf '%s/%s/cmake/%s' "$root" "$libdir" "$pkg"
            return 0
        fi
    done
    return 0
}

# Dependency installs are per-configuration. A BLAS++ built for MKL is not
# interchangeable with one built for OpenBLAS, and one built with
# gpu_backend=auto is not interchangeable with one built "none".
#
# The GPU tag is in the directory name, not only in the provenance stamp, and
# that is deliberate for a reason beyond this project: RandBLAS's installer
# uses the same RandNLA-project layout and names its own BLAS++ install
# "blaspp-<backend>-install". If we used that name too, then with a shared
# RANDNLA_PROJECT_DIR this script would rebuild over RandBLAS's BLAS++ (their
# provenance stamp is a different file, so it never matches ours), and
# RandBLAS's next run would then reuse an artifact we had replaced underneath
# it while its own stamp still claimed the original -- a silent configuration
# mismatch of exactly the kind the stamps exist to prevent. Distinct names make
# that collision impossible.
#
# Sharing dependencies between the two projects is therefore explicit rather
# than automatic: point BLASPP_INSTALL_DIR at an existing install and it is
# reused, with the conftest confirming it actually works.
BLASPP_GPU_TAG="$( [[ "$RANDNLA_PROJECT_GPU_AVAIL" == "auto" ]] && echo cuda || echo cpu )"
BLASPP_INSTALL="$RANDNLA_PROJECT_DIR/install/blaspp-$BLAS_BACKEND-$BLASPP_GPU_TAG-install"
LAPACKPP_INSTALL="$RANDNLA_PROJECT_DIR/install/lapackpp-$BLAS_BACKEND-$BLASPP_GPU_TAG-install"
RANDOM123_INSTALL="$RANDNLA_PROJECT_DIR/install/random123"

EXTERNAL_BLASPP=0
EXTERNAL_LAPACKPP=0
EXTERNAL_RANDOM123=0
BLASPP_CMAKE_DIR=""
LAPACKPP_CMAKE_DIR=""
RANDOM123_DIR=""
BLASPP_LIB_DIR=""
LAPACKPP_LIB_DIR=""

note ""
note "Dependency discovery:"
if [[ -n "${BLASPP_INSTALL_DIR:-}" ]]; then
    BLASPP_CMAKE_DIR="$(find_cmake_config "$BLASPP_INSTALL_DIR" blaspp)"
    if [[ -n "$BLASPP_CMAKE_DIR" ]]; then
        EXTERNAL_BLASPP=1
        BLASPP_LIB_DIR="$(dirname "$(dirname "$BLASPP_CMAKE_DIR")")"
        note "  [blaspp]    external install: $BLASPP_INSTALL_DIR"
    else
        note "  [blaspp]    BLASPP_INSTALL_DIR is set but holds no blasppConfig.cmake; building from source."
    fi
fi
if [[ -n "${LAPACKPP_INSTALL_DIR:-}" ]]; then
    LAPACKPP_CMAKE_DIR="$(find_cmake_config "$LAPACKPP_INSTALL_DIR" lapackpp)"
    if [[ -n "$LAPACKPP_CMAKE_DIR" ]]; then
        EXTERNAL_LAPACKPP=1
        LAPACKPP_LIB_DIR="$(dirname "$(dirname "$LAPACKPP_CMAKE_DIR")")"
        note "  [lapackpp]  external install: $LAPACKPP_INSTALL_DIR"
    else
        note "  [lapackpp]  LAPACKPP_INSTALL_DIR is set but holds no lapackppConfig.cmake; building from source."
    fi
fi
if [[ -n "${RANDOM123_INSTALL_DIR:-}" && -f "$RANDOM123_INSTALL_DIR/include/Random123/philox.h" ]]; then
    EXTERNAL_RANDOM123=1
    RANDOM123_DIR="$RANDOM123_INSTALL_DIR/include/"
    note "  [random123] external install: $RANDOM123_INSTALL_DIR"
fi

# What we would build, and therefore what a prior install must match to be
# reused. The GPU setting is part of the stamp because a CUDA-linked BLAS++
# cannot stand in for a CPU-only one -- without it, switching --gpu to --no-gpu
# silently reuses the wrong artifact.
BLASPP_STAMP_BASE="$BLASPP_URL@$BLASPP_REF backend=$BLAS_BACKEND libs=$BLAS_LIBRARIES_ARG gpu=$RANDNLA_PROJECT_GPU_AVAIL"
LAPACKPP_STAMP_BASE="$LAPACKPP_URL@$LAPACKPP_REF backend=$BLAS_BACKEND libs=$BLAS_LIBRARIES_ARG gpu=$RANDNLA_PROJECT_GPU_AVAIL"
RANDOM123_STAMP="$RANDOM123_URL@$RANDOM123_REF"

#==============================================================================
# Step accounting, computed up front so "[3/12]" means something.
#   BLAS++    3 (source, configure, build) -- the reuse path spends 3 skips
#   LAPACK++  2 (source+configure, build)
#   Random123 1
#   RandLAPACK 2, verification 1
#   extras 2, benchmarks 2
#==============================================================================
BUILD_BLASPP=$(( EXTERNAL_BLASPP ? 0 : 1 ))
BUILD_LAPACKPP=$(( EXTERNAL_LAPACKPP ? 0 : 1 ))
BUILD_RANDOM123=$(( EXTERNAL_RANDOM123 ? 0 : 1 ))

if (( FRESH )); then
    rm -rf "$RANDNLA_PROJECT_DIR/build"
fi
mkdir -p "$RANDNLA_PROJECT_DIR/build"

TOTAL_STEPS=$(( BUILD_BLASPP * 3 + BUILD_LAPACKPP * 2 + BUILD_RANDOM123 + 3 ))
(( WANT_EXTRAS ))     && TOTAL_STEPS=$(( TOTAL_STEPS + 2 )) || true
(( WANT_BENCHMARKS )) && TOTAL_STEPS=$(( TOTAL_STEPS + 2 )) || true
note ""

# RandBLAS is a submodule pinned to an exact commit and stays authoritative.
git -C "$REPO_DIR" submodule init   >> "$LOG" 2>&1
git -C "$REPO_DIR" submodule update >> "$LOG" 2>&1

#==============================================================================
# BLAS++.
#
# The integer width is settled by asking BLAS++ to configure at each width in
# WIDTH_ORDER until one succeeds, then reading back what it actually built.
# Each attempt gets a clean build directory: BLAS++ caches its detection
# results, and re-running cmake over a directory where detection previously
# failed regenerates blas/defines.h WITHOUT the Fortran-mangling and backend
# defines, which then breaks every downstream compile through LAPACK_GLOBAL in
# a way that looks nothing like the original failure.
#==============================================================================
BLASPP_SRC="$RANDNLA_PROJECT_DIR/lib/blaspp"
BLAS_INT_RESOLVED=""

configure_blaspp_at_width() {
    local width="$1" build="$RANDNLA_PROJECT_DIR/build/blaspp-build-$width"
    rm -rf "$build"; mkdir -p "$build"
    cmake -S "$BLASPP_SRC" -B "$build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$BLASPP_INSTALL" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -Dgpu_backend="$RANDNLA_PROJECT_GPU_AVAIL" \
        -Dblas_int="$width" \
        -Dbuild_tests=OFF \
        "${BLASPP_BACKEND_FLAGS[@]}" "${OPENMP_FLAGS[@]}" >> "$LOG" 2>&1
}

if (( BUILD_BLASPP )); then
    PRIOR_WIDTH="$(cat "$BLASPP_INSTALL/.randlapack-width" 2>/dev/null || true)"
    REUSABLE=0
    if (( ! FRESH )) && [[ -n "$PRIOR_WIDTH" ]] && \
       stamp_matches "$BLASPP_INSTALL" "$BLASPP_STAMP_BASE width=$PRIOR_WIDTH"; then
        for w in "${WIDTH_ORDER[@]}"; do
            if [[ "$w" == "$PRIOR_WIDTH" ]]; then REUSABLE=1; break; fi
        done
        if (( ! REUSABLE )); then
            note "  [blaspp] existing install is $PRIOR_WIDTH but this run wants ${WIDTH_ORDER[0]}; rebuilding."
        fi
    fi

    if (( REUSABLE )); then
        BLAS_INT_RESOLVED="$PRIOR_WIDTH"
        BLASPP_CMAKE_DIR="$(find_cmake_config "$BLASPP_INSTALL" blaspp)"
        BLASPP_LIB_DIR="$(dirname "$(dirname "$BLASPP_CMAKE_DIR")")"
        skip_step "BLAS++ source ... already present"
        skip_step "BLAS++ ... reusing the $BLAS_INT_RESOLVED install"
        skip_step "BLAS++ ... already built"
    else
        if source_is_current "$BLASPP_SRC" "$BLASPP_URL" "$BLASPP_REF"; then
            skip_step "BLAS++ source ... already at $BLASPP_REF"
        else
            run_step "Fetching BLAS++ ($BLASPP_REF)" \
                clone_pinned "$BLASPP_URL" "$BLASPP_SRC" "$BLASPP_REF"
        fi

        STEP=$((STEP + 1))
        printf '%s[%d/%d]%s Configuring BLAS++ ... ' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF"
        for width in "${WIDTH_ORDER[@]}"; do
            if configure_blaspp_at_width "$width"; then BLAS_INT_RESOLVED="$width"; break; fi
        done
        if [[ -z "$BLAS_INT_RESOLVED" ]]; then
            printf '%sFAILED%s\n' "$C_ERR" "$C_OFF" >&2
            printf '\nBLAS++ could not find a usable %s BLAS at any of: %s\nFull output: %s\n' \
                "$BLAS_BACKEND" "${WIDTH_ORDER[*]}" "$LOG" >&2
            case "$BLAS_BACKEND" in
                openblas) printf '\n  sudo apt install libopenblas-dev      # Debian, Ubuntu\n  brew install openblas                 # macOS\n' >&2 ;;
                mkl)      printf '\n  Set MKLROOT, or source the oneAPI setvars script.\n' >&2 ;;
            esac
            exit 1
        fi
        printf '%sdone%s (requested %s)\n' "$C_OK" "$C_OFF" "$BLAS_INT_RESOLVED"

        run_build_step "Building and installing BLAS++" \
            cmake --build "$RANDNLA_PROJECT_DIR/build/blaspp-build-$BLAS_INT_RESOLVED" \
                -j "$JOBS" --target install

        # The width actually built, read back from BLAS++'s own generated header
        # rather than inferred from which configure attempt succeeded. Those can
        # differ: BLAS++ probes int32 before int64, while blas_int only filters
        # which *library names* to consider. For MKL that is enough, because
        # mkl_intel_lp64 and mkl_intel_ilp64 are different libraries. For
        # OpenBLAS there is only -lopenblas, so asking for int64 and getting a
        # successful configure tells us nothing -- an LP64 OpenBLAS passes the
        # int32 probe and is accepted. Trusting the request here would stamp an
        # LP64 install as ILP64, and every later run would reuse it believing
        # it had ILP64.
        if grep -q '^#define BLAS_ILP64' "$BLASPP_INSTALL/include/blas/defines.h" 2>/dev/null; then
            BLAS_INT_BUILT="int64"
        else
            BLAS_INT_BUILT="int32"
        fi

        # Accelerate must be Apple's NEW interface. The legacy one (LAPACK
        # 3.2.1) computes gesdd incorrectly on Apple Silicon and lacks
        # routines BQRRP needs, so a silent fall-back would produce a library
        # that is wrong at runtime. Failing here, right after the BLAS++
        # build, names the actual cause; letting it ride would surface later
        # as a gesdd conftest failure or as wrong driver results.
        if [[ "$BLAS_BACKEND" == "accelerate" ]] \
                && ! grep -q 'ACCELERATE_NEW_LAPACK' "$BLASPP_INSTALL/include/blas/defines.h" 2>/dev/null; then
            die "BLAS++ selected Apple's LEGACY Accelerate interface, whose gesdd returns wrong results (issue #165). The new interface needs macOS 13.3 or newer. On older macOS, use --blas=openblas (after 'brew install openblas')."
        fi
        if [[ "$BLAS_INT_BUILT" != "$BLAS_INT_RESOLVED" ]]; then
            note "  [blaspp] requested $BLAS_INT_RESOLVED, BLAS++ selected $BLAS_INT_BUILT"
        fi
        BLAS_INT_RESOLVED="$BLAS_INT_BUILT"

        if [[ "$BLAS_INT_RESOLVED" == "int32" && "${WIDTH_ORDER[0]}" == "int64" ]]; then
            record_warning "No ILP64 $BLAS_BACKEND was available, so BLAS++ was built LP64 (32-bit BLAS integers). RandLAPACK works either way -- BLAS++ and LAPACK++ throw rather than truncate if a dimension exceeds the range -- but individual matrix dimensions are then capped at 2^31. BLAS++ only ever looks for plain -lopenblas, so an ILP64 OpenBLAS has to be named explicitly with --blas=custom --blas-libraries=... --blas-int=ilp64."
        fi

        write_stamp "$BLASPP_INSTALL" "$BLASPP_STAMP_BASE width=$BLAS_INT_RESOLVED"
        printf '%s\n' "$BLAS_INT_RESOLVED" > "$BLASPP_INSTALL/.randlapack-width"
        BLASPP_CMAKE_DIR="$(find_cmake_config "$BLASPP_INSTALL" blaspp)"
        BLASPP_LIB_DIR="$(dirname "$(dirname "$BLASPP_CMAKE_DIR")")"
    fi
fi
[[ -n "$BLASPP_CMAKE_DIR" ]] || die "BLAS++ was installed but blasppConfig.cmake could not be located under $BLASPP_INSTALL"

#==============================================================================
# LAPACK++, built against the BLAS++ resolved above.
#==============================================================================
LAPACKPP_SRC="$RANDNLA_PROJECT_DIR/lib/lapackpp"
if (( BUILD_LAPACKPP )); then
    LAPACKPP_STAMP="$LAPACKPP_STAMP_BASE width=$BLAS_INT_RESOLVED"
    if (( ! FRESH )) && stamp_matches "$LAPACKPP_INSTALL" "$LAPACKPP_STAMP"; then
        LAPACKPP_CMAKE_DIR="$(find_cmake_config "$LAPACKPP_INSTALL" lapackpp)"
        LAPACKPP_LIB_DIR="$(dirname "$(dirname "$LAPACKPP_CMAKE_DIR")")"
        skip_step "LAPACK++ ... reusing existing install"
        skip_step "LAPACK++ ... already built"
    else
        configure_lapackpp() {
            local build="$RANDNLA_PROJECT_DIR/build/lapackpp-build"
            rm -rf "$build"; mkdir -p "$build"
            if ! source_is_current "$LAPACKPP_SRC" "$LAPACKPP_URL" "$LAPACKPP_REF"; then
                clone_pinned "$LAPACKPP_URL" "$LAPACKPP_SRC" "$LAPACKPP_REF"
            fi
            cmake -S "$LAPACKPP_SRC" -B "$build" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX="$LAPACKPP_INSTALL" \
                -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
                -Dgpu_backend="$RANDNLA_PROJECT_GPU_AVAIL" \
                -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
                -Dbuild_tests=OFF \
                "${LAPACKPP_BACKEND_FLAGS[@]}" "${OPENMP_FLAGS[@]}"
        }
        run_step "Fetching and configuring LAPACK++ ($LAPACKPP_REF)" configure_lapackpp
        run_build_step "Building and installing LAPACK++" \
            cmake --build "$RANDNLA_PROJECT_DIR/build/lapackpp-build" -j "$JOBS" --target install
        write_stamp "$LAPACKPP_INSTALL" "$LAPACKPP_STAMP"
        LAPACKPP_CMAKE_DIR="$(find_cmake_config "$LAPACKPP_INSTALL" lapackpp)"
        LAPACKPP_LIB_DIR="$(dirname "$(dirname "$LAPACKPP_CMAKE_DIR")")"
    fi
fi
[[ -n "$LAPACKPP_CMAKE_DIR" ]] || die "LAPACK++ was installed but lapackppConfig.cmake could not be located under $LAPACKPP_INSTALL"

#==============================================================================
# Random123. Header-only: fetch and use in place.
#==============================================================================
if (( BUILD_RANDOM123 )); then
    if (( ! FRESH )) && stamp_matches "$RANDOM123_INSTALL" "$RANDOM123_STAMP"; then
        skip_step "Random123 ... reusing existing install"
    else
        run_step "Fetching Random123 ($RANDOM123_REF)" \
            clone_pinned "$RANDOM123_URL" "$RANDOM123_INSTALL" "$RANDOM123_REF"
        write_stamp "$RANDOM123_INSTALL" "$RANDOM123_STAMP"
    fi
    RANDOM123_DIR="$RANDOM123_INSTALL/include/"
fi

#==============================================================================
# RandLAPACK.
#==============================================================================
RANDLAPACK_BUILD="$RANDNLA_PROJECT_DIR/build/RandLAPACK-build"
mkdir -p "$RANDLAPACK_BUILD"

run_step "Configuring RandLAPACK" \
    cmake -S "$RL_SRC" -B "$RANDLAPACK_BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DRequireCUDA="$RANDLAPACK_CUDA" \
        -Dlapackpp_DIR="$LAPACKPP_CMAKE_DIR" \
        -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
        -DRandom123_DIR="$RANDOM123_DIR" \
        -DCMAKE_INSTALL_PREFIX="$RANDLAPACK_INSTALL_DIR" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -DBUILD_TESTS=OFF -DRandLAPACK_BUILD_TESTS=ON \
        "${OPENMP_FLAGS[@]}"
run_build_step "Building and installing RandLAPACK" \
    cmake --build "$RANDLAPACK_BUILD" -j "$JOBS" --target install

RANDLAPACK_CMAKE_DIR="$(find_cmake_config "$RANDLAPACK_INSTALL_DIR" RandLAPACK)"
RANDLAPACK_LIB_DIR="$(dirname "$(dirname "$RANDLAPACK_CMAKE_DIR")")"

#==============================================================================
# Verification.
#
# Compile, link and *run* a program against the finished install. Configuring
# successfully is not the same as producing something that works, and one
# failure mode here is genuinely silent: BLAS++ and LAPACK++ guard the
# int64_t -> blas_int downcast, so an oversized dimension throws, but that guard
# keys off sizeof(blas_int) as declared by the *header*. If the headers say
# 64-bit while the library actually loaded is LP64, the guard compiles out and
# 64-bit values reach routines reading 32 bits -- which surfaces not as wrong
# numbers but as nonsense control values, an absurd workspace size, and a run
# that dies in allocation or never finishes. Nothing catches that by inspection;
# only running something does.
#
# The test goes through BLAS++ and LAPACK++ rather than raw dgemm_, because that
# is how RandLAPACK reaches them, and it calls gesdd specifically: that is the
# routine Apple's legacy Accelerate gets wrong, so a broken SVD shows up here
# rather than inside somebody's RSVD results.
#==============================================================================
CONFTEST_DIR="$RANDNLA_PROJECT_DIR/build/conftest"
rm -rf "$CONFTEST_DIR"; mkdir -p "$CONFTEST_DIR/src"

cat > "$CONFTEST_DIR/src/CMakeLists.txt" <<'CONFTEST_CMAKE'
cmake_minimum_required(VERSION 3.21)
project(randlapack_conftest CXX)
find_package(RandLAPACK REQUIRED)
add_executable(conftest conftest.cc)
# RandLAPACK::RandLAPACK, not the bare name: the namespaced alias is what the
# installed package exports and what carries the transitive BLAS++/LAPACK++
# interface, including their include directories. Linking bare "RandLAPACK"
# compiles until the first #include <blas.hh>. This mirrors what
# benchmark/CMakeLists.txt links.
target_link_libraries(conftest RandLAPACK::RandLAPACK)
CONFTEST_CMAKE

cat > "$CONFTEST_DIR/src/conftest.cc" <<'CONFTEST_CC'
// Exercise the whole stack the way RandLAPACK does: a BLAS++ gemm and a
// LAPACK++ gesdd, with the numbers checked. A half-linked BLAS, a
// header/library integer-width disagreement, or Apple's broken
// divide-and-conquer gesdd all show up here rather than in a user's results.
#include <blas.hh>
#include <lapack.hh>
#include <blas/defines.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
#if defined(BLAS_ILP64)
    std::printf("blas_ilp64=1\n");
#else
    std::printf("blas_ilp64=0\n");
#endif

    // A 6x3 matrix with known singular values: columns scaled 3, 2, 1 from an
    // orthonormal basis, so the singular values must come back {3, 2, 1}.
    const int64_t m = 6, n = 3;
    std::vector<double> A(m * n, 0.0);
    A[0 + 0 * m] = 3.0;
    A[1 + 1 * m] = 2.0;
    A[2 + 2 * m] = 1.0;

    // Round-trip through gemm first: C = A^T A, whose eigenvalues are {9,4,1}.
    std::vector<double> C(n * n, 0.0);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               n, n, m, 1.0, A.data(), m, A.data(), m, 0.0, C.data(), n);
    const double expect_diag[3] = {9.0, 4.0, 1.0};
    for (int64_t i = 0; i < n; ++i) {
        if (std::fabs(C[i + i * n] - expect_diag[i]) > 1e-10) {
            std::printf("FAIL: gemm gave C[%lld,%lld] = %f, expected %f\n",
                        (long long)i, (long long)i, C[i + i * n], expect_diag[i]);
            return 1;
        }
    }

    // Then gesdd, the routine Apple's legacy Accelerate computes incorrectly.
    std::vector<double> S(n), U(m * n), VT(n * n);
    int64_t info = lapack::gesdd(lapack::Job::SomeVec, m, n, A.data(), m,
                                 S.data(), U.data(), m, VT.data(), n);
    if (info != 0) {
        std::printf("FAIL: gesdd returned info = %lld\n", (long long)info);
        return 1;
    }
    const double expect_sv[3] = {3.0, 2.0, 1.0};
    for (int64_t i = 0; i < n; ++i) {
        if (std::fabs(S[i] - expect_sv[i]) > 1e-9) {
            std::printf("FAIL: gesdd gave singular value %lld = %f, expected %f\n",
                        (long long)i, S[i], expect_sv[i]);
            return 1;
        }
    }
    std::printf("OK\n");
    return 0;
}
CONFTEST_CC

verify_install() {
    cmake -S "$CONFTEST_DIR/src" -B "$CONFTEST_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$RANDLAPACK_INSTALL_DIR" \
        -DRandLAPACK_DIR="$RANDLAPACK_CMAKE_DIR" \
        -Dlapackpp_DIR="$LAPACKPP_CMAKE_DIR" \
        -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
        -DRandom123_DIR="$RANDOM123_DIR" \
        -DCMAKE_BUILD_RPATH="$BLASPP_LIB_DIR;$LAPACKPP_LIB_DIR;$RANDLAPACK_LIB_DIR" \
        "${OPENMP_FLAGS[@]}" >> "$LOG" 2>&1
    cmake --build "$CONFTEST_DIR/build" -j "$JOBS" >> "$LOG" 2>&1
    "$CONFTEST_DIR/build/conftest" > "$CONFTEST_DIR/output.txt" 2>&1
    grep -q '^OK$' "$CONFTEST_DIR/output.txt"
}
run_step "Verifying the install links and runs" verify_install
cat "$CONFTEST_DIR/output.txt" >> "$LOG"

# Read the width back from what was actually compiled rather than trusting what
# this script asked for; they differ whenever BLAS++ came from elsewhere.
case "$(sed -n 's/^blas_ilp64=//p' "$CONFTEST_DIR/output.txt" | head -n1)" in
    1) OBSERVED_WIDTH="ILP64 (64-bit BLAS integers)" ;;
    0) OBSERVED_WIDTH="LP64 (32-bit BLAS integers)" ;;
    *) OBSERVED_WIDTH="unknown" ;;
esac

#==============================================================================
# Extras and benchmarks.
#
# Built by default, unlike RandBLAS's examples: these need only RandLAPACK,
# BLAS++, LAPACK++ and Random123, every one of which is already built by this
# point, so there is nothing extra to fetch and nothing new that can fail.
#
# If GPU support is disabled AND blaspp was built from source, keep them from
# auto-detecting CUDA. With an external blaspp, its own config dictates whether
# CUDAToolkit is required.
#==============================================================================
DISABLE_CUDA_FLAG=()
if [[ "$RANDLAPACK_CUDA" == "OFF" ]] && (( ! EXTERNAL_BLASPP )); then
    DISABLE_CUDA_FLAG=(-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE)
fi

configure_subproject() {  # <source-subdir> <build-dir>
    cmake -S "$RL_SRC/$1" -B "$2" \
        -DCMAKE_BUILD_TYPE=Release \
        -DFETCHCONTENT_BASE_DIR="$RANDNLA_PROJECT_DIR/build/fetchcontent-cache/" \
        -DRandLAPACK_DIR="$RANDLAPACK_CMAKE_DIR" \
        -Dlapackpp_DIR="$LAPACKPP_CMAKE_DIR" \
        -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
        -DRandom123_DIR="$RANDOM123_DIR" \
        -DCMAKE_BUILD_RPATH="$BLASPP_LIB_DIR;$LAPACKPP_LIB_DIR;$RANDLAPACK_LIB_DIR" \
        "${DISABLE_CUDA_FLAG[@]}" "${OPENMP_FLAGS[@]}"
}

if (( WANT_EXTRAS )); then
    run_step "Configuring extras" \
        configure_subproject "extras" "$RANDNLA_PROJECT_DIR/build/extras-build"
    run_build_step "Building extras" \
        cmake --build "$RANDNLA_PROJECT_DIR/build/extras-build" -j "$JOBS"
fi
if (( WANT_BENCHMARKS )); then
    run_step "Configuring benchmarks" \
        configure_subproject "benchmark" "$RANDNLA_PROJECT_DIR/build/benchmark-build"
    run_build_step "Building benchmarks" \
        cmake --build "$RANDNLA_PROJECT_DIR/build/benchmark-build" -j "$JOBS"
fi

#==============================================================================
# Shell config: opt-in only. The default prints what to add and touches nothing.
#==============================================================================
if [[ "$(basename "${SHELL:-bash}")" == "zsh" ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$UNAME_S" == "Darwin" ]]; then
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
    note "Added RANDNLA_PROJECT_DIR and RANDNLA_PROJECT_GPU_AVAIL to $SHELL_RC (open a new shell to pick them up)."
fi

#==============================================================================
# Summary.
#==============================================================================
printf '\n%s%sRandLAPACK installed successfully.%s\n\n' "$C_OK" "$C_BOLD" "$C_OFF"
printf '  Backend            %s, %s\n' "$BLAS_BACKEND" "$OBSERVED_WIDTH"
printf '  GPU support        %s\n' "$( [[ "$RANDLAPACK_CUDA" == "ON" ]] && echo "CUDA" || echo "disabled" )"
printf '  OpenMP             %s\n' "$( ((WANT_OPENMP)) && echo enabled || echo disabled )"
printf '  Project layout     %s\n' "$RANDNLA_PROJECT_DIR"
printf '  Installed library  %s\n' "$RANDLAPACK_INSTALL_DIR"
(( WANT_EXTRAS ))     && printf '  Extras             %s\n' "$RANDNLA_PROJECT_DIR/build/extras-build"
(( WANT_BENCHMARKS )) && printf '  Benchmarks         %s\n' "$RANDNLA_PROJECT_DIR/build/benchmark-build"
printf '  Full build log     %s\n' "$LOG"

if (( ${#WARNINGS[@]} )); then
    printf '\n%s%d warning(s) from this run:%s\n' "$C_WARN" "${#WARNINGS[@]}" "$C_OFF"
    for w in "${WARNINGS[@]}"; do printf '  - %s\n' "$w"; done
fi

printf '\n  Run the test suite:\n    ctest --test-dir %s\n' "$RANDLAPACK_BUILD"
printf '\n  Consume from CMake with:\n    -DRandLAPACK_DIR=%s\n' "$RANDLAPACK_CMAKE_DIR"

if [[ "$MODIFY_RC" != "1" ]]; then
    printf '\n  The benchmark scripts expect two environment variables. This script\n'
    printf '  does not edit your shell config; add them yourself if needed:\n'
    printf '    %s\n    %s\n' "$EXPORT_DIR" "$EXPORT_GPU"
    printf '  (or re-run with --modify-rc)\n'
fi
printf '\n'
