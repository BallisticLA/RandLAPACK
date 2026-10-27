# Dependencies, versions and tested setups

Where RandLAPACK has been tested, which dependency versions it builds, and
why. The installers take care of all of this for you; this page is for when
you need the details.

## Key terms

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

## Tested setups

Last updated: 2026-10-26.

| Hardware | OS | Compiler | Backend, integer width | OpenMP | Built with | Where, when |
|---|---|---|---|---|---|---|
| x86-64 | Ubuntu 24.04.5 | GCC 13.3.0 | OpenBLAS 0.3.26, LP64 | on | installer; by hand (Release, and Debug + ASan) | CI [install-script] `install-linux`, [core-linux] `build` and `build-asan`; 2026-10-26 |
| x86-64 | Ubuntu 24.04.5 | GCC 13.3.0 | OpenBLAS 0.3.26 (`libopenblas64`), ILP64 | on | installer | CI [install-script] `install-linux-openblas64`; 2026-10-26 |
| Apple Silicon (arm64) | macOS 26.6.2 | Apple Clang 21.0.0 | Accelerate, ILP64 | on (Homebrew `libomp`) | installer | CI [install-script] `install-macos`; 2026-10-26 |
| Apple Silicon (arm64) | macOS 26.6.2 | Apple Clang 21.0.0 | Accelerate, LP64 | on (Homebrew `libomp`) | installer | CI [install-script] `install-macos-lp64`; 2026-10-26 |
| Apple Silicon (arm64) | macOS 26.6.2 | Apple Clang 21.0.0 | Accelerate, LP64 | off | by hand (Release, and Debug + ASan) | CI [core-macos] `build` and `build-asan`; 2026-10-26 |
| Apple Silicon (arm64) | macOS 26.6.2 | Apple Clang 21.0.0 | Homebrew OpenBLAS 0.3.34, LP64 | on (Homebrew `libomp`) | installer (`--blas=openblas`) | CI [install-script] `install-macos`; 2026-10-26 |
| x86-64 | Windows Server 2022 | MSVC 19.44 (Visual Studio 2022) | oneMKL 2026.1, ILP64 | on | installer; by hand | CI [install-script] `install-windows`, [core-windows] `build-windows`; 2026-10-26 |
| x86-64 | Windows Server 2022 | MSVC 19.44 (Visual Studio 2022) | oneMKL 2026.1, ILP64 | off | by hand, also with oneMKL under a path with spaces | CI [core-windows] `build-windows` and `windows-spaced-path`; 2026-10-26 |
| x86-64 | Windows Server 2022 | MSVC 19.44 (Visual Studio 2022) | OpenBLAS 0.3.34, LP64 | off | by hand | CI [core-windows] `build-windows`; 2026-10-26 |
| x86-64 (Intel Core Ultra 9 386H) | Ubuntu 26.04 under WSL2 | GCC 15.2.0 | oneMKL 2026.0, ILP64 | on | installer (`--blas=mkl`); by hand (the recipe in [INSTALL_MANUAL.md](INSTALL_MANUAL.md)) | local; 2026-10-26 |
| x86-64 (Intel Core Ultra 9 386H), NVIDIA GeForce RTX 5070 Ti Laptop GPU (driver 610.88) | Ubuntu 26.04 under WSL2 | GCC 15.2.0, CUDA 13.3 | oneMKL 2026.0, ILP64 | on | installer (`--gpu --no-benchmarks --blas=mkl`) | local; 2026-10-26 |

Other setups may work, but are untested; report results on
[our Discord server](https://discord.gg/R4qj8Er9YW) or in a
[GitHub issue](https://github.com/BallisticLA/RandLAPACK/issues).

- **In every row, RandLAPACK's full test suite passed.** In "Built with",
  "installer" means `install.sh` or `install.ps1`
  ([README](../README.md#install)), and "by hand" means a step-by-step
  CMake build, as in [INSTALL_MANUAL.md](INSTALL_MANUAL.md) or CI's core
  lanes. ASan is AddressSanitizer, a memory-error checker.
- **CI** rows ran on GitHub's hosted runners. Each workflow name links to
  its run, where every job's log can be checked; [CI.md](CI.md) describes
  each lane.
- **GPU.** No CI job builds or runs the GPU code
  ([#215](https://github.com/BallisticLA/RandLAPACK/issues/215)). The GPU row
  is a fresh clone of `main` installed with `install.sh`; all 498 tests
  passed, 22 of them on the GPU. That run predates
  [#232](https://github.com/BallisticLA/RandLAPACK/pull/232), which fixed
  the two GPU issues open then:
  [#213](https://github.com/BallisticLA/RandLAPACK/issues/213) (an abort in
  shuffled or repeated test runs) and
  [#214](https://github.com/BallisticLA/RandLAPACK/issues/214)
  (intermittent CQRRPT_GPU results).

[core-linux]: https://github.com/BallisticLA/RandLAPACK/actions/runs/40406849472
[core-macos]: https://github.com/BallisticLA/RandLAPACK/actions/runs/40406849581
[core-windows]: https://github.com/BallisticLA/RandLAPACK/actions/runs/40406849425
[install-script]: https://github.com/BallisticLA/RandLAPACK/actions/runs/40406849467

## Pinned versions

For a given RandLAPACK revision, the installers always build the same
dependency sources:

| Dependency | Pinned to | Where the pin lives |
|---|---|---|
| BLAS++, LAPACK++ | the BallisticLA forks, tag `ballistic-2026.10.22` | [`installers/dependency-pins.env`](../installers/dependency-pins.env): each fork's URL, full commit hash and tag |
| RandBLAS | one commit | the `RandBLAS` submodule, and `RandLAPACK_RandBLAS_PIN` in `CMakeLists.txt` |
| Random123 | release `v1.14.0` | `installers/install.sh` and the Windows installer |
| Eigen, Spectra, fast_matrix_market | the source archive of one commit, checked by SHA256 | `extras/CMakeLists.txt` and `benchmark/CMakeLists.txt` ([benchmark/README.md](../benchmark/README.md#pinned-downloads-for-maintainers)) |
| Windows downloads (oneMKL, Intel OpenMP, OpenBLAS, GoogleTest) | exact versions, checked by SHA256 or tag | [INSTALL_WINDOWS.md, section 5](INSTALL_WINDOWS.md#5-what-the-installer-downloads-and-at-which-versions) |

## Why BLAS++ and LAPACK++ come from forks

RandLAPACK builds BLAS++ and LAPACK++ from the BallisticLA forks
([blaspp](https://github.com/BallisticLA/blaspp),
[lapackpp](https://github.com/BallisticLA/lapackpp)), not from an upstream
release.

- **What is pinned.**
  [`installers/dependency-pins.env`](../installers/dependency-pins.env)
  records each fork's URL, a full commit hash, and a tag on that commit.
  `install.sh`, `install.ps1` and every CI lane read this one file, and the
  manual recipe
  ([INSTALL_MANUAL.md, section 3](INSTALL_MANUAL.md#3-building-the-dependencies))
  reads it too.
- **Why not upstream.** The latest upstream release, `v2025.05.28`, lacks
  fixes RandLAPACK relies on. Some are already on upstream `master`. The rest
  are seven upstream pull requests that are open but not yet merged or
  released, so only the forks have them.

| Fix | Upstream pull requests | Where it is |
|---|---|---|
| Library paths that contain spaces | [blaspp#137](https://github.com/icl-utk-edu/blaspp/pull/137), [lapackpp#92](https://github.com/icl-utk-edu/lapackpp/pull/92) | forks only |
| Re-configuring a build directory without losing the settings BLAS++ and LAPACK++ detected | [blaspp#141](https://github.com/icl-utk-edu/blaspp/pull/141), [blaspp#142](https://github.com/icl-utk-edu/blaspp/pull/142), [lapackpp#94](https://github.com/icl-utk-edu/lapackpp/pull/94) | forks only |
| `lapack_symbol_suffix`, for symbol-suffixed LAPACK libraries | [lapackpp#91](https://github.com/icl-utk-edu/lapackpp/pull/91) | forks only |
| `gemqrt` undefined behaviour on an empty dimension, which an icpx build turns into a crash (BQRRP makes such a call) | [lapackpp#93](https://github.com/icl-utk-edu/lapackpp/pull/93) | forks only |
| Apple's new Accelerate interface | [blaspp#134](https://github.com/icl-utk-edu/blaspp/pull/134), [lapackpp#88](https://github.com/icl-utk-edu/lapackpp/pull/88) | upstream `master` and forks |
| LAPACK++ compiling against Accelerate's ILP64 | [lapackpp#90](https://github.com/icl-utk-edu/lapackpp/pull/90) | upstream `master` and forks |
| Discovery of an ILP64 OpenBLAS (`libopenblas64`), and `blas_symbol_suffix` | [blaspp#129](https://github.com/icl-utk-edu/blaspp/pull/129) | upstream `master` and forks |
| The MSVC build fixes | [blaspp#132](https://github.com/icl-utk-edu/blaspp/pull/132), [lapackpp#87](https://github.com/icl-utk-edu/lapackpp/pull/87) | upstream `master` and forks |

Each fork's `BALLISTIC.md`
([blaspp](https://github.com/BallisticLA/blaspp/blob/main/BALLISTIC.md),
[lapackpp](https://github.com/BallisticLA/lapackpp/blob/main/BALLISTIC.md))
lists what it carries. Issue
[#204](https://github.com/BallisticLA/RandLAPACK/issues/204) lists each fix
with its upstream pull request and tracks the return to an upstream release.
[CONTRIBUTING.md](../CONTRIBUTING.md#moving-the-blas-and-lapack-pins)
describes how a pin is moved.

## RandBLAS is a pinned submodule

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

## CUDA and GCC versions

GPU support is off by default. It needs:

- the CUDA Toolkit, 12.4.1 or later (12.9.0 recommended), which includes
  cuBLAS and cuSOLVER;
- an NVIDIA driver, v580 or later recommended for CUDA 12.9.0;
- a GCC version your CUDA release accepts: CUDA has strict GCC version
  requirements (table below);
- CMake 3.23 or later (3.24 or later with CUDA 13);
- BLAS++ and LAPACK++ built with `-Dgpu_backend=cuda` instead of the
  `-Dgpu_backend=none` of the manual recipe
  ([INSTALL_MANUAL.md, section 3](INSTALL_MANUAL.md#3-building-the-dependencies)).

Then configure RandLAPACK with `-DRequireCUDA=ON` (`--gpu` for
`install.sh`). If CMake finds no CUDA Toolkit, it prints "CUDA not found"
and builds without GPU support.

| CUDA | GCC | Status | Notes |
|---|---|---|---|
| 13.3.0 | up to 15.x | Supported | Tested with GCC 15.2 |
| **12.9.0** | **13.x** (up to 14.x) | **Recommended** | Tested with GCC 13.3.0 and driver v581.80; `nvcc` rejects GCC above 14 |
| 12.4.1 | 13.x | Supported | Minimum supported version |

The table is not exhaustive. The Supported Compilers table in NVIDIA's CUDA
Installation Guide for Linux lists the host compilers each CUDA release
supports, and the limits `nvcc` enforces may differ from what is documented.
If you hit compiler version errors, try an older GCC.

We recommend **CUDA 12.9.0 with GCC 13.3.0**, for best compatibility and to
avoid known issues in earlier CUDA versions (CUDA 12.4.x, for example, had
problems parsing Intel AMX intrinsics with GCC 13.x). Verified working as of
2025-11-26: CUDA 12.9.0, GCC 13.3.0, CMake 3.31.9 and driver v581.80.
