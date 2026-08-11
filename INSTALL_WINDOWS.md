# Installing RandLAPACK on native Windows

This guide covers building RandLAPACK with Microsoft's Visual C++ compiler
(MSVC) on Windows -- no WSL, no Cygwin. It assumes no prior Windows
development experience. If you are on Linux or macOS, use
[INSTALL.md](INSTALL.md) and `install.sh` instead.

## 1. Quick start

Install the two prerequisites (skip any you already have), from a regular
PowerShell window:

```powershell
winget install Git.Git
winget install Microsoft.VisualStudio.2022.Community --override "--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"
```

Then open **"Developer PowerShell for VS 2022"** from the Start menu (this is
important -- see section 3), and run:

```powershell
git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
cd RandLAPACK
.\install\install.ps1
```

That is the whole install. The script downloads a pinned, checksum-verified
copy of Intel oneMKL, builds the remaining dependencies, builds RandLAPACK,
and runs its test suite. Everything lands in a sibling `RandNLA-project\`
directory; re-running the script reuses what is already built.

## 2. How this differs from Linux and macOS

If you are used to Unix systems, four things explain nearly everything this
installer does differently:

| | Linux / macOS | Windows |
|---|---|---|
| Getting BLAS/LAPACK | one package-manager command (`apt install libopenblas-dev`, `brew install openblas`) | no system package manager for libraries; the installer downloads pinned, SHA256-verified binaries directly from the vendor (Intel's NuGet packages, OpenBLAS's release archives) |
| Where the compiler lives | `gcc`/`clang` always on PATH | MSVC (`cl.exe`) exists only inside a "Developer" shell that Visual Studio sets up per session |
| Finding shared libraries at run time | the binary itself remembers where its libraries are (RPATH), plus system-wide loader paths | executables have no such memory; Windows searches the executable's **own directory first** and PATH **last**, so the installer copies ("stages") every needed DLL next to each executable |
| Default BLAS backend | OpenBLAS (Linux CI), Accelerate (macOS CI) | Intel oneMKL, ILP64 sequential (fastest on typical Windows x64 machines, and enables RandBLAS's MKL-accelerated sparse routines) |

Two smaller differences: OpenMP with MSVC needs the `/openmp:llvm` runtime
(the only MSVC mode that accepts RandLAPACK's 64-bit loop indices), which the
build selects automatically; and GPU support is not available on native
Windows. The build tool is Ninja everywhere, which Visual Studio bundles.

The practical consequence of the third row is worth internalizing: **on
Windows you never need to edit PATH for RandLAPACK**. If an executable is
staged, it runs; if you write your own program against RandLAPACK, either
call the provided staging helper (section 6) or copy the DLLs next to your
`.exe`. This "app-local deployment" is the idiomatic Windows layout -- it is
what Visual Studio's own package manager does by default.

## 3. Prerequisites, in detail

- **Visual Studio 2022** (Community is free; Build Tools also works) with the
  **"Desktop development with C++"** workload. That workload includes MSVC,
  the Windows SDK, CMake, and Ninja -- you do not install those separately.
- **Git** (any recent version).
- A network connection for the first run (dependency downloads; roughly
  200 MB for the default backend).

The installer checks all of this up front and prints a fix-it command for
anything missing. The most common mistake is running from a *regular*
PowerShell: `cl.exe`, `cmake`, and `ninja` are only on PATH inside
**Developer PowerShell for VS 2022** (Start menu, or "Developer Command
Prompt" if you prefer cmd).

## 4. Choosing a BLAS/LAPACK backend

RandLAPACK does all its heavy arithmetic through BLAS++/LAPACK++, which sit
on a BLAS/LAPACK library of your choice. On Windows the installer supports:

| Backend | Flag | What you get | How it is obtained |
|---|---|---|---|
| **oneMKL** (default) | none needed | fastest option on most x64 CPUs; 64-bit integers (ILP64); MKL-accelerated sparse routines in RandBLAS | an already-installed oneAPI is discovered automatically (via `MKLROOT`, `ONEAPI_ROOT`, or the default install location); otherwise Intel's official NuGet packages are downloaded, pinned by version and SHA256 |
| **OpenBLAS** | `-Backend openblas` | solid free backend; 32-bit integers (LP64); RandBLAS's portable sparse fallbacks replace the MKL-only accelerations | official OpenBLAS release binaries, pinned and checksum-verified; the archive is self-contained and includes full LAPACK |
| **Custom / bring-your-own** | `-Backend custom -BlasLibraries <paths>` | anything BLAS++/LAPACK++ can link -- e.g. AMD AOCL, a local ILP64 OpenBLAS build | you provide the import libraries (and their DLL directory via `-BackendBinDir`); the installer verifies them with a small link-and-run check before building anything |

Notes for the custom path: AMD AOCL downloads sit behind a click-through
license page, so the installer cannot fetch them -- download AOCL yourself,
then point `-BlasLibraries` at its `.lib` files. The custom path is expected
to work with any well-formed backend but is not exercised by our CI; oneMKL
and OpenBLAS are.

## 5. `install.ps1` reference

```
-ProjectDir <path>    Where dependencies/builds/installs go
                      (default: ..\RandNLA-project next to the clone).
-Backend <name>       mkl (default) | openblas | custom.
-MklRoot <path>       Use this specific oneMKL install (oneAPI layout);
                      skips discovery and download. Backend mkl only.
-BlasLibraries <p;p>  Backend custom: semicolon-separated .lib paths.
-LapackLibraries <p>  Backend custom: LAPACK .lib paths, if separate from BLAS.
-BackendBinDir <path> Backend custom: directory holding the backend's DLLs.
-BlasInt lp64|ilp64   Backend custom: the library's integer width (default lp64).
-BlasFortran <name>   Backend custom: BLAS++ name-mangling hint (e.g. "add").
-DependencyRoot <p>   Where the dependency stack lives (default: <ProjectDir>\install).
-Fresh                Reconfigure RandLAPACK from scratch (dependencies are
                      still reused; delete <ProjectDir>\install subdirectories
                      to force dependency rebuilds).
-SkipTests            Skip the test suite.
```

Worked examples:

```powershell
# Default: oneMKL, discovered or downloaded.
.\install\install.ps1

# OpenBLAS instead of MKL.
.\install\install.ps1 -Backend openblas

# You already installed the oneAPI toolkit somewhere non-standard.
.\install\install.ps1 -MklRoot "D:\intel\oneAPI\mkl\latest"

# AMD AOCL, downloaded manually beforehand.
.\install\install.ps1 -Backend custom `
    -BlasLibraries "C:\AOCL\lib\AOCL-LibBlis-Win-MT-dll.lib;C:\AOCL\lib\AOCL-LibFlame-Win-MT-dll.lib" `
    -BackendBinDir "C:\AOCL\bin"
```

## 6. Runtime DLLs: what "staging" means

A Windows executable that links a DLL-based library must be able to find
those DLLs when it starts. Windows looks in the executable's own directory
first and PATH last; there is no RPATH. RandLAPACK's build therefore copies
every DLL an executable needs (BLAS++, LAPACK++, and the BLAS backend's
runtime) into the directory of that executable -- tests and benchmarks work
out of the box, from any shell, with no environment preparation.

For your own project, the installed CMake package carries the same helper:

```cmake
find_package(RandLAPACK REQUIRED)
add_executable(myprog main.cc)
target_link_libraries(myprog RandLAPACK)
randlapack_stage_runtime_dlls(myprog)   # no-op on non-Windows platforms
```

If you prefer not to use the helper, copy the DLLs from the backend's `bin`
directory (the installer prints it at the end) next to your `.exe`.

## 7. Troubleshooting

- **"cl.exe is not on PATH"**: you are in a regular shell. Open "Developer
  PowerShell for VS 2022" and re-run. If Visual Studio is missing entirely,
  the preflight message includes the winget install command.
- **A download fails with a hash mismatch**: the pinned artifact changed
  upstream or the download was corrupted. Re-run once; if it persists, open
  an issue -- do not bypass the check.
- **Corporate proxy**: the installer downloads with `curl.exe`, which honors
  the standard `HTTPS_PROXY` environment variable.
- **First build is slow**: Windows Defender real-time scanning inspects
  every new object file. The build is CPU-bound regardless; subsequent
  reruns reuse everything.
- **Wrong results or crashes only inside an activated conda environment**:
  conda ships MKL/OpenBLAS DLLs under the *same filenames* as ours and puts
  its `Library\bin` on PATH when an environment is active. RandLAPACK's own
  staged executables are immune (the exe's folder outranks PATH), but a
  program of yours that relies on PATH can silently pick up conda's copies.
  Stage your executable (section 6) and the problem disappears. To
  deliberately build *against* a conda-provided MKL instead, pass
  `-MklRoot "<env>\Library"` (requires conda's `mkl-devel` package).
- **Path-length errors deep in dependency builds**: keep the project
  directory short (e.g. `C:\s\RandLAPACK`), or enable Windows long paths.
- **Full reset**: delete `<ProjectDir>\build` and the relevant
  `<ProjectDir>\install\*` subdirectories, then re-run the installer.

CI runs the exact flows above on every pull request; `docs/CI.md` describes
the lanes, and `.github\scripts\windows\run-ci.ps1 -Task Core
-SetupDependencies [-Backend openblas]` reproduces a CI lane locally.
