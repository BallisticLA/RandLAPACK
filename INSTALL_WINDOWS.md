# Installing RandLAPACK on native Windows

This guide covers building RandLAPACK with Microsoft's Visual C++ compiler
(MSVC) on Windows -- no WSL, no Cygwin. It assumes no prior Windows
development experience. If you are on Linux or macOS, use
[INSTALL.md](INSTALL.md) and `install.sh` instead.

## 1. Quick start

Install the two prerequisites (skip any you already have), from a regular
PowerShell window:

```powershell
winget install --id Git.Git --exact

winget install --id Microsoft.VisualStudio.2022.BuildTools --exact `
  --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Build Tools is the compiler without the IDE. If you would rather have the full
IDE, swap the second line for
`--id Microsoft.VisualStudio.2022.Community` with
`--add Microsoft.VisualStudio.Workload.NativeDesktop`. Either way
`--includeRecommended` matters: it pulls in "C++ CMake tools for Windows",
which is what supplies CMake and Ninja, so you do not install those
separately. And `--wait` matters: without it winget returns while the Visual
Studio installer is still running, which looks like it finished.

Then open **"x64 Native Tools Command Prompt for VS 2022"** from the Start
menu (the exact entry matters -- see section 3), and run:

```bat
git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
cd RandLAPACK
powershell -ExecutionPolicy Bypass -File .\install\install.ps1
```

Two details in that last line, both deliberate. It is a `cmd` prompt, so the
PowerShell script is launched rather than typed directly. And Windows blocks
PowerShell scripts by default (`Restricted` policy on a fresh machine), so
`-ExecutionPolicy Bypass` lets this one script run without permanently
loosening a machine-wide security setting.

That is the whole install. The script uses an Intel oneMKL you already have,
and otherwise downloads a pinned copy for you, then builds the remaining
dependencies, builds RandLAPACK, and runs its test suite. Everything lands in
a sibling `RandNLA-project\` directory; re-running the script reuses what is
already built.

Unlike Linux and macOS, you are not expected to install a BLAS library first.
Windows has no system location for third-party libraries, so projects acquire
their own -- that is what vcpkg, Conan, and NuGet exist for. Anything the
installer downloads goes inside `RandNLA-project\`, never into your system,
and deleting that directory removes it completely.

If no oneMKL is found, the installer explains what it searched and asks before
downloading anything:

```
No existing oneMKL found (checked -MklRoot, $env:MKLROOT, $env:ONEAPI_ROOT,
and C:\Program Files (x86)\Intel\oneAPI\mkl\latest).

A pinned, checksum-verified copy (~155 MB) can be downloaded into
  ...\RandNLA-project\install
It is used only by this project: nothing is installed system-wide, no PATH
or registry changes, and deleting that directory removes it completely.

Download oneMKL now? [Y/n]
```

Answering no prints the alternatives and stops. Questions are skipped when the
installer is not attached to a terminal (a script, a pipeline, CI), taking the
default shown in brackets; `-Yes` skips them in an interactive session too.

If you would rather supply the library yourself, use `-MklRoot` to point at an
existing oneMKL, or `-NoDownload` to make a missing one an error instead of a
download. See §4.

## 2. How this differs from Linux and macOS

If you are used to Unix systems, four things explain nearly everything this
installer does differently:

| | Linux / macOS | Windows |
|---|---|---|
| Getting BLAS/LAPACK | you install one first (`apt install libopenblas-dev`, `brew install openblas`); the installer errors without it | no system location for libraries exists, so the installer discovers an existing oneMKL and otherwise fetches a pinned copy into the project directory. Pass `-NoDownload` for the Linux/macOS behavior |
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
anything missing.

Two mistakes account for nearly every failed Windows install, and both are
about *which shell you start from*:

1. **A regular PowerShell.** `cl.exe`, `cmake`, and `ninja` are on PATH only
   inside a Visual Studio "developer" shell, which sets them up per session.
2. **A developer shell of the wrong architecture.** This one is easy to hit
   because the obvious Start-menu entries are the wrong ones: **"Developer
   PowerShell for VS 2022"** and **"Developer Command Prompt for VS 2022"**
   both default to a **32-bit (x86)** toolchain. RandLAPACK and every BLAS
   backend here are 64-bit, and a 32-bit linker cannot use an x64 import
   library. Use **"x64 Native Tools Command Prompt for VS 2022"** instead.

Confirm you are in the right place with:

```
where cl
```

The path it prints must contain `Hostx64\x64`. If it prints
`INFO: Could not find files for the given pattern(s)`, you are in a plain
shell with no compiler; if it contains `Hostx86\x86`, you are in a 32-bit
developer shell. Use `where cl` rather than running `cl` alone: in the failing
cases `cl` either is not found or prints a usage banner, and neither reads as
"wrong shell". The installer's preflight check catches this too, so you cannot
get far down the wrong path.

**If the Start-menu entry is missing or named differently** (Build Tools, a
different edition, or a Visual Studio version other than 2022), this works
from any Command Prompt regardless of edition or version -- it asks Visual
Studio's own locator where it is installed:

```bat
for /f "usebackq delims=" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvars64.bat"
```

The `-products *` matters: without it `vswhere` reports only full Visual
Studio editions and silently finds nothing on a Build Tools install.

### Script execution policy

Windows refuses to run PowerShell scripts at all under its default
`Restricted` policy, with "running scripts is disabled on this system". The
quick start sidesteps this per-invocation with `-ExecutionPolicy Bypass`,
which is the smallest hammer: it applies to that one process and changes
nothing about the machine. If you would rather allow local scripts
permanently, this is the conventional setting, and it affects only your own
account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Choosing a BLAS/LAPACK backend

RandLAPACK does all its heavy arithmetic through BLAS++/LAPACK++, which sit
on a BLAS/LAPACK library of your choice. On Windows the installer supports:

| Backend | Flag | What you get | How it is obtained |
|---|---|---|---|
| **oneMKL** (default) | none needed | fastest option on most x64 CPUs; 64-bit integers (ILP64); MKL-accelerated sparse routines in RandBLAS | discovered from an existing install via `-MklRoot`, `MKLROOT`, `ONEAPI_ROOT`, or the default oneAPI location; otherwise Intel's official NuGet packages are downloaded, pinned by version and SHA256. `-NoDownload` turns "not found" into an error |
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
                      skips discovery. Backend mkl only. Invalid paths are
                      an error, never a silent fallback to something else.
-NoDownload           Fail instead of downloading a backend that was not
                      found locally. The default fetches one into
                      <ProjectDir>; nothing is installed system-wide, and
                      deleting <ProjectDir> removes it. With -Backend
                      openblas this always fails, because OpenBLAS has no
                      canonical Windows location to discover -- use
                      -Backend custom to supply your own.
-NoOpenMP             Build serially. The default enables OpenMP through
                      MSVC's /openmp:llvm runtime, the only mode that
                      accepts RandLAPACK's 64-bit loop indices; a serial
                      build is fully functional too.
-Yes                  Skip interactive questions, taking each documented
                      default. Questions are already skipped when stdin is
                      not a terminal, so CI never needs this.
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

- **"running scripts is disabled on this system"**: Windows' default
  PowerShell execution policy. Launch it as the quick start does
  (`powershell -ExecutionPolicy Bypass -File .\install\install.ps1`), or see
  "Script execution policy" in section 3.
- **"cl.exe is not on PATH"**: you are in a regular shell. Open "x64 Native
  Tools Command Prompt for VS 2022" and re-run. If Visual Studio is missing
  entirely, the preflight message includes the winget install command.
- **"cl.exe targets x86"**: you are in a developer shell of the wrong
  architecture (see section 3). Open "x64 Native Tools Command Prompt for VS
  2022" instead. If an earlier run already got as far as building
  dependencies, delete the `RandNLA-project` directory before retrying:
  dependencies are reused when present, and the ones configured by the 32-bit
  compiler will keep failing no matter which shell you re-run from.
- **"BLAS library not found" from BLAS++, with oneMKL clearly installed**:
  almost always the 32-bit shell above, on a version of the installer that
  predates the preflight check. The x86 linker rejects the x64 import
  library, and BLAS++ can only report that its probe did not link.
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
