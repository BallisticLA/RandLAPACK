# Installing RandLAPACK on native Windows

This guide covers building RandLAPACK with Microsoft's Visual C++ compiler
(MSVC) on Windows -- no WSL, no Cygwin. It assumes no prior Windows
development experience. If you are on Linux or macOS, use
[INSTALL.md](INSTALL.md) and `install.sh` instead.

## 1. Quick start

RandLAPACK expects the same things on Windows as on Linux and macOS: a **C++
compiler, CMake, Ninja and Git**, available in whatever terminal you choose to
work in. The installer does not supply them and will not install them for you;
it checks for them and stops with a fix-it command if any are missing.

If you already have those, skip to step 3.

**1. Get a toolchain** (one way; use your own if you prefer). Visual Studio's
C++ workload provides MSVC, the Windows SDK, CMake and Ninja together:

```powershell
winget install --id Git.Git --exact

winget install --id Microsoft.VisualStudio.2022.BuildTools --exact `
  --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Build Tools is the compiler without the IDE; `--id Microsoft.VisualStudio.2022.Community`
with `--add Microsoft.VisualStudio.Workload.NativeDesktop` works equally well.
`--includeRecommended` is what pulls in "C++ CMake tools for Windows", which
supplies CMake and Ninja. `--wait` matters too: without it winget returns while
the Visual Studio installer is still running, which looks like it finished.

**2. Make the toolchain visible to your terminal.** This is the one real
difference from Unix: MSVC is never on `PATH` globally, only inside a
"developer environment" that Visual Studio sets up per session. Any of these
gets you one, and they are equivalent — pick whichever fits how you work:

- Open **"x64 Native Tools Command Prompt for VS 2022"** from the Start menu.
- Or, in any Command Prompt, ask Visual Studio where it is and load it (works
  for any edition or version):

  ```bat
  for /f "usebackq delims=" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvars64.bat"
  ```

- Or use any shell your editor or build tool already sets up, as long as it
  gives you an **x64** toolchain (see section 3 — this is worth checking, the
  obvious Start-menu entries give you a 32-bit one).

Whatever you pick, this must show a path containing `Hostx64\x64`:

```
where cl
```

**3. Build:**

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

You are not expected to install a BLAS library first, unlike on Linux and
macOS. If the installer cannot find one it says what it searched and asks
before downloading a pinned copy into the project directory -- nothing is
installed system-wide. Section 4 covers the choices.

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
call the provided staging helper (section 7) or copy the DLLs next to your
`.exe`. This "app-local deployment" is the idiomatic Windows layout -- it is
what Visual Studio's own package manager does by default.

## 3. Prerequisites, in detail

You provide these; the installer does not:

- **A C++ compiler.** MSVC, from Visual Studio 2022 or its Build Tools, with
  the **"Desktop development with C++"** workload.
- **CMake and Ninja.** Both ship with that workload's "C++ CMake tools for
  Windows" component, so a separate install is usually unnecessary. Your own
  copies are fine if they are on `PATH`.
- **Git** (any recent version).
- A network connection for the first run, to fetch the BLAS/LAPACK backend and
  build dependencies (see section 5). Roughly 200 MB for the default backend.

The installer checks all of this up front and prints a fix-it command for
anything missing.

Two mistakes account for nearly every failed Windows install, and both are
about *which shell you start from*:

1. **A regular PowerShell.** `cl.exe`, `cmake` and `ninja` are on `PATH` only
   inside a Visual Studio developer environment, which is set up per session.
2. **A developer shell of the wrong architecture.** This is easy to hit
   because the obvious Start-menu entries are the wrong ones: **"Developer
   PowerShell for VS 2022"** and **"Developer Command Prompt for VS 2022"**
   both default to a **32-bit (x86)** toolchain. RandLAPACK and every BLAS
   backend here are 64-bit, and a 32-bit linker cannot use an x64 import
   library. Note that shell bitness is not a usable signal: "Developer Command
   Prompt" is itself a 64-bit process and still selects x86 tools.

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

**ILP64 and LP64** describe the integer width a BLAS library uses for matrix
dimensions -- 64-bit and 32-bit respectively. It matters only if you mix
libraries built for different widths; the installer keeps it consistent for
you, and the practical difference is that ILP64 lets RandBLAS use oneMKL's
accelerated sparse routines.

### When no oneMKL is found

The installer says what it searched and asks before downloading anything:

```
No oneMKL found, and the download was declined.

  Searched:  -MklRoot (not given)
             $env:MKLROOT           (not set)
             $env:ONEAPI_ROOT       (not set)
             C:\Program Files (x86)\Intel\oneAPI\mkl\latest

  Any of these works:
    winget install --id Intel.oneMKL --exact
    -MklRoot "<path to ...\mkl\latest>"   use a copy you already have
    -Backend openblas                     use OpenBLAS instead
    re-run and answer yes (or pass -Yes)  download a pinned oneMKL,
                                          ~155 MB, into this project only
```

Answering no stops with those options. Questions are skipped entirely when the
installer is not attached to a terminal -- a script, a pipeline, CI -- taking
the default shown in brackets; `-Yes` skips them in an interactive session too.

Notes for the custom path: AMD AOCL downloads sit behind a click-through
license page, so the installer cannot fetch them -- download AOCL yourself,
then point `-BlasLibraries` at its `.lib` files. The custom path is expected
to work with any well-formed backend but is not exercised by our CI; oneMKL
and OpenBLAS are.

## 5. What the installer downloads, and at which versions

The installer never installs a compiler, CMake or Ninja -- those are yours to provide, exactly
as on Linux and macOS (see §3). What it *can* fetch is the BLAS/LAPACK backend and RandLAPACK's
own build-time dependencies. Everything lands under `<ProjectDir>`, is pinned to an exact
version, and is verified by checksum where the source publishes archives:

| Component | Version | Source | Verified by |
|---|---|---|---|
| Intel oneMKL | **2026.1.0.226** | `intelmkl.devel/redist.win-x64` on nuget.org | SHA256 |
| OpenBLAS | **0.3.34** | official GitHub release binaries | SHA256 |
| GoogleTest | **v1.18.0** | release tag | git tag |
| Random123 | **v1.14.0** | release tag | git tag |
| BLAS++ | commit `3057185` | icl-utk-edu/blaspp | git commit |
| LAPACK++ | commit `40b9d0d` | icl-utk-edu/lapackpp | git commit |

All are fetched from the project's canonical upstream and pinned to an immutable reference, so
a given RandLAPACK revision always builds the same dependency versions.

**oneMKL is only downloaded if you do not already have one.** If an existing oneAPI is
discovered (§4), the installer uses *your* version, whatever that is, and downloads nothing.
The version above therefore applies only to the no-oneMKL case.

**Two deliberate exceptions to "pin a stable release".** BLAS++ and LAPACK++ are pinned to
commits rather than to their latest release, `v2025.05.28`, because that release predates the
two one-line MSVC fixes this build needs (icl-utk-edu/blaspp#132 and
icl-utk-edu/lapackpp#87, both merged upstream on
2026-08-06). They move to a release tag as soon as one includes those fixes. Everything else is
a stable, released version.

## 6. `install.ps1` reference

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

## 7. Runtime DLLs: what "staging" means

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

## 8. Troubleshooting

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
  Stage your executable (section 7) and the problem disappears. To
  deliberately build *against* a conda-provided MKL instead, pass
  `-MklRoot "<env>\Library"` (requires conda's `mkl-devel` package).
- **Path-length errors deep in dependency builds**: keep the project
  directory short (e.g. `C:\s\RandLAPACK`), or enable Windows long paths.
- **Full reset**: delete `<ProjectDir>\build` and the relevant
  `<ProjectDir>\install\*` subdirectories, then re-run the installer.

CI runs the exact flows above on every pull request; `docs/CI.md` describes
the lanes, and `.github\scripts\windows\run-ci.ps1 -Task Core
-SetupDependencies [-Backend openblas]` reproduces a CI lane locally.
