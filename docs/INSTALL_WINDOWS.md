# Installing RandLAPACK on native Windows

This guide builds RandLAPACK with Microsoft's Visual C++ compiler (MSVC) on
Windows, without WSL or Cygwin. It assumes no prior Windows development
experience. On Linux or macOS, use `install.sh` instead
([INSTALL_SCRIPT.md](INSTALL_SCRIPT.md));
[DEPENDENCIES.md](DEPENDENCIES.md#key-terms) explains the key terms
(backend, BLAS++, ILP64, pin, ...).

## 1. Quick start

You need a **C++ compiler, CMake, Ninja and Git**. The installer does not
install them; it checks for them and, if any is missing, stops with a
command that fixes it. You do **not** need a BLAS library: the
installer uses an Intel oneMKL you already have, and otherwise asks before
downloading a pinned copy into the project directory (section 4).

1. **Get a toolchain**, unless you have one (this is one way; use your own
   if you prefer). Visual Studio's C++ workload provides MSVC, the Windows
   SDK, CMake and Ninja together:
   ```powershell
   winget install --id Git.Git --exact

   winget install --id Microsoft.VisualStudio.2022.BuildTools --exact `
     --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```
   Section 3 explains these options and the alternatives.
2. **Open an x64 developer shell.** MSVC is never on `PATH` globally, only
   inside a "developer environment" that Visual Studio sets up per session.
   Open **"x64 Native Tools Command Prompt for VS 2022"** from the Start
   menu, not "Developer PowerShell/Command Prompt for VS 2022": those give a
   32-bit toolchain. (Section 3 lists other ways to get the right shell.)
   Then check:
   ```
   where.exe cl
   ```
   The path it prints must contain `Hostx64\x64`.
3. **Build.** The prompt usually starts in Visual Studio's own folder, where
   you cannot write, so first move to a short folder of your own:
   ```bat
   mkdir C:\src & cd /d C:\src
   git clone --recursive https://github.com/BallisticLA/RandLAPACK.git
   cd RandLAPACK
   powershell -ExecutionPolicy Bypass -File .\installers\install.ps1
   ```
   `-ExecutionPolicy Bypass` lets this one script run although Windows
   blocks PowerShell scripts by default (section 3).

That is the whole install. The script finds or downloads oneMKL, builds the
remaining dependencies, builds RandLAPACK, and runs its test suite.
Everything lands in a sibling `RandNLA-project\` directory; re-running the
script reuses what is already built.

## 2. How this differs from Linux and macOS

If you are used to Unix systems, four things explain nearly everything this
installer does differently:

| | Linux / macOS | Windows |
|---|---|---|
| Getting BLAS/LAPACK | Linux: you install one first (`apt install libopenblas-dev`), and the installer errors without it. macOS: Accelerate is built in | no system location for libraries exists, so the installer discovers an existing oneMKL and otherwise fetches a pinned copy into the project directory. Pass `-NoDownload` for the Linux/macOS behavior |
| Where the compiler lives | `gcc`/`clang` always on PATH | MSVC (`cl.exe`) exists only inside a "Developer" shell that Visual Studio sets up per session |
| Finding shared libraries at run time | the binary itself remembers where its libraries are (RPATH), plus system-wide loader paths | executables have no such memory; Windows searches the executable's **own directory first** and PATH **last**, so the installer copies ("stages") every needed DLL next to each executable |
| Default BLAS backend | OpenBLAS (Linux CI), Accelerate (macOS CI) | Intel oneMKL, ILP64, multithreaded through Intel's OpenMP runtime |

The practical consequence of the third row: **on Windows you do not need to
edit PATH for RandLAPACK** (section 7 has the one exception).

Smaller differences:

- OpenMP with MSVC needs the `/openmp:llvm` mode, the only MSVC mode that
  accepts RandLAPACK's 64-bit loop indices. The build selects it
  automatically.
- With oneMKL, RandLAPACK's OpenMP code runs on Intel's OpenMP runtime, the
  one oneMKL's threading layer uses, so a single thread pool serves both (on
  Linux, GCC's runtime plays that role).
- GPU support is not available on native Windows.
- The installer builds with Ninja, which Visual Studio bundles, rather than
  make.

## 3. Prerequisites, in detail

You provide these; the installer does not:

- **A C++ compiler:** MSVC, from Visual Studio 2022 or its Build Tools, with
  the **"Desktop development with C++"** workload.
- **CMake and Ninja.** Both ship with that workload's "C++ CMake tools for
  Windows" component, so a separate install is usually unnecessary. Your own
  copies are fine if they are on `PATH`.
- **Git** (any recent version).
- **A network connection** for the first run, to fetch the BLAS/LAPACK
  backend and build dependencies (section 5): roughly 200 MB for the default
  backend.

### About the winget commands

- Build Tools is the compiler without the IDE.
  `--id Microsoft.VisualStudio.2022.Community` with
  `--add Microsoft.VisualStudio.Workload.NativeDesktop` works equally well.
- `--includeRecommended` is what pulls in "C++ CMake tools for Windows",
  which supplies CMake and Ninja.
- `--wait` matters: without it, winget returns while the Visual Studio
  installer is still running, which looks like it finished.

### Starting from the right shell

Two mistakes account for nearly every failed Windows install, and both are
about *which shell you start from*:

1. **A regular PowerShell.** `cl.exe`, `cmake` and `ninja` are on `PATH`
   only inside a Visual Studio developer environment, which is set up per
   session.
2. **A developer shell of the wrong architecture.** This is easy to hit,
   because the obvious Start-menu entries are the wrong ones:
   **"Developer PowerShell for VS 2022"** and **"Developer Command Prompt for
   VS 2022"** both default to a **32-bit (x86)** toolchain. RandLAPACK and
   every backend here are 64-bit, and a 32-bit linker cannot use an x64
   import library. Shell bitness is no help: "Developer Command Prompt" is
   itself a 64-bit process and still selects x86 tools.

Any of these gives you an x64 developer environment; they are equivalent:

- **"x64 Native Tools Command Prompt for VS 2022"** from the Start menu.
- **From any Command Prompt**, ask Visual Studio where it is installed and
  load it. This works for any edition or version (Build Tools, another
  edition, a Visual Studio other than 2022), so use it if the Start-menu
  entry is missing or named differently:
  ```bat
  for /f "usebackq delims=" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvars64.bat"
  ```
  The `-products *` matters: without it, `vswhere` reports only full Visual
  Studio editions and silently finds nothing on a Build Tools install.
- **Any shell your editor or build tool sets up**, as long as it gives you an
  x64 toolchain.

Confirm with `where.exe cl`. The path it prints must contain `Hostx64\x64`:

- `INFO: Could not find files for the given pattern(s)` means a plain shell
  with no compiler;
- a path containing `Hostx86\x86` means a 32-bit developer shell.

Use `where.exe cl` rather than `cl` alone: in the failing cases, `cl` is
either not found or prints a usage banner, and neither reads as "wrong
shell". (In PowerShell, plain `where` means something else, hence the
`.exe`.) The installer's preflight check catches this too, so you cannot
get far down the wrong path.

### Script execution policy

Under its default `Restricted` policy, Windows refuses to run PowerShell
scripts at all ("running scripts is disabled on this system"). The quick
start runs from a `cmd` prompt, so it launches the PowerShell script through
`powershell` rather than typing it directly, and adds
`-ExecutionPolicy Bypass`. That is the smallest hammer: it applies to that one
process and changes nothing about the machine. To allow local scripts
permanently, this conventional setting affects only your own account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Choosing a BLAS/LAPACK backend

RandLAPACK does all its heavy arithmetic through BLAS++ and LAPACK++, which
sit on a backend of your choice. On Windows the installer supports:

| Backend | Flag | What you get | How it is obtained |
|---|---|---|---|
| **oneMKL** (default) | none needed | multithreaded through Intel's OpenMP runtime; 64-bit integers (ILP64) | discovered from an existing install via `-MklRoot`, `MKLROOT`, `ONEAPI_ROOT`, or the default oneAPI location; otherwise Intel's official NuGet packages are downloaded, pinned by version and SHA256. Intel's OpenMP runtime comes from the same oneAPI install, or else from its own pinned NuGet packages (~2 MB). `-NoDownload` turns "not found" into an error |
| **OpenBLAS** | `-Backend openblas` | solid free backend; multithreaded with its own thread pool; 32-bit integers (LP64) | official OpenBLAS release binaries, pinned and checksum-verified; the archive is self-contained and includes full LAPACK |
| **Custom** (bring your own) | `-Backend custom -BlasLibraries <paths>` | anything BLAS++/LAPACK++ can link, e.g. AMD AOCL, a local ILP64 OpenBLAS build | you provide the import libraries (and their DLL directory via `-BackendBinDir`); the installer verifies them with a small link-and-run check before building anything |

- **Integer width.** ILP64 and LP64 mean 64-bit and 32-bit integers for
  matrix dimensions. Mixing libraries built for different widths breaks
  silently, so the installer keeps the width consistent for you. Beyond
  that, LP64 caps every single matrix dimension at 2^31 - 1 (about 2.1
  billion); [INSTALL_SCRIPT.md, section 6.3](INSTALL_SCRIPT.md#63-does-lp64-limit-randlapack)
  has more.
- **Paths with spaces work.** The installer hands BLAS++ and LAPACK++ the
  libraries where they are, including a oneMKL at Intel's default location
  (`C:\Program Files (x86)\Intel\oneAPI\mkl\latest`) and `-BlasLibraries` or
  `-LapackLibraries` paths under a folder such as `C:\Program Files`. CI
  builds against a oneMKL copied to such a path to keep this working.
- **RandBLAS's oneMKL-accelerated sparse routines are not active** in the
  installer's Windows builds yet. BLAS++ marks oneMKL as MKL only when
  `mkl.h` is on the compiler's INCLUDE path while it is configured (for
  example in a oneAPI `setvars` shell), which the installer does not
  arrange, so RandBLAS uses its portable sparse code.

### When no oneMKL is found

The installer says what it searched and asks "Download oneMKL now? [Y/n]"
before downloading anything. If you answer no, it stops with these options:

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

Questions are skipped entirely when the installer is not attached to a
terminal (a script, a pipeline, CI), taking the default shown in brackets;
`-Yes` skips them in an interactive session too.

### Custom backends

AMD AOCL downloads sit behind a click-through license page, so the installer
cannot fetch them: download AOCL yourself, then point `-BlasLibraries` at its
`.lib` files (example in section 6). CI exercises `-Backend custom` with
oneMKL's import libraries (the `windows-spaced-path` job); AOCL itself is not
tested.

## 5. What the installer downloads, and at which versions

Besides the BLAS/LAPACK backend, the installer can fetch RandLAPACK's own
build-time dependencies. Everything lands under `<ProjectDir>`, is pinned to
an exact version, and is verified by checksum where the source publishes
archives:

| Component | Version | Source | Verified by |
|---|---|---|---|
| Intel oneMKL | **2026.1.0.226** | `intelmkl.devel/redist.win-x64` on nuget.org | SHA256 |
| Intel OpenMP (for oneMKL) | **2026.1.0.239** | `intelopenmp.devel/redist.win` on nuget.org | SHA256 |
| OpenBLAS | **0.3.34** | official GitHub release binaries | SHA256 |
| GoogleTest | **v1.18.0** | release tag | git tag |
| Random123 | **v1.14.0** | release tag | git tag |
| BLAS++ | the commit in `installers/dependency-pins.env` | [BallisticLA/blaspp](https://github.com/BallisticLA/blaspp), a fork of icl-utk-edu/blaspp | git commit |
| LAPACK++ | the commit in `installers/dependency-pins.env` | [BallisticLA/lapackpp](https://github.com/BallisticLA/lapackpp), a fork of icl-utk-edu/lapackpp | git commit |

- **Every component is pinned to an immutable reference**, so a given
  RandLAPACK revision always builds the same dependency versions. All are
  stable, released versions from the project's canonical upstream, except
  BLAS++ and LAPACK++.
- **BLAS++ and LAPACK++ come from the BallisticLA forks**, pinned by full
  commit hash and tag in
  [`installers/dependency-pins.env`](../installers/dependency-pins.env).
  That one file is read by this installer, by `install.sh` on Linux and
  macOS, and by CI.
  [DEPENDENCIES.md](DEPENDENCIES.md#why-blas-and-lapack-come-from-forks)
  explains why, and lists the fixes the forks carry, among them the MSVC
  build fixes this build needs.
- **oneMKL is downloaded only if you do not already have one.** If an
  existing oneAPI is discovered (section 4), the installer uses *your*
  version, whatever that is, together with the Intel OpenMP runtime from the
  same install. Only if that install has no Intel OpenMP beside it is the
  pinned Intel OpenMP (~2 MB) fetched alone. The oneMKL version above
  therefore applies only when no oneMKL is found.

## 6. `install.ps1` reference

```
-ProjectDir <path>    Where dependencies/builds/installs go (default:
                      $env:RANDNLA_PROJECT_DIR if set, otherwise
                      ..\RandNLA-project next to the clone).
-Prefix <path>        Install RandLAPACK itself here instead of
                      <ProjectDir>\install\RandLAPACK-install. Dependencies
                      still go in the project directory.
-ModifyEnvironment    Persist RANDNLA_PROJECT_DIR for your user account.
                      The default touches nothing and prints the setx
                      command instead.
-Backend <name>       mkl (default) | openblas | custom.
-MklRoot <path>       Use this specific oneMKL install (oneAPI layout);
                      skips discovery. Backend mkl only. Invalid paths are
                      an error, never a silent fallback to something else.
-NoDownload           Fail instead of downloading a backend (or the Intel
                      OpenMP runtime oneMKL needs) that was not
                      found locally. The default fetches one into
                      <ProjectDir>; nothing is installed system-wide, and
                      deleting <ProjectDir> removes it. With -Backend
                      openblas this always fails, because OpenBLAS has no
                      canonical Windows location to discover; use
                      -Backend custom to supply your own.
-NoOpenMP             Build RandLAPACK's own loops serially; the BLAS
                      backend keeps its own threading. The default enables
                      OpenMP through MSVC's /openmp:llvm mode, the only
                      one that accepts RandLAPACK's 64-bit loop indices; a
                      serial build is fully functional too. Add -Fresh
                      when switching an existing build either way.
-Yes                  Skip interactive questions, taking each documented
                      default. Questions are already skipped when stdin is
                      not a terminal, so CI never needs this.
-BlasLibraries <p;p>  Backend custom: semicolon-separated .lib paths for
                      BLAS and LAPACK both (the installer's link check
                      calls a LAPACK routine).
-LapackLibraries <p>  Backend custom: LAPACK .lib paths for LAPACK++ only,
                      if separate; still list them in -BlasLibraries.
-BackendBinDir <path> Backend custom: directory holding the backend's DLLs.
-BlasInt lp64|ilp64   Backend custom: the library's integer width (default lp64).
-BlasFortran <name>   Backend custom: BLAS++ name-mangling hint (e.g. "add").
-DependencyRoot <p>   Where the dependency stack lives (default: <ProjectDir>\install).
-Fresh                Reconfigure RandLAPACK from scratch. Dependencies are
                      still reused unless their pinned source or the backend
                      libraries changed, in which case BLAS++/LAPACK++ are
                      rebuilt automatically; delete <ProjectDir>\install
                      subdirectories to force a rebuild.
-SkipTests            Skip the test suite.
```

Worked examples, each one line, run from the same x64 developer prompt as
the quick start:

```bat
REM Default: oneMKL, discovered or downloaded.
powershell -ExecutionPolicy Bypass -File .\installers\install.ps1

REM OpenBLAS instead of oneMKL (the CI-tested form; see section 7 on OpenMP).
powershell -ExecutionPolicy Bypass -File .\installers\install.ps1 -Backend openblas -NoOpenMP

REM You already installed the oneAPI toolkit somewhere non-standard.
powershell -ExecutionPolicy Bypass -File .\installers\install.ps1 -MklRoot "D:\intel\oneAPI\mkl\latest"

REM AMD AOCL, downloaded manually beforehand.
powershell -ExecutionPolicy Bypass -File .\installers\install.ps1 -Backend custom -BlasLibraries "C:\AOCL\lib\AOCL-LibBlis-Win-MT-dll.lib;C:\AOCL\lib\AOCL-LibFlame-Win-MT-dll.lib" -BackendBinDir "C:\AOCL\bin"
```

## 7. Runtime DLLs: what "staging" means

A Windows executable that links a DLL-based library must find those DLLs
when it starts. Windows looks in the executable's own directory first and in
PATH last; there is no RPATH. RandLAPACK's build therefore copies every DLL
an executable needs (BLAS++, LAPACK++, and the backend's runtime) into that
executable's directory. Tests and benchmarks work out of the box, from any
shell, with no environment preparation. This "app-local deployment" is the
idiomatic Windows layout: Visual Studio's own package manager does the same
by default.

One exception: with `-Backend openblas` and OpenMP on (the default), each
executable also needs MSVC's OpenMP runtime, `libomp140.x86_64.dll`, which
staging does not copy. `-NoOpenMP` avoids it; CI tests OpenBLAS only that
way.

For your own project ([USING.md](USING.md) has a complete example), the
installed CMake package carries the same helper:

```cmake
find_package(RandLAPACK REQUIRED)
add_executable(myprog main.cc)
target_link_libraries(myprog RandLAPACK::RandLAPACK)
randlapack_stage_runtime_dlls(myprog)   # no-op on non-Windows platforms
```

Without the helper, copy the DLLs next to your `.exe` yourself: those in
the `bin` folders of the BLAS++ and LAPACK++ installs, and those in the
backend's DLL directories. The installer prints the backend's directories at
the end; with oneMKL there are two, oneMKL's `bin` and Intel OpenMP's.

## 8. Troubleshooting

- **"running scripts is disabled on this system"**: Windows' default
  PowerShell execution policy. Launch the installer as the quick start does
  (`powershell -ExecutionPolicy Bypass -File .\installers\install.ps1`), or
  see "Script execution policy" in section 3.
- **"cl.exe is not on PATH"**: you are in a regular shell. Open "x64 Native
  Tools Command Prompt for VS 2022" and re-run. If Visual Studio is missing
  entirely, the preflight message includes the winget install command.
- **"cl.exe targets x86"**: you are in a developer shell of the wrong
  architecture (section 3). Open "x64 Native Tools Command Prompt for VS
  2022" instead. If an earlier run already got as far as building
  dependencies, delete the `RandNLA-project` directory before retrying:
  dependencies are reused when present, and the ones configured by the
  32-bit compiler will keep failing no matter which shell you re-run from.
- **"BLAS library not found" from BLAS++, with oneMKL clearly installed**:
  almost always the 32-bit shell above. The x86 linker rejects the x64
  import library, and BLAS++ can only report that its probe did not link.
- **Tests run far slower with `ctest -j N`**: with oneMKL, each test process
  starts a full set of threads (one per core), so N processes oversubscribe
  the machine. Run `ctest` without `-j`, or set `OMP_NUM_THREADS` to a small
  number for parallel test runs; it sizes the thread pool that oneMKL and
  RandLAPACK's OpenMP code share.
- **A download fails with a hash mismatch**: the pinned artifact changed
  upstream or the download was corrupted. Re-run once; if it persists, open
  an issue. Do not bypass the check.
- **Corporate proxy**: the installer downloads with `curl.exe`, which honors
  the standard `HTTPS_PROXY` environment variable.
- **First build is slow**: Windows Defender real-time scanning inspects every
  new object file. The build is CPU-bound regardless; later runs reuse
  everything.
- **Wrong results or crashes only inside an activated conda environment**:
  conda ships oneMKL and OpenBLAS DLLs under the *same filenames* as ours,
  and puts its `Library\bin` on PATH when an environment is active.
  RandLAPACK's own staged executables are immune (the executable's folder
  outranks PATH), but a program of yours that relies on PATH can silently
  pick up conda's copies.
  Stage your executable (section 7) and the problem disappears. To build
  *against* a conda-provided oneMKL deliberately, pass
  `-MklRoot "<env>\Library"` (requires conda's `mkl-devel` package).
- **Path-length errors deep in dependency builds**: keep the project
  directory short (the default `C:\src\RandNLA-project` after the quick start),
  or enable Windows long paths.
- **Full reset**: delete `<ProjectDir>\build` and the relevant
  `<ProjectDir>\install\*` subdirectories, then re-run the installer.

CI runs these same flows on pull requests; [CI.md](CI.md) describes the
lanes and how to reproduce one locally.
