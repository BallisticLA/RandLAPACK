# RandLAPACK's GitHub CI

Five workflows run on every push to `main` and on every pull request,
except pull requests into the `cqrrp-gpu-benchmarking` branch:

- **core-linux, core-macos, core-windows** replicate the build recipe by
  hand, step by step, so a failure points at a specific step.
- **install-script** runs the user-facing installers end to end, so the
  commands we tell users to type are themselves under test.
- **dependency-pins** checks the BLAS++/LAPACK++ pins that every lane builds.

Each operating system is thus covered by two workflows, a core one and the
installer one.

| Workflow | Job | Runner | Validates | Required check? |
|----------|-----|--------|-----------|-----------------|
| core-linux | `build` | ubuntu-latest | Release build + tests, extras build + tests, benchmark compile (OpenBLAS) | **yes** (`build`) |
| core-linux | `build-asan` | ubuntu-latest | Debug + AddressSanitizer build and test run | no |
| core-macos | `build` | macos-latest | same as Linux `build`, on Apple's toolchain (Accelerate new interface, ILP64, no OpenMP) | **yes** (`build`) |
| core-macos | `build-asan` | macos-latest | Debug + AddressSanitizer build and test run, with `detect_container_overflow=0` (Homebrew's GoogleTest is not built with ASan, which makes libc++'s container checks report false overflows) | no |
| core-windows | `windows-toolchain-guards` | windows-2022 | Architecture-guard decision table, plus assertions that `install.ps1` *refuses* a real x86 and a cross-compiled arm64 toolchain, launched exactly as the docs prescribe | no |
| core-windows | `build-windows` | windows-2022 | MSVC build + tests: oneMKL ILP64 serial, oneMKL ILP64 OpenMP (`/openmp:llvm`), OpenBLAS LP64 serial ("serial" = RandLAPACK's own loops; the BLAS backend is multithreaded in every leg); each leg ends with a stripped-PATH run of a staged test executable | no |
| core-windows | `windows-spaced-path` | windows-2022 | oneMKL ILP64 copied under `Program Files (x86)\Intel\oneAPI\mkl\latest`, once through `-Backend mkl -MklRoot` and once through `-Backend custom`: fresh BLAS++/LAPACK++ builds whose exported config must name that spaced path, then the test suite | no |
| dependency-pins | `pins` | ubuntu-latest | `installers/dependency-pins.env`: each commit exists at its URL, a non-empty tag points at its commit, and the pins match the RandBLAS submodule's | no |
| install-script | `install-linux` | ubuntu-latest | `install.sh`: fresh install, idempotent re-run, dependency-discovery path | **yes** |
| install-script | `install-linux-openblas64` | ubuntu-latest | `install.sh --blas=openblas` with Debian/Ubuntu's `libopenblas64` as the only OpenBLAS: BLAS++ must come out ILP64 and link `libopenblas64`, then the test suite | no |
| install-script | `install-macos` | macos-latest | `install.sh`: fresh install (default = Accelerate new interface, ILP64, no Homebrew BLAS), idempotent re-run, an explicit `--blas-int=ilp64` re-run | **yes** |
| install-script | `install-macos-openblas` | macos-latest | `install.sh --blas=openblas` with Homebrew's OpenBLAS, which is LP64 only, so the installer must fall back to LP64; then the test suite | no |
| install-script | `install-macos-lp64` | macos-latest | `install.sh --blas-int=lp64` on Accelerate: the new interface with an LP64 BLAS++, extras and benchmarks built, then the library and extras tests | no |
| install-script | `install-windows` | windows-2022 | `installers/install.ps1`: fresh install, idempotent re-run | no |

## Required checks

- A pull request cannot merge while `build`, `install-linux` or
  `install-macos` fails, and it must be up to date with `main`.
- Required-check names match **job** names, not "workflow / job" display
  names.
- The Windows jobs, the `pins` job, the two `build-asan` jobs and the
  newest installer lanes (`install-linux-openblas64`, `install-macos-lp64`,
  `install-macos-openblas`) are deliberately not required yet. They become
  candidates once they have a green track record. `install-macos-openblas`
  was split out of the required `install-macos`, so it is the first
  candidate.

## Reproducing CI locally

- **Linux/macOS core recipe:** follow the steps in the workflow file; they
  are ordinary `cmake` + `make` invocations.
- **Installer lanes:** `bash install.sh --yes --no-gpu` from a fresh clone.
- **Windows**, from an x64 developer prompt in the repository root:
  `powershell -ExecutionPolicy Bypass -File .github\scripts\windows\run-ci.ps1 -Task Core -SetupDependencies`
  (add `-Backend openblas` for the OpenBLAS leg) reproduces a core-windows
  leg, and
  `powershell -ExecutionPolicy Bypass -File .\installers\install.ps1`
  reproduces install-windows.

## Things that are deliberate (do not "fix" without reading this)

### Triggers

- **Workflows fire on `pull_request`, and on `push` only for `main`.**
  Firing on `push` for every branch as well would run each commit of a pull
  request twice: same SHA, same result, double the minutes. A branch with an
  open pull request is covered by the `pull_request` event, and `main` still
  gets a run of its own. A branch with no pull request open gets CI on demand
  through `workflow_dispatch`.
- **`concurrency` supersedes in-flight runs, for pull requests only.**
  Pushing a fix cancels the previous run instead of leaving it going to
  completion. On `main`, `cancel-in-progress` is deliberately false: every
  commit there should be validated, not just the newest.

### Linux and macOS

- **core-macos builds BLAS++ with `-Dblas=accelerate` and hard-fails if
  Apple's new Accelerate interface is not selected.** The legacy interface
  returns wrong SVDs
  ([INSTALL_MANUAL.md, "Apple Accelerate"](INSTALL_MANUAL.md#apple-accelerate-macos)
  explains). Both core-macos jobs therefore `grep` for
  `ACCELERATE_NEW_LAPACK` in the installed `defines.h` and fail loudly
  rather than let a silent fallback to the legacy interface back in.
- **core-macos builds BLAS++ ILP64** (`-Dblas_int=int64`), the width
  `install.sh` picks on macOS by default, and checks `defines.h` for
  `BLAS_ILP64` the same way. `install-macos-lp64` is the one lane that builds
  Accelerate LP64; it runs the library and extras tests and builds the
  benchmarks, always with OpenMP. Accelerate LP64 has no AddressSanitizer run
  and no build without OpenMP.
- **core-linux runs its tests in parallel**: `ctest -j "$(nproc)"` for the
  library tests, and `ctest -j 8 --repeat until-fail:5` for the extras, which
  take about a second. `gtest_discover_tests` makes every test case its own
  process, so test cases that share a file or a directory fail here, as they
  do under a local `ctest -j`; serial runs never caught that.
  `OMP_NUM_THREADS` is set low on those steps so that the parallel tests do
  not oversubscribe the runner. The other jobs still run their tests
  serially.
- **The RandBLAS submodule's own tests do not run here**
  (`-DBUILD_TESTS=OFF` in the core recipes). RandBLAS's CI already tests the
  pinned commit, and rebuilding its roughly 480 tests in every RandLAPACK job
  roughly doubled job times. `install.sh` and `install.ps1` turn them off the
  same way, so every installer lane skips them too.
- **No job builds or runs the GPU code**
  ([#215](https://github.com/BallisticLA/RandLAPACK/issues/215)). No job
  configures with `-DRequireCUDA=ON`, and the GPU drivers compile only in
  CUDA translation units, so a change that breaks GPU code still passes CI.
  Build and run the GPU tests locally when you touch GPU code.

### Windows

- **The Windows matrix is deliberately small** (CI-cost budget). The two mkl
  legs cover the default backend, serial and OpenMP; ONE openblas leg covers
  non-MKL provisioning and the LP64 build. There is no openblas-openmp leg
  (OpenMP x MSVC is backend-orthogonal and covered by mkl-openmp), no second
  install-script backend leg (the installer's `-Backend` forwarding is thin;
  provisioning is covered by the openblas core leg), and no Windows ASan
  lane.
- **`windows-spaced-path` builds its dependencies uncached, on purpose.**
  Intel installs oneMKL under `C:\Program Files (x86)\Intel\oneAPI` by
  default, and upstream BLAS++ and LAPACK++ split library paths on spaces. The
  pinned forks carry the fix
  ([blaspp#137](https://github.com/icl-utk-edu/blaspp/pull/137),
  [lapackpp#92](https://github.com/icl-utk-edu/lapackpp/pull/92)), so
  `setup.ps1` passes the libraries where they are. This job is the regression
  check. It copies oneMKL and Intel OpenMP to a path with spaces and
  parentheses, builds BLAS++ and LAPACK++ against it in a fresh dependency
  root, and fails unless BLAS++'s exported config names that path (and, for
  `-Backend custom`, which also passes the libraries as `LAPACK_LIBRARIES`,
  LAPACK++'s does too). A cached build would not exercise the configure step
  the fix lives in.
- **The Windows guard job tests the documented *user* path, which the build
  matrix structurally cannot.** Every build leg initializes MSVC with an
  explicit `arch: x64` under `pwsh`, so none of them exercises the shell the
  install docs tell users to open. A 32-bit toolchain there fails three
  layers down, as BLAS++ reporting "BLAS library not found" (an x86 linker
  cannot use an x64 import library). `windows-toolchain-guards` closes the
  gap from two directions in **one** job. Every check asserts a *refusal* or
  runs pure logic, so each takes seconds and runner start-up dominates;
  splitting them across jobs would multiply the Windows runner count for no
  coverage.
    - The decision table (`test-toolchain-guard.ps1`) drives the guard over
      every architecture. This is the only way to cover **arm64** and
      **arm**: we cannot build for them, so no build leg can ever test them.
    - The integration steps (`assert-toolchain-refused.ps1`) run the real
      installer under a real x86 toolchain and under an `amd64_arm64`
      cross-compiling one, which yields an arm64-targeting `cl.exe` on
      ordinary x64 hardware, so no ARM runner is needed. They launch via
      `cmd` + `-ExecutionPolicy Bypass` under Windows PowerShell 5.1, so the
      documented invocation stays under test too, not just the guard.
    - A step asserting a *failure* must reset `$LASTEXITCODE` before
      returning: `shell: powershell` exits the step with whatever it holds,
      so the intentional non-zero code would report a passing assertion as
      red.

  x86 and arm64 must fail with *different* messages: x86 is a wrong-shell
  mistake with a one-command fix, arm64 an unsupported platform (no oneMKL
  build exists for it). The tests pin that distinction.
- **The architecture check lives in one file, dot-sourced by both callers.**
  `.github/scripts/windows/toolchain-arch.ps1` is shared by `install.ps1` and
  `setup.ps1`, which run independently (CI calls `setup.ps1` alone, users
  call `install.ps1`). Sharing the file avoids keeping two copies in sync.
- **The architecture check reads several signals, not just `cl.exe`'s
  banner.** It prefers `VSCMD_ARG_TGT_ARCH`, then the
  `bin\Host<host>\<target>\cl.exe` path convention, and only then the banner.
  The banner alone would be wrong on a localized Visual Studio, where the
  words around the architecture are translated, and a missed detection here
  fails *open*, silently allowing the exact configuration the check exists
  to reject.
- **CI passes `-Yes`, and prompts are TTY-gated regardless.** `setup.ps1` can
  ask questions (currently: whether to download oneMKL, and whether you
  already have OpenBLAS). Any prompt reaching a runner would block until the
  job times out, so `$script:Interactive` is false whenever stdin is
  redirected or the session is non-interactive, and every question must have a
  defensible unattended default. `action.yml` and `run-ci.ps1` additionally
  pass `-Yes`, so the intent survives any future change to that detection.
  This mirrors `install.sh`, which computes `INTERACTIVE` from `[[ -t 0 ]]`
  plus `--yes` for the same reason.
- **Backends are auto-provisioned by default; `-NoDownload` opts out.**
  Windows has no system prefix for third-party libraries, so per-project
  acquisition (what vcpkg, Conan and NuGet exist for) is the ordinary
  practice rather than a workaround, and it is what makes a one-command
  install possible on a bare machine. oneMKL is still *discovered* first
  (`-MklRoot`, `MKLROOT`, `ONEAPI_ROOT`, default oneAPI path), and the
  download is announced rather than silent. `-NoDownload` gives the stricter
  Linux/macOS behavior, where `install.sh` expects a system BLAS and errors
  without one. Everything downloaded lands under the dependency root, so a
  runner's cache and a user's project directory stay self-contained.
- **The MSVC OpenMP flavor is forced to `/openmp:llvm` in RandLAPACK's own
  CMake** (`CMake/rl_build_options.cmake` and the benchmark project), not
  just in RandBLAS. RandLAPACK's `find_package(OpenMP)` runs before the
  submodule's guard, so without this, classic `-openmp` gets cached. Under
  it, MSVC silently ignores the `collapse` clause (warning C4849) that
  `rl_rpchol` relies on, and the "openmp" leg would not test what its name
  claims.
- **Windows executables are staged, not PATH-dependent.** The BLAS backend
  enters BLAS++ as raw library paths, which `TARGET_RUNTIME_DLLS` cannot see,
  so `RANDLAPACK_RUNTIME_DLL_DIRS` stages the backend DLLs beside every
  test/benchmark executable (app-local deployment). The stripped-PATH CI step
  is the regression gate; do not remove it. A process-PATH prepend remains
  in `run-ci.ps1` and in `setup.ps1` (which `install.ps1` runs).
  RandLAPACK's executables do not rely on it. It was kept for the RandBLAS
  submodule's own test executables, which no Windows build compiles since
  `install.ps1` turned them off; removing it is a separate change.

### Dependencies

- **BLAS++ and LAPACK++ come from the BallisticLA forks, pinned in one
  file.** [`installers/dependency-pins.env`](../installers/dependency-pins.env)
  names the [blaspp](https://github.com/BallisticLA/blaspp) and
  [lapackpp](https://github.com/BallisticLA/lapackpp) forks and a full commit
  hash for each, plus a tag on those commits.
  `install.sh`, `setup.ps1` and every workflow read it, and the
  `pins` job keeps it honest.
  [DEPENDENCIES.md](DEPENDENCIES.md#why-blas-and-lapack-come-from-forks)
  explains why the forks exist and lists the fixes they carry.
- **The pins are commits, never branch names, and the cache keys carry
  them.** A branch tip moves, so a cache hit could restore a different
  revision than a miss builds.
- **Every other Windows dependency is pinned to an immutable ref** and comes
  from its canonical upstream.
- **No lane builds the extras against MKL.** The extras `ToeplitzLinOp` and
  its test need MKL's FFT and are built only when RandBLAS has MKL support;
  core-linux's extras build uses OpenBLAS and skips them, and the Windows
  MKL legs build core only. The same holds for
  `benchmark/bench_toeplitz_ls`, which the benchmark builds skip without
  MKL. Run that test and compile that benchmark locally.

## Caches

Dependencies (BLAS++, LAPACK++, Random123, plus oneMKL, Intel OpenMP,
OpenBLAS and GoogleTest on Windows) are built or downloaded once and cached;
a cache-miss run is several minutes slower than a warm one.

| Cache key prefix | Used by | Lives beside |
|------------------|---------|--------------|
| `core-deps-<OS>-v<N>` | core-linux, core-macos (both jobs) | `runner.workspace`, the checkout's parent (`*-install`) |
| `installer-deps-<OS>-v<N>` | install-linux | `deps-install/` in the workspace |
| `windows-*-r<N>` (per backend + shared) | core-windows (`build-windows`, `windows-spaced-path`), install-windows | `runner.workspace\windows-deps` |

- **To force a rebuild** against fresh upstream clones, bump the `-v`/`-r`
  suffix in the relevant key. When a recipe in `setup.ps1` changes, bump the
  `-r<N>` on the affected keys in `action.yml`.
- **Changing a pin rebuilds BLAS++/LAPACK++ with no suffix bump.** They are
  built at the commits in `installers/dependency-pins.env`, and every key
  that caches them carries those commits.
- **Random123 is unpinned in the core lanes**, which clone its default
  branch, so bumping `-v<N>` is also how they pick up a new Random123.
- **The core-macos key carries "accelerate" and "ilp64" tags**, because
  earlier caches hold dependencies built without `-Dblas=accelerate`, or
  LP64.
- **The Windows keys use MANUAL revision literals** (plus the pinned
  BLAS++/LAPACK++ commits), never `hashFiles()`. That helper resolves paths
  relative to the workspace root, and the install-script workflow checks the
  repo out under `RandLAPACK\`, where `hashFiles()` silently returns an empty
  string, so two workflows would read and write *different* caches.
- **Cache paths start from `runner.workspace` and never contain `..`.**
  `actions/cache` refuses to save a path with `..` in it ("Relative pathing
  '.' and '..' is not allowed"), so a cache spelled that way restores nothing
  and is never saved, without failing the job.
