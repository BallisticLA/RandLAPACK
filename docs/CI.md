# RandLAPACK's GitHub CI

Four workflows run on every push and pull request. They divide into two
lanes per operating system: the **core** workflows hand-replicate the build
recipe step by step (so a failure points at a specific step), and the
**install-script** workflow runs the user-facing installers end to end (so
the thing we tell users to type is itself under test).

| Workflow | Job | Runner | Validates | Required check? |
|----------|-----|--------|-----------|-----------------|
| core-linux | `build` | ubuntu-latest | Release build + tests, extras build + tests, benchmark compile (OpenBLAS) | **yes** (`build`) |
| core-linux | `build-asan` | ubuntu-latest | Debug + AddressSanitizer build and test run | no |
| core-macos | `build` | macos-latest | same as Linux `build`, on Apple's toolchain | **yes** (`build`) |
| core-macos | `build-asan` | macos-latest | Debug + AddressSanitizer build and test run | no |
| core-windows | `build-windows` | windows-2022 | MSVC build + tests: oneMKL ILP64 serial, oneMKL ILP64 OpenMP (`/openmp:llvm`), OpenBLAS LP64 serial; each leg ends with a stripped-PATH run of a staged test executable | no (new) |
| install-script | `install-linux` | ubuntu-latest | `install.sh`: fresh install, idempotent re-run, dependency-discovery path | **yes** |
| install-script | `install-macos` | macos-latest | `install.sh`: fresh install, idempotent re-run | **yes** |
| install-script | `install-windows` | windows-2022 | `install/install.ps1`: fresh install, idempotent re-run | no (new) |

Required-check names match **job** names (`build`, `install-linux`,
`install-macos`), not "workflow / job" display names. The two Windows jobs
and the two `build-asan` jobs are deliberately not required yet; they become
candidates once they have a green track record.

## Things that are deliberate (do not "fix" without reading this)

- **`TestQB.Polynomial_Decay_general1` is QUARANTINED on macOS -- THIS IS
  TEMPORARY AND MUST BE REVERTED.** The test fails on Apple Silicon because
  Apple's default (old) Accelerate LAPACK has a broken divide-and-conquer
  `gesdd`. It was previously left failing as a canary, which made core-macos
  permanently red and trained everyone to ignore the job -- a red CI lane that
  is always red reports nothing.

  **Revert once Apple's new Accelerate interface is in use**, i.e. when both
  of these land (order matters -- lapackpp#88 needs the `defines.h` from
  blaspp#134):
    - <https://github.com/icl-utk-edu/blaspp/pull/134> -- New Apple Accelerate
      support, a rebased and completed continuation of the stalled
      <https://github.com/icl-utk-edu/blaspp/pull/74>
    - <https://github.com/icl-utk-edu/lapackpp/pull/88> -- Support Apple's new
      Accelerate interface

  Then RandLAPACK PR #155 (retire legacy-Accelerate accommodations) becomes
  mergeable, `TestQB.Polynomial_Decay_general1` should pass on its own, and
  the "DELETE THE macOS SUPPRESSION" warning below will fire. Delete the
  marked block in `.github/workflows/core-macos.yaml` (both `build` and
  `build-asan`) and this entry.

  The canary is deliberately preserved: the test is still *run*, separately,
  and cannot fail the job. Every macOS run prints a warning that the
  suppression is active, and if the test ever **passes** the job prints a
  louder one telling you to delete the quarantine -- which is exactly the
  signal the old always-red arrangement was supposed to provide. A local fix
  exists (reference SVD via `gesvd`) but is intentionally unmerged.
- **The RandBLAS submodule's own tests do not run here**
  (`-DBUILD_TESTS=OFF` in the core recipes). The pinned commit is already
  tested by RandBLAS's CI; rebuilding its ~450 tests in every RandLAPACK job
  roughly doubled job times. The install-script lanes keep them, preserving
  exactly what a user's install builds.
- **The Windows matrix is deliberately small** (CI-cost budget): the two
  mkl legs cover the default backend serial + OpenMP, and ONE openblas leg
  covers non-MKL provisioning and the LP64 build. There is no
  openblas-openmp leg (OpenMP x MSVC is backend-orthogonal and covered by
  mkl-openmp), no second install-script backend leg (the installer's
  `-Backend` forwarding is thin; provisioning is covered by the openblas
  core leg), and no Windows ASan lane.
- **Windows executables are staged, not PATH-dependent.** The BLAS backend
  enters BLAS++ as raw library paths, which `TARGET_RUNTIME_DLLS` cannot
  see, so `RANDLAPACK_RUNTIME_DLL_DIRS` stages the backend DLLs beside every
  test/benchmark executable (app-local deployment). The stripped-PATH CI
  step is the regression gate; do not remove it. An internal process-PATH
  prepend remains in run-ci.ps1/install.ps1 only for the RandBLAS
  submodule's own test executables, until RandBLAS gains the same staging
  (planned follow-up).
- **BLAS++ and LAPACK++ on Windows come from BallisticLA fork branches**
  (`BallisticLA/blaspp@remove-symv-debug-print`,
  `BallisticLA/lapackpp@msvc-direct-includes`) carrying two one-line MSVC
  fixes. Upstream PRs are open (blaspp #132, lapackpp #87); once merged, the
  clones in `.github/actions/setup-randlapack-deps-windows/setup.ps1` move
  back to upstream master.

## Caches

Dependencies (BLAS++, LAPACK++, Random123 — plus oneMKL and GoogleTest on
Windows) are built once and cached; a cache-miss run is several minutes
slower than a warm one.

| Cache key prefix | Used by | Lives beside |
|------------------|---------|--------------|
| `core-deps-<OS>-v<N>` | core-linux, core-macos (both jobs) | the workspace (`../*-install`) |
| `installer-deps-<OS>-v<N>` | install-linux | `deps-install/` in the workspace |
| `windows-*-r<N>` (per backend + shared) | core-windows, install-windows | `..\windows-deps` |

To force a rebuild against fresh upstream clones, bump the `-v`/`-r` suffix
in the relevant key. The Windows keys use MANUAL revision literals only --
never `hashFiles()`: that helper resolves paths relative to the workspace
root, and the install-script workflow checks the repo out under
`RandLAPACK\`, where the glob matches nothing and `hashFiles()` silently
returns an empty string. The result was two workflows reading and writing
*different* caches while appearing to share one. When a recipe in
`setup.ps1` changes, bump the `-r<N>` on the affected keys in `action.yml`.
BLAS++/LAPACK++ track upstream default branches, so a stale cache also
means frozen upstream — bump the suffix when upstream matters.

## Reproducing CI locally

- Linux/macOS core recipe: follow the steps in the workflow file; they are
  ordinary `cmake` + `make` invocations.
- Installer lanes: `bash install.sh --yes --no-gpu` from a fresh clone.
- Windows (from an MSVC developer prompt in the repository root):
  `.github\scripts\windows\run-ci.ps1 -Task Core -SetupDependencies
  [-Backend openblas]` reproduces a core-windows leg; `.\install\install.ps1`
  reproduces install-windows.
