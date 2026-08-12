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
| core-windows | `build-windows` | windows-2022 | MSVC + oneMKL (ILP64, sequential) build + tests, serial (no OpenMP) | no (new) |
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

  **Revert as soon as Mark has reviewed the BLAS++ / LAPACK++ work migrating
  to the new Accelerate interface.** Delete the marked block in
  `.github/workflows/core-macos.yaml` (both `build` and `build-asan`) and this
  entry.

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
- **Windows builds are serial (OpenMP off) for now.** RandLAPACK's OpenMP
  loops need MSVC's `/openmp:llvm` runtime (64-bit indices, `collapse`),
  which RandBLAS's build system does not select yet; see BallisticLA/RandBLAS#184.
  When that lands, core-windows grows an OpenMP leg.
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
| `windows-*-v<N>` (five keys) | core-windows, install-windows | `..\windows-deps` |

To force a rebuild against fresh upstream clones, bump the `-v` suffix in
the relevant key. The Windows keys are additionally salted with a hash of
`setup.ps1`, so editing that script invalidates them automatically.
BLAS++/LAPACK++ track upstream default branches, so a stale cache also
means frozen upstream — bump the suffix when upstream matters.

## Reproducing CI locally

- Linux/macOS core recipe: follow the steps in the workflow file; they are
  ordinary `cmake` + `make` invocations.
- Installer lanes: `bash install.sh --yes --no-gpu` from a fresh clone.
- Windows (from an MSVC developer prompt in the repository root):
  `.github\scripts\windows\run-ci.ps1 -Task Core -SetupDependencies`
  reproduces core-windows; `.\install\install.ps1` reproduces
  install-windows.
