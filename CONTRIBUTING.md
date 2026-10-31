# Contributing to RandLAPACK

Questions, ideas, and half-formed plans are all welcome on
[our Discord server](https://discord.gg/R4qj8Er9YW). This page covers the
mechanics of contributing code.

## Getting a working setup

1. Run `bash install.sh` from your clone
   ([docs/INSTALL_SCRIPT.md](docs/INSTALL_SCRIPT.md); for a manual build see
   [docs/INSTALL_MANUAL.md](docs/INSTALL_MANUAL.md), and for native Windows
   [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md)). With GoogleTest
   installed, the installer builds the test suite.
2. Check that the tests pass before you change anything (the installer's
   final summary prints this build directory; it differs if you set
   `RANDNLA_PROJECT_DIR`):
   ```shell
   OMP_NUM_THREADS=1 ctest --test-dir ../RandNLA-project/build/RandLAPACK-build -j 4
   ```
3. After editing headers, rebuild and re-run the tests from that same build
   directory. RandLAPACK is header-only, so downstream projects (extras,
   benchmarks, your own code) see your changes only after an install from
   the RandLAPACK build directory (`cmake --build . --target install`, which
   is `make install` with the default generator).

## How the library fits together

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first. Its
[Conventions](docs/ARCHITECTURE.md#conventions) section lists the rules
review enforces: column-major layout, `int64_t` dimensions, raw buffers
rather than `std::vector`, throwing `RandLAPACK::Error` rather than
asserting, randomness only through `RandBLAS::RNGState`, and how GPU code is
guarded. [devnotes/idioms.md](devnotes/idioms.md) explains the C++ idioms we
use deliberately (templated callables for duck typing, C++20 concepts,
zero-copy transpose views, and so on).

## RandBLAS and the dependency pins

The RandBLAS submodule is read-only from this repository's point of view.
Develop RandBLAS in its own clone (see
[docs/DEPENDENCIES.md, "RandBLAS is a pinned submodule"](docs/DEPENDENCIES.md#randblas-is-a-pinned-submodule)).

### Moving the RandBLAS submodule

1. Point the submodule only at a commit on RandBLAS's `main`. For a
   squash-merged pull request, that is the merge commit; the pull request's
   own commits are not on `main`.
2. In the same commit, update `RandLAPACK_RandBLAS_PIN` in `CMakeLists.txt`.
3. Commit both before you configure. Configuring a git checkout compares the
   variable with the committed submodule commit (`git ls-tree HEAD`) and
   fails while the two disagree.

### Moving the BLAS++ and LAPACK++ pins

BLAS++ and LAPACK++ are pinned in `installers/dependency-pins.env`, which
every installer and CI lane reads
([docs/DEPENDENCIES.md](docs/DEPENDENCIES.md#why-blas-and-lapack-come-from-forks)
explains the forks). The file must match its copy in the RandBLAS submodule,
and CI's `pins` job checks both the file and the match (see
[docs/CI.md](docs/CI.md)). A pin therefore moves in RandBLAS first:

1. Move the pin in RandBLAS.
2. Once that change is on RandBLAS's `main`, make one RandLAPACK commit that:
   - moves the submodule to it and updates `RandLAPACK_RandBLAS_PIN`;
   - copies the new `*_REF` and `*_TAG` lines into
     `installers/dependency-pins.env`;
   - updates the tag in the pinned-versions table of
     [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md#pinned-versions), the only
     doc that names it.

## Tests

Every behavioral change needs a test in `test/` (GoogleTest; the tree mirrors
the source layout, and a new file goes into the source list in
`test/CMakeLists.txt`). Run the suite locally before opening a PR. CI runs it
on Linux, macOS and native Windows, both through hand-written build recipes
and through the installers themselves, and a PR cannot merge while a
required check fails ([docs/CI.md](docs/CI.md) lists the lanes and which are
required).

**Tests must be safe to run in parallel.** `gtest_discover_tests` runs every
test case as its own process, `ctest -j` runs those processes at the same
time, and CI's core-linux job does exactly that. Two test cases must
therefore never share a file or a directory. A test that needs files gives
each case its own directory under `std::filesystem::temp_directory_path()`
and removes it afterwards, never a fixed or `rand()`-named path. `ScratchDir`
in `extras/test/ext_scratch_dir.hh` does this for the extras tests, and the
fixture in `test/misc/test_matrix_io.cc` for the core suite.

**Coverage is measured, not required.** The `coverage` workflow builds the
suite with `-DRandLAPACK_ENABLE_COVERAGE=ON` (GCC, Debug) and reports which
library lines and branches the tests execute; read its report for the lines
your PR adds. Template code that nothing instantiates is invisible to it
rather than counted as uncovered, so a new template that no test instantiates goes into
`test/misc/test_instantiations.cc` ([docs/CI.md](docs/CI.md#coverage) has the
details and the local command).

**GPU tests are not run by CI.** CI neither compiles nor runs them
(`test/**/*.cu`; [#215](https://github.com/BallisticLA/RandLAPACK/issues/215)).
If your change touches GPU code, build and run them on a machine with CUDA,
and say in the PR that you did.

## Experiment branches

We conduct proof-of-concept and benchmarking experiments under version
control. Create a branch like

```
git checkout -b experiments/riley-svdidea-220311
```

The branch name always starts with `experiments/<your name>`, ideally
followed by keywords and a YYMMDD date. Push it to BallisticLA/RandLAPACK if
you want to share it. If you reach a clean example you may want to cite
later, mark the commit with a
[git tag](https://en.wikibooks.org/wiki/Git/Advanced#Tags).

## Pull requests

- Keep PRs reviewable: one logical change per commit, and present-tense
  commit messages that explain *why*, not just what.
- Reference the issues a PR fixes (`fixes #NNN`) in the commit message or PR
  body, not in code comments.
- CI compiles the benchmarks but never runs them
  ([benchmark/README.md](benchmark/README.md)). If your change is
  performance-motivated, include measurements in the PR body (machine, BLAS
  backend, sizes, medians over repeated runs).
