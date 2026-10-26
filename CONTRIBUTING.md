# Contributing to RandLAPACK

Questions, ideas, and half-formed plans are all welcome on
[our Discord server](https://discord.gg/R4qj8Er9YW). This page covers the
mechanics of contributing code.

## Getting a working setup

Run `bash install.sh` from your clone (see [docs/INSTALL.md](docs/INSTALL.md)
for manual installation, and
[docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) for native Windows).
With GoogleTest installed, the installer builds the test suite; verify your
baseline before changing anything:

```shell
ctest --test-dir ../RandNLA-project/build/RandLAPACK-build
```

After editing headers, rebuild and re-run the tests from that same build
directory. Because RandLAPACK is header-only, downstream projects (extras,
benchmarks, your own code) pick up changes only after an install from the
RandLAPACK build directory (`cmake --build . --target install`, which is
`make install` with the default generator).

## How the library fits together

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first, and
[devnotes/idioms.md](devnotes/idioms.md) for the C++ idioms we use
deliberately (templated callables for duck typing, C++20 concepts, zero-copy
transpose views, and so on).

Conventions enforced in review:

- Column-major layout, `int64_t` dimensions and indices, element type as a
  template parameter.
- Raw buffers for matrix/vector outputs and workspaces, no `std::vector` for
  anything sized by the problem in new code. An output the routine allocates
  or grows is passed as a pointer reference (`T*&`), as `RSVD::call` does
  with its factors.
- New code throws (`randlapack_require`, `RandLAPACK::Error`) rather than
  asserts, with messages that state what was violated and the offending
  values.
- Randomness only through `RandBLAS::RNGState` (counter-based Philox), never
  through `std::rand` or `std::mt19937`, so results stay reproducible.
- The RandBLAS submodule is read-only from this repository's point of view:
  develop RandBLAS in its own clone (see
  [docs/INSTALL.md, "RandBLAS is a pinned submodule"](docs/INSTALL.md#randblas-is-a-pinned-submodule)).
  Point the submodule only at a commit on RandBLAS's `main` (for a
  squash-merged pull request, the merge commit; the pull request's own
  commits are not on `main`). The commit that moves the submodule also
  updates `RandLAPACK_RandBLAS_PIN` in `CMakeLists.txt`. Configuring a git
  checkout compares the variable with the committed submodule commit
  (`git ls-tree HEAD`) and fails while the two disagree, so commit both
  together before you configure.
- BLAS++ and LAPACK++ are pinned in `installers/dependency-pins.env`, which
  every installer and CI lane reads and which must match the copy in the
  RandBLAS submodule; CI's `pins` job checks both (see
  [docs/CI.md](docs/CI.md)). A pin therefore moves in RandBLAS first. Once
  that change is on RandBLAS's `main`, one RandLAPACK commit moves the
  submodule to it, updates `RandLAPACK_RandBLAS_PIN`, and copies the new
  pins into `installers/dependency-pins.env`.

## Tests

Every behavioral change needs a test in `test/` (GoogleTest; the tree
mirrors the source layout, and a new file goes into the source list in
`test/CMakeLists.txt`). Run the suite locally before opening a PR; CI
runs it on Linux, macOS and native Windows, both through hand-written build
recipes and through the installers themselves, and a PR cannot merge while a
required check fails ([docs/CI.md](docs/CI.md) lists the lanes and which are
required). CI neither compiles nor runs the GPU tests (`test/**/*.cu`;
[#215](https://github.com/BallisticLA/RandLAPACK/issues/215)), so if your
change touches GPU code, build and run them on a machine with CUDA and say
in the PR that you did.

Tests must be safe to run in parallel. `gtest_discover_tests` runs every
test case as its own process, `ctest -j` runs those processes at the same
time, and CI's core-linux job does exactly that, so two test cases must
never share a file or a directory. A test that needs files gives each case
its own directory under `std::filesystem::temp_directory_path()` and removes
it afterwards, never a fixed or `rand()`-named path: `ScratchDir` in
`extras/test/ext_scratch_dir.hh` does this for the extras tests, and the
fixture in `test/misc/test_matrix_io.cc` for the core suite.

A rule worth knowing for GPU code: `gpu_functions/rl_cuda_kernels.cuh`
derives `USE_CUDA` from `__CUDACC__`, so `.cu` files no longer define it
themselves, and `RandLAPACK.hh` includes the GPU layer only when a CUDA
compiler is processing it. Inside `rl_cuda_kernels.cuh`, kernels and their
launches go inside the `#ifdef USE_CUDA` regions and host-callable wrappers
outside, so the header still compiles without nvcc (the wrappers then throw
when called).

## Experiment branches

We conduct proof-of-concept and benchmarking experiments under version
control. Create a branch like

```
git checkout -b experiments/riley-svdidea-220311
```

The branch name always has the prefix `experiments/<your name>` and ideally
keywords plus a YYMMDD date. Push it to BallisticLA/RandLAPACK if you want
to share it. If you reach a clean example you may want to cite later, mark
the commit with a [git tag](https://en.wikibooks.org/wiki/Git/Advanced#Tags).

## Pull requests

- Keep PRs reviewable: one logical change per commit, present-tense commit
  messages that explain *why*, not just what.
- Reference the issues a PR fixes (`fixes #NNN`) in the commit message or PR
  body rather than in code comments.
- Benchmarks are compiled in CI on Linux and macOS (by the core build jobs
  and by the default installer lanes) but not executed; if your change is
  performance-motivated, include measurements in the PR body (machine, BLAS
  provider, sizes, medians over repeated runs).
