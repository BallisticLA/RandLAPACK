# Contributing to RandLAPACK

Questions, ideas, and half-formed plans are all welcome on
[our Discord server](https://discord.gg/R4qj8Er9YW). This page covers the
mechanics of contributing code.

## Getting a working setup

Run `bash install.sh` from your clone (see [docs/INSTALL.md](docs/INSTALL.md) for
manual installation). The installer builds the test suite; verify your
baseline before changing anything:

```shell
ctest --test-dir ../RandNLA-project/build/RandLAPACK-build
```

After editing headers, rebuild and re-run the tests from that same build
directory. Because RandLAPACK is header-only, downstream projects (extras,
benchmarks, your own code) pick up changes only after `make install` in the
RandLAPACK build directory.

## How the library fits together

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first, and `devnotes/idioms.md` for
the C++ idioms we use deliberately (duck-typed callables, caller-provided
workspaces, and so on).

Conventions enforced in review:

- Column-major layout, `int64_t` dimensions and indices, element type as a
  template parameter.
- Raw caller-provided buffers for matrix/vector outputs and workspaces; no
  `std::vector` for anything sized by the problem.
- New code throws (`randlapack_require`, `RandLAPACK::Error`) rather than
  asserts, with messages that state what was violated and the offending
  values.
- Randomness only through `RandBLAS::RNGState` (counter-based Philox), never
  through `std::rand` or `std::mt19937`, so results stay reproducible.
- The RandBLAS submodule is read-only from this repository's point of view:
  develop RandBLAS in its own clone (see docs/INSTALL.md, "RandBLAS is a pinned
  submodule").

## Tests

Every behavioral change needs a test in `test/` (GoogleTest; the tree
mirrors the source layout). Run the suite locally before opening a PR; CI
runs it on Linux and macOS, both through the hand-written recipes and
through `install.sh` itself, and a PR cannot merge with failing required
checks. GPU tests (`test/**/*.cu`) only run on machines with CUDA; if your
change touches GPU code, say in the PR whether you ran them.

A quirk worth knowing: translation units that use the GPU headers define
`USE_CUDA` themselves (see `test/drivers/test_bqrrp_gpu.cu`), while the GPU
*benchmark* does not; anything added to `gpu_functions/rl_cuda_kernels.cuh`
must compile in both settings (host-callable helpers go outside the
`#ifdef USE_CUDA` region, kernels inside).

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
- Benchmarks are compile-checked in CI but not executed; if your change is
  performance-motivated, include measurements in the PR body (machine, BLAS
  provider, sizes, medians over repeated runs).
