# RandLAPACK conventions

These rules govern library code under `RandLAPACK/`. Code written before them is being
converted one driver family at a time. Until a routine is converted it keeps its old
interface, though it may already use the helpers below internally.

The rules share one idea, modeled on cuSOLVER's handles and parameter objects: **a driver
object is a handle**. It holds the options, the workspace it reuses across calls, the results
whose size it cannot know in advance, and diagnostics about the last call. `call` is the
function that does the work.

## Configuration: enums, templates and data

| What is being chosen | Mechanism | Examples |
|---|---|---|
| Which algorithm | A driver class | CQRRPT, BQRRP, RSVD, REVD2, ABRIK, CQRRT |
| Which of RandLAPACK's own implementations of a step | An `enum class` in the driver's `Options` | QRCP variant, tall-QR variant, power-iteration stabilization, basis orthogonalization, QB deflation |
| Which numbers | Fields of `Options` with documented defaults | tolerances, block sizes, passes, oversampling |
| What the user brings: the problem's data and math | A template parameter constrained by a C++20 concept | the operand (`DenseLinOp`, sparse, composite or user operators), kernel element oracles, preconditioner operators, seminorm callbacks |
| A starting point or deterministic choice | Optional data passed in | an initial basis or sketch as `T*` |
| Context and state | The driver object | options, workspace, results, `info()`, timing |

**The test:** how the problem is solved is ours, so it is an enum; what is solved belongs to the
user, so it is a concept-constrained template. Enums and templates answer different questions
and mix freely, as cuSOLVER mixes enum algorithm modes with caller-supplied data. There are no
user-supplied strategy objects and no virtual algorithm bases. When a test needs a deterministic
component, the library accepts the data that component would produce (for example an initial
basis) as an optional input.

- `Options` is a plain aggregate with default member initializers, passed to the constructor and
  reachable through `options()`. A driver has no loose public configuration members.
- An enum is shared between drivers only when its values are the same for all of them, so that a
  value a driver does not support stays a compile error. A value a driver cannot use with the
  operand it was given (for example explicit deflation on a non-dense operator) throws.
- Power-iteration stabilization (CholQR, HQR, PLU or none) and basis orthogonalization (CholQR or
  HQR) are separate options: PLU does not produce orthonormal columns.
- One name per concept across the library: `tol`, `block_size`, `verbose`.
- A concept requires everything the constrained code calls, so a type that does not fit fails at
  the constraint rather than inside the body.

## Memory

- No `std::vector` under `RandLAPACK/`. With its default allocator a vector holds ordinary host
  memory, its member functions cannot run in device code, and constructing it with a size
  value-initializes every element from one thread. Tests and benchmarks may use vectors for their
  own buffers.
- Interfaces take `T*` with sizes and leading dimensions.
- Owned memory is held by RAII types: `std::unique_ptr` and the helpers in `RandLAPACK/rl_memory.hh`,
  or members of those types. `new[]`, `delete[]`, `malloc`, `calloc`, `realloc` and `free` appear
  only inside `rl_memory.hh`. Per-call scratch is `std::unique_ptr<T[]>` from
  `util::make_buffer` (uninitialized) or `util::make_zeroed_buffer`. Every return and every
  exception then releases it, so failure paths need no cleanup code.
- Nothing may rely on memory being zero unless it zeroed it.
- A class with owning members is move-only unless copying is part of its documented API; then it
  defines a deep copy and tests it.
- **Operators hold no mutable scratch.** An operator's apply allocates its own, so two threads may
  apply one operator at once.
- **Operators follow the BLAS rule: when `beta == 0`, `C` is not read.** Callers rely on it when
  they pass uninitialized scratch as `C`.
- GPU drivers can be move-constructed but not move-assigned: a defaulted move assignment would
  replace the queue before releasing the old device buffers through it.

## Workspace

- A driver owns its workspace as grow-only buffers (`util::GrowBuffer`), each sized by one private
  sizing function that `call` also uses, so a size query and the actual use cannot drift.
  `ensure` hands out scratch; `grow` extends results and keeps their prefix.
- Every driver offers `workspace_size(...)` (host and device bytes, plus the bound on rank-adaptive
  results), `reserve(...)`, `release_workspace()`, `release_results()` and `workspace_bytes()`.
  Callers never pass workspace; they may query it and reserve it ahead of a timed call.
- After `reserve`, a call does not grow the driver's own buffers or those of its components,
  for the options in effect when it reserved. Operand applies, diagnostics and dependencies
  (LAPACK++ allocates inside some routines) may still allocate.
- Workspace persists until `release_workspace()` or destruction, so a long-lived handle keeps
  its largest buffers (dense QB's copy of A, for example).
- A driver holds its components by value; a component's buffers and results count as the parent's
  workspace, so the parent's `workspace_size` includes them and its `release_workspace()` frees
  them.
- Free functions and cold paths allocate per call with the RAII helpers.
- A driver object is not safe to call from two threads at once; use one object per thread.

## Results

- Outputs whose size the caller knows (QR factors, `tau`, pivots) are caller-allocated, as in
  LAPACK.
- Outputs whose size depends on the rank reached (QB's Q and B, RSVD's U, S and V, REVD2's
  eigenpairs) are owned by the driver, grow with the rank actually reached, and are read through
  `MatrixView` accessors. Unless the driver was reserved, their capacity stays within
  `max(block_size, 2 x the largest rank reached since construction or release_results())`. A view
  is valid until the next non-const call on the driver or until the driver is moved or destroyed.
  A component's results are not valid after its parent's call.

## Dense and operator entry points

When an algorithm accepts both a dense matrix and an operator, it has one implementation over the
operator concept, and the dense entry point wraps its buffer in `DenseLinOp` and calls it (RS, RF,
QB, RSVD). A step that genuinely differs between the two becomes an option of that one
implementation: QB deflates either an explicit copy of A (dense storage only, `m*n` extra memory)
or implicitly through the operator (about `2(m+n)rb` extra flops per application at accepted rank
`r` and block size `b`), and explicit deflation is cheaper once `r > mn/((p+2)(m+n))`, where `p` is
the row sketcher's number of passes over the data. `Deflation::automatic` keeps the entry point's choice: explicit from the dense entry
point, implicit from the operator one.

Algorithms that differ throughout stay separate drivers: CQRRT works in place on a dense A, while
its operator version cannot modify A, works through operator products and is Q-less by default. QR
with column pivoting needs explicit entries and takes only dense input. RPCholesky takes an element
oracle, and KRILL a kernel operator that must also supply entries; a general operator provides
neither.

## Errors and numerical outcomes

Two channels, following LAPACK's INFO, cuSOLVER's status and device info, and Demmel et al.,
"Proposed Consistent Exception Handling for the BLAS and LAPACK" (arXiv:2207.09281):

1. **A violated contract throws `RandLAPACK::Error`**: bad dimensions, null buffers, incompatible
   options, too small a buffer. Allocation failure is `std::bad_alloc`. Exceptions from BLAS++,
   LAPACK++ and RandBLAS propagate and are documented. The library prints nothing outside verbose
   or timing modes.
2. **What the numerics did is returned as `RandLAPACK::Status`** (`RandLAPACK/rl_status.hh`). A
   driver's `call` returns it, and the detail (stop reason, rank reached, iterations, failing
   stage, LAPACK info) goes in `info()`. An early stop that still gives a usable result (QB
   stopping on error growth or at the rank limit) is `Success` with the reason in `info()`. Free
   functions return `Status`, take the RNG state as an in/out parameter, and return other results
   through out-parameters.

- No sub-routine status is discarded: each one maps to a `Status` or is a documented recovery.
  `Status` is `[[nodiscard]]`, so dropping one is a compiler warning.
- Every iteration has a hard limit, and stopping tests are written so that a NaN neither loops
  forever nor counts as converged.
- Exceptions never escape an OpenMP parallel region: a callback called inside one has its first
  exception captured and rethrown after the region.
- After an exception the driver holds no leaked memory and can be called again; outputs may be
  partly written. On the GPU, a sticky CUDA error poisons the context and is the exception.

## NaN and Inf

A NaN or Inf in an input reaches the output or is reported, never a finite answer that looks
valid. With `Options::check_finite` (on by default) a driver scans the entries of its dense inputs
that it actually reads: `m` rows of each `lda` column, and for symmetric input the stored triangle
(both triangles in full-storage modes); padding and an unstored triangle are never read. A violation
returns `Status::NonFiniteInput`. Operator inputs cannot be scanned, so a driver scans its outputs
and returns `Status::NonFiniteResult`. With the scan off, input finiteness is the caller's
responsibility, and the guarantee covers only values the computation reads.

## GPU code

- GPU drivers allocate, copy, set and synchronize device memory through BLAS++ and call solvers
  through LAPACK++, which carry CUDA, ROCm and SYCL backends. The CUDA kernel layer keeps its stream
  signatures.
- Release paths never throw: device buffer destructors catch and drop errors and keep releasing,
  since a throwing destructor would end the host process (a MATLAB or Python session).
- Every exit from a call that has enqueued work drains the queue first, so a failed call never
  returns while the device still reads the caller's memory.
- CUDA error checks throw `RandLAPACK::Error`; they never abort.

## Tests

- The test programs in `test/` define `RandLAPACK_POISON_WORKSPACE` and
  `RandLAPACK_ENABLE_TEST_HOOKS` for every translation unit. Poisoning fills memory handed out
  uninitialized with NaN (or `-1` for integers), so code that reads stale or uninitialized memory
  fails. `RL_TEST_FAULT("name")` lets a test force a failure path that no input reaches
  deterministically.
- A driver's tests include a reuse test (call on one input, then on another, and compare with a
  fresh object), NaN and Inf injection at entries the driver reads, and a test for each `Status`
  it can return.
- A template that no test instantiates goes into `test/misc/test_instantiations.cc`, so that
  coverage reports count it (see [CI.md](CI.md#coverage)). A GPU template goes into
  `test/drivers/test_instantiations_gpu.cu`, so that the CUDA build compiles all of it.
