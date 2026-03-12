# C++ Idioms in RandLAPACK

This document catalogs the notable C++ idioms used in RandLAPACK, with
rationale and examples. It is intended for contributors and reviewers.


## 1. Templated callables for duck typing

**Where**: `rl_rpchol.hh`, `rl_determiter.hh` (pcg), `rl_krill.hh`, `rl_pdkernels.hh`

Many RandLAPACK algorithms accept operator arguments as unconstrained template
parameters (e.g., `typename FUNC_T`, `typename SEMINORM`). The only requirement
is that the object is callable with a specific signature. This is duck typing:
if it quacks like a matrix-vector product, it works.

Lambda functions are the typical way users supply these callables:

```cpp
// rpcholesky expects K_stateless(i, j) to return a kernel value.
auto kernel = [&](int64_t i, int64_t j) -> double {
    // compute K(i,j) from data
    return std::exp(-gamma * dist(i, j));
};
rp_cholesky(n, kernel, k, S, F, b, state);
```

```cpp
// pcg expects a seminorm callable: val = seminorm(dim, s, R)
auto seminorm = [](int64_t rows, int64_t cols, double* R) {
    return blas::nrm2(rows * cols, R, 1);
};
pcg(G, H, s, seminorm, tol, max_iters, N, X, verbose);
```

Templated lambdas (C++20 `auto` parameters) enable even more generic callables.
This is the closest C++ gets to Python-style duck typing while retaining
compile-time type safety and zero-overhead abstraction.


## 2. C++20 concepts for constrained templates

**Where**: `linops/rl_concepts.hh`, all linop files, `rl_syps.hh`, `rl_revd2.hh`

Where duck typing (idiom 1) leaves requirements implicit, C++20 concepts make
them explicit. `LinearOperator` and `SymmetricLinearOperator` are concepts that
require specific member variables (`n_rows`, `n_cols` or `dim`) and a callable
operator with a GEMM-like or SYMM-like signature.

```cpp
// example signature using a C++20 concept in the template parameter
template <linops::SymmetricLinearOperator SLO>
void run_algorithm(SLO &A, int64_t &k, T tol, std::vector<T> &V);
```

Benefits over unconstrained `typename`:
- The compiler rejects non-conforming types at the call site with a clear
  "constraint not satisfied" message, instead of deep template instantiation errors.
- The function signature documents what the type must provide.

RandBLAS uses the same pattern with `SketchingOperator`.


## 3. `mutable` for logically-const members

**Where**: `linops/rl_sparse_linop.hh` (`SparseLinOp::A_sp`)

The `mutable` keyword allows a member to be modified even through a const
reference to the enclosing object.

```cpp
mutable SpMat A_sp;
```

SparseLinOp's block-view methods (`row_block`, `col_block`, `submatrix`) are
`const` because they don't logically modify the operator. However, they must
pass `A_sp` to RandBLAS free functions (e.g., `csr_row_block`) that take
`SpMat&` (non-const), because those functions read structural fields that
aren't declared const in RandBLAS. Declaring `A_sp` as `mutable` resolves this
without casting away const or dropping const from the block methods.

The underlying matrix data is never modified through `A_sp`.


## 4. `shared_ptr` for mixed ownership in return values

**Where**: `linops/rl_composite_linop.hh`, `linops/rl_sparse_linop.hh`

Block-view methods (`row_block`, `col_block`, `submatrix`) return new linop
objects that may or may not own their underlying data. The returned object
stores a `shared_ptr` to heap-allocated block data when it creates new data,
or leaves the `shared_ptr` null when it borrows from the parent.

In `CompositeOperator`, there are three ownership patterns:
- **User-constructed**: both `shared_ptr`s are null; caller owns the operands.
- **row_block()**: `owned_left_` holds the new row-sliced left operand;
  `right_op` still references the original.
- **submatrix()**: both `owned_left_` and `owned_right_` hold new block views.

Declaration order matters: the `shared_ptr` members are declared *before* the
reference members, so C++ destruction order (reverse of declaration) destroys
the references first, then releases the `shared_ptr` data they pointed into.


## 5. Type-erased ownership with `shared_ptr<void>`

**Where**: `linops/rl_sparse_linop.hh` (`SparseLinOp::block_owner_`)

`SparseLinOp` block views can hold different block types (`CSRRowBlockView`,
`CSRColBlock`, `CSCColBlockView`, `CSCRowBlock`) depending on the sparse
format and block orientation. Rather than exposing the block type in the class
template, the owner is stored as `std::shared_ptr<void>`.

This works because `shared_ptr` captures the correct destructor at
construction time (via its deleter), so the pointed-to object is properly
destroyed even though the `shared_ptr`'s template parameter is `void`. This
gives type-erased ownership without virtual dispatch or additional template
parameters.

```cpp
// At construction (inside col_block()):
auto block = std::make_shared<CSRColBlock<T, sint_t>>(...);
// Stored as shared_ptr<void> — destructor for CSRColBlock is remembered.
block_owner_ = block;
```


## 6. `if constexpr` for compile-time type dispatch

**Where**: `linops/rl_sparse_linop.hh`, `misc/rl_util.hh`, RandBLAS `spmm_dispatch.hh`

`if constexpr` evaluates the condition at compile time. The branch not taken
is discarded entirely — it doesn't even need to be valid code for the given
template instantiation. This replaces SFINAE or tag dispatch for choosing
between sparse formats.

```cpp
// SparseLinOp::make_view — create a non-owning view depending on format
if constexpr (std::is_same_v<SpMat, CSR>) {
    return CSR(src.n_rows, src.n_cols, src.nnz, src.vals, src.rowptr, src.colidxs);
} else if constexpr (std::is_same_v<SpMat, CSC>) {
    return CSC(src.n_rows, src.n_cols, src.nnz, src.vals, src.rowidxs, src.colptr);
} else {
    static_assert(!std::is_same_v<SpMat, SpMat>,
        "make_view only supports CSR, CSC, and COO sparse formats");
}
```

The `static_assert` trick in the else branch fires only if that branch is
actually instantiated, giving a clear error for unsupported types.


## 7. `requires` expressions for capability detection

**Where**: `linops/rl_sparse_linop.hh` (sketch operator dispatch)

A `requires` expression tests at compile time whether certain operations are
valid for a type. Combined with `if constexpr`, this enables dispatching based
on what a type *can do* rather than what it *is* — capability-based rather
than identity-based.

```cpp
if constexpr (requires { S.buff; S.layout; S.dist; }) {
    // Dense sketch operator: has a buffer we can read directly.
    // Extract buffer and use SpMM.
} else {
    // Sparse sketch operator: use spgemm.
}
```

This avoids needing separate overloads for dense vs. sparse sketch operators
within the same function body.


## 8. Structured bindings

**Where**: `linops/rl_sparse_linop.hh`, `linops/rl_dense_linop.hh`,
`linops/rl_composite_linop.hh`, RandBLAS `util.hh`

C++17 structured bindings destructure a returned pair/tuple into named
variables, making dimension-query code more readable.

```cpp
auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
randblas_require(rows_A == n_rows);
randblas_require(cols_A == n_cols);
```

Without structured bindings this would require separate calls or temporary
variables for each dimension.


## 9. Lightweight transpose views (zero-copy reinterpretation)

**Where**: RandBLAS `sparse_data/` (CSR/CSC transpose), `linops/rl_sparse_views.hh`

Transposing a sparse matrix doesn't copy data. Instead, a CSR matrix is
reinterpreted as CSC (and vice versa) by swapping the roles of row and column
index arrays. The same physical memory is used to represent both the original and its
transpose.

RandBLAS's `right_spmm` exploits this: rather than implementing a separate
right-multiply kernel, it transforms the operand flags and layout, then
delegates to `left_spmm`. This halves the number of kernel implementations
needed.

This pattern applies more broadly: whenever a mathematical operation can be
expressed as a reinterpretation of existing data rather than a transformation,
prefer the reinterpretation.
