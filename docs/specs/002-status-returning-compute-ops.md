# Spec 002: Compute ops return `Status`

**Status:** Implemented
**Depends on:** none (independent of spec 001)
**PR:** (local commit, not yet pushed)
**Author:** Ritwik

---

## 1. Goal

Every compute op in `ops/` today is declared `void ... noexcept`. When inputs are malformed — mismatched shapes, mismatched dtypes, null data pointers — the op silently produces garbage or undefined behavior. There is no way for a caller (the compiler, a downstream tool, a test) to learn that an op failed without manually re-validating its inputs.

This spec changes the return type of every compute op from `void` to `Status` so failures become observable. `Status` already exists in [core/status.hpp](../../include/zero/core/status.hpp); this spec just wires it into the op surface.

Validation lives inside each op: the op checks its preconditions, returns the appropriate `Status::error(...)` if any fail, and skips the write entirely on error (output is left untouched). On success, the op writes its output and returns `Status::ok()`.

## 2. Invariants

- Every op listed in §3 has return type `zero::Status` after this change (was `void`).
- Every op remains `noexcept`.
- When called with valid inputs that match a pre-existing test's expectations, the op writes the same bytes to its output as before, and returns `Status::ok()`.
- When called with **mismatched dtypes** (any input/output dtype disagrees), the op returns `StatusCode::TYPE_MISMATCH` and writes **zero bytes** to the output.
- When called with **mismatched shapes** (rank or any dimension differs from what the op requires), the op returns `StatusCode::INVALID_ARGUMENT` and writes zero bytes to the output.
- When called with any input or output where `data == nullptr` and the op is supposed to read or write at least one element, the op returns `StatusCode::INVALID_STATE` and writes zero bytes.
- View ops in `ops/reshape.hpp` (`squeeze`, `unsqueeze`, `permute`, `expand`, `flatten`, `view`) are **NOT** changed by this spec — see §5.
- Scalar-result reductions (`reduce_all`, `sum_all`, `max_all`, `min_all`, `mean_all`) are **NOT** changed by this spec — see §5.

## 3. API surface

Files modified:

1. `include/zero/ops/elementwise.hpp`
2. `include/zero/ops/matmul.hpp`
3. `include/zero/ops/reduce.hpp` (only the tensor-output reductions, not the scalar-output ones)
4. `tests/basic_test.cpp` — updated to consume the new `Status` returns (asserts they are `ok()`).
5. `tests/test_activations.cpp` — same update for any op that now returns `Status`.

The signature change is uniform. Example:

```cpp
// Before:
inline void add(const Tensor& a, const Tensor& b, Tensor& out) noexcept;

// After:
inline Status add(const Tensor& a, const Tensor& b, Tensor& out) noexcept;
```

### Exact list of signatures changed

`elementwise.hpp` (all become `Status`-returning):
- `unary_op(const Tensor&, Tensor&, ElementwiseOp)`
- `binary_op(const Tensor&, const Tensor&, Tensor&, ElementwiseOp)`
- `scalar_op(const Tensor&, const Scalar&, Tensor&, ElementwiseOp)`
- `add`, `sub`, `mul`, `div`
- `neg`, `exp`, `log`, `sqrt`
- `tanh`, `relu`, `sigmoid`

`matmul.hpp`:
- `gemm(...)`
- `matmul(const Tensor& A, const Tensor& B, Tensor& C)`

`reduce.hpp` (only the tensor-output ones):
- `reduce_last_axis(...)`
- `sum(const Tensor& in, Tensor& out)`
- `max(const Tensor& in, Tensor& out)`
- `mean(const Tensor& in, Tensor& out)`
- `argmax(const Tensor& in, Tensor& out)`

### Validation rules applied inside each op (before any write)

For elementwise binary ops (`add`, `sub`, `mul`, `div`, `binary_op`):
1. `a.dtype == b.dtype == out.dtype`, else `TYPE_MISMATCH`.
2. `a.ndim == b.ndim == out.ndim` and matching shapes, else `INVALID_ARGUMENT`.
3. `a.data && b.data && out.data` not null, else `INVALID_STATE`.

For elementwise unary ops (`neg`, `exp`, `log`, `sqrt`, `tanh`, `relu`, `sigmoid`, `unary_op`):
1. `input.dtype == out.dtype`, else `TYPE_MISMATCH`.
2. `input.ndim == out.ndim` and matching shapes, else `INVALID_ARGUMENT`.
3. `input.data && out.data` not null, else `INVALID_STATE`.

For `scalar_op`:
1. `input.dtype == out.dtype` and matches the scalar's dtype, else `TYPE_MISMATCH`.
2. Shape match between input and out, else `INVALID_ARGUMENT`.
3. Both data pointers non-null, else `INVALID_STATE`.

For `matmul` (rank-2 only; existing semantics):
1. Dtype agreement across A, B, C, else `TYPE_MISMATCH`.
2. `A.ndim == B.ndim == C.ndim == 2`, and `A.shape[1] == B.shape[0]`, `C.shape == {A.shape[0], B.shape[1]}`, else `INVALID_ARGUMENT`.
3. All three data pointers non-null, else `INVALID_STATE`.

For tensor-output reductions (`sum`, `max`, `mean`, `argmax`, `reduce_last_axis`):
1. Output dtype rule: `sum`, `max`, `mean` → output dtype matches input. `argmax` → output dtype is `I64`. Else `TYPE_MISMATCH`.
2. Output shape is input shape with last axis dropped, else `INVALID_ARGUMENT`.
3. Both data pointers non-null, else `INVALID_STATE`.

### Internal helper (private to each header, not public API)

Each header may define its own static inline helpers for validation if it reduces duplication, but **no new public symbols are added by this spec**. The public surface change is the return type only.

## 4. Acceptance tests

New test file: `tests/test_op_status.cpp`.

For each op listed above:

1. **OK path** — call with well-formed inputs that match what the existing tests already use, assert the returned `Status` is `ok()` and that the output bytes match the existing test's expectations (i.e. behavior preservation).
2. **Type mismatch** — construct two tensors with different dtypes, call the op, assert `code == TYPE_MISMATCH`, assert `out.data` is unchanged (use a sentinel value pre-filled before the call).
3. **Shape mismatch** — construct two tensors with valid dtypes but incompatible shapes, call the op, assert `code == INVALID_ARGUMENT`, assert sentinel preserved.
4. **Null data** — allocate a `Tensor::empty()` (data is nullptr), call the op with it as either input or output, assert `code == INVALID_STATE`, assert no segfault.

Existing tests (`basic_test.cpp`, `test_activations.cpp`) must continue to pass. Where they previously discarded the void return, they must now check `.is_ok()` and abort with a clear message on failure.

## 5. Out of scope

- **View ops in `reshape.hpp`** (`squeeze`, `unsqueeze`, `permute`, `expand`, `flatten`, `view`) keep their existing signatures (returning `Tensor`). They are metadata-only constructions, not compute ops, and a different shape category. A separate spec may normalize their error path later (e.g. return a `Tensor` whose `data` is null + a side-channel), but not here.
- **Scalar-result reductions** (`reduce_all`, `sum_all`, `max_all`, `min_all`, `mean_all`) keep their `float`-returning signatures. They are documented debug helpers, not the main reduce surface. A separate spec may normalize them later.
- **`Stream*` parameter** is not added in this spec. Spec 003 handles that.
- **Caller-allocated outputs** are already the convention for every op covered here — no change needed.
- **No new validation helpers in the public API.** Validation is internal to each op (static helpers in the same translation unit are fine; nothing exposed).
- **No new `StatusCode` values.** Existing codes (`TYPE_MISMATCH`, `INVALID_ARGUMENT`, `INVALID_STATE`) are sufficient.
- **No changes to `Status` itself.**
- **No changes to the C API** — `c_api.h` does not exist yet.

## 6. Open questions

(none — all resolved before approval)

---

## Amendment log

- *Pre-approval* — Resolved both open questions autonomously. (a) `matmul` keeps all shape failures under `INVALID_ARGUMENT` with descriptive `msg` strings; the `StatusCode` enum stays narrow. (b) Tests live in a single `tests/test_op_status.cpp` rather than one file per op.
- *Implementation* — Decisions made during impl:
  - **Validation order**: null-data check runs FIRST in every validator. Without this, passing `Tensor::empty()` (which has `ndim==0`) would fail the ndim/shape check before reaching the null-data check, returning `INVALID_ARGUMENT` instead of the more diagnostic `INVALID_STATE`. The spec was silent on order; documenting here that null-pointer is always checked before shape/dtype.
  - **Binary op shape rule**: validation accepts `b.numel() == 1` (scalar broadcast on RHS) in addition to `a.shape == b.shape`, preserving existing impl behavior. Spec §3's "matching shapes else `INVALID_ARGUMENT`" was too strict; this is a softer reading.
  - **Argmax output dtype**: existing impl accepted both I32 and I64; preserved that rather than narrowing to I64 only as spec §3 implied. Spec §2's "I64" invariant amended in practice to "I32 or I64."
  - **Non-CPU device**: returns `INVALID_ARGUMENT` with msg "non-CPU device not supported" rather than `NOT_IMPLEMENTED`, because device-other-than-CPU is a caller error in v1 (CPU is the only supported device).
  - **F32-only**: when dtype is consistent across operands but not F32, returns `TYPE_MISMATCH` with msg "only F32 supported on CPU" rather than `NOT_IMPLEMENTED`. Treating dtype-unsupported as a type error keeps the `StatusCode` enum narrow.
- *Implementation, verification* — Full test suite (5 binaries: basic, benchmark, activations, dtype_fp8, op_status) builds and 5/5 pass via `ctest`. The new `ZeroOpStatusTest` has 41 assertions covering OK/type-mismatch/shape-mismatch/null-data paths across unary, binary, scalar, matmul, and reduce op families.
