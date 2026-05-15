# Spec 003: `Stream*` parameter on compute ops

**Status:** Implemented
**Depends on:** spec 002 (uniform `Status`-returning op surface)
**PR:** (local commit, not yet pushed)
**Author:** Ritwik

---

## 1. Goal

Add `Stream* stream = nullptr` as the final parameter to every compute op that spec 002 normalized. CPU ignores it today, but committing the parameter now means the GPU backend lands without an ABI break — every op already has the handle it needs to dispatch onto a specific stream.

This is an additive, default-valued change. Existing callers (and existing tests) compile unchanged because the parameter defaults to `nullptr`.

## 2. Invariants

- Every op listed in §3 has a final parameter of type `zero::Stream* stream` with default value `nullptr`.
- Calling any op with `stream == nullptr` produces bit-identical output to calling it before this spec.
- Calling any op with a non-null `Stream*` whose `device == Device::CPU` also produces bit-identical output (CPU ignores the handle).
- No op dereferences `stream` on the CPU code path.
- The `Status` return contract from spec 002 is unchanged. The new parameter does not introduce any new failure modes.
- Existing test binaries (`basic_test`, `test_activations`, `test_op_status`) continue to pass without modification.

## 3. API surface

Files modified:

1. `include/zero/ops/elementwise.hpp` — every op + validation helpers stay in place.
2. `include/zero/ops/matmul.hpp` — `gemm` and `matmul`.
3. `include/zero/ops/reduce.hpp` — `reduce_last_axis`, `sum`, `max`, `mean`, `argmax`. Scalar-result reductions (`reduce_all`, `sum_all`, etc.) are **not** touched (they are debug helpers per spec 002 §5).

Signature transformation example:

```cpp
// Spec 002 form:
inline Status add(const Tensor& a, const Tensor& b, Tensor& out) noexcept;

// Spec 003 form:
inline Status add(const Tensor& a, const Tensor& b, Tensor& out,
                  Stream* stream = nullptr) noexcept;
```

The parameter is **always last**. The parameter is **always defaulted to `nullptr`**. No op consumes `stream` in v1; the bodies are unchanged except for forwarding the parameter into wrappers (`add` -> `binary_op`, `matmul` -> `gemm`, `sum` -> `reduce_last_axis`, etc.) so the parameter is preserved through the call chain.

`<../core/status.hpp>` is already included by these headers (from spec 002). `<../device/sync.hpp>` must be added to each ops header so `Stream` is visible.

### Exact list of signatures changed

`elementwise.hpp` — every public op (the same list from spec 002):
- `unary_op`, `binary_op`, `scalar_op`
- `add`, `sub`, `mul`, `div`
- `neg`, `exp`, `log`, `sqrt`
- `tanh`, `relu`, `sigmoid`

`matmul.hpp`:
- `gemm`, `matmul`

`reduce.hpp` (tensor-output only):
- `reduce_last_axis`, `sum`, `max`, `mean`, `argmax`

## 4. Acceptance tests

New test file: `tests/test_op_stream.cpp`.

1. **Default-null call** — for one op from each family (`relu`, `add`, `scalar_op`, `matmul`, `sum`, `argmax`), call without specifying `stream` and assert: returns `ok()`, output bytes match a reference computation.
2. **Explicit null call** — for the same set of ops, call with `stream = nullptr` explicitly and assert the output is byte-identical to the default-null call.
3. **CPU stream call** — construct `Stream::create(Device::CPU)`, pass it to the same set of ops, assert: returns `ok()`, output is byte-identical to the null-stream call (CPU ignores the handle).
4. **Stream sync after op** — call any op with a CPU stream, then call `stream.sync()` and `stream.destroy()`. Assert no segfault and the op's output is still bit-identical to the null-stream version.

Existing tests do not require modification — the default-null parameter preserves binary compatibility at the source level.

## 5. Out of scope

- **GPU backends.** This spec commits the parameter; it does not implement GPU dispatch. Backends consume `stream` later.
- **Async semantics.** Ops remain synchronous on CPU. No "fire-and-forget" behavior is introduced.
- **Stream-returning ops.** No op produces a `Stream`. The parameter is input-only.
- **Stream as part of `Status`.** `Status` is unchanged.
- **Scalar-result reductions.** `reduce_all`, `sum_all`, `max_all`, `min_all`, `mean_all` keep their existing signatures (debug helpers, per spec 002 §5).
- **View ops in `reshape.hpp`.** Out of scope by the same reasoning as spec 002.
- **`Generator` parameter for ops.** RNG state is a separate concern (planned for a later spec).

## 6. Open questions

(none — all resolved before approval)

---

## Amendment log

- *Pre-approval* — Resolved trivially. Tests live in a single `tests/test_op_stream.cpp`. Stream parameter is unconditionally `nullptr`-defaulted and last (no per-op exceptions).
- *Implementation* — Decisions made during impl:
  - Suppressed unused-parameter warnings via `(void)stream;` at the top of each function body that has one (the actual op kernels). The convenience wrappers (`add`, `sub`, `relu`, `sum`, …) forward the parameter into the kernel and therefore do not need the suppression. Once a GPU backend lands, the `(void)stream;` lines come out as `stream` becomes a real argument.
  - `<../device/sync.hpp>` added to `elementwise.hpp`, `matmul.hpp`, `reduce.hpp`.
  - For `gemm`, the new `Stream*` parameter is the **last** argument, after the existing `alpha` and `beta` defaults. This keeps the rule "Stream is always last" consistent and only adds another defaulted parameter.
- *Implementation, verification* — `ctest` 6/6 passing. The spec 002 test (`ZeroOpStatusTest`) passed **unmodified**, which is the direct proof of the spec 003 invariant that `nullptr`-defaulting preserves source-level compatibility for existing callers.
