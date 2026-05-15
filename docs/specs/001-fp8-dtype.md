# Spec 001: FP8 dtype enum values

**Status:** Implemented
**Depends on:** none
**PR:** (local commit, not yet pushed)
**Author:** Ritwik

---

## 1. Goal

Add `F8_E4M3` and `F8_E5M2` as recognized values in the `DType` enum so that downstream tooling (compiler, future backends) can reference fp8 tensors without an ABI break later. No kernels are added in this spec — only the type system entries and the metadata functions that operate on dtypes. Modern accelerators (H100, B200) and modern transformer inference assume fp8 exists; adding the enum values now is the cheapest possible commitment that keeps the freeze honest.

## 2. Invariants

- `DType::F8_E4M3` and `DType::F8_E5M2` exist as members of the `DType` enum.
- The numeric values of all pre-existing `DType` members are unchanged.
- `dtype_size(F8_E4M3) == 1` and `dtype_size(F8_E5M2) == 1`.
- `dtype_alignment(F8_E4M3) == 1` and `dtype_alignment(F8_E5M2) == 1`.
- `dtype_is_float(F8_E4M3) == true` and `dtype_is_float(F8_E5M2) == true`.
- `dtype_is_signed`, `dtype_is_unsigned`, `dtype_is_logical` return `false` for both.
- `dtype_name(F8_E4M3) == "f8_e4m3"` and `dtype_name(F8_E5M2) == "f8_e5m2"`.
- Every `switch(DType)` in the codebase handles both new values without falling through to `unknown` or the default branch silently.
- The repo compiles with `-Werror=switch` (or the platform equivalent) after this change.

## 3. API surface

Files modified:

1. `include/zero/core/dtype.hpp` — add the enum values and update the four metadata switches/predicates.
2. `include/zero/core/scalar.hpp` — add cases to the `debug_print` switch so it remains exhaustive under `-Werror=switch`.
3. `CMakeLists.txt` — add the strict-switch flag.

```cpp
// include/zero/core/dtype.hpp

enum class DType : uint8_t {
    // ... existing values unchanged, in their existing positions ...
    BF16     = 12,

    // New in spec 001 — appended only, never reordered.
    F8_E4M3  = 13,    ///< 8-bit float, 4-bit exponent, 3-bit mantissa (inference)
    F8_E5M2  = 14,    ///< 8-bit float, 5-bit exponent, 2-bit mantissa (training)
};
```

In `dtype.hpp`:
- `dtype_size`: add `case F8_E4M3` and `case F8_E5M2`, both return `1`.
- `dtype_alignment`: unchanged (delegates to `dtype_size`).
- `dtype_is_float`: extend the OR chain so it returns `true` for both new values.
- `dtype_is_signed`, `dtype_is_unsigned`, `dtype_is_logical`: unchanged. They already use `==` comparisons over a closed set, so they correctly return `false` for the new values without modification.
- `dtype_name`: add `case F8_E4M3` returning `"f8_e4m3"` and `case F8_E5M2` returning `"f8_e5m2"`.

In `scalar.hpp`:
- `debug_print` (under `#ifndef NDEBUG`): add `case DType::F8_E4M3` and `case DType::F8_E5M2`. Format: `Scalar(f8_e4m3: bits=0x%02x)` and `Scalar(f8_e5m2: bits=0x%02x)`. Read the underlying byte via the existing `value.u8` union member (fp8 is 1 byte; no new union field required). The `to_f32`/`to_f64`/`to_i64`/`to_bool` switches keep their `default:` branches and silently return zero for fp8, matching today's F16/BF16 behavior.

### Build system

The top-level `CMakeLists.txt` adds `-Werror=switch` (gcc/clang) and `/we4062` (msvc) to the `zero-core` interface target. Under existing `-Werror -Wall` this is largely redundant, but it makes the intent discoverable and survives any future relaxation of `-Werror`.

## 4. Acceptance tests

New test file: `tests/test_dtype_fp8.cpp`.

1. (invariants on existing values) For every pre-existing `DType` member, its underlying `uint8_t` value matches the value it had before this spec. (Hardcode the expected integers in the test — this is the regression guard against accidental reordering.)
2. (size) `dtype_size(F8_E4M3) == 1` and `dtype_size(F8_E5M2) == 1`.
3. (alignment) `dtype_alignment(F8_E4M3) == 1` and `dtype_alignment(F8_E5M2) == 1`.
4. (float predicate) `dtype_is_float` returns `true` for both new values.
5. (other predicates) `dtype_is_signed`, `dtype_is_unsigned`, `dtype_is_logical` each return `false` for both new values.
6. (name) `dtype_name(F8_E4M3)` equals the literal `"f8_e4m3"`; same for `F8_E5M2`.
7. (compile-time) The test file uses the new values in a `constexpr` context to confirm all metadata functions remain `constexpr`.

8. (debug_print exhaustiveness) The test compiles with `-Werror=switch`. A successful build of the test target is itself proof that `debug_print` in `scalar.hpp` is exhaustive over `DType`.

All existing tests in `tests/test_dtype.cpp` must continue to pass unchanged.

## 5. Out of scope

- **No kernels.** No op in `ops/` learns to handle fp8 in this spec. Attempting to `matmul` two fp8 tensors should either be rejected or fall through to the existing dtype-mismatch path. Either is acceptable for now.
- **No conversion functions.** `fp32 -> fp8` and `fp8 -> fp32` casts are explicitly deferred.
- **No `Tensor::alloc` changes.** Allocating an fp8 tensor must already work given the metadata functions; if it doesn't, fix the underlying issue in a separate spec.
- **No documentation pass.** The `CORE_RUNTIME_SPEC.md` rewrite is its own spec; this one only touches the dtype enum and its immediate metadata.
- **No FFI / C API changes.** `c_api.h` does not exist yet.
- **No other dtype additions.** Do not add `INT4`, `FP4`, `complex`, etc. in this spec, even if they seem related.

## 6. Open questions

(none — all resolved before approval)

---

## Amendment log

- *Pre-approval, before any code* — Resolved two open questions: tests live in a new `tests/test_dtype_fp8.cpp`; `-Werror=switch` is included in this spec. Expanded §3 to include `scalar.hpp::debug_print`, which becomes non-exhaustive once `-Werror=switch` is enabled. No other switch over `DType` in the codebase required changes (verified via grep across `include/` and `tests/`).
- *Implementation* — Implemented as specified. All 26 acceptance assertions pass. Full test suite (4 binaries: basic, benchmark, activations, dtype_fp8) builds cleanly under the new `-Werror=switch` flag and all 4 tests pass via `ctest`.
