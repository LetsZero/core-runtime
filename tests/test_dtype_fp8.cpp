/**
 * @file test_dtype_fp8.cpp
 * @brief Acceptance tests for spec 001 — FP8 dtype enum values.
 *
 * Tests are derived from docs/specs/001-fp8-dtype.md §4.
 * Each test is annotated with the invariant number it covers.
 *
 * Style: matches basic_test.cpp (custom ASSERT macros, no external test
 * framework). gtest is not wired into the build at the time this test
 * is added.
 */

#include <zero/zero.hpp>
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace zero;

static int failures = 0;

#define ASSERT(cond, msg)                                                       \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("FAIL: %s\n", msg);                                     \
            ++failures;                                                         \
        } else {                                                                \
            std::printf("PASS: %s\n", msg);                                     \
        }                                                                       \
    } while (0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)

int main() {
    std::printf("=== Spec 001 — FP8 dtype acceptance tests ===\n\n");

    // ─────────────────────────────────────────────────────────────────────
    // Test 1 (invariant: pre-existing enum values are unchanged)
    //
    // Hardcoded regression guard. If anyone reorders or removes a DType
    // value, this block fails immediately. ABI contract.
    // ─────────────────────────────────────────────────────────────────────
    std::printf("--- Pre-existing enum value regression ---\n");
    ASSERT_EQ(static_cast<uint8_t>(DType::F16),  0u,  "DType::F16  == 0");
    ASSERT_EQ(static_cast<uint8_t>(DType::F32),  1u,  "DType::F32  == 1");
    ASSERT_EQ(static_cast<uint8_t>(DType::F64),  2u,  "DType::F64  == 2");
    ASSERT_EQ(static_cast<uint8_t>(DType::I8),   3u,  "DType::I8   == 3");
    ASSERT_EQ(static_cast<uint8_t>(DType::I16),  4u,  "DType::I16  == 4");
    ASSERT_EQ(static_cast<uint8_t>(DType::I32),  5u,  "DType::I32  == 5");
    ASSERT_EQ(static_cast<uint8_t>(DType::I64),  6u,  "DType::I64  == 6");
    ASSERT_EQ(static_cast<uint8_t>(DType::U8),   7u,  "DType::U8   == 7");
    ASSERT_EQ(static_cast<uint8_t>(DType::U16),  8u,  "DType::U16  == 8");
    ASSERT_EQ(static_cast<uint8_t>(DType::U32),  9u,  "DType::U32  == 9");
    ASSERT_EQ(static_cast<uint8_t>(DType::U64),  10u, "DType::U64  == 10");
    ASSERT_EQ(static_cast<uint8_t>(DType::Bool), 11u, "DType::Bool == 11");
    ASSERT_EQ(static_cast<uint8_t>(DType::BF16), 12u, "DType::BF16 == 12");

    // ─────────────────────────────────────────────────────────────────────
    // Test 1 (cont.) — new fp8 values land at the documented positions.
    // ─────────────────────────────────────────────────────────────────────
    ASSERT_EQ(static_cast<uint8_t>(DType::F8_E4M3), 13u, "DType::F8_E4M3 == 13");
    ASSERT_EQ(static_cast<uint8_t>(DType::F8_E5M2), 14u, "DType::F8_E5M2 == 14");

    // ─────────────────────────────────────────────────────────────────────
    // Test 2 (invariant: dtype_size returns 1 for both fp8 types)
    // ─────────────────────────────────────────────────────────────────────
    std::printf("\n--- dtype_size ---\n");
    ASSERT_EQ(dtype_size(DType::F8_E4M3), 1u, "dtype_size(F8_E4M3) == 1");
    ASSERT_EQ(dtype_size(DType::F8_E5M2), 1u, "dtype_size(F8_E5M2) == 1");

    // ─────────────────────────────────────────────────────────────────────
    // Test 3 (invariant: dtype_alignment returns 1 for both fp8 types)
    // ─────────────────────────────────────────────────────────────────────
    std::printf("\n--- dtype_alignment ---\n");
    ASSERT_EQ(dtype_alignment(DType::F8_E4M3), 1u, "dtype_alignment(F8_E4M3) == 1");
    ASSERT_EQ(dtype_alignment(DType::F8_E5M2), 1u, "dtype_alignment(F8_E5M2) == 1");

    // ─────────────────────────────────────────────────────────────────────
    // Test 4 (invariant: dtype_is_float is true for both fp8 types)
    // ─────────────────────────────────────────────────────────────────────
    std::printf("\n--- dtype_is_float ---\n");
    ASSERT(dtype_is_float(DType::F8_E4M3), "dtype_is_float(F8_E4M3) is true");
    ASSERT(dtype_is_float(DType::F8_E5M2), "dtype_is_float(F8_E5M2) is true");

    // ─────────────────────────────────────────────────────────────────────
    // Test 5 (invariant: the other predicates return false for fp8)
    // ─────────────────────────────────────────────────────────────────────
    std::printf("\n--- other predicates (must be false) ---\n");
    ASSERT(!dtype_is_signed(DType::F8_E4M3),   "dtype_is_signed(F8_E4M3) is false");
    ASSERT(!dtype_is_signed(DType::F8_E5M2),   "dtype_is_signed(F8_E5M2) is false");
    ASSERT(!dtype_is_unsigned(DType::F8_E4M3), "dtype_is_unsigned(F8_E4M3) is false");
    ASSERT(!dtype_is_unsigned(DType::F8_E5M2), "dtype_is_unsigned(F8_E5M2) is false");
    ASSERT(!dtype_is_logical(DType::F8_E4M3),  "dtype_is_logical(F8_E4M3) is false");
    ASSERT(!dtype_is_logical(DType::F8_E5M2),  "dtype_is_logical(F8_E5M2) is false");

    // ─────────────────────────────────────────────────────────────────────
    // Test 6 (invariant: dtype_name returns the exact spec string)
    // ─────────────────────────────────────────────────────────────────────
    std::printf("\n--- dtype_name ---\n");
    ASSERT(std::strcmp(dtype_name(DType::F8_E4M3), "f8_e4m3") == 0,
           "dtype_name(F8_E4M3) == \"f8_e4m3\"");
    ASSERT(std::strcmp(dtype_name(DType::F8_E5M2), "f8_e5m2") == 0,
           "dtype_name(F8_E5M2) == \"f8_e5m2\"");

    // ─────────────────────────────────────────────────────────────────────
    // Test 7 (invariant: metadata functions remain constexpr-usable)
    //
    // If any of these static_asserts fail to compile, the metadata
    // functions lost their constexpr-ness — a regression.
    // ─────────────────────────────────────────────────────────────────────
    static_assert(dtype_size(DType::F8_E4M3) == 1,
                  "dtype_size must be constexpr and return 1 for F8_E4M3");
    static_assert(dtype_size(DType::F8_E5M2) == 1,
                  "dtype_size must be constexpr and return 1 for F8_E5M2");
    static_assert(dtype_is_float(DType::F8_E4M3),
                  "dtype_is_float must be constexpr and true for F8_E4M3");
    static_assert(dtype_is_float(DType::F8_E5M2),
                  "dtype_is_float must be constexpr and true for F8_E5M2");
    std::printf("\nPASS: constexpr static_asserts compiled successfully\n");

    // Test 8 (debug_print exhaustiveness) is implicitly satisfied by this
    // translation unit compiling at all under -Werror=switch — see spec §4.

    std::printf("\n=== Result: %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
