/**
 * @file test_op_status.cpp
 * @brief Acceptance tests for spec 002 — Status-returning compute ops.
 *
 * Tests derived from docs/specs/002-status-returning-compute-ops.md §4.
 *
 * For each affected op:
 *  1. OK path — well-formed inputs return Status::ok(); output behavior
 *     is preserved against the previous void contract.
 *  2. Type mismatch — output is left untouched and code == TYPE_MISMATCH.
 *  3. Shape mismatch — output is left untouched and code == INVALID_ARGUMENT.
 *  4. Null data — code == INVALID_STATE; no segfault.
 *
 * Sentinel pattern: outputs are pre-filled with 0xAB so we can verify
 * "writes zero bytes on error" by checking the buffer is unchanged.
 */

#include <zero/zero.hpp>
#include <cstdio>
#include <cstring>
#include <cstdint>

using namespace zero;
using namespace zero::ops;

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

// Fill a tensor's data buffer with the sentinel byte 0xAB.
static void fill_sentinel(Tensor& t) noexcept {
    if (t.data != nullptr) {
        std::memset(t.data, 0xAB, t.nbytes());
    }
}

// Check that every byte of t.data is still the sentinel.
static bool is_sentinel(const Tensor& t) noexcept {
    if (t.data == nullptr) return true;
    const uint8_t* p = static_cast<const uint8_t*>(t.data);
    for (size_t i = 0; i < t.nbytes(); ++i) {
        if (p[i] != 0xAB) return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Elementwise unary (covers unary_op + every unary wrapper)
// ─────────────────────────────────────────────────────────────────────

static void test_unary() {
    std::printf("\n--- unary ops ---\n");

    int64_t shape3[] = {3};
    Tensor a   = Tensor::alloc(shape3, 1, DType::F32);
    Tensor out = Tensor::alloc(shape3, 1, DType::F32);
    static_cast<float*>(a.data)[0] = -1.0f;
    static_cast<float*>(a.data)[1] =  0.0f;
    static_cast<float*>(a.data)[2] =  2.0f;

    // OK path: relu
    fill_sentinel(out);
    ASSERT(relu(a, out).is_ok(), "relu OK returns ok()");
    ASSERT(static_cast<float*>(out.data)[2] == 2.0f, "relu OK behavior preserved");

    // OK path: unary_op directly with NEG
    ASSERT(unary_op(a, out, ElementwiseOp::NEG).is_ok(), "unary_op(NEG) OK");
    ASSERT(static_cast<float*>(out.data)[2] == -2.0f, "NEG behavior");

    // Type mismatch: output dtype differs
    Tensor out_i32 = Tensor::alloc(shape3, 1, DType::I32);
    fill_sentinel(out_i32);
    Status s_tm = relu(a, out_i32);
    ASSERT(s_tm.code == StatusCode::TYPE_MISMATCH, "relu type mismatch -> TYPE_MISMATCH");
    ASSERT(is_sentinel(out_i32), "relu type mismatch leaves output untouched");
    out_i32.free();

    // Shape mismatch
    int64_t shape4[] = {4};
    Tensor out_bad = Tensor::alloc(shape4, 1, DType::F32);
    fill_sentinel(out_bad);
    Status s_sm = relu(a, out_bad);
    ASSERT(s_sm.code == StatusCode::INVALID_ARGUMENT, "relu shape mismatch -> INVALID_ARGUMENT");
    ASSERT(is_sentinel(out_bad), "relu shape mismatch leaves output untouched");
    out_bad.free();

    // Null data
    Tensor null_t = Tensor::empty();
    Status s_nd = relu(null_t, out);
    ASSERT(s_nd.code == StatusCode::INVALID_STATE, "relu null input -> INVALID_STATE");

    a.free();
    out.free();
}

// ─────────────────────────────────────────────────────────────────────
// Elementwise binary (covers binary_op + add/sub/mul/div + scalar broadcast)
// ─────────────────────────────────────────────────────────────────────

static void test_binary() {
    std::printf("\n--- binary ops ---\n");

    int64_t shape4[] = {4};
    Tensor a   = Tensor::alloc(shape4, 1, DType::F32);
    Tensor b   = Tensor::alloc(shape4, 1, DType::F32);
    Tensor out = Tensor::alloc(shape4, 1, DType::F32);
    for (int i = 0; i < 4; ++i) {
        static_cast<float*>(a.data)[i] = static_cast<float>(i + 1);
        static_cast<float*>(b.data)[i] = static_cast<float>(i + 5);
    }

    // OK path: add
    ASSERT(add(a, b, out).is_ok(), "add OK returns ok()");
    ASSERT(static_cast<float*>(out.data)[0] == 6.0f, "add OK behavior preserved");

    // OK path: scalar broadcast (b is numel == 1)
    int64_t scalar_shape[] = {1};
    Tensor b1 = Tensor::alloc(scalar_shape, 1, DType::F32);
    static_cast<float*>(b1.data)[0] = 10.0f;
    ASSERT(add(a, b1, out).is_ok(), "add(scalar broadcast) OK");
    ASSERT(static_cast<float*>(out.data)[0] == 11.0f, "scalar broadcast behavior");
    b1.free();

    // Type mismatch
    Tensor b_i32 = Tensor::alloc(shape4, 1, DType::I32);
    fill_sentinel(out);
    ASSERT(add(a, b_i32, out).code == StatusCode::TYPE_MISMATCH,
           "add type mismatch -> TYPE_MISMATCH");
    ASSERT(is_sentinel(out), "add type mismatch leaves output untouched");
    b_i32.free();

    // Shape mismatch (a and b same, but out wrong)
    int64_t shape3[] = {3};
    Tensor out_bad = Tensor::alloc(shape3, 1, DType::F32);
    fill_sentinel(out_bad);
    ASSERT(add(a, b, out_bad).code == StatusCode::INVALID_ARGUMENT,
           "add shape mismatch -> INVALID_ARGUMENT");
    ASSERT(is_sentinel(out_bad), "add shape mismatch leaves output untouched");
    out_bad.free();

    // Null data
    Tensor null_t = Tensor::empty();
    ASSERT(add(null_t, b, out).code == StatusCode::INVALID_STATE,
           "add null input -> INVALID_STATE");

    a.free();
    b.free();
    out.free();
}

// ─────────────────────────────────────────────────────────────────────
// scalar_op
// ─────────────────────────────────────────────────────────────────────

static void test_scalar_op() {
    std::printf("\n--- scalar_op ---\n");

    int64_t shape3[] = {3};
    Tensor a   = Tensor::alloc(shape3, 1, DType::F32);
    Tensor out = Tensor::alloc(shape3, 1, DType::F32);
    for (int i = 0; i < 3; ++i) static_cast<float*>(a.data)[i] = static_cast<float>(i);

    Scalar s = Scalar(2.0f);
    ASSERT(scalar_op(a, s, out, ElementwiseOp::MUL).is_ok(), "scalar_op(MUL) OK");
    ASSERT(static_cast<float*>(out.data)[2] == 4.0f, "scalar_op MUL behavior");

    // Type mismatch
    Tensor out_i32 = Tensor::alloc(shape3, 1, DType::I32);
    fill_sentinel(out_i32);
    ASSERT(scalar_op(a, s, out_i32, ElementwiseOp::MUL).code == StatusCode::TYPE_MISMATCH,
           "scalar_op type mismatch -> TYPE_MISMATCH");
    ASSERT(is_sentinel(out_i32), "scalar_op type mismatch leaves output untouched");
    out_i32.free();

    // Null data
    Tensor null_t = Tensor::empty();
    ASSERT(scalar_op(null_t, s, out, ElementwiseOp::MUL).code == StatusCode::INVALID_STATE,
           "scalar_op null input -> INVALID_STATE");

    a.free();
    out.free();
}

// ─────────────────────────────────────────────────────────────────────
// matmul / gemm
// ─────────────────────────────────────────────────────────────────────

static void test_matmul() {
    std::printf("\n--- matmul / gemm ---\n");

    int64_t A_shape[] = {2, 3};
    int64_t B_shape[] = {3, 2};
    int64_t C_shape[] = {2, 2};
    Tensor A = Tensor::alloc(A_shape, 2, DType::F32);
    Tensor B = Tensor::alloc(B_shape, 2, DType::F32);
    Tensor C = Tensor::alloc(C_shape, 2, DType::F32);
    for (int i = 0; i < 6; ++i) {
        static_cast<float*>(A.data)[i] = static_cast<float>(i + 1);
        static_cast<float*>(B.data)[i] = static_cast<float>(i + 1);
    }

    // OK
    ASSERT(matmul(A, B, C).is_ok(), "matmul OK returns ok()");
    ASSERT(static_cast<float*>(C.data)[0] == 22.0f, "matmul OK behavior [0,0]");
    ASSERT(static_cast<float*>(C.data)[3] == 64.0f, "matmul OK behavior [1,1]");

    // gemm OK with alpha/beta
    ASSERT(gemm(A, B, C, 1.0f, 0.0f).is_ok(), "gemm OK returns ok()");

    // Type mismatch (C wrong dtype)
    Tensor C_i32 = Tensor::alloc(C_shape, 2, DType::I32);
    fill_sentinel(C_i32);
    ASSERT(matmul(A, B, C_i32).code == StatusCode::TYPE_MISMATCH,
           "matmul type mismatch -> TYPE_MISMATCH");
    ASSERT(is_sentinel(C_i32), "matmul type mismatch leaves C untouched");
    C_i32.free();

    // Shape mismatch (inner dim wrong)
    int64_t B_bad_shape[] = {4, 2};
    Tensor B_bad = Tensor::alloc(B_bad_shape, 2, DType::F32);
    fill_sentinel(C);
    ASSERT(matmul(A, B_bad, C).code == StatusCode::INVALID_ARGUMENT,
           "matmul inner-dim mismatch -> INVALID_ARGUMENT");
    ASSERT(is_sentinel(C), "matmul inner-dim mismatch leaves C untouched");
    B_bad.free();

    // Shape mismatch (rank wrong) — under INVALID_ARGUMENT per spec decision.
    int64_t vec_shape[] = {3};
    Tensor A_vec = Tensor::alloc(vec_shape, 1, DType::F32);
    fill_sentinel(C);
    ASSERT(matmul(A_vec, B, C).code == StatusCode::INVALID_ARGUMENT,
           "matmul rank mismatch -> INVALID_ARGUMENT");
    ASSERT(is_sentinel(C), "matmul rank mismatch leaves C untouched");
    A_vec.free();

    // Null data
    Tensor null_t = Tensor::empty();
    ASSERT(matmul(null_t, B, C).code == StatusCode::INVALID_STATE,
           "matmul null input -> INVALID_STATE");

    A.free();
    B.free();
    C.free();
}

// ─────────────────────────────────────────────────────────────────────
// Reduce (sum, max, mean, argmax, reduce_last_axis)
// ─────────────────────────────────────────────────────────────────────

static void test_reduce() {
    std::printf("\n--- reduce ops ---\n");

    int64_t in_shape[]  = {2, 3};
    int64_t out_shape[] = {2};
    Tensor in  = Tensor::alloc(in_shape, 2, DType::F32);
    Tensor out = Tensor::alloc(out_shape, 1, DType::F32);
    for (int i = 0; i < 6; ++i) static_cast<float*>(in.data)[i] = static_cast<float>(i + 1);

    // OK: sum along last axis → [1+2+3, 4+5+6] = [6, 15]
    ASSERT(sum(in, out).is_ok(), "sum OK returns ok()");
    ASSERT(static_cast<float*>(out.data)[0] == 6.0f,  "sum OK behavior [0]");
    ASSERT(static_cast<float*>(out.data)[1] == 15.0f, "sum OK behavior [1]");

    // OK: max along last axis
    ASSERT(max(in, out).is_ok(), "max OK");
    ASSERT(static_cast<float*>(out.data)[0] == 3.0f, "max OK behavior");

    // OK: mean
    ASSERT(mean(in, out).is_ok(), "mean OK");
    ASSERT(static_cast<float*>(out.data)[0] == 2.0f, "mean OK behavior");

    // OK: argmax (output dtype must be I32 or I64)
    Tensor amax = Tensor::alloc(out_shape, 1, DType::I64);
    ASSERT(argmax(in, amax).is_ok(), "argmax OK");
    ASSERT(static_cast<int64_t*>(amax.data)[0] == 2, "argmax OK behavior [0]");
    amax.free();

    // Type mismatch: sum with I32 output
    Tensor out_i32 = Tensor::alloc(out_shape, 1, DType::I32);
    fill_sentinel(out_i32);
    ASSERT(sum(in, out_i32).code == StatusCode::TYPE_MISMATCH,
           "sum type mismatch -> TYPE_MISMATCH");
    ASSERT(is_sentinel(out_i32), "sum type mismatch leaves output untouched");
    out_i32.free();

    // Shape mismatch: output rank != input rank - 1
    Tensor out_bad = Tensor::alloc(in_shape, 2, DType::F32);
    fill_sentinel(out_bad);
    ASSERT(sum(in, out_bad).code == StatusCode::INVALID_ARGUMENT,
           "sum rank mismatch -> INVALID_ARGUMENT");
    ASSERT(is_sentinel(out_bad), "sum shape mismatch leaves output untouched");
    out_bad.free();

    // Null data
    Tensor null_t = Tensor::empty();
    ASSERT(sum(null_t, out).code == StatusCode::INVALID_STATE,
           "sum null input -> INVALID_STATE");

    // argmax with wrong output dtype (F32 instead of int)
    Tensor amax_bad = Tensor::alloc(out_shape, 1, DType::F32);
    fill_sentinel(amax_bad);
    ASSERT(argmax(in, amax_bad).code == StatusCode::TYPE_MISMATCH,
           "argmax wrong output dtype -> TYPE_MISMATCH");
    ASSERT(is_sentinel(amax_bad), "argmax type mismatch leaves output untouched");
    amax_bad.free();

    // reduce_last_axis direct call (covered by sum/max/mean above, but exercise once)
    ASSERT(reduce_last_axis(in, out, ReduceOp::SUM).is_ok(), "reduce_last_axis(SUM) OK");

    in.free();
    out.free();
}

int main() {
    std::printf("=== Spec 002 — Status-returning compute ops ===\n");

    test_unary();
    test_binary();
    test_scalar_op();
    test_matmul();
    test_reduce();

    std::printf("\n=== Result: %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
