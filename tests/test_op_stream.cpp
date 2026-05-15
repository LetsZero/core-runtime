/**
 * @file test_op_stream.cpp
 * @brief Acceptance tests for spec 003 — Stream* parameter on compute ops.
 *
 * Tests derived from docs/specs/003-stream-parameter.md §4.
 *
 * The Stream parameter is ignored by CPU but committed in every signature.
 * Tests verify:
 *   1. Default-null call works.
 *   2. Explicit nullptr matches default-null bit-for-bit.
 *   3. Non-null CPU stream matches null-stream bit-for-bit.
 *   4. Stream sync + destroy after an op does not segfault.
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

// True if two F32 tensors are byte-identical (same shape, same bytes).
static bool tensors_byte_equal(const Tensor& a, const Tensor& b) noexcept {
    if (a.dtype != b.dtype || a.ndim != b.ndim) return false;
    if (a.nbytes() != b.nbytes()) return false;
    return std::memcmp(a.data, b.data, a.nbytes()) == 0;
}

int main() {
    std::printf("=== Spec 003 — Stream* parameter on compute ops ===\n\n");

    Stream cpu_stream = Stream::create(Device::CPU);

    // ─────────────────────────────────────────────────────────────────
    // relu — unary representative
    // ─────────────────────────────────────────────────────────────────
    {
        std::printf("--- relu ---\n");
        int64_t shape[] = {4};
        Tensor a       = Tensor::alloc(shape, 1, DType::F32);
        Tensor o_def   = Tensor::alloc(shape, 1, DType::F32);
        Tensor o_null  = Tensor::alloc(shape, 1, DType::F32);
        Tensor o_strm  = Tensor::alloc(shape, 1, DType::F32);
        float* ad = static_cast<float*>(a.data);
        ad[0] = -1.0f; ad[1] = 0.0f; ad[2] = 0.5f; ad[3] = 2.0f;

        ASSERT(relu(a, o_def).is_ok(),                  "relu default-null ok");
        ASSERT(relu(a, o_null, nullptr).is_ok(),        "relu explicit-null ok");
        ASSERT(relu(a, o_strm, &cpu_stream).is_ok(),    "relu CPU-stream ok");
        ASSERT(tensors_byte_equal(o_def, o_null),       "relu default == explicit-null");
        ASSERT(tensors_byte_equal(o_def, o_strm),       "relu default == CPU-stream");

        a.free(); o_def.free(); o_null.free(); o_strm.free();
    }

    // ─────────────────────────────────────────────────────────────────
    // add — binary representative
    // ─────────────────────────────────────────────────────────────────
    {
        std::printf("\n--- add ---\n");
        int64_t shape[] = {4};
        Tensor a       = Tensor::alloc(shape, 1, DType::F32);
        Tensor b       = Tensor::alloc(shape, 1, DType::F32);
        Tensor o_def   = Tensor::alloc(shape, 1, DType::F32);
        Tensor o_null  = Tensor::alloc(shape, 1, DType::F32);
        Tensor o_strm  = Tensor::alloc(shape, 1, DType::F32);
        for (int i = 0; i < 4; ++i) {
            static_cast<float*>(a.data)[i] = static_cast<float>(i + 1);
            static_cast<float*>(b.data)[i] = static_cast<float>(i + 5);
        }

        ASSERT(add(a, b, o_def).is_ok(),                "add default-null ok");
        ASSERT(add(a, b, o_null, nullptr).is_ok(),      "add explicit-null ok");
        ASSERT(add(a, b, o_strm, &cpu_stream).is_ok(),  "add CPU-stream ok");
        ASSERT(tensors_byte_equal(o_def, o_null),       "add default == explicit-null");
        ASSERT(tensors_byte_equal(o_def, o_strm),       "add default == CPU-stream");

        a.free(); b.free(); o_def.free(); o_null.free(); o_strm.free();
    }

    // ─────────────────────────────────────────────────────────────────
    // scalar_op — scalar representative
    // ─────────────────────────────────────────────────────────────────
    {
        std::printf("\n--- scalar_op ---\n");
        int64_t shape[] = {3};
        Tensor a       = Tensor::alloc(shape, 1, DType::F32);
        Tensor o_def   = Tensor::alloc(shape, 1, DType::F32);
        Tensor o_strm  = Tensor::alloc(shape, 1, DType::F32);
        for (int i = 0; i < 3; ++i) static_cast<float*>(a.data)[i] = static_cast<float>(i + 1);
        Scalar s = Scalar(3.0f);

        ASSERT(scalar_op(a, s, o_def, ElementwiseOp::MUL).is_ok(),
               "scalar_op default-null ok");
        ASSERT(scalar_op(a, s, o_strm, ElementwiseOp::MUL, &cpu_stream).is_ok(),
               "scalar_op CPU-stream ok");
        ASSERT(tensors_byte_equal(o_def, o_strm), "scalar_op default == CPU-stream");

        a.free(); o_def.free(); o_strm.free();
    }

    // ─────────────────────────────────────────────────────────────────
    // matmul / gemm
    // ─────────────────────────────────────────────────────────────────
    {
        std::printf("\n--- matmul / gemm ---\n");
        int64_t A_shape[] = {2, 3};
        int64_t B_shape[] = {3, 2};
        int64_t C_shape[] = {2, 2};
        Tensor A      = Tensor::alloc(A_shape, 2, DType::F32);
        Tensor B      = Tensor::alloc(B_shape, 2, DType::F32);
        Tensor C_def  = Tensor::alloc(C_shape, 2, DType::F32);
        Tensor C_strm = Tensor::alloc(C_shape, 2, DType::F32);
        for (int i = 0; i < 6; ++i) {
            static_cast<float*>(A.data)[i] = static_cast<float>(i + 1);
            static_cast<float*>(B.data)[i] = static_cast<float>(i + 1);
        }

        ASSERT(matmul(A, B, C_def).is_ok(),                 "matmul default-null ok");
        ASSERT(matmul(A, B, C_strm, &cpu_stream).is_ok(),   "matmul CPU-stream ok");
        ASSERT(tensors_byte_equal(C_def, C_strm),           "matmul default == CPU-stream");

        // gemm with the extra alpha/beta args also takes Stream as last
        Tensor C_gemm = Tensor::alloc(C_shape, 2, DType::F32);
        ASSERT(gemm(A, B, C_gemm, 1.0f, 0.0f, &cpu_stream).is_ok(),
               "gemm with alpha/beta + CPU-stream ok");
        ASSERT(tensors_byte_equal(C_def, C_gemm), "gemm default == CPU-stream");

        A.free(); B.free(); C_def.free(); C_strm.free(); C_gemm.free();
    }

    // ─────────────────────────────────────────────────────────────────
    // sum + argmax — reduce representatives
    // ─────────────────────────────────────────────────────────────────
    {
        std::printf("\n--- reduce: sum + argmax ---\n");
        int64_t in_shape[]  = {2, 3};
        int64_t out_shape[] = {2};
        Tensor in       = Tensor::alloc(in_shape, 2, DType::F32);
        Tensor s_def    = Tensor::alloc(out_shape, 1, DType::F32);
        Tensor s_strm   = Tensor::alloc(out_shape, 1, DType::F32);
        Tensor a_def    = Tensor::alloc(out_shape, 1, DType::I64);
        Tensor a_strm   = Tensor::alloc(out_shape, 1, DType::I64);
        for (int i = 0; i < 6; ++i) static_cast<float*>(in.data)[i] = static_cast<float>(i + 1);

        ASSERT(sum(in, s_def).is_ok(),                  "sum default-null ok");
        ASSERT(sum(in, s_strm, &cpu_stream).is_ok(),    "sum CPU-stream ok");
        ASSERT(tensors_byte_equal(s_def, s_strm),       "sum default == CPU-stream");

        ASSERT(argmax(in, a_def).is_ok(),               "argmax default-null ok");
        ASSERT(argmax(in, a_strm, &cpu_stream).is_ok(), "argmax CPU-stream ok");
        ASSERT(tensors_byte_equal(a_def, a_strm),       "argmax default == CPU-stream");

        in.free(); s_def.free(); s_strm.free(); a_def.free(); a_strm.free();
    }

    // ─────────────────────────────────────────────────────────────────
    // Stream sync + destroy after an op — no segfault
    // ─────────────────────────────────────────────────────────────────
    {
        std::printf("\n--- stream lifecycle ---\n");
        Stream s = Stream::create(Device::CPU);
        int64_t shape[] = {3};
        Tensor a   = Tensor::alloc(shape, 1, DType::F32);
        Tensor out = Tensor::alloc(shape, 1, DType::F32);
        static_cast<float*>(a.data)[0] = -1.0f;
        static_cast<float*>(a.data)[1] =  0.0f;
        static_cast<float*>(a.data)[2] =  1.0f;

        ASSERT(relu(a, out, &s).is_ok(), "relu with managed stream ok");
        s.sync();
        s.destroy();
        ASSERT(static_cast<float*>(out.data)[2] == 1.0f, "output preserved after stream destroy");

        a.free(); out.free();
    }

    cpu_stream.destroy();

    std::printf("\n=== Result: %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
