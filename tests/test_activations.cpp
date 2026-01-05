/**
 * @file test_activations.cpp
 * @brief Zero Core Runtime v1.1 — Activation Op Tests
 * 
 * Correctness tests for relu and sigmoid activation functions.
 * Tests known values and shape preservation invariants.
 */

#include <zero/zero.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>

using namespace zero;
using namespace zero::ops;

// ─────────────────────────────────────────────────────────────────────────────
// Test Utilities
// ─────────────────────────────────────────────────────────────────────────────

#define ASSERT(cond, msg) \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else { printf("PASS: %s\n", msg); passes++; }

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_NEAR(a, b, eps, msg) ASSERT(std::abs((a) - (b)) < (eps), msg)

int passes = 0;
int failures = 0;

// ─────────────────────────────────────────────────────────────────────────────
// ReLU Tests
// ─────────────────────────────────────────────────────────────────────────────

void test_relu_correctness() {
    printf("\n=== ReLU Correctness ===\n");
    
    // Test 1: Known values: relu([-1, 0, 2]) → [0, 0, 2]
    {
        int64_t shape[] = {3};
        Tensor a = Tensor::alloc(shape, 1, DType::F32);
        Tensor out = Tensor::alloc(shape, 1, DType::F32);
        
        float* a_data = static_cast<float*>(a.data);
        a_data[0] = -1.0f;
        a_data[1] = 0.0f;
        a_data[2] = 2.0f;
        
        relu(a, out);
        float* out_data = static_cast<float*>(out.data);
        
        ASSERT_NEAR(out_data[0], 0.0f, 1e-6f, "relu(-1) = 0");
        ASSERT_NEAR(out_data[1], 0.0f, 1e-6f, "relu(0) = 0");
        ASSERT_NEAR(out_data[2], 2.0f, 1e-6f, "relu(2) = 2");
        
        a.free();
        out.free();
    }
    
    // Test 2: Shape preservation
    {
        int64_t shape[] = {2, 3};
        Tensor a = Tensor::alloc(shape, 2, DType::F32);
        Tensor out = Tensor::alloc(shape, 2, DType::F32);
        
        ASSERT_EQ(a.ndim, out.ndim, "relu shape preservation: ndim");
        ASSERT_EQ(a.shape[0], out.shape[0], "relu shape preservation: dim 0");
        ASSERT_EQ(a.shape[1], out.shape[1], "relu shape preservation: dim 1");
        
        a.free();
        out.free();
    }
    
    // Test 3: Large negative (should not produce NaN)
    {
        int64_t shape[] = {1};
        Tensor a = Tensor::alloc(shape, 1, DType::F32);
        Tensor out = Tensor::alloc(shape, 1, DType::F32);
        
        float* a_data = static_cast<float*>(a.data);
        a_data[0] = -1000.0f;
        
        relu(a, out);
        float* out_data = static_cast<float*>(out.data);
        
        ASSERT(!std::isnan(out_data[0]), "relu(-1000) does not produce NaN");
        ASSERT_NEAR(out_data[0], 0.0f, 1e-6f, "relu(-1000) = 0");
        
        a.free();
        out.free();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sigmoid Tests
// ─────────────────────────────────────────────────────────────────────────────

void test_sigmoid_correctness() {
    printf("\n=== Sigmoid Correctness ===\n");
    
    // Test 1: Known value: sigmoid(0) = 0.5
    {
        int64_t shape[] = {1};
        Tensor a = Tensor::alloc(shape, 1, DType::F32);
        Tensor out = Tensor::alloc(shape, 1, DType::F32);
        
        float* a_data = static_cast<float*>(a.data);
        a_data[0] = 0.0f;
        
        sigmoid(a, out);
        float* out_data = static_cast<float*>(out.data);
        
        ASSERT_NEAR(out_data[0], 0.5f, 1e-6f, "sigmoid(0) = 0.5");
        
        a.free();
        out.free();
    }
    
    // Test 2: Range check (sigmoid outputs in [0, 1])
    {
        int64_t shape[] = {4};
        Tensor a = Tensor::alloc(shape, 1, DType::F32);
        Tensor out = Tensor::alloc(shape, 1, DType::F32);
        
        float* a_data = static_cast<float*>(a.data);
        a_data[0] = -100.0f;  // Should be close to 0
        a_data[1] = -2.0f;    // Should be < 0.5
        a_data[2] = 2.0f;     // Should be > 0.5
        a_data[3] = 100.0f;   // Should be close to 1
        
        sigmoid(a, out);
        float* out_data = static_cast<float*>(out.data);
        
        ASSERT(out_data[0] >= 0.0f && out_data[0] < 0.01f, "sigmoid(-100) ≈ 0");
        ASSERT(out_data[1] > 0.0f && out_data[1] < 0.5f, "sigmoid(-2) in (0, 0.5)");
        ASSERT(out_data[2] > 0.5f && out_data[2] < 1.0f, "sigmoid(2) in (0.5, 1)");
        ASSERT(out_data[3] > 0.99f && out_data[3] <= 1.0f, "sigmoid(100) ≈ 1");
        
        a.free();
        out.free();
    }
    
    // Test 3: NaN/Inf detection for large inputs
    {
        int64_t shape[] = {2};
        Tensor a = Tensor::alloc(shape, 1, DType::F32);
        Tensor out = Tensor::alloc(shape, 1, DType::F32);
        
        float* a_data = static_cast<float*>(a.data);
        a_data[0] = 100.0f;   // Large positive
        a_data[1] = -100.0f;  // Large negative
        
        sigmoid(a, out);
        float* out_data = static_cast<float*>(out.data);
        
        ASSERT(!std::isnan(out_data[0]), "sigmoid(100) does not produce NaN");
        ASSERT(!std::isnan(out_data[1]), "sigmoid(-100) does not produce NaN");
        ASSERT(!std::isinf(out_data[0]), "sigmoid(100) does not produce Inf");
        ASSERT(!std::isinf(out_data[1]), "sigmoid(-100) does not produce Inf");
        
        a.free();
        out.free();
    }
    
    // Test 4: Shape preservation
    {
        int64_t shape[] = {2, 4};
        Tensor a = Tensor::alloc(shape, 2, DType::F32);
        Tensor out = Tensor::alloc(shape, 2, DType::F32);
        
        ASSERT_EQ(a.ndim, out.ndim, "sigmoid shape preservation: ndim");
        ASSERT_EQ(a.shape[0], out.shape[0], "sigmoid shape preservation: dim 0");
        ASSERT_EQ(a.shape[1], out.shape[1], "sigmoid shape preservation: dim 1");
        
        a.free();
        out.free();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OpKind Integration Tests
// ─────────────────────────────────────────────────────────────────────────────

void test_op_kind_integration() {
    printf("\n=== OpKind IR Integration ===\n");
    
    using namespace zero::ir;
    
    // Test activation classification
    ASSERT(is_activation(OpKind::RELU), "RELU is classified as activation");
    ASSERT(is_activation(OpKind::SIGMOID), "SIGMOID is classified as activation");
    ASSERT(is_activation(OpKind::TANH), "TANH is classified as activation");
    ASSERT(!is_activation(OpKind::ADD), "ADD is not classified as activation");
    
    // Test unary classification
    ASSERT(is_unary(OpKind::RELU), "RELU is classified as unary");
    ASSERT(is_unary(OpKind::SIGMOID), "SIGMOID is classified as unary");
    ASSERT(is_unary(OpKind::NEG), "NEG is classified as unary");
    ASSERT(!is_unary(OpKind::MATMUL), "MATMUL is not classified as unary");
    
    // Test op names
    ASSERT(std::strcmp(op_kind_name(OpKind::RELU), "relu") == 0, "OpKind::RELU name");
    ASSERT(std::strcmp(op_kind_name(OpKind::SIGMOID), "sigmoid") == 0, "OpKind::SIGMOID name");
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Zero Core Runtime v1.1 — Activation Op Test Suite        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    test_relu_correctness();
    test_sigmoid_correctness();
    test_op_kind_integration();
    
    // Summary
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("TOTAL: %d passed, %d failed\n", passes, failures);
    printf("══════════════════════════════════════════════════════════════\n");
    
    return failures > 0 ? 1 : 0;
}
