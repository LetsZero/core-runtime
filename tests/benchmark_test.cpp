/**
 * @file benchmark_test.cpp
 * @brief Comprehensive correctness and performance tests for Zero Core Runtime
 */

#include <zero/zero.hpp>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>

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

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

void fill_random(Tensor& t, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* data = static_cast<float*>(t.data);
    for (int64_t i = 0; i < t.numel(); ++i) {
        data[i] = dist(rng);
    }
}

void fill_sequential(Tensor& t) {
    float* data = static_cast<float*>(t.data);
    for (int64_t i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness Tests
// ─────────────────────────────────────────────────────────────────────────────

void test_matmul_correctness() {
    printf("\n=== MatMul Correctness ===\n");
    
    // Test 1: Identity-like multiplication
    // [1 0]   [a b]   [a b]
    // [0 1] x [c d] = [c d]
    {
        int64_t shape[] = {2, 2};
        Tensor I = Tensor::alloc(shape, 2, DType::F32);
        Tensor A = Tensor::alloc(shape, 2, DType::F32);
        Tensor C = Tensor::alloc(shape, 2, DType::F32);
        
        float* i_data = static_cast<float*>(I.data);
        i_data[0] = 1; i_data[1] = 0;
        i_data[2] = 0; i_data[3] = 1;
        
        float* a_data = static_cast<float*>(A.data);
        a_data[0] = 5; a_data[1] = 6;
        a_data[2] = 7; a_data[3] = 8;
        
        matmul(I, A, C);
        float* c_data = static_cast<float*>(C.data);
        
        ASSERT_NEAR(c_data[0], 5.0f, 1e-5f, "Identity matmul [0,0]");
        ASSERT_NEAR(c_data[1], 6.0f, 1e-5f, "Identity matmul [0,1]");
        ASSERT_NEAR(c_data[2], 7.0f, 1e-5f, "Identity matmul [1,0]");
        ASSERT_NEAR(c_data[3], 8.0f, 1e-5f, "Identity matmul [1,1]");
        
        I.free(); A.free(); C.free();
    }
    
    // Test 2: Known calculation
    // [1 2 3]   [1 4]   [1*1+2*2+3*3, 1*4+2*5+3*6]   [14 32]
    // [4 5 6] x [2 5] = [4*1+5*2+6*3, 4*4+5*5+6*6] = [32 77]
    //           [3 6]
    {
        int64_t a_shape[] = {2, 3};
        int64_t b_shape[] = {3, 2};
        int64_t c_shape[] = {2, 2};
        
        Tensor A = Tensor::alloc(a_shape, 2, DType::F32);
        Tensor B = Tensor::alloc(b_shape, 2, DType::F32);
        Tensor C = Tensor::alloc(c_shape, 2, DType::F32);
        
        float* a = static_cast<float*>(A.data);
        a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6;
        
        float* b = static_cast<float*>(B.data);
        b[0]=1; b[1]=4; b[2]=2; b[3]=5; b[4]=3; b[5]=6;
        
        matmul(A, B, C);
        float* c = static_cast<float*>(C.data);
        
        ASSERT_NEAR(c[0], 14.0f, 1e-5f, "MatMul 2x3 @ 3x2 -> [0,0]=14");
        ASSERT_NEAR(c[1], 32.0f, 1e-5f, "MatMul 2x3 @ 3x2 -> [0,1]=32");
        ASSERT_NEAR(c[2], 32.0f, 1e-5f, "MatMul 2x3 @ 3x2 -> [1,0]=32");
        ASSERT_NEAR(c[3], 77.0f, 1e-5f, "MatMul 2x3 @ 3x2 -> [1,1]=77");
        
        A.free(); B.free(); C.free();
    }
}

void test_elementwise_correctness() {
    printf("\n=== Elementwise Correctness ===\n");
    
    int64_t shape[] = {4};
    Tensor a = Tensor::alloc(shape, 1, DType::F32);
    Tensor b = Tensor::alloc(shape, 1, DType::F32);
    Tensor c = Tensor::alloc(shape, 1, DType::F32);
    
    float* a_data = static_cast<float*>(a.data);
    float* b_data = static_cast<float*>(b.data);
    a_data[0] = 1; a_data[1] = 2; a_data[2] = 3; a_data[3] = 4;
    b_data[0] = 2; b_data[1] = 2; b_data[2] = 2; b_data[3] = 2;
    
    // Add
    add(a, b, c);
    float* c_data = static_cast<float*>(c.data);
    ASSERT_NEAR(c_data[0], 3.0f, 1e-5f, "Add: 1+2=3");
    ASSERT_NEAR(c_data[3], 6.0f, 1e-5f, "Add: 4+2=6");
    
    // Sub
    sub(a, b, c);
    ASSERT_NEAR(c_data[0], -1.0f, 1e-5f, "Sub: 1-2=-1");
    ASSERT_NEAR(c_data[3], 2.0f, 1e-5f, "Sub: 4-2=2");
    
    // Mul
    mul(a, b, c);
    ASSERT_NEAR(c_data[0], 2.0f, 1e-5f, "Mul: 1*2=2");
    ASSERT_NEAR(c_data[3], 8.0f, 1e-5f, "Mul: 4*2=8");
    
    // Div
    div(a, b, c);
    ASSERT_NEAR(c_data[0], 0.5f, 1e-5f, "Div: 1/2=0.5");
    ASSERT_NEAR(c_data[3], 2.0f, 1e-5f, "Div: 4/2=2");
    
    // Exp
    a_data[0] = 0; a_data[1] = 1;
    exp(a, c);
    ASSERT_NEAR(c_data[0], 1.0f, 1e-5f, "Exp: e^0=1");
    ASSERT_NEAR(c_data[1], 2.71828f, 1e-4f, "Exp: e^1≈2.718");
    
    // Log
    a_data[0] = 1; a_data[1] = 2.71828f;
    log(a, c);
    ASSERT_NEAR(c_data[0], 0.0f, 1e-5f, "Log: ln(1)=0");
    ASSERT_NEAR(c_data[1], 1.0f, 1e-4f, "Log: ln(e)≈1");
    
    // Sqrt
    a_data[0] = 4; a_data[1] = 9;
    sqrt(a, c);
    ASSERT_NEAR(c_data[0], 2.0f, 1e-5f, "Sqrt: √4=2");
    ASSERT_NEAR(c_data[1], 3.0f, 1e-5f, "Sqrt: √9=3");
    
    a.free(); b.free(); c.free();
}

void test_reduce_correctness() {
    printf("\n=== Reduce Correctness ===\n");
    
    int64_t shape[] = {2, 3};
    Tensor a = Tensor::alloc(shape, 2, DType::F32);
    fill_sequential(a);  // [0,1,2,3,4,5]
    
    ASSERT_NEAR(sum_all(a), 15.0f, 1e-5f, "Sum: 0+1+2+3+4+5=15");
    ASSERT_NEAR(mean_all(a), 2.5f, 1e-5f, "Mean: 15/6=2.5");
    ASSERT_NEAR(max_all(a), 5.0f, 1e-5f, "Max: 5");
    ASSERT_NEAR(min_all(a), 0.0f, 1e-5f, "Min: 0");
    
    // Axis reduction
    int64_t out_shape[] = {2};
    Tensor out = Tensor::alloc(out_shape, 1, DType::F32);
    sum(a, out);
    float* out_data = static_cast<float*>(out.data);
    ASSERT_NEAR(out_data[0], 3.0f, 1e-5f, "Sum axis=-1 [0]: 0+1+2=3");
    ASSERT_NEAR(out_data[1], 12.0f, 1e-5f, "Sum axis=-1 [1]: 3+4+5=12");
    
    a.free(); out.free();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Tests
// ─────────────────────────────────────────────────────────────────────────────

void benchmark_matmul() {
    printf("\n=== MatMul Benchmark ===\n");
    
    std::mt19937 rng(42);
    Timer timer;
    
    // Various sizes
    int sizes[] = {64, 128, 256, 512};
    
    for (int n : sizes) {
        int64_t shape[] = {n, n};
        Tensor A = Tensor::alloc(shape, 2, DType::F32);
        Tensor B = Tensor::alloc(shape, 2, DType::F32);
        Tensor C = Tensor::alloc(shape, 2, DType::F32);
        
        fill_random(A, rng);
        fill_random(B, rng);
        
        // Warmup
        matmul(A, B, C);
        
        // Benchmark
        int iterations = (n <= 128) ? 10 : 3;
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            matmul(A, B, C);
        }
        double ms = timer.elapsed_ms() / iterations;
        
        // GFLOPS = 2*N^3 / (time_sec * 1e9)
        double gflops = (2.0 * n * n * n) / (ms * 1e6);
        
        printf("MatMul %dx%d: %.2f ms (%.2f GFLOPS)\n", n, n, ms, gflops);
        
        A.free(); B.free(); C.free();
    }
}

void benchmark_elementwise() {
    printf("\n=== Elementwise Benchmark ===\n");
    
    std::mt19937 rng(42);
    Timer timer;
    
    int64_t shape[] = {1024, 1024};  // 1M elements
    Tensor a = Tensor::alloc(shape, 2, DType::F32);
    Tensor b = Tensor::alloc(shape, 2, DType::F32);
    Tensor c = Tensor::alloc(shape, 2, DType::F32);
    
    fill_random(a, rng);
    fill_random(b, rng);
    
    int iterations = 100;
    
    // Add
    timer.start();
    for (int i = 0; i < iterations; ++i) add(a, b, c);
    printf("Add 1M elements: %.2f ms\n", timer.elapsed_ms() / iterations);
    
    // Mul
    timer.start();
    for (int i = 0; i < iterations; ++i) mul(a, b, c);
    printf("Mul 1M elements: %.2f ms\n", timer.elapsed_ms() / iterations);
    
    // Exp
    timer.start();
    for (int i = 0; i < iterations; ++i) exp(a, c);
    printf("Exp 1M elements: %.2f ms\n", timer.elapsed_ms() / iterations);
    
    a.free(); b.free(); c.free();
}

void benchmark_reduce() {
    printf("\n=== Reduce Benchmark ===\n");
    
    std::mt19937 rng(42);
    Timer timer;
    
    int64_t shape[] = {1024, 1024};  // 1M elements
    Tensor a = Tensor::alloc(shape, 2, DType::F32);
    fill_random(a, rng);
    
    int iterations = 100;
    
    timer.start();
    float result = 0;
    for (int i = 0; i < iterations; ++i) result = sum_all(a);
    printf("Sum 1M elements: %.2f ms (result=%.2f)\n", 
           timer.elapsed_ms() / iterations, result);
    
    timer.start();
    for (int i = 0; i < iterations; ++i) result = max_all(a);
    printf("Max 1M elements: %.2f ms (result=%.2f)\n", 
           timer.elapsed_ms() / iterations, result);
    
    a.free();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Zero Core Runtime — Test & Benchmark Suite         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    // Correctness tests
    test_matmul_correctness();
    test_elementwise_correctness();
    test_reduce_correctness();
    
    // Benchmarks
    benchmark_matmul();
    benchmark_elementwise();
    benchmark_reduce();
    
    // Summary
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("TOTAL: %d passed, %d failed\n", passes, failures);
    printf("══════════════════════════════════════════════════════════════\n");
    
    return failures > 0 ? 1 : 0;
}
