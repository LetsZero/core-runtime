#pragma once

/**
 * @file function.hpp
 * @brief Zero Core Runtime — Function Representation
 * 
 * Pure functions as nodes in the execution graph.
 * Explicit inputs, explicit outputs, no hidden state.
 */

#include "../core/tensor.hpp"
#include "../core/scalar.hpp"

#include <array>
#include <cstdint>

namespace zero {
namespace ir {

/// Maximum number of function inputs/outputs
constexpr int8_t MAX_FUNC_ARGS = 16;

/**
 * @brief Function argument descriptor
 */
struct ArgDesc {
    const char* name;    ///< Argument name
    bool is_tensor;      ///< True for tensor, false for scalar
    DType dtype;         ///< Data type
    bool is_output;      ///< True if this is an output argument
    
    constexpr ArgDesc() noexcept 
        : name(nullptr), is_tensor(true), dtype(DType::F32), is_output(false) {}
    
    constexpr ArgDesc(const char* n, bool tensor, DType dt, bool out = false) noexcept
        : name(n), is_tensor(tensor), dtype(dt), is_output(out) {}
};

/**
 * @brief Function signature
 * 
 * Describes the interface of a Zero function.
 * Used by the compiler for type checking and optimization.
 */
struct FunctionSig {
    const char* name;
    std::array<ArgDesc, MAX_FUNC_ARGS> args;
    int8_t num_inputs;
    int8_t num_outputs;
    bool is_pure;         ///< Pure functions have no side effects
    
    constexpr FunctionSig() noexcept 
        : name(nullptr), args{}, num_inputs(0), num_outputs(0), is_pure(true) {}
    
    constexpr FunctionSig(const char* n) noexcept
        : name(n), args{}, num_inputs(0), num_outputs(0), is_pure(true) {}
    
    /**
     * @brief Add input argument
     */
    void add_input(const char* arg_name, bool is_tensor, DType dtype) noexcept {
        if (num_inputs + num_outputs >= MAX_FUNC_ARGS) return;
        args[num_inputs] = ArgDesc(arg_name, is_tensor, dtype, false);
        num_inputs++;
    }
    
    /**
     * @brief Add output argument
     */
    void add_output(const char* arg_name, bool is_tensor, DType dtype) noexcept {
        if (num_inputs + num_outputs >= MAX_FUNC_ARGS) return;
        int8_t idx = num_inputs + num_outputs;
        args[idx] = ArgDesc(arg_name, is_tensor, dtype, true);
        num_outputs++;
    }
    
    /**
     * @brief Get total number of arguments
     */
    constexpr int8_t total_args() const noexcept {
        return num_inputs + num_outputs;
    }
};

/**
 * @brief Function call context
 * 
 * Holds the actual tensor/scalar values for a function call.
 */
struct FunctionCall {
    const FunctionSig* signature;
    std::array<void*, MAX_FUNC_ARGS> arg_ptrs;  ///< Pointers to Tensor or Scalar
    
    FunctionCall() noexcept : signature(nullptr), arg_ptrs{} {}
    
    explicit FunctionCall(const FunctionSig* sig) noexcept 
        : signature(sig), arg_ptrs{} {}
    
    /**
     * @brief Set tensor argument
     */
    void set_tensor(int8_t idx, Tensor* tensor) noexcept {
        if (idx < 0 || idx >= MAX_FUNC_ARGS) return;
        arg_ptrs[idx] = tensor;
    }
    
    /**
     * @brief Set scalar argument
     */
    void set_scalar(int8_t idx, Scalar* scalar) noexcept {
        if (idx < 0 || idx >= MAX_FUNC_ARGS) return;
        arg_ptrs[idx] = scalar;
    }
    
    /**
     * @brief Get tensor argument
     */
    Tensor* get_tensor(int8_t idx) const noexcept {
        if (idx < 0 || idx >= MAX_FUNC_ARGS) return nullptr;
        return static_cast<Tensor*>(arg_ptrs[idx]);
    }
    
    /**
     * @brief Get scalar argument
     */
    Scalar* get_scalar(int8_t idx) const noexcept {
        if (idx < 0 || idx >= MAX_FUNC_ARGS) return nullptr;
        return static_cast<Scalar*>(arg_ptrs[idx]);
    }
};

/**
 * @brief Function pointer type for compiled functions
 */
using CompiledFn = void (*)(FunctionCall*);

/**
 * @brief Compiled function handle
 */
struct Function {
    FunctionSig signature;
    CompiledFn entry_point;
    
    Function() noexcept : signature{}, entry_point(nullptr) {}
    
    /**
     * @brief Invoke the function
     */
    void operator()(FunctionCall* call) const noexcept {
        if (entry_point != nullptr) {
            entry_point(call);
        }
    }
};

} // namespace ir
} // namespace zero
