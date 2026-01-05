#pragma once

/**
 * @file op_kind.hpp
 * @brief Zero Core Runtime — Operation Kind Enumeration
 * 
 * Minimal semantic identity for IR operations.
 * No execution logic, no scheduling — just naming.
 * 
 * v1.1: Added activation ops (RELU, SIGMOID, TANH)
 */

#include <cstdint>

namespace zero {
namespace ir {

/**
 * @brief Enumeration of all core operation kinds
 * 
 * This enum provides semantic identity for IR nodes.
 * It does NOT define execution order or behavior.
 */
enum class OpKind : uint8_t {
    // Binary arithmetic
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    
    // Unary arithmetic
    NEG = 4,
    ABS = 5,
    
    // Transcendental
    EXP = 6,
    LOG = 7,
    SQRT = 8,
    SIN = 9,
    COS = 10,
    
    // Activations (v1.1)
    TANH = 11,
    RELU = 12,
    SIGMOID = 13,
    
    // Matrix operations
    MATMUL = 20,
    MATVEC = 21,
    
    // Reductions
    SUM = 30,
    MEAN = 31,
    MAX = 32,
    MIN = 33,
    
    // Memory operations
    LOAD = 40,
    STORE = 41,
    ALLOC = 42,
    FREE = 43,
    
    // Control flow
    BRANCH = 50,
    CALL = 51,
    RETURN = 52,
};

/**
 * @brief Get human-readable name for an OpKind
 */
constexpr const char* op_kind_name(OpKind kind) noexcept {
    switch (kind) {
        case OpKind::ADD:     return "add";
        case OpKind::SUB:     return "sub";
        case OpKind::MUL:     return "mul";
        case OpKind::DIV:     return "div";
        case OpKind::NEG:     return "neg";
        case OpKind::ABS:     return "abs";
        case OpKind::EXP:     return "exp";
        case OpKind::LOG:     return "log";
        case OpKind::SQRT:    return "sqrt";
        case OpKind::SIN:     return "sin";
        case OpKind::COS:     return "cos";
        case OpKind::TANH:    return "tanh";
        case OpKind::RELU:    return "relu";
        case OpKind::SIGMOID: return "sigmoid";
        case OpKind::MATMUL:  return "matmul";
        case OpKind::MATVEC:  return "matvec";
        case OpKind::SUM:     return "sum";
        case OpKind::MEAN:    return "mean";
        case OpKind::MAX:     return "max";
        case OpKind::MIN:     return "min";
        case OpKind::LOAD:    return "load";
        case OpKind::STORE:   return "store";
        case OpKind::ALLOC:   return "alloc";
        case OpKind::FREE:    return "free";
        case OpKind::BRANCH:  return "branch";
        case OpKind::CALL:    return "call";
        case OpKind::RETURN:  return "return";
    }
    return "unknown";
}

/**
 * @brief Check if an OpKind is an activation function
 */
constexpr bool is_activation(OpKind kind) noexcept {
    return kind == OpKind::RELU || 
           kind == OpKind::SIGMOID || 
           kind == OpKind::TANH;
}

/**
 * @brief Check if an OpKind is a unary operation
 */
constexpr bool is_unary(OpKind kind) noexcept {
    return kind == OpKind::NEG || kind == OpKind::ABS ||
           kind == OpKind::EXP || kind == OpKind::LOG ||
           kind == OpKind::SQRT || kind == OpKind::SIN ||
           kind == OpKind::COS || is_activation(kind);
}

} // namespace ir
} // namespace zero
