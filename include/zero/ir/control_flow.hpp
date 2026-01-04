#pragma once

/**
 * @file control_flow.hpp
 * @brief Zero Core Runtime — Control Flow IR Nodes
 * 
 * If/else, for, while loops that map to LLVM basic blocks.
 * No dynamic dispatch, no exceptions.
 */

#include "../core/scalar.hpp"

#include <cstdint>

namespace zero {
namespace ir {

/**
 * @brief Node types for control flow
 */
enum class ControlFlowType : uint8_t {
    IF = 0,
    FOR = 1,
    WHILE = 2,
    BLOCK = 3,  // Basic block
};

/**
 * @brief Basic block identifier
 */
struct BlockId {
    uint32_t id;
    
    constexpr BlockId() noexcept : id(0) {}
    constexpr explicit BlockId(uint32_t i) noexcept : id(i) {}
    
    constexpr bool operator==(BlockId other) const noexcept { return id == other.id; }
    constexpr bool operator!=(BlockId other) const noexcept { return id != other.id; }
};

/**
 * @brief If-else node
 * 
 * Represents a conditional branch in the execution graph.
 */
struct IfNode {
    BlockId condition_block;   ///< Block that computes condition
    BlockId then_block;        ///< Block to execute if true
    BlockId else_block;        ///< Block to execute if false (0 = no else)
    BlockId merge_block;       ///< Block after if-else
    
    constexpr IfNode() noexcept 
        : condition_block{}, then_block{}, else_block{}, merge_block{} {}
};

/**
 * @brief For loop node
 * 
 * Represents a counted loop with analyzable bounds.
 */
struct ForNode {
    BlockId init_block;        ///< Initialization block
    BlockId condition_block;   ///< Loop condition
    BlockId body_block;        ///< Loop body
    BlockId update_block;      ///< Counter update
    BlockId exit_block;        ///< Block after loop
    
    // Loop bounds (for unrolling/vectorization analysis)
    int64_t lower_bound;       ///< Known lower bound (-1 if dynamic)
    int64_t upper_bound;       ///< Known upper bound (-1 if dynamic)
    int64_t step;              ///< Loop step (1 by default)
    
    constexpr ForNode() noexcept 
        : init_block{}, condition_block{}, body_block{}, 
          update_block{}, exit_block{},
          lower_bound(-1), upper_bound(-1), step(1) {}
    
    /**
     * @brief Check if loop bounds are statically known
     */
    constexpr bool has_static_bounds() const noexcept {
        return lower_bound >= 0 && upper_bound >= 0;
    }
    
    /**
     * @brief Get trip count if known
     */
    constexpr int64_t trip_count() const noexcept {
        if (!has_static_bounds()) return -1;
        return (upper_bound - lower_bound + step - 1) / step;
    }
};

/**
 * @brief While loop node
 * 
 * Represents a condition-based loop.
 */
struct WhileNode {
    BlockId condition_block;   ///< Loop condition
    BlockId body_block;        ///< Loop body
    BlockId exit_block;        ///< Block after loop
    
    constexpr WhileNode() noexcept 
        : condition_block{}, body_block{}, exit_block{} {}
};

/**
 * @brief Branch target
 */
struct Branch {
    BlockId target;
    bool is_conditional;
    
    constexpr Branch() noexcept : target{}, is_conditional(false) {}
    constexpr explicit Branch(BlockId t, bool cond = false) noexcept 
        : target(t), is_conditional(cond) {}
};

/**
 * @brief Basic block in the control flow graph
 */
struct BasicBlock {
    BlockId id;
    uint32_t instruction_start;  ///< Start index in instruction array
    uint32_t instruction_count;  ///< Number of instructions
    Branch successors[2];        ///< At most 2 successors (conditional branch)
    int8_t num_successors;
    
    constexpr BasicBlock() noexcept 
        : id{}, instruction_start(0), instruction_count(0), 
          successors{}, num_successors(0) {}
    
    /**
     * @brief Add unconditional branch
     */
    void add_branch(BlockId target) noexcept {
        if (num_successors >= 2) return;
        successors[num_successors++] = Branch(target, false);
    }
    
    /**
     * @brief Add conditional branches (then, else)
     */
    void add_cond_branch(BlockId then_target, BlockId else_target) noexcept {
        successors[0] = Branch(then_target, true);
        successors[1] = Branch(else_target, true);
        num_successors = 2;
    }
};

} // namespace ir
} // namespace zero
