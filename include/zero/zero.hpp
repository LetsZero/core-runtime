#pragma once

/**
 * @file zero.hpp
 * @brief Zero Core Runtime — Main Include Header
 * 
 * Single header to include the entire Core Runtime.
 */

// Core primitives
#include "core/dtype.hpp"
#include "core/memory.hpp"
#include "core/tensor.hpp"
#include "core/scalar.hpp"
#include "core/struct.hpp"

// Operations
#include "ops/elementwise.hpp"
#include "ops/matmul.hpp"
#include "ops/reduce.hpp"
#include "ops/reshape.hpp"

// IR primitives
#include "ir/function.hpp"
#include "ir/control_flow.hpp"
#include "ir/op_kind.hpp"

// Device model
#include "device/device.hpp"
#include "device/sync.hpp"

/**
 * @namespace zero
 * @brief Zero Core Runtime namespace
 * 
 * The Core Runtime is the final boundary of the Zero compiler.
 * Everything is lowered to these 7 primitives:
 * 
 * 1. Tensor  - The only data container
 * 2. Scalar  - Rank-0 tensor or immediate value
 * 3. Struct  - Static aggregation of tensors/scalars
 * 4. Control Flow - if/else, for, while
 * 5. Functions - Pure by default
 * 6. Memory Model - Explicit allocation/deallocation
 * 7. Core Ops - Elementwise, MatMul, Reduce, etc.
 * 
 * @see docs/CORE_RUNTIME_SPEC.md
 */
