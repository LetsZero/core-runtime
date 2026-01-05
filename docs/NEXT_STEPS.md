# Zero — Next Steps & Architecture Guide

> This document explains how to extend Zero beyond the frozen Core Runtime.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ZERO COMPILER (Frontend)                   │
│  ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐  │
│  │  Lexer   │ →  │ Parser  │ →  │   AST    │ →  │ Semantic  │  │
│  └──────────┘    └─────────┘    └──────────┘    │  Analysis │  │
│                                                  └─────┬─────┘  │
│  High-level: List, Dict, Classes, Exceptions           │        │
│                                                        ↓        │
│                                              ┌─────────────────┐│
│                                              │    Zero IR      ││
│                                              └────────┬────────┘│
└───────────────────────────────────────────────────────┼─────────┘
                                                        │
┌───────────────────────────────────────────────────────┼─────────┐
│                      CORE RUNTIME (FROZEN)            │         │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────────────┐ │         │
│  │ Tensor │ │ Scalar │ │ Struct │ │ Core Ops        │ │         │
│  └────────┘ └────────┘ └────────┘ │ (matmul, add..) │ │         │
│  ┌────────────────────┐ ┌────────┴─────────────────┐│ │         │
│  │ Control Flow       │ │ Memory Model             ││ │         │
│  │ (if, for, while)   │ │ (alloc, free, sync)      ││ │         │
│  └────────────────────┘ └──────────────────────────┘│ │         │
│                           Reference implementations  ↓ │         │
└───────────────────────────────────────────────────────┼─────────┘
                                                        │
┌───────────────────────────────────────────────────────┼─────────┐
│                      BACKENDS (Separate Repos)        │         │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐│         │
│  │ LLVM        │    │ MLIR        │    │ CUDA PTX   ││         │
│  │ (CPU opt)   │    │ (dialects)  │    │ (GPU)      ││         │
│  └─────────────┘    └─────────────┘    └────────────┘│         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Core Architectural Principles

> **CRITICAL**: Zero IR should be _boringly small_

### The Minimal IR Philosophy

If Zero IR grows too expressive, you're recreating MLIR badly.

**Zero IR should ONLY contain:**

- ✅ Ops (load, store, add, mul, matmul, etc.)
- ✅ Activations (relu, sigmoid, tanh) — v1.1
- ✅ Loops (for, while)
- ✅ Memory (alloc, free, copy)
- ✅ Control flow (if, branch, call, return)

> **v1.1 Status**: Core semantic completeness achieved for ML graphs.
> Performance and dtype expansion are deferred to IR/LLVM backends.

**ML semantics belong in:**

- ✅ MLIR dialects (conv2d, attention, layernorm)
- ✅ Standard library (written IN Zero)
- ✅ Compiler sugar (syntactic transformations)

**Keep Zero IR minimal and stable.**

### What Zero Is Building

You are NOT building:

- ❌ "A faster Python"
- ❌ "Another ML framework"

You ARE building:

> **A language whose lowest layer is ML-native**

This is the right ambition.

---

## 📦 Repo Structure (Recommended)

```
LetsZero/
├── core-runtime/      ← FROZEN (this repo)
├── zero-compiler/     ← Frontend: Lexer, Parser, AST, Type System
├── zero-llvm/         ← Backend: Zero IR → LLVM IR
├── zero-mlir/         ← (Optional) MLIR dialect for ML ops
└── zero-stdlib/       ← Standard library written IN Zero
```

---

## 1️⃣ Connecting LLVM Backend (`zero-llvm`)

### Purpose

Transform Zero IR into optimized LLVM IR for CPU/GPU execution.

### Key Components

```cpp
// zero-llvm/include/codegen.hpp

class LLVMCodegen {
    llvm::LLVMContext ctx;
    llvm::Module* module;
    llvm::IRBuilder<> builder;

public:
    // Lower Zero IR nodes to LLVM IR
    // LLVM backend is "dumb" - it only understands:
    // - pointers, shapes, strides, loops
    // - loads, stores, calls
    // Tensor semantics are already lowered in Zero IR

    llvm::Value* emit_function(const zero::ir::Function& fn);
    llvm::Value* emit_loop(const zero::ir::Loop& loop);
    llvm::Value* emit_load(llvm::Value* ptr, llvm::Value* offset);
    llvm::Value* emit_store(llvm::Value* ptr, llvm::Value* offset, llvm::Value* val);
    llvm::Value* emit_call(const std::string& fn_name, llvm::ArrayRef<llvm::Value*> args);

    // Optimizations
    void run_optimization_passes();  // Loop vectorization, tiling

    // Output
    void emit_object_file(const std::string& path);
};
```

### Optimization Opportunities

| Zero IR Op  | LLVM Optimization                  |
| ----------- | ---------------------------------- |
| MatMul      | Loop tiling, SIMD vectorization    |
| Elementwise | Auto-vectorization (AVX-512)       |
| Reduce      | Parallel reduction patterns        |
| For loops   | Unrolling, polyhedral optimization |

---

## 2️⃣ Connecting MLIR (`zero-mlir`)

### When to Use MLIR

- Custom ML dialects (conv2d, attention, layernorm)
- Multi-level lowering (high-level → linalg → affine → LLVM)
- Better optimization for ML workloads

### Dialect Design

```mlir
// zero-mlir/dialects/ZeroOps.td

def Zero_MatMulOp : Zero_Op<"matmul"> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

// Lowering: Zero.matmul → linalg.matmul → affine loops → LLVM
```

---

## 3️⃣ High-Level Features (List, Dict, Classes)

### The Erasure Principle

> **All high-level constructs are ERASED before reaching the Core Runtime.**

```
┌────────────────────────────────────────────────────────────────┐
│  Zero Source                                                   │
│  ─────────────                                                 │
│  let x: List[f32] = [1.0, 2.0, 3.0]                           │
│  let y = x.map(fn(v) => v * 2)                                │
└─────────────────────────────┬──────────────────────────────────┘
                              │ Compiler lowers to:
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  Zero IR (what Core Runtime sees)                              │
│  ─────────────────────────────────                             │
│  %x = tensor.alloc([3], f32)                                   │
│  store %x, [1.0, 2.0, 3.0]                                    │
│  %y = tensor.alloc([3], f32)                                   │
│  for %i in 0..3:                                               │
│      %v = load %x[%i]                                          │
│      %r = mul %v, 2.0                                          │
│      store %y[%i], %r                                          │
└────────────────────────────────────────────────────────────────┘
```

### Implementation Strategy

| Feature       | Frontend Representation | Core Runtime Lowering                         |
| ------------- | ----------------------- | --------------------------------------------- |
| `List[T]`     | Dynamic array type      | `Tensor` (1D) + length field                  |
| `Dict[K,V]`   | Hash map type           | `Tensor` (keys) + `Tensor` (values) + hash fn |
| `class Foo`   | Named struct + methods  | `Struct` + standalone functions               |
| `try/catch`   | Exception node          | Control flow with error codes                 |
| `async/await` | Coroutine               | State machine in `Struct`                     |

### Example: List Implementation

```cpp
// In zero-compiler: AST → Zero IR lowering

// List[f32] becomes:
struct ZeroList {
    zero::Tensor data;      // Underlying storage
    int64_t length;         // Current length
    int64_t capacity;       // Allocated capacity
};

// list.append(x) becomes:
void list_append(ZeroList* list, float x) {
    if (list->length >= list->capacity) {
        // Grow: allocate new tensor, copy, free old
    }
    store(list->data, list->length, x);
    list->length++;
}
```

### Example: Dict Implementation

```cpp
// Dict[str, f32] becomes:
struct ZeroDict {
    zero::Tensor keys;      // String keys (or hashes)
    zero::Tensor values;    // f32 values
    zero::Tensor occupied;  // Bitmap for slots
    int64_t size;
    int64_t capacity;
};
```

---

## 4️⃣ Development Roadmap

### Phase 1: Zero Compiler MVP

- [ ] Lexer/Tokenizer
- [ ] Parser → AST
- [ ] Type checker
- [ ] Zero IR generation
- [ ] Interpreter (for testing)

### Phase 2: LLVM Backend

- [ ] Basic codegen (scalar ops)
- [ ] Tensor allocation/access
- [ ] Control flow
- [ ] Function calls
- [ ] Optimization passes

### Phase 3: High-Level Features (Limit Scope)

> **Focus**: "Enough to write training loops" — skip fancy stdlib initially

- [ ] List type + basic operations (append, index, len)
- [ ] Dict type + basic operations (get, set, has)
- [ ] String type (minimal: concat, compare)
- [ ] Error handling (basic: panic, assert)

**What to skip for now:**

- ❌ Advanced list methods (sort, filter, etc.)
- ❌ Complex string operations (regex, formatting)
- ❌ Full exception system with try/catch

### Phase 4A: ML Features (Runtime-Based)

> **CRITICAL**: Don't make autograd a compiler pass first. Many projects die here.

- [ ] Autograd (runtime tape, PyTorch-style)
  - [ ] Tape-based gradient tracking
  - [ ] Backward pass execution
  - [ ] Basic optimizer (SGD)
- [ ] CPU training works end-to-end
- [ ] Model serialization (weights only)

**Why runtime-first?**

- You need working ML examples **fast**
- IR-level AD is hard and thankless early
- Delay elegance. Ship usefulness.

### Phase 4B: Advanced Backends

- [ ] GPU backend (CUDA/PTX)
- [ ] MLIR integration
- [ ] Kernel fusion
- [ ] IR-level autograd (MLIR-friendly)

---

## 🔗 Key Interfaces

### Zero IR Format (In-Memory C++ Structs)

> **CRITICAL**: Zero IR is NOT JSON-first. JSON is only for debug/inspection.

```cpp
// zero-compiler/include/zero/ir/ir.hpp

namespace zero::ir {

enum class OpKind {
    Load, Store, Add, Mul, MatMul,
    Loop, Branch, Call, Return
};

struct Value {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
};

struct Op {
    OpKind kind;
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::map<std::string, Attribute> attrs;
};

struct BasicBlock {
    std::string name;
    std::vector<Op*> ops;
    BasicBlock* next;
};

struct Function {
    std::string name;
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::vector<BasicBlock*> blocks;
};

struct Module {
    std::vector<Function*> functions;

    // Binary serialization (flatbuffers/protobuf/custom)
    void serialize_to_binary(const std::string& path);
    static Module* deserialize_from_binary(const std::string& path);

    // JSON = debug dump only
    std::string to_json_debug() const;
};

} // namespace zero::ir
```

**Why not JSON-first?**

- JSON IR kills compile speed
- Makes pattern matching painful
- LLVM/MLIR integrations will hate you
- **Think: JSON is printf for IR, not the IR itself**

### C API for Core Runtime

```c
// For embedding or FFI
extern "C" {
    ZeroTensor* zero_tensor_alloc(int64_t* shape, int ndim, ZeroDType dtype);
    void zero_tensor_free(ZeroTensor* t);
    void zero_matmul(ZeroTensor* a, ZeroTensor* b, ZeroTensor* c);
    // ...
}
```

---

## ✅ Checklist Before Starting New Repo

1. **Clone Core Runtime** — Use as submodule or copy headers
2. **Define Zero IR spec** — What nodes, what serialization format?
3. **Decide on LLVM version** — Currently targeting LLVM 17+
4. **Set up CI** — Test on Linux, macOS, Windows

---

## 📝 Summary of Key Architectural Decisions

1. **Zero IR is NOT JSON-first** — In-memory C++ structs + binary serialization. JSON = debug only.
2. **LLVM backend is "dumb"** — Only understands pointers, loops, loads, stores. No Tensor semantics.
3. **Autograd is runtime-first** — PyTorch-style tape before IR-level AD.
4. **Zero IR is boringly small** — Ops, loops, memory, control flow. That's it.
5. **ML semantics live elsewhere** — MLIR dialects, stdlib, compiler sugar.

---

_"The Core Runtime is the law. Everything else is just syntax sugar."_
