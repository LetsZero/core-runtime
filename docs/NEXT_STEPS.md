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
    llvm::Value* emit(const zero::ir::Function& fn);
    llvm::Value* emit(const zero::Tensor& tensor);
    llvm::Value* emit_matmul(llvm::Value* A, llvm::Value* B);

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

### Phase 3: High-Level Features

- [ ] List type + operations
- [ ] Dict type + operations
- [ ] String type
- [ ] Error handling

### Phase 4: ML Features

- [ ] Autograd (reverse-mode AD)
- [ ] GPU backend (CUDA)
- [ ] Model serialization

---

## 🔗 Key Interfaces

### Zero IR Format (JSON/Binary)

```json
{
  "functions": [
    {
      "name": "forward",
      "inputs": [{ "name": "x", "type": "tensor<f32, [?, 784]>" }],
      "outputs": [{ "name": "y", "type": "tensor<f32, [?, 10]>" }],
      "body": [
        { "op": "matmul", "args": ["x", "weight1"], "result": "h1" },
        { "op": "relu", "args": ["h1"], "result": "h1_act" },
        { "op": "matmul", "args": ["h1_act", "weight2"], "result": "y" }
      ]
    }
  ]
}
```

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

_"The Core Runtime is the law. Everything else is just syntax sugar."_
