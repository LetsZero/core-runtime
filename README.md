<p align="center">
  <img src="https://avatars.githubusercontent.com/u/249721810?s=400&u=75e07d4ed7e9c23ec8f2db301fa89a4942ff1806&v=4" width="200" alt="Zero Logo">
</p>

<h1 align="center">Zero — Core Runtime</h1>

<p align="center">
  <strong>The Immutable Substrate for High-Performance ML.</strong><br>
  <code>v1.0.0 — FROZEN</code>
</p>

> ⚠️ **FROZEN CORE** — This repository is now frozen. Updates will be rare, highly scrutinized, and focused exclusively on bug fixes.

## 🧊 Status: FROZEN

The Core Runtime implements all 7 primitives defined in `docs/CORE_RUNTIME_SPEC.md`:

| Primitive        | Header                | Description                           |
| ---------------- | --------------------- | ------------------------------------- |
| **Tensor**       | `core/tensor.hpp`     | The only data container, O(1) views   |
| **Scalar**       | `core/scalar.hpp`     | Rank-0 tensor, compile-time constants |
| **Struct**       | `core/struct.hpp`     | Static aggregation, no methods        |
| **Control Flow** | `ir/control_flow.hpp` | if/for/while → LLVM basic blocks      |
| **Functions**    | `ir/function.hpp`     | Pure by default, explicit I/O         |
| **Memory**       | `core/memory.hpp`     | Explicit allocation, device placement |
| **Core Ops**     | `ops/*.hpp`           | Elementwise, MatMul, Reduce, Reshape  |

## 🔧 Build

```bash
cmake -B build -DZERO_BUILD_TESTS=ON
cmake --build build --config Release
./build/tests/Release/zero_basic_test
./build/tests/Release/zero_benchmark
```

## 🧪 Tests

- **58 correctness tests** — All pass ✅
- **Benchmarks** — MatMul, Elementwise, Reduce

## 📐 Architecture

```
Zero Source → Parser → AST → Zero IR
                              ↓
                    ┌─────────────────┐
                    │  Core Runtime   │  ← THIS REPO (FROZEN)
                    │  (Pure C++20)   │
                    └────────┬────────┘
                             ↓ Zero IR
                    ┌─────────────────┐
                    │  LLVM Backend   │  ← Separate repo
                    └─────────────────┘
```

## 🔒 Frozen Core Commitment

> _"If it can be implemented in Zero, it does not belong in the Core."_

This repository follows a **"Freeze Early"** policy. The fundamental primitives are verified and stable. Future language features will be built on top of this frozen substrate.
