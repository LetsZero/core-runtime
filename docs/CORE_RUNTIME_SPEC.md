# **Zero — Core Runtime (Minimal & Sufficient)**

> **Everything else is erased before this layer.**

The Core Runtime is the final boundary of the Zero compiler. It is the language of execution, not the language of expression. If a high-level feature cannot be lowered into these seven primitives, it cannot exist in Zero.

---

### **1. Tensor**
The *only* real data container. Tensors are treated as first-class primitives, not library objects.
*   **Attributes:** Shape, strides, dtype, device (CPU/GPU/NPU).
*   **Data:** Raw memory pointer.
*   **Operations:** Views (slice, reshape, transpose) must be O(1) metadata changes.
*   **Constraint:** No lists. No dicts. No objects.

### **2. Scalar**
A rank-0 tensor or immediate value. 
*   **Usage:** Loop bounds, hyperparameters, small constants.
*   **Optimization:** Scalars must be easily inlined or constant-folded by the LLVM backend.

### **3. Struct (Plain Data Only)**
Static aggregation of tensors and scalars. 
*   **Usage:** Grouping model state, optimizer parameters, or gradient sets.
*   **Rules:** 
    *   No methods or inheritance.
    *   No mutation semantics (handled by explicit store).
    *   Memory layout must be deterministic and known at compile-time.

### **4. Control Flow**
Only what lowers cleanly to LLVM Basic Blocks and Branches.
*   **Supported:** `if / else`, `for`, `while`.
*   **Rules:** 
    *   No dynamic dispatch or virtual tables.
    *   No exceptions inside compute kernels.
    *   Prefer analyzable bounds to allow for loop unrolling and vectorization.

### **5. Functions (Pure by Default)**
The building blocks of the execution graph.
*   **Requirements:** Explicit inputs, explicit outputs.
*   **Transparency:** No hidden global state. 
*   **Result:** This allows the compiler to treat function calls as nodes in a mathematical graph, enabling aggressive fusion.

### **6. Memory & Device Model**
Explicit, manual, and non-magical.
*   **Includes:** Allocation/Deallocation, device placement (Host vs. Device), explicit copy/sync, and strict aliasing rules.
*   **Compiler Role:** The compiler must fully understand the "side effects" of every memory operation to prevent race conditions.

### **7. Core Tensor Ops**
A minimal, orthogonal set of mathematical primitives.
*   **Set:** Elementwise (add, mul, etc.), MatMul, Convolution, Reduction (sum, max), Broadcast, and Reshape.
*   **Composition:** Every complex ML operation (LayerNorm, Attention, etc.) must be a composition of these core ops or a specialized lowering.

---

## **🚫 What is NOT in Core Runtime**

The following features are **explicitly excluded** from the Core. They may exist in the Zero frontend for developer convenience, but they are **erased** (lowered) before reaching the runtime:

*   ❌ **Collections:** Dict, Map, Set, Tuple, List.
*   ❌ **OOP:** Classes, Inheritance, Virtual methods, Objects.
*   ❌ **Error Handling:** Exceptions, Try/Catch.
*   ❌ **Flexibility:** Hashing, Dynamic polymorphism, Runtime reflection.

---

## **🧠 One-Line Mental Model**

> **If it cannot be represented as Tensors + Structs + Functions + Control Flow, it does not belong in the Core Runtime.**

---

## **Why this is enough**
By restricting the core to these primitives, Zero gains the power to implement:
1.  **Autograd:** Via graph traversal of pure functions.
2.  **Optimizers:** Via struct-based parameter updates.
3.  **Graph Compilers:** Because the limited scope makes the code "easy" for the compiler to reason about.

---

## **🔒 Final Recommendation**
**Freeze this spec early.** 
By locking these seven categories, you ensure that while the Zero language grows more "convenient" (adding syntax sugar, better types, etc.), the underlying engine remains a lean, high-speed machine that never slows down.