#pragma once

/**
 * @file memory.hpp
 * @brief Zero Core Runtime — Memory Allocation Primitives
 * 
 * Explicit, manual memory management with device placement.
 * No hidden allocations, no garbage collection.
 */

#include "dtype.hpp"
#include "../device/device.hpp"

#include <cstdlib>
#include <cstring>
#include <new>

namespace zero {

/**
 * @brief Memory allocation result
 */
struct MemoryBlock {
    void* ptr;
    size_t size;
    size_t alignment;
    Device device;
};

/**
 * @brief Allocate aligned memory on the specified device
 * 
 * @param size      Number of bytes to allocate
 * @param alignment Required alignment (must be power of 2)
 * @param device    Target device for allocation
 * @return Pointer to allocated memory, or nullptr on failure
 */
inline void* mem_alloc(size_t size, size_t alignment, Device device) noexcept {
    if (size == 0) return nullptr;
    
    // For now, only CPU allocation is implemented
    // GPU/NPU backends will extend this
    if (device != Device::CPU) {
        return nullptr; // Not yet implemented
    }

#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    // posix_memalign requires alignment >= sizeof(void*) and power of 2
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/**
 * @brief Free memory allocated by mem_alloc
 * 
 * @param ptr    Pointer to memory block
 * @param device Device where memory was allocated
 */
inline void mem_free(void* ptr, Device device) noexcept {
    if (ptr == nullptr) return;
    
    if (device != Device::CPU) {
        return; // GPU/NPU backends will extend this
    }

#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Allocate zeroed memory
 */
inline void* mem_alloc_zero(size_t size, size_t alignment, Device device) noexcept {
    void* ptr = mem_alloc(size, alignment, device);
    if (ptr != nullptr) {
        std::memset(ptr, 0, size);
    }
    return ptr;
}

/**
 * @brief Copy memory between locations (same device)
 * 
 * @param dst  Destination pointer
 * @param src  Source pointer
 * @param size Number of bytes to copy
 */
inline void mem_copy(void* dst, const void* src, size_t size) noexcept {
    if (dst != nullptr && src != nullptr && size > 0) {
        std::memcpy(dst, src, size);
    }
}

/**
 * @brief Calculate total bytes needed for a tensor
 * 
 * @param shape Array of dimension sizes
 * @param ndim  Number of dimensions
 * @param dtype Data type
 * @return Total bytes required
 */
constexpr size_t calc_tensor_bytes(const int64_t* shape, int8_t ndim, DType dtype) noexcept {
    if (ndim == 0) return dtype_size(dtype); // Scalar
    
    size_t numel = 1;
    for (int8_t i = 0; i < ndim; ++i) {
        numel *= static_cast<size_t>(shape[i]);
    }
    return numel * dtype_size(dtype);
}

/**
 * @brief Calculate strides for contiguous memory layout (row-major)
 * 
 * @param shape   Array of dimension sizes
 * @param ndim    Number of dimensions
 * @param dtype   Data type
 * @param strides Output array for strides (must have ndim elements)
 */
inline void calc_contiguous_strides(
    const int64_t* shape, 
    int8_t ndim, 
    DType dtype,
    int64_t* strides
) noexcept {
    if (ndim == 0) return;
    
    int64_t stride = static_cast<int64_t>(dtype_size(dtype));
    for (int8_t i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

} // namespace zero
