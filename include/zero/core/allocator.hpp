#pragma once

/**
 * @file allocator.hpp
 * @brief Zero Core Runtime — Abstract Allocator Interface
 * 
 * Pluggable memory allocation for custom allocators (pools, arenas, etc.)
 * Default implementation uses system aligned malloc.
 */

#include "../device/device.hpp"

#include <cstddef>
#include <cstdlib>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace zero {

// ─────────────────────────────────────────────────────────────────────────────
// Abstract Allocator Interface
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Abstract allocator interface for pluggable memory management
 * 
 * Implementations must be thread-safe if used from multiple threads.
 */
struct Allocator {
    virtual ~Allocator() = default;
    
    /**
     * @brief Allocate memory with specified alignment
     * 
     * @param size      Number of bytes to allocate
     * @param alignment Required alignment (must be power of 2)
     * @param device    Target device for allocation
     * @return Pointer to allocated memory, or nullptr on failure
     */
    virtual void* alloc(size_t size, size_t alignment, Device device) noexcept = 0;
    
    /**
     * @brief Free previously allocated memory
     * 
     * @param ptr    Pointer to memory block (nullptr is safe)
     * @param device Device where memory was allocated
     */
    virtual void free(void* ptr, Device device) noexcept = 0;
    
    /**
     * @brief Get allocator name for debugging
     */
    virtual const char* name() const noexcept = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// System Allocator (Default)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Default system allocator using platform aligned malloc
 * 
 * This is the default allocator used by the runtime.
 */
struct SystemAllocator final : Allocator {
    void* alloc(size_t size, size_t alignment, Device device) noexcept override {
        if (size == 0) return nullptr;
        
        // Only CPU allocation for now
        if (device != Device::CPU) {
            return nullptr;
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
    
    void free(void* ptr, Device device) noexcept override {
        if (ptr == nullptr) return;
        
        if (device != Device::CPU) {
            return; // GPU/NPU backends will extend this
        }

#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
    }
    
    const char* name() const noexcept override { return "system"; }
    
    /**
     * @brief Get singleton instance
     */
    static SystemAllocator* instance() noexcept {
        static SystemAllocator alloc;
        return &alloc;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Global Allocator Access
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {
    inline Allocator*& global_allocator_ptr() noexcept {
        static Allocator* ptr = SystemAllocator::instance();
        return ptr;
    }
}

/**
 * @brief Get the current global allocator
 */
inline Allocator* get_allocator() noexcept {
    return detail::global_allocator_ptr();
}

/**
 * @brief Set the global allocator
 * 
 * @warning Not thread-safe. Call at startup before any allocations.
 * @param alloc Pointer to allocator (must outlive all allocations)
 */
inline void set_allocator(Allocator* alloc) noexcept {
    if (alloc != nullptr) {
        detail::global_allocator_ptr() = alloc;
    }
}

} // namespace zero
