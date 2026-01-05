#pragma once

/**
 * @file sync.hpp
 * @brief Zero Core Runtime — Device Synchronization
 * 
 * Memory copy and synchronization primitives between devices.
 */

#include "device.hpp"
#include "../core/memory.hpp"
#include "../core/tensor.hpp"

namespace zero {

/**
 * @brief Copy direction for device transfers
 */
enum class CopyDir : uint8_t {
    HOST_TO_HOST = 0,
    HOST_TO_DEVICE = 1,
    DEVICE_TO_HOST = 2,
    DEVICE_TO_DEVICE = 3,
};

/**
 * @brief Determine copy direction from source and destination devices
 */
constexpr CopyDir get_copy_direction(Device src, Device dst) noexcept {
    if (src == Device::CPU && dst == Device::CPU) return CopyDir::HOST_TO_HOST;
    if (src == Device::CPU) return CopyDir::HOST_TO_DEVICE;
    if (dst == Device::CPU) return CopyDir::DEVICE_TO_HOST;
    return CopyDir::DEVICE_TO_DEVICE;
}

/**
 * @brief Synchronous memory copy
 * 
 * @param dst      Destination pointer
 * @param src      Source pointer
 * @param size     Number of bytes
 * @param dst_dev  Destination device
 * @param src_dev  Source device
 * @return True on success
 */
inline bool device_copy(
    void* dst,
    const void* src,
    size_t size,
    Device dst_dev,
    Device src_dev
) noexcept {
    CopyDir dir = get_copy_direction(src_dev, dst_dev);
    
    // For now, only CPU-to-CPU is implemented
    if (dir == CopyDir::HOST_TO_HOST) {
        mem_copy_cpu(dst, src, size);
        return true;
    }
    
    // GPU/NPU backends will implement other paths
    return false;
}

/**
 * @brief Copy tensor to another device
 * 
 * @param input  Source tensor
 * @param device Target device
 * @return New tensor on target device
 */
inline Tensor tensor_to_device(const Tensor& input, Device device) noexcept {
    if (input.device == device) {
        // Same device, return view
        Tensor view = input;
        view.owns_data = false;
        return view;
    }
    
    // Allocate on target device
    Tensor output = Tensor::alloc(input.shape.data(), input.ndim, input.dtype, device);
    if (output.data == nullptr) {
        return Tensor::empty();
    }
    
    // Copy data
    if (!device_copy(output.data, input.data, input.nbytes(), device, input.device)) {
        output.free();
        return Tensor::empty();
    }
    
    return output;
}

/**
 * @brief Synchronize device execution
 * 
 * Blocks until all operations on the device are complete.
 */
inline void device_sync(Device device) noexcept {
    if (device == Device::CPU) {
        // CPU is always synchronous
        return;
    }
    
    // GPU/NPU backends will implement async sync
}

/**
 * @brief Stream handle for async operations
 */
struct Stream {
    uint64_t handle;
    Device device;
    
    Stream() noexcept : handle(0), device(Device::CPU) {}
    
    /**
     * @brief Create a new stream
     */
    static Stream create(Device dev) noexcept {
        Stream s;
        s.device = dev;
        // GPU/NPU backends will create actual stream
        return s;
    }
    
    /**
     * @brief Synchronize this stream
     */
    void sync() const noexcept {
        device_sync(device);
    }
    
    /**
     * @brief Destroy the stream
     */
    void destroy() noexcept {
        // GPU/NPU backends will destroy stream
        handle = 0;
    }
};

/**
 * @brief Async copy with stream
 */
inline bool device_copy_async(
    void* dst,
    const void* src,
    size_t size,
    Device dst_dev,
    Device src_dev,
    Stream* stream
) noexcept {
    // For CPU, async is same as sync
    if (dst_dev == Device::CPU && src_dev == Device::CPU) {
        return device_copy(dst, src, size, dst_dev, src_dev);
    }
    
    // GPU/NPU backends will implement async copy
    (void)stream;
    return false;
}

} // namespace zero
