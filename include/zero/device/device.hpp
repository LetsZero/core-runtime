#pragma once

/**
 * @file device.hpp
 * @brief Zero Core Runtime — Device Abstraction
 * 
 * Minimal device enumeration and capabilities.
 * Backend-specific implementations will extend this.
 */

#include <cstdint>

namespace zero {

/**
 * @brief Supported compute devices
 */
enum class Device : uint8_t {
    CPU = 0,    // Host CPU
    GPU = 1,    // CUDA/ROCm GPU
    NPU = 2,    // Neural Processing Unit
};

/**
 * @brief Get human-readable device name
 */
constexpr const char* device_name(Device device) noexcept {
    switch (device) {
        case Device::CPU: return "cpu";
        case Device::GPU: return "gpu";
        case Device::NPU: return "npu";
    }
    return "unknown";
}

/**
 * @brief Check if device is available
 * 
 * CPU is always available. GPU/NPU availability depends on backends.
 */
constexpr bool device_available(Device device) noexcept {
    // For now, only CPU is available in the core runtime
    // GPU/NPU backends will override this
    return device == Device::CPU;
}

} // namespace zero
