#pragma once

/**
 * @file status.hpp
 * @brief Zero Core Runtime — Lightweight Status Model
 * 
 * Non-throwing, opt-in error system for validation.
 * No exceptions, no allocations.
 */

#include <cstdint>

namespace zero {

/**
 * @brief Status codes for Zero operations
 */
enum class StatusCode : uint8_t {
    OK = 0,                 // Success
    INVALID_ARGUMENT = 1,   // Bad parameter value
    OUT_OF_BOUNDS = 2,      // Index or size exceeded
    ALLOCATION_FAILED = 3,  // Memory allocation failed
    TYPE_MISMATCH = 4,      // Incompatible types
    INVALID_STATE = 5,      // Object in invalid state
    NOT_IMPLEMENTED = 6,    // Feature not available
};

/**
 * @brief Lightweight status result
 * 
 * Non-owning, stack-only. Used for validation in debug
 * and propagation in compiler-generated code.
 */
struct Status {
    StatusCode code;
    const char* msg;  // Static string, not owned
    
    constexpr Status() noexcept : code(StatusCode::OK), msg(nullptr) {}
    constexpr Status(StatusCode c, const char* m = nullptr) noexcept : code(c), msg(m) {}
    
    static constexpr Status ok() noexcept {
        return Status(StatusCode::OK);
    }
    
    static constexpr Status error(StatusCode c, const char* m = nullptr) noexcept {
        return Status(c, m);
    }
    
    constexpr bool is_ok() const noexcept {
        return code == StatusCode::OK;
    }
    
    constexpr bool is_error() const noexcept {
        return code != StatusCode::OK;
    }
    
    constexpr explicit operator bool() const noexcept {
        return is_ok();
    }
};

// ─────────────────────────────────────────────────────────────────────
// Convenience Status Factories
// ─────────────────────────────────────────────────────────────────────

namespace status {
    constexpr Status OK = Status::ok();
    
    inline Status invalid_argument(const char* msg = nullptr) noexcept {
        return Status::error(StatusCode::INVALID_ARGUMENT, msg);
    }
    
    inline Status out_of_bounds(const char* msg = nullptr) noexcept {
        return Status::error(StatusCode::OUT_OF_BOUNDS, msg);
    }
    
    inline Status allocation_failed(const char* msg = nullptr) noexcept {
        return Status::error(StatusCode::ALLOCATION_FAILED, msg);
    }
    
    inline Status type_mismatch(const char* msg = nullptr) noexcept {
        return Status::error(StatusCode::TYPE_MISMATCH, msg);
    }
    
    inline Status invalid_state(const char* msg = nullptr) noexcept {
        return Status::error(StatusCode::INVALID_STATE, msg);
    }
}

} // namespace zero
