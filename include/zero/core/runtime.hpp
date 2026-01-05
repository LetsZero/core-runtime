#pragma once

/**
 * @file runtime.hpp
 * @brief Zero Core Runtime — Global Runtime Configuration
 * 
 * Seed control and deterministic mode for reproducible execution.
 */

#include <cstdint>

namespace zero {

/**
 * @brief Global runtime configuration
 * 
 * Controls reproducibility and determinism settings.
 */
struct RuntimeConfig {
    uint64_t seed = 0;            ///< Global seed for reproducibility
    bool deterministic = false;   ///< Force deterministic ops (may be slower)
    
    /**
     * @brief Get singleton instance
     */
    static RuntimeConfig& instance() noexcept {
        static RuntimeConfig config;
        return config;
    }

private:
    RuntimeConfig() = default;
};

// ─────────────────────────────────────────────────────────────────────────────
// Seed Control API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Set global seed for reproducibility
 * 
 * Also enables deterministic mode.
 * 
 * @param seed The seed value
 */
inline void set_seed(uint64_t seed) noexcept {
    RuntimeConfig::instance().seed = seed;
    RuntimeConfig::instance().deterministic = true;
}

/**
 * @brief Get current global seed
 */
inline uint64_t get_seed() noexcept {
    return RuntimeConfig::instance().seed;
}

/**
 * @brief Check if deterministic mode is enabled
 */
inline bool is_deterministic() noexcept {
    return RuntimeConfig::instance().deterministic;
}

/**
 * @brief Enable/disable deterministic mode without setting seed
 */
inline void set_deterministic(bool enabled) noexcept {
    RuntimeConfig::instance().deterministic = enabled;
}

} // namespace zero
