#pragma once

/**
 * @file struct.hpp
 * @brief Zero Core Runtime — Struct Primitive
 * 
 * Static aggregation of tensors and scalars.
 * No methods, no inheritance. Plain data only.
 */

#include "dtype.hpp"
#include "tensor.hpp"
#include "scalar.hpp"
#include "status.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace zero {

/// Maximum number of fields in a struct
constexpr int8_t MAX_STRUCT_FIELDS = 32;

/**
 * @brief Field type enumeration
 */
enum class FieldType : uint8_t {
    TENSOR = 0,
    SCALAR = 1,
};

/**
 * @brief Optional tensor metadata for model IO contracts
 */
struct TensorMeta {
    int8_t rank;              // -1 means dynamic/unknown
    const int64_t* shape;     // nullptr means dynamic shape
    DType dtype;
    
    constexpr TensorMeta() noexcept 
        : rank(-1), shape(nullptr), dtype(DType::F32) {}
    
    constexpr TensorMeta(int8_t r, const int64_t* s, DType dt) noexcept
        : rank(r), shape(s), dtype(dt) {}
};

/**
 * @brief Field descriptor for struct layout
 */
struct FieldDesc {
    const char* name;           ///< Field name (for debugging)
    size_t offset;              ///< Byte offset in struct
    FieldType type;             ///< Tensor or Scalar
    DType dtype;                ///< Data type (for scalars)
    bool is_optional;           ///< True if field can be null
    bool is_trainable;          ///< True if field participates in training
    const TensorMeta* meta;     ///< Optional tensor shape/dtype metadata
    
    constexpr FieldDesc() noexcept 
        : name(nullptr), offset(0), type(FieldType::TENSOR), dtype(DType::F32),
          is_optional(false), is_trainable(false), meta(nullptr) {}
    
    constexpr FieldDesc(const char* n, size_t off, FieldType t, DType dt = DType::F32) noexcept
        : name(n), offset(off), type(t), dtype(dt),
          is_optional(false), is_trainable(false), meta(nullptr) {}
};

/**
 * @brief Struct layout descriptor
 * 
 * Describes the memory layout of a Zero struct.
 * Used by the compiler to generate efficient access code.
 */
struct StructLayout {
    std::array<FieldDesc, MAX_STRUCT_FIELDS> fields;
    int8_t num_fields;
    size_t total_size;
    
    constexpr StructLayout() noexcept 
        : fields{}, num_fields(0), total_size(0) {}
    
    /**
     * @brief Add a tensor field
     */
    void add_tensor(const char* name, bool optional = false, bool trainable = false) noexcept {
        if (num_fields >= MAX_STRUCT_FIELDS) return;
        
        // Align to 8 bytes for pointers
        total_size = (total_size + 7) & ~static_cast<size_t>(7);
        
        fields[num_fields] = FieldDesc(name, total_size, FieldType::TENSOR);
        fields[num_fields].is_optional = optional;
        fields[num_fields].is_trainable = trainable;
        total_size += sizeof(Tensor);
        num_fields++;
    }
    
    /**
     * @brief Add a scalar field
     */
    void add_scalar(const char* name, DType dtype) noexcept {
        if (num_fields >= MAX_STRUCT_FIELDS) return;
        
        size_t align = dtype_alignment(dtype);
        total_size = (total_size + align - 1) & ~(align - 1);
        
        fields[num_fields] = FieldDesc(name, total_size, FieldType::SCALAR, dtype);
        total_size += dtype_size(dtype);
        num_fields++;
    }
    
    /**
     * @brief Get field by index
     */
    const FieldDesc* get_field(int8_t index) const noexcept {
        if (index < 0 || index >= num_fields) return nullptr;
        return &fields[index];
    }
    
    /**
     * @brief Find field by name
     */
    const FieldDesc* find_field(const char* name) const noexcept {
        for (int8_t i = 0; i < num_fields; ++i) {
            if (fields[i].name != nullptr && std::strcmp(fields[i].name, name) == 0) {
                return &fields[i];
            }
        }
        return nullptr;
    }
    
    /**
     * @brief Validate layout (debug-only checks)
     */
    Status validate() const noexcept {
        if (num_fields < 0 || num_fields > MAX_STRUCT_FIELDS) {
            return status::invalid_argument("invalid num_fields");
        }
        
        // Check for duplicate names
        for (int8_t i = 0; i < num_fields; ++i) {
            if (fields[i].name == nullptr) continue;
            for (int8_t j = i + 1; j < num_fields; ++j) {
                if (fields[j].name != nullptr && 
                    std::strcmp(fields[i].name, fields[j].name) == 0) {
                    return status::invalid_argument("duplicate field name");
                }
            }
        }
        
        return status::OK;
    }
    
#ifndef NDEBUG
    void dump() const noexcept {
        std::printf("StructLayout: %d fields, %zu bytes\n", num_fields, total_size);
        for (int8_t i = 0; i < num_fields; ++i) {
            const FieldDesc& f = fields[i];
            std::printf("  [%d] %s: offset=%zu type=%s",
                i, f.name ? f.name : "(null)", f.offset,
                f.type == FieldType::TENSOR ? "tensor" : "scalar");
            if (f.is_optional) std::printf(" optional");
            if (f.is_trainable) std::printf(" trainable");
            std::printf("\n");
        }
    }
#else
    void dump() const noexcept {}
#endif
};

/**
 * @brief Runtime struct instance
 * 
 * Holds the actual data for a struct instance.
 */
struct StructData {
    void* data;                 ///< Raw memory block
    const StructLayout* layout; ///< Layout descriptor
    bool owns_data;             ///< True if instance owns its memory
    
    /**
     * @brief Allocate a new struct instance
     */
    static StructData alloc(const StructLayout* layout) noexcept {
        StructData s;
        s.layout = layout;
        s.data = nullptr;
        s.owns_data = false;
        
        if (layout == nullptr) return s;
        
        s.data = mem_alloc(layout->total_size, 8, Device::CPU);
        s.owns_data = (s.data != nullptr);
        if (s.data != nullptr) {
            std::memset(s.data, 0, layout->total_size);
        }
        return s;
    }
    
    /**
     * @brief Wrap external memory (non-owning view)
     */
    static StructData wrap(void* external, const StructLayout* layout) noexcept {
        StructData s;
        s.data = external;
        s.layout = layout;
        s.owns_data = false;  // Never owns external memory
        return s;
    }
    
    /**
     * @brief Check if this is a non-owning view
     */
    bool is_view() const noexcept {
        return !owns_data;
    }
    
    /**
     * @brief Clone this struct (deep or shallow)
     * 
     * @param deep If true, allocates new memory and copies data.
     *             If false, returns a non-owning view.
     */
    StructData clone(bool deep) const noexcept {
        if (!deep) {
            return wrap(data, layout);
        }
        
        StructData copy = alloc(layout);
        if (copy.data != nullptr && data != nullptr && layout != nullptr) {
            mem_copy_cpu(copy.data, data, layout->total_size);
        }
        return copy;
    }
    
    /**
     * @brief Get pointer to a field
     */
    void* field_ptr(int8_t index) const noexcept {
        if (layout == nullptr || data == nullptr) return nullptr;
        const FieldDesc* field = layout->get_field(index);
        if (field == nullptr) return nullptr;
        return static_cast<uint8_t*>(data) + field->offset;
    }
    
    /**
     * @brief Get tensor field
     */
    Tensor* tensor_field(int8_t index) const noexcept {
        return static_cast<Tensor*>(field_ptr(index));
    }
    
    /**
     * @brief Get scalar field value
     */
    Scalar scalar_field(int8_t index) const noexcept {
        if (layout == nullptr) return Scalar();
        const FieldDesc* field = layout->get_field(index);
        if (field == nullptr || field->type != FieldType::SCALAR) {
            return Scalar();
        }
        return Scalar::from_bytes(field_ptr(index), field->dtype);
    }
    
    /**
     * @brief Set scalar field value
     */
    void set_scalar(int8_t index, const Scalar& value) noexcept {
        if (layout == nullptr) return;
        const FieldDesc* field = layout->get_field(index);
        if (field == nullptr || field->type != FieldType::SCALAR) return;
        value.to_bytes(field_ptr(index));
    }
    
    /**
     * @brief Reset to empty state (frees if owning)
     */
    void reset() noexcept {
        free();
        layout = nullptr;
    }
    
    /**
     * @brief Free owned memory
     */
    void free() noexcept {
        if (owns_data && data != nullptr) {
            mem_free(data, Device::CPU);
            data = nullptr;
            owns_data = false;
        }
    }
};

} // namespace zero
