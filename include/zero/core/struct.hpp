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

#include <array>
#include <cstdint>
#include <cstring>

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
 * @brief Field descriptor for struct layout
 */
struct FieldDesc {
    const char* name;      ///< Field name (for debugging)
    size_t offset;         ///< Byte offset in struct
    FieldType type;        ///< Tensor or Scalar
    DType dtype;           ///< Data type (for scalars)
    
    constexpr FieldDesc() noexcept 
        : name(nullptr), offset(0), type(FieldType::TENSOR), dtype(DType::F32) {}
    
    constexpr FieldDesc(const char* n, size_t off, FieldType t, DType dt = DType::F32) noexcept
        : name(n), offset(off), type(t), dtype(dt) {}
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
    void add_tensor(const char* name) noexcept {
        if (num_fields >= MAX_STRUCT_FIELDS) return;
        
        // Align to 8 bytes for pointers
        total_size = (total_size + 7) & ~static_cast<size_t>(7);
        
        fields[num_fields] = FieldDesc(name, total_size, FieldType::TENSOR);
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
        s.data = mem_alloc(layout->total_size, 8, Device::CPU);
        s.owns_data = (s.data != nullptr);
        if (s.data != nullptr) {
            std::memset(s.data, 0, layout->total_size);
        }
        return s;
    }
    
    /**
     * @brief Get pointer to a field
     */
    void* field_ptr(int8_t index) const noexcept {
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
        const FieldDesc* field = layout->get_field(index);
        if (field == nullptr || field->type != FieldType::SCALAR) return;
        value.to_bytes(field_ptr(index));
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
