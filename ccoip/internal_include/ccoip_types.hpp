#pragma once

#include <array>
#include <c++/12/stdexcept>
#include <cstdint>
#include <string>

namespace ccoip {
    enum ccoip_data_type_t {
        ccoipUint4 = 0, // only supported for quantization
        ccoipUint8 = 1,
        ccoipInt8 = 2,
        ccoipUint16 = 3,
        ccoipUint32 = 4,
        ccoipInt16 = 5,
        ccoipInt32 = 6,
        ccoipUint64 = 7,
        ccoipInt64 = 8,
        ccoipFloat16 = 9,
        ccoipBFloat16 = 10,
        ccoipFloat = 11,
        ccoipDouble = 12,
    };

    enum ccoip_device_type_t {
        ccoipDeviceCpu = 0,
        ccoipDeviceCuda = 1,
    };
    enum ccoip_hash_type_t {
        ccoipHashSimple = 0,
        ccoipHashCrc32 = 1,
    };

    enum ccoip_reduce_op_t {
        ccoipOpSet = 0,
        ccoipOpSum = 1,
        ccoipOpAvg = 2,
        ccoipOpProd = 3,
        ccoipOpMax = 4,
        ccoipOpMin = 5
    };

    enum ccoip_quantization_algorithm_t {
        ccoipQuantizationNone = 0,
        ccoipQuantizationMinMax = 1,
        ccoipQuantizationZeroPointScale = 2,
    };

    [[nodiscard]] inline size_t ccoip_data_type_size(const ccoip_data_type_t datatype) {
        switch (datatype) {
            case ccoipUint4:
                throw std::runtime_error("ccoipUint4 does not have a byte size");
            case ccoipUint8:
            case ccoipInt8:
                return 1;
            case ccoipUint16:
            case ccoipInt16:
            case ccoipFloat16:
            case ccoipBFloat16:
                return 2;
            case ccoipUint32:
            case ccoipInt32:
            case ccoipFloat:
                return 4;
            case ccoipUint64:
            case ccoipInt64:
            case ccoipDouble:
                return 8;
        }
        return 0;
    }

    [[nodiscard]] inline size_t ccoip_data_type_size_bits(const ccoip_data_type_t datatype) {
        switch (datatype) {
            case ccoipUint4:
                return 4;
            case ccoipUint8:
            case ccoipInt8:
                return 8;
            case ccoipUint16:
            case ccoipInt16:
            case ccoipFloat16:
            case ccoipBFloat16:
                return 16;
            case ccoipUint32:
            case ccoipInt32:
            case ccoipFloat:
                return 32;
            case ccoipUint64:
            case ccoipInt64:
            case ccoipDouble:
                return 64;
        }
        return 0;
    }

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> > >
    [[nodiscard]]
    constexpr ccoip_data_type_t ccoip_data_type_from_type() {
        if constexpr (std::is_same_v<T, std::int8_t>) {
            return ccoipInt8;
        } else if constexpr (std::is_same_v<T, std::uint8_t>) {
            return ccoipUint8;
        } else if constexpr (std::is_same_v<T, std::int16_t>) {
            return ccoipInt16;
        } else if constexpr (std::is_same_v<T, std::uint16_t>) {
            return ccoipUint16;
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
            return ccoipInt32;
        } else if constexpr (std::is_same_v<T, std::uint32_t>) {
            return ccoipUint32;
        } else if constexpr (std::is_same_v<T, std::int64_t>) {
            return ccoipInt64;
        } else if constexpr (std::is_same_v<T, std::uint64_t>) {
            return ccoipUint64;
        } else if constexpr (std::is_same_v<T, float>) {
            return ccoipFloat;
        } else if constexpr (std::is_same_v<T, double>) {
            return ccoipDouble;
        } else {
            static_assert(std::is_arithmetic_v<T>, "Unsupported type");
        }
        return ccoipUint8;
    }
};


constexpr std::size_t CCOIP_UUID_N_BYTES = 16;
using ccoip_uuid = std::array<uint8_t, CCOIP_UUID_N_BYTES>;

typedef uint8_t boolean;

struct ccoip_uuid_t {
    ccoip_uuid data;

    friend bool operator==(const ccoip_uuid_t &lhs, const ccoip_uuid_t &rhs) {
        return lhs.data == rhs.data;
    }

    friend bool operator!=(const ccoip_uuid_t &lhs, const ccoip_uuid_t &rhs) {
        return !(lhs == rhs);
    }
};

static_assert(sizeof(ccoip_uuid_t) == 16);

// Custom hash specialization for uuid_t
template<>
struct std::hash<ccoip_uuid_t> {
    std::size_t operator()(const ccoip_uuid_t &uuid) const noexcept {
        // Combine the hash of each byte in the UUID array
        std::size_t hash_value = 0;
        for (const auto &byte: uuid.data) {
            hash_value = (hash_value * 31) + byte; // or use a better mixing function if desired
        }
        return hash_value;
    }
};

inline std::string uuid_to_string(const ccoip_uuid_t &uuid) {
    std::string uuid_str;
    uuid_str.reserve(CCOIP_UUID_N_BYTES * 2 + 4);
    for (const auto &byte: uuid.data) {
        char byte_str[3];
        snprintf(byte_str, sizeof(byte_str), "%02X", byte);
        uuid_str += byte_str;
        if (uuid_str.size() == 8 || uuid_str.size() == 13 || uuid_str.size() == 18 || uuid_str.size() == 23) {
            uuid_str += "-";
        }
    }
    return uuid_str;
}
