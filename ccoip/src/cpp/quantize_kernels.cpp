#include "quantize_kernels.hpp"

#include <unordered_map>
#include <pccl_log.hpp>

#ifndef _MSC_VER
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline __forceinline
#endif

template<typename T>
struct minmax_t {
    T min;
    T max;
};

#ifndef _MSC_VER
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif

template<typename T>
FORCE_INLINE minmax_t<T> findMinAndMax(const T *RESTRICT src, const size_t count) {
    minmax_t<T> result{};
    result.min = std::numeric_limits<T>::max();
    result.max = std::numeric_limits<T>::min();
    for (size_t i = 0; i < count; ++i) {
        result.min = std::min(result.min, src[i]);
        result.max = std::max(result.max, src[i]);
    }
    return result;
}

using namespace ccoip::internal::quantize;

template<typename O, typename I, typename = std::enable_if_t<
             std::is_integral_v<O> && std::is_unsigned_v<O> && std::is_floating_point_v<I>>>
[[nodiscard]]
FORCE_INLINE static DeQuantizationMetaData minMaxQuant(O *RESTRICT dst, const I *RESTRICT src,
                                                       const size_t count) {
    minmax_t<I> minmax = findMinAndMax<I>(src, count);
    const auto dif = static_cast<double>(minmax.max - minmax.min);
    if (dif != 0.0) {
        const auto dif_inv = 1.0 / dif;
        for (size_t i = 0; i < count; ++i) {
            const double relative = (static_cast<double>(src[i]) - static_cast<double>(minmax.min)) * dif_inv;
            dst[i] = static_cast<O>(relative * static_cast<double>(std::numeric_limits<O>::max()));
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            const auto relative = static_cast<double>(src[i] - minmax.min);
            dst[i] = static_cast<O>(relative * static_cast<double>(std::numeric_limits<O>::max()));
        }
    }
    return DeQuantizationMetaData::MakeMinMax(minmax.min, minmax.max);
}


template<typename O, typename I, typename = std::enable_if_t<
             std::is_integral_v<O> && std::is_unsigned_v<O> && std::is_floating_point_v<I>>>
FORCE_INLINE static void minMaxQuantAndDequant(I *RESTRICT dst, const I *RESTRICT src,
                                               const size_t count) {
    minmax_t<I> minmax = findMinAndMax<I>(src, count);
    const auto dif = static_cast<double>(minmax.max - minmax.min);
    if (dif != 0.0) {
        const double dif_inv = 1.0 / dif;
        for (size_t i = 0; i < count; ++i) {
            const double relative = (static_cast<double>(src[i]) - static_cast<double>(minmax.min)) * dif_inv;
            O quantized_value = static_cast<O>(relative * static_cast<double>(std::numeric_limits<O>::max()));
            dst[i] = deQuantizeMinMaxScalar(quantized_value, minmax.min, minmax.max);
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            const auto relative = static_cast<double>(src[i] - minmax.min);
            O quantized_value = static_cast<O>(relative * static_cast<double>(std::numeric_limits<O>::max()));
            dst[i] = deQuantizeMinMaxScalar(quantized_value, minmax.min, minmax.max);
        }
    }
}

struct DtypeVariant {
    ccoip::ccoip_data_type_t dst_type;
    ccoip::ccoip_data_type_t src_type;

    friend bool operator==(const DtypeVariant &lhs, const DtypeVariant &rhs) {
        return lhs.dst_type == rhs.dst_type
               && lhs.src_type == rhs.src_type;
    }

    friend bool operator!=(const DtypeVariant &lhs, const DtypeVariant &rhs) {
        return !(lhs == rhs);
    }
};

template<>
struct std::hash<DtypeVariant> {
    std::size_t operator()(const DtypeVariant &pair) const noexcept {
        return std::hash<ccoip::ccoip_data_type_t>()(pair.dst_type)
               ^ std::hash<ccoip::ccoip_data_type_t>()(pair.src_type);
    }
};

typedef DeQuantizationMetaData (*quant_fn_t)(void *dst, const void *src, size_t count);

template<typename O, typename I, typename = std::enable_if_t<
             std::is_integral_v<O> && std::is_unsigned_v<O> && std::is_floating_point_v<I>>>
static constexpr quant_fn_t make_quantfn(DeQuantizationMetaData (*func)(O *, const I *, size_t)) {
    return reinterpret_cast<quant_fn_t>(func);
}


DeQuantizationMetaData ccoip::internal::quantize::performMinMaxQuantization(
        const std::span<std::byte> &dst_span,
        const std::span<const std::byte> &src_bytes,
        const ccoip_data_type_t quantized_type, const ccoip_data_type_t data_type) {
    std::unordered_map<DtypeVariant, quant_fn_t> minmax_map = {
            // uint8 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
            {{ccoipFloat, ccoipUint8}, make_quantfn(&minMaxQuant<uint8_t, float>)},
            {{ccoipDouble, ccoipUint8}, make_quantfn(&minMaxQuant<uint8_t, double>)},

            // uint16 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
            {{ccoipFloat, ccoipUint16}, make_quantfn(&minMaxQuant<uint16_t, float>)},
            {{ccoipDouble, ccoipUint16}, make_quantfn(&minMaxQuant<uint16_t, double>)},

            // uint32 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
            {{ccoipFloat, ccoipUint32}, make_quantfn(&minMaxQuant<uint32_t, float>)},
            {{ccoipDouble, ccoipUint32}, make_quantfn(&minMaxQuant<uint32_t, double>)},

            // uint64 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
            {{ccoipFloat, ccoipUint64}, make_quantfn(&minMaxQuant<uint64_t, float>)},
            {{ccoipDouble, ccoipUint64}, make_quantfn(&minMaxQuant<uint64_t, double>)},
    };

    const DtypeVariant key{data_type, quantized_type};
    const auto it = minmax_map.find(key);
    if (it == minmax_map.end()) {
        LOG(FATAL) << "Unsupported data type combination: dst_type=" << key.dst_type
                << ", src_type=" << key.src_type
                << "; This likely means that nobody in their right minds would request this data type combination because there is no meaningful use case for it.";
        return DeQuantizationMetaData{};
    }

    const quant_fn_t fn = it->second;
    const size_t count = src_bytes.size_bytes() / ccoip_data_type_size(data_type);
    if (dst_span.size_bytes() / ccoip_data_type_size(quantized_type) != count) {
        LOG(FATAL) << "Mismatched sizes for quantization: dst_size=" << dst_span.size_bytes()
                << ", src_size=" << src_bytes.size_bytes()
                << ", count=" << count;
        return DeQuantizationMetaData{};
    }
    return fn(dst_span.data(), src_bytes.data(), count);
}


typedef void (*fused_quant_and_dequant_fn_t)(void *dst, const void *src, size_t count);

template<typename O, typename I, typename = std::enable_if_t<
             std::is_integral_v<O> && std::is_unsigned_v<O> && std::is_floating_point_v<I>>>
static constexpr fused_quant_and_dequant_fn_t make_fused_quant_and_dequant_fn(
        void (*func)(I *, const I *, size_t)) {
    return reinterpret_cast<fused_quant_and_dequant_fn_t>(func);
}

void ccoip::internal::quantize::performMinMaxQuantizationAndDequantization(const std::span<std::byte> &dst_span,
                                                                           const std::span<const std::byte> &src_bytes,
                                                                           const ccoip_data_type_t quantized_type,
                                                                           const ccoip_data_type_t data_type) {
    std::unordered_map<DtypeVariant, fused_quant_and_dequant_fn_t> minmax_map = {
            // uint8 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
            {{ccoipFloat, ccoipUint8},
             make_fused_quant_and_dequant_fn<uint8_t>(&minMaxQuantAndDequant<uint8_t, float>)},
            {{ccoipDouble, ccoipUint8},
             make_fused_quant_and_dequant_fn<uint8_t>(&minMaxQuantAndDequant<uint8_t, double>)},

            // uint16 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
            {{ccoipFloat, ccoipUint16},
             make_fused_quant_and_dequant_fn<uint16_t>(&minMaxQuantAndDequant<uint16_t, float>)},
            {{ccoipDouble, ccoipUint16},
             make_fused_quant_and_dequant_fn<uint16_t>(&minMaxQuantAndDequant<uint16_t, double>)},

            // uint32 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
            {{ccoipFloat, ccoipUint32},
             make_fused_quant_and_dequant_fn<uint32_t>(&minMaxQuantAndDequant<uint32_t, float>)},
            {{ccoipDouble, ccoipUint32},
             make_fused_quant_and_dequant_fn<uint32_t>(&minMaxQuantAndDequant<uint32_t, double>)},

            // uint64 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
            {{ccoipFloat, ccoipUint64},
             make_fused_quant_and_dequant_fn<uint64_t>(&minMaxQuantAndDequant<uint64_t, float>)},
            {{ccoipDouble, ccoipUint64},
             make_fused_quant_and_dequant_fn<uint64_t>(&minMaxQuantAndDequant<uint64_t, double>)},
    };

    const DtypeVariant key{data_type, quantized_type};
    const auto it = minmax_map.find(key);
    if (it == minmax_map.end()) {
        LOG(FATAL) << "Unsupported data type combination: dst_type=" << key.dst_type
                << ", src_type=" << key.src_type
                << "; This likely means that nobody in their right minds would request this data type combination because there is no meaningful use case for it.";
        return;
    }

    const fused_quant_and_dequant_fn_t fn = it->second;
    const size_t count = src_bytes.size_bytes() / ccoip_data_type_size(data_type);

    // don't use quantized_type here! Input and output types are the same.
    if (dst_span.size_bytes() / ccoip_data_type_size(data_type) != count) {
        LOG(FATAL) << "Mismatched sizes for quantization: dst_size=" << dst_span.size_bytes()
                << ", src_size=" << src_bytes.size_bytes()
                << ", count=" << count;
        return;
    }

    fn(dst_span.data(), src_bytes.data(), count);
}
