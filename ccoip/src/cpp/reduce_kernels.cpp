#include "reduce_kernels.hpp"

#include <cassert>
#include <unordered_map>
#include <pccl_log.hpp>
#include <quantize.hpp>
#include <quantize_kernels.hpp>
#include <network_order_utils.hpp>

#include <piquant_utils.hpp>
#include <thread>

#ifndef _MSC_VER
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline __forceinline
#endif


#ifndef _MSC_VER
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif

template<typename T>
struct DeQuantizationMetaDataInternal {
    T min_value;
    T max_value;

    FORCE_INLINE static DeQuantizationMetaDataInternal From(
        const ccoip::internal::quantize::DeQuantizationMetaData &meta_data) {
        using namespace tinysockets;
        DeQuantizationMetaDataInternal result{};

        assert(meta_data.min_value.size() == sizeof(T));
        assert(meta_data.max_value.size() == sizeof(T));

        T min_value = *reinterpret_cast<const T *>(meta_data.min_value.data());
        T max_value = *reinterpret_cast<const T *>(meta_data.max_value.data());

        if constexpr (std::is_integral_v<T>) {
            // Convert from big-endian (network order) to host byte order
            min_value = network_order_utils::network_to_host(min_value);
            max_value = network_order_utils::network_to_host(max_value);
        }
        // floats and doubles are always in host byte order

        result.min_value = min_value;
        result.max_value = max_value;

        return result;
    }
};

struct Set {
private:
    template<typename D, typename S>
    FORCE_INLINE static void apply_NoQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count) {
        std::memcpy(dst, src, count * sizeof(D));
    }

    template<typename D, typename S>
    FORCE_INLINE static void apply_MinMaxQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count, D min_val, D max_val) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            dst[i] = ccoip::internal::quantize::deQuantizeMinMaxScalar(src[i], min_val, max_val);
        }
    }

public:
    template<typename D, typename S, ccoip::ccoip_quantization_algorithm_t quant_algo>
    FORCE_INLINE static void apply(D *RESTRICT dst, const S *RESTRICT src, const size_t count,
                                   const ccoip::internal::quantize::DeQuantizationMetaData &meta_data) {
        if constexpr (quant_algo == ccoip::ccoipQuantizationNone) {
            apply_NoQuant(dst, src, count);
        } else if constexpr (quant_algo == ccoip::ccoipQuantizationMinMax) {
            DeQuantizationMetaDataInternal<D> meta_data_internal = DeQuantizationMetaDataInternal<D>::From(meta_data);
            apply_MinMaxQuant(dst, src, count, meta_data_internal.min_value, meta_data_internal.max_value);
        }
    }
};

struct Sum {
private:
    template<typename D, typename S>
    FORCE_INLINE static void apply_NoQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            dst[i] += src[i];
        }
    }

    template<typename D, typename S>
    FORCE_INLINE static void apply_MinMaxQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count, D min_val, D max_val) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            dst[i] += ccoip::internal::quantize::deQuantizeMinMaxScalar(src[i], min_val, max_val);
        }
    }

public:
    template<typename D, typename S, ccoip::ccoip_quantization_algorithm_t quant_algo>
    FORCE_INLINE static void apply(D *RESTRICT dst, const S *RESTRICT src, const size_t count,
                                   const ccoip::internal::quantize::DeQuantizationMetaData &meta_data) {
        if constexpr (quant_algo == ccoip::ccoipQuantizationNone) {
            apply_NoQuant(dst, src, count);
        } else if constexpr (quant_algo == ccoip::ccoipQuantizationMinMax) {
            DeQuantizationMetaDataInternal<D> meta_data_internal = DeQuantizationMetaDataInternal<D>::From(meta_data);
            apply_MinMaxQuant(dst, src, count, meta_data_internal.min_value, meta_data_internal.max_value);
        }
    }
};

struct Prod {
private:
    template<typename D, typename S>
    FORCE_INLINE static void apply_NoQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            dst[i] *= src[i];
        }
    }

    template<typename D, typename S>
    FORCE_INLINE static void apply_MinMaxQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count, D min_val, D max_val) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            dst[i] *= ccoip::internal::quantize::deQuantizeMinMaxScalar(src[i], min_val, max_val);
        }
    }

public:
    template<typename D, typename S, ccoip::ccoip_quantization_algorithm_t quant_algo>
    FORCE_INLINE static void apply(D *RESTRICT dst, const S *RESTRICT src, const size_t count,
                                   const ccoip::internal::quantize::DeQuantizationMetaData &meta_data) {
        if constexpr (quant_algo == ccoip::ccoipQuantizationNone) {
            apply_NoQuant(dst, src, count);
        } else if constexpr (quant_algo == ccoip::ccoipQuantizationMinMax) {
            DeQuantizationMetaDataInternal<D> meta_data_internal = DeQuantizationMetaDataInternal<D>::From(meta_data);
            apply_MinMaxQuant(dst, src, count, meta_data_internal.min_value, meta_data_internal.max_value);
        }
    }
};

struct Min {
private:
    template<typename D, typename S, typename UpT>
    FORCE_INLINE static void apply_NoQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            UpT temp = std::min<UpT>(static_cast<UpT>(dst[i]),
                                     static_cast<UpT>(src[i]));
            dst[i] = static_cast<D>(temp);
        }
    }

    template<typename D, typename S, typename UpT>
    FORCE_INLINE static void apply_MinMaxQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count, D min_val, D max_val) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            UpT temp = std::min<UpT>(static_cast<UpT>(dst[i]),
                                     static_cast<UpT>(ccoip::internal::quantize::deQuantizeMinMaxScalar(src[i], min_val, max_val)));
            dst[i] = static_cast<D>(temp);
        }
    }

public:
    template<typename D, typename S, ccoip::ccoip_quantization_algorithm_t quant_algo>
    FORCE_INLINE static void apply(D *RESTRICT dst, const S *RESTRICT src, const size_t count,
                                   const ccoip::internal::quantize::DeQuantizationMetaData &meta_data) {
        using UpT = std::common_type_t<D, S>;
        if constexpr (quant_algo == ccoip::ccoipQuantizationNone) {
            apply_NoQuant<D, S, UpT>(dst, src, count);
        } else if constexpr (quant_algo == ccoip::ccoipQuantizationMinMax) {
            DeQuantizationMetaDataInternal<D> meta_data_internal = DeQuantizationMetaDataInternal<D>::From(meta_data);
            apply_MinMaxQuant<D, S, UpT>(dst, src, count, meta_data_internal.min_value, meta_data_internal.max_value);
        }
    }
};

struct Max {
private:
    template<typename D, typename S, typename UpT>
    FORCE_INLINE static void apply_NoQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            UpT temp = std::max<UpT>(static_cast<UpT>(dst[i]),
                                     static_cast<UpT>(src[i]));
            dst[i] = static_cast<D>(temp);
        }
    }

    template<typename D, typename S, typename UpT>
    FORCE_INLINE static void apply_MinMaxQuant(D *RESTRICT dst, const S *RESTRICT src, const size_t count, D min_val, D max_val) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            UpT temp = std::max<UpT>(static_cast<UpT>(dst[i]),
                                     static_cast<UpT>(ccoip::internal::quantize::deQuantizeMinMaxScalar(src[i], min_val, max_val)));
            dst[i] = static_cast<D>(temp);
        }
    }
public:
    template<typename D, typename S, ccoip::ccoip_quantization_algorithm_t quant_algo>
    FORCE_INLINE static void apply(D *RESTRICT dst, const S *RESTRICT src, const size_t count,
                                   const ccoip::internal::quantize::DeQuantizationMetaData &meta_data) {
        using UpT = std::common_type_t<D, S>;
        if constexpr (quant_algo == ccoip::ccoipQuantizationNone) {
            apply_NoQuant<D, S, UpT>(dst, src, count);
        } else if constexpr (quant_algo == ccoip::ccoipQuantizationMinMax) {
            DeQuantizationMetaDataInternal<D> meta_data_internal = DeQuantizationMetaDataInternal<D>::From(meta_data);
            apply_MinMaxQuant<D, S, UpT>(dst, src, count, meta_data_internal.min_value, meta_data_internal.max_value);
        }
    }
};

struct DtypeQuantVariant {
    const ccoip::ccoip_data_type_t dst_type;
    const ccoip::ccoip_data_type_t src_type;
    const ccoip::ccoip_quantization_algorithm_t quantization_algorithm;

    friend bool operator==(const DtypeQuantVariant &lhs, const DtypeQuantVariant &rhs) {
        return lhs.dst_type == rhs.dst_type
               && lhs.src_type == rhs.src_type
               && lhs.quantization_algorithm == rhs.quantization_algorithm;
    }

    friend bool operator!=(const DtypeQuantVariant &lhs, const DtypeQuantVariant &rhs) {
        return !(lhs == rhs);
    }
};

template<>
struct std::hash<DtypeQuantVariant> {
    std::size_t operator()(const DtypeQuantVariant &pair) const noexcept {
        return std::hash<ccoip::ccoip_data_type_t>()(pair.dst_type)
               ^ std::hash<ccoip::ccoip_data_type_t>()(pair.src_type)
               ^ std::hash<ccoip::ccoip_quantization_algorithm_t>()(pair.quantization_algorithm);
    }
};

typedef void (*reduce_fn_t)(void *dst,
                            const void *src,
                            size_t count,
                            const ccoip::internal::quantize::DeQuantizationMetaData &meta_data);

template<typename S, typename D>
static constexpr reduce_fn_t make_redfn(void (*func)(D *, const S *, size_t,
                                                     const ccoip::internal::quantize::DeQuantizationMetaData &)) {
    return reinterpret_cast<reduce_fn_t>(func);
}


template<typename Op>
FORCE_INLINE void doReduceDataType(const std::span<std::byte> &dst,
                                   const std::span<const std::byte> &src,
                                   const ccoip::ccoip_data_type_t dst_type,
                                   const ccoip::ccoip_data_type_t src_type,
                                   const ccoip::ccoip_quantization_algorithm_t quantization_algorithm,
                                   const ccoip::internal::quantize::DeQuantizationMetaData &meta_data) {
    const size_t dst_element_size = ccoip_data_type_size(dst_type);
    const size_t src_element_size = ccoip_data_type_size(src_type);

    assert(dst.size_bytes() % dst_element_size == 0);
    assert(src.size_bytes() % src_element_size == 0);

    const size_t num_elements = dst.size_bytes() / dst_element_size;
    assert(num_elements == src.size_bytes() / src_element_size);

    static std::unordered_map<DtypeQuantVariant, reduce_fn_t> reduce_functions = {
        // int8 input with all signed equal or higher bit accumulation types [ int8_t <= d <= int64_t ] (no quantization)
        // signed input type means unsigned accumulation type is not allowed.
        {{ccoip::ccoipInt8, ccoip::ccoipInt8, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int8_t, int8_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipInt16, ccoip::ccoipInt8, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int16_t, int8_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipInt32, ccoip::ccoipInt8, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int32_t, int8_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipInt64, ccoip::ccoipInt8, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int64_t, int8_t, ccoip::ccoipQuantizationNone>)},

        // int8 input with quantization is not meaningful for min max quantization as signed integers are not guaranteed to be in the range [0, max_int]

        // uint8 input with all unsigned equal or higher bit accumulation types [ uint8_t <= d <= uint64_t ] (no quantization)
        // unsigned input type means signed accumulation type is not meaningful as the sum of positive integers can never be negative unless an overflow occurs.
        {{ccoip::ccoipUint8, ccoip::ccoipUint8, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint8_t, uint8_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipUint16, ccoip::ccoipUint8, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint16_t, uint8_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipUint32, ccoip::ccoipUint8, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint32_t, uint8_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipUint64, ccoip::ccoipUint8, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint64_t, uint8_t, ccoip::ccoipQuantizationNone>)},

        // uint8 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
        {{ccoip::ccoipFloat, ccoip::ccoipUint8, ccoip::ccoipQuantizationMinMax}, make_redfn(&Op::template apply<float, uint8_t, ccoip::ccoipQuantizationMinMax>)},
        {{ccoip::ccoipDouble, ccoip::ccoipUint8, ccoip::ccoipQuantizationMinMax}, make_redfn(&Op::template apply<double, uint8_t, ccoip::ccoipQuantizationMinMax>)},

        // int16 input with all signed equal or higher bit accumulation types [ int16_t <= d <= int64_t ] (no quantization)
        {{ccoip::ccoipInt16, ccoip::ccoipInt16, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int16_t, int16_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipInt32, ccoip::ccoipInt16, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int32_t, int16_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipInt64, ccoip::ccoipInt16, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int64_t, int16_t, ccoip::ccoipQuantizationNone>)},

        // int16 input with quantization is not meaningful for min max quantization as signed integers are not guaranteed to be in the range [0, max_int]

        // uint16 input with all unsigned equal or higher bit accumulation types [ uint16_t <= d <= uint64_t ] (no quantization)
        {{ccoip::ccoipUint16, ccoip::ccoipUint16, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint16_t, uint16_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipUint32, ccoip::ccoipUint16, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint32_t, uint16_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipUint64, ccoip::ccoipUint16, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint64_t, uint16_t, ccoip::ccoipQuantizationNone>)},

        // uint16 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
        {{ccoip::ccoipFloat, ccoip::ccoipUint16, ccoip::ccoipQuantizationMinMax}, make_redfn(&Op::template apply<float, uint16_t, ccoip::ccoipQuantizationMinMax>)},
        {{ccoip::ccoipDouble, ccoip::ccoipUint16, ccoip::ccoipQuantizationMinMax}, make_redfn(&Op::template apply<double, uint16_t, ccoip::ccoipQuantizationMinMax>)},

        // int32 input with all signed equal or higher bit accumulation types [ int32_t <= d <= int64_t ] (no quantization)
        {{ccoip::ccoipInt32, ccoip::ccoipInt32, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int32_t, int32_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipInt64, ccoip::ccoipInt32, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int64_t, int32_t, ccoip::ccoipQuantizationNone>)},

        // int32 input with quantization is not meaningful for min max quantization as signed integers are not guaranteed to be in the range [0, max_int]

        // uint32 input with all unsigned equal or higher bit accumulation types [ uint32_t <= d <= uint64_t ] (no quantization)
        {{ccoip::ccoipUint32, ccoip::ccoipUint32, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint32_t, uint32_t, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipUint64, ccoip::ccoipUint32, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint64_t, uint32_t, ccoip::ccoipQuantizationNone>)},

        // uint32 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
        {{ccoip::ccoipFloat, ccoip::ccoipUint32, ccoip::ccoipQuantizationMinMax}, make_redfn(&Op::template apply<float, uint32_t, ccoip::ccoipQuantizationMinMax>)},
        {{ccoip::ccoipDouble, ccoip::ccoipUint32, ccoip::ccoipQuantizationMinMax}, make_redfn(&Op::template apply<double, uint32_t, ccoip::ccoipQuantizationMinMax>)},

        // int64 input with all signed equal or higher bit accumulation types [ int64_t <= d ] (no quantization)
        {{ccoip::ccoipInt64, ccoip::ccoipInt64, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<int64_t, int64_t, ccoip::ccoipQuantizationNone>)},

        // int64 input with quantization is not meaningful for min max quantization as signed integers are not guaranteed to be in the range [0, max_int]

        // uint64 input with all unsigned equal or higher bit accumulation types [ uint64_t <= d ] (no quantization)
        {{ccoip::ccoipUint64, ccoip::ccoipUint64, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<uint64_t, uint64_t, ccoip::ccoipQuantizationNone>)},

        // uint64 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
        {{ccoip::ccoipFloat, ccoip::ccoipUint64, ccoip::ccoipQuantizationMinMax}, make_redfn(&Op::template apply<float, uint64_t, ccoip::ccoipQuantizationMinMax>)},
        {{ccoip::ccoipDouble, ccoip::ccoipUint64, ccoip::ccoipQuantizationMinMax}, make_redfn(&Op::template apply<double, uint64_t, ccoip::ccoipQuantizationMinMax>)},

        // float input with all equivalent or higher precision accumulation types [ float <= d <= double ] (no quantization)
        {{ccoip::ccoipFloat, ccoip::ccoipFloat, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<float, float, ccoip::ccoipQuantizationNone>)},
        {{ccoip::ccoipDouble, ccoip::ccoipFloat, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<double, float, ccoip::ccoipQuantizationNone>)},

        // float input with quantization does not make sense; quantization means input is de-quantized. For that the input must be a quantized type (i.e. uint)

        // double input with all equivalent or higher precision accumulation types [ double = d ] (no quantization)
        {{ccoip::ccoipDouble, ccoip::ccoipDouble, ccoip::ccoipQuantizationNone}, make_redfn(&Op::template apply<double, double, ccoip::ccoipQuantizationNone>)},

        // double input with quantization does not make sense; quantization means input is de-quantized. For that the input must be a quantized type (i.e. uint)
    };

    const DtypeQuantVariant key{dst_type, src_type, quantization_algorithm};
    const auto it = reduce_functions.find(key);
    if (it == reduce_functions.end()) {
        LOG(FATAL) << "Unsupported data type combination: dst_type=" << dst_type
                 << ", src_type=" << src_type
                 << ", quantization_algorithm=" << quantization_algorithm
                << "; This likely means that nobody in their right minds would request this data type combination because there is no meaningful use case for it.";
        return;
    }
    const reduce_fn_t fn = it->second;

    void *dst_ptr = dst.data();
    const void *src_ptr = src.data();
    fn(dst_ptr, src_ptr, num_elements, meta_data);
}

void performReduction(const std::span<std::byte> &dst,
                      const std::span<const std::byte> &src,
                      const ccoip::ccoip_data_type_t dst_type,
                      const ccoip::ccoip_data_type_t src_type,
                      const ccoip::ccoip_quantization_algorithm_t quantization_algorithm,
                      const ccoip::ccoip_reduce_op_t op,
                      const ccoip::internal::quantize::DeQuantizationMetaData &meta_data) {
    if (src_type == dst_type) {
        switch (op) {
            case ccoip::ccoipOpSet:
                doReduceDataType<Set>(dst, src, dst_type, src_type, quantization_algorithm, meta_data);
            break;
            case ccoip::ccoipOpSum:
            case ccoip::ccoipOpAvg:
                doReduceDataType<Sum>(dst, src, dst_type, src_type, quantization_algorithm, meta_data);
            break;
            case ccoip::ccoipOpProd:
                doReduceDataType<Prod>(dst, src, dst_type, src_type, quantization_algorithm, meta_data);
            break;
            case ccoip::ccoipOpMax:
                doReduceDataType<Max>(dst, src, dst_type, src_type, quantization_algorithm, meta_data);
            break;
            case ccoip::ccoipOpMin:
                doReduceDataType<Min>(dst, src, dst_type, src_type, quantization_algorithm, meta_data);
            break;
            default:
                LOG(BUG) << "Unsupported reduce operation: " << op;
            break;
        }
        return;
    }

    piquant::reduce_op reduce_op {};
    switch (op) {
        case ccoip::ccoipOpSet:
            reduce_op = piquant::reduce_op::set;
        break;
        case ccoip::ccoipOpSum:
        case ccoip::ccoipOpAvg:
            reduce_op = piquant::reduce_op::add;
        break;
        default:
            LOG(BUG) << "Unsupported reduce operation: " << op;
            break;
    }

    ccoip::internal::get_quant_ctx().dequantize(
        src,
        ccoip::internal::get_piquant_dtype(src_type),
        dst,
        ccoip::internal::get_piquant_dtype(dst_type),
        meta_data.scaleAs<float>(),
        meta_data.zeroPointAs<std::int32_t>(),
        reduce_op
    );
}

void ccoip::internal::reduce::performReduction(const std::span<std::byte> &dst,
                                               const std::span<const std::byte> &src,
                                               const ccoip_data_type_t dst_type,
                                               const ccoip_data_type_t src_type,
                                               const ccoip_quantization_algorithm_t quantization_algorithm,
                                               const ccoip_reduce_op_t op,
                                               const quantize::DeQuantizationMetaData &meta_data) {
    return ::performReduction(dst, src, dst_type, src_type, quantization_algorithm, op, meta_data);
}

template<typename T>
FORCE_INLINE static void performAvgFinalization(T *dst, const size_t count, const size_t world_size) {
    const T value = static_cast<T>(world_size);
    #pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        dst[i] = dst[i] / value;
    }
}

void ccoip::internal::reduce::performReduceFinalization(const std::span<std::byte> &dst,
                                                        const ccoip_data_type_t accumulator_type,
                                                        const size_t world_size, const ccoip_reduce_op_t op) {
    switch (op) {
        case ccoipOpAvg: {
            switch (accumulator_type) {
                case ccoipInt8: {
                    const size_t count = dst.size_bytes() / sizeof(int8_t);
                    performAvgFinalization(reinterpret_cast<int8_t *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipUint8: {
                    const size_t count = dst.size_bytes() / sizeof(uint8_t);
                    performAvgFinalization(reinterpret_cast<uint8_t *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipInt16: {
                    const size_t count = dst.size_bytes() / sizeof(int16_t);
                    performAvgFinalization(reinterpret_cast<int16_t *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipUint16: {
                    const size_t count = dst.size_bytes() / sizeof(uint16_t);
                    performAvgFinalization(reinterpret_cast<uint16_t *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipInt32: {
                    const size_t count = dst.size_bytes() / sizeof(int32_t);
                    performAvgFinalization(reinterpret_cast<int32_t *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipUint32: {
                    const size_t count = dst.size_bytes() / sizeof(uint32_t);
                    performAvgFinalization(reinterpret_cast<uint32_t *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipInt64: {
                    const size_t count = dst.size_bytes() / sizeof(int64_t);
                    performAvgFinalization(reinterpret_cast<int64_t *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipUint64: {
                    const size_t count = dst.size_bytes() / sizeof(uint64_t);
                    performAvgFinalization(reinterpret_cast<uint64_t *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipFloat: {
                    const size_t count = dst.size_bytes() / sizeof(float);
                    performAvgFinalization(reinterpret_cast<float *>(dst.data()), count, world_size);
                    break;
                }
                case ccoipDouble: {
                    const size_t count = dst.size_bytes() / sizeof(double);
                    performAvgFinalization(reinterpret_cast<double *>(dst.data()), count, world_size);
                    break;
                }
                default: {
                    LOG(ERR) << "Unsupported data type: " << accumulator_type;
                    break;
                }
            }
        }
        default: break;
    }
}
