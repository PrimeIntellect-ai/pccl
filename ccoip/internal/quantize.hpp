#pragma once

#include <cassert>
#include <span>
#include <type_traits>
#include <vector>
#include <cstring>
#include <ccoip_types.hpp>
#include <network_order_utils.hpp>
#include <pccl_log.hpp>

namespace ccoip::internal::quantize {

    enum class DeQuantizationMetaType {
        MIN_MAX,
        ZERO_POINT_SCALE
    };
    /// Contains meta-data needed for de-quantization.
    struct DeQuantizationMetaData {
        /// Type of metadata
        /// Determines whether min_value & max_value or zero_point & scale are used.
        DeQuantizationMetaType meta_type;

        /// Defines the data type for how to interpret @code min_value@endcode and @code max_value@endcode.
        ccoip_data_type_t min_max_data_type;

        /// Defines the data type for how to interpret @code zero_point@endcode and @code scale@endcode.
        ccoip_data_type_t zero_point_type;
        ccoip_data_type_t scale_type;

        /// Minimum value summary statistic
        std::vector<uint8_t> min_value;

        /// Maximum value summary statistic
        std::vector<uint8_t> max_value;

        /// Zero point
        std::vector<uint8_t> zero_point;

        /// Scale factor
        std::vector<uint8_t> scale;

        template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        [[nodiscard]]
        static DeQuantizationMetaData MakeMinMax(T min_value, T max_value) {
            using namespace tinysockets;
            constexpr ccoip_data_type_t data_type = ccoip_data_type_from_type<T>();

            DeQuantizationMetaData meta_data{};
            meta_data.meta_type = DeQuantizationMetaType::MIN_MAX;
            meta_data.min_max_data_type = data_type;

            const size_t data_type_size = ccoip_data_type_size(data_type);
            assert(data_type_size == sizeof(T));

            if constexpr (std::is_integral_v<T>) {
                // if is integer, convert to network byte order
                min_value = network_order_utils::host_to_network(min_value);
                max_value = network_order_utils::host_to_network(max_value);
            }

            // floats and doubles are always in host byte order

            meta_data.min_value.resize(data_type_size);
            meta_data.max_value.resize(data_type_size);

            std::memcpy(meta_data.min_value.data(), &min_value, data_type_size);
            std::memcpy(meta_data.max_value.data(), &max_value, data_type_size);

            return meta_data;
        }

        template<typename Z, typename S, typename = std::enable_if_t<std::is_integral_v<Z> && std::is_floating_point_v<S>>>
        [[nodiscard]]
        static DeQuantizationMetaData MakeZeroPointScale(Z zero_point, S scale) {
            using namespace tinysockets;
            constexpr ccoip_data_type_t zero_point_type = ccoip_data_type_from_type<Z>();
            constexpr ccoip_data_type_t scale_type = ccoip_data_type_from_type<S>();

            DeQuantizationMetaData meta_data{};
            meta_data.meta_type = DeQuantizationMetaType::ZERO_POINT_SCALE;
            meta_data.zero_point_type = zero_point_type;
            meta_data.scale_type = scale_type;

            const size_t zero_point_type_size = ccoip_data_type_size(zero_point_type);
            const size_t scale_type_size = ccoip_data_type_size(scale_type);
            assert(zero_point_type_size == sizeof(Z));
            assert(scale_type_size == sizeof(S));

            if constexpr (std::is_integral_v<Z>) {
                // if is integer, convert to network byte order
                zero_point = network_order_utils::host_to_network(zero_point);
            }

            if constexpr (std::is_integral_v<S>) {
                // if is integer, convert to network byte order
                scale = network_order_utils::host_to_network(scale);
            }

            meta_data.scale_type = ccoip_data_type_from_type<S>();
            meta_data.zero_point_type = ccoip_data_type_from_type<Z>();

            // floats and doubles are always in host byte order

            meta_data.zero_point.resize(zero_point_type_size);
            meta_data.scale.resize(scale_type_size);

            std::memcpy(meta_data.zero_point.data(), &zero_point, zero_point_type_size);
            std::memcpy(meta_data.scale.data(), &scale, scale_type_size);

            return meta_data;
        }

        template <typename T> requires std::is_floating_point_v<T>
        [[nodiscard]] T scaleAs() const {
            switch (scale_type) {
                case ccoipFloat: {
                    assert(scale.size() == sizeof(float));
                    return static_cast<T>(*reinterpret_cast<const float *>(scale.data()));
                }
                case ccoipDouble: {
                    assert(scale.size() == sizeof(double));
                    return static_cast<T>(*reinterpret_cast<const double *>(scale.data()));
                }
                default: LOG(BUG) << "Unsupported scale type: " << scale_type; std::abort();
            }
        }

        template <typename T> requires std::is_integral_v<T>
        [[nodiscard]] T zeroPointAs() const {
            switch (zero_point_type) {
                case ccoipInt8: {
                    assert(zero_point.size() == sizeof(std::int8_t));
                    return static_cast<T>(*reinterpret_cast<const std::int8_t *>(zero_point.data()));
                }
                case ccoipUint8: {
                    assert(zero_point.size() == sizeof(std::uint8_t));
                    return static_cast<T>(*reinterpret_cast<const std::uint8_t *>(zero_point.data()));
                }
                case ccoipInt16: {
                    assert(zero_point.size() == sizeof(std::int16_t));
                    return static_cast<T>(*reinterpret_cast<const std::int16_t *>(zero_point.data()));
                }
                case ccoipUint16: {
                    assert(zero_point.size() == sizeof(std::uint16_t));
                    return static_cast<T>(*reinterpret_cast<const std::uint16_t *>(zero_point.data()));
                }
                case ccoipInt32: {
                    assert(zero_point.size() == sizeof(std::int32_t));
                    return static_cast<T>(*reinterpret_cast<const std::int32_t *>(zero_point.data()));
                }
                case ccoipUint32: {
                    assert(zero_point.size() == sizeof(std::uint32_t));
                    return static_cast<T>(*reinterpret_cast<const std::uint32_t *>(zero_point.data()));
                }
                case ccoipInt64: {
                    assert(zero_point.size() == sizeof(std::int64_t));
                    return static_cast<T>(*reinterpret_cast<const std::int64_t *>(zero_point.data()));
                }
                case ccoipUint64: {
                    assert(zero_point.size() == sizeof(std::uint64_t));
                    return static_cast<T>(*reinterpret_cast<const std::uint64_t *>(zero_point.data()));
                }
                default: LOG(BUG) << "Unsupported zero point type: " << zero_point_type;std::abort();
            }
        }

    };

    [[nodiscard]] DeQuantizationMetaData performQuantization(const std::span<std::byte> &dst_span,
                                                             const std::span<const std::byte> &src_span,
                                                             ccoip_quantization_algorithm_t quantization_algorithm,
                                                             ccoip_data_type_t quantized_type,
                                                             ccoip_data_type_t data_type);

    // This method is needed because we need a fast way to pretend we quantized and de-quantized.
    // The reason for this is that to ensure the result is the same for all peers, we need to
    // pretend the peer that has access to the original un-quantized data also just receives
    // the same quantized data as other peers. For this reason, we need to quantize and de-quantize locally.
    // However, we want to just do it fused and fast, hence this method.
    void performQuantizationAndDequantization(const std::span<std::byte> &dst_span,
                                              const std::span<const std::byte> &src_span,
                                              ccoip_quantization_algorithm_t quantization_algorithm,
                                              ccoip_data_type_t quantized_type,
                                              ccoip_data_type_t data_type);
};
