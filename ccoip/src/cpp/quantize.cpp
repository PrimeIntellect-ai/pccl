#include "quantize.hpp"

#include <ccoip_types.hpp>
#include <pccl_log.hpp>
#include <quantize_kernels.hpp>
#include <span>
#include <piquant_utils.hpp>


ccoip::internal::quantize::DeQuantizationMetaData ccoip::internal::quantize::performQuantization(
    const std::span<std::byte> &dst_span, const std::span<const std::byte> &src_span,
    const ccoip_quantization_algorithm_t quantization_algorithm,
    const ccoip_data_type_t quantized_type, const ccoip_data_type_t data_type) {
    std::optional<piquant::dtype> quant_type = get_piquant_dtype(quantized_type);
    std::optional<piquant::dtype> dequant_type = get_piquant_dtype(data_type);
    if (!quant_type.has_value() || !dequant_type.has_value()) {
        LOG(BUG) << "Unsupported quantized type: " << quantized_type;
        return {};
    }
    switch (quantization_algorithm) {
        case ccoipQuantizationNone: {
            LOG(BUG) << "performQuantization should never be called with ccoipQuantizationNone.";
            return {};
        }
        case ccoipQuantizationMinMax: {
            return performMinMaxQuantization(dst_span, src_span, quantized_type, data_type);
        }
        case ccoipQuantizationZeroPointScale: {
            std::pair<float, std::int64_t> quant_params{};
            std::optional<piquant::dtype> quant_type = get_piquant_dtype(quantized_type);
            std::optional<piquant::dtype> dequant_type = get_piquant_dtype(data_type);
            if (!quant_type.has_value() || !dequant_type.has_value()) {
                LOG(BUG) << "Unsupported quantized type: " << quantized_type;
                return {};
            }
            switch (data_type) {
                case ccoipFloat:
                    quant_params = get_quant_ctx().compute_quant_config_from_data(
                        std::span{
                            reinterpret_cast<const piquant::fp32_t*>(src_span.data()), src_span.size_bytes() / sizeof(piquant::fp32_t)
                        },
                        *quant_type
                    );
                    break;
                case ccoipBFloat16:
                    quant_params = get_quant_ctx().compute_quant_config_from_data(
                        std::span{
                            reinterpret_cast<const piquant::bfp16_t*>(src_span.data()), src_span.size_bytes() / sizeof(piquant::bfp16_t)
                        },
                        *quant_type
                    );
                    break;
                default: {
                    LOG(BUG) << "Unsupported data type for quantization: " << data_type;
                    return {};
                }
            }
            auto [scale, zp] = quant_params;
            get_quant_ctx().quantize(
                src_span,
                *dequant_type,
                dst_span,
                *quant_type,
                scale,
                zp,
                piquant::round_mode::nearest
            );
            return DeQuantizationMetaData::MakeZeroPointScale(zp, scale);
        }
        default: {
            LOG(BUG) << "Unsupported quantization algorithm: " << quantization_algorithm;
            return {};
        }
    }
}

void ccoip::internal::quantize::performQuantizationAndDequantization(const std::span<std::byte> &dst_span,
                                                                     const std::span<const std::byte> &src_span,
                                                                     const ccoip_quantization_algorithm_t
                                                                     quantization_algorithm,
                                                                     const ccoip_data_type_t quantized_type,
                                                                     const ccoip_data_type_t data_type) {
    switch (quantization_algorithm) {
        case ccoipQuantizationNone: {
            LOG(BUG) << "performQuantization should never be called with ccoipQuantizationNone.";
            return;
        }
        case ccoipQuantizationMinMax: {
            performMinMaxQuantizationAndDequantization(dst_span, src_span, quantized_type, data_type);
            return;
        }
        case ccoipQuantizationZeroPointScale: {
            std::pair<float, std::int64_t> quant_params{};
            std::optional<piquant::dtype> quant_type = get_piquant_dtype(quantized_type);
            std::optional<piquant::dtype> dequant_type = get_piquant_dtype(data_type);
            if (!quant_type.has_value() || !dequant_type.has_value()) {
                LOG(BUG) << "Unsupported quantized type: " << quantized_type;
                return;
            }
            switch (data_type) {
                case ccoipFloat:
                    quant_params = get_quant_ctx().compute_quant_config_from_data(
                        std::span{
                            reinterpret_cast<const piquant::fp32_t*>(src_span.data()), src_span.size_bytes() / sizeof(piquant::fp32_t)
                        },
                        *quant_type
                    );
                    break;
                case ccoipBFloat16:
                    quant_params = get_quant_ctx().compute_quant_config_from_data(
                        std::span{
                            reinterpret_cast<const piquant::bfp16_t*>(src_span.data()), src_span.size_bytes() / sizeof(piquant::bfp16_t)
                        },
                        *quant_type
                    );
                    break;
                default: {
                    LOG(BUG) << "Unsupported data type for quantization: " << data_type;
                }
            }
            auto [scale, zp] = quant_params;
            get_quant_ctx().quantize_dequantize_fused(
                src_span,
                *dequant_type,
                dst_span,
                *quant_type,
                scale,
                zp,
                piquant::round_mode::nearest,
                piquant::reduce_op::set
            );
            return;
        }
        default: {
            LOG(BUG) << "Unsupported quantization algorithm: " << quantization_algorithm;
        }
    }
}
