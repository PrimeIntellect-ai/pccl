#include "quantize.hpp"

#include <ccoip_types.hpp>
#include <pccl_log.hpp>
#include <quantize_kernels.hpp>
#include <piquant_utils.hpp>
#include <span>

ccoip::internal::quantize::DeQuantizationMetaData ccoip::internal::quantize::performQuantization(const std::span<std::byte> &dst_span, const std::span<const std::byte> &src_span,
                                                                             const ccoip_quantization_algorithm_t quantization_algorithm,
                                                                             const ccoip_data_type_t quantized_type, const ccoip_data_type_t data_type) {
    switch (quantization_algorithm) {
        case ccoipQuantizationNone: {
            LOG(BUG) << "performQuantization should never be called with ccoipQuantizationNone.";
            return {};
        }
        case ccoipQuantizationMinMax: {
            return performMinMaxQuantization(dst_span, src_span, quantized_type, data_type);
        }
        case ccoipQuantizationZeroPointScale: {
            if (data_type != ccoipFloat) { // TODO
                LOG(BUG) << "ZeroPointScale quantization is only supported for float data type.";
                return {};
            }
            auto [scale, zp] = get_quant_ctx().compute_quant_config_from_data(
                std::span<const float> {reinterpret_cast<const float *>(src_span.data()), src_span.size_bytes()/sizeof(float)},
                get_piquant_dtype(quantized_type)
            );
            get_quant_ctx().quantize(
                src_span,
                get_piquant_dtype(data_type),
                dst_span,
                get_piquant_dtype(quantized_type),
                scale,
                zp,
                piquant::round_mode::nearest // TODO
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
        const std::span<const std::byte> &src_span, const ccoip_quantization_algorithm_t quantization_algorithm,
        const ccoip_data_type_t quantized_type, const ccoip_data_type_t data_type) {
    switch (quantization_algorithm) {
        case ccoipQuantizationNone: {
            LOG(BUG) << "performQuantization should never be called with ccoipQuantizationNone.";
            return;
        }
        case ccoipQuantizationMinMax: {
            performMinMaxQuantizationAndDequantization(dst_span, src_span, quantized_type, data_type);
            return;
        }
        default: {
            LOG(BUG) << "Unsupported quantization algorithm: " << quantization_algorithm;
        }
    }
}
