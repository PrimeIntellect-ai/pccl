#pragma once

#include <optional>
#include <thread>

#include <piquant.hpp>

namespace ccoip::internal {
    inline piquant::context &get_quant_ctx() {
        static piquant::context s_ctx{std::max(1u, std::thread::hardware_concurrency())};
        return s_ctx;
    }

    [[nodiscard]] inline std::optional<piquant::dtype> get_piquant_dtype(const ccoip_data_type_t type) {
        switch (type) {
            case ccoipUint2:
                return piquant::dtype::uint2;
            case ccoipUint4:
                return piquant::dtype::uint4;
            case ccoipInt8:
            case ccoipUint8:
                // for quantization, we treat int8 and uint8 the same way
                return piquant::dtype::uint8;
            case ccoipFloat:
                return piquant::dtype::f32;
            case ccoipBFloat16:
                return piquant::dtype::bf16;
            // Todo: add support for sub-byte types like uint4, uint2 etc
            default:
                return std::nullopt;
        }
    }
} // namespace ccoip::internal
