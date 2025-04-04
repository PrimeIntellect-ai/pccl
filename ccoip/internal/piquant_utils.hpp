#pragma once

#include <thread>

#include <piquant.hpp>

namespace ccoip::internal {
    inline piquant::context& get_quant_ctx() {
        static piquant::context s_ctx {std::max(1u, std::thread::hardware_concurrency())};
        return s_ctx;
    }

    [[nodiscard]] inline piquant::dtype get_piquant_dtype(const ccoip::ccoip_data_type_t type) {
        switch (type) {
            case ccoip::ccoipInt8: return piquant::dtype::int8;
            case ccoip::ccoipUint8: return piquant::dtype::uint8;
            case ccoip::ccoipInt16: return piquant::dtype::int16;
            case ccoip::ccoipUint16: return piquant::dtype::uint16;
            case ccoip::ccoipInt32: return piquant::dtype::int32;
            case ccoip::ccoipUint32: return piquant::dtype::uint32;
            case ccoip::ccoipInt64: return piquant::dtype::int64;
            case ccoip::ccoipUint64: return piquant::dtype::uint64;
            case ccoip::ccoipFloat: return piquant::dtype::f32;
            case ccoip::ccoipDouble: return piquant::dtype::f64;
        }
        throw std::logic_error{"Unsupported data type"};
    }
}
