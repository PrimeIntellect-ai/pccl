#include <cstdint>
#include <cstddef>

#if defined(_MSC_VER)
// For Windows: intrin.h provides __cpuid(), _mm_crc32_u8, etc.
#include <intrin.h>
#else
// For GCC/Clang on x86:
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

//------------------- x86 SSE4.2 / PCLMUL Implementation -------------
#if defined(__x86_64__) || defined(_M_X64)
// "fancy" PCLMUL version:
[[nodiscard]] uint32_t CRC32_x86_sse42_pclmul(const void *data, size_t size) {
    if (!data || !size) [[unlikely]] return 0;
    const auto *buf = static_cast<const std::uint8_t *>(data);

    auto clmul_scalar = [](const uint32_t a, const uint32_t b) noexcept -> __m128i {
        return _mm_clmulepi64_si128(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b), 0);
    };
    // exponent polynomial generator (xnmodp):
    auto xnmodp = [](std::uint64_t n) noexcept -> std::uint32_t {
        std::uint64_t stack = ~static_cast<std::uint64_t>(1);
        std::uint32_t acc, low;
        for (; n > 191; n = (n >> 1) - 16)
            stack = (stack << 1) + (n & 1);
        stack = ~stack;
        acc = 0x80000000u >> (n & 31);
        for (n >>= 5; n; --n) acc = _mm_crc32_u32(acc, 0);
        while ((low = stack & 1), stack >>= 1) {
            __m128i x = _mm_cvtsi32_si128(acc);
            std::uint64_t y = _mm_cvtsi128_si64(_mm_clmulepi64_si128(x, x, 0));
            acc = _mm_crc32_u64(0, y << low);
        }
        return acc;
    };
    auto crc_shift = [&](const uint32_t crc, const size_t sz) noexcept -> __m128i {
        return clmul_scalar(crc, xnmodp((sz << 3) - 33));
    };

    std::uint32_t crc = ~0u;
    // Align on 8 if possible
    while (size && (reinterpret_cast<std::uintptr_t>(buf) & 7)) {
        crc = _mm_crc32_u8(crc, *buf++);
        --size;
    }

    // Bulk approach (illustrative)
    if (size >= 32) {
        std::size_t klen = ((size - 8) / 24) << 3;
        std::uint32_t crc1 = 0;
        std::uint32_t crc2 = 0;

        while (size >= 32) {
            crc = _mm_crc32_u64(crc, *reinterpret_cast<const std::uint64_t *>(buf));
            crc1 = _mm_crc32_u64(crc1, *reinterpret_cast<const std::uint64_t *>(buf + klen));
            crc2 = _mm_crc32_u64(crc2, *reinterpret_cast<const std::uint64_t *>(buf + (klen << 1)));
            buf += 8;
            size -= 24;
        }
        __m128i vc0 = crc_shift(crc, (klen << 1) + 8);
        __m128i vc1 = crc_shift(crc1, klen + 8);

        std::uint64_t vc = _mm_extract_epi64(_mm_xor_si128(vc0, vc1), 0);
        // Final 8 bytes
        buf += (klen << 1);
        crc = crc2;
        crc = _mm_crc32_u64(crc, *reinterpret_cast<const std::uint64_t *>(buf) ^ vc);
        buf += 8;
        size -= 8;
    }

    // Remainder 8 bytes
    while (size >= 8) {
        crc = _mm_crc32_u64(crc, *reinterpret_cast<const std::uint64_t *>(buf));
        buf += 8;
        size -= 8;
    }
    // leftover
    while (size > 0) {
        crc = _mm_crc32_u8(crc, *buf++);
        --size;
    }
    return ~crc;
}
#endif // x86
