#include <array>
#include <bit>
#include <cstdint>
#include <atomic>
#include <mutex>

// For demonstration, we include some detection headers for Linux/Android:
#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#else
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>    // or you can roll your own asm-based cpuid
#include <immintrin.h>
#elif defined(__aarch64__)
// For __crc32cb, __crc32cd, etc. we need <arm_acle.h>. For neon/pmull:
#include <arm_neon.h>
#include <arm_acle.h>
#endif
#endif

namespace ccoip::hash_utils {
    namespace {
        //------------------------------------------------------------------//
        // 1) Define a structure to hold the CPU feature flags we care about
        //------------------------------------------------------------------//

        struct CPUFeatures {
            bool hasSSE42 = false; // x86 SSE4.2
            bool hasPCLMUL = false; // x86 PCLMULQDQ
            bool hasArmCRC = false; // ARMv8 CRC instructions
            bool hasArmPMULL = false; // ARMv8 PMULL instructions
        };

        //------------------------------------------------------------------//
        // 2) Detect CPU features exactly once
        //------------------------------------------------------------------//
        // We'll do a platform-specific approach for x86 and for ARM.
        // Everything else falls back to "no advanced features."

        CPUFeatures detectCPUFeatures() {
            CPUFeatures caps{};

#if defined(__x86_64__) || defined(_M_X64)
            // Standard x86 CPUID-based detection
            // ----------------------------------
            //
            // We'll check for SSE4.2 and PCLMUL. You can refine or expand as needed.
            // We'll do a minimal approach using __get_cpuid_count if available:
            //
            //   EAX=1 => standard feature bits in ECX/EDX
            //   EAX=7 => extended features in EBX/ECX/EDX
            //
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

            // 1) Basic feature bits (SSE4.2 = bit 20 of ECX for functionId=1)
            if (__get_cpuid_max(0, nullptr) >= 1) {
                __get_cpuid(1, &eax, &ebx, &ecx, &edx);
                // SSE4.2 = ECX[20]
                caps.hasSSE42 = (ecx & (1 << 20)) != 0;
            }

            // 2) Extended feature bits in EAX=7 (PCLMUL = bit 1 of ECX for functionId=1, but let's check more standard approach)
            if (__get_cpuid_max(0, nullptr) >= 7) {
                __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
                // PCLMULQDQ is actually bit 1 of ECX for functionId=1, or bit 10 of ECX for EAX=7 depending on your toolchain docs.
                // The more standard check is in functionId=1, ECX[1].
                // We'll do a direct check from the original functionId=1 result. But to keep it simple, let's do:

                // Actually we want to re-run functionId=1 for PCLMUL:
                // PCLMUL is ECX[1] from functionId=1 => so let's do it above after the call to __get_cpuid(1).
                // We'll do it properly:
            }

            // Actually let's do it more straightforwardly:
            if (__get_cpuid_max(0, nullptr) >= 1) {
                __get_cpuid(1, &eax, &ebx, &ecx, &edx);
                // PCLMULQDQ is ECX[1]
                caps.hasPCLMUL = (ecx & (1 << 1)) != 0;
            }

#elif defined(__aarch64__)
            // For ARMv8, we can do a Linux-based approach with getauxval:
            // We want to see if HWCAP_CRC32 and HWCAP_PMULL are set.
            // This of course only works on Linux with the correct headers.
            // For other OSes, you'll need a different approach or just assume.
#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    // HWCAP_CRC32 is 1<<7 on Linux/AArch64
    // HWCAP_PMULL is 1<<4 on Linux/AArch64
    // As of asm/hwcap.h:
    //   #define HWCAP_PMULL (1 << 4)
    //   #define HWCAP_CRC32 (1 << 7)
    caps.hasArmCRC   = (hwcap & HWCAP_CRC32) != 0;
    caps.hasArmPMULL = (hwcap & HWCAP_PMULL) != 0;
#endif
#else
            // On other platforms: we assume no advanced CRC instructions
#endif

            return caps;
        }

        //------------------------------------------------------------------//
        // 3) Provide the specialized implementations as *separate* functions
        //------------------------------------------------------------------//
        //
        // We'll simply place the original code for each specialized path into
        // its own static function. Then we'll pick which one to call at runtime.

        //------------------- ARMv8 PMULL + CRC32 Implementation -------------
#if defined(__aarch64__)
// Keep the original ARM code you had, but now in a function, *unconditionally*.
// This will compile only if your compiler supports those intrinsics + inline asm.
[[nodiscard]] static uint32_t CRC32_armv8(const void *data, size_t size)
{
    if (!data || !size) [[unlikely]] return 0;

    const auto* buf = static_cast<const std::uint8_t*>(data);
    // Define your inline-assembly helpers from your code:
    auto clmul_lo_e = [](uint64x2_t a, uint64x2_t b, uint64x2_t c) noexcept -> uint64x2_t {
        uint64x2_t r;
        __asm__ __volatile__(
            "pmull %0.1q, %2.1d, %3.1d\n"
            "eor %0.16b, %0.16b, %1.16b\n"
            : "=w"(r), "+w"(c) : "w"(a), "w"(b)
        );
        return r;
    };
    auto clmul_hi_e = [](uint64x2_t a, uint64x2_t b, uint64x2_t c) noexcept -> uint64x2_t {
        uint64x2_t r;
        __asm__ __volatile__(
            "pmull2 %0.1q, %2.2d, %3.2d\n"
            "eor %0.16b, %0.16b, %1.16b\n"
            : "=w"(r), "+w"(c) : "w"(a), "w"(b)
        );
        return r;
    };

    std::uint32_t crc = ~0u;

    // Align to 8 if possible
    while (size && ((reinterpret_cast<std::uintptr_t>(buf) & 7) != 0)) {
        crc = __crc32cb(crc, *buf++);
        --size;
    }
    // possibly consume one 8-byte chunk
    if (((reinterpret_cast<std::uintptr_t>(buf) & 8) != 0) && (size >= 8)) {
        crc = __crc32cd(crc, *reinterpret_cast<const std::uint64_t*>(buf));
        buf += 8;
        size -= 8;
    }

    // Bulk PMULL logic from your code
    if (size >= 192) {
        // Prefetch some data
        uint64x2_t x0 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf)), y0;
        uint64x2_t x1 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+16)), y1;
        uint64x2_t x2 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+32)), y2;
        uint64x2_t x3 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+48)), y3;
        uint64x2_t x4 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+64)), y4;
        uint64x2_t x5 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+80)), y5;
        uint64x2_t x6 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+96)), y6;
        uint64x2_t x7 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+112)), y7;
        uint64x2_t x8 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+128)), y8;
        uint64x2_t x9 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+144)), y9;
        uint64x2_t x10 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+160)), y10;
        uint64x2_t x11 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+176)), y11;
        uint64x2_t k;
        {
            static constexpr uint64_t k_ alignas(16)[] = {0xa87ab8a8, 0xab7aff2a};
            k = vld1q_u64(k_);
        }
        x0 = veorq_u64( (uint64x2_t){crc, 0}, x0);
        buf += 192;
        size -= 192;

        while (size >= 192) {
            y0  = clmul_lo_e(x0, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf)));
            x0  = clmul_hi_e(x0, k, y0);
            y1  = clmul_lo_e(x1, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+16)));
            x1  = clmul_hi_e(x1, k, y1);
            y2  = clmul_lo_e(x2, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+32)));
            x2  = clmul_hi_e(x2, k, y2);
            y3  = clmul_lo_e(x3, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+48)));
            x3  = clmul_hi_e(x3, k, y3);
            y4  = clmul_lo_e(x4, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+64)));
            x4  = clmul_hi_e(x4, k, y4);
            y5  = clmul_lo_e(x5, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+80)));
            x5  = clmul_hi_e(x5, k, y5);
            y6  = clmul_lo_e(x6, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+96)));
            x6  = clmul_hi_e(x6, k, y6);
            y7  = clmul_lo_e(x7, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+112)));
            x7  = clmul_hi_e(x7, k, y7);
            y8  = clmul_lo_e(x8, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+128)));
            x8  = clmul_hi_e(x8, k, y8);
            y9  = clmul_lo_e(x9, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+144)));
            x9  = clmul_hi_e(x9, k, y9);
            y10 = clmul_lo_e(x10, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+160)));
            x10 = clmul_hi_e(x10, k, y10);
            y11 = clmul_lo_e(x11, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+176)));
            x11 = clmul_hi_e(x11, k, y11);

            buf += 192;
            size -= 192;
        }
        {
            static constexpr uint64_t k_ alignas(16)[] = {0xf20c0dfe, 0x493c7d27};
            k = vld1q_u64(k_);
        }
        y0  = clmul_lo_e(x0, k, x1);
        x0  = clmul_hi_e(x0, k, y0);
        y2  = clmul_lo_e(x2, k, x3);
        x2  = clmul_hi_e(x2, k, y2);
        y4  = clmul_lo_e(x4, k, x5);
        x4  = clmul_hi_e(x4, k, y4);
        y6  = clmul_lo_e(x6, k, x7);
        x6  = clmul_hi_e(x6, k, y6);
        y8  = clmul_lo_e(x8, k, x9);
        x8  = clmul_hi_e(x8, k, y8);
        y10 = clmul_lo_e(x10, k, x11);
        x10 = clmul_hi_e(x10, k, y10);

        {
            static constexpr uint64_t k_ alignas(16)[] = {0x3da6d0cb, 0xba4fc28e};
            k = vld1q_u64(k_);
        }
        y0  = clmul_lo_e(x0, k, x2);
        x0  = clmul_hi_e(x0, k, y0);
        y4  = clmul_lo_e(x4, k, x6);
        x4  = clmul_hi_e(x4, k, y4);
        y8  = clmul_lo_e(x8, k, x10);
        x8  = clmul_hi_e(x8, k, y8);

        {
            static constexpr uint64_t k_ alignas(16)[] = {0x740eef02, 0x9e4addf8};
            k = vld1q_u64(k_);
        }
        y0  = clmul_lo_e(x0, k, x4);
        x0  = clmul_hi_e(x0, k, y0);
        x4  = x8;
        y0  = clmul_lo_e(x0, k, x4);
        x0  = clmul_hi_e(x0, k, y0);

        crc = __crc32cd(0, vgetq_lane_u64(x0, 0));
        crc = __crc32cd(crc, vgetq_lane_u64(x0, 1));
    }

    // Handle remainder in 8-byte increments
    while (size >= 8) {
        crc = __crc32cd(crc, *reinterpret_cast<const std::uint64_t*>(buf));
        buf  += 8;
        size -= 8;
    }
    // Final leftover bytes
    while (size > 0) {
        crc = __crc32cb(crc, *buf++);
        --size;
    }

    return ~crc;
}
#endif // __aarch64__

        //------------------- x86 SSE4.2 / PCLMUL Implementation -------------
#if defined(__x86_64__) || defined(_M_X64)

        // The x86 specialized code from your snippet, in function form:
        [[nodiscard]] uint32_t CRC32_x86_pclmul(const void *data, size_t size) {
            if (!data || !size) [[unlikely]] return 0;
            const auto *buf = static_cast<const std::uint8_t *>(data);

            // Some helper inlines from your snippet
            auto clmul_scalar = [](uint32_t a, uint32_t b) noexcept -> __m128i {
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
            auto crc_shift = [&](uint32_t crc, size_t sz) noexcept -> __m128i {
                return clmul_scalar(crc, xnmodp((sz << 3) - 33));
            };

            std::uint32_t crc = ~0u;
            // Align on 8 if possible
            while (size && (reinterpret_cast<std::uintptr_t>(buf) & 7)) {
                crc = _mm_crc32_u8(crc, *buf++);
                --size;
            }

            // Bulk approach from snippet:
            if (size >= 32) {
                std::size_t klen = ((size - 8) / 24) << 3;
                std::uint32_t crc1 = 0;
                std::uint32_t crc2 = 0;

                while (size >= 32) {
                    crc = _mm_crc32_u64(crc, *reinterpret_cast<const std::uint64_t *>(buf));
                    crc1 = _mm_crc32_u64(crc1, *reinterpret_cast<const std::uint64_t *>(buf + klen));
                    crc2 = _mm_crc32_u64(crc2, *reinterpret_cast<const std::uint64_t *>(buf + (klen << 1)));
                    buf += 8;
                    size -= 24; // 8 + (8 + 8) from the snippet's approach
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

        // A simpler SSE4.2-only version (without the PCLMUL fancy approach):
        [[nodiscard]] static uint32_t CRC32_x86_sse42(const void *data, size_t size) {
            if (!data || !size) [[unlikely]] return 0;
            const auto *buf = static_cast<const std::uint8_t *>(data);

            std::uint32_t crc = ~0u;
            // Align
            while (size && (reinterpret_cast<std::uintptr_t>(buf) & 7)) {
                crc = _mm_crc32_u8(crc, *buf++);
                --size;
            }
            // 8-byte chunks
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

#endif // x86 specializations

        //------------------- Generic (no hardware accel) --------------------
        [[nodiscard]] uint32_t CRC32_generic(const void *data, size_t size) {
            if (!data || !size) [[unlikely]] return 0;
            const auto *buf = static_cast<const std::uint8_t *>(data);

            static constexpr std::array<std::uint32_t, 256> crc_lut = {
                0x00000000, 0xf26b8303, 0xe13b70f7, 0x1350f3f4,
                0xc79a971f, 0x35f1141c, 0x26a1e7e8, 0xd4ca64eb,
                0x8ad958cf, 0x78b2dbcc, 0x6be22838, 0x9989ab3b,
                0x4d43cfd0, 0xbf284cd3, 0xac78bf27, 0x5e133c24,
                0x105ec76f, 0xe235446c, 0xf165b798, 0x030e349b,
                0xd7c45070, 0x25afd373, 0x36ff2087, 0xc494a384,
                0x9a879fa0, 0x68ec1ca3, 0x7bbcef57, 0x89d76c54,
                0x5d1d08bf, 0xaf768bbc, 0xbc267848, 0x4e4dfb4b,
                0x20bd8ede, 0xd2d60ddd, 0xc186fe29, 0x33ed7d2a,
                0xe72719c1, 0x154c9ac2, 0x061c6936, 0xf477ea35,
                0xaa64d611, 0x580f5512, 0x4b5fa6e6, 0xb93425e5,
                0x6dfe410e, 0x9f95c20d, 0x8cc531f9, 0x7eaeb2fa,
                0x30e349b1, 0xc288cab2, 0xd1d83946, 0x23b3ba45,
                0xf779deae, 0x05125dad, 0x1642ae59, 0xe4292d5a,
                0xba3a117e, 0x4851927d, 0x5b016189, 0xa96ae28a,
                0x7da08661, 0x8fcb0562, 0x9c9bf696, 0x6ef07595,
                0x417b1dbc, 0xb3109ebf, 0xa0406d4b, 0x522bee48,
                0x86e18aa3, 0x748a09a0, 0x67dafa54, 0x95b17957,
                0xcba24573, 0x39c9c670, 0x2a993584, 0xd8f2b687,
                0x0c38d26c, 0xfe53516f, 0xed03a29b, 0x1f682198,
                0x5125dad3, 0xa34e59d0, 0xb01eaa24, 0x42752927,
                0x96bf4dcc, 0x64d4cecf, 0x77843d3b, 0x85efbe38,
                0xdbfc821c, 0x2997011f, 0x3ac7f2eb, 0xc8ac71e8,
                0x1c661503, 0xee0d9600, 0xfd5d65f4, 0x0f36e6f7,
                0x61c69362, 0x93ad1061, 0x80fde395, 0x72966096,
                0xa65c047d, 0x5437877e, 0x4767748a, 0xb50cf789,
                0xeb1fcbad, 0x197448ae, 0x0a24bb5a, 0xf84f3859,
                0x2c855cb2, 0xdeeedfb1, 0xcdbe2c45, 0x3fd5af46,
                0x7198540d, 0x83f3d70e, 0x90a324fa, 0x62c8a7f9,
                0xb602c312, 0x44694011, 0x5739b3e5, 0xa55230e6,
                0xfb410cc2, 0x092a8fc1, 0x1a7a7c35, 0xe811ff36,
                0x3cdb9bdd, 0xceb018de, 0xdde0eb2a, 0x2f8b6829,
                0x82f63b78, 0x709db87b, 0x63cd4b8f, 0x91a6c88c,
                0x456cac67, 0xb7072f64, 0xa457dc90, 0x563c5f93,
                0x082f63b7, 0xfa44e0b4, 0xe9141340, 0x1b7f9043,
                0xcfb5f4a8, 0x3dde77ab, 0x2e8e845f, 0xdce5075c,
                0x92a8fc17, 0x60c37f14, 0x73938ce0, 0x81f80fe3,
                0x55326b08, 0xa759e80b, 0xb4091bff, 0x466298fc,
                0x1871a4d8, 0xea1a27db, 0xf94ad42f, 0x0b21572c,
                0xdfeb33c7, 0x2d80b0c4, 0x3ed04330, 0xccbbc033,
                0xa24bb5a6, 0x502036a5, 0x4370c551, 0xb11b4652,
                0x65d122b9, 0x97baa1ba, 0x84ea524e, 0x7681d14d,
                0x2892ed69, 0xdaf96e6a, 0xc9a99d9e, 0x3bc21e9d,
                0xef087a76, 0x1d63f975, 0x0e330a81, 0xfc588982,
                0xb21572c9, 0x407ef1ca, 0x532e023e, 0xa145813d,
                0x758fe5d6, 0x87e466d5, 0x94b49521, 0x66df1622,
                0x38cc2a06, 0xcaa7a905, 0xd9f75af1, 0x2b9cd9f2,
                0xff56bd19, 0x0d3d3e1a, 0x1e6dcdee, 0xec064eed,
                0xc38d26c4, 0x31e6a5c7, 0x22b65633, 0xd0ddd530,
                0x0417b1db, 0xf67c32d8, 0xe52cc12c, 0x1747422f,
                0x49547e0b, 0xbb3ffd08, 0xa86f0efc, 0x5a0489ff,
                0x8ecee914, 0x7ca56a17, 0x6ff599e3, 0x9d9e1ae0,
                0xd3d3e1ab, 0x21b862a8, 0x32e8915c, 0xc083125f,
                0x141976b4, 0xe672f5b7, 0xf5220643, 0x07498540,
                0x590ab964, 0xab613a67, 0xb831c99b, 0x4a5a4a90,
                0x9e902e7b, 0x6cfbad78, 0x7fab5e8c, 0x8dc0dd8f,
                0xe330a81a, 0x115b2b19, 0x020bd8ed, 0xf0605bee,
                0x24aa3f05, 0xd6c1bc06, 0xc5914ff2, 0x37faccf1,
                0x69e9f0d5, 0x9b8273d6, 0x88d28022, 0x7ab90321,
                0xae7367ca, 0x5c18e4c9, 0x4f48173d, 0xbd23943e,
                0xf36e6f75, 0x0105ec76, 0x12551f82, 0xe03e9c81,
                0x34f4f86a, 0xc69f7b69, 0xd5cf889d, 0x27a40b9e,
                0x79b737ba, 0x8bdcb4b9, 0x988c474d, 0x6ae7c44e,
                0xbe2da0a5, 0x4c4623a6, 0x5f16d052, 0xad7d5351
            };

            std::uint32_t crc = ~0u;
            for (std::size_t i = 0; i < size; ++i) {
                crc = (crc >> 8) ^ crc_lut[static_cast<uint8_t>(crc) ^ buf[i]];
            }
            return ~crc;
        }

        //------------------------------------------------------------------//
        // 4) Provide a one-time initialization and a global dispatcher
        //------------------------------------------------------------------//

        // We'll store a function pointer to the best implementation we detect.
        using CRC32Func = uint32_t (*)(const void *, size_t);

        std::once_flag g_initOnce;
        CRC32Func g_bestCRC32 = nullptr;

        void initBestCRC32() {
            // Detect features
            static const CPUFeatures caps = detectCPUFeatures();

            // Decide the best function pointer
#if defined(__aarch64__)
    // If on ARM64, check for CRC + PMULL
    if (caps.hasArmCRC && caps.hasArmPMULL) {
        g_bestCRC32 = &CRC32_armv8;
        return;
    }
#endif

#if defined(__x86_64__) || defined(_M_X64)
            // If on x86_64, check SSE4.2/PCLMUL
            if (caps.hasSSE42) {
                if (caps.hasPCLMUL) {
                    g_bestCRC32 = &CRC32_x86_pclmul;
                    return;
                } else {
                    g_bestCRC32 = &CRC32_x86_sse42;
                    return;
                }
            }
#endif

            // Fallback
            g_bestCRC32 = &CRC32_generic;
        }

        //------------------------------------------------------------------//
        // 5) The public-facing CRC32 function
        //------------------------------------------------------------------//
    } // end anonymous namespace

    [[nodiscard]] uint32_t CRC32(const void *data, const size_t size) {
        // Ensure we only do CPU feature detection once
        std::call_once(g_initOnce, initBestCRC32);

        // Now just dispatch to whichever function was chosen
        return g_bestCRC32(data, size);
    }
} // namespace ccoip::hash_utils
