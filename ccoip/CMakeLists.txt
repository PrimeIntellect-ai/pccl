add_library(ccoip_internal STATIC
        src/cpp/ccoip_master.cpp src/cpp/ccoip_client_handler.cpp src/cpp/ccoip_master_handler.cpp
        src/cpp/bandwidth_store.cpp src/cpp/benchmark_runner.cpp src/cpp/topolgy_optimizer.cpp
        src/cpp/ccoip_packets.cpp src/cpp/ccoip_packet_buffer.cpp src/cpp/ccoip_client.cpp src/cpp/uuid_utils.cpp
        src/cpp/ccoip_master_state.cpp src/cpp/ccoip_client_state.cpp src/cpp/reduce_kernels.cpp src/cpp/reduce.cpp src/cpp/quantize.cpp
        src/cpp/quantize_kernels.cpp src/cpp/simplehash/simplehash_cuda.cpp)

add_library(crc32_cpu STATIC src/cpp/crc32/crc32_cpu.cpp)
add_library(simplehash_cpu STATIC src/cpp/simplehash/simplehash_cpu.cpp)


if (PCCL_BUILD_OPENMP_SUPPORT)
    target_link_libraries(simplehash_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
endif ()

if(${IS_AMD64})
    add_library(crc32_amd64_sse42_pcmul STATIC src/cpp/crc32/crc32_amd64_sse42_pcmul.cpp)
    target_compile_options(crc32_amd64_sse42_pcmul PUBLIC "-msse4.2" "-mpclmul")

    add_library(crc32_amd64_sse42 STATIC src/cpp/crc32/crc32_amd64_sse42.cpp)
    target_compile_options(crc32_amd64_sse42 PUBLIC "-msse4.2")

    target_link_libraries(crc32_cpu PRIVATE crc32_amd64_sse42 crc32_amd64_sse42_pcmul)
elseif(${IS_ARM64})
    add_library(crc32_arm64 STATIC src/cpp/crc32/crc32_arm64.cpp)
    target_compile_options(crc32_arm64 PUBLIC "-march=armv8-a+simd+crc")
    target_link_libraries(crc32_cpu PRIVATE crc32_arm64)
endif()

target_link_libraries(ccoip_internal PRIVATE crc32_cpu)
target_link_libraries(ccoip_internal PRIVATE simplehash_cpu)
target_link_libraries(ccoip_internal PRIVATE piquant)

target_link_libraries(ccoip_internal PUBLIC tinysockets)
target_link_libraries(ccoip_internal PRIVATE pccl_log)
target_link_libraries(ccoip_internal PRIVATE tsp)
target_link_libraries(ccoip_internal PRIVATE threadpool)

if (PCCL_BUILD_CUDA_SUPPORT)
    target_link_libraries(ccoip_internal PUBLIC CUDA::cuda_driver)
    target_compile_definitions(ccoip_internal PRIVATE PCCL_HAS_CUDA_SUPPORT)

    add_library(simplehash_cuda STATIC src/cuda/simplehash_cuda.cu)
    set_target_properties(simplehash_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF)

    set_target_properties(simplehash_cuda PROPERTIES
            CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86;87;89;90;90a"
    )
    target_link_libraries(simplehash_cuda PUBLIC CUDA::cuda_driver)
    target_link_libraries(ccoip_internal PRIVATE simplehash_cuda)
endif ()

# add stub C++ language targets ccoip_private & ccoip_public
add_library(ccoip_private STATIC cmake_hacks/dummy.cpp)
add_library(ccoip_public STATIC cmake_hacks/dummy.cpp)

target_include_directories(ccoip_internal PUBLIC internal)
target_include_directories(ccoip_internal PUBLIC internal_include)
target_include_directories(ccoip_internal PUBLIC include)

target_include_directories(ccoip_private PUBLIC internal_include)
target_include_directories(ccoip_private PUBLIC public_include)
target_include_directories(ccoip_public PUBLIC public_include)

# we need to hide ccoip_internal, hence no INTERFACE possible
target_link_libraries(ccoip_private PRIVATE ccoip_internal)
target_link_libraries(ccoip_public PRIVATE ccoip_internal)

add_subdirectory(tests)