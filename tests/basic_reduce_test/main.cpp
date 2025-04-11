#include <cassert>
#include <ccoip.h>
#include <chrono>
#include <iostream>
#include <pccl.h>
#include <random>
#include <thread>

#include "../../log/include/pccl_log.hpp"

void panic(const int exit_code) { exit(exit_code); }

#define PCCL_CHECK(status)                                                                                             \
    {                                                                                                                  \
        pcclResult_t status_val = status;                                                                              \
        if (status_val != pcclSuccess) {                                                                               \
            std::cerr << "Error: " << status_val << std::endl;                                                         \
            panic(1);                                                                                                  \
        }                                                                                                              \
    }

void fill_uniform(float *data, const size_t count) {
    std::mt19937 gen(42);
    std::uniform_real_distribution dis(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dis(gen);
    }
}

#define MAX_STEPS 1000

int main() {
    PCCL_CHECK(pcclInit());

    pcclComm_t *communicator{};
    constexpr pcclCommCreateParams_t params{.master_address = {.inet = {.protocol = inetIPv4, .ipv4 = {127, 0, 0, 1}},
                                                               .port = CCOIP_PROTOCOL_PORT_MASTER},
                                            .peer_group = 0};
    PCCL_CHECK(pcclCreateCommunicator(&params, &communicator));
    PCCL_CHECK(pcclConnect(communicator));

    int world_size{};
    pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size);

    constexpr size_t n_elements = 1024 * 1024 * 4;
    const auto weights = new float[n_elements];
    fill_uniform(weights, n_elements);

    const auto gradients = new float[n_elements];
    fill_uniform(gradients, n_elements);

    // Create shared state
    pcclTensorInfo_t infos[1] = {
        {
            .name = "weights",
            .data = weights,
            .count = n_elements,
            .datatype = pcclFloat,
            .device_type = pcclDeviceCpu,
            .allow_content_inequality = true
        }
    };

    pcclSharedState_t shared_state{.revision = 0, .count = 1, .infos = infos};

    size_t i = 0;

    while (true) {
        i++;
        if (i > 1) {
            PCCL_CHECK(pcclUpdateTopology(communicator));
            PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
        }
        
        if (world_size > 1) {
            // PCCL_CHECK(pcclOptimizeTopology(communicator));
            PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
        }

        if (world_size < 2) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        pcclSharedStateSyncInfo_t sync_info{};
        PCCL_CHECK(pcclSynchronizeSharedState(communicator, &shared_state, &sync_info));
        if (i > 2) {
            assert(sync_info.rx_bytes == 0);
        }

        pcclAsyncReduceOp_t async_op{};
        pcclReduceInfo_t reduce_info{};
        auto start = std::chrono::high_resolution_clock::now();
        pcclResult_t result;

        do {
            constexpr pcclReduceDescriptor_t desc{
                    .count = n_elements,
                    .op = pcclSum,
                    .tag = 0,
                    .src_descriptor = {.datatype = pcclFloat, .distribution_hint = PCCL_DISTRIBUTION_HINT_NONE},
                    .quantization_options = {.quantized_datatype = pcclUint8, .algorithm = pcclQuantZeroPointScale},
                    //.quantization_options = {.quantized_datatype = pcclFloat, .algorithm = pcclQuantZeroPointScale},
            };
            pcclAllReduceAsync(gradients, weights, &desc, communicator, &async_op);
            result = pcclAwaitAsyncReduce(&async_op, &reduce_info);
            pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size);
            LOG(INFO) << "pcclAllReduce status " << result;
        } while (result != pcclSuccess && world_size > 1);

        auto end = std::chrono::high_resolution_clock::now();
        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "All reduce finished: Rx-Bytes:" << reduce_info.rx_bytes << "; Tx-Bytes:" << reduce_info.tx_bytes << "; Revision: " << shared_state.revision << std::endl;
        const double mb_per_second = static_cast<double>(reduce_info.rx_bytes + reduce_info.tx_bytes) / 1e6 /
                                     (static_cast<double>(time_ms) / 1e3);
        std::cout << "Bandwidth: " << mb_per_second << " MB/s" << std::endl;

        // print first 10 elements of the result
        for (size_t j = 0; j < 10; ++j) {
            std::cout << weights[j] << " ";
        }
        std::cout << std::endl;
        shared_state.revision++;
    }

    PCCL_CHECK(pcclDestroyCommunicator(communicator));
}
