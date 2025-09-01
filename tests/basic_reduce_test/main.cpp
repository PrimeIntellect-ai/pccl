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

std::mt19937 gen(42);

void fill_uniform(float *data, const size_t count) {
    std::uniform_real_distribution dis(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dis(gen);
    }
}

#define MAX_STEPS 1000

static int GetIntEnvVar(const std::string &var_name, const int default_value) {
    const char *env_var = std::getenv(var_name.c_str());
    if (env_var == nullptr) {
        return default_value;
    }
    try {
        return std::stoi(env_var);
    } catch (const std::invalid_argument &) {
        std::cerr << "Invalid value for environment variable " << var_name << ": " << env_var << std::endl;
        return default_value;
    }
}

int main() {
    PCCL_CHECK(pcclInit());

    constexpr uint32_t connection_pool_size = 32;

    pcclComm_t *communicator{};
    const pcclCommCreateParams_t params{.master_address = {.inet = {.protocol = inetIPv4, .ipv4 = {127, 0, 0, 1}},
                                                               .port = CCOIP_PROTOCOL_PORT_MASTER},
                                            .peer_group = 0,
                                            .p2p_connection_pool_size = connection_pool_size};
    PCCL_CHECK(pcclCreateCommunicator(&params, &communicator));
    PCCL_CHECK(pcclConnect(communicator));

    const size_t num_reduce_ops = 32;
    const size_t max_in_flight = 32;
    const size_t mib = 16;

    int world_size{};
    pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size);

    const size_t n_elements = 1024 * 1024 * mib;
    const auto weights = new float[n_elements]{};

    std::vector<float *> gradients{};
    gradients.reserve(num_reduce_ops);
    for (size_t i = 0; i < num_reduce_ops; ++i) {
        gradients.push_back(new float[n_elements]{});
    }

    // Create a shared state
    pcclTensorInfo_t infos[1] = {{.name = "weights",
                                  .data = weights,
                                  .count = n_elements,
                                  .datatype = pcclFloat,
                                  .device_type = pcclDeviceCpu,
                                  .allow_content_inequality = false}};

    pcclSharedState_t shared_state{.revision = 0, .count = 1, .infos = infos};

    size_t i = 0;
    size_t num_syncs = 0;
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

        for (auto &gradient: gradients) {
            // fill_uniform(gradient, n_elements);
        }

        pcclSharedStateSyncInfo_t sync_info{};
        PCCL_CHECK(pcclSynchronizeSharedState(communicator, &shared_state,
                                              PCCL_SHARED_STATE_SYNC_STRATEGY_ENFORCE_POPULAR, &sync_info));
        num_syncs++;
        if (num_syncs > 1) {
            // Assert we never receive data. Only send.
            // Failure of this assert would indicate peer drift.
            assert(sync_info.rx_bytes == 0);
        }

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<pcclReduceOpDescriptor_t> descriptors(num_reduce_ops);
        for (size_t j = 0; j < num_reduce_ops; ++j) {
            const pcclReduceDescriptor_t desc{
                    .count = n_elements,
                    .op = pcclSum,
                    .tag = j,
                    .src_descriptor = {.datatype = pcclFloat, .distribution_hint = pcclDistributionNone},
                    .quantization_options = {.quantized_datatype = pcclUint2, .algorithm = pcclQuantZeroPointScale},
            };
            descriptors[j] =
                    pcclReduceOpDescriptor_t{.sendbuf = gradients[j], .recvbuf = gradients[j], .descriptor = desc};
        }

        pcclReduceInfo_t reduce_info{};
        PCCL_CHECK(pcclAllReduceMultipleWithRetry(descriptors.data(), descriptors.size(), communicator, &reduce_info,
                                                  max_in_flight));

        auto end = std::chrono::high_resolution_clock::now();
        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "All reduce finished: Rx-Bytes:" << reduce_info.rx_bytes << "; Tx-Bytes:" << reduce_info.tx_bytes
                  << "; Revision: " << shared_state.revision << std::endl;
        const double mb_per_second = static_cast<double>(reduce_info.rx_bytes + reduce_info.tx_bytes) / 1e6 /
                                     (static_cast<double>(time_ms) / 1e3);
        std::cout << "Bandwidth: " << mb_per_second << " MB/s" << std::endl;
        std::cout << "Time: " << time_ms << " ms" << std::endl;

        for (int j = 0; j < num_reduce_ops; ++j) {
            for (int k = 0; k < n_elements; ++k) {
                weights[k] += gradients[j][k];
            }
        }
        for (size_t j = 0; j < num_reduce_ops; ++j) {
            weights[j] /= static_cast<float>(num_reduce_ops);
        }

        // print first 10 elements of the result
        for (size_t j = 0; j < 10; ++j) {
            std::cout << weights[j] << " ";
        }
        std::cout << std::endl;
        shared_state.revision++;
    }

    PCCL_CHECK(pcclDestroyCommunicator(communicator));
}
