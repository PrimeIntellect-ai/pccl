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

    std::vector<float *> all_weights{};
    std::vector<float *> all_gradients{};

    constexpr size_t num_weights = 12;

    constexpr size_t n_elements = 1024 * 1024 * 256;

    for (size_t i = 0; i < num_weights; i++) {
        auto weights = new float[n_elements];
        auto gradients = new float[n_elements];
        all_weights.push_back(weights);
        all_gradients.push_back(gradients);
    }


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

        std::cout << "Start all reduce" << std::endl;

        std::vector<pcclReduceOpDescriptor_t> descriptors{};
        for (size_t j = 0; j < num_weights; j++) {
            pcclReduceOpDescriptor_t descriptor{
                .sendbuf = all_gradients[j],
                .recvbuf = all_weights[j],
                .descriptor = pcclReduceDescriptor_t {
                    .count = num_weights,
                    .tag = j,
                    .src_descriptor = pcclReduceOperandDescriptor_t {
                        .datatype = pcclFloat,
                        .distribution_hint = PCCL_DISTRIBUTION_HINT_NONE
                    }
                }
            };
            descriptors.push_back(descriptor);
        }

        pcclReduceInfo_t reduce_info{};
        pcclAllReduceMultipleWithRetry(descriptors.data(), num_weights, communicator, &reduce_info, 8);

        pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size);

        std::cout << "All reduces finished" << std::endl;
    }

    PCCL_CHECK(pcclDestroyCommunicator(communicator));
}
