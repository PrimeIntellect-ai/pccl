#include "pccl.h"
#include "pccl_internal.hpp"
#include <optional>
#include <ccoip_master.hpp>
#include <pccl_log.hpp>
#include <unordered_set>

static constinit bool pccl_initialized = false;

#define PCCL_VALIDATE_INITIALIZED() \
    if (!pccl_initialized) { \
        return (pcclNotInitialized); \
    }

static pcclResult_t internalPcclInit() {
    return pcclSuccess;
}

pcclResult_t pcclInit() {
    if (pccl_initialized) {
        return pcclSuccess;
    }
    PCCL_ERR_PROPAGATE(internalPcclInit());
    pccl_initialized = true;
    return pcclSuccess;
}

pcclResult_t pcclCreateCommunicator(const pcclCommCreateParams_t *params, pcclComm_t **comm_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(comm_out != nullptr, pcclInvalidArgument);
    *comm_out = new pcclComm_t(*params);
    return pcclSuccess;
}

pcclResult_t pcclGetAttribute(const pcclComm_t *communicator,
                              const pcclAttribute_t attribute,
                              int *p_attribute_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    PCCL_VALIDATE(p_attribute_out != nullptr, pcclInvalidArgument);

    int value{};
    switch (attribute) {
        case PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE: {
            value = static_cast<int>(communicator->ccoip_client->getGlobalWorldSize());
            break;
        }
        case PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE: {
            value = static_cast<int>(communicator->ccoip_client->getLocalWorldSize());
            break;
        }
        case PCCL_ATTRIBUTE_NUM_DISTINCT_PEER_GROUPS: {
            value = static_cast<int>(communicator->ccoip_client->getNumDistinctPeerGroups());
            break;
        }
        case PCCL_ATTRIBUTE_LARGEST_PEER_GROUP_WORLD_SIZE: {
            value = static_cast<int>(communicator->ccoip_client->getLargestPeerGroupWorldSize());
            break;
        }
        default: {
            [[unlikely]] return pcclInvalidArgument;
        }
    }
    *p_attribute_out = value;
    return pcclSuccess;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    if (communicator->ccoip_client != nullptr) {
        if (!communicator->ccoip_client->interrupt()) [[unlikely]] {
            return pcclInvalidUsage;
        }
        if (!communicator->ccoip_client->join()) [[unlikely]] {
            return pcclInvalidUsage;
        }
    }
    delete communicator;
    return pcclSuccess;
}

pcclResult_t pcclConnect(pcclComm_t *communicator) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client == nullptr, pcclInvalidUsage);
    communicator->ccoip_client = std::make_unique<ccoip::CCoIPClient>(communicator->params.master_address,
                                                                      communicator->params.peer_group);

    pcclResult_t status = pcclSuccess;
    if (!communicator->ccoip_client->connect()) {
        LOG(ERR) << "Failed to establish connection to master";
        if (!communicator->ccoip_client->interrupt()) [[unlikely]] {
            LOG(ERR) << "Failed to interrupt client after connection failure";
            status = pcclInternalError;
            goto failure;
        }
        if (!communicator->ccoip_client->join()) [[unlikely]] {
            LOG(ERR) << "Failed to join client after connection failure";
            status = pcclInternalError;
            goto failure;
        }
        status = pcclMasterConnectionFailed;
        goto failure;
    }
    return status;
failure:
    communicator->ccoip_client = nullptr;
    return status;
}

static std::optional<ccoip::ccoip_data_type_t> getCCoIPDataType(const pcclDataType_t datatype) {
    switch (datatype) {
        case pcclUint8:
            return ccoip::ccoipUint8;
        case pcclUint16:
            return ccoip::ccoipUint16;
        case pcclUint32:
            return ccoip::ccoipUint32;
        case pcclUint64:
            return ccoip::ccoipUint64;
        case pcclInt8:
            return ccoip::ccoipInt8;
        case pcclInt16:
            return ccoip::ccoipInt16;
        case pcclInt32:
            return ccoip::ccoipInt32;
        case pcclInt64:
            return ccoip::ccoipInt64;
        case pcclFloat:
            return ccoip::ccoipFloat;
        case pcclDouble:
            return ccoip::ccoipDouble;
        default:
            return std::nullopt;
    }
}

static std::optional<ccoip::ccoip_device_type_t> getCCoIPDeviceType(const pcclDeviceType_t device_type) {
    switch (device_type) {
        case pcclDeviceCpu:
            return ccoip::ccoipDeviceCpu;
        case pcclDeviceCuda:
            return ccoip::ccoipDeviceCuda;
        default:
            return std::nullopt;
    }
}

static std::optional<ccoip::ccoip_quantization_algorithm_t>
getCCoIPQuantizationAlgorithm(const pcclQuantizationAlgorithm_t algorithm) {
    switch (algorithm) {
        case pcclQuantNone:
            return ccoip::ccoipQuantizationNone;
        case pcclQuantMinMax:
            return ccoip::ccoipQuantizationMinMax;
        case pcclQuantZeroPointScale:
            return ccoip::ccoipQuantizationZeroPointScale;
    }
    return std::nullopt;
}

static std::optional<ccoip::ccoip_reduce_op_t> getCCoIPReduceOp(const pcclRedOp_t op) {
    switch (op) {
        case pcclSum:
            return ccoip::ccoip_reduce_op_t::ccoipOpSum;
        case pcclAvg:
            return ccoip::ccoip_reduce_op_t::ccoipOpAvg;
        case pcclProd:
            return ccoip::ccoip_reduce_op_t::ccoipOpProd;
        case pcclMax:
            return ccoip::ccoip_reduce_op_t::ccoipOpMax;
        case pcclMin:
            return ccoip::ccoip_reduce_op_t::ccoipOpMin;
        default:
            return std::nullopt;
    }
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclUpdateTopology(pcclComm_t *communicator) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);

    // if there are any async collective operations running, we cannot update the topology
    if (communicator->ccoip_client->isAnyCollectiveComsOpRunning()) {
        return pcclPendingAsyncOps;
    }

    // accept new peers; this will block until we have a valid connection to each peer
    if (!communicator->ccoip_client->acceptNewPeers()) {
        return pcclUpdateTopologyFailed;
    }

    return pcclSuccess;
}

pcclResult_t pcclArePeersPending(const pcclComm_t *communicator, bool *pending_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(pending_out != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);

    bool pending = false;
    if (!communicator->ccoip_client->arePeersPending(pending)) {
        return pcclInvalidUsage;
    }
    *pending_out = pending;

    return pcclSuccess;
}

pcclResult_t pcclOptimizeTopology(const pcclComm_t *communicator) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);

    // optimize the topology
    if (communicator->ccoip_client->getGlobalWorldSize() > 1) {
        if (!communicator->ccoip_client->optimizeTopology()) {
            return pcclTopologyOptimizationFailed;
        }
    } else {
        return pcclInvalidUsage;
    }

    return pcclSuccess;
}

pcclResult_t pcclAllReduceAsync(const void *sendbuff, void *recvbuff,
                                const pcclReduceDescriptor_t *descriptor,
                                const pcclComm_t *communicator,
                                pcclAsyncReduceOp_t *reduce_handle_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    PCCL_VALIDATE(reduce_handle_out != nullptr, pcclInvalidArgument);

    const size_t count = descriptor->count;
    const auto datatype = descriptor->src_descriptor.datatype;
    const auto quantization_algorithm = descriptor->quantization_options.algorithm;
    const auto quantized_datatype = quantization_algorithm == pcclQuantNone
                                        ? datatype
                                        : descriptor->quantization_options.quantized_datatype;
    const auto op = descriptor->op;
    const auto tag = descriptor->tag;

    const auto ccoip_data_type = getCCoIPDataType(datatype);
    if (!ccoip_data_type) {
        return pcclInvalidArgument;
    }
    const auto ccoip_quantized_data_type = getCCoIPDataType(quantized_datatype);
    if (!ccoip_quantized_data_type) {
        return pcclInvalidArgument;
    }
    const auto ccoip_quantization_algorithm = getCCoIPQuantizationAlgorithm(quantization_algorithm);
    if (!ccoip_quantization_algorithm) {
        return pcclInvalidArgument;
    }
    const auto ccoip_op = getCCoIPReduceOp(op);
    if (!ccoip_op) {
        return pcclInvalidArgument;
    }

    int local_world_size{};
    PCCL_ERR_PROPAGATE(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE, &local_world_size));

    if (local_world_size < 2) {
        return pcclTooFewPeers;
    }

    if (!communicator->ccoip_client->allReduceAsync(sendbuff, recvbuff, count, *ccoip_data_type,
                                                    *ccoip_quantized_data_type,
                                                    *ccoip_quantization_algorithm, *ccoip_op, tag)) {
        return pcclInvalidArgument;
    }

    *reduce_handle_out = pcclAsyncReduceOp_t{
        .comm = const_cast<pcclComm_t *>(communicator),
        .tag = tag,
    };
    return pcclSuccess;
}

pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff,
                           const pcclReduceDescriptor_t *descriptor,
                           const pcclComm_t *communicator,
                           pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    pcclAsyncReduceOp_t reduce_handle{};
    PCCL_ERR_PROPAGATE(pcclAllReduceAsync(sendbuff, recvbuff, descriptor, communicator, &reduce_handle));
    PCCL_ERR_PROPAGATE(pcclAwaitAsyncReduce(&reduce_handle, reduce_info_out));
    return pcclSuccess;
}


pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle,
                                  pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(reduce_handle != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(reduce_handle->comm != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(reduce_handle->comm->ccoip_client != nullptr, pcclInvalidUsage);

    if (!reduce_handle->comm->ccoip_client->joinAsyncReduce(reduce_handle->tag)) {
        return pcclRankConnectionLost;
    }

    if (reduce_info_out != nullptr) {
        std::optional<ccoip::ccoip_reduce_info_t> info{};
        if (!reduce_handle->comm->ccoip_client->getAsyncReduceInfo(reduce_handle->tag, info)) [[unlikely]] {
            return pcclInvalidUsage;
        }

        if (!info) [[unlikely]] {
            return pcclInternalError;
        }

        reduce_info_out->local_world_size = info->world_size;
        reduce_info_out->tx_bytes = info->tx_bytes;
        reduce_info_out->rx_bytes = info->rx_bytes;
    }

    return pcclSuccess;
}


pcclResult_t pcclAllReduceMultipleWithRetry(const pcclReduceOpDescriptor_t *descriptors, const size_t count,
                                            const pcclComm_t *communicator,
                                            pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out,
                                            const int max_in_flight) {
    PCCL_VALIDATE_INITIALIZED();

    PCCL_VALIDATE(descriptors != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(count > 0, pcclInvalidArgument);
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    PCCL_VALIDATE(max_in_flight > 0, pcclInvalidArgument);

    int local_world_size{};
    if (pcclGetAttribute(communicator, PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE, &local_world_size) != pcclSuccess) {
        return pcclInvalidUsage;
    }

    if (local_world_size < 2) {
        return pcclTooFewPeers;
    }

    size_t total_tx = 0;
    size_t total_rx = 0;

    uint32_t in_flight = 0;

    std::vector<std::optional<pcclAsyncReduceOp_t>> reduce_handles{};
    reduce_handles.resize(count);

    // launch as many async all reduce operations up-front
    // as possible, up to the max_in_flight limit
    for (size_t i = 0; i < count; ++i) {
        const pcclReduceOpDescriptor_t &op_descriptor = descriptors[i];

        pcclAsyncReduceOp_t handle{};
        PCCL_ERR_PROPAGATE(pcclAllReduceAsync(
            op_descriptor.recvbuf,
            op_descriptor.recvbuf,
            &op_descriptor.descriptor,
            communicator,
            &handle
        ));
        reduce_handles[i] = handle;

        in_flight++;

        if (in_flight >= static_cast<uint32_t>(max_in_flight)) {
            break;
        }
    }

    // stores all indices of completed reduce operations
    // in range [0, count)
    std::unordered_set<size_t> completed_ops{};

    while (local_world_size > 1) {
        bool all_done = true;
        for (size_t i = 0; i < count; ++i) {
            const auto &reduce_handle_opt = reduce_handles[i];
            if (reduce_handle_opt == std::nullopt) {
                // no reduce handle exists for this operation yet,
                // so it needs to be launched.
                if (completed_ops.contains(i)) {
                    continue;
                }
                // the reduce handle is not yet completed
                if (in_flight >= static_cast<uint32_t>(max_in_flight)) {
                    continue;
                }

                const pcclReduceOpDescriptor_t &op_descriptor = descriptors[i];

                pcclAsyncReduceOp_t handle{};
                PCCL_ERR_PROPAGATE(pcclAllReduceAsync(
                    op_descriptor.recvbuf,
                    op_descriptor.recvbuf,
                    &op_descriptor.descriptor,
                    communicator,
                    &handle
                ));
                reduce_handles[i] = handle;

                in_flight++;
            }

            const auto &reduce_handle = reduce_handle_opt.value();

            pcclReduceInfo_t reduce_info{};
            const pcclResult_t reduce_status = pcclAwaitAsyncReduce(&reduce_handle, &reduce_info);
            PCCL_ERR_PROPAGATE(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE, &local_world_size));

            if (reduce_status != pcclSuccess) {
                reduce_handles[i] = std::nullopt;

                // Wait for all ongoing ops to finish or fail before retry
                for (size_t j = 0; j < count; ++j) {
                    if (j == i) {
                        continue;
                    }
                    const auto &h_j = reduce_handles[j];
                    if (h_j != std::nullopt) {
                        const pcclResult_t s_j = pcclAwaitAsyncReduce(&h_j.value(), nullptr);
                        if (s_j == pcclSuccess) {
                            completed_ops.insert(j);
                        }
                        in_flight--;
                    }
                    reduce_handles[i] = std::nullopt;
                }

                // some async operation failed, we are not done yet
                all_done = false;
            }

            // success for this handle
            reduce_handles[i] = std::nullopt;
            completed_ops.insert(i);

            total_tx += reduce_info.tx_bytes;
            total_rx += reduce_info.rx_bytes;

            in_flight--;
        }
        if (all_done) {
            break;
        }
    }

    if (reduce_info_out != nullptr) {
        *reduce_info_out = pcclReduceInfo_t{
            .local_world_size = static_cast<uint32_t>(local_world_size),
            .tx_bytes = total_tx,
            .rx_bytes = total_rx,
        };
    }

    if (local_world_size == 1) {
        // if we are alone, just finalize all handles and return
        for (size_t i = 0; i < count; ++i) {
            const auto &reduce_handle_opt = reduce_handles[i];
            if (reduce_handle_opt != std::nullopt) {
                pcclAwaitAsyncReduce(&reduce_handle_opt.value(), nullptr); // don't care about status
            }
        }
        return pcclTooFewPeers;
    }

    return pcclSuccess;
}

inline size_t pcclDataTypeSize(const pcclDataType_t datatype) {
    switch (datatype) {
        case pcclUint8:
        case pcclInt8:
            return 1;
        case pcclUint16:
            return 2;
        case pcclUint32:
        case pcclInt32:
        case pcclFloat:
            return 4;
        case pcclUint64:
        case pcclInt64:
        case pcclDouble:
            return 8;
        default:
            return 0;
    }
}

pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *communicator, pcclSharedState_t *shared_state,
                                        pcclSharedStateSyncInfo_t *PCCL_NULLABLE sync_info_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);

    // if there are any async collective operations running, we cannot sync the shared state
    if (communicator->ccoip_client->isAnyCollectiveComsOpRunning()) {
        return pcclPendingAsyncOps;
    }

    // sync shared state
    ccoip_shared_state_t shared_state_internal{};
    shared_state_internal.revision = shared_state->revision;
    for (size_t i = 0; i < shared_state->count; ++i) {
        const pcclTensorInfo_t &entry = shared_state->infos[i];
        const size_t entry_bytes = entry.count * pcclDataTypeSize(entry.datatype);
        auto ccoip_data_type = getCCoIPDataType(entry.datatype);
        if (!ccoip_data_type) {
            return pcclInvalidArgument;
        }
        auto device_type = getCCoIPDeviceType(entry.device_type);
        if (!device_type) {
            return pcclInvalidArgument;
        }

#ifndef PCCL_HAS_CUDA_SUPPORT
        if (device_type == pcclDeviceCuda) {
            LOG(WARN) <<
                    "PCCL is not built with CUDA support. Please use a cuda-enabled distribution of PCCL to use cuda tensors with PCCL!";
            return pcclInvalidArgument;
        }
#endif

        shared_state_internal.entries.push_back(ccoip_shared_state_entry_t{
            .key = entry.name,
            .data_type = *ccoip_data_type,
            .device_type = *device_type,
            .data_ptr = entry.data,
            .data_size = entry_bytes,
            .allow_content_inequality = entry.allow_content_inequality
        });
    }
    ccoip_shared_state_sync_info_t info{};
    if (!communicator->ccoip_client->syncSharedState(shared_state_internal, info)) {
        return pcclInvalidUsage;
    }

    // revision may change after sync
    shared_state->revision = shared_state_internal.revision;

    if (sync_info_out != nullptr) {
        *sync_info_out = pcclSharedStateSyncInfo_t{
            .tx_bytes = info.tx_bytes,
            .rx_bytes = info.rx_bytes,
        };
    }
    return pcclSuccess;
}

struct pcclMasterInstanceState_t {
    std::unique_ptr<ccoip::CCoIPMaster> master_handler;
};

pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address, pcclMasterInstance_t **p_master_handle_out) {
    PCCL_VALIDATE(p_master_handle_out != nullptr, pcclInvalidArgument);
    *p_master_handle_out = new pcclMasterInstance_t{
        .master_handler = std::make_unique<ccoip::CCoIPMaster>(listen_address),
    };
    return pcclSuccess;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclRunMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(master_instance->master_handler != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->launch()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclInterruptMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(master_instance->master_handler != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->interrupt()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(master_instance->master_handler != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->join()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

pcclResult_t pcclDestroyMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(master_instance->master_handler != nullptr, pcclInvalidArgument);
    master_instance->master_handler = nullptr;
    delete master_instance;
    return pcclSuccess;
}

pcclResult_t pcclGetBuildInfo(pcclBuildInfo_t *info) {
    PCCL_VALIDATE(info != nullptr, pcclInvalidArgument);
#ifdef PCCL_HAS_CUDA_SUPPORT
    info->has_cuda_support = true;
#else
    info->has_cuda_support = false;
#endif
    return pcclSuccess;
}
