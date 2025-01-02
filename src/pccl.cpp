#include "pccl.h"
#include "pccl_internal.hpp"
#include <optional>
#include <ccoip_master.hpp>

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

    switch (attribute) {
        case PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE: {
            const size_t world_size = communicator->ccoip_client->getWorldSize();
            *p_attribute_out = static_cast<int>(world_size);
            break;
        }
        default: {
            [[unlikely]] return pcclInvalidArgument;
        }
    }
    return pcclSuccess;
}

pcclResult_t pcclTopologySaveGraph(const pcclComm_t *communicator, const char *filename) {
    return pcclSuccess;
}

pcclResult_t pcclSaveReducePlan(const pcclComm_t *communicator, const char *filename) {
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
        if (!communicator->ccoip_client->interrupt()) [[unlikely]] {
            status = pcclInternalError;
            goto cleanup;
        }
        if (!communicator->ccoip_client->join()) [[unlikely]] {
            status = pcclInternalError;
            goto cleanup;
        }
        status = pcclMasterConnectionFailed;
        goto cleanup;
    }
    if (!communicator->ccoip_client->updateTopology()) [[unlikely]] {
        status = pcclMasterConnectionFailed;
        goto cleanup;
    }
    cleanup:
    communicator->ccoip_client = nullptr;
    return status;
}

static std::optional<ccoip::ccoip_data_type_t> getCCoIPDataType(const pcclDataType_t datatype) {
    switch (datatype) {
        case pcclUint8: return ccoip::ccoipUint8;
        case pcclUint16: return ccoip::ccoipUint16;
        case pcclUint32: return ccoip::ccoipUint32;
        case pcclUint64: return ccoip::ccoipUint64;
        case pcclInt8: return ccoip::ccoipInt8;
        case pcclInt16: return ccoip::ccoipInt16;
        case pcclInt32: return ccoip::ccoipInt32;
        case pcclInt64: return ccoip::ccoipInt64;
        case pcclFloat: return ccoip::ccoipFloat;
        case pcclDouble: return ccoip::ccoipDouble;
    }
    return std::nullopt;
}

static std::optional<ccoip::ccoip_reduce_op_t> getCCoIPReduceOp(const pcclRedOp_t op) {
    switch (op) {
        case pcclSum: return ccoip::ccoip_reduce_op_t::ccoipOpSum;
        case pcclAvg: return ccoip::ccoip_reduce_op_t::ccoipOpAvg;
        case pcclProd: return ccoip::ccoip_reduce_op_t::ccoipOpProd;
        case pcclMax: return ccoip::ccoip_reduce_op_t::ccoipOpMax;
        case pcclMin: return ccoip::ccoip_reduce_op_t::ccoipOpMin;
    }
    return std::nullopt;
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
        return pcclInvalidUsage;
    }

    // update the topology
    if (!communicator->ccoip_client->updateTopology()) [[unlikely]] {
        return pcclInvalidUsage;
    }

    return pcclSuccess;
}

pcclResult_t pcclAllReduceAsync(const void *sendbuff, void *recvbuff, const size_t count, const pcclDataType_t datatype,
                                const pcclRedOp_t op, const uint64_t tag, const pcclComm_t *communicator,
                                pcclAsyncReduceOp_t *reduce_handle_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    PCCL_VALIDATE(reduce_handle_out != nullptr, pcclInvalidArgument);

    auto ccoip_data_type = getCCoIPDataType(datatype);
    if (!ccoip_data_type) {
        return pcclInvalidArgument;
    }
    auto ccoip_op = getCCoIPReduceOp(op);
    if (!ccoip_op) {
        return pcclInvalidArgument;
    }
    if (!communicator->ccoip_client->allReduceAsync(sendbuff, recvbuff, count, *ccoip_data_type, *ccoip_op, tag)) {
        return pcclInvalidUsage;
    }

    *reduce_handle_out = pcclAsyncReduceOp_t{
        .comm = const_cast<pcclComm_t *>(communicator),
        .tag = tag,
    };
    return pcclSuccess;
}

pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff, const size_t count, const pcclDataType_t datatype,
                           const pcclRedOp_t op, const uint64_t tag, const pcclComm_t *communicator,
                           pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    pcclAsyncReduceOp_t reduce_handle{};
    pcclAllReduceAsync(sendbuff, recvbuff, count, datatype, op, tag, communicator, &reduce_handle);
    pcclAwaitAsyncReduce(&reduce_handle, reduce_info_out);
    return pcclSuccess;
}


pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle,
                                  pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(reduce_handle != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(reduce_handle->comm != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(reduce_handle->comm->ccoip_client != nullptr, pcclInvalidUsage);

    if (!reduce_handle->comm->ccoip_client->joinAsyncReduce(reduce_handle->tag)) {
        return pcclInvalidUsage;
    }

    if (reduce_info_out != nullptr) {
        std::optional<ccoip::ccoip_reduce_info_t> info{};
        if (!reduce_handle->comm->ccoip_client->getAsyncReduceInfo(reduce_handle->tag, info)) [[unlikely]] {
            return pcclInvalidUsage;
        }

        if (!info) [[unlikely]] {
            return pcclInternalError;
        }

        reduce_info_out->world_size = info->world_size;
        reduce_info_out->tx_bytes = info->tx_bytes;
        reduce_info_out->rx_bytes = info->rx_bytes;
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
        shared_state_internal.entries.push_back(ccoip_shared_state_entry_t{
            .key = entry.name,
            .data_type = *ccoip_data_type,
            .value = std::span(static_cast<std::byte *>(entry.data), entry_bytes),
            .allow_content_inequality = entry.allow_content_inequality
        });
    }
    ccoip_shared_state_sync_info_t info{};
    if (!communicator->ccoip_client->syncSharedState(shared_state_internal, info)) [[unlikely]] {
        return pcclInvalidUsage;
    }
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
