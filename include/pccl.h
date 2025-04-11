#pragma once

#include <ccoip_inet.h>

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#include <stdbool.h>
#endif

#ifdef _MSC_VER
#define PCCL_EXPORT __declspec(dllexport)
#else
#define PCCL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum pcclResult_t {
    pcclSuccess = 0,
    pcclNotInitialized = 1,
    pcclInternalError = 2,
    pcclInvalidArgument = 3,
    pcclInvalidUsage = 4,
    pcclTooFewPeers = 5,
    pcclMasterConnectionFailed = 6,
    pcclRankConnectionFailed = 7,
    pcclRankConnectionLost = 8,
    pcclNoSharedStateAvailable = 9,
    pcclPendingAsyncOps = 10,
    pcclUpdateTopologyFailed = 11,
    pcclTopologyOptimizationFailed = 12,
} pcclResult_t;

typedef enum pcclDataType_t {
    pcclUint8 = 0,
    pcclInt8 = 1,
    pcclInt16 = 2,
    pcclUint16 = 3,
    pcclUint32 = 4,
    pcclInt32 = 5,
    pcclUint64 = 6,
    pcclInt64 = 7,
    pcclFloat = 8,
    pcclDouble = 9
} pcclDataType_t;

typedef enum pcclDeviceType_t {
    pcclDeviceCpu = 0,
    pcclDeviceCuda = 1,
} pcclDeviceType_t;

typedef enum pcclRedOp_t {
    pcclSum,
    pcclAvg,
    pcclProd,
    pcclMax,
    pcclMin
} pcclRedOp_t;

typedef enum pcclAttribute_t {
    /// Total number of peers part of the run
    PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE = 1,

    /// Number of peers in the peer group that this peer is part of
    PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE = 2,

    /// Number of distinct peer groups in the run
    PCCL_ATTRIBUTE_NUM_DISTINCT_PEER_GROUPS = 3,

    /// Number of peers in the largest peer group
    PCCL_ATTRIBUTE_LARGEST_PEER_GROUP_WORLD_SIZE = 4,

} pcclAttribute_t;

typedef struct pcclCommCreateParams_t {
    /**
     * The address of the master node to connect to.
     */
    ccoip_socket_address_t master_address;

    /**
     * The world is split into peer groups, where each peer group is a set of peers that can communicate with each other.
     * Shared state distribution and collective communications operations will span only across peers in the same peer group.
     * To allow all peers to communicate with each other, all peers must be in the same peer group, e.g. by setting this to 0.
     * The peer group is a 32-bit unsigned integer whose identity determines the peer group the client is part of.
     */
    uint32_t peer_group;
} pcclCommCreateParams_t;

typedef struct pcclComm_t pcclComm_t;

typedef struct pcclRankInfo_t pcclRankInfo_t;

typedef struct pcclReduceInfo_t {
    /// The world size used for the reduce operation. Is equal to the number of peers that participated in the operation if completed successfully.
    uint32_t local_world_size;
    uint64_t tx_bytes;
    uint64_t rx_bytes;
} pcclReduceInfo_t;

typedef enum pcclDistributionHint_t {
    PCCL_DISTRIBUTION_HINT_NONE = 0,
    PCCL_NORMAL_DISTRIBUTION_HINT = 1,
    PCCL_UNIFORM_DISTRIBUTION_HINT = 2
} pcclDistributionHint_t;

typedef struct pcclReduceOperandDescriptor_t {
    pcclDataType_t datatype;
    pcclDistributionHint_t distribution_hint;
} pcclReduceOperandDescriptor_t;

typedef enum pcclQuantizationAlgorithm_t {
    pcclQuantNone = 0,
    pcclQuantMinMax = 1,
    pcclQuantZeroPointScale = 2
} pcclQuantizationAlgorithm_t;

typedef struct pcclQuantizationOptions_t {
    pcclDataType_t quantized_datatype;
    pcclQuantizationAlgorithm_t algorithm;
} pcclQuantizationOptions_t;

typedef struct pcclReduceDescriptor_t {
    size_t count;
    pcclRedOp_t op;
    uint64_t tag;
    pcclReduceOperandDescriptor_t src_descriptor;
    pcclQuantizationOptions_t quantization_options;
} pcclReduceDescriptor_t;

typedef struct pcclReduceSingleDescriptor_t {
    void *sendbuf;
    void *recvbuf;
    pcclReduceDescriptor_t descriptor;
} pcclReduceOpDescriptor_t;

typedef struct pcclAsyncReduceOp_t {
    pcclComm_t *comm;
    uint64_t tag;
} pcclAsyncReduceOp_t;

/**
 * Uniquely identifies a rank in a communicator.
 * Even if a rank ordinal is reused after a rank is removed and a new rank added in its place, the UUID will be unique.
 */
typedef struct pcclRankUuid_t {
    uint8_t data[16];
} pcclRankUuid_t;

typedef struct pcclTensorInfo_t {
    const char *name;
    void *data;
    size_t count;
    pcclDataType_t datatype;
    pcclDeviceType_t device_type;
    bool allow_content_inequality;
} pcclTensorInfo_t;

typedef struct pcclSharedState_t {
    uint64_t revision;
    size_t count;
    pcclTensorInfo_t *infos;
} pcclSharedState_t;

typedef struct pcclSharedStateSyncInfo_t {
    uint64_t tx_bytes;
    uint64_t rx_bytes;
} pcclSharedStateSyncInfo_t;

typedef struct pcclMasterInstanceState_t pcclMasterInstance_t;


typedef struct pcclBuildInfo_t {
    /**
     * Whether this pccl build was compiled with CUDA support
     */
    bool has_cuda_support;
} pcclBuildInfo_t;

#define PCCL_NULLABLE /* nothing */

/**
 * Initializes the pccl library.
 * Must be called before pccl library functions are used.
 */
PCCL_EXPORT pcclResult_t pcclInit(void);

/**
 * Creates a new communicator.
 * @param params Parameters to create the communicator with.
 * @param comm_out Pointer to the communicator to be created.
 * @return @code pcclSuccess@endcode if the communicator was created successfully.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 */
PCCL_EXPORT pcclResult_t pcclCreateCommunicator(const pcclCommCreateParams_t *params,
                                                pcclComm_t **comm_out);

/**
 * Gets an attribute of a communicator.
 * @param communicator The communicator to get the attribute from.
 * @param attribute The attribute type to retrieve the value of.
 * @param p_attribute_out Pointer to the attribute value to be set.
 *
 * @return @code pcclSuccess@endcode if the attribute was successfully retrieved.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 * @return @code pcclInvalidArgument@endcode if the communicator or pAttributeOut is null or if the specified attribute type is unknown.
 */
PCCL_EXPORT pcclResult_t pcclGetAttribute(const pcclComm_t *communicator, pcclAttribute_t attribute,
                                          int *p_attribute_out);

/**
 * Destroys a communicator.
 * Will block until the threads associated with the communicator have exited.
 *
 * @param communicator The communicator to destroy.
 *
 * @return pcclSuccess if the communicator was destroyed successfully.
 */
PCCL_EXPORT pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator);

/**
 * Establishes a connection to a master node & waits until all peers have connected.
 * This function can block for a long time depending on how frequently the existing peers agree to accept new peers.
 * This function must be called on a communicator for the communicator to be usable.
 *
 * @param communicator The communicator to connect to the master node.
 *
 * @return @code pcclSuccess@endcode if the connection was established successfully.
 * @return @code pcclInvalidArgument@endcode if the communicator is null.
 * @return @code pcclInvalidUsage@endcode if the communicator is already connected to a master node.
 */
PCCL_EXPORT pcclResult_t pcclConnect(pcclComm_t *communicator);

/**
 * Update the topology of a communicator if required.
 * Topology updates are required when new peers join, in which case @code pcclUpdateTopology@endcode will
 * automatically handle connection establishment with the new peer(s).
 * Topology updates can also be triggered by the master node in response to bandwidth changes or other events.
 * This function will block until the topology update is complete.
 *
 * @param communicator The communicator to update the topology of.
 *
 * @return @code pcclSuccess@endcode if the topology was updated successfully.
 * @return @code pcclUpdateTopologyFailed@endcode e.g. if p2p connections could not be established with new peers.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 */
PCCL_EXPORT pcclResult_t pcclUpdateTopology(pcclComm_t *communicator);

/**
 * Returns whether the communicator has pending peers to accept.
 * This function can be used to determine if @ref pcclUpdateTopolgy needs to be called.
 * If this function returns true, it is recommended to call @ref pcclUpdateTopology to avoid keeping pending peers waiting.
 * If this function returns false, the call to @ref pcclUpdateTopology can be skipped without risk of delaying pending peers.
 * This is useful if async collective communications are ongoing that would otherwise have to be awaited before calling @ref pcclUpdateTopology.
 * All peers must call this function jointly. Only once all peers have called @ref pcclArePeersPending will this function unblock - just like @ref pcclUpdateTopology and other phase-changing functions.
 * NOTE: This function could technically return a state that becomes dirty the next moment, so very unluckily timed peer joins would be skipped if @ref pcclUpdateTopology is then not invoked based on the return value of this function.
 * The worst that can happen here is that the peer is accepted in a subsequent call to @ref pcclUpdateTopology, which would also have been the result if the peer joined single digit milliseconds later without employing the are peers pending guard.
 *
 * @param communicator The communicator to check for pending peers.
 * @param pending_out Pointer to the boolean to be set to true if there are pending peers to accept, false otherwise.
 *
 * @return @code pcclSuccess@endcode if the pending status was successfully retrieved.
 * @return @code pcclInvalidArgument@endcode if the communicator or pending_out is null.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 */
PCCL_EXPORT pcclResult_t pcclArePeersPending(const pcclComm_t *communicator, bool *pending_out);


/**
 * Called to optimize the topology of a communicator.
 * After topology updates, it is recommended that the topology be optimized to improve performance of collective communications operations.
 * @param communicator the communicator
 * @return @code pcclSuccess@endcode if the topology was successfully optimized / unchanged without error.
 * @return @code pcclTopologyOptimizationFailed@endcode if an internal error occurred during the topology optimization.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 */
PCCL_EXPORT pcclResult_t pcclOptimizeTopology(const pcclComm_t *communicator);

/**
 * Performs an all reduce operation on a communicator. Blocks until the all reduce is complete.
 *
 * @param sendbuff The buffer to send data from.
 * @param recvbuff The buffer to receive data into.
 * @param descriptor Descriptor containing parameters for configuring the reduce operation.
 * @param communicator The communicator to perform the operation on.
 * @param reduce_info_out The reduce info to be filled with information about the operation.
 *
 * @return @code pcclSuccess@endcode if the all reduce operation was successful.
 * @return @code pcclRankConnectionLost@endcode if the connection to a peer was lost during the operation either gracefully or due to a network error.
 * @return @code pcclInvalidArgument@endcode if the communicator, sendbuff, recvbuff, count is less or equal to zero, or tag is less than zero.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 * @return @code pcclInvalidUsage@endcode if the communicator is not connected to a master node.
 */
PCCL_EXPORT pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff,
                                       const pcclReduceDescriptor_t *descriptor,
                                       const pcclComm_t *communicator,
                                       pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out);

/**
* Performs an all reduce operation on a communicator. Async version of @code pcclAllReduce@endcode.
*
* @param sendbuff The buffer to send data from.
* @param recvbuff The buffer to receive data into.
* @param descriptor Descriptor containing parameters for configuring the reduce operation.
* @param communicator The communicator to perform the operation on.
* @param reduce_handle_out The reduce op handle to be filled with an async handle to the operation.
*
* @return @code pcclSuccess@endcode if the all reduce operation was successful.
* @return @code pcclInvalidArgument@endcode if the communicator, sendbuff, recvbuff, count is less or equal to zero, or tag is less than zero.
* @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
* @return @code pcclInvalidUsage@endcode if the communicator is not connected to a master node.
* @return @code pcclInvalidArgument@endcode if the reduce handle output is null.
*/
PCCL_EXPORT pcclResult_t pcclAllReduceAsync(const void *sendbuff, void *recvbuff,
                                            const pcclReduceDescriptor_t *descriptor,
                                            const pcclComm_t *communicator,
                                            pcclAsyncReduceOp_t *reduce_handle_out);


/**
 * Performs multiple all reduces concurrently.
 * If any of the all reduce operations fail, the function will await all outstanding operations and retry the failed ones.
 * The function will not complete until all operations have completed successfully or the local world size has dropped below 2.
 *
 * @note Different reduce operations may have been performed with different local world sizes if peers dropped out during the operation.
 * The local world size populated in the reduce info will be the local world size after all operations have completed. No veracity guarantees are made about this value beyond for heuristic usage.
 *
 * @param descriptors the array of descriptors describing all reduce operations to perform. Each descriptor contains the send and receive buffers, the count, the operation type, and the tag.
 * @param count the number of descriptors in the array. This is the number of all reduce operations to perform.
 * @param communicator the communicator to perform the all reduce operations on.
 * @param reduce_info_out the reduce info to be filled with information about the operation.
 * @param max_in_flight the maximum number of all reduce operations to perform concurrently. This is the maximum number of all reduce operations that can be in flight at any given time. It is expected that the actual number of in-flight operations will be less than this value.
 * @return @code pcclSuccess@endcode if all reduce operations were successful.
 * @return @code pcclInvalidArgument@endcode if the descriptors array is null, the count is less than or equal to zero, the communicator is null, or the max_in_flight is less than or equal to zero.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 * @return @code pcclInvalidUsage@endcode if the communicator is not connected to a master node.
 */
PCCL_EXPORT pcclResult_t pcclAllReduceMultipleWithRetry(const pcclReduceOpDescriptor_t *descriptors,
                                                        size_t count,
                                                        const pcclComm_t *communicator,
                                                        pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out,
                                                        int max_in_flight);

/**
 * Awaits the completion of an async reduce operation. Blocks until the operation is complete.
 *
 * @param reduce_handle The handle to the async reduce operation.
 * @param reduce_info_out The reduce info to be filled with information about the operation.
 *
 * @return @code pcclSuccess@endcode if the async reduce operation was successful.
 * @return @code pcclInvalidArgument@endcode if the reduce handle is null or invalid.
*  @return @code pcclRankConnectionLost@endcode if the connection to a peer was lost during the operation either gracefully or due to a network error.
 */
PCCL_EXPORT pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle,
                                              pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out);

/**
 * Synchronizes the shared state between all peers that are currently accepted.
 * If the shared state revision of this peer is outdated, the shared state will be updated.
 * The function will not unblock until it is confirmed all peers have the same shared state revision.
 * This function will fail if the world size is less than 2.
 * @param communicator The communicator to synchronize the shared state on.
 * @param shared_state The shared state referencing user-owned data to be synchronized.
 * @param sync_info_out shared state synchronization info.
 * @return @code pcclSuccess@endcode if the shared state was synchronized successfully.
 * @return @code pcclInvalidArgument@endcode if the communicator or shared_state is null.
 * @return @code pcclInvalidUsage@endcode if the communicator is not connected to a master node.
 */
PCCL_EXPORT pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *communicator,
                                                    pcclSharedState_t *shared_state,
                                                    pcclSharedStateSyncInfo_t *PCCL_NULLABLE sync_info_out);

/**
 * Creates a master node handle.
 * @param listen_address The address to listen for incoming connections on.
 * @param p_master_handle_out Pointer to the master node handle to be created.
 * @return @code pcclSuccess@endcode if the master node handle was created successfully.
 */
PCCL_EXPORT pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address,
                                          pcclMasterInstance_t **p_master_handle_out);

/**
 * Runs a master node. This function is non-blocking.
 * @param master_instance The master node handle to run.
 * @return @code pcclSuccess@endcode if the master node was run successfully.
 * @return @code pcclInvalidArgument@endcode if the master handle is already running.
 */
PCCL_EXPORT pcclResult_t pcclRunMaster(pcclMasterInstance_t *master_instance);

/**
 * Interrupts a master node.
 * @param master_instance The master node handle to interrupt.
 * @return @code pcclSuccess@endcode if the master node was interrupted successfully.
 */
PCCL_EXPORT pcclResult_t pcclInterruptMaster(pcclMasterInstance_t *master_instance);

/**
 * Awaits termination of a master node. This function is blocking.
 * @param master_instance The master node handle to await termination of.
 * @return @code pcclSuccess@endcode if the master node was terminated successfully.
 * @return @code pcclInvalidArgument@endcode if the master handle is not running / was never interrupted.
 */
PCCL_EXPORT pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t *master_instance);

/**
 * Destroys a master node. Must only be called after pcclMasterAwaitTermination has been called and returned.
 * @param master_instance The master node handle to destroy.
 * @return @code pcclSuccess@endcode if the master node was destroyed successfully.
 */
PCCL_EXPORT pcclResult_t pcclDestroyMaster(pcclMasterInstance_t *master_instance);

/**
 * Obtains build info of this pccl build
 * @param info output where the build information will be written to
 * @return @code pcclSuccess@endcode
 */
PCCL_EXPORT pcclResult_t pcclGetBuildInfo(pcclBuildInfo_t *info);

#ifdef __cplusplus
}
#endif
