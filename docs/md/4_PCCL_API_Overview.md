# PCCL API Overview

## Initialization and Finalization

### `pcclInit`

```c
pcclResult_t pcclInit(void);
```

Initializes the PCCL library. Must be called once in a process before any other PCCL functions.

- Returns `pcclSuccess` on success, or an error code (e.g., pcclSystemError) if something basic (like a socket layer)
  can’t be initialized.

### `pcclDestroyCommunicator`

```c
pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator);
```

Destroys a communicator object previously created, blocking until all internal threads and connections for that
communicator have cleanly exited.

- Typically called at the end of a training process (or whenever you no longer need the communicator).
- The library remains valid, so you could create another communicator afterwards if you wish.

PCCL does not strictly require a global “finalization” call after you destroy all communicators.

## Master-Related Functions

A separate set of functions manage the master node, which is responsible for orchestrating the run. In typical usage,
you either use these calls directly in your own “master” process or run the provided `ccoip_master` executable that does
it internally.

### `pcclCreateMaster`

```c
pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address,
                              pcclMasterInstance_t **p_master_handle_out);
```

Creates a new master node handle, binding to the specified IP and port for listening to incoming client connections (by
default, port `48148`).

### `pcclRunMaster`

```c
pcclResult_t pcclRunMaster(pcclMasterInstance_t *master_instance);
```

Starts the master node’s event loop non-blockingly. After this call, the master is actively listening to clients and
orchestrating their states.

- Returns `pcclSuccess` or `pcclInvalidArgument` if the handle was invalid or already running.

### `pcclInterruptMaster`

```c
pcclResult_t pcclInterruptMaster(pcclMasterInstance_t *master_instance);
```

Signals the master to shut down gracefully. Typically invoked when you want to stop the entire run.

### `pcclMasterAwaitTermination`

```c
pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t *master_instance);
```

Blocks the calling thread until the master node is fully terminated (i.e., the event loop ends and no more connections
remain open).

### `pcclDestroyMaster`

```c
pcclResult_t pcclDestroyMaster(pcclMasterInstance_t *master_instance);
```

Frees resources associated with the master handle. Must only be called after a successful
`pcclMasterAwaitTermination()`.

### Typical Master Lifecycle

1. `pcclCreateMaster` ⇒ obtains a handle
2. `pcclRunMaster` ⇒ starts listening + the master’s event loop, returns immediately
3. Optionally handle OS signals (e.g., ^C) to call `pcclInterruptMaster`
4. `pcclMasterAwaitTermination` ⇒ blocks until master finishes
5. `pcclDestroyMaster` ⇒ finalize

## Creating & Managing Communicators

Clients that want to join a run do the following:

### `pcclCreateCommunicator`

```c
pcclResult_t pcclCreateCommunicator(const pcclCommCreateParams_t *params,
                                    pcclComm_t **comm_out);
```

Allocates and initializes a communicator object.

- Parameters:
    - `parameters->master_address`: The `ccoip_socket_address_t` specifying the master’s IP and port.
    - `parameters->peer_group`: An integer identifying the group of peers that will do collectives among themselves (0
      is a common default if you only have one group).
- Returns `pcclSuccess`, or `pcclNotInitialized` if `pcclInit` wasn’t called, or `pcclInvalidArgument` if any pointer is
  null.

### `pcclConnect`

```c
pcclResult_t pcclConnect(pcclComm_t *communicator);
```

Establishes a connection to the master node and blocks until the master orchestrates acceptance. This can take time if
the run is busy or if existing peers have not yet reached a suitable “accept new peers” phase.

- If successful, the communicator can be used for `pcclAllReduce` and `pcclSynchronizeSharedState`.
- If it fails, you might see `pcclMasterConnectionFailed` or `pcclInvalidUsage`.

### `pcclUpdateTopology`

```c
pcclResult_t pcclUpdateTopology(pcclComm_t *communicator);
```

Triggers a “vote to accept new peers” and/or remove dropped ones. If all accepted peers in this group call
`pcclUpdateTopology` around the same time, the master transitions them to `CONNECTING_TO_PEERS` (or uses existing
connections for the stable ones). This may also detect that certain peers have vanished, removing them from the ring.

- If new peers are waiting to join, they become accepted after a successful consensus.
- Typically called once per iteration in a stable spot of your training loop.
- If anything fails (e.g., a peer can’t connect to new peers/newly assigned neighbors), you might get
  `pcclUpdateTopologyFailed` status. The recommended action is to retry the optimization. The `Hello World` example from
  section 2 implements this best practice.

### `pcclAreNewPeersPending`

```c
pcclResult_t pcclAreNewPeersPending(const pcclComm_t *communicator, bool *new_peers_pending_out);
```

Writes into `new_peers_pending_out` whether the communicator has pending peers to accept.
This function can be used to determine if `pcclUpdateTopolgy` needs to be called.
If the pending peers state is true, it is recommended to call `pcclUpdateTopology` to avoid keeping pending peers waiting.
If the pending peers state is false, the call to `pcclUpdateTopology` can be skipped without risk of delaying pending peers.
This is useful if async collective communications are ongoing that would otherwise have to be awaited before calling `pcclUpdateTopology`.
All peers must call this function jointly. Only once all peers have called `pcclArePeersPending` will this function unblock - just like `pcclUpdateTopology` and other phase-changing functions.

Note: This function could technically output a state that becomes dirty the next moment, so very unluckily timed peer joins would be skipped if `pcclUpdateTopology` is then not invoked based on the return value of this function.
The worst that can happen here is that the peer is accepted in a subsequent call to `pcclUpdateTopology`, which would also have been the result if the peer joined single digit milliseconds later without employing the are peers pending guard.


### `pcclOptimizeTopology`

```c
pcclResult_t pcclOptimizeTopology(const pcclComm_t *communicator);
```

Enters a “topology optimization” phase, where the master node reorders the ring or other topology to improve throughput.
The master may request missing bandwidth measurements from peers, which may trigger point to point bandwidth tests. This
will take some time and the method will block until all work requested by the master necessary for topology optimization
is complete. The master will then return an improved topology, which peers will immediately adopt by re-establishing p2p
connections via a similar process as `pcclUpdateTopology`, but without admitting new peers into the run.

- `Key`: This does *not* let in brand-new peers; Use `pcclUpdateTopology` for that.
- If anything fails (e.g., a peer can’t connect to newly assigned neighbors), you might get
  `pcclTopologyOptimizationFailed` status. The recommended action is to retry the optimization. The `Hello World`
  example from section 2 implements this best practice.

## Collective Operations

Currently, PCCL provides ring-based All-Reduce. Over time, more collectives (All-Gather, Scatter, etc.) may be
introduced. While currently PCCL only supports CPU buffers for collective operations, future versions may support GPU
buffers as well.

### `pcclAllReduce`

```c
typedef enum pcclDataType_t {
    pcclUint8 = 0,
    pcclInt8 = 1,
    ...
    pcclFloat = 8,
    pcclDouble = 9
} pcclDataType_t;


typedef enum pcclDistributionHint_t {
    PCCL_DISTRIBUTION_HINT_NONE = 0,
    PCCL_NORMAL_DISTRIBUTION_HINT = 1,
    PCCL_UNIFORM_DISTRIBUTION_HINT = 2
} pcclDistributionHint_t;

typedef struct pcclReduceOperandDescriptor_t {
    pcclDataType_t datatype;
    pcclDistributionHint_t distribution_hint;
} pcclReduceOperandDescriptor_t;

typedef enum pcclRedOp_t {
    pcclSum,
    pcclAvg,
    pcclProd,
    pcclMax,
    pcclMin
} pcclRedOp_t;

typedef struct pcclReduceDescriptor_t {
    size_t count;
    pcclRedOp_t op;
    uint64_t tag;
    pcclReduceOperandDescriptor_t src_descriptor;
    pcclQuantizationOptions_t quantization_options;
} pcclReduceDescriptor_t;

pcclResult_t pcclAllReduce(const void *sendbuff,
                           void *recvbuff,
                           const pcclReduceDescriptor_t *descriptor,
                           const pcclComm_t *communicator,
                           pcclReduceInfo_t *reduce_info_out);
```

A blocking All-Reduce call:

- `sendbuff`, `recvbuff`: Pointers to user-allocated memory. If they’re the same pointer, in-place reduce is done. (sendbuff == recvbuff is recommended for performance to avoid an internal memcpy).
- `descriptor`: A `pcclReduceDescriptor_t` structure that describes the reduce operation.
  - `descriptor->count`: Number of elements to reduce. This is the number of elements in the send buffer.
  - `descriptor->op`: The kind of reduce operation to perform (e.g., sum, min, max, etc.). This is a `pcclRedOp_t` enum.
  - `descriptor->tag`: A unique tag for this operation. Should be unique for each operation per step. This tag must not collide with respect to currently ongoing all reduces.
  - `descriptor->src_descriptor`: A `pcclReduceOperandDescriptor_t` structure that describes the source buffer. This is used to specify the type of data in the send buffer and how it should be interpreted.
    - `descriptor->src_descriptor.datatype`: The data type of the elements in the send buffer. This is a `pcclDataType_t` enum.
    - `descriptor->src_descriptor.distribution_hint`: A hint about the distribution of the data in the send buffer. This is a `pcclDistributionHint_t` enum.
  - `descriptor->quantization_options`:  If you want to compress data in flight.
- `reduce_info_out`: (optional) stats about how many bytes were transmitted or received.

#### Error Handling:

- `pcclSuccess`:  if the ring-based reduce completed normally.
- `pcclRankConnectionLost`: if a peer vanished mid-collective. You typically re-try the same reduce.
- `pcclInvalidUsage`: if you haven’t connected or if the communicator is in an illegal state.

### `pcclAllReduceAsync(...) / pcclAwaitAsyncReduce(...)`
```c
pcclResult_t pcclAllReduceAsync(const void *sendbuff,
                                void *recvbuff,
                                const pcclReduceDescriptor_t *descriptor,
                                const pcclComm_t *communicator,
                                pcclAsyncReduceOp_t *reduce_handle_out);

pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle,
                                  pcclReduceInfo_t *reduce_info_out);
```

Async all reduce operations are similar to the blocking version, but they return immediately and require a separate call
to `pcclAwaitAsyncReduce` to wait for completion.
Before calling functions like `pcclUpdateTopology`, `pcclOptimizeTopology`, or `pcclSynchronizeSharedState`, you should
ensure that all outstanding async operations have completed.

Note: `pcclAllReduceAsync` and `pcclAwaitAsyncReduce` may be called from a different thread than the main thread registered by the communicator.
PCCL will enforce that user facing apis are called from the main thread.
For `pcclAllReduceAsync` and `pcclAwaitAsyncReduce` however, the user is responsible for ensuring the following contract:
- `pcclAwaitAsyncReduce` must always be called from the same thread that called `pcclAllReduceAsync`. No guarantees about behavior of awaiting handles created by different threads are provided.
- The main thread registered by the communicator must not call other user facing apis while the concurrent thread is performing and awaiting collective communications operations using `pcclAllReduceAsync` and `pcclAwaitAsyncReduce`.
- The main thread must await the work done by the concurrent thread before calling other user facing apis itself again.
- Veracity of values returned by `pcclGetAttribute` calls for attributes `PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE` and `PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE` are only guaranteed up-to-date on the thread that called `pcclAwaitAsyncReduce`. Different threads may see outdated values and should only be relied upon for logging purposes.

The communicator registered main-thread may however call `pcclAreNewPeersPending` even if there are outstanding async operations issued by other threads.
If `pcclAreNewPeersPending` returns `true`, the main thread should call `pcclUpdateTopology` to ensure that the new peers are accepted into the run.
However, before doing so, the main thread must ensure that all outstanding async operations have completed awaiting the concurrent thread which must call `pcclAwaitAsyncReduce`.
If `pcclAreNewPeersPending` returns `false`, the main thread can safely skip calling `pcclUpdateTopology` without risking delaying the acceptance of new peers.
This is useful such that new compute can be started from the main thread without waiting for the completion of outstanding async operations without risking delaying the acceptance of new peers.

#### Error Handling:
Error handling is similar to the blocking version, but depending on whether the error occurs during the actual reduce operation
or is already apparent when calling `pcclAwaitAsyncReduce`, the error will be returned either by `pcclAllReduceAsync` or `pcclAwaitAsyncReduce`.
Generally, under correct usage you can expect `pcclAllReduceAsync` never to fail, while `pcclAwaitAsyncReduce` requires special recovery handling.

### `pcclAllReduceMultipleWithRetry`
```c
typedef struct pcclReduceSingleDescriptor_t {
    void *sendbuf;
    void *recvbuf;
    pcclReduceDescriptor_t descriptor;
} pcclReduceOpDescriptor_t;

pcclResult_t pcclAllReduceMultipleWithRetry(const pcclReduceOpDescriptor_t *descriptors,
                                            size_t count,
                                            const pcclComm_t *communicator,
                                            pcclReduceInfo_t *PCCL_NULLABLE reduce_info_out,
                                            int max_in_flight);
```
Performs multiple all reduces concurrently.
If any of the all reduce operations fail, the function will await all outstanding operations and retry the failed ones.
The function will not complete until all operations have completed successfully or the local world size has dropped below 2.
While it is possible to replicate the semantings of this function with `pcclAllReduceAsync` and `pcclAwaitAsyncReduce`,
we do not recommend doing so, as the retry logic is non-trivial and requires special care to ensure that the retry does not deadlock.

NOTE: Different reduce operations may have been performed with different local world sizes if peers dropped out during the operation.
The local world size populated in the reduce info will be the local world size after all operations have completed. No veracity guarantees are made about this value beyond for heuristic usage.

- `descriptors`: An array of `pcclReduceSingleDescriptor_t` structures, each containing the send and receive buffers and the reduce descriptor.
- `count`: The number of reduce operations to perform. Equal to the size of the `descriptors` array.
- `communicator`: The communicator object to use for the operation.
- `reduce_info_out`: (optional) stats about how many bytes were transmitted or received.
- `max_in_flight`: The maximum number of concurrent operations to perform. This is useful for limiting the number of concurrent operations to avoid overwhelming the network or the system.

### Shared-State Synchronization

`pcclSynchronizeSharedState`

```c
pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *communicator,
                                        pcclSharedState_t *shared_state,
                                        pcclSharedStateSyncInfo_t *sync_info_out);
```

Performs a group-wide check of user-defined “shared state” arrays (e.g., model or optimizer parameters).

- `shared_state->revision`: Must increment exactly by +1 each time. If you skip a number or reuse an old revision, the
  master rejects it (`REVISION_INCREMENT_VIOLATION`).
- `shared_state->infos`: An array of `pcclTensorInfo_t`, each describing a key string, pointer to memory, type, size,
  and a flag `allow_content_inequality`.
- If any mismatch is found (by hashing each tensor), that peer is told to re-request data from a designated “correct”
  peer.
- Once the entire group is consistent at that revision, the function returns.
- `sync_info_out->tx_bytes` / `sync_info_out->rx_bytes`: Tells you how many bytes were actually transmitted or received
  in that sync (often zero if you are already identical).

Common Usage:

1. You do local training steps that produce new weights.
2. Increment shared_state->revision by 1.
3. Call pcclSynchronizeSharedState(...).
4. If all peers have the same resulting data, no actual data transfer happens (zero bytes). Otherwise, some
   re-transmission occurs to unify them.

## Querying Attributes

### `pcclGetAttribute`

```c
pcclResult_t pcclGetAttribute(const pcclComm_t *communicator,
                              pcclAttribute_t attribute,
                              int *p_attribute_out);
```

Retrieves specific integer-valued attributes from the communicator. Currently:

- `PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE`: The total number of “accepted” peers in the run (master not included, as it is not considered a peer and has a strictly administrative role).
- `PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE`: The number of peers in the same peer group as this communicator.
- `PCCL_ATTRIBUTE_NUM_DISTINCT_PEER_GROUPS`: The number of distinct peer groups defined in the run. A peer group is considered defined if at least one peer has declared a particular integer value to be its peer group and has been accepted into the run.

Common use:

```c
int world_size;
pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size);
```

`Important`: This captures a snapshot of the current state.
The currently tracked attributes may change after every pccl api call.
If veracity of the attribute is important (e.g. to base application-logic branching decision on), the user must ensure that the attribute is up-to-date by re-obtaining it after
the relevant api call.
E.g. `PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE` may change after `pcclConnect`, `pcclUpdateTopology`, `pcclOptimizeTopology`, and after collective operations.

## Error Handling & Logging

### Error Returns

Most functions return a `pcclResult_t`, an enum with values like:

- `pcclSuccess` (0)
- `pcclNotInitialized`, `pcclInvalidArgument`, `pcclInvalidUsage`
- `pcclMasterConnectionFailed`, `pcclRankConnectionFailed`, `pcclRankConnectionLost`
- `pcclRemoteError`, etc.

If functions do not explicitly mention a recommended action, it is typical wise to panic or otherwise handle the error
in a non-recoverable way.

## Logging

PCCL uses an internal logging mechanism that writes debug/info/error messages to stdout/stderr, controlled by the
environment variable `PCCL_LOG_LEVEL`. For example:

- `export PCCL_LOG_LEVEL=DEBUG` (Linux/macOS)
- `set PCCL_LOG_LEVEL=DEBUG` (Windows cmd)

Logging can be helpful for diagnosing deadlocks or mismatched states.