# Common Footguns

## FSDP + PCCL

When attempting to combine FSDP and PCCL, a constellation of the following two facts about
FSDP and PCCL respectively can lead to deadlocks:

- FSDP's internal all-gather operations are blocking.
- PCCL's `pcclConnect()` will not unblock until the pre-existing set of peers have accepted the newcomer.

If an application fails to accept peers that are MPI-ranks of itself, subsequent MPI rank processes
will be effectively locked out of the run, resulting in a deadlock as soon as a pre-existing peer
hits a blocking all-gather operation, e.g. during a model forward.

To avoid this, we recommend checking if the global world size is less than the largest peer group world size times the
MPI world size.

E.g. if you have 8 FSDP ranks "traditional MPI ranks", then each of those ranks will hold a different shard of the model
weights.
Ranks that have the same shard of the model weights will be in the same PCCL peer group.

This effectively forms a 2D matrix of MPI ranks and the PCCL dynamic membership dimension.

A fully populated grid of ranks would look as follows:

|                 | **PCCL Rank 1** | **PCCL Rank 2** | **PCCL Rank 3** | **PCCL Rank 4** |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| **FSDP Rank 0** | HOST=1,GPU=0    | HOST=1,GPU=0    | HOST=2,GPU=0    | HOST=3,GPU=0    |
| **FSDP Rank 1** | HOST=1,GPU=1    | HOST=1,GPU=1    | HOST=2,GPU=1    | HOST=3,GPU=1    |
| **FSDP Rank 2** | HOST=1,GPU=2    | HOST=1,GPU=2    | HOST=2,GPU=2    | HOST=3,GPU=2    |
| **FSDP Rank 3** | HOST=1,GPU=3    | HOST=1,GPU=3    | HOST=2,GPU=3    | HOST=3,GPU=3    |
| **FSDP Rank 4** | HOST=1,GPU=4    | HOST=1,GPU=4    | HOST=2,GPU=4    | HOST=3,GPU=4    |
| **FSDP Rank 5** | HOST=1,GPU=5    | HOST=1,GPU=5    | HOST=2,GPU=5    | HOST=3,GPU=5    |
| **FSDP Rank 6** | HOST=1,GPU=6    | HOST=1,GPU=6    | HOST=2,GPU=6    | HOST=3,GPU=6    |
| **FSDP Rank 7** | HOST=1,GPU=7    | HOST=1,GPU=7    | HOST=2,GPU=7    | HOST=3,GPU=7    |

However, any sparse population of ranks is possible, e.g. not all MPI ranks of any PCCL peer may have not started or
been accepted into
the PCCL run yet.

We cannot run any FSDP forwards until truly all MPI-ranks of any given PCCL peer have been accepted into the run.
Any other configuration will lead to deadlocks.

To check if the grid is fully populated, we simply check the following:

```python
global_world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)
largest_peer_group_size = communicator.get_attribute(Attribute.LARGEST_PEER_GROUP_WORLD_SIZE)

if global_world_size < (mpi_config.mpi_world_size * largest_peer_group_size):
    # still wait for more peers
    pass
```

In combination with an async DiLoCo accept-pump which is only active if peers are truly pending, this would look as
follows:

```python
local_iter_num = 0

while True:
    local_iter_num += 1

    if local_iter_num > 1:
        logger.info("Checking are_peers_pending...")
        while True:
            try:
                if communicator.are_peers_pending():
                    logger.info(
                        "Join-Candidate peers pending; awaiting concurrent collective operations to accept new peers...")
                    if all_reduce_thread is not None:
                        all_reduce_thread.join()
                    communicator.update_topology()
                    topology_updated = True
                break
            except PCCLError as e:
                logger.info(f"Updating PCCL topology failed {e}, retrying...")
                time.sleep(1)

    global_world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)  # obtain global world-size after join
    largest_peer_group_size = communicator.get_attribute(Attribute.LARGEST_PEER_GROUP_WORLD_SIZE)
    mpi_ranks_pending = global_world_size < (mpi_config.mpi_world_size * largest_peer_group_size)

    if mpi_ranks_pending:
        time.sleep(1)
        continue # wait for more peers
```

If the sharding strategy differs between PCCL ranks, we recommend using a single-process per peer approach without using
the concept of PCCL peer groups at all.
This may mean a dedicated master PCCL process which memory-maps the different shards of the FSDP subprocesses into its
memory space
into one contiguously addressable memory region to then reference in the PCCL shared state. Alternatively, the master
process could gather and scatter the state from its worker processes, however, at the cost of avoidable memcopies.

### Reduced fault tolerance when FSDP is used

When using FSDP, the fault tolerance of PCCL is significantly reduced. If a peer drops out and is not restarted,
the remaining peers will block forever during the next all-gather operation.
This can cause the run to stall until the NCCL collective operation timeout is exceeded.
PCCL has no way of determining that the rest of the peers are effectively
dead along until those other processes die because of said timeout.


## Connecting via 127.0.0.1
When connecting via `127.0.0.1`, the master node will also see the peer's IP as `127.0.0.1` and will be logged as such.
When other peers that do not connect to the master via loopback attempt to connect, they would obtain return information
referencing `127.0.0.1` - attempting to p2p connect to themselves.
For this reason, the master disallows connections of non-loopback peers when at least one peer has connected via loopback.
It is recommended to always connect via the public IP of the master node to avoid this issue.

## Deterministic Advancement of Shared State
PCCL expects application developers to deterministically advance the shared state from collective communication results such that the shared state content
of all peers is bitwise identical.
If the user intends to break this contract, they must set the `allow_content_inequality` flag to `true` in the shared state
or alternatively tolerate periodic retransmissions of the shared state.
PCCL will support distance-aware hashing of the shared state in future versions such that retransmissions
are more infrequent instead of for all intents and purposes occurring every single step when `allow_content_inequality` remains `false`.
When using the `allow_content_inequality` flag, the user is solely responsible for ensuring that peer drift does not occur practically
and that all shared state tensors are "logically identical" such that obtaining any version on join is tolerable.
We recommend spending the extra effort to polish the application logic to ensure that shared state advancement is deterministic
and to rule out peer drift categorically.

This can be accomplished with most modern optimizers - which are merely elementwise operations, where no GPU-indeterminism can appear.
Regardless of whether the shared state is placed on the GPU or CPU, it is easy to accidentally lose this property through:

- Not referencing all optimizer state including momentum buffers in the shared state
- Creating new tensor instances instead of using in-place operations (like `.copy_()`) (which may even segfault when using CPU, or worse on GPU)

It is also crucial that any approximations used in the optimizer are not only deterministic but are implemented identically on all peers.
This can lead to issues where when peers utilize different architectures, the shared state will diverge and unnecessary retransmissions will be encountered.

PCCL does allow the shared state to be placed on different device types (CPU/GPU) on different peers.
E.g. the behavior of `__nv_expf` is not going to be replicated exactly by the CPU implementation of `expf` and vice versa, which will lead to divergence.
We recommend that when the user intends to allow for peers with different GPU & CPU architectures to join that they implement their own easily replicable approximations for the shared state advancement.
