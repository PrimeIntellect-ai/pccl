# Common Footguns

## FSDP + PCCL

When attempting to combine FSDP and PCCL, a constellation of the following to facts about
FSDP and PCCL respectively can lead to deadlocks:

- FSDP's internal all-gather operations are blocking.
- PCCL's `pcclConnect()` will not unblock until the pre-existing set of peers have accepted the newcomer.

If an application fails to accept peers that are MPI-ranks of itself, subsequent MPI rank processes
will be effectively locked out of the run, resulting in a deadlock as soon as a pre-existing peer
hits a blocking all-gather operation, e.g. during a model forward.

To avoid this, we recommend checking if the global world size is less than the largest peer group world size times the
mpi world size.

E.g. if you have 8 FSDP ranks "traditional MPI ranks", then each of those ranks will hold a different shard of the model
weights.
Ranks that have the same shard of the model weights will be in the same PCCL peer group.

This effectively forms a 2D matrix of MPI ranks and the PCCL dynamic membership dimension.

A fully populated grid of ranks would look as follows:

|                 | **PCCL Rank 1**               | **PCCL Rank 2**               | **PCCL Rank 3**               | **PCCL Rank 4**               |
|-----------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| **FSDP Rank 0** | (host:0, gpu:0, peer_group:0) | (host:1, gpu:0, peer_group:0) | (host:2, gpu:0, peer_group:0) | (host:3, gpu:0, peer_group:0) |
| **FSDP Rank 1** | (host:0, gpu:1, peer_group:1) | (host:1, gpu:1, peer_group:1) | (host:2, gpu:1, peer_group:1) | (host:3, gpu:1, peer_group:1) |
| **FSDP Rank 2** | (host:0, gpu:2, peer_group:2) | (host:1, gpu:2, peer_group:2) | (host:2, gpu:2, peer_group:2) | (host:3, gpu:2, peer_group:2) |
| **FSDP Rank 3** | (host:0, gpu:3, peer_group:3) | (host:1, gpu:3, peer_group:3) | (host:2, gpu:3, peer_group:3) | (host:3, gpu:3, peer_group:3) |
| **FSDP Rank 4** | (host:0, gpu:4, peer_group:4) | (host:1, gpu:4, peer_group:4) | (host:2, gpu:4, peer_group:4) | (host:3, gpu:4, peer_group:4) |
| **FSDP Rank 5** | (host:0, gpu:5, peer_group:5) | (host:1, gpu:5, peer_group:5) | (host:2, gpu:5, peer_group:5) | (host:3, gpu:5, peer_group:5) |
| **FSDP Rank 6** | (host:0, gpu:6, peer_group:6) | (host:1, gpu:6, peer_group:6) | (host:2, gpu:6, peer_group:6) | (host:3, gpu:6, peer_group:6) |
| **FSDP Rank 7** | (host:0, gpu:7, peer_group:7) | (host:1, gpu:7, peer_group:7) | (host:2, gpu:7, peer_group:7) | (host:3, gpu:7, peer_group:7) |

However, any sparse population of ranks is possible, e.g. not all MPI ranks of any PCCL peer may have not started or been accepted into
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

In combination with an async DiLoCo accept-pump which is only active if peers are truly pending, this would look as follows:
```python
if local_iter_num > 1 or local_world_size == 1:
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

If the sharding strategy differs between PCCL ranks, we recommend using a single-process per peer approach without using the concept of PCCL peer groups at all.
This may mean a dedicated master PCCL process which memory-maps the different shards of the FSDP subprocesses into its memory space
into one contiguously addressable memory region to then reference in the PCCL shared state. Alternatively, the master process could gather and scatter the state from its worker processes, however, at the cost of avoidable memcopies.
