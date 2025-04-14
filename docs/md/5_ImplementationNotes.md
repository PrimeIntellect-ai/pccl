# Implementation Notes

This section offers a behind-the-scenes look at how PCCL is implemented. While most users won’t need these details to run or integrate PCCL, it can be useful for:
- Debugging issues in cross-platform socket code
- Understanding how concurrency is managed

## TinySockets
PCCL relies on standard TCP sockets for:
- *Master* connections (long-lived to the orchestrator)
- *Peer-to-Peer* ring connections (one or two per ring neighbor)
- *Shared-State Distribution* “one-off” ephemeral connections (similar to HTTP GET/POST style transfers)
- *Bandwidth Benchmark* connections (short-lived, used to measure throughput between pairs)

### Queued Socket Mechanism
For the master connection—and potentially any socket that might carry messages for multiple “logical” operations—PCCL uses a queued socket approach. That is, we maintain an internal queue of incoming packets, and let each part of the library read only the packets intended for it by consuming only packets that match a particular predicate. This helps avoid concurrency issues where multiple threads might accidentally consume each other’s data.

### Dedicated RX/TX Threads
PCCL uses dedicated RX/TX threads for sending and receiving concurrently.
Threads add read or write requests to a queue for a particular tag, and data will be sent or received on that threads behalf.
The RX thread will read from the socket and dispatch the data to the correct queue given the received tag.
The TX thread will send data from the queue to the socket while prepending the tag for distinction.
Waking up the TX thread is done via [threadpark](https://github.com/PrimeIntellect-ai/threadpark/tree/main), a custom-built lightweight thread parking library that utilizing futex-like apis on all major operating systems to facilitate efficient wakes.

## Firewalls & Ports
- By default, the master listens on port `48148`
- Each peer tries to bind to a small range ([`48149`..`48151`]) for p2p, shared-state distribution, and bandwidth test sockets.
  However, these ports are not defacto static as is the case with most network protocols. Rather, these ports are “bump allocated“ where initially the implementation tries to bind to the target port (e.g. `48151` for the benchmark socket, or `48149` for the shared state server), but if this fails, the next higher port is tried until a free port is found. This ensures multiple peers can run on the same machine without port conflicts. Peers will "find" each other by reporting their ports to the master, which will inturn share this information with other peers.

- `Important`: For wide-area or internet usage, you must open these ports in your firewall & forward them to your computer when behind NAT. When only hosting one peer per IP address, only opening port `48149`, `48150`, `48151` is required.
  When hosting more peers per IP address (e.g behind NAT), the recommended approach is to open a port range above `48148` proportional to the amount of peers using this IP address.


## Master Orchestration

The master runs a single-threaded event loop (currently using `libuv`) that listens for new clients, processes control packets (e.g. “vote to accept new peers,” “optimize topology,” etc.), and updates the central state machine in `CCoIPMasterState`.


### Single-Threaded, Deterministic State
Since the master is conceptually a big “authoritative state machine”. Each inbound request from a client (join, sync, etc.) triggers a state update or transition. Particular state transitions trigger confirmation packets to be sent to clients to proceed.

### Master Crash = End of Run
If the master node process crashes or is forcibly killed, the peer side eventually sees “lost connection” errors. There is no built-in “master re-election” or replication. The recommended approach is to:

1. Simply restart the master process
2. Have peers automatically reconnect to the new master (e.g. fisrt panic the application & relaunch via a script on unsucessful exit code)
3. Load from checkpoint to restore the shared state.

It should be noted that CCoIP itself is not designed to "retain" shared state, simply to distribute it while a run is ongoing. If a run does indeed crawl to a halt, the shared state is lost. Therefore, it is recommended that peers save their own shared state to disk periodically, and reload it on restart.
As PCCL guarantees bit-identical shared state among all peers at all times, it is expected that after peers load their saved shared state from disk and begin synchronizing the shared state again, that all previously connected peers will unanimously agree on the shared state hashes and continue from there.

However, it should be noted that a crash of the master process is very unlikely.

### Topology Optimization (Bandwidth Tests & TSP)
One of PCCL’s features is "bandwidth-aware ring ordering". Since ring-based reduce can be bottlenecked by the slowest link, it helps to measure peer-to-peer throughput and reorder accordingly.

1. **Bandwidth Store**: The master keeps an asymmetric cost matrix (`BandwidthStore`) of measured bandwidth from peer A to peer B.
2. **Benchmark Requests**: When a peer calls `pcclOptimizeTopology`, the master identifies missing edges (i.e., pairs not yet measured) and instructs the relevant peer(s) to do a quick TCP test.
3. **TSP Heuristic:** The master uses a traveling-salesman “shortest path” (or “highest bandwidth”) approach to find a ring ordering that tries to maximize total link speed. For small problems `world_size <= 16` an exact solution will be attempted in a set timeout limit, for larger problems it might attempt a simpler heuristic (path of immediate “closest” peer, random tour or ant colony optimization with 2 Opt & 3 Opt local search, etc.). If an optimal solution cannot be found, the master may start a “moonshot” approach in the background to either target an optimal solution for higher `world_size` or to continue improving the current solution heuristically.
   Once a better ring is found, p2p connections will be re-established in that order (without letting brand-new peers in) the next time the client calls `pcclUpdateTopology`. The clients adopt the new ring as soon as they collectively vote and connect to each new neighbor.
   Solutions that are found immediately as part of the topology optimization phase without a background “moonshot” are adopted immediately by the peers as part of `pcclOptimizeTopology` in a fashion similar to `pcclUpdateTopology`, but without admitting newcomers into the run while still “going through the same motions” of voting to establish p2p connections by peers, followed by distribution of the p2p connection information by the master to said peers, along with subsequent connection establishment performed by the peers followed by subsequent confirmation of the connection establishment to the master.

### Shared-State Hashing & Distribution

When you call `pcclSynchronizeSharedState`, each peer does:

1. **Hash Each Tensor**:
- On CPU: SimpleHash (with OpenMP optimizations, if supported by compiler) or a CRC32-based approach (with SSE/AVX/ARM Neon optimizations if available).
- On GPU: SimpleHash, a custom deterministic kernel if compiled with CUDA support.

2. **Report Revision & Hash**: The peer sends these to the master for that group’s “mask election.”
3. **Master Chooses a Mask**: By popularity, it decides which set of (keys, hashes) is canonical. Peers that deviate are assigned to fetch the updated data from a designated “correct” peer via ephemeral connections.
4. **One-Increment Rule**:
   The master checks that each new `shared_state->revision` is exactly 1 higher than before. If not, you see a `REVISION_INCREMENT_VIOLATION`.
   If it were to ever happen that the last peer that distributed the shared state leaves the run, the shared state is effectively lost.
   Because new peers will have freshly initialized revision counters, which thus triggers a mismatch, no peer will ever be able to synchronize the shared state successfully again until the master is restarted.
   It is the responsibility of the application developer to ensure to periodically checkpoint the shared state.
   In the best case, newly joining peers load the most recent shared state from disk, and because the shared
   state revision matches the expected value, said peer will become shared state distributor, restoring the normal flow of operations.
   If the peer only periodically checkpoints shared state, the load may not result in restoring the exact last seen shared state revision.
   The master will of course not accept this revision, even though it might be the best available recovery choice.
   It is intended that in such a scenario the master must be restarted for purposes of safety and consistency of behavior.

5. Dirty Keys: If a peer’s local hash for “weight_1” mismatches the mask, the peer sets up a direct ephemeral TCP connection to the distributing peer to its shared state distribution socket. After transmission, the content is hashed again and compared to the expected value. If the hash matches, the peer proceeds. On hash mismatch, the call to `pcclSynchronizeSharedState` will return an error code.

In practice, if your model steps are *bitwise deterministic across* peers, the “dirty keys” scenario rarely happens. But it remains crucial for newly joining peers who need a full checkpoint or for accidental drift scenarios.
In the ideal case, the training code may even assert that no data is ever *received* during shared state synchronization after the first local iteration, where it may obtain the current popular shared state from the set of pre-existing peers post joining the run.

### Ring-Reduce Implementation
PCCL’s All-Reduce uses a pipeline ring approach:

1. `Reduce-Scatter`: `In world_size-1` steps, each peer’s chunk is successively passed around the ring and accumulated (e.g., sum).
2. `Reduce-Gather`: Another `world_size-1` steps to distribute final chunks so everyone ends up with the fully reduced array.

#### Chunking & Quantization
- The library divides the buffer among peers, so each rank “owns” a slice.
- It optionally quantizes data if `pcclQuantMinMax` or others are selected. This can help reduce link usage on slower WAN connections. As quantization is also performed using optimized SIMD instructions, the overhead should be negligible for most WAN and even Ethernet connections.

## Concurrency and Threading Model

### Master
- **Single Thread**: The master’s event loop is not multithreaded. This ensures consistent updates to `CCoIPMasterState`.

### Client

- **Blocking API**: By default, calls like `pcclAllReduce` or `pcclSynchronizeSharedState` block the caller until the operation finishes (or fails).
- **Async All-Reduce**: `pcclAllReduceAsync` spawns an internal thread or creates a logical execution plan and returns immediately. The user can poll for completion or wait on a condition variable. This concurrency model is useful for overlapping computation with communication, or for running multiple operations in parallel.
  This will be intended API design in future versions of PCCL.
- **Queued-Sockets:** If two internal threads might read from the same socket, PCCL enforces a queue mechanism to route matching packets to the correct consumer via predicate matching.

### Overall Rule: One Operation at a Time (Per Group)
Because the master enforces that the entire group do the same operation in lockstep, you rarely need your own concurrency around these calls.
The library expects you not to overlap, say, an All-Reduce with a shared-state sync in the same communicator - neither through means
of the native concurrency features implemented by PCCL (`pcclAllReduceAsync`, `pcclAwaitAsyncReduce`), nor through the use of concurrent threads.
The only exception to this rule is that it is allowed to launch multiple async collective communications without awaiting the respective previous handle - in other words, multiple async collective communications operations may be in flight at the same time.
Note however that even in this case the overarching "Operation" is "performing collective communications operations" and *only* performing collective communications operations.

### Threadsafe
PCCL is generally not threadsafe and should only ever be used from the main thread (except in case of stated exceptions).
PCCL will generally enforce that public facing apis are called on the main thread registered for the communicator.

#### There are some exceptions to this rule:
PCCL allows the user to launch and await multiple async collective operations from threads other than the main thread, as long as the main thread
does not call other user facing function on the same communicator in the meantime.
The async work ongoing on the other thread must be awaited before the main thread can call user facing api functions on the same communicator again.
The main thread may however call the `pcclAreNewPeersPending` function while collective communications operations are being scheduled and awaited by a different thread.
If `pcclAreNewPeersPending` returns true, the main thread should call `pcclUpdateTopology` to accept the new peers into the run.
Before doing so, the main thread should await the completion of the ongoing collective communication operations on the other thread.
However, the main thread should not call `pcclAwaitAsyncReduce` on handles created by the other thread. No guarantees are made about the behavior of awaiting handles created by different threads.
Instead, the main thread should await the completion of the work performed by the other thread which will itself call `pcclAwaitAsyncReduce` on the handles it created.