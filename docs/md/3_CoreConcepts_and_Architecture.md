# Core Concepts & Architecture

This section dives deeper into how PCCL orchestrates a fault-tolerant run, dynamically accepts (or rejects) new peers, and tracks collective operations with its central master node. We’ll look at the client–master model, the phases and states of each client, and the key data structures that define the library’s internal “state machine.”

## Master–Client Model
Unlike MPI or other HPC libraries that rely on all ranks starting in tandem, PCCL uses a central master as a lightweight orchestrator:

- **Clients (Peers)**: Each training process or participant creates a PCCL communicator and connects to the master. Once recognized, it joins an ongoing session or “run.”

- **Master Node:** A single process that listens for new connections, tracks who is currently “accepted,” and coordinates transitions through various states (e.g. “accept new peers,” “run collectives,” “sync shared state”).

### What the Master Tracks
1. Registered Clients vs. Accepted Clients
- *Registered*: A peer that has connected but not yet reached the final “Accepted” phase. While “registered,” a peer can’t do collective ops.
- *Accepted*: Fully integrated into the run and allowed to perform All-Reduce or shared-state sync with other accepted peers in the same peer group.

2. Ring Topology & Bandwidth Metrics

- **Ring Order**: The master maintains a ring order, which determines the order of pipeline stages for the All-Reduce operation. This order can be optimized based on measured bandwidth between peers.

- **Bandwidth Metrics**: The master requests bandwidth info from client-to-client benchmarks during topology optimization and (re-)solves a traveling-salesman-like problem to optimize the ring order.
Depending on the size of the problem, the master may start a "moonshot" optimization as a background task, where either continuous heuristic improvement upon the solution or an exact solution is attempted. Depending on whether this yielded a better solution, peers will adopt the new topology during the next invocation of `pcclOptimizeTopology()`, if no peers joined in the meantime - making the optimization no longer applicable and thus discarded.
However, for any call to `pcclOptimizeTopology()` with no current optimization in progress, the master will immediately provide the best topology that can be found within a small computational timeout budget, which peers will adopt immediately.

**NOTE**: PCCL benchmarks by sending and receiving bytes as fast as possible. PCCL will not take into consideration the possible CPU-overhead introduced by e.g. quantization during an all reduce. If a peer is bottlenecked by its CPU and cannot utilize its full available bandwidth during an ongoing collective operation, PCCL may settle on a suboptimal topology.
PCCL also asserts idealized pseudo-fullduplex bandwidth to accelerate the bandwidth testing process: PCCL will assert that sending does not degrade receving performance or the other way around. If a peer has high peak receiving bandwidth, but sending degrades this side substantially, the topology optimization may not find the optimal solution.

3. Shared State

- The master verifies that each peer’s “shared state revision” matches the group’s expected next revision. If not, the peer is considered out-of-date (and might request or distribute data in the subsequent sync phase).
Additionally, the master tracks "content hash popularity", meaning it decides which peers have outdated/dirty shared state and require retransmission. Retransmission is performed in a p2p manner, the master simply decides what peers have to request shared state transmission from what peer.

4. Collective Operations

- The master also tracks which collectives are in progress (if any) for each peer group and ensures no conflicting operations (like also trying to accept new peers or performing shared state synchronization at the same moment).

## States & Phases: ConnectionPhase, ConnectionState, and More
PCCL organizes each client’s position in the run as a combination of:

- **ConnectionPhase**: Are we a just-arrived peer (`PEER_REGISTERED`) or a fully integrated one (`PEER_ACCEPTED`)?
- **ConnectionState**: Which “transitional” sub-phase is the peer in (e.g. voting to accept new peers, optimizing topology, waiting to finalize shared-state sync, etc.)?
- **CollectiveCommunicationState** (per tag):  If the peer has launched multiple concurrent All-Reduces, each operation can be in “vote to initiate,” “perform,” or “vote to complete” states.

Below is a simplified overview:

1. ConnectionPhase
- `PEER_REGISTERED`: The peer has connected to the master but not been accepted by existing peers. Restricts the client to minimal actions (e.g. establishing p2p connections if it becomes and accept candidate or else merely waiting).
- `PEER_ACCEPTED`: The peer is fully recognized and can participate in All-Reduce or shared-state sync within its group.

2. ConnectionState
- `IDLE`: Default "do nothing" state.
- `VOTE_ACCEPT_NEW_PEERS`: The peer is voting to let newly registered peers join. Once *all* peers do this, the run transitions to establishing p2p connections with the new arrivals.
- `VOTE_NO_NEW_PEERS_ESTABLISH_P2P_CONNECTIONS`: State for re-establishing or re-wiring p2p connections (e.g. after a topology change) without allowing brand-new peers. Typically triggered by `pcclOptimizeTopology()` to update the p2p connections to reflect the neighbors of the new ring order.
- `OPTIMIZE_TOPOLOGY / VOTE_OPTIMIZE_TOPOLOGY`: Master instructs peers to measure point-to-point bandwidth to peers that are missing from the master-managed bandwidth store. If not all requested operations necessary for topology optimizations are completed by peers, all peers enter the `OPTIMIZE_TOPOLOGY_FAILED` state, in which the only legal action is to retry the optimization.
- `COLLECTIVE_COMMUNICATIONS_RUNNING`: A “coarse” umbrella state indicating that at least one collective operation is currently in progress for some tag. The library uses per-tag `CollectiveCommunicationState` to track each operation’s finer details.

3. CollectiveCommunicationState (per operation "tag")
- `VOTE_INITIATE_COLLECTIVE_COMMS`: All peers are voting to launch an All-Reduce for a given tag.
- `PERFORM_COLLECTIVE_COMMS`: The ring-based reduce is active. If a peer fails mid-operation, the master marks the attempt as failed.
- `VOTE_COMPLETE_COLLECTIVE_COMMS`: After the ring finishes locally, each peer declares done. Once all peers do so, the operation is final.

### Why So Many States?
This fine-grained approach ensures that the master and all peers remain in **unanimous agreement** on the next action.
Since PCCL does not rely on a single global barrier (like `MPI_Barrier`) for every possible transition, each step in the run (accepting peers, establishing connections, synchronizing states, etc.) is a carefully orchestrated micro-consensus. The advantage is that a single dropped connection or tardy peer can be gracefully handled; the run can progress with those who remain.
With this architecture, PCCL can handle a wide range of failure conditions without risking a complete job crash. Every IO-failure induced crash and or breaking of bit parity of shared state is considered a bug in PCCL as long as the application is following best practices.

### Peer Groups & Communicators
Each communicator is tied to a **peer group** (an integer) that the master uses to partition the run. Only peers in the same group:

- Participate in each other’s All-Reduce or shared-state ops.
- Are forced to remain in lockstep by the library’s state machine for that group’s *collective calls* and *shared-state sync* operations. Other operations still require unanimous agreement across all peers across all peer groups.


### Shared-State Mask & Mismatch Checks

For each peer group, the master keeps a *“mask”* that enumerates which tensors (by string key) must be identical across peers, along with their hash. During synchronization:
- Each peer provides a candidate set of `(key, hash_type, hash_value)` via `pcclSynchronizeSharedState`.
- The master elects the most popular candidate set as the “mask.”
- Peers that deviate (outdated revision or mismatched hash) are directed to request data from a designated up-to-date peer.
- Once all participants converge, they vote to finalize the sync, returning each peer to `IDLE`.

Note: `hash_type` is currently always `simplehash`. Simple-hash is a non-cryptographic parallelizable hash function implemented on both cpu & cuda.
PCCL thus does not make implicit assertions about device placement for shared state tensors. These hashes are not cryptographically secure nor designed to resist adversarial attacks, but are sufficient for detecting silent divergences in shared state.

### Fault Tolerance & Dynamic Peers

**If a peer disappears** (or times out on the ring) mid-collective:
- The All-Reduce call returns an error to other peers.
- Meanwhile, the master eventually drops the vanished peer from the ring after it fails to respond.
- The remaining peers can simply retry the operation.

Note: in the worst case, for each retry there might be one less peer in the ring until there is but one peer left. In this case, the user should break out of the re-try loop and discard all data collected in the current step and wait to accept new peers until a world size of at least two is reached again.
This best practice is reflected in the `Hello World` example code of section 2.

#### Accepting new peers:
When a brand-new client calls `pcclConnect`, it enters `PEER_REGISTERED`. The existing accepted peers must at some stable moment vote to `VOTE_ACCEPT_NEW_PEERS`. Once all do, the group transitions to establishing p2p connections. The master then transitions them to `CONNECTING_TO_PEERS`, where peers will establish p2p connections with the new arrivals.
Each peer has a p2p socket that it listens on for incoming connections. During p2p connection establishment, the master sends the list of p2p connections each peer has to establish according to the current topology.
Each peer establishes an TX and an RX connection to each peer it has to connect to. The TX connection is established by the peer to the opposite peer's p2p socket, while the RX connection is established by the opposite peer to this peer's p2p socket.
The peers then establish these connections and send a confirmation back to the master. The master then transitions newcomers to `PEER_ACCEPTED` and returns their connection state to `IDLE`. P2P connection establishment may also fail, in which case the only legal action for clients is to retry the connection establishment, potentially with a different set of peers, if e.g. a peer has disconnected/has been kicked from the run due to a failure or unexpected/malicious behavior.

### One Operation at a Time (Per Group)
Because each phase (update topology, optimize topology, sync shared state, run collective operations) requires all accepted peers in that group to do the same action, multiple distinct operations can’t truly run in parallel.

