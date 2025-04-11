# PCCL (Prime Collective Communications Library)

# Introduction


## What is PCCL?
The Prime Collective Communications Library (PCCL) is a fault-tolerant collective-communications framework designed to run over standard TCP/IP networks—including wide-area and internet-scale environments. It facilitates distributed training of larger models by letting multiple peers dynamically join or leave a training run at any point, with minimal risk of bringing the entire process to a halt.

Unlike classic HPC-focused libraries such as *MPI* or vendor-specific solutions like *NCCL*, PCCL does not require all processes to start together in a tightly coupled, homogeneous cluster. Instead, it employs a *central master* node—a lightweight orchestrator—that tracks which peers are currently part of the run and coordinates the collective operations (e.g. All-Reduce, model state synchronization). This architecture allows:

### Dynamic Membership
Peers can appear or disappear for any reason (scaling out training, a machine crashing, or a user intentionally leaving). The master updates the run topology and ensures surviving peers can continue without deadlock.

### Fault Tolerance
Because each peer routinely checks in with the master and with one another (in a ring topology by default), a dropped or unresponsive peer triggers an error but does not irrevocably crash the entire job. The user can typically retry the operation or allow the master to remove the failed peer before proceeding.

### Shared state synchronization
Beyond simple collectives like All-Reduce, PCCL provides a built-in system to keep large model parameters or optimizer states in sync across all participants. Each peer references user-owned CPU or GPU memory for these tensors, and PCCL verifies identical content (by hashing) to prevent silent divergences in training.

### Dynamic Topology & Bandwidth Awareness
PCCL can periodically measure point-to-point bandwidth among peers, then optimize the ring order (or other topologies) to improve throughput. While specialized for ring-based collective operations, this design can adapt to different network speeds (e.g., node to node over the public Internet).


By default, PCCL handles the complexities of connecting peers over standard TCP sockets. It also implements features such as chunked data transfer, built-in hashing for detecting mismatched parameters, and a TSP-based (traveling-salesman) heuristic to rearrange ring orders for potentially higher performance.

In short, PCCL aims to combine the elasticity and internet-friendliness of a “game-server” style model with the collective ops and large-tensor capabilities typically found in cluster-based deep-learning frameworks—offering an approachable, if somewhat experimental, alternative to rigid HPC solutions.


## Key Features

### Dynamic Membership
Peers can join or leave the run at any time—whether due to scaling decisions, network failures, or user action. The system’s central master node updates the run topology accordingly, preventing a single dropout from crashing the entire collective.

### Central Master Orchestration
A lightweight “master” process keeps track of all currently accepted peers, assigns them to peer groups, and coordinates which collective or shared-state operation is valid at any moment. Peers only need to know their immediate ring neighbors and the master’s address.

### All-Reduce and Other Collective Ops
PCCL supports ring-based All-Reduce (summing, averaging, min/max, etc.) with CPU- (and in the future GPU-side) buffers.

###  Built-In Shared-State Sync
Beyond simple collectives, PCCL provides a mechanism to keep large model parameters or optimizer states identical across peers. Each peer references its local CPU or GPU memory as “shared state,” and PCCL enforces bitwise equality (unless explicitly relaxed). If a mismatch is detected (via hashing), the correct data is retransmitted from a “popular” peer.

### Fault Tolerance & Automatic Recovery
If a peer disappears mid-collective, the operation fails gracefully rather than halting the entire job. The user can retry the collective, while the master arranges to exclude the failed peer. This model is more forgiving than MPI-like systems requiring all ranks to remain present.

### Bandwidth-Aware Topology Optimization
PCCL can benchmark peer-to-peer links and solve a traveling-salesman-like problem to reorder the ring for better throughput. That helps accommodate nodes connected at different speeds (e.g., across continents over a standard internet connection).

### Cross-Platform, CPU/GPU Support
PCCL is written in C++ with portability to Linux, Windows, and macOS. It supports pointers to host or CUDA device memory for collectives and shared-state. While not as hardware-specialized as NCCL, it aims for wide compatibility—including Apple Silicon (ARM64) and x86_64.

### No Specialized Network Hardware Required
Uses standard TCP sockets. Suitable for clusters without InfiniBand or HPC fabrics, or even cloud VMs with only public IP addresses.

### Ports
A PCCL master must always be on the following port:
- `48148` - Master endpoint

A PCCL peer by default will bind to the following ports:
- `48149` - Shared state transmission endpoint
- `48150` - P2P connection endpoint
- `48151` - bandwidth testing endpoint

If those ports cannot be bound to, PCCL will attempt to bind to the next available port above the respective port.
This is useful e.g. if you are hosting multiple peers on the same host, where there would otherwise be a port conflict.
We recommend port forwarding a small range of ports above `48148`, proportional to the number of peers you want to run on the same host.

## Project Status / Roadmap

### Currently Active Development
PCCL is under active development. While core features like ring-based All-Reduce, shared-state synchronization, and fault tolerance are functional, several aspects are still evolving and may change.

#### Limitations
- **Async Collectives**: The library exposes asynchronous calls (`pcclAllReduceAsync`).
- **Master Crash Handling**: If the master node fails, connected peers will eventually see connection errors. Automatic “master failover” is not yet implemented, so one must typically restart training from a saved checkpoint.
- **GPU Support**: While GPU pointers are supported for shared state synchronization, PCCL’s ring reduce targets CPU-managed transfers over TCP.

#### Planned Improvements
- **Enhanced Quantization Schemes**: Current min–max quantization may expand to include stochastic rounding, 4-bit quantization or compression schemes for reducing bandwidth.

### Intended Use Cases
The library is primarily aimed at training scenarios where nodes may run over commodity Ethernet or WAN connections, or in dynamic settings where peers are expected to join and non-gracefully leave.
PCCL does not directly compete with traditional MPI libraries with respect to saturating high-speed interconnects. 
PCCL requires Ethernet or TCP/IP connectivity, and is not designed for low-latency, high-bandwidth fabrics like InfiniBand or Cray Aries.