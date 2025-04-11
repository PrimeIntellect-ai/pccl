#include "ccoip_client_handler.hpp"
#include <ccoip.h>
#include <future>
#include <hash_utils.hpp>
#include <pccl_log.hpp>
#include "ccoip_types.hpp"

#include <benchmark_runner.hpp>
#include <guard_utils.hpp>
#include <list>
#include <reduce.hpp>
#include <reduce_kernels.hpp>
#include <thread_guard.hpp>
#include <win_sock_bridge.h>
#include "ccoip_inet_utils.hpp"
#include "ccoip_packets.hpp"

#ifdef PCCL_HAS_CUDA_SUPPORT
#include <cuda.h>
#endif

ccoip::CCoIPClientHandler::CCoIPClientHandler(const ccoip_socket_address_t &address,
                                              const uint32_t peer_group) : master_socket(address),
                                                                           // Both p2p_socket and shared_state_socket listen to the first free port above the specified port number
                                                                           // as the constructor with inet_addr and above_port is called, which will bump on failure to bind.
                                                                           // this is by design and the chosen ports will be communicated to the master, which will then distribute
                                                                           // this information to clients to then correctly establish connections. The protocol does not assert these
                                                                           // ports to be static; The only asserted static port is the master listening port.
                                                                           p2p_socket({address.inet.protocol, {}, {}},
                                                                               CCOIP_PROTOCOL_PORT_P2P),
                                                                           shared_state_socket(
                                                                               {address.inet.protocol, {}, {}},
                                                                               CCOIP_PROTOCOL_PORT_SHARED_STATE),
                                                                           benchmark_socket(
                                                                               {address.inet.protocol, {}, {}},
                                                                               CCOIP_PROTOCOL_PORT_BANDWIDTH_BENCHMARK),
                                                                           peer_group(peer_group) {
}

bool ccoip::CCoIPClientHandler::connect() {
    setMainThread(std::this_thread::get_id());
    THREAD_GUARD(main_thread_id);

    if (accepted) {
        LOG(WARN) << "CCoIPClientHandler::connect() called while already connected";
        return false;
    }

    // start listening on p2p socket
    {
        p2p_socket.setJoinCallback([this](const ccoip_socket_address_t &client_address,
                                          std::unique_ptr<tinysockets::BlockingIOSocket> &socket) {
            // maximize send and receive buffer sizes
            socket->maximizeSendBuffer();
            socket->maximizeReceiveBuffer();

            const auto hello_packet_opt = socket->receivePacket<P2PPacketHello>();
            if (!hello_packet_opt) {
                LOG(ERR) << "Failed to receive P2PPacketHello from " << ccoip_sockaddr_to_str(client_address);
                if (!socket->closeConnection()) {
                    LOG(ERR) << "Failed to close connection to " << ccoip_sockaddr_to_str(client_address);
                }
                return;
            }
            const auto &hello_packet = hello_packet_opt.value();

            p2p_connections_rx_mutex.lock();
            const auto &rx_socket = p2p_connections_rx[hello_packet.peer_uuid] = std::make_unique<
                                        tinysockets::MultiplexedIOSocket>(
                                        socket->getSocketFd(), tinysockets::ConnectionModeFlags::MODE_RX);
            p2p_connections_rx_mutex.unlock();

            if (!rx_socket->run()) {
                LOG(FATAL) << "Failed to start MultiplexedIOSocket for P2P connection with "
                        << ccoip_sockaddr_to_str(client_address);
                return;
            }

            // send hello ack packet only after our rx socket is confirmed running
            if (!socket->sendPacket<P2PPacketHelloAck>({})) {
                LOG(ERR) << "Failed to send P2PPacketHelloAck to " << ccoip_sockaddr_to_str(client_address);
                if (!socket->closeConnection()) {
                    LOG(ERR) << "Failed to close connection to " << ccoip_sockaddr_to_str(client_address);
                }
                return;
            }
            LOG(INFO) << "P2P connection established with " << ccoip_sockaddr_to_str(client_address);
        });

        if (!p2p_socket.listen()) {
            LOG(ERR) << "Failed to bind P2P socket " << p2p_socket.getListenPort();
            return false;
        }
        if (!p2p_socket.runAsync()) [[unlikely]] {
            LOG(ERR) << "Failed to start P2P socket thread";
            return false;
        }
        LOG(INFO) << "P2P socket listening on port " << p2p_socket.getListenPort() << "...";
    }

    // start listening with shared state distribution server
    {
        shared_state_socket.addReadCallback(
            [this](const ccoip_socket_address_t &client_address, const std::span<std::uint8_t> &data) {
                onSharedStateClientRead(client_address, data);
            });
        if (!shared_state_socket.listen()) {
            LOG(ERR) << "Failed to bind shared state socket " << shared_state_socket.getListenPort();
            return false;
        }
        if (!shared_state_socket.runAsync()) [[unlikely]] {
            LOG(ERR) << "Failed to start shared state socket thread";
            return false;
        }
        shared_state_server_thread_id = shared_state_socket.getServerThreadId();
        LOG(INFO) << "Shared state socket listening on port " << shared_state_socket.getListenPort() << "...";
    }

    // start listening with bandwidth benchmark socket
    {
        benchmark_socket.setJoinCallback([this](const ccoip_socket_address_t &client_address,
                                                const std::unique_ptr<tinysockets::BlockingIOSocket> &socket) {
            int socket_fd = socket->getSocketFd();

            // if there is an ongoing benchmark
            if (benchmark_thread_opt.has_value()) {
                // and said benchmark is complete
                if (benchmark_complete_state) {
                    // join the thread
                    if (benchmark_thread_opt->joinable()) {
                        benchmark_thread_opt->join();
                    }
                } else {
                    // if the benchmark is not complete, tell the incoming client to go away
                    LOG(DEBUG) << "Rejecting incoming benchmark connection from "
                            << ccoip_sockaddr_to_str(client_address)
                            << " because a benchmark is already running. Telling it to retry later...";
                    B2CPacketBenchmarkServerIsBusy packet{};
                    packet.is_busy = true;
                    if (!socket->sendPacket<B2CPacketBenchmarkServerIsBusy>(packet)) {
                        LOG(WARN) << "Failed to send B2CPacketBenchmarkServerIsBusy to "
                                << ccoip_sockaddr_to_str(client_address);
                    }
                    if (!socket->closeConnection()) {
                        LOG(WARN) << "Failed to close connection to " << ccoip_sockaddr_to_str(client_address)
                                << " after rejecting benchmark connection with busy signal";
                    }
                    return;
                }
            }

            // tell the incoming client that it can start the benchmark
            B2CPacketBenchmarkServerIsBusy packet{};
            packet.is_busy = false;
            if (!socket->sendPacket<B2CPacketBenchmarkServerIsBusy>(packet)) {
                LOG(WARN) << "Failed to send B2CPacketBenchmarkServerIsBusy to "
                        << ccoip_sockaddr_to_str(client_address);
            }

            benchmark_complete_state.store(false);
            std::thread benchmark_thread([client_address, socket_fd, this] {
                NetworkBenchmarkHandler handler{};
                if (!handler.runBlocking(socket_fd)) {
                    // we have an accept backlog of 1, so this is fine and intended.
                    LOG(WARN) << "Failed to run network benchmark with " << ccoip_sockaddr_to_str(client_address);
                }
                LOG(INFO) << "Network benchmark finished with client " << ccoip_sockaddr_to_str(client_address);
                benchmark_complete_state.store(true);
            });

            benchmark_thread_opt = std::move(benchmark_thread);
        });
        if (!benchmark_socket.listen()) {
            LOG(ERR) << "Failed to bind bandwidth benchmark socket " << benchmark_socket.getListenPort();
            return false;
        }
        if (!benchmark_socket.runAsync()) [[unlikely]] {
            LOG(ERR) << "Failed to start bandwidth benchmark socket thread";
            return false;
        }
        LOG(INFO) << "Network bandwidth benchmark socket listening on port " << benchmark_socket.getListenPort()
                << "...";
    }

    if (!master_socket.establishConnection()) {
        LOG(ERR) << "Failed to establish master socket connection";
        return false;
    }

    if (!master_socket.run()) {
        LOG(ERR) << "Failed to start master socket thread";
        return false;
    }

    // send join request packet to master
    C2MPacketRequestSessionRegistration join_request{};
    join_request.p2p_listen_port = p2p_socket.getListenPort();
    join_request.shared_state_listen_port = shared_state_socket.getListenPort();
    join_request.bandwidth_benchmark_listen_port = benchmark_socket.getListenPort();
    join_request.peer_group = peer_group;

    if (!master_socket.sendPacket<C2MPacketRequestSessionRegistration>(join_request)) {
        LOG(ERR) << "Failed to send C2MPacketRequestSessionJoin to master";
        return false;
    }

    // receive join response packet from master
    const auto response = master_socket.receivePacket<M2CPacketSessionRegistrationResponse>();
    if (!response) {
        LOG(ERR) << "Failed to receive M2CPacketSessionJoinResponse from master";
        return false;
    }
    if (!response->accepted) {
        LOG(ERR) << "Master rejected join request";
        return false;
    }
    client_state.setAssignedUUID(response->assigned_uuid);
    const auto result = establishP2PConnections();
    if (result == RETRY_NEEDED) {
        accepted = true;
        // even if the initial p2p connection establishment fails due to unfortunate timing
        // of other peers dropping or becoming unreachable we will still be considered as a newly accepted peer,
        // even though p2p connections have not successfully been established. Here we have to retry like any other peer.
        if (!requestAndEstablishP2PConnections(true)) {
            // accept_new_peers must be true by definition because we just got accepted
            return false;
        }
    } else if (result == FAILED) {
        LOG(ERR) << "Failed to establish p2p connections after connecting to master.";
        return false;
    }
    accepted = true;
    return true;
}

bool ccoip::CCoIPClientHandler::requestAndEstablishP2PConnections(const bool accept_new_peers) {
    if (!accepted) {
        LOG(BUG) <<
                "CCoIPClientHandler::reestablishP2PConnections() was called before peer was accepted into the run. This is a bug!";
        return false;
    }

    if (!master_socket.isOpen()) {
        LOG(ERR) << "Failed to accept new peers: Client socket has been closed; This may mean the client was kicked by "
                "the master";
        return false;
    }

    if (client_state.isAnyCollectiveComsOpRunning()) {
        LOG(ERR) << "Failed to request and establish p2p connections: A collective communications operation is running";
        return false;
    }

    EstablishP2PConnectionResult result;
    do {
        C2MPacketRequestEstablishP2PConnections packet{};
        packet.accept_new_peers = accept_new_peers;
        if (!master_socket.sendPacket<C2MPacketRequestEstablishP2PConnections>(packet)) {
            return false;
        }
        result = establishP2PConnections();
    } while (result == RETRY_NEEDED);
    if (result == FAILED) {
        return false;
    }
    connection_revision++;
    return true;
}

bool ccoip::CCoIPClientHandler::arePeersPending(bool &pending_out) {
    if (!accepted) {
        LOG(BUG) <<
                "CCoIPClientHandler::arePeersPending() was called before peer was accepted into the run. This is a bug!";
        return false;
    }

    if (!master_socket.isOpen()) {
        LOG(ERR) << "Failed to accept check if peers are pending: Client socket has been closed; This may mean the client was kicked by "
                "the master";
        return false;
    }

    // collective ops may be running in the meantime
    const C2MPacketCheckPeersPending packet{};
    if (!master_socket.sendPacket<C2MPacketCheckPeersPending>(packet)) {
        return false;
    }
    const auto peers_pending_opt = master_socket.receivePacket<M2CPacketPeersPendingResponse>();
    if (!peers_pending_opt) {
        LOG(WARN) << "Failed to receive M2CPacketPeersPending from master";
        return false;
    }
    pending_out = peers_pending_opt->peers_pending;
    return true;
}

bool ccoip::CCoIPClientHandler::syncSharedState(ccoip_shared_state_t &shared_state,
                                                ccoip_shared_state_sync_info_t &info_out) {
    if (!accepted) {
        LOG(WARN) << "CCoIPClientHandler::syncSharedState() before CCoIPClientHandler::connect() was called. Establish "
                "master connection first before performing client actions.";
        return false;
    }

    if (!master_socket.isOpen()) {
        LOG(ERR) << "Failed to sync shared state: Client socket has been closed; This may mean the client was kicked "
                "by the master";
        return false;
    }

    if (client_state.isAnyCollectiveComsOpRunning()) {
        return false;
    }

#ifdef PCCL_HAS_CUDA_SUPPORT
    // convert CUDA runtime api pointers
    for (auto &entry: shared_state.entries) {
        if (entry.device_type == ccoipDeviceCuda) {
            // make sure we have a driver api pointer, and not a runtime pointer.
            // this method will for driver api pointers return the same pointer and for
            // runtime api pointers obtain the driver api compatible device pointer.
            CUdeviceptr device_ptr{};
            if (cuPointerGetAttribute(&device_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                      reinterpret_cast<CUdeviceptr>(entry.data_ptr)) != CUDA_SUCCESS) {
                LOG(FATAL) << "Could not convert CUDA runtime api pointer into a device pointer. Are you sure this "
                        "address is valid? Supplied pointer: "
                        << entry.data_ptr;
                return false;
            }
            entry.data_ptr = reinterpret_cast<void *>(device_ptr);
        }
    }
#endif

    // prepare shared state hashes
    std::vector<SharedStateHashEntry> shared_state_hashes{};
    shared_state_hashes.reserve(shared_state.entries.size());
    for (const auto &entry: shared_state.entries) {
        auto &key = entry.key;
        uint64_t hash = 0;
        ccoip_hash_type_t hash_type{};
        if (!entry.allow_content_inequality) {
            if (entry.device_type == ccoipDeviceCpu) {
                // hash = hash_utils::CRC32(entry.data_ptr, entry.data_size);
                // hash_type = ccoipHashCrc32;
                hash = hash_utils::simplehash_cpu(entry.data_ptr, entry.data_size);
                hash_type = ccoipHashSimple;
            } else if (entry.device_type == ccoipDeviceCuda) {
#ifndef PCCL_HAS_CUDA_SUPPORT
                LOG(BUG) << "PCCL is not built with CUDA support. We shouldn't even have gotten so far without CUDA "
                        "support when referencing CUDA tensors. This is a bug!";
                return false;
#else
                hash = hash_utils::simplehash_cuda(entry.data_ptr, entry.data_size);
                hash_type = ccoipHashSimple;
#endif
            } else {
                LOG(BUG) << "Unknown device type: " << entry.device_type
                        << " encountered during shared state hash preparation. This should have been caught earlier "
                        "and is a bug.";
                return false;
            }
        }
        shared_state_hashes.push_back(SharedStateHashEntry{
            .key = key,
            .hash = hash,
            .hash_type = hash_type,
            .data_type = entry.data_type,
            .allow_content_inequality = entry.allow_content_inequality
        });
    }

    // inside this block, we are guaranteed in the shared state distribution phase
    {
        // As long as we are inside this block, client_state.isSyncingSharedState() will return true

        // ATTENTION: We create this guard even before we vote to enter the shared state sync state on the server side.
        // So on the client side this phase either means that we have already successfully voted to enter the shared
        // state sync phase or that we would want to be in the shared state sync phase. This is necessary because we
        // check this state of being in the shared state sync phase during shared state distribution. We turn down
        // requests that come in while we are not in the shared state sync phase. However, if we were to start the
        // shared state sync phase AFTER sending the shared state sync vote packet, not all clients receive the
        // confirmation packet from the master after a successful vote at the same time. If you are unlucky, this
        // discrepancy can be so bad that some clients even handle this packet and send a shared state request to a
        // shared state distributor that is not yet aware that the shared state sync phase has started. It will then
        // respond with state SHARED_STATE_NOT_DISTRIBUTED. Because we are basically only using this phase state for
        // this purpose anyway, this is a good solution to avoid yet another vote that would only really solve a
        // conceptual problem, not a problem introduced by data dependencies.
        guard_utils::phase_guard guard([this, &shared_state] { client_state.beginSyncSharedStatePhase(shared_state); },
                                       [this] { client_state.endSyncSharedStatePhase(); });

        // vote for shared state sync
        C2MPacketSyncSharedState packet{};
        packet.shared_state_revision = shared_state.revision;
        packet.shared_state_hashes = shared_state_hashes;
        if (!master_socket.sendPacket<C2MPacketSyncSharedState>(packet)) {
            LOG(ERR) << "Failed to sync shared state: Failed to send C2MPacketSyncSharedState to master";
            return false;
        }
        // wait for confirmation from master that all peers have voted to sync the shared state
        const auto response = master_socket.receivePacket<M2CPacketSyncSharedState>();
        if (!response) {
            LOG(ERR) << "Failed to sync shared state: Failed to receive M2CPacketSyncSharedState response from master";
            return false;
        }

        if (response->is_outdated) {
            // check if distributor address is empty
            if (response->distributor_address.port == 0 && response->distributor_address.inet.protocol == inetIPv4 &&
                response->distributor_address.inet.ipv4.data[0] == 0 &&
                response->distributor_address.inet.ipv4.data[1] == 0 &&
                response->distributor_address.inet.ipv4.data[2] == 0 &&
                response->distributor_address.inet.ipv4.data[3] == 0) {
                LOG(ERR) << "Failed to sync shared state: Shared state distributor address is empty";
                return false;
            }

            // if shared state is outdated, request shared state from master
            tinysockets::BlockingIOSocket req_socket(response->distributor_address);
            if (!req_socket.establishConnection()) {
                LOG(ERR)
                        << "Failed to sync shared state: Failed to establish connection with shared state distributor: "
                        << ccoip_sockaddr_to_str(response->distributor_address);
                return false;
            }
            C2SPacketRequestSharedState request{};
            request.requested_keys.reserve(response->outdated_keys.size());
            for (const auto &key: response->outdated_keys) {
                request.requested_keys.push_back(key);
            }
            if (!req_socket.sendPacket<C2SPacketRequestSharedState>(request)) {
                LOG(ERR) << "Failed to sync shared state: Failed to send C2SPacketRequestSharedState to distributor";
                return false;
            }
            const auto shared_state_response = req_socket.receivePacket<S2CPacketSharedStateResponse>();
            if (!shared_state_response) {
                LOG(ERR) << "Failed to sync shared state: Failed to receive S2CPacketSharedStateResponse from "
                        "distributor";
                return false;
            }
            if (shared_state_response->status != SharedStateResponseStatus::SUCCESS) {
                LOG(ERR) << "Failed to sync shared state: Shared state distributor returned status "
                        << shared_state_response->status;
                return false;
            }
            // update shared state
            shared_state.revision = shared_state_response->revision;

            std::unordered_map<std::string, const SharedStateEntry *> new_entries{};
            std::unordered_map<std::string, const ccoip_shared_state_entry_t *> dst_entries{};
            for (const auto &entry: shared_state.entries) {
                dst_entries[entry.key] = &entry;
            }
            for (const auto &entry: shared_state_response->entries) {
                new_entries[entry.key] = &entry;
            }

            // write new content of outdated shared state entries
            for (size_t i = 0; i < response->outdated_keys.size(); i++) {
                const auto &outdated_key_name = response->outdated_keys[i];
                auto entry_it = dst_entries.find(outdated_key_name);
                if (entry_it == dst_entries.end()) {
                    LOG(ERR) << "Failed to sync shared state: Received data for unknown key " << outdated_key_name;
                    return false;
                }
                auto &dst_entry = *entry_it->second;

                if (new_entries.contains(outdated_key_name)) {
                    const auto &new_entry = new_entries[outdated_key_name];
                    if (new_entry->size_bytes != dst_entry.data_size) {
                        LOG(ERR) << "Failed to sync shared state: Shared state entry size mismatch for key "
                                << dst_entry.key;
                        return false;
                    }
                    if (dst_entry.device_type == ccoipDeviceCpu) {
                        // if cpu, receive directly into the cpu buffer
                        std::span dst_span(static_cast<std::byte *>(dst_entry.data_ptr), dst_entry.data_size);
                        if (req_socket.receiveRawData(dst_span, new_entry->size_bytes) != new_entry->size_bytes) {
                            LOG(ERR) << "Failed receive all bytes expected for shared state entry during shared state "
                                    "content transmission! Did peer disconnect unexpectedly?";
                            return false;
                        }
                    } else if (dst_entry.device_type == ccoipDeviceCuda) {
#ifndef PCCL_HAS_CUDA_SUPPORT
                        if (dst_entry.device_type == ccoipDeviceCuda) {
                            LOG(BUG) << "PCCL is not built with CUDA support. We shouldn't even have gotten so far "
                                    "without CUDA support when referencing CUDA tensors. This is a bug!";
                            return false;
                        }
#else
                        std::unique_ptr<std::byte> dst_ptr(new std::byte[dst_entry.data_size]);
                        std::span dst_span(dst_ptr.get(), dst_entry.data_size);
                        if (req_socket.receiveRawData(dst_span, new_entry->size_bytes) != new_entry->size_bytes) {
                            LOG(ERR) << "Failed receive all bytes expected for shared state entry during shared state "
                                    "content transmission! Did peer disconnect unexpectedly?";
                            return false;
                        }
                        if (cuMemcpyHtoD_v2(reinterpret_cast<CUdeviceptr>(dst_entry.data_ptr), dst_ptr.get(),
                                            dst_entry.data_size) != CUDA_SUCCESS) {
                            LOG(FATAL)
                                    << "Failed to copy host to device memory while trying to write out shared state "
                                    "transmission response content! Is shared state referenced memory still valid?";
                            return false;
                        }
#endif
                    } else {
                        LOG(BUG) << "Unknown device type encountered while trying to write out shared state "
                                "transmission response content. This should have been caught earlier and is a bug.";
                        return false;
                    }

                    info_out.rx_bytes += new_entry->size_bytes;

                    // check if hash is correct if allow_content_inequality is False
                    if (dst_entry.allow_content_inequality) {
                        continue;
                    }
                    if (i < response->expected_hashes.size()) {
                        uint64_t expected_hash = response->expected_hashes[i];
                        ccoip_hash_type_t expected_hash_type = response->expected_hash_types[i];

                        if (expected_hash_type == ccoipHashSimple) {
#ifdef PCCL_HAS_CUDA_SUPPORT
                            if (dst_entry.device_type == ccoipDeviceCuda) {
                                uint64_t actual_hash = hash_utils::simplehash_cuda(
                                    dst_entry.data_ptr, dst_entry.data_size);
                                if (actual_hash != expected_hash) {
                                    LOG(ERR) <<
                                            "Shared state distributor transmitted incorrect shared state entry for key "
                                            << dst_entry.key << ": Expected hash " << expected_hash << " but got "
                                            << actual_hash;
                                    return false;
                                }
                            }
#else
                            if (dst_entry.device_type == ccoipDeviceCuda) {
                                LOG(BUG) << "PCCL is not built with CUDA support. We shouldn't even have gotten so far "
                                        "without CUDA support when referencing CUDA tensors. This is a bug!";
                                return false;
                            }
#endif
                            if (dst_entry.device_type == ccoipDeviceCpu) {
                                uint64_t actual_hash = hash_utils::simplehash_cpu(
                                    dst_entry.data_ptr, dst_entry.data_size);
                                if (actual_hash != expected_hash) {
                                    LOG(ERR) <<
                                            "Shared state distributor transmitted incorrect shared state entry for key "
                                            << dst_entry.key << ": Expected hash " << expected_hash << " but got "
                                            << actual_hash;
                                    return false;
                                }
                            }
                        }

                        if (expected_hash_type == ccoipHashCrc32) {
                            if (dst_entry.device_type == ccoipDeviceCuda) {
                                LOG(FATAL) << "CRC32 is currently not supported on CUDA devices.";
                                return false;
                            }
                            uint64_t actual_hash = hash_utils::CRC32(dst_entry.data_ptr, dst_entry.data_size);
                            if (actual_hash != expected_hash) {
                                LOG(ERR) << "Shared state distributor transmitted incorrect shared state entry for key "
                                        << dst_entry.key << ": Expected hash " << expected_hash << " but got "
                                        << actual_hash;
                                return false;
                            }
                        }
                    } else {
                        LOG(WARN) << "Master did not transmit expected hash for shared state entry " << dst_entry.key
                                << "; Skipping hash check...";
                    }
                }
            }
        }
        // indicate to master that shared state distribution is complete
        if (!master_socket.sendPacket<C2MPacketDistSharedStateComplete>({})) {
            LOG(ERR) << "Failed to sync shared state: Failed to send C2MPacketSyncSharedStateComplete to master";
            return false;
        }

        // wait for confirmation from master that all peers have synced the shared state
        if (const auto sync_response = master_socket.receivePacket<M2CPacketSyncSharedStateComplete>();
            !sync_response) {
            LOG(ERR) << "Failed to sync shared state: Failed to receive M2CPacketSyncSharedStateComplete response from "
                    "master";
            return false;
        }
        info_out.tx_bytes = client_state.getSharedStateSyncTxBytes();
        client_state.resetSharedStateSyncTxBytes();
    }
    return true;
}

bool ccoip::CCoIPClientHandler::optimizeTopology() {
    bool topology_optimization_complete = false;
    do {
        if (!master_socket.sendPacket<C2MPacketOptimizeTopology>({})) {
            return false;
        }
        const auto optimize_response = master_socket.receivePacket<M2CPacketOptimizeTopologyResponse>();
        if (!optimize_response) {
            return false;
        }
        if (!optimize_response->bw_benchmark_requests.empty()) {
            auto remaining_requests = std::list(optimize_response->bw_benchmark_requests.begin(),
                                                optimize_response->bw_benchmark_requests.end());
            auto iterator = remaining_requests.begin();

            // back off if we have already attempted to benchmark this peer recently to avoid spamming a busy server
            std::unordered_map<ccoip_uuid_t, std::chrono::steady_clock::time_point> last_attempt{};

            while (!remaining_requests.empty()) {
                if (iterator == remaining_requests.end()) {
                    // if we have recent attempts for all peers, wait for a bit
                    if (std::ranges::all_of(remaining_requests, [&last_attempt](const BenchmarkRequest &request) {
                        return last_attempt.contains(request.to_peer_uuid) &&
                               std::chrono::steady_clock::now() - last_attempt[request.to_peer_uuid] <
                               std::chrono::seconds(1);
                    })) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    }
                    iterator = remaining_requests.begin();
                }
                auto request = *iterator;
                if (request.from_peer_uuid != client_state.getAssignedUUID()) {
                    LOG(ERR) << "Received bandwidth benchmark request for peer "
                            << uuid_to_string(request.from_peer_uuid)
                            << " but this peer is not the target of the request. Ignoring...";
                    iterator = remaining_requests.erase(iterator);
                    continue;
                }
                if (last_attempt.contains(request.to_peer_uuid)) {
                    auto last_attempt_time = last_attempt[request.to_peer_uuid];
                    auto now = std::chrono::steady_clock::now();
                    if (now - last_attempt_time < std::chrono::seconds(1)) {
                        // try again later
                        ++iterator;
                        continue;
                    }
                }
                NetworkBenchmarkRunner runner(request.to_peer_benchmark_endpoint);
                const NetworkBenchmarkRunner::BenchmarkResult result = runner.runBlocking();

                if (result == NetworkBenchmarkRunner::BenchmarkResult::SUCCESS) {
                    const auto output_mbits_per_second = runner.getOutputBandwidthMbitsPerSecond();
                    C2MPacketReportPeerBandwidth packet{};
                    packet.to_peer_uuid = request.to_peer_uuid;
                    packet.bandwidth_mbits_per_second = output_mbits_per_second;
                    if (!master_socket.sendPacket<C2MPacketReportPeerBandwidth>(packet)) {
                        return false;
                    }
                    iterator = remaining_requests.erase(iterator);
                } else if (result == NetworkBenchmarkRunner::BenchmarkResult::BENCHMARK_SERVER_BUSY || result ==
                           NetworkBenchmarkRunner::BenchmarkResult::SEND_FAILURE) {
                    // try again later
                    // Even when we encounter a send failure, it is worth trying again because we actually did manage to establish a connection at first.
                    // We will check again to see if the peer is truly gone, and if it is, the next benchmark run with be a connection failure status anyway.
                    ++iterator;
                    last_attempt[request.to_peer_uuid] = std::chrono::steady_clock::now();
                } else {
                    // benchmark failed, simply log and continue
                    LOG(WARN) << "Failed to run network benchmark on endpoint "
                            << ccoip_sockaddr_to_str(request.to_peer_benchmark_endpoint)
                            << ". Ignoring and continuing...";
                    iterator = remaining_requests.erase(iterator);
                }
            }
        }
        if (const C2MPacketOptimizeTopologyWorkComplete complete_packet{};
            !master_socket.sendPacket<C2MPacketOptimizeTopologyWorkComplete>(complete_packet)) {
            return false;
        }
        const auto response = master_socket.receivePacket<M2CPacketOptimizeTopologyComplete>();
        if (!response) {
            return false;
        }
        topology_optimization_complete = response->success;
        client_state.updateTopology(response->ring_reduce_order);
    } while (!topology_optimization_complete);

    if (!requestAndEstablishP2PConnections(false)) {
        return false;
    }

    return true;
}

bool ccoip::CCoIPClientHandler::interrupt() {
    if (interrupted) {
        return false;
    }

    // interrupt all p2p connections
    {
        std::shared_lock lock(p2p_connections_rx_mutex);
        for (const auto &p2p_entry: p2p_connections_rx) {
            if (!p2p_entry.second->interrupt()) [[unlikely]] {
                return false;
            }
            p2p_entry.second->join();
        }
    }

    for (const auto &p2p_entry: p2p_connections_tx) {
        if (!p2p_entry.second->interrupt()) [[unlikely]] {
            return false;
        }
        p2p_entry.second->join();
    }
    if (!p2p_socket.interrupt()) [[unlikely]] {
        return false;
    }
    if (!shared_state_socket.interrupt()) [[unlikely]] {
        return false;
    }
    if (!benchmark_socket.interrupt()) [[unlikely]] {
        return false;
    }
    if (!master_socket.interrupt()) {
        // this can happen, if the connection was closed before, e.g. when kicked. We don't care about this case.
    }
    interrupted = true;
    return true;
}

bool ccoip::CCoIPClientHandler::join() {
    // await all running collective communications operations
    for (const auto tag: client_state.getRunningCollectiveComsOpTags()) {
        if (!client_state.joinAsyncCollectiveOp(tag)) {
            return false;
        }
    }

    // join all p2p connections
    {
        std::shared_lock lock(p2p_connections_rx_mutex);
        for (const auto &p2p_entry: p2p_connections_rx) {
            p2p_entry.second->join();
        }
    }

    for (const auto &p2p_entry: p2p_connections_tx) {
        p2p_entry.second->join();
    }

    p2p_socket.join();
    shared_state_socket.join();
    benchmark_socket.join();
    master_socket.join();
    if (benchmark_thread_opt.has_value()) {
        if (benchmark_thread_opt->joinable()) {
            benchmark_thread_opt->join();
        }
    }
    return true;
}

bool ccoip::CCoIPClientHandler::isInterrupted() const { return interrupted; }

ccoip::CCoIPClientHandler::~CCoIPClientHandler() {
    if (!interrupt()) {
        // do nothing, it may already have been interrupted by the user
    }
    if (!join()) [[unlikely]] {
        // do nothing here in destructor
    }
}

ccoip::CCoIPClientHandler::EstablishP2PConnectionResult ccoip::CCoIPClientHandler::establishP2PConnections() {
    // Note: intentionally no thread guard here, as this may be called when all reduces fail, which may be scheduled on concurrent threads

    // wait for connection info packet
    const auto connection_info_packet_opt = master_socket.receivePacket<M2CPacketP2PConnectionInfo>();
    if (!connection_info_packet_opt) {
        LOG(ERR) << "Failed to receive new peers packet";
        return FAILED;
    }
    const auto &connection_info_packet = connection_info_packet_opt.value();
    LOG(DEBUG) << "Received M2CPacketNewPeers from master";

    bool all_peers_connected = true;
    std::vector<ccoip_uuid_t> failed_peers{};
    client_state.setGlobalWorldSize(connection_info_packet.global_world_size);
    client_state.setLocalWorldSize(connection_info_packet.local_world_size);
    client_state.setNumDistinctPeerGroups(connection_info_packet.num_distinct_peer_groups);
    client_state.setLargestPeerGroupWorldSize(connection_info_packet.largest_peer_group_world_size);

    if (!connection_info_packet.unchanged) {
        LOG(DEBUG) << "New peers list has changed";

        // establish p2p connections
        for (auto &peer: connection_info_packet.all_peers) {
            // check if connection already exists
            if (p2p_connections_tx.contains(peer.peer_uuid)) {
                LOG(DEBUG) << "P2P connection with peer " << uuid_to_string(peer.peer_uuid) << " already exists";
                continue;
            }
            if (!establishP2PConnection(peer)) {
                LOG(ERR) << "Failed to establish P2P connection with peer " << uuid_to_string(peer.peer_uuid);
                all_peers_connected = false;
                failed_peers.push_back(peer.peer_uuid);
                break;
            }
        }

        // close p2p tx connections that are no longer needed
        for (auto it = p2p_connections_tx.begin(); it != p2p_connections_tx.end();) {
            const auto &peer_uuid = it->first;
            if (std::ranges::find_if(connection_info_packet.all_peers, [peer_uuid](const PeerInfo &peer) {
                return peer.peer_uuid == peer_uuid;
            }) == connection_info_packet.all_peers.end()) {
                LOG(DEBUG) << "Closing p2p connection with peer " << uuid_to_string(it->first);
                if (!closeP2PConnection(it->first, *it->second)) {
                    LOG(WARN) << "Failed to close p2p tx connection with peer " << uuid_to_string(it->first);
                }
                it = p2p_connections_tx.erase(it);
            } else {
                ++it;
            }
        }

        // close p2p rx connections that are no longer needed
        {
            std::unique_lock lock(p2p_connections_rx_mutex);
            for (auto it = p2p_connections_rx.begin(); it != p2p_connections_rx.end();) {
                const auto &peer_uuid = it->first;
                if (std::ranges::find_if(connection_info_packet.all_peers, [peer_uuid](const PeerInfo &peer) {
                    return peer.peer_uuid == peer_uuid;
                }) == connection_info_packet.all_peers.end()) {
                    LOG(DEBUG) << "Interrupting p2p rx connection with peer " << uuid_to_string(it->first);
                    if (!it->second->interrupt()) [[unlikely]] {
                        LOG(WARN) << "Failed to close p2p rx connection with peer " << uuid_to_string(it->first);
                    }
                    it = p2p_connections_rx.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    // send packet to this peer has established its p2p connections
    C2MPacketP2PConnectionsEstablished packet{};
    packet.success = all_peers_connected;

    // We report the fact that we couldn't establish p2p connections with this peer to the master.
    // This might indicate that traffic doesn't get routed between the peers, however they both might be able
    // to ping the master or other peers.
    // If there is any failed connection, we will retry to establish p2p connections.
    // This time, the master will not pair us with this peer.
    // If we are unlucky enough to get paired with another peer we can't connect to, this process
    // repeats until the master either kicks us because in the worst case we can't ping any peer,
    // but beyond that we might not be able to ping at least N peers, where N is the minimum amount of neighbors
    // required to sustain the topology.
    packet.failed_peers = failed_peers;

    if (!master_socket.sendPacket<C2MPacketP2PConnectionsEstablished>(packet)) {
        LOG(ERR) << "Failed to send P2P connections established packet";
        return FAILED;
    }

    // wait for response from master, indicating ALL peers have established their
    // respective p2p connections
    const auto response = master_socket.receivePacket<M2CPacketP2PConnectionsEstablished>();
    if (!response) {
        LOG(ERR) << "Failed to receive P2P connections established response";
        return FAILED;
    }
    if (!all_peers_connected && response->success) {
        LOG(BUG) << "Master indicated that all peers have established their p2p connections, but this peer has not and "
                "should have reported this to the master. This is a bug!";
        return FAILED;
    }
    if (!response->success) {
        LOG(ERR) << "Master indicated that p2p connection establishment has failed; Retrying...";
        return RETRY_NEEDED;
    }
    client_state.updateTopology(response->ring_reduce_order);
    return SUCCESS;
}

bool ccoip::CCoIPClientHandler::establishP2PConnection(const PeerInfo &peer) {
    // Note: intentionally no thread guard here, because this function may be called when an all reduce fails,
    // which is an operation that is allowed to be scheduled by threads other
    // than the communicator registered main thread.

    LOG(DEBUG) << "Establishing P2P connection with peer " << uuid_to_string(peer.peer_uuid);
    tinysockets::BlockingIOSocket socket(peer.p2p_listen_addr);
    if (!socket.establishConnection()) {
        LOG(ERR) << "Failed to establish P2P connection with peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    LOG(DEBUG) << "Established socket connection with peer " << uuid_to_string(peer.peer_uuid) <<
            "; Starting p2p handshake...";

    // maximize send and receive buffer sizes
    socket.maximizeSendBuffer();
    socket.maximizeReceiveBuffer();

    P2PPacketHello hello_packet{};
    hello_packet.peer_uuid = client_state.getAssignedUUID();
    if (!socket.sendPacket<P2PPacketHello>(hello_packet)) {
        LOG(ERR) << "Failed to send hello packet to peer " << uuid_to_string(peer.peer_uuid);
    }
    if (const auto response = socket.receivePacket<P2PPacketHelloAck>(); !response) {
        LOG(ERR) << "Failed to receive hello ack from peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    LOG(DEBUG) << "P2P handshake with peer " << uuid_to_string(peer.peer_uuid) << " successful.";
    auto [it, inserted] = p2p_connections_tx.emplace(
        peer.peer_uuid,
        std::make_unique<tinysockets::MultiplexedIOSocket>(socket.getSocketFd(), socket.getConnectSockAddr(),
                                                           tinysockets::ConnectionModeFlags::MODE_TX));
    if (!inserted) {
        LOG(ERR) << "P2P connection with peer " << uuid_to_string(peer.peer_uuid) << " already exists";
        return false;
    }
    if (!it->second->run()) {
        LOG(FATAL) << "Failed to start MultiplexedIOSocket for P2P connection with " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    if (!client_state.registerPeer(peer.p2p_listen_addr, peer.peer_uuid)) {
        LOG(ERR) << "Failed to register peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::closeP2PConnection(const ccoip_uuid_t &uuid, tinysockets::MultiplexedIOSocket &socket) {
    if (!socket.interrupt()) [[unlikely]] {
        LOG(BUG) << "Failed to close connection with peer " << uuid_to_string(uuid);
        return false;
    }

    if (!client_state.unregisterPeer(socket.getConnectSockAddr())) [[unlikely]] {
        LOG(ERR) << "Failed to unregister peer " << uuid_to_string(uuid)
                << ". This means the client was already unregistered; This is a bug!";
        return false;
    }

    // Wait for the socket-thread to exit
    // After the thread exits, the socket is guaranteed to be closed
    socket.join();
    return true;
}

void ccoip::CCoIPClientHandler::handleSharedStateRequest(const ccoip_socket_address_t &client_address,
                                                         const C2SPacketRequestSharedState &packet) {
    THREAD_GUARD(shared_state_server_thread_id);

    LOG(DEBUG) << "Received shared state distribution request from " << ccoip_sockaddr_to_str(client_address);

    std::vector<ccoip_shared_state_entry_t> entries_to_send{};

    S2CPacketSharedStateResponse response{};

    // check if we are in shared state sync phase
    if (!client_state.isSyncingSharedState()) {
        LOG(WARN) << "Received shared state request from " << ccoip_sockaddr_to_str(client_address)
                << " while not in shared state sync phase; Responding with status=SHARED_STATE_NOT_DISTRIBUTED";
        response.status = SHARED_STATE_NOT_DISTRIBUTED;
        goto end;
    }

    // construct the packet containing meta-data
    {
        response.status = SharedStateResponseStatus::SUCCESS;
        const auto &shared_state = client_state.getCurrentSharedState();
        response.revision = shared_state.revision;
        for (const auto &requested_key: packet.requested_keys) {
            const auto it = std::ranges::find_if(
                shared_state.entries,
                [&requested_key](const ccoip_shared_state_entry_t &entry) { return entry.key == requested_key; });
            if (it == shared_state.entries.end()) {
                response.entries.clear();
                entries_to_send.clear();
                response.status = UNKNOWN_SHARED_STATE_KEY;
                goto end;
            }
            response.entries.push_back(SharedStateEntry{.key = requested_key, .size_bytes = it->data_size});
            entries_to_send.push_back(*it);
            client_state.trackSharedStateTxBytes(it->data_size);
        }
    }
    LOG(DEBUG) << "Distributing shared state revision " << response.revision;
end:
    if (!shared_state_socket.sendPacket(client_address, response)) {
        LOG(ERR) << "Failed to send shared state response to " << ccoip_sockaddr_to_str(client_address);
    }

    [[maybe_unused]] uint32_t last_device_ordinal = -1; // this is needed for CUDA support

    for (const auto &entry: entries_to_send) {
        if (entry.device_type == ccoipDeviceCpu) {
            // if the tensor is already on the cpu, send directly
            if (!shared_state_socket.sendRawPacket(
                client_address, std::span(static_cast<const std::byte *>(entry.data_ptr), entry.data_size))) {
                LOG(ERR) << "Failed to send shared state data to client " << ccoip_sockaddr_to_str(client_address);
            }
        } else if (entry.device_type == ccoipDeviceCuda) {
#ifndef PCCL_HAS_CUDA_SUPPORT
            if (entry.device_type == ccoipDeviceCuda) {
                LOG(BUG) << "PCCL is not built with CUDA support. We shouldn't even have gotten so far without CUDA "
                        "support when referencing CUDA tensors. This is a bug!";
                return;
            }
#else
            const auto device_ptr = reinterpret_cast<CUdeviceptr>(entry.data_ptr);

            // get device the pointer is on
            int device_ordinal{};
            if (const CUresult result =
                        cuPointerGetAttribute(&device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, device_ptr);
                result != CUDA_SUCCESS) {
                const char *error_name{};
                cuGetErrorName(result, &error_name);
                const char *error_string{};
                cuGetErrorString(result, &error_string);

                LOG(FATAL) << "Failed to get device ordinal for device pointer " << device_ptr << "; "
                        << std::string(error_name) << ": " << std::string(error_string);
                return;
            }

            if (device_ordinal != last_device_ordinal) {
                last_device_ordinal = device_ordinal;

                // get device from device ordinal
                CUdevice device{};
                if (const CUresult result = cuDeviceGet(&device, device_ordinal); result != CUDA_SUCCESS) {
                    const char *error_name{};
                    cuGetErrorName(result, &error_name);
                    const char *error_string{};
                    cuGetErrorString(result, &error_string);

                    LOG(FATAL) << "Failed to get device handle from ordinal for device ordinal " << device_ordinal
                            << "; " << std::string(error_name) << ": " << std::string(error_string);
                    return;
                }
                CUcontext primary_ctx{};

                if (const CUresult result = cuDevicePrimaryCtxRetain(&primary_ctx, device); result != CUDA_SUCCESS) {
                    const char *error_name{};
                    cuGetErrorName(result, &error_name);
                    const char *error_string{};
                    cuGetErrorString(result, &error_string);

                    LOG(FATAL) << "Failed to retain primary context for device with ordinal " << device_ordinal << "; "
                            << std::string(error_name) << ": " << std::string(error_string);
                    return;
                }

                if (const CUresult result = cuCtxSetCurrent(primary_ctx); result != CUDA_SUCCESS) {
                    const char *error_name{};
                    cuGetErrorName(result, &error_name);
                    const char *error_string{};
                    cuGetErrorString(result, &error_string);

                    LOG(FATAL) << "Failed to set primary context for device with ordinal " << device_ordinal << "; "
                            << std::string(error_name) << ": " << std::string(error_string);
                    return;
                }
            }

            std::unique_ptr<std::byte[]> host_buffer(new std::byte[entry.data_size]);
            if (const auto result = cuMemcpyDtoH_v2(host_buffer.get(), device_ptr, entry.data_size);
                result != CUDA_SUCCESS) {
                const char *error_name{};
                cuGetErrorName(result, &error_name);
                const char *error_string{};
                cuGetErrorString(result, &error_string);

                LOG(FATAL)
                        << "Failed to copy cuda device memory to host while serving shared state transmission request; "
                        << std::string(error_name) << ": " << std::string(error_string);
                return;
            }
            if (!shared_state_socket.sendRawPacket(client_address, std::span(host_buffer.get(), entry.data_size))) {
                LOG(ERR) << "Failed to send shared state data to client " << ccoip_sockaddr_to_str(client_address);
            }
#endif
        } else {
            LOG(BUG) << "Unknown device type " << entry.device_type
                    << " encountered while serving shared state transmission request. This should have been caught "
                    "earlier and is a bug.";
            return;
        }
    }
}

void ccoip::CCoIPClientHandler::onSharedStateClientRead(const ccoip_socket_address_t &client_address,
                                                        const std::span<std::uint8_t> &data) {
    THREAD_GUARD(shared_state_server_thread_id);
    PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
    if (const auto packet_type = buffer.read<uint16_t>(); packet_type == C2SPacketRequestSharedState::packet_id) {
        C2SPacketRequestSharedState packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2SPacketRequestSharedState from "
                    << ccoip_sockaddr_to_str(client_address);
            return;
        }
        handleSharedStateRequest(client_address, packet);
    } else {
        LOG(ERR) << "Unknown packet type " << packet_type << " from " << ccoip_sockaddr_to_str(client_address);
        if (!shared_state_socket.closeClientConnection(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to close connection with " << ccoip_sockaddr_to_str(client_address);
        }
    }
}


bool ccoip::CCoIPClientHandler::allReduceAsync(const void *sendbuff, void *recvbuff, const size_t count,
                                               const ccoip_data_type_t datatype,
                                               const ccoip_data_type_t quantized_data_type,
                                               const ccoip_quantization_algorithm_t quantization_algorithm,
                                               const ccoip_reduce_op_t op, const uint64_t tag) {
    // NOTE: intentionally no thread guard here!

    if (client_state.isCollectiveComsOpRunning(tag)) {
        // tag is already in use
        return false;
    }


    // Set connection revision active at the time of launching the collective comms operation.
    // If the connection revision has changed since then,
    // we know that p2p connections were re-established in the meantime.
    client_state.setCollectiveConnectionRevision(tag, connection_revision);

    // intentional copies for thread-safety
    // p2p_connections_tx & p2p_connections_rx should only be modified by the main thread,
    // but we will read them in the async operation
    std::unordered_map p2p_connections_tx(this->p2p_connections_tx);

    p2p_connections_rx_mutex.lock_shared();
    std::unordered_map p2p_connections_rx(this->p2p_connections_rx);
    p2p_connections_rx_mutex.unlock_shared();

    const auto ring_order = client_state.getRingOrder();

    if (!client_state.launchAsyncCollectiveOp(tag, [this, sendbuff, recvbuff, count, datatype, quantized_data_type,
                                                  quantization_algorithm, op, tag,
                                                  ring_order,
                                                  p2p_connections_tx, p2p_connections_rx
                                              ](std::promise<bool> &promise) {
                                                  LOG(DEBUG) << "Vote to commence all reduce operation with tag " <<
                                                          tag;

                                                  bool aborted = false;

                                                  const auto reduce_fun = [&] {
                                                      // vote commence collective comms operation and await consensus
                                                      {
                                                          C2MPacketCollectiveCommsInitiate initiate_packet{};
                                                          initiate_packet.tag = tag;
                                                          initiate_packet.count = count;
                                                          initiate_packet.data_type = datatype;
                                                          initiate_packet.op = op;
                                                          if (!master_socket.sendPacket<
                                                              C2MPacketCollectiveCommsInitiate>(initiate_packet)) {
                                                              return std::pair{false, false};
                                                          }

                                                          // As long as we do not expect packets of types here that the main thread also expects and claims
                                                          // for normal operation (such as commencing other concurrent collective comms operations), we can
                                                          // safely use receiveMatchingPacket packet here to receive packets from the master socket. Given
                                                          // that M2CPacketCollectiveCommsCommence is only expected in this context, we will never "steal"
                                                          // anyone else's packet nor will the main thread steal our packet.
                                                          const auto response = master_socket.receiveMatchingPacket<
                                                              M2CPacketCollectiveCommsCommence>(
                                                              [tag](const M2CPacketCollectiveCommsCommence &
                                                          packet) {
                                                                  return packet.tag == tag;
                                                              });

                                                          if (!response) {
                                                              return std::pair{false, false};
                                                          }
                                                          LOG(DEBUG) <<
                                                                  "Received M2CPacketCollectiveCommsCommence for tag "
                                                                  << tag
                                                                  << "; Collective communications consensus reached";
                                                      }


                                                      // no need to actually all reduce when there is no second peer.
                                                      if (ring_order.size() < 2) {
                                                          return std::pair{false, false};
                                                      }

                                                      const auto &client_uuid = client_state.getAssignedUUID();

                                                      // find position in ring order
                                                      const auto it = std::ranges::find(ring_order, client_uuid);
                                                      if (it == ring_order.end()) {
                                                          return std::pair{false, false};
                                                      }
                                                      const size_t position = std::distance(ring_order.begin(), it);
                                                      const size_t byte_size = count * ccoip_data_type_size(datatype);

                                                      const std::span send_span(
                                                          static_cast<const std::byte *>(sendbuff), byte_size);
                                                      const std::span recv_span(
                                                          static_cast<std::byte *>(recvbuff), byte_size);

                                                      // perform pipeline ring reduce
                                                      auto [success, abort_packet_received] =
                                                              reduce::pipelineRingReduce(client_state, master_socket,
                                                                  tag, send_span, recv_span, datatype,
                                                                  quantized_data_type, op, quantization_algorithm,
                                                                  position, ring_order.size(), ring_order,
                                                                  p2p_connections_tx,
                                                                  p2p_connections_rx
                                                              );
                                                      return std::pair{success, abort_packet_received};
                                                  };
                                                  auto [success, abort_packet_received] = reduce_fun();
                                                  if (!success) {
                                                      LOG(WARN) <<
                                                              "An IO error occurred during the all reduce; Aborting collective communications operation...";
                                                  }
                                                  if (abort_packet_received) {
                                                      LOG(WARN) <<
                                                              "Received abort packet during all reduce with tag " << tag
                                                              << ". Considering all reduce aborted.";
                                                      aborted = true;
                                                      success = false;
                                                  }
                                                  if (![&] {
                                                      // vote collective comms operation complete and await consensus
                                                      C2MPacketCollectiveCommsComplete complete_packet{};
                                                      complete_packet.tag = tag;
                                                      complete_packet.was_aborted =
                                                              success == false && ring_order.size() > 1;

                                                      if (!master_socket.sendPacket<C2MPacketCollectiveCommsComplete>(
                                                          complete_packet)) {
                                                          return false;
                                                      }
                                                      LOG(DEBUG) << "Sent C2MPacketCollectiveCommsComplete for tag " <<
                                                              tag;

                                                      if (!abort_packet_received) {
                                                          const auto abort_response = master_socket.
                                                                  receiveMatchingPacket<M2CPacketCollectiveCommsAbort>(
                                                                      [tag](const M2CPacketCollectiveCommsAbort &
                                                                  packet) {
                                                                          return packet.tag == tag;
                                                                      });
                                                          if (!abort_response) {
                                                              return false;
                                                          }
                                                          aborted = abort_response->aborted;
                                                      }

                                                      const auto complete_response =
                                                              master_socket.receiveMatchingPacket<
                                                                  M2CPacketCollectiveCommsComplete>(
                                                                  [tag](const M2CPacketCollectiveCommsComplete &
                                                              packet) {
                                                                      return packet.tag == tag;
                                                                  });

                                                      if (!complete_response) {
                                                          return false;
                                                      }
                                                      if (aborted) {
                                                          LOG(WARN) <<
                                                                  "Collective communications operation was aborted";
                                                          return false;
                                                      }
                                                      return true;
                                                  }()) {
                                                      success = false;
                                                  }
                                                  promise.set_value(success);
                                                  return success;
                                              })) [[unlikely]] {
        return false;
    }

    return true;
}

bool ccoip::CCoIPClientHandler::joinAsyncReduce(const uint64_t tag) {
    // NOTE: intentionally no thread guard here

    if (!client_state.joinAsyncCollectiveOp(tag)) [[unlikely]] {
        return false;
    }
    const auto failure_opt = client_state.hasCollectiveComsOpFailed(tag);
    if (!failure_opt) [[unlikely]] {
        LOG(WARN) << "Collective coms op with tag " << tag << " was either not started or has not yet finished";
        return false;
    }
    if (*failure_opt) {
        // Discard all receive data in multiplexed p2p sockets for the tag of the failed reduce operation.
        // Otherwise, it can be that old data will be mistaken to be
        // good data in the next all reduce, which will inadvertently
        // throw off byte counters where one peer could have "received all the data"
        // while another peer is not yet done sending it because there was old data
        // that caused this discrepancy.
        // We also know that post joining the reduce operation, all peers have indicated completion of the
        // collection operation, and we thus know that we will not receive more p2p data and in turn
        // that discarding all data at this point safely discards all data still associated with the old aborted
        // collective communication operation.
        {
            std::shared_lock lock(p2p_connections_rx_mutex);
            for (auto &[peer_uuid, socket]: p2p_connections_rx) {
                socket->discardReceivedData(tag);
            }
        }

        // In that next invocation, we need to have the new topology ready
        // as necessitated that a peer just dropped, causing the abort.
        // We thus re-establish p2p connections according to the new topology as determined by the master,
        // and obtain the new topology.
        // However, we only want to re-establish once. There might be multiple concurrent all reduce operations
        // in flight, all of which could fail, and thus independently trigger a re-establishment of p2p connections,
        // which would also cause restarted all reduce operations to fail, causing a cascade of all-reduce
        // failures forever onwards.
        // We thus check if the connection revision has changed since the start of the all reduce.
        // If it has, we know that connections were already re-established, and we can safely skip this step.
        const auto revision_at_collective_begin_opt = client_state.getCollectiveConnectionRevision(tag);
        if (connection_revision == *revision_at_collective_begin_opt) {
            if (!requestAndEstablishP2PConnections(false)) {
                LOG(ERR) <<
                        "Failed to request and establish p2p connections after collective comms operation was aborted";
                return false;
            }
        }
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::getAsyncReduceInfo(const uint64_t tag, std::optional<ccoip_reduce_info_t> &info_out) {
    const auto world_size_opt = client_state.getCollectiveComsWorldSize(tag);
    const auto tx_bytes_opt = client_state.getCollectiveComsTxBytes(tag);
    const auto rx_bytes_opt = client_state.getCollectiveComsRxBytes(tag);

    // NOTE: TX bytes may be std::nullopt
    if (!world_size_opt) {
        // world_size_opt is a good indicator of whether this is the second invocation of getAsyncReduceInfo
        return false;
    }
    const auto world_size = *world_size_opt;
    const auto tx_bytes = tx_bytes_opt.has_value() ? *tx_bytes_opt : 0;
    const auto rx_bytes = rx_bytes_opt.has_value() ? *rx_bytes_opt : 0;
    info_out = ccoip_reduce_info_t{.tag = tag, .world_size = world_size, .tx_bytes = tx_bytes, .rx_bytes = rx_bytes};

    // we have to clear this information right here because otherwise it would just keep piling up in the hash maps
    // if the user uses new tag numbers for every collective comms operation.
    client_state.resetCollectiveComsWorldSize(tag);
    client_state.resetCollectiveComsTxBytes(tag);
    client_state.resetCollectiveComsRxBytes(tag);
    return true;
}

bool ccoip::CCoIPClientHandler::isAnyCollectiveComsOpRunning() {
    return client_state.isAnyCollectiveComsOpRunning();
}

size_t ccoip::CCoIPClientHandler::getGlobalWorldSize() const { return client_state.getGlobalWorldSize(); }

size_t ccoip::CCoIPClientHandler::getLocalWorldSize() const { return client_state.getLocalWorldSize(); }

size_t ccoip::CCoIPClientHandler::getNumDistinctPeerGroups() const { return client_state.getNumDistinctPeerGroups(); }

size_t ccoip::CCoIPClientHandler::getLargestPeerGroupWorldSize() const { return client_state.getLargestPeerGroupWorldSize(); }

void ccoip::CCoIPClientHandler::setMainThread(const std::thread::id main_thread_id) {
    this->main_thread_id = main_thread_id;
    client_state.setMainThread(main_thread_id);
}
