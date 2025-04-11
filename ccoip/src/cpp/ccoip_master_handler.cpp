#include "ccoip_master_handler.hpp"

#include <ccoip.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <thread_guard.hpp>
#include <tinysockets.hpp>
#include <topology_optimizer.hpp>
#include <uuid_utils.hpp>

ccoip::CCoIPMasterHandler::CCoIPMasterHandler(const ccoip_socket_address_t &listen_address) : server_socket(
        listen_address),
    topology_optimization_threadpool(4, 64) {
    topology_optimization_threadpool.startup();

    server_socket.addReadCallback([this](const ccoip_socket_address_t &client_address, const std::span<uint8_t> &data) {
        onClientRead(client_address, data);
    });
    server_socket.addCloseCallback(
        [this](const ccoip_socket_address_t &client_address) { onClientDisconnect(client_address); });
}

bool ccoip::CCoIPMasterHandler::run() {
    if (!server_socket.listen()) {
        return false;
    }
    LOG(DEBUG) << "CCoIPMasterHandler listening on port " << server_socket.getListenPort();
    if (!server_socket.runAsync()) {
        return false;
    }
    server_thread_id = server_socket.getServerThreadId();
    running = true;
    return true;
}

bool ccoip::CCoIPMasterHandler::interrupt() {
    if (interrupted) {
        return true;
    }
    if (!server_socket.interrupt()) {
        return false;
    }
    interrupted = true;
    return true;
}

bool ccoip::CCoIPMasterHandler::join() {
    if (!running) {
        return false;
    }
    server_socket.join();
    return true;
}

bool ccoip::CCoIPMasterHandler::kickClient(const ccoip_socket_address_t &client_address) const {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Kicking client " << ccoip_sockaddr_to_str(client_address);
    if (!server_socket.closeClientConnection(client_address)) [[unlikely]] {
        return false;
    }
    return true;
}

void ccoip::CCoIPMasterHandler::onClientRead(const ccoip_socket_address_t &client_address,
                                             const std::span<uint8_t> &data) {
    THREAD_GUARD(server_thread_id);
    PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
    if (const auto packet_type = buffer.read<uint16_t>();
        packet_type == C2MPacketRequestSessionRegistration::packet_id) {
        C2MPacketRequestSessionRegistration packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketRequestSessionJoin from "
                    << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleRequestSessionJoin(client_address, packet);
    } else if (packet_type == C2MPacketRequestEstablishP2PConnections::packet_id) {
        C2MPacketRequestEstablishP2PConnections packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketRequestEstablishP2PConnections from "
                    << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleEstablishP2PConnections(client_address, packet);
    } else if (packet_type == C2MPacketP2PConnectionsEstablished::packet_id) {
        C2MPacketP2PConnectionsEstablished packet{};
        if (!packet.deserialize(buffer)) [[unlikely]] {
            LOG(ERR) << "Failed to deserialize C2MPacketP2PConnectionsEstablished from "
                    << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleP2PConnectionsEstablished(client_address, packet);
    } else if (packet_type == C2MPacketCheckPeersPending::packet_id) {
        C2MPacketCheckPeersPending packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketCheckPeersPending from "
                    << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleCheckPeersPending(client_address, packet);
    } else if (packet_type == C2MPacketOptimizeTopology::packet_id) {
        C2MPacketOptimizeTopology packet{};
        packet.deserialize(buffer);
        handleOptimizeTopology(client_address, packet);
    } else if (packet_type == C2MPacketReportPeerBandwidth::packet_id) {
        C2MPacketReportPeerBandwidth packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketReportPeerBandwidth from "
                    << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleReportPeerBandwidth(client_address, packet);
    } else if (packet_type == C2MPacketOptimizeTopologyWorkComplete::packet_id) {
        C2MPacketOptimizeTopologyWorkComplete packet{};
        packet.deserialize(buffer);
        handleOptimizeTopologyWorkComplete(client_address, packet);
    } else if (packet_type == C2MPacketSyncSharedState::packet_id) {
        C2MPacketSyncSharedState packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketSyncSharedState from " << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleSyncSharedState(client_address, packet);
    } else if (packet_type == C2MPacketDistSharedStateComplete::packet_id) {
        C2MPacketDistSharedStateComplete packet{};
        packet.deserialize(buffer);
        handleSyncSharedStateComplete(client_address, packet);
    } else if (packet_type == C2MPacketCollectiveCommsInitiate::packet_id) {
        C2MPacketCollectiveCommsInitiate packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketCollectiveCommsInitiate from "
                    << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleCollectiveCommsInitiate(client_address, packet);
    } else if (packet_type == C2MPacketCollectiveCommsComplete::packet_id) {
        C2MPacketCollectiveCommsComplete packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketCollectiveCommsComplete from "
                    << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleCollectiveCommsComplete(client_address, packet);
    } else {
        LOG(ERR) << "Unknown packet type " << packet_type << " from " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
    }
}

void ccoip::CCoIPMasterHandler::handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                                         const C2MPacketRequestSessionRegistration &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketRequestSessionJoin from " << ccoip_sockaddr_to_str(client_address);

    // check if peer has already joined
    if (server_state.isClientRegistered(client_address)) {
        LOG(WARN) << "Peer " << ccoip_sockaddr_to_str(client_address) << " has already joined";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    const uint16_t p2p_listen_port = packet.p2p_listen_port;
    const uint16_t shared_state_listen_port = packet.shared_state_listen_port;
    const uint16_t bandwidth_benchmark_listen_port = packet.bandwidth_benchmark_listen_port;


    // generate uuid for new peer
    ccoip_uuid_t new_uuid{};
    uuid_utils::generate_uuid(new_uuid.data);

    // send response to new peer
    M2CPacketSessionRegistrationResponse response{};
    response.accepted = true;
    response.assigned_uuid = new_uuid;

    // register client uuid
    if (!server_state.registerClient(
        client_address,
        CCoIPClientVariablePorts{
            .p2p_listen_port = p2p_listen_port,
            .shared_dist_state_listen_port = shared_state_listen_port,
            .bandwidth_benchmark_listen_port = bandwidth_benchmark_listen_port
        },
        packet.peer_group, new_uuid)) [[unlikely]] {
        LOG(ERR) << "Failed to register client " << ccoip_sockaddr_to_str(client_address);
        response.accepted = false;
    }

    // send response to new peer
    if (!server_socket.sendPacket(client_address, response)) [[unlikely]] {
        LOG(ERR) << "Failed to send M2CPacketJoinResponse to " << ccoip_sockaddr_to_str(client_address);
    }

    // if this is the first peer, simply send empty p2p connection information
    if (server_state.getClientSocketAddresses().size() == 1) {
        sendP2PConnectionInformation(true);
    } else {
        // otherwise still check if we have consensus to accept new peers
        if (!checkEstablishP2PConnectionConsensus()) [[unlikely]] {
            LOG(BUG) << "checkAcceptNewPeersConsensus() failed. This is a bug";
        }
    }
}

bool ccoip::CCoIPMasterHandler::checkP2PConnectionsEstablished() {
    // check if at least one client is waiting for other peers
    bool any_waiting = false;
    for (const auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
        const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
        if (!peer_info_opt) [[unlikely]] {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();

        // all connection phases are legal (both REGISTERED & ACCEPTED)
        if (peer_info.connection_state == WAITING_FOR_OTHER_PEERS ||
            peer_info.connection_state == CONNECTING_TO_PEERS_FAILED) {
            any_waiting = true;
            break;
        }
    }
    if (!any_waiting) {
        return false;
    }

    // send establish new peers packets to all clients
    if (server_state.p2pConnectionsEstablishConsensus()) {
        LOG(DEBUG) << "All clients have declared that they have established P2P connections";

        bool any_failed = false;
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto &peer_info = peer_info_opt->get();
            if (peer_info.connection_state == CONNECTING_TO_PEERS_FAILED) {
                any_failed = true;
                break;
            }
        }
        // send confirmation packets to all clients
        if (!server_state.transitionToP2PConnectionsEstablishedPhase(any_failed || peer_dropped)) [[unlikely]] {
            peer_dropped = false; // reset peer_dropped state, as we are returning
            LOG(BUG) << "Failed to transition to P2P connections established phase;";
            return false;
        }

        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto &peer_info = peer_info_opt->get();
            if (peer_info.connection_phase == PEER_REGISTERED && peer_info.connection_state == IDLE) {
                // ignore clients that have not made the cut for the current peer acceptance phase
                continue;
            }
            M2CPacketP2PConnectionsEstablished packet{};
            if (peer_dropped) {
                packet.success = false;
            } else {
                packet.success = !any_failed;
            }

            // TODO: implement real topology optimization,
            //  for now we assert ring reduce and return the ring order to be ascending order of client uuids
            const auto topology = server_state.getRingTopology(peer_info.peer_group, false);

            packet.ring_reduce_order = topology;

            if (!server_socket.sendPacket<M2CPacketP2PConnectionsEstablished>(peer_address, packet)) {
                LOG(ERR) << "Failed to send M2CPacketP2PConnectionsEstablished to "
                        << ccoip_sockaddr_to_str(peer_address);
            }
        }
        peer_dropped = false; // reset peer_dropped state
    }
    return true;
}

bool ccoip::CCoIPMasterHandler::checkQueryPeersPendingConsensus() {
    if (server_state.queryPendingPeersConsensus()) {
        const bool has_pending_peers = server_state.hasPendingPeers();
        M2CPacketPeersPendingResponse packet{};
        packet.peers_pending = has_pending_peers;

        // send query peers pending response packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto client_info_opt = server_state.getClientInfo(peer_uuid);
            if (!client_info_opt) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto client_info = client_info_opt->get();
            if (client_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }
            if (!server_socket.sendPacket<M2CPacketPeersPendingResponse>(peer_address, packet)) {
                LOG(ERR) << "Failed to send M2CPacketPeersPending to " << ccoip_sockaddr_to_str(peer_address);
            }
        }
        server_state.resetVoteQueryPendingPeers();
    }
    return true;
}

bool ccoip::CCoIPMasterHandler::checkEstablishP2PConnectionConsensus() {
    // check if all clients have voted to accept new peers
    if (server_state.acceptNewPeersConsensus()) {
        peer_dropped = false;
        // reset peer dropped state, as we are starting a completely new p2p connection establishment phase
        if (!server_state.transitionToP2PEstablishmentPhase(true)) [[unlikely]] {
            LOG(BUG) << "Failed to transition to P2P establishment phase. This is a bug!";
            return false;
        }
        // we do include REGISTERED peers in the world size stats because they will become accepted,
        // if this phase completes successfully, which must happen before peers are allowed to proceed.
        sendP2PConnectionInformation(true);
    }

    // check if all clients have voted to establish p2p connections without accepting new peers
    if (server_state.noAcceptNewPeersEstablishP2PConnectionsConsensus()) {
        peer_dropped = false;
        // reset peer dropped state, as we are starting a completely new p2p connection establishment phase
        if (!server_state.transitionToP2PEstablishmentPhase(false)) [[unlikely]] {
            LOG(BUG) << "Failed to transition to P2P establishment phase. This is a bug!";
            return false;
        }
        // we don't include REGISTERED peers in the world size stats because they will not get accepted
        // here because peers have voted not to do so.
        sendP2PConnectionInformation(false);
    }
    return true;
}

#define CALLSITE_TOPOLOGY_OPTIMIZATION_START __COUNTER__

bool ccoip::CCoIPMasterHandler::checkTopologyOptimizationConsensus() {
    // check if all clients have voted to optimize the topology
    if (server_state.optimizeTopologyConsensus()) {
        if (!server_state.transitionToTopologyOptimizationPhase()) [[unlikely]] {
            LOG(BUG) << "Failed to transition to topology optimization phase. This is a bug!";
            return false;
        }

        // send optimize topology response packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            if (const auto &peer_info = peer_info_opt->get(); peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }
            M2CPacketOptimizeTopologyResponse packet{};

            std::vector<BenchmarkRequest> benchmark_requests{};
            const auto &bandwidth_entries = server_state.getMissingBandwidthEntries(peer_uuid);
            for (const auto &bandwidth_entry: bandwidth_entries) {
                const auto &from_peer_uuid = bandwidth_entry.from_peer_uuid;

                // There is no need to send benchmark requests to peers to which they are not addressed.
                // A benchmark request is formulated to be executed by the "from" peer, which transfers data to the "to"
                // peer. The "from" peer reports back to the master with the measured bandwidth during transfer. Hence,
                // the "from" peer is the one that initiates the benchmark, and thus the peer that needs to be addressed
                // by this request.
                if (from_peer_uuid != peer_uuid) {
                    continue;
                }

                const auto &to_peer_uuid = bandwidth_entry.to_peer_uuid;
                const auto to_peer_info_opt = server_state.getClientInfo(to_peer_uuid);
                if (!to_peer_info_opt) [[unlikely]] {
                    LOG(BUG) << "Client " << uuid_to_string(to_peer_uuid) << " not found";
                    continue;
                }
                const auto &peer_info = to_peer_info_opt->get();
                if (peer_info.connection_phase != PEER_ACCEPTED) {
                    continue;
                }
                BenchmarkRequest request{
                    from_peer_uuid, to_peer_uuid,
                    ccoip_socket_address_t{
                        .inet = peer_info.socket_address.inet,
                        .port = peer_info.variable_ports.bandwidth_benchmark_listen_port,
                    }
                };
                LOG(DEBUG) << "Requesting bandwidth information from " << uuid_to_string(from_peer_uuid) << " to "
                        << uuid_to_string(to_peer_uuid)
                        << "; Endpoint: " << ccoip_sockaddr_to_str(request.to_peer_benchmark_endpoint);

                benchmark_requests.push_back(request);
            }
            packet.bw_benchmark_requests = benchmark_requests;
            if (!server_socket.sendPacket<M2CPacketOptimizeTopologyResponse>(peer_address, packet)) {
                LOG(ERR) << "Failed to send C2MPacketOptimizeTopology to " << ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
    return true;
}

void ccoip::CCoIPMasterHandler::performTopologyOptimization(const uint32_t peer_group) {
    const auto task_it = topology_optimization_tasks.find(peer_group);

    bool moonshot = false;
    if (task_it != topology_optimization_tasks.end()) {
        moonshot = true; // run moonshot optimization if this is the second time we are optimizing the topology

        // if we have an ongoing optimization task, we join it first
        task_it->second.join();
    }

    // launch topology optimization task, if solution is not optimal
    if (!server_state.isTopologyOptimal(peer_group)) {
        {
            auto future = topology_optimization_threadpool.scheduleTask(
                [this, peer_group] {
                    std::vector<ccoip_uuid_t> new_topology{};
                    bool is_optimal = false;
                    bool has_improved = false;
                    if (!server_state.
                        performTopologyOptimization(peer_group, false, new_topology, is_optimal, has_improved)) {
                        LOG(WARN) << "Failed to perform topology optimization!";
                        return;
                    }
                    if (has_improved) {
                        return;
                    }
                    if (!server_state.setRingTopology(peer_group, new_topology, is_optimal)) {
                        LOG(BUG) << "Failed to update topology. This means we tried to update a topology when it was "
                                "already optimal. This is a bug!";
                    }
                });
            topology_optimization_tasks.insert({peer_group, std::move(future)});
        }

        const auto &future = topology_optimization_tasks.at(peer_group);
        if (!moonshot) {
            // await non-moonshot optimization immediately
            future.join();
        }
    }
}


bool ccoip::CCoIPMasterHandler::checkTopologyOptimizationCompletionConsensus() {
    // check if at least one client is waiting for topology optimization to complete
    if (server_state.topologyOptimizationCompleteConsensus()) {
        /*
         * check if all bandwidth benchmark requests have been completed
        const bool success = server_state.isBandwidthStoreFullyPopulated();

        if (!success) {
            LOG(WARN) << "Not all bandwidth benchmarks have been completed! Topology optimization failed. Retrying...";
            LOG(DEBUG) << "Number of registered peers in bandwidth store: "
                       << server_state.getNumBandwidthStoreRegisteredPeers();
        }*/

        // We don't actually assert that the bandwidth store is fully populated.
        // not all peers may be able to talk to any other peer. The ATSP solver will navigate around those challenges
        // and find the optimal tour taking into account the routing limitations.
        constexpr bool success = true;

        // Mark the edges that are still missing as unreachable so we don't try them again.
        // if we don't do this, we would constantly be running a benchmark that will time out, only stalling
        // the run with every invocation of optimize topology.
        if (!server_state.isBandwidthStoreFullyPopulated()) {
            for (const auto &peer_uuid: server_state.getCurrentlyAcceptedPeers()) {
                const auto missing_entries = server_state.getMissingBandwidthEntries(peer_uuid);
                for (const auto &missing_entry: missing_entries) {
                    server_state.markBandwidthEntryUnreachable(missing_entry);
                }
            }
        }

        if (!server_state.endTopologyOptimizationPhase(!success)) {
            LOG(BUG) << "Failed to end topology optimization phase. This is a bug!";
            return false;
        }

        if (success) {
            for (const uint32_t peer_group: server_state.getExistingPeerGroups()) {
                if (!server_state.isTopologyOptimal(peer_group)) {
                    performTopologyOptimization(peer_group);
                }
            }
        }

        // send topology optimization complete packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            if (const auto &peer_info = peer_info_opt->get(); peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }
            M2CPacketOptimizeTopologyComplete packet{};
            packet.success = success;
            if (!server_socket.sendPacket<M2CPacketOptimizeTopologyComplete>(peer_address, packet)) {
                LOG(ERR) << "Failed to send M2CPacketTopologyOptimizationComplete to "
                        << ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
    return true;
}


std::optional<ccoip_socket_address_t>
ccoip::CCoIPMasterHandler::findBestSharedStateTxPeer(const ccoip_uuid_t &peer_uuid) {
    const auto info_opt = server_state.getClientInfo(peer_uuid);
    if (!info_opt) [[unlikely]] {
        LOG(BUG) << "Client " << uuid_to_string(peer_uuid) << " not found";
        return std::nullopt;
    }
    const auto &info = info_opt->get();

    // TODO: should be topology aware
    // for now, just return the first peer that distributes the shared state
    for (const auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
        const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
        if (!peer_info_opt) [[unlikely]] {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();
        if (peer_info.peer_group != info.peer_group) {
            continue;
        }
        if (peer_info.connection_phase == PEER_ACCEPTED && peer_info.connection_state == DISTRIBUTE_SHARED_STATE) {
            return ccoip_socket_address_t{peer_address.inet, peer_info.variable_ports.shared_dist_state_listen_port};
        }
    }
    return std::nullopt;
}

bool ccoip::CCoIPMasterHandler::checkSyncSharedStateConsensus(const uint32_t peer_group) {
    bool any_waiting = false;
    // check if at least one client is waiting for shared state sync
    for (const auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
        const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
        if (!peer_info_opt) [[unlikely]] {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();
        if (peer_info.peer_group != peer_group || peer_info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (peer_info.connection_state == VOTE_SYNC_SHARED_STATE) {
            any_waiting = true;
            break;
        }
    }
    if (!any_waiting) {
        return false;
    }

    // check if all clients have voted to sync shared state
    if (server_state.syncSharedStateConsensus(peer_group)) {
        // elect mask from candidates
        if (!server_state.electSharedStateMask(peer_group)) [[unlikely]] {
            LOG(BUG) << "Failed to elect shared state mask; This may indicate no mask candidates were provided, "
                    "despite the fact that shared state sync consensus was reached. This is a bug!";
            return false;
        }

        // check for mismatched shared state entries; violating clients will be marked.
        // the mismatch status can be queried via server_state.getSharedStateMismatchStatus()
        if (!server_state.checkMaskSharedStateMismatches(peer_group)) {
            LOG(WARN) << "Failed to check shared state mask mismatch. No current mask set.";
            return false;
        }

        // kick all clients with mismatched shared state
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto &peer_info = peer_info_opt->get();
            if (peer_info.peer_group != peer_group || peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }
            const auto status_opt = server_state.getSharedStateMismatchStatus(peer_uuid);
            if (!status_opt) {
                LOG(BUG) << "No shared state mismatch status found for client " << ccoip_sockaddr_to_str(peer_address)
                        << " after checkMaskSharedStateMismatches was invoked. This is a bug!";
                return false;
            }
            if (*status_opt == CCoIPMasterState::KEY_SET_MISMATCH) {
                LOG(WARN) << "Kicking client " << ccoip_sockaddr_to_str(peer_address)
                        << " due to shared state mismatch";
                if (!kickClient(peer_address)) [[unlikely]] {
                    LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(peer_address);
                    return false;
                }
                return true; // this is not a failure case; we have kicked a client and should recover
            }
        }

        if (!server_state.transitionToSharedStateSyncPhase(peer_group)) [[unlikely]] {
            LOG(BUG) << "Failed to transition to shared state distribution phase; This is a bug!";
            return false;
        }

        // send confirmation packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto &peer_info = peer_info_opt->get();
            if (peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }

            if (peer_info.peer_group != peer_group) {
                continue;
            }

            // only these are valid states for accepted clients to be in after the shared state voting phase
            if (peer_info.connection_state != DISTRIBUTE_SHARED_STATE &&
                peer_info.connection_state != REQUEST_SHARED_STATE) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " in state "
                        << peer_info.connection_state
                        << " but expected DISTRIBUTE_SHARED_STATE or REQUEST_SHARED_STATE";
                continue;
            }

            M2CPacketSyncSharedState response{};

            // if the state is REQUEST_SHARED_STATE, then transitionToSharedStateDistributionPhase has
            // determined that the shared state hash does not match; notify the client to re-request the shared state
            const bool needs_update = peer_info.connection_state == REQUEST_SHARED_STATE;
            response.is_outdated = needs_update;
            if (needs_update) {
                if (auto best_peer_opt = findBestSharedStateTxPeer(peer_uuid)) {
                    response.distributor_address = *best_peer_opt;

                    auto outdated_keys = server_state.getOutdatedSharedStateKeys(peer_uuid);
                    response.outdated_keys = outdated_keys;

                    for (const auto &key: outdated_keys) {
                        response.expected_hashes.push_back(server_state.getSharedStateEntryHash(peer_group, key));
                        const auto hash_type_opt = server_state.getSharedStateEntryHashType(peer_group, key);
                        if (!hash_type_opt) {
                            LOG(BUG) << "Hash type for shared state entry " << key
                                    << " could not be found in peer group " << peer_group
                                    << " despite the fact that a hash has been recorded. This is a bug.";
                            return false;
                        }
                        response.expected_hash_types.push_back(*hash_type_opt);
                    }
                } else {
                    LOG(ERR) << "No peer found to distribute shared state to " << ccoip_sockaddr_to_str(peer_address)
                            << " while peers is marked to request shared state.";
                }
            }
            if (!server_socket.sendPacket<M2CPacketSyncSharedState>(peer_address, response)) {
                LOG(ERR) << "Failed to send M2CPacketSyncSharedState to " << ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
    return true;
}

bool ccoip::CCoIPMasterHandler::checkSyncSharedStateCompleteConsensus(const uint32_t peer_group) {
    bool any_waiting = false;
    // check if at least one client is waiting for shared state sync completion
    for (const auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
        const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
        if (!peer_info_opt) [[unlikely]] {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();
        if (peer_info.peer_group != peer_group || peer_info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (peer_info.connection_state == VOTE_COMPLETE_SHARED_STATE_SYNC) {
            any_waiting = true;
            break;
        }
    }
    if (!any_waiting) {
        return false;
    }
    // check if all clients have voted to distribute shared state complete
    if (server_state.syncSharedStateCompleteConsensus(peer_group)) {
        if (!server_state.endSharedStateSyncPhase(peer_group)) [[unlikely]] {
            LOG(BUG) << "Failed to end shared state distribution phase; This is a bug!";
            return false;
        }

        // send confirmation packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto &peer_info = peer_info_opt->get();
            if (peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }
            if (peer_info.peer_group != peer_group) {
                continue;
            }

            // because endSharedStateSyncPhase() was already invoked,
            // the only valid state for accepted clients to be in after a successful vote
            // on shared state distribution completion is IDLE
            if (peer_info.connection_state != IDLE) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " in state "
                        << peer_info.connection_state << " but expected IDLE";
                continue;
            }

            // send confirmation packet
            if (!server_socket.sendPacket<M2CPacketSyncSharedStateComplete>(peer_address, {})) {
                LOG(ERR) << "Failed to send M2CPacketSyncSharedStateComplete to "
                        << ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
    return true;
}

bool ccoip::CCoIPMasterHandler::checkCollectiveCommsInitiateConsensus(const uint32_t peer_group, const uint64_t tag) {
    // check if all clients have voted to initiate the collective communications operation
    const auto has_collective_consensus = server_state.collectiveCommsInitiateConsensus(peer_group, tag);
    LOG(DEBUG) << "Has collective comms initiate consensus in checkCollectiveCommsInitiateConsensus: "
            << has_collective_consensus;

    if (has_collective_consensus) {
        if (!server_state.transitionToPerformCollectiveCommsPhase(peer_group, tag)) {
            LOG(BUG) << "Failed to transition to collective communications initiate phase; This is a bug";
            return false;
        }

        // send confirmation packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto &peer_info = peer_info_opt->get();
            if (peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }

            if (peer_info.peer_group != peer_group) {
                continue;
            }

            // because transitionToPerformCollectiveCommsPhase() was already invoked,
            // the only valid state for accepted clients to be in after a successful vote
            // on collective comms initiation is COLLECTIVE_COMMUNICATIONS_RUNNING
            if (peer_info.connection_state != COLLECTIVE_COMMUNICATIONS_RUNNING) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " in state "
                        << peer_info.connection_state << " but expected COLLECTIVE_COMMUNICATIONS_RUNNING";
                continue;
            }

            // send confirmation packet
            M2CPacketCollectiveCommsCommence confirm_packet{};
            confirm_packet.tag = tag;
            if (!server_socket.sendPacket<M2CPacketCollectiveCommsCommence>(peer_address, confirm_packet)) {
                LOG(ERR) << "Failed to send M2CPacketCollectiveCommsCommence to "
                        << ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
    return true;
}

bool ccoip::CCoIPMasterHandler::checkCollectiveCommsCompleteConsensus(const uint32_t peer_group, const uint64_t tag) {
    // check if all clients have voted to complete the collective communications operation
    const auto has_collective_consensus = server_state.collectiveCommsCompleteConsensus(peer_group, tag);
    LOG(DEBUG) << "Collective comms complete consensus in checkCollectiveCommsCompleteConsensus: " <<
            has_collective_consensus;
    if (has_collective_consensus) {
        const auto is_collective_aborted = server_state.isCollectiveCommsOperationAborted(peer_group, tag);
        LOG(DEBUG) << "Collective comms operation aborted in checkCollectiveCommsCompleteConsensus: " <<
                is_collective_aborted;
        if (!is_collective_aborted) {
            // if this operation was never aborted, send abort packets with abort state = false
            // we do this such that for each collective comms operation, there is exactly one abort packet
            // sent to each peer - either during the operation to signal an abort, or after the operation
            // to signal that the operation was not aborted.
            // We do this so that we always expect exactly one abort packet for each operation, which avoids
            // problems relating these packets being handled in the next collective comms operation due to
            // timing issues, where it would falsely be interpreted as an abort for the next operation.
            sendCollectiveCommsAbortPackets(peer_group, tag, false);
        }

        if (!server_state.transitionToCollectiveCommsCompletePhase(peer_group, tag)) {
            LOG(BUG) << "Failed to transition to collective communications complete phase; This is a bug";
            return false;
        }

        // send confirmation packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            auto &peer_info = peer_info_opt->get();
            if (peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }

            if (peer_info.peer_group != peer_group) {
                continue;
            }

            // because transitionToCollectiveCommsCompletePhase() was already invoked,
            // the only valid state for accepted clients to be in after a successful vote
            // on collective comms completion is IDLE (or COLLECTIVE_COMMUNICATIONS_RUNNING when multiple collective ops are running concurrently)
            if (peer_info.connection_state != IDLE && peer_info.connection_state != COLLECTIVE_COMMUNICATIONS_RUNNING) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " in state "
                        << peer_info.connection_state << " but expected IDLE";
                continue;
            }

            // send confirmation packet
            M2CPacketCollectiveCommsComplete confirm_packet{};
            confirm_packet.tag = tag;
            if (!server_socket.sendPacket<M2CPacketCollectiveCommsComplete>(peer_address, confirm_packet)) {
                LOG(ERR) << "Failed to send M2CPacketCollectiveCommsComplete to "
                        << ccoip_sockaddr_to_str(peer_address);
            }
            peer_info.collective_coms_states.erase(tag);
        }
    }
    return true;
}

void ccoip::CCoIPMasterHandler::handleP2PConnectionsEstablished(const ccoip_socket_address_t &client_address,
                                                                const C2MPacketP2PConnectionsEstablished &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketP2PConnectionsEstablished from " << ccoip_sockaddr_to_str(client_address);

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) [[unlikely]] {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    const auto client_uuid = client_uuid_opt.value();

    if (!server_state.markP2PConnectionsEstablished(client_uuid, packet.success, packet.failed_peers)) [[unlikely]] {
        LOG(WARN) << "Failed to mark P2P connections established for " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    if (!checkP2PConnectionsEstablished()) {
        LOG(BUG) << "checkP2PConnectionsEstablished() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling a p2p connection established packet. This should never happen!";
    }
}

void ccoip::CCoIPMasterHandler::handleCheckPeersPending(const ccoip_socket_address_t &client_address,
                                                        const C2MPacketCheckPeersPending &) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketCheckPeersPending from " << ccoip_sockaddr_to_str(client_address);

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    const auto client_uuid = client_uuid_opt.value();

    if (!server_state.voteQueryWaitingPeersPending(client_uuid)) {
        LOG(WARN) << "Failed to mark peers pending for " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    if (!checkQueryPeersPendingConsensus()) {
        LOG(BUG) << "checkPeersPendingConsensus() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling a check peers pending packet. This should never happen!";
    }
}


void ccoip::CCoIPMasterHandler::handleOptimizeTopology(const ccoip_socket_address_t &client_address,
                                                       const C2MPacketOptimizeTopology &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketOptimizeTopology from " << ccoip_sockaddr_to_str(client_address);

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    if (const auto client_uuid = client_uuid_opt.value(); !server_state.voteOptimizeTopology(client_uuid)) [[unlikely]]
    {
        LOG(WARN) << "Failed to vote to optimize topology from " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    if (!checkTopologyOptimizationConsensus()) {
        LOG(BUG) << "checkTopologyOptimizationConsensus() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling collective comms initiate packet. This should never happen.";
    }
}

void ccoip::CCoIPMasterHandler::handleReportPeerBandwidth(const ccoip_socket_address_t &client_address,
                                                          const C2MPacketReportPeerBandwidth &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketReportPeerBandwidth from " << ccoip_sockaddr_to_str(client_address);

    const auto from_client_uuid_opt = server_state.findClientUUID(client_address);
    if (!from_client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    const auto from_client_uuid = from_client_uuid_opt.value();
    server_state.storePeerBandwidth(from_client_uuid, packet.to_peer_uuid, packet.bandwidth_mbits_per_second);
}

void ccoip::CCoIPMasterHandler::handleOptimizeTopologyWorkComplete(const ccoip_socket_address_t &client_address,
                                                                   const C2MPacketOptimizeTopologyWorkComplete &) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketOptimizeTopologyWorkComplete from " << ccoip_sockaddr_to_str(client_address);

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    if (const auto client_uuid = client_uuid_opt.value(); !server_state.voteTopologyOptimizationComplete(client_uuid)) [
        [unlikely]] {
        LOG(WARN) << "Failed to vote to optimize topology work complete from " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    if (!checkTopologyOptimizationCompletionConsensus()) {
        LOG(BUG) << "checkTopologyOptimizationCompletionConsensus() failed for "
                << ccoip_sockaddr_to_str(client_address)
                << " when handling collective comms initiate packet. This should never happen.";
    }
}

void ccoip::CCoIPMasterHandler::handleSyncSharedState(const ccoip_socket_address_t &client_address,
                                                      const C2MPacketSyncSharedState &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketSyncSharedState from " << ccoip_sockaddr_to_str(client_address);

    // obtain client uuid from client address
    ccoip_uuid_t client_uuid{}; {
        const auto client_uuid_opt = server_state.findClientUUID(client_address);
        if (!client_uuid_opt) [[unlikely]] {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        client_uuid = client_uuid_opt.value();
    }

    // check if shared state request follows the right "mask" as in:
    // 1. requests same keys, dtype, etc. as the other clients; mismatch will result in a kick
    // 2. hash of the shared state is the same as the hash of the shared state of the other clients; mismatch will
    //    result in the client being notified to re-request the shared state entry whose hash does not match.
    // If the client is the first to sync shared state, then that peer's shared state is the reference
    // "mask" for the other clients.
    const CCoIPMasterState::SharedStateMismatchStatus status =
            server_state.isNewRevisionLegal(client_uuid, packet.shared_state_revision);

    if (status == CCoIPMasterState::REVISION_INCREMENT_VIOLATION) {
        LOG(WARN) << "Shared state revision increment violation for " << ccoip_sockaddr_to_str(client_address)
                << ". Please make sure all clients increment the shared state revision by exactly 1 each time before "
                "synchronizing shared state; Client will be kicked.";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    if (status == CCoIPMasterState::KEY_SET_MISMATCH) {
        LOG(WARN) << "Shared state key set mismatch for " << ccoip_sockaddr_to_str(client_address)
                << ". Please make sure all clients have the same shared state keys; Client will be kicked.";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    // vote for sync shared state
    if (!server_state.voteSyncSharedState(client_uuid)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to sync shared state from " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    server_state.voteSharedStateMask(client_uuid, packet.shared_state_hashes);

    const auto info_opt = server_state.getClientInfo(client_uuid);
    if (!info_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        return;
    }
    if (const auto &info = info_opt->get(); !checkSyncSharedStateConsensus(info.peer_group)) {
        LOG(BUG) << "checkSyncSharedStateConsensus() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling shared state sync packet. This should never happen.";
    }
}

void ccoip::CCoIPMasterHandler::handleSyncSharedStateComplete(const ccoip_socket_address_t &client_address,
                                                              const C2MPacketDistSharedStateComplete &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketSyncSharedStateComplete from " << ccoip_sockaddr_to_str(client_address);

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    const auto client_uuid = client_uuid_opt.value();
    if (!server_state.voteDistSharedStateComplete(client_uuid)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to distribute shared state complete from "
                << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    const auto info_opt = server_state.getClientInfo(client_uuid);
    if (!info_opt) {
        LOG(WARN) << "Client " << uuid_to_string(client_uuid) << " not found";
        return;
    }
    if (const auto &info = info_opt->get(); !checkSyncSharedStateCompleteConsensus(info.peer_group)) {
        LOG(BUG) << "checkSyncSharedStateCompleteConsensus() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling shared state sync complete packet. This should never happen.";
    }
}

void ccoip::CCoIPMasterHandler::handleCollectiveCommsInitiate(const ccoip_socket_address_t &client_address,
                                                              const C2MPacketCollectiveCommsInitiate &packet) {
    THREAD_GUARD(server_thread_id);

    LOG(DEBUG) << "Received C2MPacketCollectiveCommsInitiate from " << ccoip_sockaddr_to_str(client_address);
    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    const auto client_uuid = client_uuid_opt.value();
    if (!server_state.voteCollectiveCommsInitiate(client_uuid, packet.tag)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to initiate a collective communications operation from "
                << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    const auto info_opt = server_state.getClientInfo(client_uuid);
    if (!info_opt) {
        LOG(WARN) << "Client " << uuid_to_string(client_uuid) << " not found";
        return;
    }
    if (const auto &info = info_opt->get();
        !checkCollectiveCommsInitiateConsensus(info.peer_group, packet.tag)) {
        LOG(BUG) << "checkCollectiveCommsInitiateConsensus() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling collective comms initiate packet. This should never happen.";
    }
}

void ccoip::CCoIPMasterHandler::sendCollectiveCommsAbortPackets(const uint32_t peer_group, const uint64_t tag,
                                                                const bool aborted) {
    M2CPacketCollectiveCommsAbort abort_packet{};
    abort_packet.tag = tag;
    abort_packet.aborted = aborted;

    for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
        const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
        if (!peer_info_opt) [[unlikely]] {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();
        if (peer_info.peer_group != peer_group || peer_info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (peer_info.connection_state != COLLECTIVE_COMMUNICATIONS_RUNNING) {
            continue;
        }
        if (!server_socket.sendPacket<M2CPacketCollectiveCommsAbort>(peer_address, abort_packet)) {
            LOG(ERR) << "Failed to send M2CPacketCollectiveCommsAbort to " << ccoip_sockaddr_to_str(peer_address);
        }
        LOG(DEBUG) << "Sent abort packet to " << ccoip_sockaddr_to_str(peer_address)
                << " with abort state: " << aborted << " for tag " << tag;
    }
}

void ccoip::CCoIPMasterHandler::handleCollectiveCommsComplete(const ccoip_socket_address_t &client_address,
                                                              const C2MPacketCollectiveCommsComplete &packet) {
    THREAD_GUARD(server_thread_id);

    LOG(DEBUG) << "Received C2MPacketCollectiveCommsComplete from " << ccoip_sockaddr_to_str(client_address);

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    const auto client_uuid = client_uuid_opt.value();
    if (!server_state.voteCollectiveCommsComplete(client_uuid, packet.tag)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to complete a collective communications operation from "
                << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    const auto info_opt = server_state.getClientInfo(client_uuid);
    if (!info_opt) {
        LOG(WARN) << "Client " << uuid_to_string(client_uuid) << " not found";
        return;
    }

    const auto &info = info_opt->get();

    if (packet.was_aborted) {
        if (server_state.abortCollectiveCommsOperation(info.peer_group, packet.tag)) {
            // this peer is the first to report an abort; notify all other peers
            sendCollectiveCommsAbortPackets(info.peer_group, packet.tag, true);
        }
    }

    if (!checkCollectiveCommsCompleteConsensus(info.peer_group, packet.tag)) {
        LOG(BUG) << "checkCollectiveCommsCompleteConsensus() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling collective comms complete packet. This should never happen.";
    }
}

void ccoip::CCoIPMasterHandler::onClientDisconnect(const ccoip_socket_address_t &client_address) {
    THREAD_GUARD(server_thread_id);

    LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(client_address) << " disconnected";
    if (!server_state.isClientRegistered(client_address)) {
        // client disconnected before ever requesting to register
        LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(client_address)
                << " disconnected before registering. Strange, but not impossible";
        return;
    }

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        return;
    }
    const auto client_uuid = client_uuid_opt.value();
    const auto client_info_opt = server_state.getClientInfo(client_uuid);
    if (!client_info_opt) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        return;
    }
    const ClientInfo client_info = client_info_opt->get();
    if (!server_state.unregisterClient(client_address)) {
        LOG(WARN) << "Failed to unregister client " << ccoip_sockaddr_to_str(client_address);
        return;
    }

    // the client disconnecting might result in consensus changes for ongoing votes;
    // If the client that left was the only outstanding vote to e.g. accept new peers, then we need to re-check
    // the consensus and send response packets as necessary.
    if (!checkEstablishP2PConnectionConsensus()) [[unlikely]] {
        LOG(BUG) << "checkAcceptNewPeersConsensus() failed. This is a bug";
    }

    // If a client disconnects, and we are in the middle of a p2p establishment phase,
    // some of those p2p connections might have been successfully established before a problematic peer leaves,
    // maybe said peer has even already confirmed successful establishment to the master,
    // but those p2p connections are now outdated and do not reflect the topology as it will be when the client
    // disconnection is fully handled by the master.
    // This is why we in this case set a flag to indicate that the establishment phase must fail.
    // This flag is respected in the checkP2PConnectionsEstablished phase.
    peer_dropped = true;

    if (!checkP2PConnectionsEstablished()) {
        LOG(DEBUG) << "checkP2PConnectionsEstablished() returned false; This likely means no clients are waiting for "
                "other peers and is expected during disconnects.";
    }

    if (!checkQueryPeersPendingConsensus()) {
        LOG(BUG) << "checkPeersPendingConsensus() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling a check peers pending packet. This should never happen!";
    }

    if (!checkSyncSharedStateConsensus(client_info.peer_group)) {
        LOG(DEBUG) << "checkSyncSharedStateConsensus() returned false; This likely means no clients are waiting for "
                "shared state sync and is expected during disconnects.";
    }

    if (!checkSyncSharedStateCompleteConsensus(client_info.peer_group)) {
        LOG(DEBUG) << "checkSyncSharedStateCompleteConsensus() returned false; This likely means no clients are "
                "waiting for shared state sync completion and is expected during disconnects.";
    }

    const auto ongoing_tags = server_state.getOngoingCollectiveCommsOpTags(client_info.peer_group);
    std::string ongoing_tags_str{};
    for (const auto &tag: ongoing_tags) {
        ongoing_tags_str += std::to_string(tag) + ", ";
    }
    LOG(DEBUG) << "Ongoing collective comms operations for peer group " << client_info.peer_group << ": "
            << ongoing_tags_str;
    for (const auto &tag: ongoing_tags) {
        if (!checkCollectiveCommsInitiateConsensus(client_info.peer_group, tag)) {
            LOG(DEBUG) << "checkCollectiveCommsInitiateConsensus() returned false; This likely means no clients are "
                    "waiting for collective comms initiation and is expected during disconnects.";
        }
        if (!checkCollectiveCommsCompleteConsensus(client_info.peer_group, tag)) {
            LOG(DEBUG) << "checkCollectiveCommsCompleteConsensus() returned false; This likely means no clients are "
                    "waiting for collective comms completion and is expected during disconnects.";
        }
        const auto collective_running = server_state.isCollectiveOperationRunning(client_info.peer_group, tag);
        LOG(DEBUG) << "Collective comms operation with tag " << tag << " is running: " << collective_running;
        if (collective_running) {
            const auto collective_aborted = server_state.abortCollectiveCommsOperation(client_info.peer_group, tag);
            LOG(DEBUG) << "Collective comms operation with tag " << tag << " was aborted: " << collective_aborted;
            if (collective_aborted) {
                // In this case, the master is the first to notice that a collective op needs to be aborted
                // due to a client leaving that participated in said operation.
                // Clients may report io failures themselves which also results in the collective op being aborted,
                // but so must the master.
                sendCollectiveCommsAbortPackets(client_info.peer_group, tag, true);
            }
        }
    }
}

void ccoip::CCoIPMasterHandler::handleEstablishP2PConnections(const ccoip_socket_address_t &client_address,
                                                              const C2MPacketRequestEstablishP2PConnections &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketEstablishP2PConnections from " << ccoip_sockaddr_to_str(client_address);

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    if (const auto client_uuid = client_uuid_opt.value();
        !server_state.voteEstablishP2PConnections(client_uuid, packet.accept_new_peers)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to establish p2p connections (with or without accepting new peers) from "
                << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    if (!checkEstablishP2PConnectionConsensus()) [[unlikely]] {
        LOG(BUG) << "checkAcceptNewPeersConsensus() failed for " << ccoip_sockaddr_to_str(client_address)
                << " when handling request establish p2p connections packet. This should never happen.";
    }
}

ccoip::CCoIPMasterHandler::~CCoIPMasterHandler() {
    topology_optimization_threadpool.shutdown();
}

void ccoip::CCoIPMasterHandler::sendP2PConnectionInformation(const bool changed, const ClientInfo &peer_info,
                                                             const bool include_registered) {
    M2CPacketP2PConnectionInfo new_peers{};
    new_peers.unchanged = !changed;
    new_peers.global_world_size = server_state.getGlobalWorldSize();
    new_peers.local_world_size = server_state.getLocalWorldSize(peer_info.peer_group, include_registered);
    new_peers.num_distinct_peer_groups = server_state.getNumDistinctPeerGroups(include_registered);
    new_peers.largest_peer_group_world_size = server_state.getLargestPeerGroupWorldSize(include_registered);

    const auto peer_address = peer_info.socket_address;

    if (changed) {
        const auto peers = server_state.getPeersForClient(peer_address, peer_info, include_registered); // get the peers for the client
        new_peers.all_peers.reserve(peers.size());
        for (const auto &client_info: peers) {
            // construct a new peers packet
            new_peers.all_peers.push_back(
                {
                    .p2p_listen_addr = ccoip_socket_address_t{
                        .inet = client_info.socket_address.inet,
                        .port = client_info.variable_ports.p2p_listen_port
                    },
                    .peer_uuid = client_info.client_uuid
                });
        }
    }

    LOG(DEBUG) << "Sending p2p connection information to peer " << ccoip_sockaddr_to_str(peer_address);
    if (!server_socket.sendPacket(peer_address, new_peers)) {
        LOG(ERR) << "Failed to send M2CPacketNewPeers to " << ccoip_sockaddr_to_str(peer_address);
    }
}

void ccoip::CCoIPMasterHandler::sendP2PConnectionInformation(const bool include_registered) {
    const bool changed = server_state.hasPeerListChanged(CALLSITE_P2P_CONNECTION_INFORMATION) ||
                         server_state.hasTopologyChanged(CALLSITE_P2P_CONNECTION_INFORMATION);

    // send establish new peers packets to all clients
    for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
        const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
        if (!peer_info_opt) [[unlikely]] {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();
        if (peer_info.connection_state != CONNECTING_TO_PEERS) {
            continue;
        }

        // for all connected clients
        sendP2PConnectionInformation(changed, peer_info, include_registered);
    }
}
