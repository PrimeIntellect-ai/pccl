#include "ccoip_master_handler.hpp"

#include <ccoip.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <thread_guard.hpp>
#include <tinysockets.hpp>
#include <uuid_utils.hpp>

ccoip::CCoIPMasterHandler::CCoIPMasterHandler(const ccoip_socket_address_t &listen_address) : server_socket(
    listen_address) {
    server_socket.addReadCallback([this](const ccoip_socket_address_t &client_address, const std::span<uint8_t> &data) {
        onClientRead(client_address, data);
    });
    server_socket.addCloseCallback([this](const ccoip_socket_address_t &client_address) {
        onClientDisconnect(client_address);
    });
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
            LOG(ERR) << "Failed to deserialize C2MPacketRequestSessionJoin from " << ccoip_sockaddr_to_str(
                client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleRequestSessionJoin(client_address, packet);
    } else if (packet_type == C2MPacketAcceptNewPeers::packet_id) {
        C2MPacketAcceptNewPeers packet{};
        packet.deserialize(buffer);
        handleAcceptNewPeers(client_address, packet);
    } else if (packet_type == C2MPacketP2PConnectionsEstablished::packet_id) {
        C2MPacketP2PConnectionsEstablished packet{};
        packet.deserialize(buffer);
        handleP2PConnectionsEstablished(client_address, packet);
    } else if (packet_type == C2MPacketGetTopologyRequest::packet_id) {
        C2MPacketGetTopologyRequest packet{};
        packet.deserialize(buffer);
        handleGetTopologyRequest(client_address, packet);
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
            LOG(ERR) << "Failed to deserialize C2MPacketCollectiveCommsInitiate from " << ccoip_sockaddr_to_str(
                client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        handleCollectiveCommsInitiate(client_address, packet);
    } else if (packet_type == C2MPacketCollectiveCommsComplete::packet_id) {
        C2MPacketCollectiveCommsComplete packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketCollectiveCommsComplete from " << ccoip_sockaddr_to_str(
                client_address);
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

    // generate uuid for new peer
    ccoip_uuid_t new_uuid{};
    uuid_utils::generate_uuid(new_uuid.data);

    // send response to new peer
    M2CPacketSessionRegistrationResponse response{};
    response.accepted = true;
    response.assigned_uuid = new_uuid;

    // register client uuid
    if (!server_state.registerClient(client_address,
                                     CCoIPClientVariablePorts{p2p_listen_port, shared_state_listen_port},
                                     packet.peer_group,
                                     new_uuid)) [[
        unlikely]] {
        LOG(ERR) << "Failed to register client " << ccoip_sockaddr_to_str(client_address);
        response.accepted = false;
    }

    // send response to new peer
    if (!server_socket.sendPacket(client_address, response)) [[unlikely]] {
        LOG(ERR) << "Failed to send M2CPacketJoinResponse to " << ccoip_sockaddr_to_str(client_address);
    }

    // if this is the first peer, simply send empty p2p connection information
    if (server_state.getClientSocketAddresses().size() == 1) {
        sendP2PConnectionInformation();
    } else {
        // otherwise still check if we have consensus to accept new peers
        checkAcceptNewPeersConsensus();
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

        if (peer_info.connection_state == WAITING_FOR_OTHER_PEERS) {
            any_waiting = true;
            break;
        }
    }
    if (!any_waiting) {
        return false;
    }

    // send establish new peers packets to all clients
    if (server_state.p2pConnectionsEstablishConsensus()) {
        // send confirmation packets to all clients
        if (!server_state.transitionToP2PConnectionsEstablishedPhase()) [[unlikely]] {
            LOG(WARN) << "Failed to transition to P2P connections established phase;";
            return false;
        }

        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            if (const auto &peer_info = peer_info_opt->get(); peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }
            if (!server_socket.sendPacket<C2MPacketP2PConnectionsEstablished>(peer_address, {})) {
                LOG(ERR) << "Failed to send M2CPacketP2PConnectionsEstablished to " << ccoip_sockaddr_to_str(
                    peer_address
                );
            }
        }
    }
    return true;
}

void ccoip::CCoIPMasterHandler::checkAcceptNewPeersConsensus() {
    // check if all clients have voted to accept new peers
    if (server_state.acceptNewPeersConsensus()) {
        server_state.transitionToP2PEstablishmentPhase();
        sendP2PConnectionInformation();
    }
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
            LOG(BUG) <<
                    "Failed to elect shared state mask; This may indicate no mask candidates were provided, despite the fact that shared state sync consensus was reached. This is a bug!";
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
                LOG(WARN) << "Kicking client " << ccoip_sockaddr_to_str(peer_address) <<
                        " due to shared state mismatch";
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
            if (peer_info.connection_state != DISTRIBUTE_SHARED_STATE && peer_info.connection_state !=
                REQUEST_SHARED_STATE) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) <<
                        " in state " << peer_info.connection_state <<
                        " but expected DISTRIBUTE_SHARED_STATE or REQUEST_SHARED_STATE";
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
                    if (outdated_keys.empty()) {
                        outdated_keys = server_state.getSharedStateKeys(peer_group);
                    }
                    response.outdated_keys = outdated_keys;

                    for (const auto &key: outdated_keys) {
                        response.expected_hashes.push_back(server_state.getSharedStateEntryHash(peer_group, key));
                    }
                } else {
                    LOG(ERR) << "No peer found to distribute shared state to " << ccoip_sockaddr_to_str(peer_address) <<
                            " while peers is marked to request shared state.";
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
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) <<
                        " in state " << peer_info.connection_state <<
                        " but expected IDLE";
                continue;
            }

            // send confirmation packet
            if (!server_socket.sendPacket<M2CPacketSyncSharedStateComplete>(peer_address, {})) {
                LOG(ERR) << "Failed to send M2CPacketSyncSharedStateComplete to " <<
                        ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
    return true;
}

bool ccoip::CCoIPMasterHandler::checkCollectiveCommsInitiateConsensus(const uint32_t peer_group,
                                                                      const uint64_t tag,
                                                                      bool topology_changed) {
    // check if all clients have voted to initiate the collective communications operation
    if (server_state.collectiveCommsInitiateConsensus(peer_group, tag)) {
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
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) <<
                        " in state " << peer_info.connection_state <<
                        " but expected COLLECTIVE_COMMUNICATIONS_RUNNING";
                continue;
            }

            // send confirmation packet
            M2CPacketCollectiveCommsCommence confirm_packet{};
            confirm_packet.tag = tag;
            confirm_packet.require_topology_update = topology_changed;
            if (!server_socket.sendPacket<M2CPacketCollectiveCommsCommence>(peer_address, confirm_packet)) {
                LOG(ERR) << "Failed to send M2CPacketCollectiveCommsCommence to " <<
                        ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
    return true;
}

bool ccoip::CCoIPMasterHandler::checkCollectiveCommsCompleteConsensus(const uint32_t peer_group, const uint64_t tag) {
    // check if all clients have voted to complete the collective communications operation
    if (server_state.collectiveCommsCompleteConsensus(peer_group, tag)) {
        if (!server_state.isCollectiveCommsOperationAborted(peer_group, tag)) {
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
            // on collective comms completion is IDLE
            if (peer_info.connection_state != IDLE) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) <<
                        " in state " << peer_info.connection_state <<
                        " but expected IDLE";
                continue;
            }

            // send confirmation packet
            M2CPacketCollectiveCommsComplete confirm_packet{};
            confirm_packet.tag = tag;
            if (!server_socket.sendPacket<M2CPacketCollectiveCommsComplete>(peer_address, confirm_packet)) {
                LOG(ERR) << "Failed to send M2CPacketCollectiveCommsComplete to " <<
                        ccoip_sockaddr_to_str(peer_address);
            }
            peer_info.collective_coms_states.erase(tag);
        }
    }
    return true;
}

void ccoip::CCoIPMasterHandler::handleP2PConnectionsEstablished(const ccoip_socket_address_t &client_address,
                                                                const C2MPacketP2PConnectionsEstablished &) {
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
    if (const auto client_uuid = client_uuid_opt.value();
        !server_state.markP2PConnectionsEstablished(client_uuid)) [[unlikely]] {
        LOG(WARN) << "Failed to mark P2P connections established for " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    if (!checkP2PConnectionsEstablished()) {
        LOG(BUG) << "checkP2PConnectionsEstablished() failed for " << ccoip_sockaddr_to_str(client_address) <<
                " when handling a p2p connection established packet. This should never happen!";
    }
}

void ccoip::CCoIPMasterHandler::handleGetTopologyRequest(const ccoip_socket_address_t &client_address,
                                                         const C2MPacketGetTopologyRequest &) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketGetTopologyRequest from " << ccoip_sockaddr_to_str(client_address);

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
    auto client_info_opt = server_state.getClientInfo(client_uuid);
    if (!client_info_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        return;
    }
    const auto &client_info = client_info_opt->get();
    const auto peer_group = client_info.peer_group;

    // TODO: implement real topology optimization,
    //  for now we assert ring reduce and return the ring order to be ascending order of client uuids
    std::vector<ccoip_uuid_t> topology{};
    for (const auto &[peer_uuid, _]: server_state.getClientEntrySet()) {
        const auto peer_info_opt = server_state.getClientInfo(peer_uuid);
        if (!peer_info_opt) {
            LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found";
            continue;
        }
        if (const auto &peer_info = peer_info_opt->get();
            peer_info.connection_phase != PEER_ACCEPTED || peer_info.peer_group != peer_group) {
            continue;
        }
        topology.push_back(peer_uuid);
    }
    std::ranges::sort(topology, [](const ccoip_uuid_t &a, const ccoip_uuid_t &b) {
        int cmp = 0;
        for (size_t i = 0; i < CCOIP_UUID_N_BYTES; i++) {
            cmp = a.data[i] - b.data[i];
            if (cmp != 0) {
                return cmp < 0;
            }
        }
        return false;
    });

    M2CPacketGetTopologyResponse response{};
    response.ring_reduce_order = topology;
    if (!server_socket.sendPacket<M2CPacketGetTopologyResponse>(client_address, response)) {
        LOG(ERR) << "Failed to send M2CPacketTopologyResponse to " << ccoip_sockaddr_to_str(client_address);
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
    const CCoIPMasterState::SharedStateMismatchStatus status = server_state
            .isNewRevisionLegal(client_uuid, packet.shared_state_revision);

    if (status == CCoIPMasterState::REVISION_INCREMENT_VIOLATION) {
        LOG(WARN) << "Shared state revision increment violation for " << ccoip_sockaddr_to_str(client_address) <<
                ". Please make sure all clients increment the shared state revision by exactly 1 each time before synchronizing shared state; Client will be kicked.";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    if (status == CCoIPMasterState::KEY_SET_MISMATCH) {
        LOG(WARN) << "Shared state key set mismatch for " << ccoip_sockaddr_to_str(client_address) <<
                ". Please make sure all clients have the same shared state keys; Client will be kicked.";
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
        LOG(BUG) << "checkSyncSharedStateConsensus() failed for " << ccoip_sockaddr_to_str(client_address) <<
                " when handling shared state sync packet. This should never happen.";
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
        LOG(WARN) << "Failed to vote to distribute shared state complete from " <<
                ccoip_sockaddr_to_str(client_address);
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
        LOG(BUG) << "checkSyncSharedStateCompleteConsensus() failed for " << ccoip_sockaddr_to_str(client_address) <<
                " when handling shared state sync complete packet. This should never happen.";
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
        LOG(WARN) << "Failed to vote to initiate a collective communications operation from " << ccoip_sockaddr_to_str(
            client_address);
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
        !checkCollectiveCommsInitiateConsensus(info.peer_group, packet.tag, false)) {
        LOG(BUG) << "checkCollectiveCommsInitiateConsensus() failed for " << ccoip_sockaddr_to_str(client_address) <<
                " when handling collective comms initiate packet. This should never happen.";
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
        LOG(DEBUG) << "Sent abort packet to " << ccoip_sockaddr_to_str(peer_address) << " with abort state: " <<
                aborted;
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
        LOG(WARN) << "Failed to vote to complete a collective communications operation from " <<
                ccoip_sockaddr_to_str(client_address);
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
        LOG(BUG) << "checkCollectiveCommsCompleteConsensus() failed for " << ccoip_sockaddr_to_str(client_address) <<
                " when handling collective comms complete packet. This should never happen.";
    }
}

void ccoip::CCoIPMasterHandler::onClientDisconnect(const ccoip_socket_address_t &client_address) {
    THREAD_GUARD(server_thread_id);

    LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(client_address) << " disconnected";
    if (!server_state.isClientRegistered(client_address)) {
        // client disconnected before ever requesting to register
        LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " disconnected before registering. Strange, but not impossible";
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
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
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
    checkAcceptNewPeersConsensus();
    if (!checkP2PConnectionsEstablished()) {
        LOG(INFO) <<
                "checkP2PConnectionsEstablished() returned false; This likely means no clients are waiting for other peers and is expected during disconnects.";
    }
    if (!checkSyncSharedStateConsensus(client_info.peer_group)) {
        LOG(INFO) <<
                "checkSyncSharedStateConsensus() returned false; This likely means no clients are waiting for shared state sync and is expected during disconnects.";
    }
    if (!checkSyncSharedStateCompleteConsensus(client_info.peer_group)) {
        LOG(INFO) <<
                "checkSyncSharedStateCompleteConsensus() returned false; This likely means no clients are waiting for shared state sync completion and is expected during disconnects.";
    }

    for (const auto &tag: server_state.getOngoingCollectiveComsOpTags(client_info.peer_group)) {
        if (!checkCollectiveCommsCompleteConsensus(client_info.peer_group, tag)) {
            LOG(INFO) <<
                    "checkCollectiveCommsCompleteConsensus() returned false; This likely means no clients are waiting for collective comms completion and is expected during disconnects.";
        }
        if (!checkCollectiveCommsInitiateConsensus(client_info.peer_group, tag, true)) {
            LOG(INFO) <<
                    "checkCollectiveCommsInitiateConsensus() returned false; This likely means no clients are waiting for collective comms initiation and is expected during disconnects.";
        }
    }
}

void ccoip::CCoIPMasterHandler::handleAcceptNewPeers(const ccoip_socket_address_t &client_address,
                                                     const C2MPacketAcceptNewPeers &) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketAcceptNewPeers from " << ccoip_sockaddr_to_str(client_address);

    const auto client_uuid_opt = server_state.findClientUUID(client_address);
    if (!client_uuid_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }
    if (const auto client_uuid = client_uuid_opt.value();
        !server_state.voteAcceptNewPeers(client_uuid)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to accept new peers from " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    checkAcceptNewPeersConsensus();
}

ccoip::CCoIPMasterHandler::~CCoIPMasterHandler() = default;

void ccoip::CCoIPMasterHandler::sendP2PConnectionInformation() {
    const bool unchanged = !server_state.hasPeerListChanged();

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
        M2CPacketNewPeers new_peers{};
        new_peers.unchanged = unchanged;

        auto peers = server_state.getPeersForClient(peer_address); // get the peers for the client
        new_peers.new_peers.reserve(peers.size());
        for (const auto &client_info: peers) {
            // construct a new peers packet
            new_peers.new_peers.push_back({
                .p2p_listen_addr = ccoip_socket_address_t{
                    .inet = client_info.socket_address.inet,
                    .port = client_info.variable_ports.p2p_listen_port
                },
                .peer_uuid = client_info.client_uuid
            });
        }

        if (!server_socket.sendPacket(peer_address, new_peers)) {
            LOG(ERR) << "Failed to send M2CPacketNewPeers to " << ccoip_sockaddr_to_str(peer_address);
        }
    }
}
