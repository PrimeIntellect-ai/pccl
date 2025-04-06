#include "ccoip_master_state.hpp"

#include <ccoip_packets.hpp>
#include <functional>
#include <pccl_log.hpp>
#include <thread_guard.hpp>
#include <topology_optimizer.hpp>

inline bool isLocalAddress(const ccoip_socket_address_t &client_address) {
    if (client_address.inet.protocol == inetIPv4) {
        if (client_address.inet.ipv4.data[0] == 127) {
            // all 127.x.x.x addresses are local
            return true;
        }
    } else if (client_address.inet.protocol == inetIPv6) {
        // check if all zero except position 15
        bool is_local = true;
        for (int i = 0; i < 15; i++) {
            if (client_address.inet.ipv6.data[i] != 0) {
                is_local = false;
                break;
            }
        }
        if (client_address.inet.ipv6.data[15] != 1) {
            is_local = false;
        }
        return is_local;
    }
    return false;
}

bool ccoip::CCoIPMasterState::registerClient(const ccoip_socket_address_t &client_address,
                                             const CCoIPClientVariablePorts &variable_ports,
                                             const uint32_t peer_group,
                                             const ccoip_uuid_t uuid) {
    // If it is a local address, only allow it to join if only local clients have joined.
    // If only local clients have joined, do not allow any non-local clients to join.
    // This is crucial because the clients would have to talk to the peers through a different IP
    // other than 127.0.0.1 because they would just hit themselves.
    // If the master cannot tell the real IP of the client, we cannot reliably inform clients how to reach each other.
    {
        const bool is_local = isLocalAddress(client_address);

        if (!client_info.empty()) {
            bool all_existing_local = true;
            for (const auto &[_, client_info]: client_info) {
                if (!isLocalAddress(client_info.socket_address)) {
                    all_existing_local = false;
                    break;
                }
            }
            if (all_existing_local && !is_local) {
                LOG(WARN) << "Non-Local client " << ccoip_sockaddr_to_str(client_address)
                        << " cannot join because only local clients have joined";
                return false;
            }
            if (!all_existing_local && is_local) {
                LOG(WARN) << "Local client " << ccoip_sockaddr_to_str(client_address)
                        << " cannot join because non-local clients have joined";
                return false;
            }
        }
    }

    if (isClientRegistered(client_address)) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " already registered";
        return false;
    }

    const auto internal_address = ccoip_socket_to_internal(client_address);
    client_uuids[internal_address] = uuid;
    uuid_clients[uuid] = internal_address;
    client_info[uuid] = ClientInfo{
        .client_uuid = uuid,
        .connection_phase = PEER_REGISTERED,
        .connection_state = IDLE,
        .socket_address = client_address,
        .variable_ports = variable_ports,
        .peer_group = peer_group
    };

    // set all callsites in peer_list_changed to true
    for (auto &[_, has_changed]: peer_list_changed) {
        has_changed = true;
    }

    // if this is the first client, consider it as voting to accept new peers
    if (client_uuids.size() == 1) {
        auto &info = client_info[uuid];
        info.connection_phase = PEER_ACCEPTED; // consider it accepted
        onPeerAccepted(info);
        if (!voteEstablishP2PConnections(uuid, true)) [[unlikely]] {
            LOG(WARN) << "Failed to vote to accept new peers for first client "
                    << ccoip_sockaddr_to_str(client_address);
            return false;
        }
        if (!acceptNewPeersConsensus()) [[unlikely]] {
            LOG(BUG) << "Inconsistent state: the first and only client voted to accept new peers, but not all clients "
                    "have voted";
            return false;
        }
        if (!transitionToP2PEstablishmentPhase(true)) [[unlikely]] {
            LOG(BUG) << "Failed to transition to P2P establishment phase; This is a bug";
        }
    }
    return true;
}

void ccoip::CCoIPMasterState::onPeerAccepted(const ClientInfo &info) {
    const auto uuid = info.client_uuid;
    int world_size = 0;
    for (const auto &[uuid, info]: client_info) {
        if (info.connection_phase == PEER_ACCEPTED) {
            world_size++;
        }
    }
    LOG(DEBUG) << "Peer " << uuid_to_string(uuid)
            << " has been accepted (PEER_ACCEPTED). New world size: " << world_size;
    if (!bandwidth_store.registerPeer(uuid)) {
        LOG(BUG) << "Failed to register bandwidth data for client " << uuid_to_string(uuid)
                << "; This means the peer was already registered. This is a bug";
    }
}

bool ccoip::CCoIPMasterState::unregisterClient(const ccoip_socket_address_t &client_address) {
    if (!isClientRegistered(client_address)) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not registered";
        return false;
    }
    const auto internal_address = ccoip_socket_to_internal(client_address);
    if (const auto it = client_uuids.find(internal_address); it != client_uuids.end()) {
        if (!uuid_clients.erase(it->second)) {
            LOG(BUG) << "Client with UUID " << uuid_to_string(it->second)
                    << " not found in uuid->sockaddr mapping. This means bi-directional mapping for client UUIDs is "
                    "inconsistent";
            return false;
        }
        const auto info_it = client_info.find(it->second);
        if (info_it == client_info.end()) {
            LOG(BUG) << "ClientInfo of client with UUID " << uuid_to_string(it->second)
                    << " not found in uuid->ClientInfo mapping. This means client info mapping is inconsistent";
            return false;
        }
        const auto &info = info_it->second;
        const uint32_t peer_group = info.peer_group;
        const auto connection_phase = info.connection_phase;
        client_info.erase(info_it);

        const auto peer_uuid = it->second;

        if (connection_phase == PEER_ACCEPTED) {
            if (!bandwidth_store.unregisterPeer(peer_uuid)) {
                LOG(BUG) << "Failed to unregister bandwidth data for client " << uuid_to_string(peer_uuid)
                        << "; This means the peer was already unregistered or never registered. This is a bug";
                return false;
            }
        }

        // remove from all voting sets
        votes_accept_new_peers.erase(peer_uuid);
        votes_establish_p2p_connections.erase(peer_uuid);
        votes_optimize_topology.erase(peer_uuid);
        votes_complete_topology_optimization.erase(peer_uuid);
        votes_sync_shared_state[peer_group].erase(peer_uuid);
        votes_sync_shared_state_complete[peer_group].erase(peer_uuid);
        client_uuids.erase(it);

        // set all callsites in peer_list_changed to true
        for (auto &[_, has_changed]: peer_list_changed) {
            has_changed = true;
        }
    } else {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
    }
    return true;
}

bool ccoip::CCoIPMasterState::isClientRegistered(const ccoip_socket_address_t &client_address) const {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    return client_uuids.contains(internal_address);
}

bool ccoip::CCoIPMasterState::voteEstablishP2PConnections(const ccoip_uuid_t &peer_uuid, const bool accept_new_peers) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to accept new peers for unregistered client " << uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to accept new peers
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to accept new peers in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to accept new peers, the client must be idle
    // or in the CONNECTING_TO_PEERS_FAILED state
    if (info.connection_state != IDLE && info.connection_state != CONNECTING_TO_PEERS_FAILED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to accept new peers in state " << info.connection_state;
        return false;
    }

    // set the client state to vote to accept new peers
    if (accept_new_peers) {
        info.connection_state = VOTE_ACCEPT_NEW_PEERS;
        if (auto [_, inserted] = votes_accept_new_peers.insert(info.client_uuid); !inserted) {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " found in votes_accept_new_peers set, but was in IDLE state before voting. This is a bug";
            return false;
        }
    } else {
        info.connection_state = VOTE_NO_NEW_PEERS_ESTABLISH_P2P_CONNECTIONS;
        if (auto [_, inserted] = votes_establish_p2p_connections.insert(info.client_uuid); !inserted) {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " found in votes_establish_p2p_connections set, but was in IDLE state before voting. This is a "
                    "bug";
            return false;
        }
    }
    return true;
}

bool ccoip::CCoIPMasterState::voteQueryWaitingPeersPending(const ccoip_uuid_t &peer_uuid) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot mark client " << uuid_to_string(peer_uuid) << " as waiting for peers";
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot be marked as waiting for other peers
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot be marked as waiting for other peers in phase " << info.connection_phase;
        return false;
    }

    // if the client has already voted to query whether peers are pending, it cannot vote again
    if (info.vote_query_pending_peers == VOTE_QUERY_PENDING_PEERS) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " is already marked as waiting for other peers";
        return false;
    }
    info.vote_query_pending_peers = VOTE_QUERY_PENDING_PEERS;
    return true;
}


bool ccoip::CCoIPMasterState::hasPendingPeers() {
    for (const auto &[_, info]: client_info) {
        if (info.connection_phase == PEER_REGISTERED) {
            return true;
        }
    }
    return false;
}

void ccoip::CCoIPMasterState::resetVoteQueryPendingPeers() {
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.vote_query_pending_peers == NO_VOTE_INITIATED_QUERY_PENDING_PEERS) {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " found to be in NO_VOTE_INITIATED_QUERY_PENDING_PEERS state when vote is reset. This is a bug";
        }
        info.vote_query_pending_peers = NO_VOTE_INITIATED_QUERY_PENDING_PEERS;
    }
}

bool ccoip::CCoIPMasterState::voteOptimizeTopology(const ccoip_uuid_t &peer_uuid) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to optimize topology for unregistered client " << uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to optimize the topology
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to optimize topology in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to optimize the topology, the client must be idle or in the OPTIMIZE_TOPOLOGY_FAILED state
    if (info.connection_state != IDLE && info.connection_state != OPTIMIZE_TOPOLOGY_FAILED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to optimize topology in state " << info.connection_state;
        return false;
    }

    // set the client state to vote to optimize the topology
    info.connection_state = VOTE_OPTIMIZE_TOPOLOGY;
    if (auto [_, inserted] = votes_optimize_topology.insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " found in votes_optimize_topology set, but was in IDLE state before voting. This is a bug";
        return false;
    }

    return true;
}

bool ccoip::CCoIPMasterState::voteTopologyOptimizationComplete(const ccoip_uuid_t &peer_uuid) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to optimize topology for unregistered client " << uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to complete the topology optimization phase
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to optimize topology in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to complete the topology optimization phase, the client must be in the OPTIMIZE_TOPOLOGY state
    if (info.connection_state != OPTIMIZE_TOPOLOGY) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to complete topology optimization in state " << info.connection_state;
        return false;
    }

    // set the client state to vote to end the topology optimization phase
    info.connection_state = VOTE_COMPLETE_TOPOLOGY_OPTIMIZATION;
    if (auto [_, inserted] = votes_complete_topology_optimization.insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " found in votes_complete_topology_optimization set, but was in OPTIMIZE_TOPOLOGY state before "
                "voting. This is a bug";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::voteSyncSharedState(const ccoip_uuid_t &peer_uuid) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to sync shared state for unregistered client " << uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to sync shared state
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to sync shared state in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to sync shared state, the client must be idle
    if (info.connection_state != IDLE) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to sync shared state in state " << info.connection_state;
        return false;
    }

    const uint32_t peer_group = info.peer_group;

    // set the client state to vote to sync shared state
    info.connection_state = VOTE_SYNC_SHARED_STATE;
    if (auto [_, inserted] = votes_sync_shared_state[peer_group].insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " found in votes_sync_shared_state set, but was in IDLE state before voting. This is a bug";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::voteDistSharedStateComplete(const ccoip_uuid_t &peer_uuid) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to sync shared state for unregistered client " << uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to complete shared state distribution
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to distribute shared state in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to complete the distribute shared state phase, the client must be in either
    // the DISTRIBUTE_SHARED_STATE or REQUEST_SHARED_STATE state
    if (info.connection_state != DISTRIBUTE_SHARED_STATE && info.connection_state != REQUEST_SHARED_STATE) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to distribute shared state in state " << info.connection_state;
        return false;
    }

    const uint32_t peer_group = info.peer_group;

    // set the client state to vote to end the distribute shared state phase
    info.connection_state = VOTE_COMPLETE_SHARED_STATE_SYNC;
    if (auto [_, inserted] = votes_sync_shared_state_complete[peer_group].insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " found in votes_dist_shared_state_complete set, but was in DISTRIBUTE_SHARED_STATE or "
                "REQUEST_SHARED_STATE state before voting. This is a bug";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::voteCollectiveCommsInitiate(const ccoip_uuid_t &peer_uuid, const uint64_t tag) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to initiate collective communications operation for unregistered client "
                << uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to initiate a collective communications operation
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to initiate collective communications operation in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to initiate a collective comms operation, the client must be idle already in the collective
    // communications running state; see meaning of COLLECTIVE_COMMUNICATIONS_RUNNING state in ccoip_master_state.hpp
    if (info.connection_state != IDLE && info.connection_state != COLLECTIVE_COMMUNICATIONS_RUNNING) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to initiate collective communications operation in state " << info.connection_state;
        return false;
    }

    // to vote to initiate a collective communications operation, the client must not already be in the process of
    // voting to initiate one or in the process of performing a collective communications operation known by the same
    // tag
    auto ccomms_state_it = info.collective_coms_states.find(tag);
    if (ccomms_state_it != info.collective_coms_states.end()) {
        auto &ccomms_state = ccomms_state_it->second;
        if (ccomms_state == PERFORM_COLLECTIVE_COMMS) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " is in the PERFORM_COLLECTIVE_COMMS state for tag " << tag
                    << ". Before voting to initiate a new collective communications operation, the client must "
                    "complete the current one.";
        } else if (ccomms_state == VOTE_INITIATE_COLLECTIVE_COMMS) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " is already voting to initiate a collective communications operation for tag " << tag;
        } else {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " is in an unknown collective communications state " << ccomms_state << " for tag " << tag;
        }
        return false;
    }

    // set the client state to vote to initiate a collective communications operation
    info.connection_state = COLLECTIVE_COMMUNICATIONS_RUNNING;
    info.collective_coms_states[tag] = VOTE_INITIATE_COLLECTIVE_COMMS;

    return true;
}

bool ccoip::CCoIPMasterState::voteCollectiveCommsComplete(const ccoip_uuid_t &peer_uuid, const uint64_t tag) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to complete collective communications operation for unregistered client "
                << uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to complete a collective communications operation
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot vote to complete collective communications operation in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to complete a collective comms operation, the client must be in the process of performing one
    // see meaning of PERFORM_COLLECTIVE_COMMS state in ccoip_master_state.hpp
    auto ccomms_state_it = info.collective_coms_states.find(tag);
    if (ccomms_state_it == info.collective_coms_states.end()) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " is not in the process of performing a collective communications operation for tag " << tag
                << ". Before voting to complete a collective communications operation, the client must vote to "
                "initiate one.";
        return false;
    }
    auto &ccomms_state = ccomms_state_it->second;
    if (ccomms_state != PERFORM_COLLECTIVE_COMMS) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) << " is in the " << ccomms_state
                << " state for tag " << tag
                << ". Before voting to complete a collective communications operation, the client must be in the "
                "PERFORM_COLLECTIVE_COMMS state.";
        return false;
    }

    // set the client state to vote to complete a collective communications operation
    ccomms_state = VOTE_COMPLETE_COLLECTIVE_COMMS;
    return true;
}

bool ccoip::CCoIPMasterState::markP2PConnectionsEstablished(const ccoip_uuid_t &client_uuid, const bool success,
                                                            const std::vector<ccoip_uuid_t> &failed_peers) {
    const auto info_opt = getClientInfo(client_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to accept new peers for unregistered client " << uuid_to_string(client_uuid);
        return false;
    }
    auto &info = info_opt->get();
    if (info.connection_state != CONNECTING_TO_PEERS) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                << " cannot mark p2p connections established in state " << info.connection_state;
        return false;
    }
    if (!success) {
        // add failed peers to un-reachability map
        {
            auto &unreachable_peers = unreachability_map[client_uuid];
            for (const auto &peer: failed_peers) {
                const auto failed_peer_info_opt = getClientInfo(peer);
                if (!failed_peer_info_opt) {
                    // ignore invalid failed peer entries because that likely just means that the master has
                    // already handled the disconnection and deregistered said peer, making it no longer valid.
                    continue;
                }
                unreachable_peers.insert(peer);
            }
            if (unreachable_peers.size() == client_info.size()) {
                LOG(WARN) << "Peer " << ccoip_sockaddr_to_str(info.socket_address)
                        << " cannot communicate with any other peer! Peer will be kicked.";
                return false; // returning false here results in a kick.
            } {
                const auto topo_opt = buildReachableRingTopology(info.peer_group);
                // if we cannot build a tour anymore, client will be kicked
                if (!topo_opt) {
                    LOG(WARN) << "Peer " << ccoip_sockaddr_to_str(info.socket_address)
                            << " cannot communicate with enough peers such that any ring tour is possible! Peer will "
                            "be kicked.";
                    return false; // returning false here results in a kick.
                }
                if (!setRingTopology(info.peer_group, *topo_opt, false)) {
                    LOG(BUG) << "Failed to set ring topology for peer group " << info.peer_group
                            << " after p2p connections establishment failure; Topology was already optimal. This is a bug";
                }
            }
        }
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) << " failed to establish p2p connections";
    }
    info.connection_state = success ? WAITING_FOR_OTHER_PEERS : CONNECTING_TO_PEERS_FAILED;
    return true;
}

bool ccoip::CCoIPMasterState::transitionToP2PConnectionsEstablishedPhase(const bool failure) {
    for (auto &[_, info]: client_info) {
        if (info.connection_phase == PEER_REGISTERED && info.connection_state == IDLE) {
            // ignore clients that have not made the cut for the current peer acceptance phase
            continue;
        }
        if (failure) {
            if (info.connection_state == WAITING_FOR_OTHER_PEERS) {
                info.connection_state = CONNECTING_TO_PEERS_FAILED;

                // we still accept the new peer into the run despite the fact the p2p connection establishment
                // phase is considered a failure and ought to be re-tried by the clients.
                // However, in order to do so, we must accept the client.
                // It will still be stuck in the CONNECTING_TO_PEERS_FAILED state though (just like all other clients),
                // so no "damage" can be done from this client caused by this state transition.
                // It will be stuck in this state until it successfully establishes all p2p connections needed for operation.
                if (info.connection_phase == PEER_REGISTERED) {
                    info.connection_phase = PEER_ACCEPTED; // update connection phase to PEER_ACCEPTED
                    onPeerAccepted(info);
                }
            } else if (info.connection_state != CONNECTING_TO_PEERS_FAILED) {
                LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) << " in state "
                        << info.connection_state
                        << " but expected WAITING_FOR_OTHER_PEERS or CONNECTING_TO_PEERS_FAILED";
                return false;
            }
        } else {
            if (info.connection_state == WAITING_FOR_OTHER_PEERS) {
                info.connection_state = IDLE;
                if (info.connection_phase == PEER_REGISTERED) {
                    info.connection_phase = PEER_ACCEPTED; // update connection phase to PEER_ACCEPTED
                    onPeerAccepted(info);
                }
            } else {
                LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) << " in state "
                        << info.connection_state << " but expected WAITING_FOR_OTHER_PEERS";
                return false;
            }
        }
    }
    // all clients have established p2p connections
    return true;
}

bool ccoip::CCoIPMasterState::endSharedStateSyncPhase(const uint32_t peer_group) {
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.peer_group != peer_group) {
            continue;
        }
        if (info.connection_state == VOTE_COMPLETE_SHARED_STATE_SYNC) {
            info.connection_state = IDLE;
        } else {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) << " in state "
                    << info.connection_state << " while terminating shared state distribution phase.";
            return false;
        }
    }
    shared_state_mask[peer_group].clear();
    shared_state_mask_candidates[peer_group].clear();
    shared_state_hashes[peer_group].clear();
    shared_state_hash_types[peer_group].clear();
    shared_state_statuses[peer_group].clear();
    shared_state_dirty_keys[peer_group].clear();
    votes_sync_shared_state_complete[peer_group].clear();
    return true;
}


bool ccoip::CCoIPMasterState::endTopologyOptimizationPhase(const bool failed) {
    if (failed) {
        bandwidth_store.printBandwidthStore();
    }
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.connection_state == VOTE_COMPLETE_TOPOLOGY_OPTIMIZATION) {
            if (failed) {
                info.connection_state = OPTIMIZE_TOPOLOGY_FAILED;
            } else {
                info.connection_state = IDLE;
            }
        } else {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) << " in state "
                    << info.connection_state << " while terminating topology optimization phase.";
            return false;
        }
    }
    votes_complete_topology_optimization.clear();
    return true;
}

bool ccoip::CCoIPMasterState::p2pConnectionsEstablishConsensus() const {
    size_t num_connecting_peers = 0;
    size_t num_voting_peers = 0;
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == WAITING_FOR_OTHER_PEERS || info.connection_state == CONNECTING_TO_PEERS_FAILED) {
            num_voting_peers++;
        }
        if (info.connection_state == WAITING_FOR_OTHER_PEERS || info.connection_state == CONNECTING_TO_PEERS ||
            info.connection_state == CONNECTING_TO_PEERS_FAILED) {
            num_connecting_peers++;
        }
    }
    return num_voting_peers == num_connecting_peers;
}

bool ccoip::CCoIPMasterState::syncSharedStateCompleteConsensus(const uint32_t peer_group) {
    size_t voting_clients = 0;
    size_t n_accepted_peers = 0;
    for (const auto &[_, info]: client_info) {
        if (info.peer_group != peer_group) {
            continue;
        }
        if (info.connection_state == VOTE_COMPLETE_SHARED_STATE_SYNC) {
            voting_clients++;
        }
        if (info.connection_phase == PEER_ACCEPTED) {
            n_accepted_peers++;
        }
    }
    if (voting_clients != votes_sync_shared_state_complete[peer_group].size()) {
        LOG(BUG) << "Mismatch in number of clients voting to sync shared state between client_info and "
                "votes_sync_shared_state_complete";
        return false;
    }
    return voting_clients == n_accepted_peers;
}

bool ccoip::CCoIPMasterState::collectiveCommsInitiateConsensus(const uint32_t peer_group, const uint64_t tag) {
    size_t voting_clients = 0;
    size_t n_accepted_peers = 0;
    for (const auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.peer_group != peer_group) {
            continue;
        }
        auto ccomms_state_it = info.collective_coms_states.find(tag);
        if (ccomms_state_it != info.collective_coms_states.end()) {
            if (ccomms_state_it->second == VOTE_INITIATE_COLLECTIVE_COMMS) {
                voting_clients++;
            }
        }
        n_accepted_peers++;
    }
    return voting_clients == n_accepted_peers;
}

bool ccoip::CCoIPMasterState::collectiveCommsCompleteConsensus(const uint32_t peer_group, const uint64_t tag) {
    size_t voting_clients = 0;
    size_t n_accepted_peers = 0;
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.peer_group != peer_group) {
            continue;
        }
        auto ccomms_state_it = info.collective_coms_states.find(tag);
        if (ccomms_state_it != info.collective_coms_states.end()) {
            if (ccomms_state_it->second == VOTE_COMPLETE_COLLECTIVE_COMMS) {
                voting_clients++;
            }
        }
        n_accepted_peers++;
    }
    return voting_clients == n_accepted_peers;
}

bool ccoip::CCoIPMasterState::transitionToPerformCollectiveCommsPhase(const uint32_t peer_group, const uint64_t tag) {
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.peer_group != peer_group) {
            continue;
        }
        if (info.connection_state != COLLECTIVE_COMMUNICATIONS_RUNNING) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " cannot transition to perform collective communications phase in state "
                    << info.connection_state;
            return false;
        }
        auto ccomms_state_it = info.collective_coms_states.find(tag);
        if (ccomms_state_it == info.collective_coms_states.end()) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " not found in collective communications states for tag " << tag;
            return false;
        }
        if (ccomms_state_it->second != VOTE_INITIATE_COLLECTIVE_COMMS) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) << " in state "
                    << ccomms_state_it->second << " but expected VOTE_INITIATE_COLLECTIVE_COMMS";
            return false;
        }
        ccomms_state_it->second = PERFORM_COLLECTIVE_COMMS;
    }
    return true;
}

bool ccoip::CCoIPMasterState::transitionToCollectiveCommsCompletePhase(const uint32_t peer_group, const uint64_t tag) {
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.peer_group != peer_group) {
            continue;
        }
        if (info.connection_state != COLLECTIVE_COMMUNICATIONS_RUNNING) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " cannot transition to perform collective communications phase in state "
                    << info.connection_state;
            return false;
        }
        auto ccomms_state_it = info.collective_coms_states.find(tag);
        if (ccomms_state_it == info.collective_coms_states.end()) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " not found in collective communications states for tag " << tag;
            return false;
        }

        if (ccomms_state_it->second != VOTE_COMPLETE_COLLECTIVE_COMMS) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) << " in state "
                    << ccomms_state_it->second << " but expected VOTE_COMPLETE_COLLECTIVE_COMMS";
            return false;
        }

        LOG(DEBUG) << "Collective communications operation for tag " << tag << " complete for client "
                << ccoip_sockaddr_to_str(info.socket_address);

        info.collective_coms_states.erase(ccomms_state_it);
        if (info.collective_coms_states.empty()) {
            info.connection_state = IDLE;
        }
    }
    collective_comms_op_abort_states[peer_group].erase(tag);
    return true;
}

bool ccoip::CCoIPMasterState::abortCollectiveCommsOperation(const uint32_t peer_group, const uint64_t tag) {
    auto &peer_group_map = collective_comms_op_abort_states[peer_group];
    if (const auto it = peer_group_map.find(tag); it != peer_group_map.end()) {
        if (it->second) {
            return false;
        }
    }
    peer_group_map[tag] = true;
    return true;
}

bool ccoip::CCoIPMasterState::isCollectiveCommsOperationAborted(const uint32_t peer_group, const uint64_t tag) const {
    const auto pg_it = collective_comms_op_abort_states.find(peer_group);
    if (pg_it == collective_comms_op_abort_states.end()) {
        return false;
    }
    const auto &peer_group_map = pg_it->second;
    const auto tag_it = peer_group_map.find(tag);
    if (tag_it == peer_group_map.end()) {
        return false;
    }
    return tag_it->second;
}

bool ccoip::CCoIPMasterState::isCollectiveOperationRunning(const uint32_t peer_group, const uint64_t tag) const {
    // find any peer
    for (const auto &[_, info]: client_info) {
        if (info.peer_group != peer_group) {
            continue;
        }

        const auto tag_it = info.collective_coms_states.find(tag);
        if (tag_it == info.collective_coms_states.end()) {
            continue;
        }

        // if one peer of the peer group is in the running state, we know the operation is still running.
        // only if all have transitioned out of this state, are we ready to conclude the collective comms op
        if (tag_it->second == PERFORM_COLLECTIVE_COMMS) {
            return true;
        }
    }
    return false;
}

bool ccoip::CCoIPMasterState::syncSharedStateConsensus(const uint32_t peer_group) {
    size_t voting_clients = 0;
    size_t n_accepted_peers = 0;
    for (const auto &[_, info]: client_info) {
        if (info.peer_group != peer_group) {
            continue;
        }
        if (info.connection_state == VOTE_SYNC_SHARED_STATE) {
            voting_clients++;
        }
        if (info.connection_phase == PEER_ACCEPTED) {
            n_accepted_peers++;
        }
    }
    if (voting_clients != votes_sync_shared_state[peer_group].size()) {
        LOG(BUG) << "Mismatch in number of clients voting to sync shared state between client_info and "
                "votes_sync_shared_state";
        return false;
    }
    return voting_clients == n_accepted_peers;
}

bool ccoip::CCoIPMasterState::transitionToP2PEstablishmentPhase(const bool accept_new_peers) {
    if (accept_new_peers) {
        // set connection state of all peers to connecting to peers
        for (auto &[_, info]: client_info) {
            // No client, REGISTER or ACCEPTED is exempt from being set to CONNECTING_TO_PEERS here.
            // If you are REGISTERED when we do this transition, good. If you are ACCEPTED, you had to have voted to
            // accept new peers.
            info.connection_state = CONNECTING_TO_PEERS;
        }
        // clear all votes to accept new peers
        votes_accept_new_peers.clear();
    } else {
        // set connection state of all peers to connecting to peers
        for (auto &[_, info]: client_info) {
            // only already accepted peers will enter the CONNECTING_TO_PEERS state, when accept_new_peers is false
            if (info.connection_phase != PEER_ACCEPTED) {
                continue;
            }
            info.connection_state = CONNECTING_TO_PEERS;
        }
        // clear all votes to accept new peers
        votes_establish_p2p_connections.clear();
    }
    return true;
}

bool ccoip::CCoIPMasterState::transitionToTopologyOptimizationPhase() {
    // set connection state of all peers to optimize topology
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            // clients that are not accepted cannot vote to optimize topology, hence they are not considered here.
            continue;
        }
        if (info.connection_state != VOTE_OPTIMIZE_TOPOLOGY) {
            // all clients should be in the VOTE_OPTIMIZE_TOPOLOGY state when transitioning to the OPTIMIZE_TOPOLOGY
            // state
            return false;
        }
        info.connection_state = OPTIMIZE_TOPOLOGY;
    }
    // clear all votes to optimize topology
    votes_optimize_topology.clear();
    return true;
}

std::vector<ccoip::ClientInfo>
ccoip::CCoIPMasterState::getPeersForClient(const ccoip_socket_address_t &client_address,
                                           const ClientInfo &client_info) {
    const auto client_uuid = client_info.client_uuid;

    std::vector<ClientInfo> peers{};

    // for now, simply return all clients except the client itself
    const auto ring_topology = getRingTopology(client_info.peer_group);
    if (ring_topology.empty()) {
        LOG(WARN) << "Ring topology is empty";
        return peers;
    }

    // find the index of the client in the ring topology
    size_t client_index = 0;
    for (size_t i = 0; i < ring_topology.size(); i++) {
        if (ring_topology[i] == client_uuid) {
            client_index = i;
            break;
        }
    }

    if (ring_topology.size() == 1) {
        return peers;
    }

    const size_t previous = client_index == 0 ? ring_topology.size() - 1 : client_index - 1;
    const size_t next = client_index == ring_topology.size() - 1 ? 0 : client_index + 1;

    const ccoip_uuid_t &previous_uuid = ring_topology[previous];
    const ccoip_uuid_t &next_uuid = ring_topology[next];

    // find the client info for the previous and next clients
    const auto previous_info_it = this->client_info.find(previous_uuid);
    if (previous_info_it == this->client_info.end()) {
        LOG(WARN) << "Client " << uuid_to_string(previous_uuid) << " not found";
        return peers;
    }
    const auto next_info_it = this->client_info.find(next_uuid);
    if (next_info_it == this->client_info.end()) {
        LOG(WARN) << "Client " << uuid_to_string(next_uuid) << " not found";
        return peers;
    }
    peers.push_back(previous_info_it->second);
    if (next != previous) {
        // prevent adding the same peer twice
        peers.push_back(next_info_it->second);
    }
    return peers;
}

std::vector<ccoip_socket_address_t> ccoip::CCoIPMasterState::getClientSocketAddresses() {
    std::vector<ccoip_socket_address_t> addresses{};
    addresses.reserve(client_uuids.size());
    for (const auto &[internal_address, _]: client_uuids) {
        addresses.push_back(internal_to_ccoip_sockaddr(internal_address));
    }
    return addresses;
}

std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t>> ccoip::CCoIPMasterState::getClientEntrySet() {
    std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t>> entries{};
    entries.reserve(client_uuids.size());
    for (const auto &[internal_address, uuid]: client_uuids) {
        entries.emplace_back(uuid, internal_to_ccoip_sockaddr(internal_address));
    }
    return entries;
}

bool ccoip::CCoIPMasterState::hasPeerListChanged(const uint32_t callsite) {
    auto &changed = peer_list_changed[callsite];
    if (changed) {
        changed = false;
        return true;
    }
    return false;
}

bool ccoip::CCoIPMasterState::hasTopologyChanged(const uint32_t callsite) {
    auto &changed = topology_changed[callsite];
    if (changed) {
        changed = false;
        return true;
    }
    return false;
}

std::unordered_set<ccoip_uuid_t> ccoip::CCoIPMasterState::getCurrentlyAcceptedPeers() {
    // get current accepted peers
    std::unordered_set<ccoip_uuid_t> current_accepted_peers{};
    for (const auto &[_, info]: client_info) {
        if (info.connection_phase == PEER_ACCEPTED) {
            current_accepted_peers.insert(info.client_uuid);
        }
    }
    return current_accepted_peers;
}


ccoip::CCoIPMasterState::SharedStateMismatchStatus
ccoip::CCoIPMasterState::isNewRevisionLegal(const ccoip_uuid_t &peer_uuid, const uint64_t revision) {
    SharedStateMismatchStatus status = SUCCESSFUL_MATCH;
    const auto info_it = client_info.find(peer_uuid);

    if (info_it == client_info.end()) {
        LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found";
        status = KEY_SET_MISMATCH;
    }
    const uint32_t peer_group = info_it->second.peer_group;
    const uint64_t next_shared_state_revision = this->next_shared_state_revision[peer_group];

    if (next_shared_state_revision != 0) {
        if (revision < next_shared_state_revision) {
            status = REVISION_OUTDATED;
        } else if (revision > next_shared_state_revision) {
            status = REVISION_INCREMENT_VIOLATION;
        }
    } else if (revision > 0) {
        // We allow arbitrary values when the next expected state revision is zero; This is needed for resuming
        this->next_shared_state_revision[peer_group] = revision;
    }

    shared_state_statuses[peer_group][peer_uuid] = status;
    return status;
}

void ccoip::CCoIPMasterState::voteSharedStateMask(const ccoip_uuid_t &peer_uuid,
                                                  const std::vector<SharedStateHashEntry> &entries) {
    const auto info_it = client_info.find(peer_uuid);
    if (info_it == client_info.end()) {
        LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found";
        return;
    }
    const uint32_t peer_group = info_it->second.peer_group;
    shared_state_mask_candidates[peer_group].emplace_back(peer_uuid, entries);
}

class SharedStateHashEntryList {
public:
    std::vector<ccoip::SharedStateHashEntry> entries;

    // Equality operator to compare two SharedStateHashEntryLists
    bool operator==(const SharedStateHashEntryList &other) const { return entries == other.entries; }

    // Custom hash function for SharedStateHashEntryList
    [[nodiscard]] size_t hash() const {
        size_t hash_value = 0;
        for (const auto &entry: entries) {
            hash_value ^= std::hash<ccoip::SharedStateHashEntry>{}(entry) << 1;
        }
        return hash_value;
    }
};

template<>
struct std::hash<SharedStateHashEntryList> {
    size_t operator()(const SharedStateHashEntryList &entry_list) const noexcept { return entry_list.hash(); }
};

struct SharedStateElectionStats {
    size_t num_votes;
    int priority = 0;
};

bool ccoip::CCoIPMasterState::electSharedStateMask(const uint32_t peer_group) {
    // count the number of votes for each distinct candidate
    std::unordered_map<SharedStateHashEntryList, SharedStateElectionStats> election_stats{};
    for (const auto &[uuid, entries]: shared_state_mask_candidates[peer_group]) {
        SharedStateHashEntryList entry_list{.entries = entries};
        auto &stats = election_stats[entry_list];
        stats.num_votes++;

        if (const auto status_opt = getSharedStateMismatchStatus(uuid)) {
            const SharedStateMismatchStatus status = *status_opt;
            if (status == SUCCESSFUL_MATCH) {
                stats.priority = 1;
            } else if (status == REVISION_OUTDATED) {
                stats.priority = -2;
            } else {
                stats.priority = -1;
            }
        }
    }

    int max_priority = INT32_MIN;
    for (const auto &[uuid, stats]: election_stats) {
        max_priority = stats.priority > max_priority ? stats.priority : max_priority;
    }

    // get the candidate with the most votes
    SharedStateHashEntryList winning_candidate{};
    size_t max_votes = 0;
    for (const auto &[candidate, stats]: election_stats) {
        if (stats.priority != max_priority) {
            continue;
        }
        if (stats.num_votes > max_votes) {
            max_votes = stats.num_votes;
            winning_candidate = candidate;
        }
    }

    shared_state_mask[peer_group] = winning_candidate.entries;
    return true;
}

bool ccoip::CCoIPMasterState::checkMaskSharedStateMismatches(const uint32_t peer_group) {
    if (shared_state_mask[peer_group].empty()) {
        LOG(WARN) << "No shared state mask candidates found for peer group " << peer_group;
        return false;
    }

    const auto &mask_entries = shared_state_mask[peer_group];

    // check for shared state mask mismatch
    for (const auto &[uuid, entries]: shared_state_mask_candidates[peer_group]) {
        if (entries.size() != mask_entries.size()) {
            LOG(WARN) << "Shared state mask mismatch for client " << uuid_to_string(uuid) << " in peer group "
                    << peer_group;
        }

        const SharedStateMismatchStatus old_status = shared_state_statuses[peer_group][uuid];
        if (old_status != SUCCESSFUL_MATCH && old_status != REVISION_OUTDATED) {
            continue;
        }

        SharedStateMismatchStatus status = SUCCESSFUL_MATCH;
        for (size_t i = 0; i < mask_entries.size(); i++) {
            const auto &mask_entry = mask_entries[i];
            const auto &entry = entries[i];
            if (mask_entry.key != entry.key) {
                LOG(WARN) << "Shared state mask key mismatch for client " << uuid_to_string(uuid) << " in peer group "
                        << peer_group << " for key " << mask_entry.key;
                status = KEY_SET_MISMATCH;
            }
            if (mask_entry.allow_content_inequality != entry.allow_content_inequality) {
                LOG(WARN) << "Shared state mask allow_content_inequality mismatch for client " << uuid_to_string(uuid)
                        << " in peer group " << peer_group << " for key " << mask_entry.key;
                status = KEY_SET_MISMATCH;
            }
            if (mask_entry.data_type != entry.data_type) {
                LOG(WARN) << "Shared state mask data_type mismatch for client " << uuid_to_string(uuid)
                        << " in peer group " << peer_group << " for key " << mask_entry.key;
                status = KEY_SET_MISMATCH;
            }
            if (mask_entry.hash_type != entry.hash_type) {
                LOG(WARN) << "Shared state mask hash_type mismatch for client " << uuid_to_string(uuid)
                        << " in peer group " << peer_group << " for key " << mask_entry.key;
                status = KEY_SET_MISMATCH;
            }
            if (status != KEY_SET_MISMATCH && mask_entry.hash != entry.hash) {
                LOG(WARN) << "Shared state mask content mismatch for client " << uuid_to_string(uuid)
                        << " in peer group " << peer_group << " for key " << mask_entry.key;
                status = CONTENT_HASH_MISMATCH;

                // we actually need to catch & track all mismatches here for content hashes, so we can't break early
                // when content hash mismatch occurs
                shared_state_hashes[peer_group][mask_entry.key] = mask_entry.hash;
                shared_state_hash_types[peer_group][mask_entry.key] = mask_entry.hash_type;
                shared_state_dirty_keys[peer_group][uuid].push_back(mask_entry.key);
            }

            // no break possible on content hash mismatch
            if (status == KEY_SET_MISMATCH) {
                break;
            }
        }
        if (old_status == SUCCESSFUL_MATCH) {
            shared_state_statuses[peer_group][uuid] = status;
        }
    }

    // identify content hash mismatches
    return true;
}

void ccoip::CCoIPMasterState::storePeerBandwidth(const ccoip_uuid_t from, const ccoip_uuid_t to,
                                                 const double send_bandwidth_mpbs) {
    if (!bandwidth_store.storeBandwidth(from, to, send_bandwidth_mpbs)) {
        LOG(BUG) << "Failed to store bandwidth for client " << uuid_to_string(from) << " to " << uuid_to_string(to)
                << ". This likely means the peer was never registered. This is a bug.";
    }
}

std::optional<double> ccoip::CCoIPMasterState::getPeerBandwidthMbps(const ccoip_uuid_t from, const ccoip_uuid_t to) {
    return bandwidth_store.getBandwidthMbps(from, to);
}

std::vector<ccoip::bandwidth_entry> ccoip::CCoIPMasterState::getMissingBandwidthEntries(const ccoip_uuid_t peer) {
    auto missing_entries = bandwidth_store.getMissingBandwidthEntries(peer);

    // remove bandwidth entries that are considered unreachable
    for (auto it = missing_entries.begin(); it != missing_entries.end();) {
        const auto &[from_peer_uuid, to_peer_uuid] = *it;

        // check if considered unreachable
        if (unreachability_map[from_peer_uuid].contains(to_peer_uuid)) {
            it = missing_entries.erase(it);
        } else {
            ++it;
        }
    }

    return missing_entries;
}

void ccoip::CCoIPMasterState::markBandwidthEntryUnreachable(const bandwidth_entry &bandwidth_entry) {
    unreachability_map[bandwidth_entry.from_peer_uuid].insert(bandwidth_entry.to_peer_uuid);
}


bool ccoip::CCoIPMasterState::isBandwidthStoreFullyPopulated() const {
    return bandwidth_store.isBandwidthStoreFullyPopulated();
}

size_t ccoip::CCoIPMasterState::getNumBandwidthStoreRegisteredPeers() const {
    return bandwidth_store.getNumberOfRegisteredPeers();
}

std::optional<ccoip::CCoIPMasterState::SharedStateMismatchStatus>
ccoip::CCoIPMasterState::getSharedStateMismatchStatus(const ccoip_uuid_t &peer_uuid) {
    const auto info_opt = client_info.find(peer_uuid);
    if (info_opt == client_info.end()) {
        LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found";
        return std::nullopt;
    }
    const auto peer_group = info_opt->second.peer_group;
    const auto group_responses = shared_state_statuses[peer_group];
    if (const auto it = group_responses.find(peer_uuid); it != group_responses.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool ccoip::CCoIPMasterState::transitionToSharedStateSyncPhase(const uint32_t peer_group) {
    for (auto &[_, info]: client_info) {
        if (info.peer_group != peer_group) {
            continue;
        }
        if (info.connection_state == VOTE_SYNC_SHARED_STATE) {
            const auto status_opt = getSharedStateMismatchStatus(info.client_uuid);
            if (!status_opt) [[unlikely]] {
                LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                        << " in state VOTE_SYNC_SHARED_STATE but no shared state mismatch status found";
                return false;
            }
            const auto status = *status_opt;
            if (status == KEY_SET_MISMATCH) {
                LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                        << " is in state KEY_SET_MISMATCH after shared state voting phase ended!";
                return false;
            }
            info.connection_state = status == SUCCESSFUL_MATCH ? DISTRIBUTE_SHARED_STATE : REQUEST_SHARED_STATE;
        } else if (info.connection_phase == PEER_ACCEPTED) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address)
                    << " in phase PEER_ACCEPTED but not in state VOTE_SYNC_SHARED_STATE after shared state voting "
                    "phase ended";
            return false;
        }
    }
    votes_sync_shared_state[peer_group].clear();
    next_shared_state_revision[peer_group]++;
    return true;
}

std::optional<ccoip_uuid_t> ccoip::CCoIPMasterState::findClientUUID(const ccoip_socket_address_t &client_address) {
    if (const auto it = client_uuids.find(ccoip_socket_to_internal(client_address)); it != client_uuids.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool ccoip::CCoIPMasterState::acceptNewPeersConsensus() const {
    std::unordered_set<ccoip_uuid_t> voting_peers{};
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == VOTE_ACCEPT_NEW_PEERS || info.connection_phase == PEER_REGISTERED) {
            // if the client is in the PEER_REGISTERED state, it is not yet accepted
            // by the other peers, so it cannot vote to accept new peers.
            // However, to reach the total of all clients, we need to include it in the count.
            voting_peers.insert(info.client_uuid);
        }
    }
    return voting_peers.size() == client_uuids.size();
}

bool ccoip::CCoIPMasterState::queryPendingPeersConsensus() const {
    if (client_info.empty()) {
        return false;
    }
    for (const auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }

        if (info.vote_query_pending_peers != VOTE_QUERY_PENDING_PEERS) {
            return false;
        }
    }
    return true;
}

bool ccoip::CCoIPMasterState::noAcceptNewPeersEstablishP2PConnectionsConsensus() const {
    std::unordered_set<ccoip_uuid_t> voting_peers{};
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == VOTE_NO_NEW_PEERS_ESTABLISH_P2P_CONNECTIONS ||
            info.connection_phase == PEER_REGISTERED) {
            voting_peers.insert(info.client_uuid);
        }
    }
    return voting_peers.size() == client_uuids.size();
}

bool ccoip::CCoIPMasterState::optimizeTopologyConsensus() const {
    std::unordered_set<ccoip_uuid_t> voting_peers{};
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == VOTE_OPTIMIZE_TOPOLOGY || info.connection_phase == PEER_REGISTERED) {
            // if the client is in the PEER_REGISTERED state, it is not yet accepted
            // by the other peers, so it cannot vote to optimize the topology.
            // However, to reach the total of all clients, we need to include it in the count.
            voting_peers.insert(info.client_uuid);
        }
    }
    return voting_peers.size() == client_uuids.size();
}

bool ccoip::CCoIPMasterState::topologyOptimizationCompleteConsensus() const {
    std::unordered_set<ccoip_uuid_t> voting_peers{};
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == VOTE_COMPLETE_TOPOLOGY_OPTIMIZATION || info.connection_phase == PEER_REGISTERED) {
            // if the client is in the PEER_REGISTERED state, it is not yet accepted
            // by the other peers, so it cannot vote to complete the topology optimization.
            // However, to reach the total of all clients, we need to include it in the count.
            voting_peers.insert(info.client_uuid);
        }
    }
    return voting_peers.size() == client_uuids.size();
}

std::optional<std::reference_wrapper<ccoip::ClientInfo>>
ccoip::CCoIPMasterState::getClientInfo(const ccoip_uuid_t &client_uuid) {
    const auto it = client_info.find(client_uuid);
    if (it == client_info.end()) {
        return std::nullopt;
    }
    return it->second;
}

uint64_t ccoip::CCoIPMasterState::getSharedStateRevision(const uint32_t peer_group) {
    return next_shared_state_revision[peer_group];
}

std::vector<std::string> ccoip::CCoIPMasterState::getOutdatedSharedStateKeys(const ccoip_uuid_t peer_uuid) {
    const auto info_it = client_info.find(peer_uuid);
    if (info_it == client_info.end()) {
        LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found when querying outdated shared state keys";
        return std::vector<std::string>{};
    }
    const uint32_t peer_group = info_it->second.peer_group;
    return shared_state_dirty_keys[peer_group][peer_uuid];
}

uint64_t ccoip::CCoIPMasterState::getSharedStateEntryHash(const uint32_t peer_group, const std::string &key) {
    auto &group_state_hashes = shared_state_hashes[peer_group];
    const auto it = group_state_hashes.find(key);
    if (it == group_state_hashes.end()) {
        return 0;
    }
    return it->second;
}

std::optional<ccoip::ccoip_hash_type_t> ccoip::CCoIPMasterState::getSharedStateEntryHashType(const uint32_t peer_group,
    const std::string &key) {
    auto &group_state_hashes = shared_state_hash_types[peer_group];
    const auto it = group_state_hashes.find(key);
    if (it == group_state_hashes.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::vector<std::string> ccoip::CCoIPMasterState::getSharedStateKeys(const uint32_t peer_group) {
    std::vector<std::string> keys{};
    const auto &group_state_mask = shared_state_mask[peer_group];
    keys.reserve(group_state_mask.size());
    for (const auto &entry: group_state_mask) {
        keys.push_back(entry.key);
    }
    return keys;
}

std::vector<uint64_t> ccoip::CCoIPMasterState::getOngoingCollectiveCommsOpTags(const uint32_t peer_group) {
    std::unordered_set<uint64_t> tags{};
    for (const auto &[_, info]: client_info) {
        if (info.peer_group != peer_group) {
            continue;
        }
        for (const auto &[tag, _]: info.collective_coms_states) {
            tags.insert(tag);
        }
    }
    return std::vector(tags.begin(), tags.end());
}

bool ccoip::CCoIPMasterState::performTopologyOptimization(
    const uint32_t peer_group,
    const bool moonshot,
    std::vector<ccoip_uuid_t> &new_topology, bool &is_optimal, bool &has_improved) {
    if (moonshot) {
        auto topology = getRingTopology(peer_group);

        if (!topology_is_optimal[peer_group]) {
            bool topology_has_improved = false;
            bool topology_is_optimal = false;
            if (!TopologyOptimizer::ImproveTopologyMoonshot(bandwidth_store, topology, topology_is_optimal,
                                                            topology_has_improved)) {
                LOG(WARN) << "Failed to optimize topology";
                return false;
            }
            new_topology = topology;
            is_optimal = topology_is_optimal;
            has_improved = topology_has_improved;
        }
    } else {
        auto topology = getRingTopology(peer_group);
        if (!topology_is_optimal[peer_group]) {
            bool topology_is_optimal = false;
            if (!TopologyOptimizer::OptimizeTopology(bandwidth_store, topology, topology_is_optimal)) {
                LOG(WARN) << "Failed to optimize topology";
                return false;
            }
            new_topology = topology;
            is_optimal = topology_is_optimal;
            has_improved = true;
        } else {
            has_improved = false;
        }
    }
    return true;
}

bool ccoip::CCoIPMasterState::isTopologyOptimal(const uint32_t peer_group) { return topology_is_optimal[peer_group]; }

std::vector<ccoip_uuid_t> ccoip::CCoIPMasterState::buildBasicRingTopology(const uint32_t peer_group) {
    std::vector<ccoip_uuid_t> topology{};
    for (const auto &[peer_uuid, _]: getClientEntrySet()) {
        const auto peer_info_opt = getClientInfo(peer_uuid);
        if (!peer_info_opt) {
            LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();
        if (peer_info.connection_phase != PEER_ACCEPTED && peer_info.connection_state != CONNECTING_TO_PEERS) {
            // if the peer is not accepted and is not in the process of connecting to peers, ignore it
            // a newly joined peer will be in the REGISTERED state, but will be part of the new topology.
            // To make sure it is included in the ring topology, there is an exception to the PEER_ACCEPTED rule:
            // To move from REGISTERED to ACCEPTED, the peer must first establish p2p connections with the other peers.
            // In order for the new peer to be included on the list of connections, it must be part of the ring
            // topology, hence the CONNECTING_TO_PEERS state also allows the peer to be included in the ring topology.
            // It doesn't make sense to include other states after CONNECTING_TO_PEERS, as this peer will move to the
            // REGISTERED state shortly, if p2p connection establishment succeeds.
            continue;
        }
        if (peer_info.peer_group != peer_group) {
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
    return topology;
}

std::optional<std::vector<ccoip_uuid_t>>
ccoip::CCoIPMasterState::buildReachableRingTopology(const uint32_t peer_group) {
    std::vector<ccoip_uuid_t> remaining_peers{};
    for (const auto &[peer_uuid, _]: getClientEntrySet()) {
        const auto peer_info_opt = getClientInfo(peer_uuid);
        if (!peer_info_opt) {
            LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();
        if (peer_info.connection_phase != PEER_ACCEPTED && peer_info.connection_state != CONNECTING_TO_PEERS) {
            // if the peer is not accepted and is not in the process of connecting to peers, ignore it
            // a newly joined peer will be in the REGISTERED state, but will be part of the new topology.
            // To make sure it is included in the ring topology, there is an exception to the PEER_ACCEPTED rule:
            // To move from REGISTERED to ACCEPTED, the peer must first establish p2p connections with the other peers.
            // In order for the new peer to be included on the list of connections, it must be part of the ring
            // topology, hence the CONNECTING_TO_PEERS state also allows the peer to be included in the ring topology.
            // It doesn't make sense to include other states after CONNECTING_TO_PEERS, as this peer will move to the
            // REGISTERED state shortly, if p2p connection establishment succeeds.
            continue;
        }
        if (peer_info.peer_group != peer_group) {
            continue;
        }
        remaining_peers.push_back(peer_uuid);
    }
    // Edge case: if we have 0 or 1 peers, we can trivially return a "ring" or nothing
    if (remaining_peers.size() <= 1) {
        if (remaining_peers.empty()) {
            return std::nullopt; // No peers -> no ring
        }
        // A single peer could be considered a degenerate ring if that makes sense in your context
        return std::vector{remaining_peers.front()};
    }

    // 2. Build adjacency: an unordered_map<uuid, unordered_set<uuid>> for bidirectional reachability
    std::unordered_map<ccoip_uuid_t, std::unordered_set<ccoip_uuid_t>> adjacency;
    adjacency.reserve(remaining_peers.size());

    for (const auto &u: remaining_peers) {
        for (const auto &v: remaining_peers) {
            if (u == v) {
                continue; // Skip self
            }
            // Check if v is NOT in u's unreachable set AND u is NOT in v's unreachable set
            const bool u_reaches_v = unreachability_map[u].contains(v);
            const bool v_reaches_u = unreachability_map[v].contains(u);
            if (u_reaches_v && v_reaches_u) {
                // They can reach each other -> add to adjacency
                adjacency[u].insert(v);
            }
        }
    }

    // 3. Backtracking helper to attempt building a Hamiltonian cycle
    std::vector<ccoip_uuid_t> path;
    path.reserve(remaining_peers.size());

    std::unordered_set<ccoip_uuid_t> visited;
    visited.reserve(remaining_peers.size());

    // Recursive function: returns true if it successfully forms a ring
    std::function<bool(const ccoip_uuid_t &)> backtrack = [&](const ccoip_uuid_t &current) -> bool {
        // If we've visited all peers, check if we can close the cycle
        if (path.size() == remaining_peers.size()) {
            // Check adjacency to the start to form a cycle
            const auto &start = path.front();
            if (adjacency[current].contains(start)) {
                // We found a ring
                return true;
            }
            // Can't close the cycle -> fail this path
            return false;
        }

        // Otherwise, try each neighbor of the current node
        for (const auto &neighbor: adjacency[current]) {
            if (!visited.contains(neighbor)) {
                // Mark neighbor visited, add to path, recurse
                visited.insert(neighbor);
                path.push_back(neighbor);

                if (backtrack(neighbor)) {
                    return true;
                }

                // Backtrack
                visited.erase(neighbor);
                path.pop_back();
            }
        }
        return false; // No valid path from here
    };

    // 4. Try each peer as a potential starting point
    for (const auto &start: remaining_peers) {
        path.clear();
        visited.clear();

        visited.insert(start);
        path.push_back(start);

        if (backtrack(start)) {
            // If successful, `path` now represents a valid ring
            return path;
        }
    }

    // If we tried all starts and never returned a valid ring, no ring is possible
    return std::nullopt;
}

bool ccoip::CCoIPMasterState::setRingTopologyUnsafe(const uint32_t peer_group,
                                                    const std::vector<ccoip_uuid_t> &new_topology,
                                                    const bool optimal) {
    if (topology_is_optimal[peer_group]) {
        const auto &ring_topology = ring_topologies[peer_group];
        if (new_topology.size() != ring_topology.size()) {
            topology_is_optimal[peer_group] = false;
        }
    }
    if (topology_is_optimal[peer_group]) {
        LOG(WARN) << "Update topology called when topology is already optimal!";
        return false;
    }
    ring_topologies[peer_group] = new_topology;
    topology_is_optimal[peer_group] = optimal;

    // mark topology changed
    for (auto &[_, changed]: topology_changed) {
        changed = true;
    }
    return true;
}

static bool isTopologyDirty(const std::vector<ccoip_uuid_t> &ring_topology,
                            const std::unordered_map<ccoip_uuid_t, ccoip::ClientInfo> &client_info,
                            const uint64_t local_world_size) {
    if (ring_topology.size() != local_world_size) {
        return true;
    }
    for (const auto &peer_uuid: ring_topology) {
        if (!client_info.contains(peer_uuid)) {
            return true;
        }
    }
    return false;
}

std::vector<ccoip_uuid_t> ccoip::CCoIPMasterState::getRingTopology(const uint32_t peer_group) {
    std::unique_lock lock{topology_mutex};
    if (const uint64_t local_world_size = getLocalWorldSize(peer_group);
        isTopologyDirty(ring_topologies[peer_group], client_info, local_world_size)) {
        const auto new_topology = buildBasicRingTopology(peer_group);
        if (!setRingTopologyUnsafe(peer_group, new_topology, false)) {
            LOG(BUG) << "Topology was dirty, but failed to update it!";
        }
    }
    return ring_topologies.at(peer_group);
}

std::vector<uint32_t> ccoip::CCoIPMasterState::getExistingPeerGroups() {
    std::vector<uint32_t> peer_groups{};
    peer_groups.reserve(client_info.size());
    for (const auto &[_, info]: client_info) {
        peer_groups.push_back(info.peer_group);
    }
    return peer_groups;
}

bool ccoip::CCoIPMasterState::setRingTopology(const uint32_t peer_group,
                                              const std::vector<ccoip_uuid_t> &new_topology,
                                              const bool optimal) {
    std::unique_lock lock{topology_mutex};
    return setRingTopologyUnsafe(peer_group, new_topology, optimal);
}

uint64_t ccoip::CCoIPMasterState::getGlobalWorldSize() const {
    return client_info.size();
}

uint64_t ccoip::CCoIPMasterState::getLocalWorldSize(const uint32_t peer_group) const {
    uint64_t world_size = 0;
    for (const auto &[_, info]: client_info) {
        if (info.peer_group == peer_group) {
            world_size++;
        }
    }
    return world_size;
}
