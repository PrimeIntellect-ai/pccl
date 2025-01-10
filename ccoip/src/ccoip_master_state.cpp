#include "ccoip_master_state.hpp"

#include <ccoip_packets.hpp>
#include <pccl_log.hpp>

bool ccoip::CCoIPMasterState::registerClient(const ccoip_socket_address_t &client_address,
                                             const CCoIPClientVariablePorts &variable_ports,
                                             const uint32_t peer_group,
                                             const ccoip_uuid_t uuid) {
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
    peer_list_changed = true;

    // if this is the first client, consider it as voting to accept new peers
    if (client_uuids.size() == 1) {
        client_info[uuid].connection_phase = PEER_ACCEPTED; // consider it accepted
        if (!voteAcceptNewPeers(uuid)) [[unlikely]] {
            LOG(WARN) << "Failed to vote to accept new peers for first client " <<
                    ccoip_sockaddr_to_str(client_address);
            return false;
        }
        if (!acceptNewPeersConsensus()) [[unlikely]] {
            LOG(BUG) <<
                    "Inconsistent state: the first and only client voted to accept new peers, but not all clients have voted";
            return false;
        }
        transitionToP2PEstablishmentPhase();
    }
    return true;
}

bool ccoip::CCoIPMasterState::unregisterClient(const ccoip_socket_address_t &client_address) {
    if (!isClientRegistered(client_address)) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not registered";
        return false;
    }
    const auto internal_address = ccoip_socket_to_internal(client_address);
    if (const auto it = client_uuids.find(internal_address); it != client_uuids.end()) {
        if (!uuid_clients.erase(it->second)) {
            LOG(BUG) << "Client with UUID " << uuid_to_string(it->second) <<
                    " not found in uuid->sockaddr mapping. This means bi-directional mapping for client UUIDs is inconsistent";
            return false;
        }
        const auto info_it = client_info.find(it->second);
        if (info_it == client_info.end()) {
            LOG(BUG) << "ClientInfo of client with UUID " << uuid_to_string(it->second) <<
                    " not found in uuid->ClientInfo mapping. This means client info mapping is inconsistent";
            return false;
        }
        const uint32_t peer_group = info_it->second.peer_group;
        client_info.erase(info_it);

        // remove from all voting sets
        votes_accept_new_peers.erase(it->second);
        votes_p2p_connections_established.erase(it->second);
        votes_sync_shared_state[peer_group].erase(it->second);
        votes_sync_shared_state_complete[peer_group].erase(it->second);
        client_uuids.erase(it);

        peer_list_changed = true;
    } else {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
    }
    return true;
}

bool ccoip::CCoIPMasterState::isClientRegistered(const ccoip_socket_address_t &client_address) const {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    return client_uuids.contains(internal_address);
}

bool ccoip::CCoIPMasterState::voteAcceptNewPeers(const ccoip_uuid_t &peer_uuid) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to accept new peers for unregistered client " << uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to accept new peers
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to accept new peers in phase "
                << info.connection_phase;
        return false;
    }
    // in order to vote to accept new peers, the client must be idle
    if (info.connection_state != IDLE) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to accept new peers in state "
                << info.connection_state;
        return false;
    }

    // set the client state to vote to accept new peers
    info.connection_state = VOTE_ACCEPT_NEW_PEERS;
    if (auto [_, inserted] = votes_accept_new_peers.insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " found in votes_accept_new_peers set, but was in IDLE state before voting. This is a bug";
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
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to sync shared state in phase "
                << info.connection_phase;
        return false;
    }

    // in order to vote to sync shared state, the client must be idle
    if (info.connection_state != IDLE) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to sync shared state in state "
                << info.connection_state;
        return false;
    }

    const uint32_t peer_group = info.peer_group;

    // set the client state to vote to sync shared state
    info.connection_state = VOTE_SYNC_SHARED_STATE;
    if (auto [_, inserted] = votes_sync_shared_state[peer_group].insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " found in votes_sync_shared_state set, but was in IDLE state before voting. This is a bug";
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
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to distribute shared state in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to distribute shared state, the client must be in either
    // the DISTRIBUTE_SHARED_STATE or REQUEST_SHARED_STATE state
    if (info.connection_state != DISTRIBUTE_SHARED_STATE && info.connection_state != REQUEST_SHARED_STATE) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to distribute shared state in state " << info.connection_state;
        return false;
    }

    const uint32_t peer_group = info.peer_group;

    // set the client state to vote to distribute shared state
    info.connection_state = VOTE_COMPLETE_SHARED_STATE_SYNC;
    if (auto [_, inserted] = votes_sync_shared_state_complete[peer_group].insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " found in votes_dist_shared_state_complete set, but was in DISTRIBUTE_SHARED_STATE or REQUEST_SHARED_STATE state before voting. This is a bug";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::voteCollectiveCommsInitiate(const ccoip_uuid_t &peer_uuid,
                                                          const uint64_t tag) {
    const auto info_opt = getClientInfo(peer_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to initiate collective communications operation for unregistered client " <<
                uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to initiate a collective communications operation
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to initiate collective communications operation in phase "
                << info.connection_phase;
        return false;
    }

    // in order to vote to initiate a collective comms operation, the client must be idle already in the collective communications running state;
    // see meaning of COLLECTIVE_COMMUNICATIONS_RUNNING state in ccoip_master_state.hpp
    if (info.connection_state != IDLE && info.connection_state != COLLECTIVE_COMMUNICATIONS_RUNNING) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to initiate collective communications operation in state "
                << info.connection_state;
        return false;
    }

    // to vote to initiate a collective communications operation, the client must not already be in the process of voting to initiate one
    // or in the process of performing a collective communications operation known by the same tag
    auto ccomms_state_it = info.collective_coms_states.find(tag);
    if (ccomms_state_it != info.collective_coms_states.end()) {
        auto &ccomms_state = ccomms_state_it->second;
        if (ccomms_state == PERFORM_COLLECTIVE_COMMS) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " is in the PERFORM_COLLECTIVE_COMMS state for tag " << tag <<
                    ". Before voting to initiate a new collective communications operation, the client must complete the current one.";
        } else if (ccomms_state == VOTE_INITIATE_COLLECTIVE_COMMS) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " is already voting to initiate a collective communications operation for tag " << tag;
        } else {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " is in an unknown collective communications state " << ccomms_state <<
                    " for tag " << tag;
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
        LOG(WARN) << "Cannot vote to complete collective communications operation for unregistered client " <<
                uuid_to_string(peer_uuid);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to complete a collective communications operation
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot vote to complete collective communications operation in phase "
                << info.connection_phase;
        return false;
    }

    // in order to vote to complete a collective comms operation, the client must be in the process of performing one
    // see meaning of PERFORM_COLLECTIVE_COMMS state in ccoip_master_state.hpp
    auto ccomms_state_it = info.collective_coms_states.find(tag);
    if (ccomms_state_it == info.collective_coms_states.end()) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " is not in the process of performing a collective communications operation for tag " << tag <<
                ". Before voting to complete a collective communications operation, the client must vote to initiate one.";
        return false;
    }
    auto &ccomms_state = ccomms_state_it->second;
    if (ccomms_state != PERFORM_COLLECTIVE_COMMS) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " is in the " << ccomms_state << " state for tag " << tag <<
                ". Before voting to complete a collective communications operation, the client must be in the PERFORM_COLLECTIVE_COMMS state.";
        return false;
    }

    // set the client state to vote to complete a collective communications operation
    ccomms_state = VOTE_COMPLETE_COLLECTIVE_COMMS;
    return true;
}

bool ccoip::CCoIPMasterState::markP2PConnectionsEstablished(const ccoip_uuid_t &client_uuid) {
    const auto info_opt = getClientInfo(client_uuid);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to accept new peers for unregistered client " << uuid_to_string(client_uuid);
        return false;
    }
    auto &info = info_opt->get();
    if (info.connection_state != CONNECTING_TO_PEERS) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " cannot mark p2p connections established in state " << info.connection_state;
        return false;
    }
    info.connection_state = WAITING_FOR_OTHER_PEERS;
    if (auto [_, inserted] = votes_p2p_connections_established.insert(info.client_uuid); !inserted) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                " found in votes_p2p_connections_established set, but was in CONNECTING_TO_PEERS state before voting";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::transitionToP2PConnectionsEstablishedPhase() {
    for (auto &[_, info]: client_info) {
        if (info.connection_phase == PEER_REGISTERED && info.connection_state == IDLE) {
            // ignore clients that have not made the cut for the current peer acceptance phase
            continue;
        }
        if (info.connection_state == WAITING_FOR_OTHER_PEERS) {
            info.connection_state = IDLE;
            if (info.connection_phase == PEER_REGISTERED) {
                info.connection_phase = PEER_ACCEPTED; // update connection phase to PEER_ACCEPTED
            }
        } else {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " in state " << info.connection_state << " but expected WAITING_FOR_OTHER_PEERS";
            return false;
        }
    }
    // all clients have established p2p connections
    votes_p2p_connections_established.clear();
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
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " in state " << info.connection_state <<
                    " while terminating shared state distribution phase.";
            return false;
        }
    }
    shared_state_mask[peer_group].clear();
    shared_state_mask_candidates[peer_group].clear();
    shared_state_hashes[peer_group].clear();
    shared_state_statuses[peer_group].clear();
    shared_state_dirty_keys[peer_group].clear();
    votes_sync_shared_state_complete[peer_group].clear();
    return true;
}

bool ccoip::CCoIPMasterState::p2pConnectionsEstablishConsensus() const {
    size_t num_connecting_peers = 0;
    std::unordered_set<ccoip_uuid_t> voting_peers{};
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == WAITING_FOR_OTHER_PEERS) {
            voting_peers.insert(info.client_uuid);
        }
        if (info.connection_state == WAITING_FOR_OTHER_PEERS || info.connection_state == CONNECTING_TO_PEERS) {
            num_connecting_peers++;
        }
    }
    return voting_peers.size() == num_connecting_peers;
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
        LOG(BUG) <<
                "Mismatch in number of clients voting to sync shared state between client_info and votes_sync_shared_state_complete";
        return false;
    }
    return voting_clients == n_accepted_peers;
}

bool ccoip::CCoIPMasterState::collectiveCommsInitiateConsensus(const uint32_t peer_group,
                                                               const uint64_t tag) {
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
        /*if (info.collective_coms_failure_states[tag]) {
            LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " has marked the collective communications operation for tag " << tag <<
                    " as failed; Considering the operation complete";
            return true;
        }*/
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

bool ccoip::CCoIPMasterState::transitionToPerformCollectiveCommsPhase(const uint32_t peer_group,
                                                                      const uint64_t tag) {
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.peer_group != peer_group) {
            continue;
        }
        if (info.connection_state != COLLECTIVE_COMMUNICATIONS_RUNNING) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " cannot transition to perform collective communications phase in state " << info.connection_state;
            return false;
        }
        auto ccomms_state_it = info.collective_coms_states.find(tag);
        if (ccomms_state_it == info.collective_coms_states.end()) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " not found in collective communications states for tag " << tag;
            return false;
        }
        if (ccomms_state_it->second != VOTE_INITIATE_COLLECTIVE_COMMS) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " in state " << ccomms_state_it->second <<
                    " but expected VOTE_INITIATE_COLLECTIVE_COMMS";
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
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " cannot transition to perform collective communications phase in state " << info.connection_state;
            return false;
        }
        auto ccomms_state_it = info.collective_coms_states.find(tag);
        if (ccomms_state_it == info.collective_coms_states.end()) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " not found in collective communications states for tag " << tag;
            return false;
        }

        if (ccomms_state_it->second != VOTE_COMPLETE_COLLECTIVE_COMMS) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " in state " << ccomms_state_it->second <<
                    " but expected VOTE_COMPLETE_COLLECTIVE_COMMS";
            return false;
        }

        LOG(DEBUG) << "Collective communications operation for tag " << tag << " complete for client " <<
                ccoip_sockaddr_to_str(info.socket_address);

        info.collective_coms_states.erase(ccomms_state_it);
        if (info.collective_coms_states.empty()) {
            info.connection_state = IDLE;
        }
    }
    collective_comms_op_abort_states[peer_group].clear();
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
        LOG(BUG) <<
                "Mismatch in number of clients voting to sync shared state between client_info and votes_sync_shared_state";
        return false;
    }
    return voting_clients == n_accepted_peers;
}

void ccoip::CCoIPMasterState::transitionToP2PEstablishmentPhase() {
    // set connection state of all peers to connecting to peers
    for (auto &[_, info]: client_info) {
        info.connection_state = CONNECTING_TO_PEERS;
    }
    // clear all votes to accept new peers
    votes_accept_new_peers.clear();
}

std::vector<ccoip::ClientInfo>
ccoip::CCoIPMasterState::getPeersForClient(const ccoip_socket_address_t &client_address) const {
    // get the UUID of the client
    const auto client_uuid = client_uuids.at(ccoip_socket_to_internal(client_address));

    std::vector<ClientInfo> peers{};

    // for now, simply return all clients except the client itself
    // TODO: in the future, this should be based on the topology
    for (const auto &[uuid, info]: client_info) {
        if (uuid != client_uuid) {
            peers.push_back(info);
        }
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

std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t> > ccoip::CCoIPMasterState::getClientEntrySet() {
    std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t> > entries{};
    entries.reserve(client_uuids.size());
    for (const auto &[internal_address, uuid]: client_uuids) {
        entries.emplace_back(uuid, internal_to_ccoip_sockaddr(internal_address));
    }
    return entries;
}

bool ccoip::CCoIPMasterState::hasPeerListChanged() {
    const bool changed = peer_list_changed;
    peer_list_changed = false;
    return changed;
}

ccoip::CCoIPMasterState::SharedStateMismatchStatus ccoip::CCoIPMasterState::isNewRevisionLegal(
    const ccoip_uuid_t &peer_uuid,
    const uint64_t revision
) {
    SharedStateMismatchStatus status = SUCCESSFUL_MATCH;
    const auto info_it = client_info.find(peer_uuid);

    if (info_it == client_info.end()) {
        LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found";
        status = KEY_SET_MISMATCH;
    }
    const uint32_t peer_group = info_it->second.peer_group;
    if (const uint64_t next_shared_state_revision = this->next_shared_state_revision[peer_group];
        revision < next_shared_state_revision) {
        status = REVISION_OUTDATED;
    } else if (revision > next_shared_state_revision) {
        status = REVISION_INCREMENT_VIOLATION;
    }
    shared_state_statuses[peer_group][peer_uuid] = status;
    return status;
}

ccoip::CCoIPMasterState::SharedStateMismatchStatus ccoip::CCoIPMasterState::sharedStateMatches(
    const ccoip_uuid_t &peer_uuid, const std::vector<SharedStateHashEntry> &entries
) {
    SharedStateMismatchStatus status = SUCCESSFUL_MATCH;
    const auto info_it = client_info.find(peer_uuid);

    if (info_it == client_info.end()) {
        LOG(WARN) << "Client " << uuid_to_string(peer_uuid) << " not found";
        status = KEY_SET_MISMATCH;
    }

    const uint32_t peer_group = info_it->second.peer_group;

    // compare the shared state mask with the supplied shared state entries
    {
        const auto &group_shared_state_mask = shared_state_mask[peer_group];

        if (const size_t state_mask_size = group_shared_state_mask.size(); state_mask_size != entries.size()) {
            status = KEY_SET_MISMATCH;
            goto end;
        }

        std::vector<std::string> dirty_content_keys{};
        for (const auto &entry: entries) {
            // find matching entry with same key
            const auto mask_entry_it = std::ranges::find_if(group_shared_state_mask,
                                                            [&entry](const SharedStateHashEntry &mask_entry) {
                                                                return mask_entry.key == entry.key;
                                                            });

            if (mask_entry_it == group_shared_state_mask.end()) {
                status = KEY_SET_MISMATCH;
                goto end;
            }

            const auto &mask_entry = *mask_entry_it;
            if (mask_entry.hash != entry.hash) {
                status = CONTENT_HASH_MISMATCH;
                dirty_content_keys.push_back(mask_entry.key);
            }
            if (mask_entry.allow_content_inequality != entry.allow_content_inequality) {
                status = KEY_SET_MISMATCH;
                goto end;
            }
            if (mask_entry.data_type != entry.data_type) {
                status = KEY_SET_MISMATCH;
                goto end;
            }
            if (mask_entry.num_elements != entry.num_elements) {
                status = KEY_SET_MISMATCH;
                goto end;
            }
        }
        if (!dirty_content_keys.empty()) {
            shared_state_dirty_keys[peer_group][peer_uuid] = dirty_content_keys;
        }
    }
end:
    shared_state_statuses[peer_group][peer_uuid] = status;
    return status;
}

void ccoip::CCoIPMasterState::voteSharedStateMask(
    const ccoip_uuid_t &peer_uuid,
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
    bool operator==(const SharedStateHashEntryList &other) const {
        return entries == other.entries;
    }

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
    size_t operator()(const SharedStateHashEntryList &entry_list) const noexcept {
        return entry_list.hash();
    }
};

bool ccoip::CCoIPMasterState::electSharedStateMask(const uint32_t peer_group) {
    // count the number of votes for each distinct candidate
    std::unordered_map<SharedStateHashEntryList, size_t> num_votes{};
    for (const auto &[uuid, entries]: shared_state_mask_candidates[peer_group]) {
        SharedStateHashEntryList entry_list{.entries = entries};
        num_votes[entry_list]++;
    }

    // get the candidate with the most votes
    SharedStateHashEntryList winning_candidate{};
    size_t max_votes = 0;
    for (const auto &[candidate, votes]: num_votes) {
        if (votes > max_votes) {
            max_votes = votes;
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
            LOG(WARN) << "Shared state mask mismatch for client " << uuid_to_string(uuid) <<
                    " in peer group " << peer_group;
        }

        SharedStateMismatchStatus status = SUCCESSFUL_MATCH;
        for (size_t i = 0; i < mask_entries.size(); i++) {
            const auto &mask_entry = mask_entries[i];
            const auto &entry = entries[i];
            if (mask_entry.key != entry.key) {
                LOG(WARN) << "Shared state mask key mismatch for client " << uuid_to_string(uuid) <<
                        " in peer group " << peer_group << " for key " << mask_entry.key;
                status = KEY_SET_MISMATCH;
            }
            if (mask_entry.allow_content_inequality != entry.allow_content_inequality) {
                LOG(WARN) << "Shared state mask allow_content_inequality mismatch for client " << uuid_to_string(uuid)
                        <<
                        " in peer group " << peer_group << " for key " << mask_entry.key;
                status = KEY_SET_MISMATCH;
            }
            if (mask_entry.data_type != entry.data_type) {
                LOG(WARN) << "Shared state mask data_type mismatch for client " << uuid_to_string(uuid) <<
                        " in peer group " << peer_group << " for key " << mask_entry.key;
                status = KEY_SET_MISMATCH;
            }
            if (status != KEY_SET_MISMATCH && mask_entry.hash != entry.hash) {
                LOG(WARN) << "Shared state mask content mismatch for client " << uuid_to_string(uuid) <<
                        " in peer group " << peer_group << " for key " << mask_entry.key;
                status = CONTENT_HASH_MISMATCH;

                // we actually need to catch & track all mismatches here for content hashes, so we can't break early when content hash mismatch occurs
                shared_state_hashes[peer_group][mask_entry.key] = mask_entry.hash;
                shared_state_dirty_keys[peer_group][uuid].push_back(mask_entry.key);
            }

            // no break possible on content hash mismatch
            if (status == KEY_SET_MISMATCH) {
                break;
            }
        }
        shared_state_statuses[peer_group][uuid] = status;
    }

    // identify content hash mismatches
    return true;
}


std::optional<ccoip::CCoIPMasterState::SharedStateMismatchStatus> ccoip::CCoIPMasterState::getSharedStateMismatchStatus(
    const ccoip_uuid_t &peer_uuid) {
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
                LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                        " in state VOTE_SYNC_SHARED_STATE but no shared state mismatch status found";
                return false;
            }
            const auto status = *status_opt;
            if (status == KEY_SET_MISMATCH) {
                LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                        " is in state KEY_SET_MISMATCH after shared state voting phase ended!";
                return false;
            }
            info.connection_state = status == SUCCESSFUL_MATCH ? DISTRIBUTE_SHARED_STATE : REQUEST_SHARED_STATE;
        } else if (info.connection_phase == PEER_ACCEPTED) {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(info.socket_address) <<
                    " in phase PEER_ACCEPTED but not in state VOTE_SYNC_SHARED_STATE after shared state voting phase ended";
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

std::optional<std::reference_wrapper<ccoip::ClientInfo> > ccoip::CCoIPMasterState::getClientInfo(
    const ccoip_uuid_t &client_uuid) {
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

std::vector<std::string> ccoip::CCoIPMasterState::getSharedStateKeys(const uint32_t peer_group) {
    std::vector<std::string> keys{};
    const auto &group_state_mask = shared_state_mask[peer_group];
    keys.reserve(group_state_mask.size());
    for (const auto &entry: group_state_mask) {
        keys.push_back(entry.key);
    }
    return keys;
}

std::vector<uint64_t> ccoip::CCoIPMasterState::getOngoingCollectiveComsOpTags(const uint32_t peer_group) {
    std::vector<uint64_t> tags{};
    for (const auto &[_, info]: client_info) {
        if (info.peer_group != peer_group) {
            continue;
        }
        for (const auto &[tag, _]: info.collective_coms_states) {
            tags.push_back(tag);
        }
    }
    return tags;
}
