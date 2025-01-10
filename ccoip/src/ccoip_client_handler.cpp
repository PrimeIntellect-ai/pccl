#include "ccoip_client_handler.hpp"
#include "ccoip_types.hpp"
#include <pccl_log.hpp>
#include <ccoip.h>
#include <future>
#include <hash_utils.hpp>

#include "ccoip_inet_utils.hpp"
#include "ccoip_packets.hpp"
#include <thread_guard.hpp>
#include <guard_utils.hpp>
#include <reduce_kernels.hpp>

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
                                                                           peer_group(peer_group) {
}

bool ccoip::CCoIPClientHandler::connect() {
    if (connected) {
        LOG(WARN) << "CCoIPClientHandler::connect() called while already connected";
        return false;
    }

    // start listening on p2p socket
    {
        p2p_socket.setJoinCallback([this](const ccoip_socket_address_t &client_address,
                                          std::unique_ptr<tinysockets::BlockingIOSocket> &socket) {
            const auto hello_packet_opt = socket->receivePacket<P2PPacketHello>();
            if (!hello_packet_opt) {
                LOG(ERR) << "Failed to receive P2PPacketHello from " << ccoip_sockaddr_to_str(client_address);
                if (!socket->closeConnection()) {
                    LOG(ERR) << "Failed to close connection to " << ccoip_sockaddr_to_str(client_address);
                }
                return;
            }
            const auto &hello_packet = hello_packet_opt.value();

            p2p_connections_rx[hello_packet.peer_uuid] = std::move(socket);
            if (!p2p_connections_rx[hello_packet.peer_uuid]->sendPacket<P2PPacketHelloAck>({})) {
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

    if (!establishP2PConnections()) {
        LOG(ERR) << "Failed to establish P2P connections";
        return false;
    }
    connected = true;
    return true;
}

bool ccoip::CCoIPClientHandler::acceptNewPeers() {
    if (!connected) {
        LOG(WARN) <<
                "CCoIPClientHandler::acceptNewPeers() before CCoIPClientHandler::connect() was called. Establish master connection first before performing client actions.";
        return false;
    }

    if (!master_socket.isOpen()) {
        LOG(ERR) <<
                "Failed to sync shared state: Client socket has been closed; This may mean the client was kicked by the master";
        return false;
    }

    if (client_state.isAnyCollectiveComsOpRunning()) {
        return false;
    }

    if (!master_socket.sendPacket<C2MPacketAcceptNewPeers>({})) {
        return false;
    }

    if (!establishP2PConnections()) {
        LOG(ERR) << "Failed to establish P2P connections";
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::syncSharedState(ccoip_shared_state_t &shared_state,
                                                ccoip_shared_state_sync_info_t &info_out) {
    if (!connected) {
        LOG(WARN) <<
                "CCoIPClientHandler::syncSharedState() before CCoIPClientHandler::connect() was called. Establish master connection first before performing client actions.";
        return false;
    }

    if (!master_socket.isOpen()) {
        LOG(ERR) <<
                "Failed to sync shared state: Client socket has been closed; This may mean the client was kicked by the master";
        return false;
    }

    if (client_state.isAnyCollectiveComsOpRunning()) {
        return false;
    }

    // prepare shared state hashes
    std::vector<SharedStateHashEntry> shared_state_hashes{};
    shared_state_hashes.reserve(shared_state.entries.size());
    for (const auto &entry: shared_state.entries) {
        auto &key = entry.key;
        auto &value = entry.value;
        shared_state_hashes.push_back(SharedStateHashEntry{
            .key = key,
            .hash = entry.allow_content_inequality ? 0 : hash_utils::CRC32(value.data(), value.size_bytes()),
            .data_type = entry.data_type,
            .allow_content_inequality = entry.allow_content_inequality
        });
    }

    // inside this block, we are guaranteed in the shared state distribution phase
    {
        // As long as we are inside this block, client_state.isSyncingSharedState() will return true

        // ATTENTION: We create this guard even before we vote to enter the shared state sync state on the server side.
        // So on the client side this phase either means that we have already successfully voted to enter the shared state sync phase
        // or that we would want to be in the shared state sync phase.
        // This is necessary because we check this state of being in the shared state sync phase during shared state distribution.
        // We turn down requests that come in while we are not in the shared state sync phase.
        // However, if we were to start the shared state sync phase AFTER sending the shared state sync vote packet,
        // not all clients receive the confirmation packet from the master after a successful vote at the same time.
        // If you are unlucky, this discrepancy can be so bad that some clients even handle this packet and send a shared
        // state request to a shared state distributor that is not yet aware that the shared state sync phase has started.
        // It will then respond with state SHARED_STATE_NOT_DISTRIBUTED.
        // Because we are basically only using this phase state for this purpose anyway, this is a good solution to
        // avoid yet another vote that would only really solve a conceptual problem, not a problem introduced by data dependencies.
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
            if (response->distributor_address.port == 0 &&
                response->distributor_address.inet.protocol == inetIPv4 &&
                response->distributor_address.inet.ipv4.data[0] == 0 &&
                response->distributor_address.inet.ipv4.data[1] == 0 &&
                response->distributor_address.inet.ipv4.data[2] == 0 &&
                response->distributor_address.inet.ipv4.data[3] == 0
            ) {
                LOG(ERR) << "Failed to sync shared state: Shared state distributor address is empty";
                return false;
            }

            // if shared state is outdated, request shared state from master
            tinysockets::BlockingIOSocket req_socket(response->distributor_address);
            if (!req_socket.establishConnection()) {
                LOG(ERR) <<
                        "Failed to sync shared state: Failed to establish connection with shared state distributor: " <<
                        ccoip_sockaddr_to_str(response->distributor_address);
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
                LOG(ERR) <<
                        "Failed to sync shared state: Failed to receive S2CPacketSharedStateResponse from distributor";
                return false;
            }
            if (shared_state_response->status != SUCCESS) {
                LOG(ERR) << "Failed to sync shared state: Shared state distributor returned status " <<
                        shared_state_response->status;
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
                    if (new_entry->dst_size != dst_entry.value.size_bytes()) {
                        LOG(ERR) << "Failed to sync shared state: Shared state entry size mismatch for key " <<
                                dst_entry.key;
                        return false;
                    }
                    std::memcpy(dst_entry.value.data(), new_entry->dst_buffer.get(), new_entry->dst_size);
                    info_out.rx_bytes += new_entry->dst_size;

                    // check if hash is correct if allow_content_inequality is False
                    if (dst_entry.allow_content_inequality) {
                        continue;
                    }
                    if (i < response->expected_hashes.size()) {
                        uint64_t actual_hash = hash_utils::CRC32(dst_entry.value.data(), dst_entry.value.size_bytes());
                        if (uint64_t expected_hash = response->expected_hashes[i]; actual_hash != expected_hash) {
                            LOG(ERR) << "Shared state distributor transmitted incorrect shared state entry for key " <<
                                    dst_entry.key << ": Expected hash " << expected_hash << " but got " << actual_hash;
                            return false;
                        }
                    } else {
                        LOG(WARN) << "Master did not transmit expected hash for shared state entry " << dst_entry.key <<
                                "; Skipping hash check...";
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
        if (const auto sync_response = master_socket.receivePacket<M2CPacketSyncSharedStateComplete>(); !
            sync_response) {
            LOG(ERR) <<
                    "Failed to sync shared state: Failed to receive M2CPacketSyncSharedStateComplete response from master";
            return false;
        }
        info_out.tx_bytes = client_state.getSharedStateSyncTxBytes();
        client_state.resetSharedStateSyncTxBytes();
    }
    return true;
}

bool ccoip::CCoIPClientHandler::updateTopology() {
    if (!master_socket.sendPacket<C2MPacketGetTopologyRequest>({})) {
        return false;
    }
    const auto response = master_socket.receivePacket<M2CPacketGetTopologyResponse>();
    if (!response) {
        return false;
    }
    client_state.updateTopology(response->ring_reduce_order);
    return true;
}

bool ccoip::CCoIPClientHandler::interrupt() {
    if (interrupted) {
        return false;
    }
    if (!p2p_socket.interrupt()) [[unlikely]] {
        return false;
    }
    if (!shared_state_socket.interrupt()) [[unlikely]] {
        return false;
    }
    if (!master_socket.interrupt()) {
        // this can happen, if the connection was closed before, e.g. when kicked. We don't care about this case.
    }
    interrupted = true;
    return true;
}

bool ccoip::CCoIPClientHandler::join() {
    p2p_socket.join();
    shared_state_socket.join();
    master_socket.join();
    return true;
}

bool ccoip::CCoIPClientHandler::isInterrupted() const {
    return interrupted;
}

ccoip::CCoIPClientHandler::~CCoIPClientHandler() = default;

bool ccoip::CCoIPClientHandler::establishP2PConnections() {
    // wait for new peers packet
    const auto new_peers = master_socket.receivePacket<M2CPacketNewPeers>();
    if (!new_peers) {
        LOG(ERR) << "Failed to receive new peers packet";
        return false;
    }
    LOG(DEBUG) << "Received M2CPacketNewPeers from master";
    if (!new_peers->unchanged) {
        LOG(DEBUG) << "New peers list has changed";

        // establish p2p connections
        for (auto &peer: new_peers->new_peers) {
            // check if connection already exists
            if (p2p_connections_tx.contains(peer.peer_uuid)) {
                LOG(DEBUG) << "P2P connection with peer " << uuid_to_string(peer.peer_uuid) << " already exists";
                continue;
            }
            LOG(DEBUG) << "Establishing P2P connection with peer " << uuid_to_string(peer.peer_uuid);
            if (!establishP2PConnection(peer)) {
                LOG(ERR) << "Failed to establish P2P connection with peer " << uuid_to_string(peer.peer_uuid);
                return false;
            }
        }
        // close p2p connections that are no longer needed
        for (auto it = p2p_connections_tx.begin(); it != p2p_connections_tx.end();) {
            if (std::ranges::find_if(new_peers->new_peers,
                                     [&it](const M2CPacketNewPeerInfo &peer) {
                                         return peer.peer_uuid == it->first;
                                     }) == new_peers->new_peers.end()) {
                LOG(DEBUG) << "Closing p2p connection with peer " << uuid_to_string(it->first);
                if (!closeP2PConnection(it->first, *it->second)) {
                    LOG(WARN) << "Failed to close p2p connection with peer " << uuid_to_string(it->first);
                }
                it = p2p_connections_tx.erase(it);
            } else {
                ++it;
            }
        }
    }

    // send packet to this peer has established its p2p connections
    if (!master_socket.sendPacket<C2MPacketP2PConnectionsEstablished>({})) {
        LOG(ERR) << "Failed to send P2P connections established packet";
        return false;
    }

    // wait for response from master, indicating ALL peers have established their
    // respective p2p connections
    if (const auto response = master_socket.receivePacket<M2CPacketP2PConnectionsEstablished>(); !response) {
        LOG(ERR) << "Failed to receive P2P connections established response";
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::establishP2PConnection(const M2CPacketNewPeerInfo &peer) {
    LOG(DEBUG) << "Establishing P2P connection with peer " << uuid_to_string(peer.peer_uuid);
    auto [it, inserted] = p2p_connections_tx.emplace(peer.peer_uuid,
                                                     std::make_unique<tinysockets::BlockingIOSocket>(
                                                         peer.p2p_listen_addr));
    if (!inserted) {
        LOG(ERR) << "P2P connection with peer " << uuid_to_string(peer.peer_uuid) << " already exists";
        return false;
    }
    auto &connection = it->second;
    if (!connection->establishConnection()) {
        LOG(ERR) << "Failed to establish P2P connection with peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    P2PPacketHello hello_packet{};
    hello_packet.peer_uuid = client_state.getAssignedUUID();
    if (!connection->sendPacket<P2PPacketHello>(hello_packet)) {
        LOG(ERR) << "Failed to send hello packet to peer " << uuid_to_string(peer.peer_uuid);
    }
    if (const auto response = connection->receivePacket<P2PPacketHelloAck>(); !response) {
        LOG(ERR) << "Failed to receive hello ack from peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    if (!client_state.registerPeer(peer.p2p_listen_addr, peer.peer_uuid)) {
        LOG(ERR) << "Failed to register peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::closeP2PConnection(const ccoip_uuid_t &uuid, tinysockets::BlockingIOSocket &socket) {
    if (!socket.closeConnection()) [[unlikely]] {
        LOG(BUG) << "Failed to close connection with peer " << uuid_to_string(uuid);
        return false;
    }
    if (!client_state.unregisterPeer(socket.getConnectSockAddr())) [[unlikely]] {
        LOG(BUG) << "Failed to unregister peer " << uuid_to_string(uuid) <<
                ". This means the client was already unregistered; This is a bug!";
        return false;
    }
    return true;
}

void ccoip::CCoIPClientHandler::handleSharedStateRequest(const ccoip_socket_address_t &client_address,
                                                         const C2SPacketRequestSharedState &packet) {
    THREAD_GUARD(shared_state_server_thread_id);

    LOG(DEBUG) << "Received shared state distribution request from " << ccoip_sockaddr_to_str(client_address);

    S2CPacketSharedStateResponse response{};

    // check if we are in shared state sync phase
    if (!client_state.isSyncingSharedState()) {
        LOG(WARN) << "Received shared state request from " << ccoip_sockaddr_to_str(client_address) <<
                " while not in shared state sync phase; Responding with status=SHARED_STATE_NOT_DISTRIBUTED";
        response.status = SHARED_STATE_NOT_DISTRIBUTED;
        goto end;
    }

    // construct packet referencing memory from the current shared state
    // for the set of keys requested by the client
    {
        response.status = SUCCESS;
        const auto &shared_state = client_state.getCurrentSharedState();
        response.revision = shared_state.revision;
        for (const auto &requested_key: packet.requested_keys) {
            const auto it = std::ranges::find_if(shared_state.entries,
                                                 [&requested_key](const ccoip_shared_state_entry_t &entry) {
                                                     return entry.key == requested_key;
                                                 });
            if (it == shared_state.entries.end()) {
                response.status = UNKNOWN_SHARED_STATE_KEY;
                goto end;
            }
            response.entries.push_back(SharedStateEntry{
                .key = requested_key,
                .src_buffer = std::span(reinterpret_cast<uint8_t *>(it->value.data()), it->value.size_bytes())
            });
            client_state.trackSharedStateTxBytes(it->value.size_bytes());
        }
    }
    LOG(DEBUG) << "Distributing shared state revision " << response.revision;
end:
    if (!shared_state_socket.sendPacket(client_address, response)) {
        LOG(ERR) << "Failed to send shared state response to " << ccoip_sockaddr_to_str(client_address);
    }
}

void ccoip::CCoIPClientHandler::onSharedStateClientRead(const ccoip_socket_address_t &client_address,
                                                        const std::span<std::uint8_t> &data) {
    THREAD_GUARD(shared_state_server_thread_id);
    PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
    if (const auto packet_type = buffer.read<uint16_t>();
        packet_type == C2SPacketRequestSharedState::packet_id) {
        C2SPacketRequestSharedState packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2SPacketRequestSharedState from " <<
                    ccoip_sockaddr_to_str(client_address);
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
                                               const ccoip_data_type_t datatype, const ccoip_reduce_op_t op,
                                               const uint64_t tag) {
    if (client_state.isCollectiveComsOpRunning(tag)) {
        // can't start a new collective coms op while one is already running
        return false;
    }

    if (!client_state.launchAsyncCollectiveOp(
        tag, [this, sendbuff, recvbuff, count, datatype, op, tag](std::promise<bool> &promise) {
            LOG(DEBUG) << "Vote to commence all reduce operation with tag " << tag;

            bool abort_packet_received = false;
            bool aborted = false;

            const auto receiveReduceTerm = [this, tag, &abort_packet_received, &aborted]
            (tinysockets::BlockingIOSocket &p2p_rx_socket) -> std::optional<P2PPacketReduceTerm> {
                while (true) {
                    if (!p2p_rx_socket.isOpen()) {
                        return std::nullopt;
                    }
                    if (auto term_packet = p2p_rx_socket.receivePacket<P2PPacketReduceTerm>(true)) {
                        return term_packet;
                    }
                    auto abort_packet = master_socket.receiveMatchingPacket<M2CPacketCollectiveCommsAbort>(
                        [tag](const M2CPacketCollectiveCommsAbort &packet) {
                            return packet.tag == tag;
                        }, true);
                    if (abort_packet) {
                        abort_packet_received = true;
                        aborted = abort_packet->aborted;
                        return std::nullopt;
                    }
                    std::this_thread::yield();
                }
            };
            const auto reduce_fun = [&] {
                // vote commence collective comms operation and await consensus
                {
                    C2MPacketCollectiveCommsInitiate initiate_packet{};
                    initiate_packet.tag = tag;
                    initiate_packet.count = count;
                    initiate_packet.data_type = datatype;
                    initiate_packet.op = op;
                    if (!master_socket.sendPacket<C2MPacketCollectiveCommsInitiate>(initiate_packet)) {
                        return false;
                    }

                    // As long as we do not expect packets of types here that the main thread also expects and claims for normal operation
                    // (such as commencing other concurrent collective comms operations), we can safely use receiveMatchingPacket packet here
                    // to receive packets from the master socket.
                    // Given that M2CPacketCollectiveCommsCommence is only expected in this context, we will never "steal" anyone else's packet
                    // nor will the main thread steal our packet.
                    const auto response = master_socket.receiveMatchingPacket<M2CPacketCollectiveCommsCommence>(
                        [tag](const M2CPacketCollectiveCommsCommence &packet) {
                            return packet.tag == tag;
                        });

                    if (!response) {
                        return false;
                    }
                    LOG(DEBUG) << "Received M2CPacketCollectiveCommsCommence for tag " << tag <<
                            "; Collective communications consensus reached";
                    if (response->require_topology_update) {
                        if (!updateTopology()) {
                            LOG(ERR) <<
                                    "Failed to update topology after collective comms commence indicated dirty topology";
                        }
                    }
                }

                // TODO: THIS IS SUBJECT TO CHANGE AND FOR NOW IS A HARDCODED UNOPTIMIZED NON-PIPELINED RING REDUCE.
                //  THIS ENTIRE SECTION WILL BE DELETED IN THE NEAR FUTURE
                const auto &ring_order = client_state.getRingOrder();
                const auto &client_uuid = client_state.getAssignedUUID();

                // find position in ring order
                const auto it = std::ranges::find(ring_order, client_uuid);
                if (it == ring_order.end()) {
                    return false;
                }
                const auto position = std::distance(ring_order.begin(), it);

                const size_t byte_size = count * ccoip_data_type_size(datatype);

                std::span send_span(static_cast<const std::byte *>(sendbuff), byte_size);
                std::span recv_span(static_cast<std::byte *>(recvbuff), byte_size);
                std::memcpy(recv_span.data(), send_span.data(), send_span.size());

                if (position == 0) {
                    // is first
                    // initiate ring reduce

                    // find next peer
                    const auto next_peer = ring_order[position + 1];
                    const auto next_peer_socket_opt = p2p_connections_tx.find(next_peer);
                    if (next_peer_socket_opt == p2p_connections_tx.end()) {
                        LOG(ERR) << "Failed to find p2p connection for next peer " << uuid_to_string(next_peer);
                        return false;
                    }

                    auto &next_peer_socket = next_peer_socket_opt->second;

                    P2PPacketReduceTerm packet{};
                    packet.tag = tag;
                    packet.is_reduce = true;
                    packet.data = send_span;
                    if (!next_peer_socket->sendPacket<P2PPacketReduceTerm>(packet)) {
                        LOG(ERR) << "Failed to send reduce term to next peer " << uuid_to_string(next_peer);
                        return false;
                    }
                } else {
                    // is not first
                    // wait for data from previous peer
                    const auto prev_peer = ring_order[position - 1];
                    const auto prev_peer_socket_opt = p2p_connections_rx.find(prev_peer);
                    if (prev_peer_socket_opt == p2p_connections_rx.end()) {
                        LOG(ERR) << "Failed to find p2p connection for previous peer " << uuid_to_string(prev_peer);
                        return false;
                    }
                    auto &prev_peer_socket = prev_peer_socket_opt->second;
                    const auto rx_packet = receiveReduceTerm(*prev_peer_socket);
                    if (!rx_packet) {
                        LOG(ERR) << "Failed to receive reduce term from previous peer " << uuid_to_string(prev_peer);
                        return false;
                    }
                    client_state.trackCollectiveComsRxBytes(tag, rx_packet->data.size());

                    // perform reduction
                    performReduction(recv_span, rx_packet->data, datatype, op);

                    // if is not last, send data to next peer
                    if (position < ring_order.size() - 1) {
                        // find next peer
                        const auto next_peer = ring_order[position + 1];
                        const auto next_peer_socket_opt = p2p_connections_tx.find(next_peer);
                        if (next_peer_socket_opt == p2p_connections_tx.end()) {
                            LOG(ERR) << "Failed to find p2p connection for next peer " << uuid_to_string(next_peer);
                            return false;
                        }
                        auto &next_peer_socket = next_peer_socket_opt->second;

                        P2PPacketReduceTerm tx_packet{};
                        tx_packet.tag = tag;
                        tx_packet.is_reduce = true;
                        tx_packet.data = recv_span;
                        if (!next_peer_socket->sendPacket<P2PPacketReduceTerm>(tx_packet)) {
                            LOG(ERR) << "Failed to send reduce term to next peer " << uuid_to_string(next_peer);
                            return false;
                        }
                        client_state.trackCollectiveComsTxBytes(tag, byte_size);
                    }
                }

                if (position == ring_order.size() - 1) {
                    // is last
                    // distribute to other peers
                    for (size_t i = 0; i < ring_order.size() - 1; i++) {
                        const auto peer = ring_order[i];
                        const auto peer_socket_opt = p2p_connections_tx.find(peer);
                        if (peer_socket_opt == p2p_connections_tx.end()) {
                            LOG(ERR) << "Failed to find p2p connection for peer " << uuid_to_string(peer);
                            return false;
                        }
                        auto &peer_socket = peer_socket_opt->second;
                        P2PPacketReduceTerm packet{};
                        packet.tag = tag;
                        packet.is_reduce = false;
                        packet.data = recv_span;
                        if (!peer_socket->sendPacket<P2PPacketReduceTerm>(packet)) {
                            LOG(ERR) << "Failed to send reduce term to peer " << uuid_to_string(peer);
                            return false;
                        }
                        client_state.trackCollectiveComsTxBytes(tag, byte_size);
                    }
                } else {
                    // is not last
                    // wait for final result
                    const auto final_peer = ring_order[ring_order.size() - 1];
                    const auto final_peer_socket_opt = p2p_connections_rx.find(final_peer);
                    if (final_peer_socket_opt == p2p_connections_rx.end()) {
                        LOG(ERR) << "Failed to find p2p connection for final peer " << uuid_to_string(final_peer);
                        return false;
                    }
                    auto &final_peer_socket = final_peer_socket_opt->second;
                    const auto final_rx_packet = receiveReduceTerm(*final_peer_socket);
                    if (!final_rx_packet) {
                        LOG(ERR) << "Failed to receive reduce term from final peer " << uuid_to_string(final_peer);
                        return false;
                    }
                    client_state.trackCollectiveComsRxBytes(tag, byte_size);

                    std::memcpy(recv_span.data(), final_rx_packet->data.data(), final_rx_packet->data.size());
                    // set content to final result
                }
                return true;
            };
            auto success = reduce_fun();
            if (![&] {
                // vote collective comms operation complete and await consensus
                C2MPacketCollectiveCommsComplete complete_packet{};
                complete_packet.tag = tag;
                complete_packet.was_aborted = success == false;

                if (!master_socket.sendPacket<C2MPacketCollectiveCommsComplete>(complete_packet)) {
                    return false;
                }
                LOG(DEBUG) << "Sent C2MPacketCollectiveCommsComplete for tag " << tag;

                if (!abort_packet_received) {
                    const auto abort_response = master_socket.receiveMatchingPacket<M2CPacketCollectiveCommsAbort>(
                        [tag](const M2CPacketCollectiveCommsAbort &packet) {
                            return packet.tag == tag;
                        });
                    if (!abort_response) {
                        return false;
                    }
                    aborted = abort_response->aborted;
                }

                const auto complete_response = master_socket.receiveMatchingPacket<M2CPacketCollectiveCommsComplete>(
                    [tag](const M2CPacketCollectiveCommsComplete &packet) {
                        return packet.tag == tag;
                    });

                if (!complete_response) {
                    return false;
                }
                if (aborted) {
                    // abort=true is considered a failure, where the user must re-issue
                    // the collective comms operation to try again, if so desired.
                    if (!updateTopology()) {
                        LOG(ERR) << "Failed to update topology after collective comms operation was aborted";
                    }
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
    if (!client_state.joinAsyncReduce(tag)) [[unlikely]] {
        return false;
    }
    const auto failure_opt = client_state.hasCollectiveComsOpFailed(tag);
    if (!failure_opt) [[unlikely]] {
        LOG(WARN) << "Collective coms op with tag " << tag << " was either not started or has not yet finished";
        return false;
    }
    if (*failure_opt) {
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::getAsyncReduceInfo(const uint64_t tag,
                                                   std::optional<ccoip_reduce_info_t> &info_out) {
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

bool ccoip::CCoIPClientHandler::isAnyCollectiveComsOpRunning() const {
    return client_state.isAnyCollectiveComsOpRunning();
}

size_t ccoip::CCoIPClientHandler::getWorldSize() const {
    return client_state.getWorldSize();
}
