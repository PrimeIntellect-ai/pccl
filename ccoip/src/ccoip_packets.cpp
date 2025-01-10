#include "ccoip_packets.hpp"

size_t ccoip::EmptyPacket::serialized_size = 0;

// EmptyPacket
void ccoip::EmptyPacket::serialize(PacketWriteBuffer &buffer) const {
    // do nothing
}

bool ccoip::EmptyPacket::deserialize(PacketReadBuffer &buffer) {
    // do nothing
    return true;
}

// C2MPacketRequestSessionRegistration
ccoip::packetId_t ccoip::C2MPacketRequestSessionRegistration::packet_id = C2M_PACKET_REQUEST_SESSION_REGISTRATION_ID;

void ccoip::C2MPacketRequestSessionRegistration::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint16_t>(p2p_listen_port);
    buffer.write<uint16_t>(shared_state_listen_port);
    buffer.write<uint32_t>(peer_group);
}

bool ccoip::C2MPacketRequestSessionRegistration::deserialize(PacketReadBuffer &buffer) {
    p2p_listen_port = buffer.read<uint16_t>();
    shared_state_listen_port = buffer.read<uint16_t>();
    peer_group = buffer.read<uint32_t>();
    return true;
}

// C2MPacketAcceptNewPeers
ccoip::packetId_t ccoip::C2MPacketAcceptNewPeers::packet_id = C2M_PACKET_ACCEPT_NEW_PEERS_ID;

// C2MPacketP2PConnectionsEstablished
ccoip::packetId_t ccoip::C2MPacketP2PConnectionsEstablished::packet_id = C2M_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID;

// C2MPacketGetTopologyRequest
ccoip::packetId_t ccoip::C2MPacketGetTopologyRequest::packet_id = C2M_PACKET_GET_TOPOLOGY_REQUEST_ID;

// C2MPacketSyncSharedState
ccoip::packetId_t ccoip::C2MPacketSyncSharedState::packet_id = C2M_PACKET_SYNC_SHARED_STATE_ID;

void ccoip::C2MPacketSyncSharedState::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(shared_state_revision);
    buffer.write<uint64_t>(shared_state_hashes.size());
    for (const auto &entry: shared_state_hashes) {
        buffer.writeString(entry.key);
        buffer.write<uint64_t>(entry.hash);
        buffer.write<uint64_t>(entry.num_elements);
        buffer.write<uint8_t>(static_cast<uint8_t>(entry.data_type));
        buffer.write<boolean>(entry.allow_content_inequality);
    }
}

bool ccoip::C2MPacketSyncSharedState::deserialize(PacketReadBuffer &buffer) {
    shared_state_revision = buffer.read<uint64_t>();
    const auto n_entries = buffer.read<uint64_t>();
    shared_state_hashes.reserve(n_entries);
    for (size_t i = 0; i < n_entries; i++) {
        SharedStateHashEntry entry{};
        entry.key = buffer.readString();
        entry.hash = buffer.read<uint64_t>();
        entry.num_elements = buffer.read<uint64_t>();
        entry.data_type = static_cast<ccoip_data_type_t>(buffer.read<uint8_t>());
        entry.allow_content_inequality = buffer.read<boolean>();
        shared_state_hashes.push_back(entry);
    }
    return true;
}

// C2MPacketDistSharedStateComplete
ccoip::packetId_t ccoip::C2MPacketDistSharedStateComplete::packet_id = C2M_PACKET_DIST_SHARED_STATE_COMPLETE_ID;

// M2CPacketSessionRegistrationResponse
ccoip::packetId_t ccoip::M2CPacketSessionRegistrationResponse::packet_id = M2C_PACKET_SESSION_REGISTRATION_RESPONSE_ID;


// C2MPacketCollectiveCommsInitiate
ccoip::packetId_t ccoip::C2MPacketCollectiveCommsInitiate::packet_id = C2M_PACKET_COLLECTIVE_COMMS_INITIATE_ID;

void ccoip::C2MPacketCollectiveCommsInitiate::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(tag);
    buffer.write<uint64_t>(count);
    buffer.write<uint8_t>(static_cast<uint8_t>(data_type));
    buffer.write<uint8_t>(static_cast<uint8_t>(op));
}

bool ccoip::C2MPacketCollectiveCommsInitiate::deserialize(PacketReadBuffer &buffer) {
    tag = buffer.read<uint64_t>();
    count = buffer.read<uint64_t>();
    data_type = static_cast<ccoip_data_type_t>(buffer.read<uint8_t>());
    op = static_cast<ccoip_reduce_op_t>(buffer.read<uint8_t>());
    return true;
}

// C2MPacketCollectiveCommsComplete
ccoip::packetId_t ccoip::C2MPacketCollectiveCommsComplete::packet_id = C2M_PACKET_COLLECTIVE_COMMS_COMPLETE_ID;

void ccoip::C2MPacketCollectiveCommsComplete::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(tag);
    buffer.write<boolean>(was_aborted);
}

bool ccoip::C2MPacketCollectiveCommsComplete::deserialize(PacketReadBuffer &buffer) {
    tag = buffer.read<uint64_t>();
    was_aborted = buffer.read<boolean>();
    return true;
}

void ccoip::M2CPacketSessionRegistrationResponse::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(accepted);
    buffer.writeFixedArray(assigned_uuid.data);
}

bool ccoip::M2CPacketSessionRegistrationResponse::deserialize(PacketReadBuffer &buffer) {
    accepted = buffer.read<boolean>();
    assigned_uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
    return true;
}

static void writeSocketAddress(PacketWriteBuffer &buffer, const ccoip_socket_address_t &socket_address) {
    buffer.write<bool>(socket_address.inet.protocol == inetIPv4);
    if (socket_address.inet.protocol == inetIPv4) {
        std::array<uint8_t, 4> ipv4_data{};
        for (size_t i = 0; i < 4; i++) {
            ipv4_data[i] = socket_address.inet.ipv4.data[i];
        }
        buffer.writeFixedArray(ipv4_data);
    } else if (socket_address.inet.protocol == inetIPv6) {
        std::array<uint8_t, 16> ipv6_data{};
        for (size_t i = 0; i < 16; i++) {
            ipv6_data[i] = socket_address.inet.ipv6.data[i];
        }
        buffer.writeFixedArray(ipv6_data);
    }
    buffer.write<uint16_t>(socket_address.port);
}

// M2CPacketNewPeers
void ccoip::M2CPacketNewPeers::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(unchanged);
    if (unchanged) {
        return;
    }
    buffer.write<uint64_t>(new_peers.size());
    for (auto &new_peer: new_peers) {
        writeSocketAddress(buffer, new_peer.p2p_listen_addr);
        buffer.writeFixedArray(new_peer.peer_uuid.data);
    }
}

static ccoip_socket_address_t readSocketAddress(PacketReadBuffer &buffer) {
    ccoip_socket_address_t address{};
    if (buffer.read<bool>()) {
        address.inet.protocol = inetIPv4;
        for (unsigned char &octet: address.inet.ipv4.data) {
            octet = buffer.read<uint8_t>();
        }
    } else {
        address.inet.protocol = inetIPv6;
        for (unsigned char &octet: address.inet.ipv6.data) {
            octet = buffer.read<uint8_t>();
        }
    }

    const auto port = buffer.read<uint16_t>();
    address.port = port;
    return address;
}

bool ccoip::M2CPacketNewPeers::deserialize(PacketReadBuffer &buffer) {
    unchanged = buffer.read<boolean>();
    if (unchanged) {
        return true;
    }
    const auto n_peers = buffer.read<uint64_t>();
    new_peers.reserve(n_peers);
    for (size_t i = 0; i < n_peers; i++) {
        M2CPacketNewPeerInfo peer_info{};
        peer_info.p2p_listen_addr = readSocketAddress(buffer);

        for (unsigned char &byte: peer_info.peer_uuid.data) {
            byte = buffer.read<uint8_t>();
        }
        new_peers.push_back(peer_info);
    }
    return true;
}

// M2CPacketGetTopologyResponse
ccoip::packetId_t ccoip::M2CPacketGetTopologyResponse::packet_id = M2C_PACKET_GET_TOPOLOGY_RESPONSE_ID;

void ccoip::M2CPacketGetTopologyResponse::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(ring_reduce_order.size());
    for (const auto &uuid: ring_reduce_order) {
        buffer.writeFixedArray(uuid.data);
    }
}

bool ccoip::M2CPacketGetTopologyResponse::deserialize(PacketReadBuffer &buffer) {
    const auto n_peers = buffer.read<uint64_t>();
    ring_reduce_order.reserve(n_peers);
    for (size_t i = 0; i < n_peers; i++) {
        ccoip_uuid_t uuid{};
        uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
        ring_reduce_order.push_back(uuid);
    }
    return true;
}

// M2CPacketSyncSharedState
ccoip::packetId_t ccoip::M2CPacketSyncSharedState::packet_id = M2C_PACKET_SYNC_SHARED_STATE_ID;

void ccoip::M2CPacketSyncSharedState::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(is_outdated);
    writeSocketAddress(buffer, distributor_address);
    buffer.write<uint64_t>(outdated_keys.size());
    for (const auto &key: outdated_keys) {
        buffer.writeString(key);
    }
    for (const auto &hash: expected_hashes) {
        buffer.write<uint64_t>(hash);
    }
}

bool ccoip::M2CPacketSyncSharedState::deserialize(PacketReadBuffer &buffer) {
    is_outdated = buffer.read<boolean>();
    distributor_address = readSocketAddress(buffer);
    const auto n_keys = buffer.read<uint64_t>();
    outdated_keys.reserve(n_keys);
    for (size_t i = 0; i < n_keys; i++) {
        outdated_keys.push_back(buffer.readString());
    }
    expected_hashes.reserve(n_keys);
    for (size_t i = 0; i < n_keys; i++) {
        expected_hashes.push_back(buffer.read<uint64_t>());
    }
    return true;
}

// M2CPacketCollectiveCommsCommence
ccoip::packetId_t ccoip::M2CPacketCollectiveCommsCommence::packet_id = M2C_PACKET_COLLECTIVE_COMMS_COMMENCE_ID;

void ccoip::M2CPacketCollectiveCommsCommence::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(tag);
    buffer.write<boolean>(require_topology_update);
}

bool ccoip::M2CPacketCollectiveCommsCommence::deserialize(PacketReadBuffer &buffer) {
    tag = buffer.read<uint64_t>();
    require_topology_update = buffer.read<boolean>();
    return true;
}

// M2CPacketCollectiveCommsComplete
ccoip::packetId_t ccoip::M2CPacketCollectiveCommsComplete::packet_id = M2C_PACKET_COLLECTIVE_COMMS_COMPLETE_ID;

void ccoip::M2CPacketCollectiveCommsComplete::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(tag);
}

bool ccoip::M2CPacketCollectiveCommsComplete::deserialize(PacketReadBuffer &buffer) {
    tag = buffer.read<uint64_t>();
    return true;
}

// M2CPacketCollectiveCommsAbort
ccoip::packetId_t ccoip::M2CPacketCollectiveCommsAbort::packet_id = M2C_PACKET_COLLECTIVE_COMMS_ABORT_ID;

void ccoip::M2CPacketCollectiveCommsAbort::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(tag);
    buffer.write<boolean>(aborted);
}

bool ccoip::M2CPacketCollectiveCommsAbort::deserialize(PacketReadBuffer &buffer) {
    tag = buffer.read<uint64_t>();
    aborted = buffer.read<boolean>();
    return true;
}

// P2PPacketHello
ccoip::packetId_t ccoip::P2PPacketHello::packet_id = P2P_PACKET_HELLO_ID;

void ccoip::P2PPacketHello::serialize(PacketWriteBuffer &buffer) const {
    buffer.writeFixedArray(peer_uuid.data);
}

bool ccoip::P2PPacketHello::deserialize(PacketReadBuffer &buffer) {
    peer_uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
    return true;
}

// M2CPacketSyncSharedStateComplete
ccoip::packetId_t ccoip::M2CPacketSyncSharedStateComplete::packet_id = M2C_PACKET_SYNC_SHARED_STATE_COMPLETE_ID;

// P2PPacketReduceTerm
ccoip::packetId_t ccoip::P2PPacketReduceTerm::packet_id = P2P_PACKET_REDUCE_TERM_ID;

void ccoip::P2PPacketReduceTerm::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(tag);
    buffer.write<boolean>(is_reduce);
    buffer.write<uint64_t>(data.size_bytes());
    buffer.writeContents(reinterpret_cast<const uint8_t *>(data.data()), data.size_bytes());
}

bool ccoip::P2PPacketReduceTerm::deserialize(PacketReadBuffer &buffer) {
    tag = buffer.read<uint64_t>();
    is_reduce = buffer.read<boolean>();
    const auto length = buffer.read<uint64_t>();
    if (length > buffer.remaining()) {
        return false;
    }
    dst_ptr = std::unique_ptr<std::byte[]>(new std::byte[length]);
    data = std::span(dst_ptr.get(), length);
    buffer.readContents(reinterpret_cast<uint8_t *>(dst_ptr.get()), length);
    return true;
}

// C2SPacketRequestSharedState
void ccoip::C2SPacketRequestSharedState::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(requested_keys.size());
    for (const auto &key: requested_keys) {
        buffer.writeString(key);
    }
}

bool ccoip::C2SPacketRequestSharedState::deserialize(PacketReadBuffer &buffer) {
    const auto n_keys = buffer.read<uint64_t>();
    requested_keys.reserve(n_keys);
    for (size_t i = 0; i < n_keys; i++) {
        requested_keys.push_back(buffer.readString());
    }
    return true;
}

// C2SPacketRequestSharedState
ccoip::packetId_t ccoip::C2SPacketRequestSharedState::packet_id = C2S_PACKET_REQUEST_SHARED_STATE_ID;

// S2CPacketSharedStateResponse
ccoip::packetId_t ccoip::S2CPacketSharedStateResponse::packet_id = S2C_PACKET_SHARED_STATE_RESPONSE_ID;

void ccoip::S2CPacketSharedStateResponse::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint8_t>(static_cast<uint8_t>(status));
    buffer.write<uint64_t>(revision);

    // write entries
    buffer.write<uint64_t>(entries.size());
    for (const auto &entry: entries) {
        buffer.writeString(entry.key);
        buffer.write<uint64_t>(entry.src_buffer.size_bytes());
        buffer.writeContents(entry.src_buffer.data(), entry.src_buffer.size_bytes());
    }
}

bool ccoip::S2CPacketSharedStateResponse::deserialize(PacketReadBuffer &buffer) {
    status = static_cast<SharedStateResponseStatus>(buffer.read<uint8_t>());
    revision = buffer.read<uint64_t>();

    // read entries
    const auto n_entries = buffer.read<uint64_t>();
    entries.reserve(n_entries);
    for (size_t i = 0; i < n_entries; i++) {
        const std::string key = buffer.readString();
        const auto length = buffer.read<uint64_t>();
        if (length > buffer.remaining()) {
            return false;
        }
        auto dst_buffer = std::unique_ptr<uint8_t[]>(new uint8_t[length]);
        buffer.readContents(dst_buffer.get(), length);
        entries.push_back(SharedStateEntry{
            .key = key,
            .dst_buffer = std::move(dst_buffer),
            .dst_size = length
        });
    }
    return true;
}

// M2CPacketNewPeers
ccoip::packetId_t ccoip::M2CPacketNewPeers::packet_id = M2C_PACKET_NEW_PEERS_ID;

// M2CPacketP2PConnectionsEstablished
ccoip::packetId_t ccoip::M2CPacketP2PConnectionsEstablished::packet_id = M2C_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID;

// P2PPacketHelloAck
ccoip::packetId_t ccoip::P2PPacketHelloAck::packet_id = P2P_PACKET_HELLO_ACK_ID;
