#include "ccoip_packets.hpp"

#include <pccl_log.hpp>
#include <random>

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
    buffer.write<uint16_t>(bandwidth_benchmark_listen_port);
    buffer.write<uint32_t>(peer_group);
}

bool ccoip::C2MPacketRequestSessionRegistration::deserialize(PacketReadBuffer &buffer) {
    p2p_listen_port = buffer.read<uint16_t>();
    shared_state_listen_port = buffer.read<uint16_t>();
    bandwidth_benchmark_listen_port = buffer.read<uint16_t>();
    peer_group = buffer.read<uint32_t>();
    return true;
}

// C2MPacketRequestEstablishP2PConnections
ccoip::packetId_t ccoip::C2MPacketRequestEstablishP2PConnections::packet_id =
        C2M_PACKET_REQUEST_ESTABLISH_P2P_CONNECTIONS;

void ccoip::C2MPacketRequestEstablishP2PConnections::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(accept_new_peers);
}

bool ccoip::C2MPacketRequestEstablishP2PConnections::deserialize(PacketReadBuffer &buffer) {
    accept_new_peers = buffer.read<boolean>();
    return true;
}

// C2MPacketCheckPeersPending
ccoip::packetId_t ccoip::C2MPacketCheckPeersPending::packet_id = C2M_PACKET_CHECK_PEERS_PENDING_ID;

// C2MPacketOptimizeTopology
ccoip::packetId_t ccoip::C2MPacketOptimizeTopology::packet_id = C2M_PACKET_OPTIMIZE_TOPOLOGY_ID;

// C2MPacketOptimizeTopologyWorkComplete
ccoip::packetId_t ccoip::C2MPacketOptimizeTopologyWorkComplete::packet_id =
        C2M_PACKET_OPTIMIZE_TOPOLOGY_WORK_COMPLETE_ID;

// C2MPacketP2PConnectionsEstablished
ccoip::packetId_t ccoip::C2MPacketP2PConnectionsEstablished::packet_id = C2M_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID;

void ccoip::C2MPacketP2PConnectionsEstablished::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(success);
    buffer.write<uint64_t>(failed_peers.size());
    for (const auto &[data]: failed_peers) {
        buffer.writeFixedArray(data);
    }
}

bool ccoip::C2MPacketP2PConnectionsEstablished::deserialize(PacketReadBuffer &buffer) {
    success = buffer.read<boolean>();
    const auto length = buffer.read<uint64_t>();
    for (size_t i = 0; i < length; i++) {
        ccoip_uuid_t uuid{};
        uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
        failed_peers.push_back(uuid);
    }
    return true;
}

// C2MReportPeerBandwidth
ccoip::packetId_t ccoip::C2MPacketReportPeerBandwidth::packet_id = C2M_PACKET_REPORT_PEER_BANDWIDTH_ID;

void ccoip::C2MPacketReportPeerBandwidth::serialize(PacketWriteBuffer &buffer) const {
    buffer.writeFixedArray(to_peer_uuid.data);
    buffer.write<double>(bandwidth_mbits_per_second);
}

bool ccoip::C2MPacketReportPeerBandwidth::deserialize(PacketReadBuffer &buffer) {
    to_peer_uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
    bandwidth_mbits_per_second = buffer.read<double>();
    return true;
}

// C2MPacketSyncSharedState
ccoip::packetId_t ccoip::C2MPacketSyncSharedState::packet_id = C2M_PACKET_SYNC_SHARED_STATE_ID;

void ccoip::C2MPacketSyncSharedState::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(shared_state_revision);
    buffer.write<uint64_t>(shared_state_hashes.size());
    for (const auto &entry: shared_state_hashes) {
        buffer.writeString(entry.key);
        buffer.write<uint64_t>(entry.hash);
        buffer.write<uint8_t>(static_cast<uint8_t>(entry.hash_type));
        buffer.write<uint64_t>(entry.num_elements);
        buffer.write<uint8_t>(static_cast<uint8_t>(entry.data_type));
        buffer.write<boolean>(entry.allow_content_inequality);
    }
}

bool ccoip::C2MPacketSyncSharedState::deserialize(PacketReadBuffer &buffer) {
    shared_state_revision = buffer.read<uint64_t>();
    const auto n_entries = buffer.read<uint64_t>();
    for (size_t i = 0; i < n_entries; i++) {
        SharedStateHashEntry entry{};
        entry.key = buffer.readString();
        entry.hash = buffer.read<uint64_t>();
        entry.hash_type = static_cast<ccoip_hash_type_t>(buffer.read<uint8_t>());
        entry.num_elements = buffer.read<uint64_t>();
        entry.data_type = static_cast<ccoip_data_type_t>(buffer.read<uint8_t>());
        entry.allow_content_inequality = buffer.read<boolean>();
        shared_state_hashes.push_back(entry);
    }
    return true;
}

// M2CPacketSyncSharedStateComplete
ccoip::packetId_t ccoip::M2CPacketSyncSharedStateComplete::packet_id = M2C_PACKET_SYNC_SHARED_STATE_COMPLETE_ID;

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
void ccoip::M2CPacketP2PConnectionInfo::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(unchanged);
    buffer.write<uint64_t>(global_world_size);
    buffer.write<uint64_t>(local_world_size);
    buffer.write<uint64_t>(num_distinct_peer_groups);
    buffer.write<uint64_t>(largest_peer_group_world_size);
    if (unchanged) {
        return;
    }
    buffer.write<uint64_t>(all_peers.size());
    for (auto &new_peer: all_peers) {
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

bool ccoip::M2CPacketP2PConnectionInfo::deserialize(PacketReadBuffer &buffer) {
    unchanged = buffer.read<boolean>();
    global_world_size = buffer.read<uint64_t>();
    local_world_size = buffer.read<uint64_t>();
    num_distinct_peer_groups = buffer.read<uint64_t>();
    largest_peer_group_world_size = buffer.read<uint64_t>();
    if (unchanged) {
        return true;
    }
    const auto n_peers = buffer.read<uint64_t>();
    for (size_t i = 0; i < n_peers; i++) {
        PeerInfo peer_info{};
        peer_info.p2p_listen_addr = readSocketAddress(buffer);

        for (unsigned char &byte: peer_info.peer_uuid.data) {
            byte = buffer.read<uint8_t>();
        }
        all_peers.push_back(peer_info);
    }
    return true;
}

// M2CPacketP2PConnectionsEstablished
ccoip::packetId_t ccoip::M2CPacketP2PConnectionsEstablished::packet_id = M2C_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID;

void ccoip::M2CPacketP2PConnectionsEstablished::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(success);
    buffer.write<uint64_t>(ring_reduce_order.size());
    for (const auto &uuid: ring_reduce_order) {
        buffer.writeFixedArray(uuid.data);
    }
}

bool ccoip::M2CPacketP2PConnectionsEstablished::deserialize(PacketReadBuffer &buffer) {
    success = buffer.read<boolean>();
    const auto n_peers = buffer.read<uint64_t>();
    for (size_t i = 0; i < n_peers; i++) {
        ccoip_uuid_t uuid{};
        uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
        ring_reduce_order.push_back(uuid);
    }
    return true;
}

// M2CPacketPeersPending
ccoip::packetId_t ccoip::M2CPacketPeersPendingResponse::packet_id = M2C_PACKET_PEERS_PENDING_RESPONSE_ID;

void ccoip::M2CPacketPeersPendingResponse::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(peers_pending);
}

bool ccoip::M2CPacketPeersPendingResponse::deserialize(PacketReadBuffer &buffer) {
    peers_pending = buffer.read<boolean>();
    return true;
}

// M2CPacketOptimizeTopologyResponse
ccoip::packetId_t ccoip::M2CPacketOptimizeTopologyResponse::packet_id = M2C_PACKET_OPTIMIZE_TOPOLOGY_RESPONSE_ID;

void ccoip::M2CPacketOptimizeTopologyResponse::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(bw_benchmark_requests.size());
    for (const auto &[from_peer_uuid, to_peer_uuid, endpoint_socket_address]: bw_benchmark_requests) {
        buffer.writeFixedArray(from_peer_uuid.data);
        buffer.writeFixedArray(to_peer_uuid.data);
        writeSocketAddress(buffer, endpoint_socket_address);
    }
}

bool ccoip::M2CPacketOptimizeTopologyResponse::deserialize(PacketReadBuffer &buffer) {
    const auto n_requests = buffer.read<uint64_t>();
    for (size_t i = 0; i < n_requests; i++) {
        BenchmarkRequest request{};
        request.from_peer_uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
        request.to_peer_uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
        request.to_peer_benchmark_endpoint = readSocketAddress(buffer);
        bw_benchmark_requests.push_back(request);
    }
    return true;
}

// M2CPacketOptimizeTopologyComplete
ccoip::packetId_t ccoip::M2CPacketOptimizeTopologyComplete::packet_id = M2C_PACKET_OPTIMIZE_TOPOLOGY_COMPLETE_ID;

void ccoip::M2CPacketOptimizeTopologyComplete::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(success);
    buffer.write<uint64_t>(ring_reduce_order.size());
    for (const auto &uuid: ring_reduce_order) {
        buffer.writeFixedArray(uuid.data);
    }
}

bool ccoip::M2CPacketOptimizeTopologyComplete::deserialize(PacketReadBuffer &buffer) {
    success = buffer.read<boolean>();
    const auto n_peers = buffer.read<uint64_t>();
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
    for (const auto &hash_type: expected_hash_types) {
        buffer.write<uint8_t>(hash_type);
    }
}

bool ccoip::M2CPacketSyncSharedState::deserialize(PacketReadBuffer &buffer) {
    is_outdated = buffer.read<boolean>();
    distributor_address = readSocketAddress(buffer);
    const auto n_keys = buffer.read<uint64_t>();
    for (size_t i = 0; i < n_keys; i++) {
        outdated_keys.push_back(buffer.readString());
    }
    for (size_t i = 0; i < n_keys; i++) {
        expected_hashes.push_back(buffer.read<uint64_t>());
    }
    for (size_t i = 0; i < n_keys; i++) {
        expected_hash_types.push_back(static_cast<ccoip_hash_type_t>(buffer.read<uint8_t>()));
    }
    return true;
}

// M2CPacketCollectiveCommsCommence
ccoip::packetId_t ccoip::M2CPacketCollectiveCommsCommence::packet_id = M2C_PACKET_COLLECTIVE_COMMS_COMMENCE_ID;

void ccoip::M2CPacketCollectiveCommsCommence::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(tag);
}

bool ccoip::M2CPacketCollectiveCommsCommence::deserialize(PacketReadBuffer &buffer) {
    tag = buffer.read<uint64_t>();
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

// P2PPacketReduceTerm
ccoip::packetId_t ccoip::P2PPacketDequantizationMeta::packet_id = P2P_PACKET_DEQUANTIZATION_META;

void ccoip::P2PPacketDequantizationMeta::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(tag);

    buffer.write<uint8_t>(static_cast<uint8_t>(dequantization_meta.meta_type));
    switch (dequantization_meta.meta_type) {
        case internal::quantize::DeQuantizationMetaType::MIN_MAX: {
            buffer.write<uint8_t>(static_cast<uint8_t>(dequantization_meta.min_max_data_type));
            for (const auto &val: dequantization_meta.min_value) {
                buffer.write<uint8_t>(val);
            }
            for (const auto &val: dequantization_meta.max_value) {
                buffer.write<uint8_t>(val);
            }
            break;
        }
        case internal::quantize::DeQuantizationMetaType::ZERO_POINT_SCALE: {
            buffer.write<uint8_t>(static_cast<uint8_t>(dequantization_meta.zero_point_type));
            for (const auto &val: dequantization_meta.zero_point) {
                buffer.write<uint8_t>(val);
            }
            buffer.write<uint8_t>(static_cast<uint8_t>(dequantization_meta.scale_type));
            for (const auto &val: dequantization_meta.scale) {
                buffer.write<uint8_t>(val);
            }
            break;
        }
        default: {
            LOG(BUG) << "Unsupported dequantization meta type: " << static_cast<uint8_t>(dequantization_meta.meta_type);
        }
    }
}

bool ccoip::P2PPacketDequantizationMeta::deserialize(PacketReadBuffer &buffer) {
    tag = buffer.read<uint64_t>();
    const auto meta_type = static_cast<internal::quantize::DeQuantizationMetaType>(buffer.read<uint8_t>());
    dequantization_meta.meta_type = meta_type;
    switch (meta_type) {
        case internal::quantize::DeQuantizationMetaType::MIN_MAX: {
            const auto data_type = static_cast<ccoip_data_type_t>(buffer.read<uint8_t>());
            dequantization_meta.min_max_data_type = data_type;
            const size_t n_bytes = ccoip_data_type_size(data_type);

            // this resize is safe because it is not user controlled
            // and thus not arbitrarily large
            dequantization_meta.min_value.resize(n_bytes);
            dequantization_meta.max_value.resize(n_bytes);
            for (size_t i = 0; i < n_bytes; i++) {
                dequantization_meta.min_value[i] = buffer.read<uint8_t>();
            }
            for (size_t i = 0; i < n_bytes; i++) {
                dequantization_meta.max_value[i] = buffer.read<uint8_t>();
            }
            break;
        }
        case internal::quantize::DeQuantizationMetaType::ZERO_POINT_SCALE: {
            const auto zero_point_type = static_cast<ccoip_data_type_t>(buffer.read<uint8_t>());
            dequantization_meta.zero_point_type = zero_point_type;
            const size_t zero_point_n_bytes = ccoip_data_type_size(zero_point_type);

            // this resize is safe because it is not user controlled
            // and thus not arbitrarily large
            dequantization_meta.zero_point.resize(zero_point_n_bytes);
            for (size_t i = 0; i < zero_point_n_bytes; i++) {
                dequantization_meta.zero_point[i] = buffer.read<uint8_t>();
            }

            const auto scale_type = static_cast<ccoip_data_type_t>(buffer.read<uint8_t>());
            dequantization_meta.scale_type = scale_type;
            const size_t scale_n_bytes = ccoip_data_type_size(scale_type);

            // this resize is safe because it is not user controlled
            // and thus not arbitrarily large
            dequantization_meta.scale.resize(scale_n_bytes);
            for (size_t i = 0; i < scale_n_bytes; i++) {
                dequantization_meta.scale[i] = buffer.read<uint8_t>();
            }
            break;
        }
        default: {
            LOG(WARN) << "Cannot deserialize unsupported de-quantization meta type: " << static_cast<uint8_t>(
                meta_type);
            return false;
        }
    }
    return true;
}

size_t ccoip::P2PPacketDequantizationMeta::serializedSize() const {
    size_t size = sizeof(uint64_t) + sizeof(uint8_t);
    size += sizeof(uint32_t) + dequantization_meta.min_value.size();
    size += sizeof(uint32_t) + dequantization_meta.max_value.size();
    return size;
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
    for (const auto &[key, size_bytes]: entries) {
        buffer.writeString(key);
        buffer.write<uint64_t>(size_bytes);
    }
}

bool ccoip::S2CPacketSharedStateResponse::deserialize(PacketReadBuffer &buffer) {
    status = static_cast<SharedStateResponseStatus>(buffer.read<uint8_t>());
    revision = buffer.read<uint64_t>();

    // read entries
    const auto n_entries = buffer.read<uint64_t>();
    for (size_t i = 0; i < n_entries; i++) {
        const std::string key = buffer.readString();
        const auto size_bytes = buffer.read<uint64_t>();
        entries.push_back(SharedStateEntry{
            .key = key,
            .size_bytes = size_bytes
        });
    }
    return true;
}


// M2CPacketP2PConnectionInfo
ccoip::packetId_t ccoip::M2CPacketP2PConnectionInfo::packet_id = M2C_PACKET_P2P_CONNECTION_INFO_ID;

// P2PPacketHelloAck
ccoip::packetId_t ccoip::P2PPacketHelloAck::packet_id = P2P_PACKET_HELLO_ACK_ID;

// B2CPacketBenchmarkServerIsBusy
ccoip::packetId_t ccoip::B2CPacketBenchmarkServerIsBusy::packet_id = B2C_PACKET_BENCHMARK_SERVER_IS_BUSY;

void ccoip::B2CPacketBenchmarkServerIsBusy::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(is_busy);
}

bool ccoip::B2CPacketBenchmarkServerIsBusy::deserialize(PacketReadBuffer &buffer) {
    is_busy = buffer.read<boolean>();
    return true;
}
