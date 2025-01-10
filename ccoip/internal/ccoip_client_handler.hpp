#pragma once

#include <ccoip_client.hpp>
#include <ccoip_client_state.hpp>
#include <ccoip_shared_state.hpp>
#include <ccoip_packets.hpp>

#include <tinysockets.hpp>

namespace ccoip {
    class CCoIPClientHandler {
        /// Blocking socket for connection to master node
        tinysockets::QueuedSocket master_socket;

        /// All state of the client is encapsulated in this object
        CCoIPClientState client_state;

        /// Socket listening for p2p connections
        tinysockets::BlockingIOServerSocket p2p_socket;

        /// Socket listening for shared state distribution requests
        tinysockets::ServerSocket shared_state_socket;

        /// Thread ID of the shared state server thread
        std::thread::id shared_state_server_thread_id;

        /// Open p2p connections; Tx connections (we have established this connection to the peer)
        std::unordered_map<ccoip_uuid_t, std::unique_ptr<tinysockets::BlockingIOSocket>> p2p_connections_tx;

        /// Open p2p connections; Rx connections (peer has established this connection to us)
        std::unordered_map<ccoip_uuid_t, std::unique_ptr<tinysockets::BlockingIOSocket>> p2p_connections_rx;

        /// Peer group of the client
        uint32_t peer_group;

        bool interrupted = false;

        bool connected = false;

    public:
        explicit CCoIPClientHandler(const ccoip_socket_address_t &address, uint32_t peer_group);

        [[nodiscard]] bool connect();

        [[nodiscard]] bool acceptNewPeers();

        [[nodiscard]] bool syncSharedState(ccoip_shared_state_t &shared_state,
                                           ccoip_shared_state_sync_info_t &info_out);

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool updateTopology();

        [[nodiscard]] bool join();

        [[nodiscard]] bool isInterrupted() const;

        ~CCoIPClientHandler();

        [[nodiscard]] bool allReduceAsync(const void *sendbuff, void *recvbuff, size_t count,
                                          ccoip_data_type_t datatype,
                                          ccoip_reduce_op_t op, uint64_t tag);

        [[nodiscard]] bool joinAsyncReduce(uint64_t tag);

        [[nodiscard]] bool getAsyncReduceInfo(uint64_t tag, std::optional<ccoip_reduce_info_t> &info_out);

        [[nodiscard]] bool isAnyCollectiveComsOpRunning() const;

        [[nodiscard]] size_t getWorldSize() const;


    private:
        [[nodiscard]] bool establishP2PConnections();

        [[nodiscard]] bool establishP2PConnection(const M2CPacketNewPeerInfo &peer);

        [[nodiscard]] bool closeP2PConnection(const ccoip_uuid_t &uuid, tinysockets::BlockingIOSocket &socket);

        [[nodiscard]] bool closeAllP2PConnections();

        // shared state packet handlers
        void handleSharedStateRequest(const ccoip_socket_address_t &client_address,
                                      const C2SPacketRequestSharedState &packet);

        // shared state server socket callbacks
        void onSharedStateClientRead(const ccoip_socket_address_t &client_address, const std::span<std::uint8_t> &data);
    };
}
