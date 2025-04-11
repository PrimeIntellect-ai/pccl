#pragma once

#include <ccoip_inet.h>
#include <ccoip_master_state.hpp>
#include <ccoip_packets.hpp>
#include <tinysockets.hpp>
#include <vector>
#include <thread>
#include <pithreadpool/threadpool.hpp>

namespace ccoip {
    class CCoIPMasterHandler {
        /// Server socket
        tinysockets::ServerSocket server_socket;

        /// Thread ID of the server thread.
        /// Server socket callbacks such as @code onClientRead @endcode and @code onClientDisconnect @endcode
        /// will only ever be invoked from this thread.
        std::thread::id server_thread_id;

        /// Encapsulates the state of the master node
        /// All administrative state and transactions are encapsulated in this object
        CCoIPMasterState server_state;

        /// Indicates (when the master is currently in the midst of establishing p2p connections) the state of whether
        /// peers have left. This is needed to make sure the p2p connection establishment phase is guaranteed to fail
        /// when peers drop because the p2p connection information that has been distributed in the beginning of this
        /// phase does now no longer reflect the new topology taking the leaving peer(s) into account.
        /// New up-to-date p2p connection information must be distributed to reflect the new topology
        /// and this is achieved via the fact that a failed p2p connection establishment phase will result in a re-try
        /// initiated by all clients.
        // This retry is enforced by the master via the CONNECTING_TO_PEERS_FAILED state.
        bool peer_dropped = false;

        /// Thread pool for topology optimization
        pi::threadpool::ThreadPool topology_optimization_threadpool;

        /// Tasks for topology optimization
        std::unordered_map<uint32_t, pi::threadpool::TaskFuture<pi::threadpool::void_t>> topology_optimization_tasks;

    public:
        volatile bool running = false;
        volatile bool interrupted = false;

        explicit CCoIPMasterHandler(const ccoip_socket_address_t &listen_address);

        [[nodiscard]] bool run();

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool join();

        [[nodiscard]] bool kickClient(const ccoip_socket_address_t &client_address) const;

        ~CCoIPMasterHandler();

    private:
        void sendP2PConnectionInformation(bool changed, const ClientInfo &client_info, bool include_registered);

        void sendP2PConnectionInformation(bool include_registered);

        [[nodiscard]] bool checkEstablishP2PConnectionConsensus();

        [[nodiscard]] bool checkTopologyOptimizationConsensus();

        void performTopologyOptimization(uint32_t peer_group);

        [[nodiscard]] bool checkTopologyOptimizationCompletionConsensus();

        [[nodiscard]] bool checkP2PConnectionsEstablished();

        [[nodiscard]] bool checkQueryPeersPendingConsensus();

        [[nodiscard]] bool checkSyncSharedStateConsensus(uint32_t peer_group);

        [[nodiscard]] bool checkSyncSharedStateCompleteConsensus(uint32_t peer_group);

        [[nodiscard]] bool checkCollectiveCommsInitiateConsensus(uint32_t peer_group, uint64_t tag);

        [[nodiscard]] bool checkCollectiveCommsCompleteConsensus(uint32_t peer_group, uint64_t tag);


        /// Finds the optimal peer to distribute the shared state to the specified requester.
        std::optional<ccoip_socket_address_t> findBestSharedStateTxPeer(const ccoip_uuid_t &peer_uuid);

        // packet handling functions
        void handleEstablishP2PConnections(const ccoip_socket_address_t &client_address,
                                  const C2MPacketRequestEstablishP2PConnections &packet);

        void handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                      const C2MPacketRequestSessionRegistration &packet);


        void handleP2PConnectionsEstablished(const ccoip_socket_address_t &client_address,
                                             const C2MPacketP2PConnectionsEstablished &packet);

        void handleCheckPeersPending(const ccoip_socket_address_t & client_address, const C2MPacketCheckPeersPending & packet);


        void handleOptimizeTopology(const ccoip_socket_address_t &client_address,
                                   const C2MPacketOptimizeTopology &packet);

        void handleReportPeerBandwidth(const ccoip_socket_address_t &client_address,
                                      const C2MPacketReportPeerBandwidth &packet);

        void handleOptimizeTopologyWorkComplete(const ccoip_socket_address_t &client_address,
                                               const C2MPacketOptimizeTopologyWorkComplete &packet);

        void handleSyncSharedState(const ccoip_socket_address_t &client_address,
                                   const C2MPacketSyncSharedState &packet);

        void handleSyncSharedStateComplete(const ccoip_socket_address_t &client_address,
                                           const C2MPacketDistSharedStateComplete &packet);

        void handleCollectiveCommsInitiate(const ccoip_socket_address_t &client_address,
                                           const C2MPacketCollectiveCommsInitiate &packet);

        void sendCollectiveCommsAbortPackets(uint32_t peer_group, uint64_t tag, bool aborted);

        void handleCollectiveCommsComplete(const ccoip_socket_address_t &client_address,
                                           const C2MPacketCollectiveCommsComplete &packet);

        // server socket callbacks
        void onClientRead(const ccoip_socket_address_t &client_address,
                          const std::span<uint8_t> &data);

        void onClientDisconnect(const ccoip_socket_address_t &client_address);
    };
}
