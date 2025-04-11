#pragma once

#include <ccoip_inet.h>
#include <ccoip_shared_state.hpp>

#include <optional>
#include <thread>

namespace ccoip {
    class CCoIPClientHandler;

    struct ccoip_reduce_info_t {
        uint64_t tag;
        uint32_t world_size;
        uint64_t tx_bytes;
        uint64_t rx_bytes;
    };

    class CCoIPClient {
        CCoIPClientHandler *client;
        std::thread::id main_thread_id;

    public:
        explicit CCoIPClient(const ccoip_socket_address_t &master_socket_address, uint32_t peer_group);

        CCoIPClient(const CCoIPClient &other) = delete;

        CCoIPClient(CCoIPClient &&other) = delete;

        CCoIPClient &operator=(const CCoIPClient &other) = delete;

        CCoIPClient &operator=(CCoIPClient &&other) = delete;

        /// Connect to the master
        [[nodiscard]] bool connect() const;

        /// Accept new peers if necessary and establish p2p connections
        [[nodiscard]] bool acceptNewPeers() const;

        /// Returns true if there are any peers pending to be accepted
        [[nodiscard]] bool arePeersPending(bool &pending_out) const;

        /// Synchronize the shared state
        /// @param shared_state the shared state object referencing user-owned memory to be synchronized
        /// @param info_out is populated with information about this sync operation such as the number of bytes sent/received to facilitate it
        [[nodiscard]] bool syncSharedState(ccoip_shared_state_t &shared_state,
                                           ccoip_shared_state_sync_info_t &info_out) const;

        /// Interrupt the client
        [[nodiscard]] bool interrupt() const;

        /// Performs topology optimization on the current topology
        [[nodiscard]] bool optimizeTopology() const;

        /// Launches an asynchronous all reduce operation
        [[nodiscard]] bool allReduceAsync(const void *sendbuff, void *recvbuff, size_t count,
                                          ccoip_data_type_t datatype, ccoip_data_type_t quantized_data_type,
                                          ccoip_quantization_algorithm_t quantization_algorithm,
                                          ccoip_reduce_op_t op, uint64_t tag) const;

        /// Awaits the completion of an async reduce operation
        [[nodiscard]] bool joinAsyncReduce(uint64_t tag) const;

        /// Gets the reduce info for the async op with the given tag
        /// Outputs std::nullopt if the tag is not found or the operation has not completed;
        /// Note that this function can only be called once for a particular tag; Subsequent calls will return false
        [[nodiscard]] bool getAsyncReduceInfo(uint64_t tag, std::optional<ccoip_reduce_info_t> &info_out) const;

        /// Wait for the client to gracefully terminate after interruption
        [[nodiscard]] bool join() const;

        /// Returns true if the client has been interrupted
        [[nodiscard]] bool isInterrupted() const;

        /// Returns true if there is any collective communications operation running
        [[nodiscard]] bool isAnyCollectiveComsOpRunning() const;

        /// Returns the current local world size; World size here shall mean the number of peers in the peer group that this client is part of;
        /// Includes the client itself
        [[nodiscard]] size_t getLocalWorldSize() const;

        /// Returns the current global world size; World size here shall mean the total number of peers connected across all peer groups;
        /// Includes the client itself
        [[nodiscard]] size_t getGlobalWorldSize() const;

        /// Returns the current number of peer groups defined in the run. A peer group is considered defined once a peer that specifies the particular integer value as its peer group
        /// is accepted into the run by the master.
        [[nodiscard]] size_t getNumDistinctPeerGroups() const;

        /// Returns the current number of peers in the largest peer group defined in the run. A peer group is considered defined once a peer that specifies the particular integer value as its peer group
        /// is accepted into the run by the master.
        /// This function will return the same value on all peers in the run and across all peer groups.
        [[nodiscard]] size_t getLargestPeerGroupWorldSize() const;

        /// Sets the thread id that the client considers to be the main thread.
        /// It will be the only thread that the client will allow to call certain functions.
        void setMainThread(std::thread::id main_thread_id);

        ~CCoIPClient();
    };
};
