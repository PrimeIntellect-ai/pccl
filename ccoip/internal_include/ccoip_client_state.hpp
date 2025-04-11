#pragma once

#include <ccoip_client.hpp>
#include <ccoip_inet.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <ccoip_shared_state.hpp>
#include <ccoip_types.hpp>
#include <future>
#include <optional>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <pithreadpool/threadpool.hpp>

#define DEFAULT_MAX_CONCURRENT_COLLECTIVE_OPS 16

inline int GetMaxConcurrentCollectiveOps() {
    const char *logLevel = getenv("PCCL_MAX_CONCURRENT_COLLECTIVE_OPS");
    if (logLevel == nullptr) {
        return DEFAULT_MAX_CONCURRENT_COLLECTIVE_OPS;
    }
    return std::stoi(logLevel);
}

namespace ccoip {
    class CCoIPClientState {
        /// UUID assigned to this client by the master
        ccoip_uuid_t assigned_uuid{};

        /// The global world size as determined by the master
        std::atomic<size_t> global_world_size{};

        /// The local world size as determined by the master
        std::atomic<size_t> local_world_size{};

        /// The number of distinct peer groups defined in the run as determined by the master
        std::atomic<size_t> num_distinct_peer_groups{};

        /// The number of peers in the largest peer group as determined by the master
        std::atomic<size_t> largest_peer_group_world_size{};

        /// Maps p2p listen socket addresses of respective clients to their UUID assigned by the master.
        /// Populated by @code registerPeer()@endcode when this peer establishes a p2p connection to said p2p listen socket address.
        /// Cleared by @code unregisterPeer()@endcode when a client is no longer needed due to topology changes.
        ///
        /// If a p2p connection drops, for now it is asserted that it will also disconnect from the master, triggering
        /// a topology update. TODO: Implement a more robust mechanism to handle this.
        std::unordered_map<internal_inet_socket_address_t, ccoip_uuid_t> socket_addr_to_uuid{};
        std::shared_mutex socket_addr_to_uuid_mutex{};

        /// References the current shared state to be distributed.
        /// Is only valid when @code is_syncing_shared_state@endcode is true and is otherwise empty.
        ccoip_shared_state_t current_shared_state{};
        std::shared_mutex current_shared_state_mutex{};

        /// Whether the client is currently syncing shared state.
        /// If this flag is false while the shared state distribution server receives
        /// a shared state request, the server will respond with @code SHARED_STATE_NOT_DISTRIBUTED@endcode
        std::atomic_bool is_syncing_shared_state = false;

        /// The number of bytes sent during the shared state sync phase
        /// Accessible by @code getSharedStateSyncTxBytes()@endcode
        /// Reset by @code resetSharedStateSyncTxBytes()@endcode
        std::atomic_size_t shared_state_sync_tx_bytes = 0;

        /// The number of bytes sent during ongoing collective communications operations
        /// Accessible by @code getCollectiveComsTxBytes(uint32_t tag)@endcode
        /// Reset by @code resetCollectiveComsRxBytes(uint32_t tag)@endcode
        std::unordered_map<uint64_t, size_t> collective_coms_tx_bytes{};
        std::shared_mutex collective_coms_tx_bytes_mutex{};

        /// The number of bytes received during ongoing collective communications operations
        /// Accessible by @code trackCollectiveComsRxBytes(uint32_t tag)@endcode
        /// Cleared by @code resetCollectiveComsRxBytes(uint32_t tag)@endcode
        std::unordered_map<uint64_t, std::atomic_size_t> collective_coms_rx_bytes{};
        std::shared_mutex collective_coms_rx_bytes_mutex{};

        /// The connection revision active at the start of the collective communications operation with the specified tag
        /// Accessible by @code getCollectiveConnectionRevision(uint32_t tag)@endcode
        /// Set by @code setCollectiveConnectionRevision(uint32_t tag, uint64_t revision)@endcode
        std::unordered_map<uint64_t, uint64_t> collective_coms_connection_revisions{};
        std::shared_mutex collective_coms_connection_revisions_mutex{};

        /// Maps tags of running collective communications operations to their respective world size
        /// that was valid at the time of the operation;
        /// Accessible by @code getCollectiveComsWorldSize(uint32_t tag)@endcode
        /// Cleared by @code resetCollectiveComsRxBytes(uint32_t tag)@endcode
        std::unordered_map<uint64_t, uint32_t> running_collective_coms_ops_world_size{};
        std::shared_mutex running_collective_coms_ops_world_size_mutex{};

        /// Tags of all running collective communications operations
        std::unordered_set<uint64_t> running_collective_coms_ops_tags{};
        std::shared_mutex running_collective_coms_ops_tags_mutex{};

        /// Threadpool for currently running collective communications operations
        pi::threadpool::ThreadPool collective_coms_threadpool{GetMaxConcurrentCollectiveOps(), 64};

        /// Maps tags of running collective operation tasks to their respective futures
        std::unordered_map<uint64_t, pi::threadpool::TaskFuture<pi::threadpool::void_t>> running_collective_ops{};

        /// Maps tags of running collective operation tasks to their respective failure states;
        /// These failure states are used to signal the completion of the collective operation task
        /// and indicate whether the operation was successful or not.
        /// 1=success, 0=failure, 2=not completed
        std::unordered_map<uint64_t, std::atomic<uint32_t>> running_reduce_tasks_failure_states{};
        std::shared_mutex running_reduce_tasks_failure_states_mutex{};

        // TODO: THIS IS SUBJECT TO CHANGE AND A TEMPORARY HACK!!
        //  FOR NOW ASSERT THE TOPOLOGY TO BE A SIMPLE RING REDUCE WITH A NON-PIPELINED ORDER
        std::vector<ccoip_uuid_t> ring_order;

        std::thread::id main_thread_id;

    public:
        CCoIPClientState();

        ~CCoIPClientState();

        /// Called to register a peer with the client state
        [[nodiscard]] bool registerPeer(const ccoip_socket_address_t &address, ccoip_uuid_t uuid);

        /// Called to unregister a peer with the client state
        [[nodiscard]] bool unregisterPeer(const ccoip_socket_address_t &address);

        /// Sets the uuid assigned to this client by the master
        void setAssignedUUID(const ccoip_uuid_t &new_assigned_uuid);

        /// Sets the global world size as determined by the master
        void setGlobalWorldSize(size_t new_global_world_size);

        /// Sets the local world size as determined by the master
        void setLocalWorldSize(size_t new_local_world_size);

        /// Sets the number of distinct peer groups as determined by the master
        void setNumDistinctPeerGroups(size_t new_num_distinct_peer_groups);

        /// Sets the number of peers in the largest peer group as determined by the master
        void setLargestPeerGroupWorldSize(size_t new_largest_peer_group_world_size);

        /// Returns the uuid assigned to this client by the master
        [[nodiscard]] const ccoip_uuid_t &getAssignedUUID() const;

        /// Begins the shared state synchronization phase.
        /// On the client side, this means that we are either already in the shared state sync phase after
        /// a successful vote or that we would want to be in the shared state sync phase.
        /// The reason for this is documented in ccoip::CCoIPClientHandler::syncSharedState()
        void beginSyncSharedStatePhase(const ccoip_shared_state_t &shared_state);

        /// Ends the shared state synchronization phase
        void endSyncSharedStatePhase();

        /// Returns true if the client is currently syncing shared state
        [[nodiscard]] bool isSyncingSharedState() const;

        /// Returns true if there is a collective communications operation running with the specified tag
        [[nodiscard]] bool isCollectiveComsOpRunning(uint64_t tag);

        /// Returns true if there is any collective communications operation running
        [[nodiscard]] bool isAnyCollectiveComsOpRunning();

        /// Launches an asynchronous all reduce operation
        [[nodiscard]] bool launchAsyncCollectiveOp(uint64_t tag, std::function<void(std::promise<bool> &)> &&task);

        /// Join the collective communications operation with the specified tag
        /// Returns false if no collective communications operation with the specified tag is running
        [[nodiscard]] bool joinAsyncCollectiveOp(uint64_t tag);

        /// Returns true if the collective communications operation with the specified tag failed;
        /// Returns std::nullopt if the tag is not found or the operation has not completed
        [[nodiscard]] std::optional<bool> hasCollectiveComsOpFailed(uint64_t tag);

        /// Returns the currently synced shared state
        [[nodiscard]] const ccoip_shared_state_t &getCurrentSharedState();

        /// Returns the number of bytes sent during the shared state sync phase
        [[nodiscard]] size_t getSharedStateSyncTxBytes() const;

        /// Called to track additional bytes sent during the shared state sync phase
        void trackSharedStateTxBytes(size_t tx_bytes);

        /// Resets the number of bytes sent during the shared state sync phase
        void resetSharedStateSyncTxBytes();

        /// Called to track additional bytes received during ongoing collective communications operations
        void trackCollectiveComsRxBytes(uint64_t tag, size_t rx_bytes);

        /// Resets the number of bytes received during ongoing collective communications operations
        /// @code getCollectiveComsRxBytes()@endcode will return std::nullopt after this call
        void resetCollectiveComsRxBytes(uint64_t tag);

        /// Returns the number of bytes sent during ongoing collective communications operations;
        /// Returns std::nullopt if the tag is not found
        [[nodiscard]] std::optional<size_t> getCollectiveComsRxBytes(uint64_t tag);

        /// Sets the connection revision active at the start of the collective communications operation with the specified tag
        void setCollectiveConnectionRevision(uint64_t tag, uint64_t revision);

        /// Returns the connection revision active at the start of the collective communications operation with the specified tag
        [[nodiscard]] std::optional<uint64_t> getCollectiveConnectionRevision(uint64_t tag);

        /// Called to track additional bytes sent during ongoing collective communications operations
        void trackCollectiveComsTxBytes(uint64_t tag, size_t tx_bytes);

        /// Resets the number of bytes sent during ongoing collective communications operations
        /// @code getCollectiveComsTxBytes()@endcode will return std::nullopt after this call
        void resetCollectiveComsTxBytes(uint64_t tag);

        /// Returns the number of bytes sent during ongoing collective communications operations
        /// Returns std::nullopt if the tag is not found
        [[nodiscard]] std::optional<size_t> getCollectiveComsTxBytes(uint64_t tag);

        /// Returns the world size of the collective communications operation with the specified tag
        /// Returns std::nullopt if the tag is not found
        [[nodiscard]] std::optional<uint32_t> getCollectiveComsWorldSize(uint64_t tag);

        /// Resets the world size of the collective communications operation with the specified tag
        void resetCollectiveComsWorldSize(uint64_t tag);

        // TODO: THIS IS SUBJECT TO CHANGE, FOR NOW ASSERT THE TOPOLOGY TO BE A SIMPLE RING REDUCE WITH A NON-PIPELINED ORDER
        void updateTopology(const std::vector<ccoip_uuid_t> &new_ring_order);

        // TODO: THIS IS SUBJECT TO CHANGE, FOR NOW ASSERT THE TOPOLOGY TO BE A SIMPLE RING REDUCE WITH A NON-PIPELINED ORDER
        [[nodiscard]] const std::vector<ccoip_uuid_t> &getRingOrder() const {
            return ring_order;
        }

        /// !!! A WORD OF CAUTION !!!
        /// It is recommended to read the comment in ccoip_master_state.cpp called "A WORD OF CAUTION w.r.t getLocalWorldSize and getGlobalWorldSize"
        /// There is a non-trivial amount of calling convention implied how getLocalWorldSize and getGlobalWorldSize are used.
        /// This also has implications for

        /// Returns the local world size. World size here shall mean the number of peers participating in the peer group that this client is a part of.
        [[nodiscard]] size_t getLocalWorldSize() const;

        /// Returns the global world size. World size here shall mean the total number of peers connected across all peer groups.
        [[nodiscard]] size_t getGlobalWorldSize() const;

        /// Returns the number of distinct peer groups defined in the run. A peer group is considered defined once one or more peers that declares the particular value as its peer group
        /// is accepted into the run.
        [[nodiscard]] size_t getNumDistinctPeerGroups() const;

        /// Returns the number of peers in the largest peer group defined in the run. A peer group is considered defined once on or more peers that declares the particular value as its peer group
        /// is accepted into the run.
        /// This value will return the same value on all peers in the run and across all peer groups.
        [[nodiscard]] size_t getLargestPeerGroupWorldSize() const;

        /// Sets the thread id that the client considers to be the main thread.
        void setMainThread(std::thread::id main_thread_id);

        std::vector<uint64_t> getRunningCollectiveComsOpTags();

    private:
        /// Declares a collective communications operation with the specified tag started.
        /// Returns false if a collective communications operation with the same tag is already running.
        [[nodiscard]] bool startCollectiveComsOp(uint64_t tag);

        /// Declares a collective communications operation with the specified tag ended
        /// Returns false if no collective communications operation with the specified tag is running.
        /// NOTE: asserts caller holds running_collective_coms_ops_tags_mutex
        [[nodiscard]] bool endCollectiveComsOp(uint64_t tag);
    };
};
