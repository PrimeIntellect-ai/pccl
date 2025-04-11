#pragma once

#include <bandwidth_store.hpp>
#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <ccoip_types.hpp>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace ccoip {
    enum ConnectionPhase {
        /// The client is registered with the master;
        /// Initially, the client is in this state.
        /// It does not yet participate in collective communications operations
        /// or shared state distribution.
        /// In this state, the client is not allowed to request to participate in any of the above.
        /// While in @code PEER_REGISTERED@endcode, the client is generally restricted to @code IDLE@endcode or the
        /// transitional states used for p2p establishment (@code CONNECTING_TO_PEERS@endcode or @code
        /// WAITING_FOR_OTHER_PEERS@endcode).
        /// The client is not allowed to request other phases.
        PEER_REGISTERED,

        /// Peers have accepted the client.
        /// Peers periodically accept new clients to join the running session.
        /// This is a phase unanimously agreed upon by all peers.
        /// New clients once they have established their p2p connections with all peers
        /// and confirmed to the master to have done so, will be moved to this phase.
        PEER_ACCEPTED
    };

    enum ConnectionState {
        /// The client is idle.
        /// It does not participate in any collective communications operations
        /// or shared state distribution.
        /// When the @code ConnectionPhase@endcode is @code PEER_REGISTERED@endcode,
        /// this is the only legal state for the client to be in (except @code CONNECTING_TO_PEERS@endcode & @code
        /// WAITING_FOR_OTHER_PEERS@endcode).
        IDLE,

        /// The client has voted to accept new peers.
        /// In this state, it waits for all peers to vote to accept new peers.
        VOTE_ACCEPT_NEW_PEERS,

        /// The client has voted to establish p2p connections without accepting new peers.
        /// In this state, it waits for all peers to vote to establish p2p connections without accepting new peers.
        /// This is used to re-establish p2p connections after topology changes, where the client does not want to
        /// accept new peers.
        VOTE_NO_NEW_PEERS_ESTABLISH_P2P_CONNECTIONS,

        /// When all peers have voted to accept new peers, the client
        /// will enter the p2p establishment phase.
        CONNECTING_TO_PEERS,

        /// When the client has failed to establish p2p connections with all peers,
        /// it will enter this state.
        /// In this state, the client may only attempt to retry establishing p2p connections.
        /// This means the client must vote to accept new peers again.
        /// Because this necessitates all peers to vote to accept new peers again,
        /// this failure is propagated to all clients.
        CONNECTING_TO_PEERS_FAILED,

        /// When the node has successfully established p2p connections with all peers,
        /// but other peers have not yet declared that they have established p2p connections,
        /// the client will be in the @code WAITING_FOR_OTHER_PEERS@endcode state.
        /// Once all peers have declared that they have established p2p connections,
        /// the client will return to the @code IDLE@endcode state.
        WAITING_FOR_OTHER_PEERS,

        /// The client has voted to optimize the topology.
        /// In this state, it waits for all peers to vote to optimize the topology.
        VOTE_OPTIMIZE_TOPOLOGY,

        /// When all clients have voted to optimize the topology, the client will enter the @code
        /// OPTIMIZE_TOPOLOGY@endcode state.
        /// In this state, the clients will perform the necessary work to optimize the topology.
        /// This may be no work at all, if the topology has not changed.
        /// Work may include bandwidth testing, if requested by the master.
        /// Bandwidth information is reported to the master, which will in turn update the topology to reflect the new
        /// bandwidth information.
        /// TODO: IMPLEMENT THAT CLIENTS MAY REQUEST TO RE-OPTIMIZE THE TOPOLOGY IF THEY DETECT DROP IN BANDWIDTH
        OPTIMIZE_TOPOLOGY,

        /// If the master detects that not all clients have performed all work requests necessary for topology
        /// optimization,
        /// it will enter all clients into this state. In this state, the only legal state for the client to transition
        /// into
        /// is into the @code VOTE_OPTIMIZE_TOPOLOGY@endcode state, as the client must re-request work from the master
        /// that
        /// is necessary to optimize the topology.
        OPTIMIZE_TOPOLOGY_FAILED,

        /// After performing the work necessary to optimize the topology, the client will vote to complete the topology
        /// optimization.
        /// In this state, it waits for all peers to vote to complete the topology optimization.
        /// Once all peers have declared that they have completed the topology optimization work, the client will return
        /// to the @code IDLE@endcode state.
        VOTE_COMPLETE_TOPOLOGY_OPTIMIZATION,

        /// The client has voted to synchronize shared state.
        /// In this state, it waits for all peers to vote to synchronize shared state.
        VOTE_SYNC_SHARED_STATE,

        /// When all peers of a peer group have voted to synchronize shared state, depending on whether
        /// the client has an up-to-date shared state, the client will either
        /// enter the @code DISTRIBUTE_SHARED_STATE@endcode state or the @code REQUEST_SHARED_STATE@endcode state.
        /// In the @code DISTRIBUTE_SHARED_STATE@endcode state, the client will distribute its shared state to peers
        /// that request it.
        /// Shared state distribution happens over one-off, non-persistent connections, similar to HTTP/1.1 connections.
        /// This is because of the logical mismatch between the persistent nature of the p2p connections
        /// and the non-persistent nature of shared state distribution, which may be between peers that are not
        /// topological neighbors. Thus, to avoid complexity in the p2p connections, shared state distribution
        /// is done over one-off connections.
        DISTRIBUTE_SHARED_STATE,

        /// When all peers of a peer group have voted to synchronize shared state, depending on whether
        /// the client has an up-to-date shared state, the client will either
        /// enter the @code DISTRIBUTE_SHARED_STATE@endcode state or the @code REQUEST_SHARED_STATE@endcode state.
        /// In the @code REQUEST_SHARED_STATE@endcode state, the client will request shared state from a peer
        /// the master node has designated. This node may or may not be a topological neighbor as determined
        /// by the reduce topology. The client will then initiate a one-off connection to the designated peer.
        REQUEST_SHARED_STATE,


        /// The client has voted to complete the shared state distribution phase.
        /// After each client has requested the set of shared state entries that were declared outdated and subsequent
        /// distribution was successful,
        /// the client will vote to complete the shared state distribution phase.
        /// Clients that do not have any outdated shared state entries will immediately vote to complete the shared
        /// state distribution phase.
        /// Consensus is required for the vote to complete, meaning all clients in the @code PEER_ACCEPTED@endcode phase
        /// must vote to complete the shared state distribution phase.
        /// This consensus ensures that no client proceeds beyond the shared state distribution phase until all clients
        /// have received the shared state entries that were outdated
        /// and further, that clients that could act as shared state distributors progress. While shared state
        /// distributors by definition
        /// have non-dirty shared state and will thus immediately vote to complete the shared state distribution phase,
        /// this however does not
        /// imply that these clients progress to the next phase! They will wait, as intended for all clients to vote to
        /// complete the shared state distribution phase
        /// and while doing so process incoming shared state distribution requests.
        /// After the shared state synchronization phase ends, clients return to the @code IDLE@endcode state.
        VOTE_COMPLETE_SHARED_STATE_SYNC,

        /// Because a client can launch multiple concurrent collective communications operations, it is not possible to
        /// keep
        /// track of voting states for each collective communications operation as ConnectionStates, as these are not
        /// "per connection",
        /// but "per tag".
        /// Tags correspond to exactly one currently active collective communications operation, where active shall mean
        /// either in the process of voting or in the process of being performed.
        /// Thus, this state is used to indicate that the client has voted to initiate at least one collective
        /// communications operation and
        /// may or may not be currently in the process of performing one or more collective communications operations
        /// after a successful vote.
        /// As long as there is at least one collective communications operations running, the client will remain in
        /// this state.
        /// New collective communications operation votes can be started when in this state, and votes can be finalized
        /// in this state.
        /// However, in this state, the client cannot transition to any other state until all collective communications
        /// operations have completed.
        /// Because more fine-grained state tracking is required for collective communications operations, the @code
        /// CollectiveCommunicationState@endcode enum is used
        /// to indicate the state of each collective communications operation per tag.
        /// Note that the client can be in multiple @code CollectiveCommunicationState@endcode states at the same time,
        /// one for each collective communications operation.
        COLLECTIVE_COMMUNICATIONS_RUNNING
    };

    enum CollectiveCommunicationState {
        /// The client has voted to initiate a collective communications operation with a particular tag.
        /// In this state, the client waits for all peers to vote to initiate a collective communications operation for
        /// this tag.
        /// After each client has voted to initiate a collective communications operation for this tag, the client will
        /// enter the
        /// @code PERFORM_ALL_REDUCE@endcode state for this tag.
        /// A client can be in multiple @code AllReduceState@endcode states per tag, one for each collective
        /// communications operation.
        VOTE_INITIATE_COLLECTIVE_COMMS,

        /// When all peers have voted to initiate a collective communications operation for a particular tag, the client
        /// will enter the
        /// @code PERFORM_ALL_REDUCE@endcode state for this tag.
        /// In this state, clients will perform p2p communication to perform the collective communications operation
        /// according to the
        /// topology defined by the master node.
        PERFORM_COLLECTIVE_COMMS,

        /// After a peer has performed the work necessary to complete the collective communication operation for itself,
        /// it will vote to complete the collective communications operation.
        /// Once all peers have voted to complete the collective communications operation, the collective communications
        /// operation
        /// is considered complete and fully performed by all peers.
        /// Once all peers have voted to complete the collective communications operation,
        /// there will be no collective communications state associated with this tag until a new collective
        /// communications operation is initiated.
        VOTE_COMPLETE_COLLECTIVE_COMMS
    };

    /// State of the phase-independent vote for querying if there are pending peers.
    /// In the instant of the last incoming vote for the has pending peers vote, the state of whether
    /// there are peers pending is determined by the master and returned to the client.
    enum VoteQueryHasPendingPeers {

        /// The initial state.
        /// No vote to obtain the pending peers state has been initiated by this peer.
        NO_VOTE_INITIATED_QUERY_PENDING_PEERS,

        /// The state to indicate that the client has voted to query if there are pending peers.
        VOTE_QUERY_PENDING_PEERS

    };

    struct CCoIPClientVariablePorts {
        /// The port the peer listens to for p2p connections
        uint16_t p2p_listen_port;

        /// The port the peer listens to for shared state distribution requests
        uint16_t shared_dist_state_listen_port;

        /// The port the peer listens to for benchmarking network bandwidth
        uint16_t bandwidth_benchmark_listen_port;
    };

    struct ClientInfo {
        ccoip_uuid_t client_uuid{};
        ConnectionPhase connection_phase = PEER_REGISTERED;
        ConnectionState connection_state = IDLE;
        VoteQueryHasPendingPeers vote_query_pending_peers = NO_VOTE_INITIATED_QUERY_PENDING_PEERS;

        std::unordered_map<uint64_t, CollectiveCommunicationState> collective_coms_states{};

        ccoip_socket_address_t socket_address{};
        CCoIPClientVariablePorts variable_ports{};
        uint32_t peer_group{};
    };

#define CALLSITE_TOPOLOGY_OPTIMIZATION_COMPLETE 1
#define CALLSITE_P2P_CONNECTION_INFORMATION 2

    class CCoIPMasterState {
    public:
        /// Represents the status of a comparison between the shared state mask and supplied shared state entries.
        enum SharedStateMismatchStatus {
            SUCCESSFUL_MATCH,
            KEY_SET_MISMATCH,
            CONTENT_HASH_MISMATCH,
            REVISION_OUTDATED,
            REVISION_INCREMENT_VIOLATION
        };

    private:
        /// Maps the socket address of the client to its assigned UUID
        /// Populated on successful session join, and cleared on client leave/disconnect
        std::unordered_map<internal_inet_socket_address_t, ccoip_uuid_t> client_uuids{};

        /// Maps the UUID of the client to its assigned socket address
        /// Populated identically to `client_uuids` for reverse lookups
        std::unordered_map<ccoip_uuid_t, internal_inet_socket_address_t> uuid_clients{};

        /// Maps the uuid of the client to its client information
        std::unordered_map<ccoip_uuid_t, ClientInfo> client_info{};

        /// set of all uuids that have voted to accept new peers.
        /// cleared once accept new peers consensus is reached.
        /// Removed from on client leave/disconnect.
        std::unordered_set<ccoip_uuid_t> votes_accept_new_peers{};

        /// set of all uuids that have voted to establish p2p connections without accepting new peers.
        /// cleared once p2p connections establishment consensus is reached.
        /// Removed from on client leave/disconnect.
        std::unordered_set<ccoip_uuid_t> votes_establish_p2p_connections{};

        /// set of all uuids that have voted to optimize the topology.
        /// cleared once topology optimization consensus is reached.
        /// Removed from on client leave/disconnect.
        std::unordered_set<ccoip_uuid_t> votes_optimize_topology{};

        /// set of all uuids that have voted to complete the topology optimization.
        /// cleared once topology optimization completion consensus is reached.
        /// Removed from on client leave/disconnect.
        std::unordered_set<ccoip_uuid_t> votes_complete_topology_optimization{};

        /// set of all uuids that have voted to synchronize shared state for each peer group.
        /// Peer group bin is cleared once shared state distribution consensus is reached.
        /// Removed from on client leave/disconnect.
        std::unordered_map<uint32_t, std::unordered_set<ccoip_uuid_t>> votes_sync_shared_state{};

        /// set of all uuids that have voted to complete shared state distribution for each peer group.
        /// Peer group bin is cleared once the shared state distribution phase ends.
        /// Removed from on client leave/disconnect.
        std::unordered_map<uint32_t, std::unordered_set<ccoip_uuid_t>> votes_sync_shared_state_complete{};

        /// Flag per "callsite" to indicate if the peer list has changed;
        /// Each call-site defines separate scope for checking whether the peer list has changed since the last
        /// invocation from said call site. See @code hasPeerListChanged@endcode
        std::unordered_map<uint32_t, bool> peer_list_changed{{CALLSITE_TOPOLOGY_OPTIMIZATION_COMPLETE, false},
                                                             {CALLSITE_P2P_CONNECTION_INFORMATION, false}};

        /// Flag per "callsite" to indicate if the topology has changed;
        /// Each call-site defines separate scope for checking whether the peer list has changed since the last
        /// invocation from said call site. See @code hasTopologyChanged@endcode
        std::unordered_map<uint32_t, bool> topology_changed{{CALLSITE_TOPOLOGY_OPTIMIZATION_COMPLETE, false},
                                                            {CALLSITE_P2P_CONNECTION_INFORMATION, false}};

        /// Defines the "mask" of shared state entries for each peer group.
        /// The mask is defined by the identity of the set of shared state key strings and their corresponding hashes.
        /// This mask is checked against the by clients synchronizing shared state.
        /// Mismatch of the keys results in the client being kicked.
        /// Mismatch of a hash results will result in the client being notified to re-request the shared state entry
        /// whose hash does not match.
        /// This "mask" is elected by popularity from a set of candidates provided by
        /// each peer that has voted to synchronize shared state.
        /// Peer group bin is cleared once the shared state distribution phase ends.
        std::unordered_map<uint32_t, std::vector<SharedStateHashEntry>> shared_state_mask{};

        /// Defines the list of candidates for the shared state mask for each peer group.
        /// Most popular candidate is elected as the shared state mask.
        /// Peer group bin is cleared after the shared state distribution phase ends.
        /// Removed from on client leave/disconnect.
        std::unordered_map<uint32_t, std::vector<std::pair<ccoip_uuid_t, std::vector<SharedStateHashEntry>>>>
                shared_state_mask_candidates{};

        /// Cache of the shared state hashes for all entries in the shared state mask for each peer group.
        /// by their key string.
        /// Peer group bin is cleared when the shared state distribution phase ends.
        std::unordered_map<uint32_t, std::unordered_map<std::string, uint64_t>> shared_state_hashes{};

        /// Cache of the shared state hash types for all entries in the shared state mask for each peer group.
        /// by their key string.
        /// Peer group bin is cleared when the shared state distribution phase ends.
        std::unordered_map<uint32_t, std::unordered_map<std::string, ccoip_hash_type_t>> shared_state_hash_types{};

        /// The next shared state revision that is expected from peers synchronizing shared state, intending to
        /// distribute. Shared state revisions that do not match this value will either be considered outdated or a
        /// violation of the one-increment-only rule. This value is stored per peer group. Each peer group maintains
        /// their own shared state, and thus their own revision.
        ///
        /// If a client requests to sync a shared state with a revision smaller than this,
        /// this means that the client has an outdated shared state that is outdated and needs to be updated.
        /// If a client requests a shared state with a revision larger than this, the client has violated the
        /// one-increment-only rule.
        /// It is expected that clients will either
        /// a) independently advance their shared state and remain in sync implicitly or
        /// b) request the shared state from the master node to synchronize.
        ///
        /// After the completion of a shared state voting phase, this value is incremented by the master
        /// and ANTICIPATES that clients will increment their shared state revision by one.
        std::unordered_map<uint32_t, uint64_t> next_shared_state_revision{};

        /// Maps the client UUID to the shared state mismatch status populated by the last invocation of @code
        /// sharedStateMatches@endcode. Grouped by peer group. Peer group bin is cleared when the shared state
        /// distribution phase ends.
        std::unordered_map<uint32_t, std::unordered_map<ccoip_uuid_t, SharedStateMismatchStatus>>
                shared_state_statuses{};

        /// Maps the client UUID to the set of shared state keys that have dirty content, meaning
        /// that the hash of the shared state entry does not match the hash in the shared state mask.
        /// Grouped by peer group.
        /// Peer group bin is cleared when the shared state distribution phase ends.
        std::unordered_map<uint32_t, std::unordered_map<ccoip_uuid_t, std::vector<std::string>>>
                shared_state_dirty_keys{};

        /// Boolean state for each collective communications operation tag in every peer group that indicates if the
        /// operation has been aborted or not. true = aborted false = not aborted Cleared when the collective
        /// communications operation ends.
        std::unordered_map<uint32_t, std::unordered_map<uint64_t, bool>> collective_comms_op_abort_states{};

        /// Map of all peers that a given peer cannot reach.
        /// unreachability_map[a] is the list of peers that peer a cannot reach.
        /// Populated on failed p2p connection establishment and when bandwidth benchmarking fails.
        std::unordered_map<ccoip_uuid_t, std::unordered_set<ccoip_uuid_t>> unreachability_map{};

        /// Bandwidth store for all clients
        BandwidthStore bandwidth_store{};

        // TODO: THIS IS SUBJECT TO CHANGE:

        /// Mutex for the topology states
        std::shared_mutex topology_mutex{};

        /// Map current ring topology in use for each peer group.
        /// List order defined the ring order.
        std::unordered_map<uint32_t, std::vector<ccoip_uuid_t>> ring_topologies{};

        /// whether the current topology is optimal as determined by the topology optimizer.
        /// If we reach optimal topology, we stop continuous optimization.
        std::unordered_map<uint32_t, bool> topology_is_optimal{};

    public:
        /// Registers a client
        [[nodiscard]] auto registerClient(const ccoip_socket_address_t &client_address,
                                          const CCoIPClientVariablePorts &variable_ports, uint32_t peer_group,
                                          ccoip_uuid_t uuid) -> bool;

        /// Unregisters a client
        [[nodiscard]] bool unregisterClient(const ccoip_socket_address_t &client_address);

        /// Checks if a client is registered
        [[nodiscard]] bool isClientRegistered(const ccoip_socket_address_t &client_address) const;

        /// Called when a client votes to accept new peers.
        /// All clients must vote to accept new peers before all clients start the p2p establishment phase.
        [[nodiscard]] bool voteEstablishP2PConnections(const ccoip_uuid_t &peer_uuid, bool accept_new_peers);

        /// Called when a client votes to check if peers are pending to be accepted.
        /// All clients must vote to check for pending peers before the master can determine the state of whether peers
        /// are pending at the instant of the last vote. This vote is independent of all other phases and can happen
        /// completely concurrently to any other phase. The master does not strictly enforce the lifecycle of this vote
        /// as strictly as it does for other votes. The reason for this is that peers should be able to obtain this
        /// information at any time to inform application branching for optimization purposes.
        [[nodiscard]] bool voteQueryWaitingPeersPending(const ccoip_uuid_t &peer_uuid);

        /// Called when a client votes to optimize the topology
        /// All clients must vote to optimize the topology before topology optimization can begin.
        [[nodiscard]] bool voteOptimizeTopology(const ccoip_uuid_t &peer_uuid);

        /// Called when a client votes to complete the topology optimization phase.
        /// This indicates that it has completed the work requested by the master that was necessary for the master to
        /// be able to optimize the topology. All clients must vote to complete the topology optimization phase before
        /// the topology optimization phase can end. When consensus is reached, @code
        /// endTopologyOptimizationPhase()@endcode will be called. All clients here shall mean the subset of clients
        /// that are in the @code PEER_ACCEPTED@endcode phase.
        [[nodiscard]] bool voteTopologyOptimizationComplete(const ccoip_uuid_t &peer_uuid);

        /// Called when a client votes to synchronize shared state.
        /// All clients must vote to synchronize shared state before shared state distribution can begin.
        [[nodiscard]] bool voteSyncSharedState(const ccoip_uuid_t &peer_uuid);

        /// Called when a client votes to complete the shared state distribution phase.
        /// This indicates that it has completed receiving the subset of shared state that was outdated.
        /// All clients must vote to complete shared state distribution before shared state distribution can end.
        /// When consensus is reached, @code endSharedStateDistributionPhase()@endcode will be called.
        /// All clients here shall mean the subset of clients that are in the @code PEER_ACCEPTED@endcode phase.
        [[nodiscard]] bool voteDistSharedStateComplete(const ccoip_uuid_t &peer_uuid);

        /// Called when a client votes to initiate a collective communications operation with a particular tag.
        /// @param peer_uuid the uuid of the client that is initiating the collective communications operation. The peer
        /// group of said client determines what peers will participate in the collective communications operation.
        /// @param tag the tag associated with the collective communications operation.
        /// All clients must vote to initiate a collective communications operation before the operation can begin.
        /// Clients can concurrently vote to initiate and perform multiple collective communications operations.
        [[nodiscard]] bool voteCollectiveCommsInitiate(const ccoip_uuid_t &peer_uuid, uint64_t tag);

        /// Called when a client votes to complete a collective communications operation with a particular tag.
        /// @param peer_uuid the uuid of the client that is completing the collective communications operation. The peer
        /// group of said client determines what peers will participate in the collective communications operation.
        /// @param tag the tag associated with the collective communications operation.
        /// All clients must vote to complete a collective communications operation before the operation can end.
        [[nodiscard]] bool voteCollectiveCommsComplete(const ccoip_uuid_t &peer_uuid, uint64_t tag);

        /// Returns true if all peers of the peer group have voted to synchronize the shared state
        /// All clients here shall mean the subset of clients that are in the @code PEER_ACCEPTED@endcode phase.
        [[nodiscard]] bool syncSharedStateConsensus(uint32_t peer_group);

        /// Returns true if all clients have voted to accept new peers
        [[nodiscard]] bool acceptNewPeersConsensus() const;

        /// Returns true if all clients have voted to query if there are pending peers
        [[nodiscard]] bool queryPendingPeersConsensus() const;

        /// Returns true if all clients have voted to establish p2p connections without accepting new peers
        [[nodiscard]] bool noAcceptNewPeersEstablishP2PConnectionsConsensus() const;

        /// Returns true if all clients have voted to optimize the topology
        [[nodiscard]] bool optimizeTopologyConsensus() const;

        /// Returns true if all clients have voted to complete the topology optimization phase
        [[nodiscard]] bool topologyOptimizationCompleteConsensus() const;

        /// Transition to the p2p establishment phase
        /// Triggered after unanimous voting to accept new peers
        [[nodiscard]] bool transitionToP2PEstablishmentPhase(bool accept_new_peers);

        /// Transition to the topology optimization phase
        /// Triggered after unanimous voting to optimize the topology
        [[nodiscard]] bool transitionToTopologyOptimizationPhase();

        /// Marks that p2p connections have been established by a particular client.
        /// The particular client will be transitioned to the @code WAITING_FOR_OTHER_PEERS@endcode state
        /// until all peers have declared that they have established p2p connections.
        ///
        /// Once all peers have declared that they have established p2p connections (and succeeded in doing so)
        /// clients return back to the @code IDLE@endcode state.
        /// If @code success@endcode is false, the peer has indicated that it was not able to establish p2p connections
        /// to the peers inside @code failed_peers@endcode. If this is the case, we will mark the failed peer as
        /// unreachable as from the perspective of @code peer_uuid@endcode.
        [[nodiscard]] bool markP2PConnectionsEstablished(const ccoip_uuid_t &peer_uuid, bool success,
                                                         const std::vector<ccoip_uuid_t> &failed_peers);

        /// Transition to the p2p connections established phase
        /// Triggered after all clients have declared that they have established p2p connections either successfully or
        /// not. If @code any_failed@endcode is false, the following occurs: All clients will return to the @code
        /// IDLE@endcode state. Returns false if any client is not in the @code WAITING_FOR_OTHER_PEERS@endcode state If
        /// @code any_failed@endcode is true, then the following occurs: If peers are in the @code
        /// WAITING_FOR_OTHER_PEERS@endcode state, they will be transitioned to the @code
        /// CONNECTING_TO_PEERS_FAILED@endcode state. Otherwise, all clients are expected to be in the @code
        /// CONNECTING_TO_PEERS_FAILED@endcode state.
        [[nodiscard]] bool transitionToP2PConnectionsEstablishedPhase(bool failure);

        /// Transition to the shared state distribution phase
        /// Triggered after all peers of a peer group have voted to synchronize shared state
        /// Returns false if a client is in the PEER_ACCEPTED phase but not in the VOTE_SYNC_SHARED_STATE state or if
        /// @code sharedStateMatches@endcode has not been called for all clients.
        [[nodiscard]] bool transitionToSharedStateSyncPhase(uint32_t peer_group);

        /// Called to end the shared state distribution phase
        /// Will revert all peers of the peer group to the @code IDLE@endcode state
        [[nodiscard]] bool endSharedStateSyncPhase(uint32_t peer_group);

        /// Called to end the topology optimization phase
        /// Will revert all clients to the @code IDLE@endcode state
        [[nodiscard]] bool endTopologyOptimizationPhase(bool failed);

        /// Returns true if all clients have declared that they have established p2p connections
        /// Note: that "all clients" here means the subset of clients that are joining in the current acceptNewPeers
        /// phase. E.g. if a peer joins the session after consensus has been reached to accept new peers, it will not be
        /// included in this check.
        [[nodiscard]] bool p2pConnectionsEstablishConsensus() const;

        /// Returns true if all peers of the peer group have voted to complete the shared state distribution phase
        [[nodiscard]] bool syncSharedStateCompleteConsensus(uint32_t peer_group);

        /// Returns true if all peers of the peer group have voted to initiate a collective communications operation for
        /// this tag
        [[nodiscard]] bool collectiveCommsInitiateConsensus(uint32_t peer_group, uint64_t tag);

        /// Returns true if all peers of the peer group have voted to complete a collective communications operation for
        /// this tag
        [[nodiscard]] bool collectiveCommsCompleteConsensus(uint32_t peer_group, uint64_t tag);

        /// Transition all clients of the given peer group to the phase of performing the collective communications
        /// operation with the given tag/ Returns false if the client is not in the @code
        /// COLLECTIVE_COMMUNICATIONS_RUNNING@endcode state or if not in the @code
        /// VOTE_INITIATE_COLLECTIVE_COMMS@endcode state for the specified tag.
        [[nodiscard]] bool transitionToPerformCollectiveCommsPhase(uint32_t peer_group, uint64_t tag);

        /// Transition all clients of the given peer group to the phase of completing the collective communications
        /// operation with the given tag. Returns false if the client is not in the @code
        /// COLLECTIVE_COMMUNICATIONS_RUNNING@endcode state or if not in the @code
        /// VOTE_COMPLETE_COLLECTIVE_COMMS@endcode state for the specified tag.
        [[nodiscard]] bool transitionToCollectiveCommsCompletePhase(uint32_t peer_group, uint64_t tag);

        /// Aborts the specified collective communications operation with the given tag for the given peer group.
        /// Only legal to call when all clients are in the @code COLLECTIVE_COMMUNICATIONS_RUNNING@endcode state.
        /// Abort of a collective communications operation represents separate state from the current state that being
        /// @code COLLECTIVE_COMMUNICATIONS_RUNNING@endcode. Even when collective communications operations are aborted,
        /// clients will remain in the @code COLLECTIVE_COMMUNICATIONS_RUNNING@endcode state and vote to complete the
        /// collective communications operation once the abort signal was handled and the collective communications
        /// operation thus "completes" as per the protocol. Returns true if the collective communications operation was
        /// not in the abort state before this call. Returns false if the collective communications operation was
        /// already aborted by a previous call to this function. This state is accessible via @code
        /// isCollectiveCommsOperationAborted@endcode.
        [[nodiscard]] bool abortCollectiveCommsOperation(uint32_t peer_group, uint64_t tag);

        /// Returns true if the specified collective communications operation with the given tag for the given peer
        /// group has been aborted.
        [[nodiscard]] bool isCollectiveCommsOperationAborted(uint32_t peer_group, uint64_t tag) const;

        /// @returns true, if the collective communications operation with the specified tag in the specified peer group
        /// is in the running state, meaning that the operation was commenced and not yet finished.
        [[nodiscard]] bool isCollectiveOperationRunning(uint32_t peer_group, uint64_t tag) const;

        /// @returns SharedStateMismatchStatus::SUCCESSFUL_MATCH if the specified revision is legal as the next
        /// shared state revision for the given peer group.
        /// @returns @code SharedStateMismatchStatus::REVISION_OUTDATED@endcode if the specified revision is less than
        /// the next expected revision.
        /// @returns @code SharedStateMismatchStatus::REVISION_INCREMENT_VIOLATION@endcode if the specified revision is
        /// greater than the next expected revision. The next expected shared state revision is always one larger than
        /// the previous shared state revision. This status will be retained until the shared state voting phase ends
        /// and can be queried via @code getSharedStateMismatchStatus@endcode.
        [[nodiscard]] SharedStateMismatchStatus isNewRevisionLegal(const ccoip_uuid_t &peer_uuid, uint64_t revision);

        /// Votes for a specific shared state mask; Called for each peer that has voted to synchronize shared state.
        /// After all clients have voted to synchronize shared state, the elected shared state mask will be chosen by
        /// popularity.
        void voteSharedStateMask(const ccoip_uuid_t &peer_uuid, const std::vector<SharedStateHashEntry> &entries);

        /// Returns the peers that a particular client should establish p2p connections with
        [[nodiscard]] std::vector<ClientInfo> getPeersForClient(const ccoip_socket_address_t &client_address,
                                                                const ClientInfo &client_info, bool include_registered);

        /// Returns all client socket addresses
        [[nodiscard]] std::vector<ccoip_socket_address_t> getClientSocketAddresses();

        /// Returns a list pairs of client socket addresses and their corresponding UUIDs
        [[nodiscard]] std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t>> getClientEntrySet();

        /// Returns true if the peer list has changed since the last invocation of this function for this callsite.
        /// @param callsite a unique identifier for the caller of this function
        [[nodiscard]] bool hasPeerListChanged(uint32_t callsite);

        /// Returns true if the topology has changed since the last invocation of this function for this callsite.
        /// @param callsite a unique identifier for the caller of this function
        [[nodiscard]] bool hasTopologyChanged(uint32_t callsite);

        /// Returns the set of currently accepted peers, meaning the set of peers that are currently in the @code
        /// PEER_ACCEPTED@endcode phase.
        [[nodiscard]] std::unordered_set<ccoip_uuid_t> getCurrentlyAcceptedPeers();

        /// Returns the shared state mismatch status. This status is set by @code sharedStateMatches@endcode.
        /// If @code sharedStateMatches@endcode has not been called yet since the start of the current shared state
        /// voting phase, this function will return std::nullopt.
        [[nodiscard]] std::optional<SharedStateMismatchStatus>
        getSharedStateMismatchStatus(const ccoip_uuid_t &peer_uuid);

        /// Called to elect the shared state mask for a particular peer group.
        /// The most popular candidate will be chosen as the shared state mask.
        /// This function clears the shared state mask candidates for the peer group.
        /// @returns true if the shared state mask was elected successfully.
        /// @returns false if the shared state mask could not be elected, e.g. if no candidates were provided.
        [[nodiscard]] bool electSharedStateMask(uint32_t peer_group);

        /// Called to check if each peer matches the currently elected shared state mask.
        /// For this function to be invocable, @code electSharedStateMask@endcode must have been called before and
        /// the shared state mask for this peer group must not have been cleared yet.
        /// This function will mark each client whose shared state does not match the elected mask.
        /// This status can be queried via @code getSharedStateMismatchStatus@endcode.
        /// @returns true if the mismatch check was successful.
        /// @returns false if the check could not be performed, e.g. the shared state mask is empty.
        [[nodiscard]] bool checkMaskSharedStateMismatches(uint32_t peer_group);

        /// Stores the bandwidth between two peers. The bandwidth is stored in both directions, so that the bandwidth
        /// from A to B is stored in the entry (A, B) and the bandwidth from B to A is stored in the entry (B, A).
        /// @param from uuid of the "from" peer
        /// @param to uuid of the "to" peer
        /// @param send_bandwidth_mpbs the bandwidth in mbit/s that the "from" peer can send to the "to" peer. This is
        /// different from what the "to" peer can receive from the "from" peer, which is stored in a separate entry.
        void storePeerBandwidth(ccoip_uuid_t from, ccoip_uuid_t to, double send_bandwidth_mpbs);

        /// @param from the source peer for which this bandwidth entry is the tx bandwidth (as bottleneck by either peer
        /// "from" or peer "to")
        /// @param to to the destination peer for which this bandwidth entry is the rx bandwidth (as bottleneck by
        /// either peer "from" or peer "to")
        /// @return the bandwidth in mbit/s that the "from" peer can send to the "to" peer. If the bandwidth is not
        /// stored, returns std::nullopt.
        [[nodiscard]] std::optional<double> getPeerBandwidthMbps(ccoip_uuid_t from, ccoip_uuid_t to);

        /// Returns the set of bandwidth entries that are missing for a particular peer.
        /// In a fully and correctly populated bandwidth store, each peer has entries to all other peers where it is
        /// "from". Each peer also has entries where it is the "to" peer of each other peer. Conceptually, this can be
        /// though of as an asymmetric cost matrix. Entries that have never been reported, will be returned by this
        /// function.
        [[nodiscard]] std::vector<bandwidth_entry> getMissingBandwidthEntries(ccoip_uuid_t peer);

        /// Returns true if the bandwidth store is fully populated for all peers.
        [[nodiscard]] bool isBandwidthStoreFullyPopulated() const;

        /// Returns the number of peers registered in the bandwidth store
        [[nodiscard]] size_t getNumBandwidthStoreRegisteredPeers() const;

        /// Declares the following cost edge unreachable.
        /// Unreachable bandwidth entries will not be included in the @code getMissingBandwidthEntries@endcode, despite
        /// having no entry in the bandwidth store.
        void markBandwidthEntryUnreachable(const bandwidth_entry &bandwidth_entry);

        /// Finds the client UUID from the client address; returns std::nullopt if not found
        [[nodiscard]] std::optional<ccoip_uuid_t> findClientUUID(const ccoip_socket_address_t &client_address);

        /// Returns the client info for a particular client uuid; returns std::nullopt if not found
        [[nodiscard]] std::optional<std::reference_wrapper<ClientInfo>> getClientInfo(const ccoip_uuid_t &client_uuid);

        /// Returns the current shared state revision for the given peer group. This represents the current maximum
        /// revision of the shared state that all clients have agreed upon to be the current shared state revision.
        [[nodiscard]] uint64_t getSharedStateRevision(uint32_t peer_group);

        /// Returns the set of shared state keys of a particular peer that have dirty content, meaning that the hash of
        /// the shared state entry does not match the hash in the shared state mask. If @code sharedStateMatches@endcode
        /// has not been called yet since the start of the current shared state voting phase, this function will return
        /// an empty vector.
        [[nodiscard]] std::vector<std::string> getOutdatedSharedStateKeys(ccoip_uuid_t peer_uuid);

        /// Returns the shared state entry hash for a particular key; returns 0 if the key is not found
        /// For a key to be found, it must be present in the shared state mask.
        /// The shared state mask needs to be populated by @code sharedStateMatches@endcode before calling this
        /// function.
        [[nodiscard]] uint64_t getSharedStateEntryHash(uint32_t peer_group, const std::string &key);

        /// Returns the shared state entry hash type for a particular key; returns std::nullopt if the key is not found
        /// For a key to be found, it must be present in the shared state mask.
        /// The shared state mask needs to be populated by @code sharedStateMatches@endcode before calling this
        /// function.
        [[nodiscard]] std::optional<ccoip_hash_type_t> getSharedStateEntryHashType(uint32_t peer_group,
                                                                                   const std::string &key);

        /// Returns the set of shared state keys
        [[nodiscard]] std::vector<std::string> getSharedStateKeys(uint32_t peer_group);

        /// Returns the set of ongoing collective communications operation tags for a particular peer group
        [[nodiscard]] std::vector<uint64_t> getOngoingCollectiveCommsOpTags(uint32_t peer_group);

        /// Performs topology optimization on the current topology.
        /// @param peer_group
        /// @param moonshot if true, the topology will be optimized with a moonshot approach, which will take longer but
        /// may yield better results.
        /// @param new_topology the new topology that was generated by the topology optimizer.
        /// @param is_optimal whether the new topology is optimal.
        /// @param has_improved whether the topology has changed.
        /// Returns true if the topology optimization was successful.
        [[nodiscard]] bool performTopologyOptimization(uint32_t peer_group, bool moonshot,
                                                       std::vector<ccoip_uuid_t> &new_topology, bool &is_optimal,
                                                       bool &has_improved);

        /// Returns true if the current topology is optimal
        [[nodiscard]] bool isTopologyOptimal(uint32_t peer_group);

        // TODO: THIS IS SUBJECT TO CHANGE, AS IT ASSERTS THAT THE TOPOLOGY IS A RING ALWAYS

        /// Returns the ring topology
        [[nodiscard]] std::vector<ccoip_uuid_t> getRingTopology(uint32_t peer_group, bool include_registered_peers);

        [[nodiscard]] std::vector<uint32_t> getExistingPeerGroups();

        /// Sets the current ring reduce topology and the associated state whether it is optimal
        /// NOTE: This function IS thread-safe and may be called from other threads than the main thread
        [[nodiscard]] bool setRingTopology(uint32_t peer_group, const std::vector<ccoip_uuid_t> &new_topology,
                                           bool optimal);

        /// Returns the global world size, which is the total number of clients in the session
        [[nodiscard]] uint64_t getGlobalWorldSize() const;

        /// Returns the local world size for the specified peer group
        /// @oaram peer_group the peer group to obtain the number of peers of
        /// @param include_registered whether to count peers that are currently in the REGISTERED state
        [[nodiscard]] uint64_t getLocalWorldSize(uint32_t peer_group, bool include_registered) const;

        /// Returns the number of distinct peer groups defined in the run. A peer group is considered defined once one
        /// or more peers that declares a particular value to be its peer group is accepted into the run.
        /// @param include_registered whether to count peers that are currently in the REGISTERED state
        [[nodiscard]] uint64_t getNumDistinctPeerGroups(bool include_registered) const;

        /// Returns the largest peer group's local world size. E.i. the number of peers in the largest peer group.
        /// @param include_registered whether to count peers that are currently in the REGISTERED state
        uint64_t getLargestPeerGroupWorldSize(bool include_registered);

        /// Returns true if there is any peer that has not yet been accepted into the run, aka has pending state.
        [[nodiscard]] bool hasPendingPeers();

        /// Resets the vote query pending peers state for all clients.
        /// This is invoked after the query pending vote has been completed.
        void resetVoteQueryPendingPeers();

    private:
        void onPeerAccepted(const ClientInfo &info);

        /// Builds a default non-optimized ring topology.
        /// NOTE: Does not respect the @code unreachability_map@endcode !!
        [[nodiscard]] std::vector<ccoip_uuid_t> buildBasicRingTopology(uint32_t peer_group);

        /// Builds a non-optimized ring topology while respecting the @code unreachability_map@encode
        /// Returns std::nullopt, if no tour is possible given the state of the unreachability_map
        [[nodiscard]] std::optional<std::vector<ccoip_uuid_t>> buildReachableRingTopology(uint32_t peer_group);

        /// Sets the current ring reduce topology and the associated state whether it is optimal
        /// NOTE: This function IS NOT thread-safe. It is the unsafe version of @code setRingTopology@endcode
        /// and should only be used internally when the mutex is already locked.
        [[nodiscard]] bool setRingTopologyUnsafe(uint32_t peer_group, const std::vector<ccoip_uuid_t> &new_topology,
                                                 bool optimal);
    };
} // namespace ccoip
