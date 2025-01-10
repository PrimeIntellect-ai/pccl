#pragma once

#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <ccoip_types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace ccoip {
    enum ConnectionPhase {
        /// The client is registered with the master;
        /// Initially, the client is in this state.
        /// It does not yet participate in collective communications operations
        /// or shared state distribution.
        /// In this tate, the client is not allowed to request to participate in any of the above.
        /// While in this state, the only legal state for the client to be in is @code IDLE@endcode, except
        /// @code CONNECTING_TO_PEERS@endcode and @code WAITING_FOR_OTHER_PEERS@endcode while establishing p2p connections.
        /// Only after establishment, is the client moved to the @code PEER_ACCEPTED@endcode phase.
        PEER_REGISTERED,

        /// Peers have accepted the client.
        /// Peers periodically accept new clients to join the running session.
        /// This is a phase unanimously agreed upon by all peers.
        /// New clients once they have established their p2p connections with all peers,
        /// will be moved to this phase.
        PEER_ACCEPTED
    };

    enum ConnectionState {
        /// The client is idle.
        /// It does not participate in any collective communications operations
        /// or shared state distribution.
        /// When the @code ConnectionPhase@endcode is @code PEER_REGISTERED@endcode,
        /// this is the only legal state for the client to be in (except @code CONNECTING_TO_PEERS@endcode & @code WAITING_FOR_OTHER_PEERS@endcode).
        IDLE,

        /// The client has voted to accept new peers.
        /// In this state, it waits for all peers to vote to accept new peers.
        VOTE_ACCEPT_NEW_PEERS,

        /// When all peers have voted to accept new peers, the client
        /// will enter the p2p establishment phase.
        CONNECTING_TO_PEERS,

        /// When the node has established p2p connections with all peers,
        /// but other peers have not yet declared that they have established p2p connections,
        /// the client will be in the @code WAITING_FOR_OTHER_PEERS@endcode state.
        /// Once all peers have declared that they have established p2p connections,
        /// the client will return to the @code IDLE@endcode state.
        WAITING_FOR_OTHER_PEERS,

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
        /// After each client has requested the set of shared state entries that were declared outdated and subsequent distribution was successful,
        /// the client will vote to complete the shared state distribution phase.
        /// Clients that do not have any outdated shared state entries will immediately vote to complete the shared state distribution phase.
        /// Consensus is required for the vote to complete, meaning all clients in the @code PEER_ACCEPTED@endcode phase must vote to complete the shared state distribution phase.
        /// This consensus ensures that no client proceeds beyond the shared state distribution phase until all clients have received the shared state entries that were outdated
        /// and further, that clients that could act as shared state distributors progress. While shared state distributors by definition
        /// have non-dirty shared state and will thus immediately vote to complete the shared state distribution phase, this however does not
        /// imply that these clients progress to the next phase! They will wait, as intended for all clients to vote to complete the shared state distribution phase
        /// and while doing so process incoming shared state distribution requests.
        /// After the shared state synchronization phase ends, clients return to the @code IDLE@endcode state.
        VOTE_COMPLETE_SHARED_STATE_SYNC,

        /// Because a client can launch multiple concurrent collective communications operations, it is not possible to keep
        /// track of voting states for each collective communications operation as ConnectionStates, as these are not "per connection",
        /// but "per tag".
        /// Tags correspond to exactly one currently active collective communications operation, where active shall mean
        /// either in the process of voting or in the process of being performed.
        /// Thus, this state is used to indicate that the client has voted to initiate at least one collective communications operation and
        /// may or may not be currently in the process of performing one or more collective communications operations after a successful vote.
        /// As long as there is at least one collective communications operations running, the client will remain in this state.
        /// New collective communications operation votes can be started when in this state, and votes can be finalized in this state.
        /// However, in this state, the client cannot transition to any other state until all collective communications operations have completed.
        /// Because more fine-grained state tracking is required for collective communications operations, the @code CollectiveCommunicationState@endcode enum is used
        /// to indicate the state of each collective communications operation per tag.
        COLLECTIVE_COMMUNICATIONS_RUNNING
    };

    enum CollectiveCommunicationState {
        /// The client has voted to initiate a collective communications operation with a particular tag.
        /// In this state, the client waits for all peers to vote to initiate a collective communications operation for this tag.
        /// After each client has voted to initiate a collective communications operation for this tag, the client will enter the
        /// @code PERFORM_ALL_REDUCE@endcode state for this tag.
        /// A client can be in multiple @code AllReduceState@endcode states per tag, one for each collective communications operation.
        VOTE_INITIATE_COLLECTIVE_COMMS,

        /// When all peers have voted to initiate a collective communications operation for a particular tag, the client will enter the
        /// @code PERFORM_ALL_REDUCE@endcode state for this tag.
        /// In this state, clients will perform p2p communication to perform the collective communications operation according to the
        /// topology defined by the master node.
        PERFORM_COLLECTIVE_COMMS,

        /// After a peer has performed the work necessary to complete the collective communication operation for itself,
        /// it will vote to complete the collective communications operation.
        /// Once all peers have voted to complete the collective communications operation, the collective communications operation
        /// is considered complete and fully performed by all peers.
        /// Once all peers have voted to complete the collective communications operation,
        /// there will be no collective communications state associated with this tag until a new collective communications operation is initiated.
        VOTE_COMPLETE_COLLECTIVE_COMMS
    };

    struct CCoIPClientVariablePorts {
        /// The port the peer listens to for p2p connections
        uint16_t p2p_listen_port;

        /// The port the peer listens to for shared state distribution requests
        uint16_t shared_dist_state_listen_port;
    };

    struct ClientInfo {
        ccoip_uuid_t client_uuid{};
        ConnectionPhase connection_phase = PEER_REGISTERED;
        ConnectionState connection_state = IDLE;
        std::unordered_map<uint64_t, CollectiveCommunicationState> collective_coms_states{};

        ccoip_socket_address_t socket_address{};
        CCoIPClientVariablePorts variable_ports{};
        uint32_t peer_group{};
    };

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

        /// set of all uuids that have established p2p connections.
        /// cleared once all clients have established p2p connections.
        /// Removed from on client leave/disconnect.
        std::unordered_set<ccoip_uuid_t> votes_p2p_connections_established{};

        /// set of all uuids that have voted to synchronize shared state for each peer group.
        /// Peer group bin is cleared once shared state distribution consensus is reached.
        /// Removed from on client leave/disconnect.
        std::unordered_map<uint32_t, std::unordered_set<ccoip_uuid_t> > votes_sync_shared_state{};

        /// set of all uuids that have voted to complete shared state distribution for each peer group.
        /// Peer group bin is cleared once the shared state distribution phase ends.
        /// Removed from on client leave/disconnect.
        std::unordered_map<uint32_t, std::unordered_set<ccoip_uuid_t> > votes_sync_shared_state_complete{};

        /// Flag to indicate if the peer list has changed
        /// See @code hasPeerListChanged@endcode
        bool peer_list_changed = false;

        /// Defines the "mask" of shared state entries for each peer group.
        /// The mask is defined by the identity of the set of shared state key strings and their corresponding hashes.
        /// This mask is checked against the by clients synchronizing shared state.
        /// Mismatch of the keys results in the client being kicked.
        /// Mismatch of a hash results will result in the client being notified to re-request the shared state entry
        /// whose hash does not match.
        /// This "mask" is elected by popularity from a set of candidates provided by
        /// each peer that has voted to synchronize shared state.
        /// Peer group bin is cleared once the shared state distribution phase ends.
        std::unordered_map<uint32_t, std::vector<SharedStateHashEntry> > shared_state_mask{};

        /// Defines the list of candidates for the shared state mask for each peer group.
        /// Most popular candidate is elected as the shared state mask.
        /// Peer group bin is cleared after the shared state distribution phase ends.
        /// Removed from on client leave/disconnect.
        std::unordered_map<uint32_t, std::vector<std::pair<ccoip_uuid_t, std::vector<SharedStateHashEntry> > > >
        shared_state_mask_candidates{};

        /// Cache of the shared state hashes for all entries in the shared state mask for each peer group.
        /// by their key string.
        /// Peer group bin is cleared when the shared state distribution phase ends.
        std::unordered_map<uint32_t, std::unordered_map<std::string, uint64_t> > shared_state_hashes{};

        /// The next shared state revision that is expected from peers synchronizing shared state, intending to distribute.
        /// Shared state revisions that do not match this value will either be considered outdated or a violation of the
        /// one-increment-only rule.
        /// This value is stored per peer group. Each peer group maintains their own shared state, and thus their own revision.
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

        /// Maps the client UUID to the shared state mismatch status populated by the last invocation of @code sharedStateMatches@endcode.
        /// Grouped by peer group.
        /// Peer group bin is cleared when the shared state distribution phase ends.
        std::unordered_map<uint32_t, std::unordered_map<ccoip_uuid_t, SharedStateMismatchStatus> > shared_state_statuses
                {};

        /// Maps the client UUID to the set of shared state keys that have dirty content, meaning
        /// that the hash of the shared state entry does not match the hash in the shared state mask.
        /// Grouped by peer group.
        /// Peer group bin is cleared when the shared state distribution phase ends.
        std::unordered_map<uint32_t, std::unordered_map<ccoip_uuid_t, std::vector<std::string> > >
        shared_state_dirty_keys{};

        /// Boolean state for each collective communications operation tag in every peer group that indicates if the operation has been aborted or not.
        /// true = aborted
        /// false = not aborted
        /// Cleared when the collective communications operation ends.
        std::unordered_map<uint32_t, std::unordered_map<uint64_t, bool> > collective_comms_op_abort_states{};

    public:
        /// Registers a client
        [[nodiscard]] auto registerClient(const ccoip_socket_address_t &client_address,
                                          const CCoIPClientVariablePorts &variable_ports,
                                          uint32_t peer_group,
                                          ccoip_uuid_t uuid) -> bool;

        /// Unregisters a client
        [[nodiscard]] bool unregisterClient(const ccoip_socket_address_t &client_address);

        /// Checks if a client is registered
        [[nodiscard]] bool isClientRegistered(const ccoip_socket_address_t &client_address) const;

        /// Called when a client votes to accept new peers.
        /// All clients must vote to accept new peers before all clients start the p2p establishment phase.
        [[nodiscard]] bool voteAcceptNewPeers(const ccoip_uuid_t &peer_uuid);

        /// Called when a client votes to synchronize shared state.
        /// All clients must vote to synchronize shared state before shared state distribution can begin.
        [[nodiscard]] bool voteSyncSharedState(const ccoip_uuid_t &peer_uuid);

        /// Called when a client votes to complete the shared state distribution phase.
        /// This indicates that it has completed receiving the subset of shared state that was outdated.
        /// All clients must vote to complete shared state distribution before shared state distribution can end.
        /// With the shared state distribution ending, the shared state synchronization phase also ends.
        /// When consensus is reached, @code endSharedStateDistributionPhase()@endcode will be called.
        /// All clients here shall mean the subset of clients that are in the @code PEER_ACCEPTED@endcode phase.
        [[nodiscard]] bool voteDistSharedStateComplete(const ccoip_uuid_t &peer_uuid);

        /// Called when a client votes to initiate a collective communications operation with a particular tag.
        /// @param peer_uuid the uuid of the client that is initiating the collective communications operation. The peer group of said client determines what peers will participate in the collective communications operation.
        /// @param tag the tag associated with the collective communications operation.
        /// All clients must vote to initiate a collective communications operation before the operation can begin.
        /// Clients can concurrently vote to initiate and perform multiple collective communications operations.
        [[nodiscard]] bool voteCollectiveCommsInitiate(const ccoip_uuid_t &peer_uuid, uint64_t tag);

        /// Called when a client votes to complete a collective communications operation with a particular tag.
        /// @param peer_uuid the uuid of the client that is completing the collective communications operation. The peer group of said client determines what peers will participate in the collective communications operation.
        /// @param tag the tag associated with the collective communications operation.
        /// All clients must vote to complete a collective communications operation before the operation can end.
        [[nodiscard]] bool voteCollectiveCommsComplete(const ccoip_uuid_t &peer_uuid, uint64_t tag);

        /// Returns true if all peers of the peer group have voted to synchronize the shared state
        /// All clients here shall mean the subset of clients that are in the @code PEER_ACCEPTED@endcode phase.
        [[nodiscard]] bool syncSharedStateConsensus(uint32_t peer_group);

        /// Returns true if all clients have voted to accept new peers
        [[nodiscard]] bool acceptNewPeersConsensus() const;

        /// Transition to the p2p establishment phase
        /// Triggered after unanimous voting to accept new peers
        void transitionToP2PEstablishmentPhase();

        /// Marks that p2p connections have been established by a particular client.
        /// The particular client will be transitioned to the @code WAITING_FOR_OTHER_PEERS@endcode state
        /// until all peers have declared that they have established p2p connections.
        ///
        /// Once all peers have declared that they have established p2p connections,
        /// clients return back to the @code IDLE@endcode state.
        [[nodiscard]] bool markP2PConnectionsEstablished(const ccoip_uuid_t &peer_uuid);

        /// Transition to the p2p connections established phase
        /// Triggered after all clients have declared that they have established p2p connections.
        /// This means that all clients will return to the @code IDLE@endcode state.
        /// Returns false if any client is not in the @code WAITING_FOR_OTHER_PEERS@endcode state
        [[nodiscard]] bool transitionToP2PConnectionsEstablishedPhase();

        /// Transition to the shared state distribution phase
        /// Triggered after all peers of a peer group have voted to synchronize shared state
        /// Returns false if a client is in the PEER_ACCEPTED phase but not in the VOTE_SYNC_SHARED_STATE state or if
        /// @code sharedStateMatches@endcode has not been called for all clients.
        [[nodiscard]] bool transitionToSharedStateSyncPhase(uint32_t peer_group);

        /// Called to end the shared state distribution phase
        /// Will revert all peers of the peer group to the @code IDLE@endcode state
        [[nodiscard]] bool endSharedStateSyncPhase(uint32_t peer_group);

        /// Returns true if all clients have declared that they have established p2p connections
        /// Note: that "all clients" here means the subset of clients that are joining in the current acceptNewPeers phase.
        /// E.g. if a peer joins the session after consensus has been reached to accept new peers, it will not be included in this check.
        [[nodiscard]] bool p2pConnectionsEstablishConsensus() const;

        /// Returns true if all peers of the peer group have voted to complete the shared state distribution phase
        [[nodiscard]] bool syncSharedStateCompleteConsensus(uint32_t peer_group);

        /// Returns true if all peers of the peer group have voted to initiate a collective communications operation for this tag
        [[nodiscard]] bool collectiveCommsInitiateConsensus(uint32_t peer_group, uint64_t tag);

        /// Returns true if all peers of the peer group have voted to complete a collective communications operation for this tag
        [[nodiscard]] bool collectiveCommsCompleteConsensus(uint32_t peer_group, uint64_t tag);

        /// Transition all clients of the given peer group to the phase of performing the collective communications operation with the given tag/
        /// Returns false if the client is not in the @code COLLECTIVE_COMMUNICATIONS_RUNNING@endcode state or if
        /// not in the @code VOTE_INITIATE_COLLECTIVE_COMMS@endcode state for the specified tag.
        [[nodiscard]] bool transitionToPerformCollectiveCommsPhase(uint32_t peer_group, uint64_t tag);

        /// Transition all clients of the given peer group to the phase of completing the collective communications operation with the given tag.
        /// Returns false if the client is not in the @code COLLECTIVE_COMMUNICATIONS_RUNNING@endcode state or if
        /// not in the @code VOTE_COMPLETE_COLLECTIVE_COMMS@endcode state for the specified tag.
        [[nodiscard]] bool transitionToCollectiveCommsCompletePhase(uint32_t peer_group, uint64_t tag);

        /// Aborts the specified collective communications operation with the given tag for the given peer group.
        /// Only legal to call when all clients are in the @code COLLECTIVE_COMMUNICATIONS_RUNNING@endcode state.
        /// Abort of a collective communications operation represents separate state from the current state that being @code COLLECTIVE_COMMUNICATIONS_RUNNING@endcode.
        /// Even when collective communications operations are aborted, clients will remain in the @code COLLECTIVE_COMMUNICATIONS_RUNNING@endcode state
        /// and vote to complete the collective communications operation once the abort signal was handled and the collective communications operation thus "completes" as per the protocol.
        /// Returns true if the collective communications operation was not in the abort state before this call.
        /// Returns false if the collective communications operation was already aborted by a previous call to this function.
        /// This state is accessible via @code isCollectiveCommsOperationAborted@endcode.
        [[nodiscard]] bool abortCollectiveCommsOperation(uint32_t peer_group, uint64_t tag);

        /// Returns true if the specified collective communications operation with the given tag for the given peer group has been aborted.
        [[nodiscard]] bool isCollectiveCommsOperationAborted(uint32_t peer_group, uint64_t tag) const;

        /// @returns SharedStateMismatchStatus::SUCCESSFUL_MATCH if the specified revision is legal as the next
        /// shared state revision for the given peer group.
        /// @returns @code SharedStateMismatchStatus::REVISION_OUTDATED@endcode if the specified revision is less than the next expected revision.
        /// @returns @code SharedStateMismatchStatus::REVISION_INCREMENT_VIOLATION@endcode if the specified revision is greater than the next expected revision.
        /// The next expected shared state revision is always one larger than the previous shared state revision.
        /// This status will be retained until the shared state voting phase ends and can be queried
        /// via @code getSharedStateMismatchStatus@endcode.
        [[nodiscard]] SharedStateMismatchStatus isNewRevisionLegal(
            const ccoip_uuid_t &peer_uuid,
            uint64_t revision);

        /// Returns true if the shared state entries provided match the current "mask".
        /// A "mask" is defined by the identity of the set of shared state key strings and their corresponding hashes.
        /// If @code ignore_hashes@endcode is true, only the shared state keys are compared.
        /// If this is the first client to sync shared state, then the supplied shared state will define the "mask"
        /// for subsequent checks.
        /// The "mask" is elected by popularity among peers. The most popular "mask" determines which peers
        /// are considered to have outdated or illegal shared state.
        /// This status will be retained until the shared state voting phase ends and can be queried
        /// via @code getSharedStateMismatchStatus@endcode.
        [[nodiscard]] SharedStateMismatchStatus sharedStateMatches(
            const ccoip_uuid_t &peer_uuid,
            const std::vector<SharedStateHashEntry> &entries);

        /// Votes for a specific shared state mask; Called for each peer that has voted to synchronize shared state.
        /// After all clients have voted to synchronize shared state, the elected shared state mask will be chosen by popularity.
        void voteSharedStateMask(const ccoip_uuid_t &peer_uuid, const std::vector<SharedStateHashEntry> &entries);

        /// Returns the peers that a particular client should establish p2p connections with
        [[nodiscard]] std::vector<ClientInfo> getPeersForClient(
            const ccoip_socket_address_t &client_address
        ) const;

        /// Returns all client socket addresses
        [[nodiscard]] std::vector<ccoip_socket_address_t> getClientSocketAddresses();

        /// Returns a list pairs of client socket addresses and their corresponding UUIDs
        [[nodiscard]] std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t> > getClientEntrySet();

        /// Returns true if the peer list has changed since the last invocation of this function
        [[nodiscard]] bool hasPeerListChanged();

        /// Returns the shared state mismatch status. This status is set by @code sharedStateMatches@endcode.
        /// If @code sharedStateMatches@endcode has not been called yet since the start of the current shared state voting phase,
        /// this function will return std::nullopt.
        [[nodiscard]] std::optional<SharedStateMismatchStatus> getSharedStateMismatchStatus(
            const ccoip_uuid_t &peer_uuid);

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

        /// Finds the client UUID from the client address; returns std::nullopt if not found
        [[nodiscard]] std::optional<ccoip_uuid_t> findClientUUID(const ccoip_socket_address_t &client_address);

        /// Returns the client info for a particular client uuid; returns std::nullopt if not found
        [[nodiscard]] std::optional<std::reference_wrapper<ClientInfo> > getClientInfo(const ccoip_uuid_t &client_uuid);

        /// Returns the current shared state revision for the given peer group. This represents the current maximum revision of the shared state
        /// that all clients have agreed upon to be the current shared state revision.
        [[nodiscard]] uint64_t getSharedStateRevision(uint32_t peer_group);

        /// Returns the set of shared state keys of a particular peer that have dirty content, meaning that the hash of the shared state entry
        /// does not match the hash in the shared state mask.
        /// If @code sharedStateMatches@endcode has not been called yet since the start of the current shared state voting phase,
        /// this function will return an empty vector.
        [[nodiscard]] std::vector<std::string> getOutdatedSharedStateKeys(ccoip_uuid_t peer_uuid);

        /// Returns the shared state entry hash for a particular key; returns 0 if the key is not found
        /// For a key to be found, it must be present in the shared state mask.
        /// The shared state mask needs to be populated by @code sharedStateMatches@endcode before calling this function.
        [[nodiscard]] uint64_t getSharedStateEntryHash(uint32_t peer_group, const std::string &key);

        /// Returns the set of shared state keys
        [[nodiscard]] std::vector<std::string> getSharedStateKeys(uint32_t peer_group);

        /// Returns the set of ongoing collective communications operation tags for a particular peer group
        [[nodiscard]] std::vector<uint64_t> getOngoingCollectiveComsOpTags(uint32_t peer_group);
    };
}
