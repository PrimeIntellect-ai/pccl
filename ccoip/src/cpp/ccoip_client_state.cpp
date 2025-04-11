#include "ccoip_client_state.hpp"

#include <ccoip_inet_utils.hpp>
#include <pccl_log.hpp>
#include <thread_guard.hpp>

ccoip::CCoIPClientState::CCoIPClientState() { collective_coms_threadpool.startup(); }

ccoip::CCoIPClientState::~CCoIPClientState() {
    for (const std::unordered_set<uint64_t> remaining_tags = running_collective_coms_ops_tags;
         const auto &tag: remaining_tags) {
        if (!joinAsyncCollectiveOp(tag)) {
            LOG(ERR) << "Failed to join collective comms op with tag " << tag << " during client state destruction";
        }
    }
    collective_coms_threadpool.shutdown();
}

void ccoip::CCoIPClientState::setMainThread(const std::thread::id main_thread_id) {
    this->main_thread_id = main_thread_id;
}

std::vector<uint64_t> ccoip::CCoIPClientState::getRunningCollectiveComsOpTags() {
    std::vector<uint64_t> running_collective_coms_op_tags; {
        std::shared_lock lock(running_collective_coms_ops_tags_mutex);
        running_collective_coms_op_tags.reserve(running_collective_coms_ops_tags.size());
        for (const auto tag: running_collective_coms_ops_tags) {
            running_collective_coms_op_tags.push_back(tag);
        }
    }
    return running_collective_coms_op_tags;
}

bool ccoip::CCoIPClientState::registerPeer(const ccoip_socket_address_t &address, const ccoip_uuid_t uuid) {
    // intentionally no thread guard here
    std::unique_lock lock{socket_addr_to_uuid_mutex};
    const auto internal_address = ccoip_socket_to_internal(address);
    if (const auto it = socket_addr_to_uuid.find(internal_address); it != socket_addr_to_uuid.end()) {
        return false;
    }
    socket_addr_to_uuid[internal_address] = uuid;
    return true;
}

bool ccoip::CCoIPClientState::unregisterPeer(const ccoip_socket_address_t &address) {
    // intentionally no thread guard here
    std::unique_lock lock{socket_addr_to_uuid_mutex};
    const auto internal_address = ccoip_socket_to_internal(address);
    if (const auto n = socket_addr_to_uuid.erase(internal_address); n == 0) {
        return false;
    }
    return true;
}

void ccoip::CCoIPClientState::setAssignedUUID(const ccoip_uuid_t &new_assigned_uuid) {
    THREAD_GUARD(main_thread_id);
    if (assigned_uuid.data != ccoip_uuid{}) {
        throw std::runtime_error("Can only set assigned UUID once!");
    }
    assigned_uuid = new_assigned_uuid;
}

void ccoip::CCoIPClientState::setGlobalWorldSize(const size_t new_global_world_size) {
    global_world_size.store(new_global_world_size, std::memory_order_release);
}

void ccoip::CCoIPClientState::setLocalWorldSize(const size_t new_local_world_size) {
    local_world_size.store(new_local_world_size, std::memory_order_release);
}

void ccoip::CCoIPClientState::setNumDistinctPeerGroups(const size_t new_num_distinct_peer_groups) {
    num_distinct_peer_groups.store(new_num_distinct_peer_groups, std::memory_order_release);
}

void ccoip::CCoIPClientState::setLargestPeerGroupWorldSize(const size_t new_largest_peer_group_world_size) {
    largest_peer_group_world_size.store(new_largest_peer_group_world_size, std::memory_order_release);
}

const ccoip_uuid_t &ccoip::CCoIPClientState::getAssignedUUID() const {
    // may be called concurrently, but is only set once during initialization
    return assigned_uuid;
}

void ccoip::CCoIPClientState::beginSyncSharedStatePhase(const ccoip_shared_state_t &shared_state) {
    THREAD_GUARD(main_thread_id);

    std::unique_lock lock(current_shared_state_mutex);
    current_shared_state = shared_state;
    is_syncing_shared_state.store(true, std::memory_order_release);
}

void ccoip::CCoIPClientState::endSyncSharedStatePhase() {
    THREAD_GUARD(main_thread_id);

    std::unique_lock lock(current_shared_state_mutex);
    current_shared_state = {};
    is_syncing_shared_state.store(false, std::memory_order_release);
}

bool ccoip::CCoIPClientState::isSyncingSharedState() const {
    return is_syncing_shared_state.load(std::memory_order_acquire);
}

bool ccoip::CCoIPClientState::isCollectiveComsOpRunning(const uint64_t tag) {
    std::shared_lock lock(running_collective_coms_ops_tags_mutex);
    return running_collective_coms_ops_tags.contains(tag);
}

bool ccoip::CCoIPClientState::isAnyCollectiveComsOpRunning() {
    std::shared_lock lock(running_collective_coms_ops_tags_mutex);
    return !running_collective_coms_ops_tags.empty();
}

/// !!! A WORD OF CAUTION w.r.t getLocalWorldSize and getGlobalWorldSize !!!
///
/// NOTE: getLocalWorldSize & getGlobalWorldSize are NOT guarded by THREAD_GUARD.
/// That doesn't mean that there isn't an implied contract with the caller here though.
/// The contract is that the local_world_size and global_world_size are only set
/// by the same thread that will actually call these methods.
/// Technically, the atomics here are unnecessary, given that this is the implied contract, as
/// no tearing could occur in the absence of concurrent access.
/// However, we still employ it as a defensive measure here that we don't actually rely on.
/// The library user however can break this contract with get attribute calls, so it is useful
/// to not return garbage values, even if the strict veracity of those values is not guaranteed.
/// Veracity here shall mean up-to-date-ness. Value changes may not be instantly visible to the reading thread,
/// if the writing thread is concurrent.
/// This should not happen anyways though.
/// The way you are *meant* to use these methods is that you only ever query them on the thread that is expected
/// to modify the values.
///
/// How are they modified?
/// setLocalWorldSize & setGlobalWorldSize will be called when p2p connections are (re)-established.
/// This can happen as a side effect of failed collective communications operations.
///
/// NOTE: The user may launch and await collective communications operations on any thread, while
/// other operations are only allowed on the main thread.
///
/// While a concurrent thread is performing collective communications operations,
/// the main thread MUST NOT CALL ANY user facing functions EXCEPT @ref pcclAreNewPeersPending.
/// The @ref pcclAreNewPeersPending function is an exception because it will only communicate with the master.
/// P2P connections are neither used nor potentially invalidated by this function.
///
/// This function is necessary to be called on the main thread to enable cross-step communication overlap.
/// If this function returns false, then update topology may be skipped, allowing us to not await the ongoing
/// all reduces while we launch new computation.
///
/// If this function returns true, we must await the ongoing all reduces before we can proceed.
/// HOWEVER, this does not mean simply calling await on the outstanding handles by the main thread when
/// the handles were launched by another thread. It specifically means that we await the work ongoing
/// by said concurrent thread to finish doing its business which itself will call await on the outstanding
/// handles. No guarantees are made about whether awaiting the handles launched by a different thread is safe.
size_t ccoip::CCoIPClientState::getLocalWorldSize() const { return local_world_size.load(std::memory_order_acquire); }

size_t ccoip::CCoIPClientState::getGlobalWorldSize() const { return global_world_size.load(std::memory_order_acquire); }

size_t ccoip::CCoIPClientState::getNumDistinctPeerGroups() const {
    return num_distinct_peer_groups.load(std::memory_order_acquire);
}

size_t ccoip::CCoIPClientState::getLargestPeerGroupWorldSize() const {
    return largest_peer_group_world_size.load(std::memory_order_acquire);
}

bool ccoip::CCoIPClientState::startCollectiveComsOp(const uint64_t tag) {
    std::unique_lock guard{running_collective_coms_ops_tags_mutex};
    if (running_collective_coms_ops_tags.contains(tag)) {
        return false;
    }
    running_collective_coms_ops_tags.insert(tag);
    return true;
}

bool ccoip::CCoIPClientState::endCollectiveComsOp(const uint64_t tag) {
    // no lock here (!!), caller holds lock asserted!
    if (const auto n = running_collective_coms_ops_tags.erase(tag); n == 0) {
        return false;
    }
    if (const auto n = running_collective_ops.erase(tag); n == 0) {
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientState::launchAsyncCollectiveOp(const uint64_t tag,
                                                      std::function<void(std::promise<bool> &)> &&task) {
    // NOTE: intentionally no thread guard here!

    if (!startCollectiveComsOp(tag)) [[unlikely]] {
        return false;
    }
    resetCollectiveComsTxBytes(tag);
    resetCollectiveComsRxBytes(tag);

    // retain applicable world size at the time of launch
    {
        std::unique_lock lock(running_collective_coms_ops_world_size_mutex);
        running_collective_coms_ops_world_size[tag] = getLocalWorldSize();
    }

    // set initial failure state to not completed
    {
        std::unique_lock lock(running_reduce_tasks_failure_states_mutex);
        running_reduce_tasks_failure_states[tag].store(2, std::memory_order_relaxed); // not completed
    }

    auto op_future = collective_coms_threadpool.scheduleTask([this, tag, task = std::move(task)] {
        std::promise<bool> promise{};
        task(promise);
        std::future<bool> future = promise.get_future();

        // expose the result of the task to the client state
        {
            std::unique_lock lock(running_reduce_tasks_failure_states_mutex);
            running_reduce_tasks_failure_states[tag].store(future.get() ? 1 : 0, std::memory_order_release);
        }
    });
    running_collective_ops.emplace(tag, std::move(op_future));
    return true;
}

bool ccoip::CCoIPClientState::joinAsyncCollectiveOp(const uint64_t tag) {
    // note: intentionally no thread guard here
    std::unique_lock guard{running_collective_coms_ops_tags_mutex};
    if (const auto it = running_collective_ops.find(tag); it != running_collective_ops.end()) {
        it->second.join();
        if (!endCollectiveComsOp(tag)) [[unlikely]] {
            LOG(BUG) << "Collective comms op with tag " << tag << " was not started but is being ended";
        }
        return true;
    }
    return false;
}

std::optional<bool> ccoip::CCoIPClientState::hasCollectiveComsOpFailed(const uint64_t tag) {
    // note: intentionally no thread guard here
    std::shared_lock lock(running_reduce_tasks_failure_states_mutex);
    if (const auto it = running_reduce_tasks_failure_states.find(tag);
        it != running_reduce_tasks_failure_states.end()) {
        const auto &state = it->second;
        return state.load() == 0;
    }
    return std::nullopt;
}

const ccoip_shared_state_t &ccoip::CCoIPClientState::getCurrentSharedState() {
    std::shared_lock lock(current_shared_state_mutex);
    return current_shared_state;
}


size_t ccoip::CCoIPClientState::getSharedStateSyncTxBytes() const {
    THREAD_GUARD(main_thread_id);
    return shared_state_sync_tx_bytes;
}

void ccoip::CCoIPClientState::trackSharedStateTxBytes(const size_t tx_bytes) {
    shared_state_sync_tx_bytes.fetch_add(tx_bytes, std::memory_order_relaxed);
}

void ccoip::CCoIPClientState::resetSharedStateSyncTxBytes() {
    shared_state_sync_tx_bytes.store(0, std::memory_order_release);
}

std::optional<size_t> ccoip::CCoIPClientState::getCollectiveComsRxBytes(const uint64_t tag) {
    std::shared_lock lock(collective_coms_rx_bytes_mutex);
    const auto it = collective_coms_rx_bytes.find(tag);
    if (it == collective_coms_rx_bytes.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ccoip::CCoIPClientState::setCollectiveConnectionRevision(const uint64_t tag, const uint64_t revision) {
    std::unique_lock lock(collective_coms_connection_revisions_mutex);
    collective_coms_connection_revisions[tag] = revision;
}

std::optional<uint64_t> ccoip::CCoIPClientState::getCollectiveConnectionRevision(const uint64_t tag) {
    std::shared_lock lock(collective_coms_connection_revisions_mutex);
    const auto it = collective_coms_connection_revisions.find(tag);
    if (it == collective_coms_connection_revisions.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ccoip::CCoIPClientState::trackCollectiveComsRxBytes(const uint64_t tag, const size_t rx_bytes) {
    std::unique_lock lock(collective_coms_rx_bytes_mutex);
    collective_coms_rx_bytes[tag].fetch_add(rx_bytes, std::memory_order_relaxed);
}

void ccoip::CCoIPClientState::resetCollectiveComsRxBytes(const uint64_t tag) {
    std::unique_lock lock(collective_coms_rx_bytes_mutex);
    collective_coms_rx_bytes.erase(tag);
}

std::optional<size_t> ccoip::CCoIPClientState::getCollectiveComsTxBytes(const uint64_t tag) {
    std::shared_lock lock(collective_coms_tx_bytes_mutex);
    const auto it = collective_coms_tx_bytes.find(tag);
    if (it == collective_coms_tx_bytes.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<uint32_t> ccoip::CCoIPClientState::getCollectiveComsWorldSize(const uint64_t tag) {
    std::shared_lock lock(running_collective_coms_ops_world_size_mutex);
    const auto it = running_collective_coms_ops_world_size.find(tag);
    if (it == running_collective_coms_ops_world_size.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ccoip::CCoIPClientState::resetCollectiveComsWorldSize(const uint64_t tag) {
    std::unique_lock lock(running_collective_coms_ops_world_size_mutex);
    running_collective_coms_ops_world_size.erase(tag);
}

void ccoip::CCoIPClientState::trackCollectiveComsTxBytes(const uint64_t tag, const size_t tx_bytes) {
    std::unique_lock lock(collective_coms_tx_bytes_mutex);
    collective_coms_tx_bytes[tag] += tx_bytes;
}

void ccoip::CCoIPClientState::resetCollectiveComsTxBytes(const uint64_t tag) {
    std::unique_lock lock(collective_coms_tx_bytes_mutex);
    collective_coms_tx_bytes.erase(tag);
}

void ccoip::CCoIPClientState::updateTopology(const std::vector<ccoip_uuid_t> &new_ring_order) {
    ring_order = new_ring_order;
}
