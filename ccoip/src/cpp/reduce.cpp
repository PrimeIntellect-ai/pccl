#include "reduce.hpp"

#include <algorithm>
#include <cassert>
#include <ccoip_master_state.hpp>
#include <ccoip_packets.hpp>
#include <ccoip_types.hpp>
#include <cstring>
#include <numeric>
#include <quantize.hpp>
#include <reduce_kernels.hpp>
#include <threadpark.h>
#include <tinysockets.hpp>
#include <win_sock_bridge.h>

/// Chunk size passed to the send() function of the MultiplexedIOSocket.
/// This determines the maximum size of a single local tagged chunk as managed by the multiplexer.
/// The higher the chunk size, the less overhead is incurred by the multiplexer, but it is also less fine-granular.
/// The reduce algorithm also uses this chunk size to quantize chunks as they are received. Higher chunk sizes means
/// quantization runs less frequently but potentially for longer.
#define DEFAULT_MULTIPLEX_CHUNK_SIZE size_t(8388608ull * 8)


namespace {
    size_t GetPCCLMultiplexChunkSize() {
        static size_t chunk_size = SIZE_MAX;
        if (chunk_size == SIZE_MAX) {
            // check PCCL_MULTIPLEX_CHUNK_SIZE environment variable
            const char *env_chunk_size = std::getenv("PCCL_MULTIPLEX_CHUNK_SIZE");
            if (env_chunk_size) {
                try {
                    chunk_size = std::stoull(env_chunk_size);
                } catch (...) {
                    LOG(ERR) << "Invalid value for environment variable PCCL_MULTIPLEX_CHUNK_SIZE: " << env_chunk_size;
                    chunk_size = DEFAULT_MULTIPLEX_CHUNK_SIZE;
                }
            } else {
                chunk_size = DEFAULT_MULTIPLEX_CHUNK_SIZE;
            }
        }
        return chunk_size;
    }

    /**
     * \brief Utility to compute per-rank chunk boundaries for an array
     *        of \p total_el elements across \p world_size ranks.
     *
     * Returns a vector of length \p world_size, where
     *   boundaries[r] = {start_index, end_index}
     * in element (not byte) units.
     */
    std::vector<std::pair<size_t, size_t>> computeChunkBoundaries(const size_t total_el, const size_t world_size) {
        std::vector<std::pair<size_t, size_t>> boundaries(world_size, {0, 0});
        if (world_size == 0) {
            return boundaries;
        }

        const size_t base = total_el / world_size;
        const size_t remainder = total_el % world_size;
        size_t current_start = 0;

        for (size_t r = 0; r < world_size; ++r) {
            const size_t chunk_size = base + ((r < remainder) ? 1 : 0);
            boundaries[r] = {current_start, current_start + chunk_size};
            current_start += chunk_size;
        }
        return boundaries;
    }

    /**
     * \brief Utility to create a subspan of a span, with bounds checking.
     */
    std::span<const std::byte> guarded_subspan(const std::span<const std::byte> &span, const size_t offset,
                                               const size_t count) {
        if (offset > span.size()) {
            throw std::out_of_range("Subspan out of range");
        }
        if (offset + count > span.size()) {
            throw std::out_of_range("Subspan out of range");
        }
        return span.subspan(offset, count);
    }

    /**
     * \brief Utility to create a subspan of a span, with bounds checking.
     */
    std::span<std::byte> guarded_subspan(const std::span<std::byte> &span, const size_t offset, const size_t count) {
        if (offset > span.size()) {
            throw std::out_of_range("Subspan out of range");
        }
        if (offset + count > span.size()) {
            throw std::out_of_range("Subspan out of range");
        }
        return span.subspan(offset, count);
    }

    /**
     * \brief Utility to create a subspan of a span, with bounds checking.
     */
    std::span<std::byte> guarded_subspan(const std::span<std::byte> &span, const size_t offset) {
        if (offset > span.size()) {
            throw std::out_of_range("Subspan out of range");
        }
        return span.subspan(offset);
    }

    /**
     * \brief Perform the "reduce-scatter" stage of the ring in one step (send one chunk / receive one chunk / reduce).
     *
     * - Sends the \p tx_span (possibly quantized) to the next rank.
     * - Receives from the previous rank into \p recv_buffer_span (which is sized exactly for the incoming chunk).
     * - De-quantizes and accumulates into \p rx_span.
     *
     * @return {success, abort_packet_received}
     */
    [[nodiscard]] std::pair<bool, bool> runReduceStage(
            ccoip::CCoIPClientState &client_state, tinysockets::QueuedSocket &master_socket, const uint64_t tag,
            const uint64_t seq_nr, const std::span<const std::byte> &tx_span, const std::span<std::byte> &rx_span,
            const std::span<std::byte> &recv_buffer_span,

            const ccoip::ccoip_data_type_t data_type, const ccoip::ccoip_data_type_t quantized_type,
            const ccoip::ccoip_quantization_algorithm_t quantization_algorithm, const ccoip::ccoip_reduce_op_t op,

            const size_t rank, const size_t world_size, const std::vector<ccoip_uuid_t> &ring_order,

            const std::optional<ccoip::internal::quantize::DeQuantizationMetaData> &meta_data_self,

            const std::unordered_map<ccoip_uuid_t, std::vector<std::shared_ptr<tinysockets::MultiplexedIOSocket>>>
                    &peer_tx_sockets,
            const std::unordered_map<ccoip_uuid_t, std::vector<std::shared_ptr<tinysockets::MultiplexedIOSocket>>>
                    &peer_rx_sockets) {
        using namespace tinysockets;
        using namespace ccoip::internal::reduce;
        using namespace ccoip::internal::quantize;

        // We allow tx_span.size_bytes() != recv_buffer_span.size_bytes() because
        // different chunks can have different sizes. Let’s track them separately:
        const size_t total_tx_size = tx_span.size_bytes();
        const size_t total_rx_size = recv_buffer_span.size_bytes();

        // In a ring, next rank is (rank+1) mod world_size, previous is (rank-1+world_size) mod world_size.
        const size_t rx_peer_idx = (rank + world_size - 1) % world_size;
        const size_t tx_peer_idx = (rank + 1) % world_size;
        const ccoip_uuid_t rx_peer = ring_order.at(rx_peer_idx);
        const ccoip_uuid_t tx_peer = ring_order.at(tx_peer_idx);

        const auto &tx_sockets = peer_tx_sockets.at(tx_peer);
        const auto &rx_sockets = peer_rx_sockets.at(rx_peer);

        // choose tx/rx socket based on seq nr
        const auto &tx_socket = tx_sockets.at(seq_nr % tx_sockets.size());
        const auto &rx_socket = rx_sockets.at(seq_nr % rx_sockets.size());

        // 1) Send our local de-quant metadata to next peer
        if (quantized_type != data_type) {
            ccoip::P2PPacketDequantizationMeta packet{};
            packet.tag = tag;
            packet.dequantization_meta = (meta_data_self ? *meta_data_self : DeQuantizationMetaData{});
            if (!tx_socket->sendPacket<ccoip::P2PPacketDequantizationMeta>(tag, seq_nr, packet)) {
                LOG(WARN) << "Failed to send de-quantization meta data!";
                return {false, false};
            }
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);

            // TODO: FIX THAT trackCollectiveComsTxBytes is called so often. We can batch this because it does lock onto
            client_state.trackCollectiveComsTxBytes(tag, ltv_header + packet.serializedSize());
        }

        // 2) Receive their metadata from the previous peer
        DeQuantizationMetaData received_meta_data{};
        if (quantized_type != data_type) {
            // wait until we receive a P2PPacketDequantizationMeta packet and wait for aborts in the meantime
            while (true) {
                const auto metadata_packet =
                        rx_socket->receivePacket<ccoip::P2PPacketDequantizationMeta>(tag, seq_nr, true);
                if (!metadata_packet) {
                    if (!rx_socket->isOpen()) {
                        return {false, false};
                    }
                    const auto abort_packet = master_socket.receiveMatchingPacket<ccoip::M2CPacketCollectiveCommsAbort>(
                            [tag](const ccoip::M2CPacketCollectiveCommsAbort &packet) { return packet.tag == tag; },
                            true);
                    if (abort_packet) {
                        return {true, true};
                    }
                    continue;
                }
                received_meta_data = metadata_packet->dequantization_meta;
                constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
                client_state.trackCollectiveComsRxBytes(tag, ltv_header + metadata_packet->serializedSize());
                break;
            }
        }

        // 3) Full-duplex send/recv loop
        size_t bytes_sent = 0;
        size_t bytes_recvd = 0;

        size_t no_event_ctr = 0;

        std::vector<tpark_handle_t *> tx_done_handles{};
        while (bytes_sent < total_tx_size || bytes_recvd < total_rx_size) {
            bool no_event = true;

            // 3a) Send if ready
            if (bytes_sent < total_tx_size) {
                const size_t chunk_size = std::min(GetPCCLMultiplexChunkSize(), total_tx_size - bytes_sent);
                const auto send_sub = guarded_subspan(tx_span, bytes_sent, chunk_size);
                tpark_handle_t *done_handle = nullptr;
                if (tx_socket->sendBytes(tag, seq_nr, send_sub, false, &done_handle)) {
                    no_event = false;
                    bytes_sent += send_sub.size_bytes();
                    client_state.trackCollectiveComsTxBytes(tag, send_sub.size_bytes());
                } else {
                    if (done_handle != nullptr) {
                        // tparkWait(done_handle, true);
                        tparkDestroyHandle(done_handle);
                    }
                    return {false, false};
                }
                tparkWait(done_handle, true);
                tparkDestroyHandle(done_handle);
            }

            // 3b) Receive if ready
            if (bytes_recvd < total_rx_size) {
                const auto recv_sub = guarded_subspan(recv_buffer_span, bytes_recvd);
                if (auto n_read = rx_socket->receiveBytesInplace(tag, seq_nr, recv_sub)) {
                    if (*n_read > 0) {
                        no_event = false;
                        client_state.trackCollectiveComsRxBytes(tag, *n_read);

                        const size_t quant_el_sz_bits = ccoip_data_type_size_bits(quantized_type);

                        // old_floor = how many *complete* quantized elements we had before in unit bits
                        const size_t old_floor_elem = (bytes_recvd * 8) / quant_el_sz_bits;
                        bytes_recvd += *n_read;

                        // new_floor = how many *complete* quantized elements we have now in unit bits
                        const size_t new_floor_elem = (bytes_recvd * 8) / quant_el_sz_bits;

                        // If new_floor > old_floor, we have at least one fully-received element to reduce
                        if (new_floor_elem > old_floor_elem) {
                            const size_t elem_inc = new_floor_elem - old_floor_elem;

                            // source (quantized) byte window
                            const size_t src_off_bytes = (old_floor_elem * quant_el_sz_bits) / 8;
                            const size_t src_size_bytes = (elem_inc * quant_el_sz_bits) / 8;
                            auto reduce_src_span = guarded_subspan(recv_buffer_span, src_off_bytes, src_size_bytes);

                            // destination (full-precision) byte window
                            const size_t data_type_el_sz_bits = ccoip_data_type_size_bits(data_type);
                            const size_t dst_off_bytes = (old_floor_elem * data_type_el_sz_bits) / 8;
                            const size_t dst_size_bytes = (elem_inc * data_type_el_sz_bits) / 8;
                            auto reduce_dst_span = guarded_subspan(rx_span, dst_off_bytes, dst_size_bytes);

                            // Accumulate newly arrived data into rx_span
                            performReduction(reduce_dst_span, reduce_src_span, data_type, quantized_type,
                                             quantization_algorithm, op, received_meta_data);
                        }
                    }
                } else {
                    return {false, false};
                }
            }

            if (no_event) {
                no_event_ctr++;
            } else {
                no_event_ctr = 0;
            }

            if (no_event_ctr > 100) {
                const auto abort_packet = master_socket.receiveMatchingPacket<ccoip::M2CPacketCollectiveCommsAbort>(
                        [tag](const ccoip::M2CPacketCollectiveCommsAbort &packet) { return packet.tag == tag; }, true);
                if (abort_packet) {
                    return {true, true};
                }
                no_event_ctr = 0;
            }
        }

        return {true, false};
    }

    /**
     * \brief Perform the "allgather" stage of the ring for one step (send chunk / receive chunk / copy in).
     *
     * This is like runReduceStage but without applying a reduce-op. Instead, we effectively copy the newly
     * arrived data into \p rx_span.
     *
     * @return {success, abort_packet_received}
     */
    [[nodiscard]] std::pair<bool, bool> runAllgatherStage(
            ccoip::CCoIPClientState &client_state, tinysockets::QueuedSocket &master_socket, const uint64_t tag,
            const uint64_t seq_nr, const std::span<const std::byte> &tx_span, const std::span<std::byte> &rx_span,
            const std::span<std::byte> &recv_buffer_span,

            const ccoip::ccoip_data_type_t data_type, const ccoip::ccoip_data_type_t quantized_type,
            const ccoip::ccoip_quantization_algorithm_t quantization_algorithm,

            const size_t rank, const size_t world_size, const std::vector<ccoip_uuid_t> &ring_order,

            const std::optional<ccoip::internal::quantize::DeQuantizationMetaData> &meta_data_self,
            ccoip::internal::quantize::DeQuantizationMetaData &received_meta_data_out,

            const std::unordered_map<ccoip_uuid_t, std::vector<std::shared_ptr<tinysockets::MultiplexedIOSocket>>>
                    &peer_tx_sockets,
            const std::unordered_map<ccoip_uuid_t, std::vector<std::shared_ptr<tinysockets::MultiplexedIOSocket>>>
                    &peer_rx_sockets) {
        using namespace tinysockets;
        using namespace ccoip::internal::quantize;
        using namespace ccoip::internal::reduce;

        const size_t total_tx_size = tx_span.size_bytes();
        const size_t total_rx_size = recv_buffer_span.size_bytes();

        const size_t rx_peer_idx = (rank + world_size - 1) % world_size;
        const size_t tx_peer_idx = (rank + 1) % world_size;

        const ccoip_uuid_t rx_peer = ring_order.at(rx_peer_idx);
        const ccoip_uuid_t tx_peer = ring_order.at(tx_peer_idx);

        const auto &tx_sockets = peer_tx_sockets.at(tx_peer);
        const auto &rx_sockets = peer_rx_sockets.at(rx_peer);

        // choose tx/rx socket based on seq nr
        const auto &tx_socket = tx_sockets.at(seq_nr % tx_sockets.size());
        const auto &rx_socket = rx_sockets.at(seq_nr % rx_sockets.size());

        // 1) Exchange metadata for consistency
        if (quantized_type != data_type) {
            ccoip::P2PPacketDequantizationMeta packet{};
            packet.tag = tag;
            packet.dequantization_meta = (meta_data_self ? *meta_data_self : DeQuantizationMetaData{});
            if (!tx_socket->sendPacket<ccoip::P2PPacketDequantizationMeta>(tag, seq_nr, packet)) {
                LOG(WARN) << "Failed to send de-quantization meta data in allgather!";
                return {false, false};
            }
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsTxBytes(tag, ltv_header + packet.serializedSize());
        }

        DeQuantizationMetaData received_meta_data{};
        if (quantized_type != data_type) {
            // wait until we receive a P2PPacketDequantizationMeta packet and wait for aborts in the meantime
            while (true) {
                const auto metadata_packet =
                        rx_socket->receivePacket<ccoip::P2PPacketDequantizationMeta>(tag, seq_nr, true);
                if (!metadata_packet) {
                    if (!rx_socket->isOpen()) {
                        return {false, false};
                    }
                    const auto abort_packet = master_socket.receiveMatchingPacket<ccoip::M2CPacketCollectiveCommsAbort>(
                            [tag](const ccoip::M2CPacketCollectiveCommsAbort &packet) { return packet.tag == tag; },
                            true);
                    if (abort_packet) {
                        return {true, true};
                    }
                    continue;
                }
                received_meta_data = metadata_packet->dequantization_meta;
                constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
                client_state.trackCollectiveComsRxBytes(tag, ltv_header + metadata_packet->serializedSize());
                break;
            }
        }
        received_meta_data_out = received_meta_data;

        // 2) Full-duplex send/recv
        size_t bytes_sent = 0;
        size_t bytes_recvd = 0;

        size_t no_event_ctr = 0;
        while (bytes_sent < total_tx_size || bytes_recvd < total_rx_size) {
            bool no_event = true;

            // Send
            if (bytes_sent < total_tx_size) {
                const size_t chunk_size = std::min(GetPCCLMultiplexChunkSize(), total_tx_size - bytes_sent);
                const auto send_sub = guarded_subspan(tx_span, bytes_sent, chunk_size);
                tpark_handle_t *done_handle{};
                if (tx_socket->sendBytes(tag, seq_nr, send_sub, false, &done_handle)) {
                    no_event = false;
                    bytes_sent += send_sub.size_bytes();
                    client_state.trackCollectiveComsTxBytes(tag, send_sub.size_bytes());
                } else {
                    if (done_handle != nullptr) {
                        // tparkWait(done_handle, true);
                        tparkDestroyHandle(done_handle);
                    }
                    return {false, false};
                }
                tparkWait(done_handle, true);
                tparkDestroyHandle(done_handle);
            }

            // Receive
            if (bytes_recvd < total_rx_size) {
                const auto recv_sub = guarded_subspan(recv_buffer_span, bytes_recvd);
                if (auto n_read = rx_socket->receiveBytesInplace(tag, seq_nr, recv_sub)) {
                    if (*n_read > 0) {
                        no_event = false;
                        client_state.trackCollectiveComsRxBytes(tag, *n_read);

                        const size_t quant_el_sz_bits = ccoip_data_type_size_bits(quantized_type);

                        // old_floor = how many *complete* quantized elements we had before
                        const size_t old_floor_elem = (bytes_recvd * 8) / quant_el_sz_bits;
                        bytes_recvd += *n_read;

                        // new_floor = how many *complete* quantized elements we have now
                        const size_t new_floor_elem = (bytes_recvd * 8) / quant_el_sz_bits;

                        if (new_floor_elem > old_floor_elem) {
                            const size_t elem_inc = new_floor_elem - old_floor_elem;

                            // De-quantize + copy into rx_span
                            const size_t src_off_bytes = (old_floor_elem * quant_el_sz_bits) / 8;
                            const size_t src_size_bytes = (elem_inc * quant_el_sz_bits) / 8;
                            auto copy_src_span = guarded_subspan(recv_buffer_span, src_off_bytes, src_size_bytes);

                            const size_t data_type_el_sz_bits = ccoip_data_type_size_bits(data_type);
                            const size_t dst_off_bytes = (old_floor_elem * data_type_el_sz_bits) / 8;
                            const size_t dst_size_bytes = (elem_inc * data_type_el_sz_bits) / 8;
                            auto copy_dst_span = guarded_subspan(rx_span, dst_off_bytes, dst_size_bytes);

                            // We do not accumulate for allgather, just "copy" via performReduction w/ ccoipOpSet
                            performReduction(copy_dst_span, copy_src_span, data_type, quantized_type,
                                             quantization_algorithm,
                                             ccoip::ccoipOpSet, // = copy
                                             received_meta_data);
                        }
                    }
                } else {
                    return {false, false};
                }
            }
            if (no_event) {
                no_event_ctr++;
            } else {
                no_event_ctr = 0;
            }
            if (no_event_ctr > 100) {
                const auto abort_packet = master_socket.receiveMatchingPacket<ccoip::M2CPacketCollectiveCommsAbort>(
                        [tag](const ccoip::M2CPacketCollectiveCommsAbort &packet) { return packet.tag == tag; }, true);
                if (abort_packet) {
                    return {true, true};
                }
                no_event_ctr = 0;
            }
        }
        return {true, false};
    }

#define POOLED_ALLOCATOR_MAX_ENTRIES 256
#define DISABLE_POOLED_ALLOCATOR 1

    class PooledAllocator {
        std::vector<std::pair<void *, size_t>> pool;
        std::mutex mutex;

    public:
        void *allocate(const size_t size) {
#if DISABLE_POOLED_ALLOCATOR
            return new std::byte[size];
#else
            std::unique_lock lock(mutex);
            for (auto it = pool.begin(); it != pool.end(); ++it) {
                if (it->second >= size) {
                    void *ptr = it->first;
                    pool.erase(it);
                    return ptr;
                }
            }
            auto *ptr = new std::byte[size];
            return ptr;
#endif
        }

        void release(const void *ptr, size_t size) {
#if DISABLE_POOLED_ALLOCATOR
            delete[] static_cast<const std::byte *>(ptr);
            return;
#else
            if (ptr == nullptr) {
                LOG(BUG) << "[reduce::PooledAllocator] Attempting to release nullptr";
                return;
            }
            // we trust the user to set size correctly; there is only one intended call-site anyways
            std::unique_lock lock(mutex);
            if (pool.size() >= POOLED_ALLOCATOR_MAX_ENTRIES) {
                const auto begin = pool.begin();
                delete[] static_cast<std::byte *>(begin->first);
                pool.erase(begin);
            }
            pool.emplace_back(const_cast<void *>(ptr), size);
#endif
        }

        ~PooledAllocator() {
            for (auto &[ptr, size]: pool) {
                delete[] static_cast<std::byte *>(ptr);
            }
            pool.clear();
        }
    };

#define MAX_FREE_LIST_SIZE 8

    struct allocator_free_list {
        void *ptrs[MAX_FREE_LIST_SIZE] = {};
        size_t sizes[MAX_FREE_LIST_SIZE] = {};
        PooledAllocator &allocator;

        void add(void *ptr, const size_t size) {
            for (size_t i = 0; i < MAX_FREE_LIST_SIZE; i++) {
                // NOLINT(*-loop-convert)
                if (ptrs[i] == nullptr) {
                    ptrs[i] = ptr;
                    sizes[i] = size;
                    break;
                }
            }
        }

        void remove(const void *ptr) {
            for (size_t i = 0; i < MAX_FREE_LIST_SIZE; i++) {
                // NOLINT(*-loop-convert)
                if (ptrs[i] == ptr) {
                    ptrs[i] = nullptr;
                    break;
                }
            }
        }

        ~allocator_free_list() {
            for (size_t i = 0; i < MAX_FREE_LIST_SIZE; i++) {
                // NOLINT(*-loop-convert)
                const void *ptr = ptrs[i];
                const size_t size = sizes[i];
                if (ptr != nullptr) {
                    allocator.release(ptr, size);
                }
            }
        }
    };
} // end anonymous namespace


//--------------------------------------------------------
// The main pipeline ring reduce API
//--------------------------------------------------------
std::pair<bool, bool> ccoip::reduce::pipelineRingReduce(
        CCoIPClientState &client_state, tinysockets::QueuedSocket &master_socket, const uint64_t tag,
        const uint64_t seq_nr, std::span<const std::byte> src_buf, const std::span<std::byte> &dst_buf,
        const ccoip_data_type_t data_type, const ccoip_data_type_t quantized_type, const ccoip_reduce_op_t op,
        const ccoip_quantization_algorithm_t quantization_algorithm, const size_t rank, const size_t world_size,
        const std::vector<ccoip_uuid_t> &ring_order,
        const std::unordered_map<ccoip_uuid_t, std::vector<std::shared_ptr<tinysockets::MultiplexedIOSocket>>>
                &peer_tx_sockets,
        const std::unordered_map<ccoip_uuid_t, std::vector<std::shared_ptr<tinysockets::MultiplexedIOSocket>>>
                &peer_rx_sockets) {
    using namespace ccoip::internal::quantize;
    using namespace ccoip::internal::reduce;

    thread_local PooledAllocator pooled_allocator{};
    allocator_free_list free_list{.allocator = pooled_allocator};

    // If world_size < 2, just copy and finalize
    if (world_size < 2) {
        if (dst_buf.data() != src_buf.data()) {
            std::memcpy(dst_buf.data(), src_buf.data(), src_buf.size_bytes());
        }
        performReduceFinalization(dst_buf, data_type, world_size, op);
        return {true, false};
    }

    const bool src_and_dest_identical_ptr = (src_buf.data() == dst_buf.data());

    // Backup buffer to be used to restore the src_buf if the all reduce is aborted.
    std::span<const std::byte> backup_buf = src_buf;

    // Copy local data into dst_buf so we can reduce in place
    if (!src_and_dest_identical_ptr) {
        if (dst_buf.size_bytes() != src_buf.size_bytes()) {
            LOG(BUG) << "Destination buffer size does not match source buffer size!";
            return {false, false};
        }
        // add src buf as the initial contribution of the reduction
        std::memcpy(dst_buf.data(), src_buf.data(), src_buf.size_bytes());
    } else {
        // if src and dst are identical, we need to create a backup of the source buffer because it will be modified.
        void *backup = pooled_allocator.allocate(src_buf.size_bytes());
        free_list.add(backup, src_buf.size_bytes());
        std::memcpy(backup, src_buf.data(), src_buf.size_bytes());
        backup_buf = std::span<const std::byte>(static_cast<std::byte *>(backup), src_buf.size_bytes());
    }

    const auto restore_src_buf = [src_and_dest_identical_ptr, src_buf, backup_buf] {
        if (src_and_dest_identical_ptr) {
            // this is needed so that we *actually* don't have const data changing from the perspective of the user
            // in the case where the all reduce aborts.
            // So this const cast ironically saves the perception of const-ness from the user.
            auto *src_buf_ptr = const_cast<std::byte *>(src_buf.data());
            std::memcpy(src_buf_ptr, backup_buf.data(), backup_buf.size_bytes());
        }
    };

    // Number of (unquantized) elements total
    const size_t data_type_el_size = ccoip_data_type_size(data_type);
    assert(dst_buf.size_bytes() % data_type_el_size == 0);
    const size_t total_elements = dst_buf.size_bytes() / data_type_el_size;

    // Compute chunk boundaries
    auto boundaries = computeChunkBoundaries(total_elements, world_size);

    // If using sub-byte quantization, ensure that each chunk is a multiple of the pack size
    {
        const size_t qbits = ccoip_data_type_size_bits(quantized_type);

        if (const bool using_quant = (quantized_type != data_type) && (quantization_algorithm != ccoipQuantizationNone);
            using_quant && (qbits < 8)) {
            const size_t pack = 8 / std::gcd<size_t>(qbits, static_cast<size_t>(8)); // 4->2, 2->4, 1->8

            if (total_elements % pack != 0) {
                LOG(BUG) << "Total elements " << total_elements << " not divisible by pack=" << pack
                         << " (required for " << qbits << "-bit quantization).";
                return {false, false};
            }

            const size_t total_units = total_elements / pack; // split units of "pack"
            auto unit_bounds = computeChunkBoundaries(total_units, world_size);
            boundaries.clear();
            boundaries.reserve(unit_bounds.size());
            for (auto [us, ue]: unit_bounds) {
                boundaries.emplace_back(us * pack, ue * pack); // back to element indices
            }
        } else if (qbits % 8 != 0) {
            LOG(BUG) << "Unsupported quant width: " << qbits << " (not a multiple of 8 and no sub-byte packing path).";
            return {false, false};
        }
    }

    if (quantization_algorithm == ccoipQuantizationNone && quantized_type != data_type) {
        LOG(BUG) << "Quantization disabled (algorithm=None) but quantized_type != data_type";
        return {false, false};
    }

    // Determine the maximum chunk size across all ranks to size the "receive" buffer
    const size_t quant_type_el_size_bits = ccoip_data_type_size_bits(quantized_type);
    size_t max_chunk_el = 0;
    for (size_t r = 0; r < world_size; ++r) {
        const auto [start_el, end_el] = boundaries[r];
        if (const size_t chunk_el = (end_el > start_el) ? (end_el - start_el) : 0; chunk_el > max_chunk_el) {
            max_chunk_el = chunk_el;
        }
    }

    // chunks should always be byte-aligned;
    // For this reason, we disallow uneven chunk sizes in element count when using uint4 when using quantization.
    const size_t max_chunk_size_bits_q = max_chunk_el * quant_type_el_size_bits;
    if (max_chunk_size_bits_q % 8 != 0) {
        LOG(BUG) << "Max chunk size in bits is not a multiple of 8, despite chunks being even-sized in element count";
        return {false, false};
    }

    const size_t max_chunk_size_bytes_q = max_chunk_size_bits_q / 8;
    {
        auto *recv_buffer = pooled_allocator.allocate(max_chunk_size_bytes_q);
        free_list.add(recv_buffer, max_chunk_size_bytes_q);

        std::span recv_buffer_span{static_cast<std::byte *>(recv_buffer), max_chunk_size_bytes_q};

        //----------------------------------------------------------------------
        // PHASE 1: Ring Reduce-Scatter (world_size - 1 steps)
        // After this, rank r ends up with the full sum for chunk r in its subrange.
        //----------------------------------------------------------------------
        for (int step = 0; step < static_cast<int>(world_size) - 1; ++step) {
            // chunk_to_send = (rank - step) mod world_size
            const size_t tx_chunk_idx = (rank + world_size - step) % world_size;
            // chunk_to_receive = (rank - step - 1) mod world_size
            const size_t rx_chunk_idx = (rank + world_size - step - 1) % world_size;

            // Identify subranges
            const auto [tx_start_el, tx_end_el] = boundaries[tx_chunk_idx];
            const auto [rx_start_el, rx_end_el] = boundaries[rx_chunk_idx];
            const size_t tx_size_el = (tx_end_el > tx_start_el ? tx_end_el - tx_start_el : 0);
            const size_t rx_size_el = (rx_end_el > rx_start_el ? rx_end_el - rx_start_el : 0);

            // Subspans for sending & receiving
            std::span<const std::byte> tx_unquantized =
                    guarded_subspan(dst_buf, tx_start_el * data_type_el_size, tx_size_el * data_type_el_size);
            std::span<std::byte> rx_span =
                    guarded_subspan(dst_buf, rx_start_el * data_type_el_size, rx_size_el * data_type_el_size);

            // Possibly quantize
            void *quantized_data = nullptr;
            std::optional<DeQuantizationMetaData> meta_data;
            const size_t quant_buf_size = (tx_size_el * quant_type_el_size_bits) / 8; // this should be safe now
            if (quantized_type != data_type && quantization_algorithm != ccoipQuantizationNone && tx_size_el > 0) {
                quantized_data = pooled_allocator.allocate(quant_buf_size);
                free_list.add(quantized_data, quant_buf_size);

                std::span q_span(static_cast<std::byte *>(quantized_data), quant_buf_size);
                meta_data =
                        performQuantization(q_span, tx_unquantized, quantization_algorithm, quantized_type, data_type);
                // Now we send q_span
                tx_unquantized = std::span<const std::byte>(q_span.data(), q_span.size());
            }

            // We'll receive into the front of recv_buffer_span up to rx_size_el * quant_type_el_size
            std::span<std::byte> recv_sub =
                    guarded_subspan(recv_buffer_span, 0, (rx_size_el * quant_type_el_size_bits) / 8);

            // Perform ring exchange & reduce
            auto [success, abort_packet_received] =
                    runReduceStage(client_state, master_socket, tag, seq_nr,
                                   /*tx_span=*/tx_unquantized,
                                   /*rx_span=*/rx_span,
                                   /*recv_buffer_span=*/recv_sub, data_type, quantized_type, quantization_algorithm, op,
                                   rank, world_size, ring_order, meta_data, peer_tx_sockets, peer_rx_sockets);
            if (quantized_data != nullptr) {
                pooled_allocator.release(quantized_data, quant_buf_size);
                free_list.remove(quantized_data);
                quantized_data = nullptr;
            }
            if (!success || abort_packet_received) {
                restore_src_buf();
                return {success, abort_packet_received};
            }
            assert(quantized_data == nullptr);
        }
        pooled_allocator.release(recv_buffer, recv_buffer_span.size_bytes());
        free_list.remove(recv_buffer);
    }

    //----------------------------------------------------------------------
    // PHASE 2: Ring Allgather (world_size - 1 steps)
    // Each rank pipeline-broadcasts the chunk it owns; eventually,
    // all ranks have all chunks in dst_buf.
    //----------------------------------------------------------------------

    // NOTE: we only quantize the chunk we "own". Subsequent chunks we get to own are received quantized and forwarded
    // as such. This is to prevent double-quantization of the same data, which would lead to loss of precision. We don't
    // want to guarantee q = Q(x); Q(D(q)) = q for Q(x) being the quantization function and D(q) being the
    // de-quantization function.
    void *owned_data_ptr = nullptr;
    std::span<std::byte> owned_data_span{};
    DeQuantizationMetaData prev_meta_data{};

    // compute the maximum chunk size across all ranks
    size_t max_chunk_size_el = 0;
    for (const auto [start_el, end_el]: boundaries) {
        const auto chunk_size_el = (end_el > start_el) ? (end_el - start_el) : 0;
        if (chunk_size_el > max_chunk_size_el) {
            max_chunk_size_el = chunk_size_el;
        }
    }
    {
        void *quantized_data = nullptr;
        size_t quantized_data_size = 0;

        // Start with the chunk we "own" = (rank+1) mod world_size
        size_t current_chunk_idx = (rank + 1) % world_size;
        for (int step = 0; step < static_cast<int>(world_size) - 1; ++step) {
            // We'll send chunk = current_chunk_idx
            const auto [tx_start_el, tx_end_el] = boundaries[current_chunk_idx];
            const size_t tx_size_el = (tx_end_el > tx_start_el ? tx_end_el - tx_start_el : 0);

            std::span<std::byte> orig_tx_span =
                    guarded_subspan(dst_buf, tx_start_el * data_type_el_size, tx_size_el * data_type_el_size);

            std::span<const std::byte> tx_span = orig_tx_span;

            // The chunk we will receive is:
            // inc_idx = (current_chunk_idx - 1 + world_size) % world_size
            const size_t inc_idx = (current_chunk_idx + world_size - 1) % world_size;
            const auto [rx_start_el, rx_end_el] = boundaries[inc_idx];
            const size_t rx_size_el = (rx_end_el > rx_start_el ? rx_end_el - rx_start_el : 0);

            std::span<std::byte> rx_span =
                    guarded_subspan(dst_buf, rx_start_el * data_type_el_size, rx_size_el * data_type_el_size);

            // Possibly quantize
            std::optional<DeQuantizationMetaData> meta_data;
            if (quantized_type != data_type && quantization_algorithm != ccoipQuantizationNone && tx_size_el > 0) {
                if (owned_data_ptr == nullptr) {
                    // if this is the first stage, we quantize our own finished chunk.
                    assert(step == 0); // only in stage 0 should this ever happen.
                    if (quantized_data == nullptr) {
                        quantized_data_size = (tx_size_el * quant_type_el_size_bits) / 8;
                        quantized_data = pooled_allocator.allocate(quantized_data_size);
                        free_list.add(quantized_data, quantized_data_size);
                    }
                    std::span q_span(static_cast<std::byte *>(quantized_data),
                                     (tx_size_el * quant_type_el_size_bits) / 8);
                    meta_data = performQuantization(q_span, tx_span, quantization_algorithm, quantized_type, data_type);
                    tx_span = std::span<const std::byte>(q_span.data(), q_span.size());

                    // If quantization is enabled, this peer technically has a "higher precision accumulator" than what
                    // other peers have, and it would propagate into the final result array.
                    // And again, we need to actively destroy information for the sake of parity with other
                    // peers. What the other peer would have de-quantized, is what we need to set out data to as well.
                    performReduction(orig_tx_span, tx_span, data_type, quantized_type, quantization_algorithm,
                                     ccoipOpSet, *meta_data);
                } else {
                    // forward the quantized data we received in the previous step
                    tx_span = std::span<const std::byte>(owned_data_span.data(),
                                                         (tx_size_el * quant_type_el_size_bits) / 8);
                    meta_data = prev_meta_data;
                }
            }

            // we will hold on to the quantized data we just received and forward it verbatim in the next step.
            if (owned_data_ptr == nullptr) {
                const size_t owned_data_size = (max_chunk_size_el * quant_type_el_size_bits) / 8;

                owned_data_ptr = pooled_allocator.allocate(owned_data_size);
                free_list.add(owned_data_ptr, owned_data_size);

                owned_data_span = std::span(static_cast<std::byte *>(owned_data_ptr), owned_data_size);
            }

            // We will receive into the owned data ptr memory
            // owned_data_span has enough memory to fit the largest chunk size,
            // however we sub-span to the actual size of the chunk we are receiving.
            std::span<std::byte> recv_sub =
                    guarded_subspan(owned_data_span, 0, (rx_size_el * quant_type_el_size_bits) / 8);

            // Ring exchange (no reduce-op)
            auto [success, abort_packet_received] =
                    runAllgatherStage(client_state, master_socket, tag, seq_nr, tx_span, rx_span, recv_sub, data_type,
                                      quantized_type, quantization_algorithm, rank, world_size, ring_order, meta_data,
                                      prev_meta_data, // out
                                      peer_tx_sockets, peer_rx_sockets);
            if (!success || abort_packet_received) {
                restore_src_buf();
                return {success, abort_packet_received};
            }

            // The newly received chunk (inc_idx) becomes our "owned" chunk for the next step
            current_chunk_idx = inc_idx;
        }
        if (quantized_data != nullptr) {
            pooled_allocator.release(quantized_data, quantized_data_size);
            free_list.remove(quantized_data);
        }
    }

    pooled_allocator.release(owned_data_ptr, owned_data_span.size_bytes());
    free_list.remove(owned_data_ptr);

    //----------------------------------------------------------------------
    // Finalize for ops that need it (e.g. op = AVG, etc.)
    //----------------------------------------------------------------------
    performReduceFinalization(dst_buf, data_type, world_size, op);
    return {true, false};
}
