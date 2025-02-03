#include "reduce.hpp"

#include <cassert>
#include <ccoip_master_state.hpp>
#include <cstring>
#include <ccoip_packets.hpp>
#include <ccoip_types.hpp>
#include <quantize.hpp>
#include <reduce_kernels.hpp>
#include <tinysockets.hpp>
#include <win_sock_bridge.h>

namespace {

    /**
     * \brief Utility to compute per-rank chunk boundaries for an array
     *        of \p total_el elements across \p world_size ranks.
     *
     * Returns a vector of length \p world_size, where
     *   boundaries[r] = {start_index, end_index}
     * in element (not byte) units.
     */
    std::vector<std::pair<size_t, size_t>>
    computeChunkBoundaries(const size_t total_el, const size_t world_size) {
        std::vector<std::pair<size_t, size_t>> boundaries(world_size, {0, 0});
        if (world_size == 0) { return boundaries; }

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
     * \brief Perform the "reduce-scatter" stage of the ring in one step (send one chunk / receive one chunk / reduce).
     *
     * - Sends the \p tx_span (possibly quantized) to the next rank.
     * - Receives from the previous rank into \p recv_buffer_span (which is sized exactly for the incoming chunk).
     * - De-quantizes and accumulates into \p rx_span.
     */
    void runReduceStage(
            ccoip::CCoIPClientState &client_state,
            const uint64_t tag,
            const std::span<const std::byte> &tx_span,
            const std::span<std::byte> &rx_span,
            const std::span<std::byte> &recv_buffer_span,

            const ccoip::ccoip_data_type_t data_type,
            const ccoip::ccoip_data_type_t quantized_type,
            const ccoip::ccoip_quantization_algorithm_t quantization_algorithm,
            const ccoip::ccoip_reduce_op_t op,

            const size_t rank,
            const size_t world_size,
            const std::vector<ccoip_uuid_t> &ring_order,

            const std::optional<ccoip::internal::quantize::DeQuantizationMetaData> &meta_data_self,

            const std::unordered_map<ccoip_uuid_t,
                                     std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_tx_sockets,
            const std::unordered_map<ccoip_uuid_t,
                                     std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_rx_sockets) {
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

        const auto &tx_socket = peer_tx_sockets.at(tx_peer);
        const auto &rx_socket = peer_rx_sockets.at(rx_peer);

        // 1) Send our local de-quant metadata to next peer
        if (quantized_type != data_type) {
            ccoip::P2PPacketDequantizationMeta packet{};
            packet.tag = tag;
            packet.dequantization_meta = (meta_data_self
                                              ? *meta_data_self
                                              : DeQuantizationMetaData{});
            if (!tx_socket->sendPacket<ccoip::P2PPacketDequantizationMeta>(packet)) {
                LOG(WARN) << "Failed to send de-quantization meta data!";
                return;
            }
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsTxBytes(tag, ltv_header + packet.serializedSize());
        }

        // 2) Receive their metadata from the previous peer
        DeQuantizationMetaData received_meta_data{};
        if (quantized_type != data_type) {
            const auto packet = rx_socket->receivePacket<ccoip::P2PPacketDequantizationMeta>();
            if (!packet) {
                LOG(WARN) << "Failed to receive de-quantization meta data!";
                return;
            }
            received_meta_data = packet->dequantization_meta;
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsRxBytes(tag, ltv_header + packet->serializedSize());
        }

        // 3) Full-duplex send/recv loop
        size_t bytes_sent = 0;
        size_t bytes_recvd = 0;

        while (bytes_sent < total_tx_size || bytes_recvd < total_rx_size) {
            // Prepare poll descriptors
            std::vector<poll::PollDescriptor> descriptors;
            descriptors.reserve(2);

            std::optional<poll::PollDescriptor *> tx_desc = std::nullopt;
            std::optional<poll::PollDescriptor *> rx_desc = std::nullopt;

            if (bytes_sent < total_tx_size) {
                descriptors.push_back({tx_socket->getSocketFd(), poll::PollEvent::POLL_OUTPUT});
                tx_desc = &descriptors.back();
            }
            if (bytes_recvd < total_rx_size) {
                descriptors.push_back({rx_socket->getSocketFd(), poll::PollEvent::POLL_INPUT});
                rx_desc = &descriptors.back();
            }

            if (poll::poll(descriptors, -1) < 0) {
                const std::string error_message = strerror(errno);
                LOG(WARN) << "poll() failed: " << error_message;
                return;
            }

            // 3a) Send if ready
            if (tx_desc && (*tx_desc)->hasEvent(poll::PollEvent::POLL_OUTPUT)) {
                const auto send_sub = tx_span.subspan(bytes_sent);
                if (auto sent = send_nonblocking(send_sub, **tx_desc)) {
                    bytes_sent += *sent;
                    client_state.trackCollectiveComsTxBytes(tag, *sent);
                }
            }

            // 3b) Receive if ready
            if (rx_desc && (*rx_desc)->hasEvent(poll::PollEvent::POLL_INPUT)) {
                const auto recv_sub = recv_buffer_span.subspan(bytes_recvd);
                if (auto recvd = recv_nonblocking(recv_sub, **rx_desc)) {
                    if (*recvd > 0) {
                        client_state.trackCollectiveComsRxBytes(tag, *recvd);

                        const size_t quant_el_sz = ccoip_data_type_size(quantized_type);

                        // old_floor = how many *complete* quantized elements we had before
                        const size_t old_floor = (bytes_recvd / quant_el_sz) * quant_el_sz;
                        bytes_recvd += *recvd;
                        // new_floor = how many *complete* quantized elements we have now
                        const size_t new_floor = (bytes_recvd / quant_el_sz) * quant_el_sz;

                        // If new_floor > old_floor, we have at least one fully-received element to reduce
                        if (new_floor > old_floor) {
                            const size_t chunk_bytes = new_floor - old_floor;
                            auto reduce_src_span =
                                    recv_buffer_span.subspan(old_floor, chunk_bytes);

                            // Map that to data_type-sized output range in rx_span
                            const size_t data_type_el_sz = ccoip_data_type_size(data_type);
                            auto reduce_dst_span =
                                    rx_span.subspan((old_floor / quant_el_sz) * data_type_el_sz,
                                                    (chunk_bytes / quant_el_sz) * data_type_el_sz);

                            // Accumulate newly arrived data into rx_span
                            performReduction(reduce_dst_span,
                                             reduce_src_span,
                                             data_type,
                                             quantized_type,
                                             quantization_algorithm,
                                             op,
                                             received_meta_data);
                        }
                    }
                }
            }
        }
    }

    /**
     * \brief Perform the "allgather" stage of the ring for one step (send chunk / receive chunk / copy in).
     *
     * This is like runReduceStage but without applying a reduce-op. Instead, we effectively copy the newly
     * arrived data into \p rx_span.
     */
    void runAllgatherStage(
            ccoip::CCoIPClientState &client_state,
            const uint64_t tag,
            const std::span<const std::byte> &tx_span,
            const std::span<std::byte> &rx_span,
            const std::span<std::byte> &recv_buffer_span,

            const ccoip::ccoip_data_type_t data_type,
            const ccoip::ccoip_data_type_t quantized_type,
            const ccoip::ccoip_quantization_algorithm_t quantization_algorithm,

            const size_t rank,
            const size_t world_size,
            const std::vector<ccoip_uuid_t> &ring_order,

            const std::optional<ccoip::internal::quantize::DeQuantizationMetaData> &meta_data_self,
            ccoip::internal::quantize::DeQuantizationMetaData &received_meta_data_out,

            const std::unordered_map<ccoip_uuid_t,
                                     std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_tx_sockets,
            const std::unordered_map<ccoip_uuid_t,
                                     std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_rx_sockets) {
        using namespace tinysockets;
        using namespace ccoip::internal::quantize;
        using namespace ccoip::internal::reduce;

        const size_t total_tx_size = tx_span.size_bytes();
        const size_t total_rx_size = recv_buffer_span.size_bytes();

        const size_t rx_peer_idx = (rank + world_size - 1) % world_size;
        const size_t tx_peer_idx = (rank + 1) % world_size;

        const ccoip_uuid_t rx_peer = ring_order.at(rx_peer_idx);
        const ccoip_uuid_t tx_peer = ring_order.at(tx_peer_idx);

        const auto &tx_socket = peer_tx_sockets.at(tx_peer);
        const auto &rx_socket = peer_rx_sockets.at(rx_peer);

        // 1) Exchange metadata for consistency
        if (quantized_type != data_type) {
            ccoip::P2PPacketDequantizationMeta packet{};
            packet.tag = tag;
            packet.dequantization_meta = (meta_data_self
                                              ? *meta_data_self
                                              : DeQuantizationMetaData{});
            if (!tx_socket->sendPacket<ccoip::P2PPacketDequantizationMeta>(packet)) {
                LOG(WARN) << "Failed to send de-quantization meta data in allgather!";
                return;
            }
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsTxBytes(tag, ltv_header + packet.serializedSize());
        }

        DeQuantizationMetaData received_meta_data{};
        if (quantized_type != data_type) {
            const auto packet = rx_socket->receivePacket<ccoip::P2PPacketDequantizationMeta>();
            if (!packet) {
                LOG(WARN) << "Failed to receive de-quant meta in allgather!";
                return;
            }
            received_meta_data = packet->dequantization_meta;
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsRxBytes(tag, ltv_header + packet->serializedSize());
        }
        received_meta_data_out = received_meta_data;

        // 2) Full-duplex send/recv
        size_t bytes_sent = 0;
        size_t bytes_recvd = 0;

        while (bytes_sent < total_tx_size || bytes_recvd < total_rx_size) {
            std::vector<poll::PollDescriptor> descriptors;
            descriptors.reserve(2);

            std::optional<poll::PollDescriptor *> tx_desc = std::nullopt;
            std::optional<poll::PollDescriptor *> rx_desc = std::nullopt;

            if (bytes_sent < total_tx_size) {
                descriptors.push_back({tx_socket->getSocketFd(), poll::PollEvent::POLL_OUTPUT});
                tx_desc = &descriptors.back();
            }
            if (bytes_recvd < total_rx_size) {
                descriptors.push_back({rx_socket->getSocketFd(), poll::PollEvent::POLL_INPUT});
                rx_desc = &descriptors.back();
            }

            if (poll::poll(descriptors, -1) < 0) {
                const std::string error_message = strerror(errno);
                LOG(WARN) << "poll() failed (allgather): " << error_message;
                return;
            }

            // Send
            if (tx_desc && (*tx_desc)->hasEvent(poll::PollEvent::POLL_OUTPUT)) {
                const auto send_sub = tx_span.subspan(bytes_sent);
                if (auto sent = send_nonblocking(send_sub, **tx_desc)) {
                    bytes_sent += *sent;
                    client_state.trackCollectiveComsTxBytes(tag, *sent);
                }
            }

            // Receive
            if (rx_desc && (*rx_desc)->hasEvent(poll::PollEvent::POLL_INPUT)) {
                const auto recv_sub = recv_buffer_span.subspan(bytes_recvd);
                if (auto recvd = recv_nonblocking(recv_sub, **rx_desc)) {
                    if (*recvd > 0) {
                        client_state.trackCollectiveComsRxBytes(tag, *recvd);

                        const size_t quant_el_sz = ccoip_data_type_size(quantized_type);

                        const size_t old_floor = (bytes_recvd / quant_el_sz) * quant_el_sz;
                        bytes_recvd += *recvd;

                        if (const size_t new_floor = (bytes_recvd / quant_el_sz) * quant_el_sz; new_floor > old_floor) {
                            const size_t chunk_bytes = new_floor - old_floor;
                            // De-quantize + copy into rx_span
                            auto copy_src_span = recv_buffer_span.subspan(old_floor, chunk_bytes);

                            const size_t data_type_el_sz = ccoip_data_type_size(data_type);
                            auto copy_dst_span = rx_span.subspan(
                                    (old_floor / quant_el_sz) * data_type_el_sz,
                                    (chunk_bytes / quant_el_sz) * data_type_el_sz);

                            // We do not accumulate for allgather, just "copy" via performReduction w/ ccoipOpSet
                            performReduction(copy_dst_span,
                                             copy_src_span,
                                             data_type,
                                             quantized_type,
                                             quantization_algorithm,
                                             ccoip::ccoipOpSet, // = copy
                                             received_meta_data);
                        }
                    }
                }
            }
        }
    }
} // end anonymous namespace


//--------------------------------------------------------
// The main pipeline ring reduce API
//--------------------------------------------------------
void ccoip::reduce::pipelineRingReduce(
        CCoIPClientState &client_state,
        const uint64_t tag,
        std::span<const std::byte> src_buf,
        const std::span<std::byte> &dst_buf,
        const ccoip_data_type_t data_type,
        const ccoip_data_type_t quantized_type,
        const ccoip_reduce_op_t op,
        const ccoip_quantization_algorithm_t quantization_algorithm,
        const size_t rank,
        const size_t world_size,
        const std::vector<ccoip_uuid_t> &ring_order,
        const std::unordered_map<ccoip_uuid_t,
                                 std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_tx_sockets,
        const std::unordered_map<ccoip_uuid_t,
                                 std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_rx_sockets) {
    using namespace ccoip::internal::quantize;
    using namespace ccoip::internal::reduce;

    // If world_size < 2, just copy and finalize
    if (world_size < 2) {
        if (dst_buf.data() != src_buf.data()) {
            std::memcpy(dst_buf.data(), src_buf.data(), src_buf.size_bytes());
        }
        performReduceFinalization(dst_buf, data_type, world_size, op);
        return;
    }

    // Handle potential overlap of src_buf and dst_buf:
    std::optional<std::unique_ptr<std::byte[]>> maybe_src_copy;

    bool src_and_dest_identical_ptr = (src_buf.data() == dst_buf.data());
    {
        const auto *src_beg = src_buf.data();
        const auto *src_end = src_beg + src_buf.size_bytes();
        const auto *dst_beg = dst_buf.data();
        const auto *dst_end = dst_beg + dst_buf.size_bytes();
        const bool overlap = !((src_end <= dst_beg) || (dst_end <= src_beg));
        if (overlap) {
            maybe_src_copy = std::make_unique<std::byte[]>(src_buf.size_bytes());
            std::memcpy(maybe_src_copy->get(), src_buf.data(), src_buf.size_bytes());
            src_buf = std::span<const std::byte>(maybe_src_copy->get(), src_buf.size_bytes());
        }
    }

    // Copy local data into dst_buf so we can reduce in place
    if (!src_and_dest_identical_ptr) {
        std::memcpy(dst_buf.data(), src_buf.data(), src_buf.size_bytes());
    } else {
        // if src and dest are identical, they definitely overlap
        // which means that we just created a copy of src buf and made it the src buf, leaving the original ptr in dst_buf,
        // which contains the same data as src_buf.
    }

    // Number of (unquantized) elements total
    const size_t data_type_el_size = ccoip_data_type_size(data_type);
    assert(dst_buf.size_bytes() % data_type_el_size == 0);
    const size_t total_elements = dst_buf.size_bytes() / data_type_el_size;

    // Compute chunk boundaries
    auto boundaries = computeChunkBoundaries(total_elements, world_size);

    // Determine the maximum chunk size across all ranks to size the "receive" buffer
    const size_t quant_type_el_size = ccoip_data_type_size(quantized_type);
    size_t max_chunk_el = 0;
    for (size_t r = 0; r < world_size; ++r) {
        const auto [start_el, end_el] = boundaries[r];
        if (const size_t chunk_el = (end_el > start_el) ? (end_el - start_el) : 0; chunk_el > max_chunk_el) {
            max_chunk_el = chunk_el;
        }
    }
    const size_t max_chunk_size_bytes_q = max_chunk_el * quant_type_el_size;
    auto recv_buffer = std::make_unique<std::byte[]>(max_chunk_size_bytes_q);
    std::span recv_buffer_span{recv_buffer.get(), max_chunk_size_bytes_q};


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
                dst_buf.subspan(tx_start_el * data_type_el_size, tx_size_el * data_type_el_size);
        std::span<std::byte> rx_span =
                dst_buf.subspan(rx_start_el * data_type_el_size, rx_size_el * data_type_el_size);

        // Possibly quantize
        std::unique_ptr<std::byte[]> quantized_data;
        std::optional<DeQuantizationMetaData> meta_data;
        if (quantized_type != data_type && quantization_algorithm != ccoipQuantizationNone && tx_size_el > 0) {
            quantized_data = std::make_unique<std::byte[]>(tx_size_el * quant_type_el_size);
            std::span q_span(quantized_data.get(), tx_size_el * quant_type_el_size);
            meta_data = performQuantization(q_span,
                                            tx_unquantized,
                                            quantization_algorithm,
                                            quantized_type,
                                            data_type);
            // Now we send q_span
            tx_unquantized = std::span<const std::byte>(q_span.data(), q_span.size());
        }

        // We'll receive into the front of recv_buffer_span up to rx_size_el * quant_type_el_size
        std::span<std::byte> recv_sub =
                recv_buffer_span.subspan(0, rx_size_el * quant_type_el_size);

        // Perform ring exchange & reduce
        runReduceStage(client_state, tag,
                       /*tx_span=*/ tx_unquantized,
                       /*rx_span=*/ rx_span,
                       /*recv_buffer_span=*/ recv_sub,
                       data_type, quantized_type, quantization_algorithm, op,
                       rank, world_size, ring_order,
                       meta_data,
                       peer_tx_sockets, peer_rx_sockets);
    }

    //----------------------------------------------------------------------
    // PHASE 2: Ring Allgather (world_size - 1 steps)
    // Each rank pipeline-broadcasts the chunk it owns; eventually,
    // all ranks have all chunks in dst_buf.
    //----------------------------------------------------------------------

    // NOTE: we only quantize the chunk we "own". Subsequent chunks we get to own are received quantized and forwarded as such.
    // This is to prevent double-quantization of the same data, which would lead to loss of precision.
    // We don't want to guarantee q = Q(x); Q(D(q)) = q for Q(x) being the quantization function and D(q) being the de-quantization function.
    std::unique_ptr<std::byte[]> owned_data_ptr = nullptr;
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

    // Start with the chunk we "own" = (rank+1) mod world_size
    size_t current_chunk_idx = (rank + 1) % world_size;
    std::unique_ptr<std::byte[]> quantized_data = nullptr;
    for (int step = 0; step < static_cast<int>(world_size) - 1; ++step) {
        // We'll send chunk = current_chunk_idx
        const auto [tx_start_el, tx_end_el] = boundaries[current_chunk_idx];
        const size_t tx_size_el = (tx_end_el > tx_start_el ? tx_end_el - tx_start_el : 0);

        std::span<std::byte> orig_tx_span =
                dst_buf.subspan(tx_start_el * data_type_el_size, tx_size_el * data_type_el_size);

        std::span<const std::byte> tx_span = orig_tx_span;

        // The chunk we will receive is:
        // inc_idx = (current_chunk_idx - 1 + world_size) % world_size
        const size_t inc_idx = (current_chunk_idx + world_size - 1) % world_size;
        const auto [rx_start_el, rx_end_el] = boundaries[inc_idx];
        const size_t rx_size_el = (rx_end_el > rx_start_el ? rx_end_el - rx_start_el : 0);

        std::span<std::byte> rx_span =
                dst_buf.subspan(rx_start_el * data_type_el_size, rx_size_el * data_type_el_size);

        // Possibly quantize
        std::optional<DeQuantizationMetaData> meta_data;
        if (quantized_type != data_type && quantization_algorithm != ccoipQuantizationNone && tx_size_el > 0) {
            if (owned_data_ptr == nullptr) {
                // if this is the first stage, we quantize our own finished chunk.
                assert(step == 0); // only in stage 0 should this ever happen.
                if (quantized_data == nullptr) {
                    quantized_data = std::unique_ptr<std::byte[]>(new std::byte[tx_size_el * quant_type_el_size]);
                    // only allocate once
                }
                std::span q_span(quantized_data.get(), tx_size_el * quant_type_el_size);
                meta_data = performQuantization(q_span,
                                                tx_span,
                                                quantization_algorithm,
                                                quantized_type,
                                                data_type);
                tx_span = std::span<const std::byte>(q_span.data(), q_span.size());

                // If quantization is enabled, this peer technically has a "higher precision accumulator" than what
                // other peers have, and it would propagate into the final result array.
                // And again, we need to actively destroy information for the sake of parity with other
                // peers. What the other peer would have de-quantized, is what we need to set out data to as well.
                performReduction(orig_tx_span, tx_span, data_type, quantized_type,
                                 quantization_algorithm, ccoipOpSet, *meta_data);
            } else {
                // forward the quantized data we received in the previous step
                tx_span = std::span<const std::byte>(owned_data_span.data(), tx_size_el * quant_type_el_size);
                meta_data = prev_meta_data;
            }
        }

        // We'll receive into the front of recv_buffer_span
        std::span<std::byte> recv_sub =
                recv_buffer_span.subspan(0, rx_size_el * quant_type_el_size);

        // Ring exchange (no reduce-op)
        runAllgatherStage(client_state, tag,
                          tx_span, rx_span, recv_sub,
                          data_type, quantized_type, quantization_algorithm,
                          rank, world_size, ring_order,
                          meta_data,
                          prev_meta_data, // out
                          peer_tx_sockets, peer_rx_sockets);

        // we will hold on to the quantized data we just received and forward it verbatim in the next step.
        if (owned_data_ptr == nullptr) {
            owned_data_ptr = std::unique_ptr<std::byte[]>(new std::byte[max_chunk_size_el * quant_type_el_size]);
            owned_data_span = std::span(owned_data_ptr.get(), max_chunk_size_el * quant_type_el_size);
        }
        std::memcpy(owned_data_span.data(), recv_sub.data(), owned_data_span.size_bytes());

        // The newly received chunk (inc_idx) becomes our "owned" chunk for the next step
        current_chunk_idx = inc_idx;
    }

    //----------------------------------------------------------------------
    // Finalize for ops that need it (e.g. op = AVG, etc.)
    //----------------------------------------------------------------------
    performReduceFinalization(dst_buf, data_type, world_size, op);
}
