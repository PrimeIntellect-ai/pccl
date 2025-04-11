#include <ccoip_utils.hpp>
#include <win_sock_bridge.h>

#include <MPSCQueue.hpp>
#include <SPSCQueue.hpp>
#include <cstring>
#include <iostream>
#include <network_order_utils.hpp>
#include <shared_mutex>
#include <threadpark.h>
#include "tinysockets.hpp"

#ifndef _MSC_VER
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif

static bool configure_socket_fd(const int socket_fd) {
    constexpr int opt = 1;

    // enable TCP_NODELAY
    if (setsockoptvp(socket_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) [[unlikely]] {
        LOG(ERR) << "Failed to set TCP_NODELAY option on server socket";
        closesocket(socket_fd);
        return false;
    }

    // enable SO_BUSY_POLL if available
#ifdef SO_BUSY_POLL
    setsockoptvp(socket_fd, SOL_SOCKET, SO_BUSY_POLL, &opt, sizeof(opt));
#endif

    // enable TCP_QUICKACK if available
#ifdef TCP_QUICKACK
    setsockoptvp(socket_fd, IPPROTO_TCP, TCP_QUICKACK, &opt, sizeof(opt));
#endif
    return true;
}

struct ReceiveQueueEntry {
    uint64_t tag{};
    uint8_t *data;
    size_t data_size{};
    std::span<std::byte> data_span{};
};

struct SendQueueEntry {
    uint64_t tag{};
    const std::byte *data{};
    size_t size_bytes{};
    bool is_cloned{};
    tpark_handle_t *done_handle{};
};

#define TXRX_QUEUE_DEPTH 1024
#define POOLED_ALLOCATOR_MAX_ENTRIES 128

namespace tinysockets {
    // Note: this allocator can be this stupid because our sizes are
    // deterministic and are also the same on both RX and TX side.
    // Also note: This allocator is crucial for high all reduce performance.
    class PooledAllocator {
        std::vector<std::pair<void *, size_t>> pool;
        std::mutex mutex;

    public:
        void *allocate(const size_t size) {
            std::unique_lock lock(mutex);
            for (auto it = pool.begin(); it != pool.end(); ++it) {
                if (it->second >= size) {
                    void *ptr = it->first;
                    pool.erase(it);
                    return ptr;
                }
            }
            return malloc(size);
        }

        void release(const void *ptr, size_t size) {
            // we trust the user to set size correctly; there is only one intended call-site anyways
            std::unique_lock lock(mutex);
            if (pool.size() >= POOLED_ALLOCATOR_MAX_ENTRIES) {
                free(const_cast<void *>(ptr));
            } else {
                pool.emplace_back(const_cast<void *>(ptr), size);
            }
        }

        ~PooledAllocator() {
            for (auto &[ptr, size]: pool) {
                free(ptr);
            }
            pool.clear();
        }
    };

    struct MultiplexedIOSocketInternalState {
        MPSCQueue<SendQueueEntry> send_queue;

        std::shared_mutex receive_queues_mutex{};
        std::unordered_map<uint64_t, std::unique_ptr<::rigtorp::SPSCQueue<ReceiveQueueEntry>>> receive_queues{};

        PooledAllocator tx_allocator{};
        PooledAllocator rx_allocator{};

        tpark_handle_t *tx_park_handle = nullptr;

        MultiplexedIOSocketInternalState() : send_queue(TXRX_QUEUE_DEPTH) {}

        ~MultiplexedIOSocketInternalState() {
            if (tx_park_handle != nullptr) {
                tparkDestroyHandle(tx_park_handle);
            }
        }
    };
} // namespace tinysockets

tinysockets::MultiplexedIOSocket::MultiplexedIOSocket(const ccoip_socket_address_t &address,
                                                      const ConnectionModeFlags flags) :
    socket_fd(0), connect_sockaddr(address), flags(flags), internal_state(new MultiplexedIOSocketInternalState) {}

tinysockets::MultiplexedIOSocket::MultiplexedIOSocket(const int socket_fd, const ConnectionModeFlags flags) :
    socket_fd(socket_fd), connect_sockaddr(), flags(flags), internal_state(new MultiplexedIOSocketInternalState) {}

tinysockets::MultiplexedIOSocket::MultiplexedIOSocket(const int socket_fd, const ccoip_socket_address_t &address,
                                                      const ConnectionModeFlags flags) :
    socket_fd(socket_fd), connect_sockaddr(address), flags(flags),
    internal_state(new MultiplexedIOSocketInternalState) {}

bool tinysockets::MultiplexedIOSocket::establishConnection() {
    if (socket_fd != 0) {
        return false;
    }
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) [[unlikely]] {
        LOG(ERR) << "Failed to create socket";
        return false;
    }

    if (!configure_socket_fd(socket_fd)) [[unlikely]] {
        return false;
    }

    // connect to the server
    sockaddr_in server_address_ipv4{};
    sockaddr_in6 server_address_ipv6{};

    if (connect_sockaddr.inet.protocol == inetIPv4) {
        if (convert_to_sockaddr_ipv4(connect_sockaddr, &server_address_ipv4) == -1) [[unlikely]] {
            LOG(ERR) << "Failed to convert socket address";
            closesocket(socket_fd);
            return false;
        }
    } else if (connect_sockaddr.inet.protocol == inetIPv6) {
        if (convert_to_sockaddr_ipv6(connect_sockaddr, &server_address_ipv6) == -1) [[unlikely]] {
            LOG(ERR) << "Failed to convert socket address";
            closesocket(socket_fd);
            return false;
        }
    } else [[unlikely]] {
        LOG(ERR) << "Unsupported protocol";
        closesocket(socket_fd);
        return false;
    }

    // connect to the server based on protocol
    if (connect_sockaddr.inet.protocol == inetIPv4) {
        if (connect(socket_fd, reinterpret_cast<sockaddr *>(&server_address_ipv4), sizeof(server_address_ipv4)) < 0) {
            const std::string error_message = std::strerror(errno);
            LOG(ERR) << "Failed to connect to server; connect() failed with " << error_message;
            closesocket(socket_fd);
            return false;
        }
    } else if (connect_sockaddr.inet.protocol == inetIPv6) {
        if (connect(socket_fd, reinterpret_cast<const sockaddr *>(&server_address_ipv6), sizeof(server_address_ipv6)) <
            0) {
            const std::string error_message = std::strerror(errno);
            LOG(ERR) << "Failed to connect to server; connect() failed with " << error_message;
            closesocket(socket_fd);
            return false;
        }
    } else [[unlikely]] {
        LOG(ERR) << "Unsupported protocol";
        closesocket(socket_fd);
        return false;
    }
    return true;
}

std::optional<size_t> tinysockets::MultiplexedIOSocket::receivePacketLength() const {
    uint64_t length;
    size_t n_received = 0;
    do {
        const ssize_t i = recvvp(socket_fd, &length, sizeof(length), 0);
        if (const bool running = this->running.load(std::memory_order_acquire); i == -1 || i == 0 || !running) {
            const std::string error_message = std::strerror(errno);
            if (running) {
                LOG(ERR) << "[MultiplexedIOSocket] Failed to receive packet length with error: " << error_message;
            }
            return std::nullopt;
        }
        n_received += i;
    } while (n_received < sizeof(length));
    return network_order_utils::host_to_network(length);
}

bool tinysockets::MultiplexedIOSocket::run() {
    if (socket_fd == 0) {
        return false;
    }
    running.store(true, std::memory_order_release);
    if (flags & MODE_RX) {
        recv_thread = std::thread([this] {
            while (running.load(std::memory_order_acquire)) {
                constexpr size_t PREAMBLE_SIZE = sizeof(uint64_t) * 2;
                uint64_t preamble[2] = {};
                size_t n_received = 0;

                // Keep reading until we either receive all 16 bytes,
                // or encounter an error/remote close/interrupt.
                do {
                    const ssize_t i = recvvp(socket_fd, reinterpret_cast<uint8_t *>(preamble) + n_received, PREAMBLE_SIZE - n_received, 0);
                    const bool still_running = running.load(std::memory_order_acquire);

                    // If error, remote closure, or we’re no longer “running,” bail out.
                    if (i == -1 || i == 0 || !still_running) {
                        if (still_running) {
                            LOG(ERR) << "[MultiplexedIOSocket] Failed to receive 16-byte preamble: "
                                     << std::strerror(errno);
                        }
                        return;
                    }
                    n_received += i;
                } while (n_received < PREAMBLE_SIZE);

                uint64_t length = preamble[0];
                uint64_t tag = preamble[1];

                length = network_order_utils::network_to_host(length);
                tag = network_order_utils::network_to_host(tag);

                // safeguard against large packets
                if (length > (1024 * 1024 * 1024)) {
                    LOG(FATAL) << "Received excessive packet length " << length << "; closing connection";
                    if (!interrupt()) [[unlikely]] {
                        LOG(ERR) << "Failed to interrupt MultiplexedIOSocket";
                    }
                }

                length -= sizeof(uint64_t); // subtract the tag size

                auto *data_ptr = static_cast<uint8_t *>(internal_state->rx_allocator.allocate(length));
                std::span data{data_ptr, length};
                if (!receivePacketData(data)) {
                    if (running.load(std::memory_order_acquire)) {
                        LOG(ERR) << "Failed to receive packet data for packet with length " << length
                                 << "; error: " << std::strerror(errno) << "; exiting receive loop...";
                        if (!interrupt()) [[unlikely]] {
                            LOG(ERR) << "Failed to interrupt MultiplexedIOSocket";
                        }
                    } else {
                        LOG(INFO) << "MultiplexedIOSocket::run() interrupted, exiting receive loop...";
                    }
                    break;
                }

                if (!internal_state->receive_queues.contains(tag)) {
                    std::unique_lock guard{internal_state->receive_queues_mutex};
                    internal_state->receive_queues[tag] =
                            std::make_unique<::rigtorp::SPSCQueue<ReceiveQueueEntry>>(TXRX_QUEUE_DEPTH);
                }

                // add entry to SPMC queue of tag
                {
                    const auto raw_ptr = reinterpret_cast<std::byte *>(data_ptr);
                    std::shared_lock guard{internal_state->receive_queues_mutex};
                    const auto &queue = internal_state->receive_queues.at(tag);
                    queue->push(ReceiveQueueEntry{
                            .tag = tag,

                            // we just pass this for the sake of ownership
                            .data = data_ptr,

                            // we have to take note of the size actually allocated by the data_ptr for the allocator,
                            // because that's what we allocated.
                            // We don't expose the tag in the data span, but we need to know how much we allocated
                            .data_size = length,

                            // don't include the tag in the data span
                            .data_span = std::span(raw_ptr, data.size_bytes())});
                }
            }
        });
    }

    if (flags & MODE_TX) {
        internal_state->tx_park_handle = tparkCreateHandle();
        send_thread = std::thread([this] {
            while (running.load(std::memory_order_acquire) && socket_fd != 0) {
                const SendQueueEntry *entry{};
                {
                    tparkBeginPark(internal_state->tx_park_handle);
                    entry = internal_state->send_queue.dequeue(true);
                    if (entry == nullptr) {
                        tparkWait(internal_state->tx_park_handle, true);
                        do {
                            entry = internal_state->send_queue.dequeue(true);
                            // despite the fact that wakes are guaranteed not-spurious and insertion into the queue
                            // should happen before the wake, it still is possible that the queue is empty because
                            // seq cst only guarantees all writes before a seq_cst store are visible to other threads
                            // when we are seq_cst loading a value that was written with seq_cst by the producer thread
                            // from which it can be inferred that a previous store has occurred.
                            // Here, the producer calls wake, where we are at the mercy of the OS as to which
                            // memory model it uses for the atomic store. So we still can't assert the queue
                            // is always non-empty after a wake.
                        } while (entry == nullptr && running.load(std::memory_order_acquire));
                    } else {
                        tparkEndPark(internal_state->tx_park_handle);
                    }
                }

                if (entry == nullptr) {
                    LOG(DEBUG) << "no entry to send, exiting send loop...";
                    break;
                }

                const uint64_t preamble[2] = {
                        network_order_utils::host_to_network(
                                entry->size_bytes + sizeof(uint64_t) // size including the subsequent tag
                                ),
                        network_order_utils::host_to_network(entry->tag)};
                if (sendvp(socket_fd, preamble, sizeof(preamble), MSG_NOSIGNAL) == -1) {
                    LOG(ERR) << "Failed to send preamble for packet with tag " << entry->tag;
                    delete entry;
                    continue;
                }
                size_t n_sent = 0;
                do {
                    const ssize_t i = sendvp(socket_fd, entry->data + n_sent, entry->size_bytes - n_sent, MSG_NOSIGNAL);
                    if (i == 0) {
                        LOG(ERR) << "Connection was closed while sending packet data for packet with tag " << entry->tag
                                 << "; exiting send loop...";
                        if (!interrupt()) {
                            LOG(ERR) << "Failed to interrupt MultiplexedIOSocket";
                        }
                        break;
                    }

                    if (i == -1) {
                        // if we are sending too fast and are exceeding the kernel's send buffer, we need to sleep a
                        // bit. We will re-try sending the same data in the next iteration
                        if (errno == ENOBUFS) {
                            LOG(DEBUG) << "Kernel send buffer is full; Tried to send too much data without opposite "
                                          "peer catching up. Backing off...";
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            continue;
                        }
                        std::string error_message = std::strerror(errno);
                        LOG(ERR) << "Failed to send packet data for packet with tag " << entry->tag
                                 << " with error: " << error_message;
                        break;
                    }
                    n_sent += i;
                } while (n_sent < entry->size_bytes);
                if (entry->is_cloned) {
                    internal_state->tx_allocator.release(entry->data, entry->size_bytes);
                }
                if (entry->done_handle != nullptr) {
                    tparkWake(entry->done_handle);
                }
                delete entry;
            }
            LOG(INFO) << "MultiplexedIOSocket::run() interrupted, exiting send loop...";
        });
    }
    return true;
}

bool tinysockets::MultiplexedIOSocket::receivePacketData(std::span<std::uint8_t> &dst) const {
    if (!(flags & MODE_RX)) {
        LOG(ERR) << "MultiplexedIOSocket::receivePacketData() called on a socket without RX mode";
        return false;
    }
    size_t n_received = 0;
    do {
        const ssize_t i = recvvp(socket_fd, dst.data() + n_received, dst.size_bytes() - n_received, 0);
        if (i == 0 || i == -1) {
            // this == 0 is more important than meets the eye... Linux does not like relying on -1 for EOF
            return false;
        }
        n_received += i;
    } while (n_received < dst.size_bytes() && running && socket_fd != 0);
    return true;
}


bool tinysockets::MultiplexedIOSocket::sendBytes(const uint64_t tag, const std::span<const std::byte> &data,
                                                 const bool clone_memory, tpark_handle_t **pDoneHandleOut) const {
    if (!(flags & MODE_TX)) {
        LOG(ERR) << "MultiplexedIOSocket::sendBytes() called on a socket without TX mode";
        return false;
    }
    if (!running.load(std::memory_order_acquire)) {
        LOG(ERR) << "MultiplexedIOSocket::sendBytes() called on a socket that is not running";
        return false;
    }
    auto *entry = new SendQueueEntry();
    entry->tag = tag;
    entry->size_bytes = data.size_bytes();
    if (clone_memory) {
        auto *cloned_ptr = static_cast<std::byte *>(internal_state->tx_allocator.allocate(data.size_bytes()));
        entry->data = cloned_ptr;
        std::memcpy(cloned_ptr, data.data(), data.size());
    } else {
        entry->data = data.data();
    }
    entry->is_cloned = clone_memory;

    if (pDoneHandleOut != nullptr) {
        entry->done_handle = tparkCreateHandle();
        tparkBeginPark(entry->done_handle);
        *pDoneHandleOut = entry->done_handle;
    }

    {
        if (!internal_state->send_queue.enqueue(entry, true)) {
            LOG(ERR) << "MultiplexedIOSocket::sendBytes() failed to enqueue data; MPSC queue is full";
            return false;
        }
        tparkWake(internal_state->tx_park_handle);
    }
    return true;
}

std::optional<ssize_t> tinysockets::MultiplexedIOSocket::receiveBytesInplace(const uint64_t tag,
                                                                             const std::span<std::byte> &data) const {
    while (true) {
        if (!running.load(std::memory_order_acquire) || socket_fd == 0) {
            return std::nullopt;
        }
        std::shared_lock lock{internal_state->receive_queues_mutex};
        if (!internal_state->receive_queues.contains(tag)) {
            continue;
        }
        const auto &queue = internal_state->receive_queues.at(tag);
        const auto entry_ptr = queue->front();
        if (entry_ptr == nullptr) {
            return 0;
        }
        const auto entry = *entry_ptr;
        queue->pop();

        if (entry.tag != tag) {
            LOG(BUG) << "Obtained packet from SPMCQueue with unexpected tag " << entry.tag << "; expected " << tag;
            continue;
        }
        // ensure buffer is large enough
        if (data.size_bytes() < entry.data_span.size_bytes()) {
            LOG(ERR) << "Buffer is too small to receive data; expected " << entry.data_span.size_bytes()
                     << " bytes but got " << data.size_bytes();
            continue;
        }
        std::memcpy(data.data(), entry.data_span.data(), entry.data_span.size_bytes());

        internal_state->rx_allocator.release(entry.data, entry.data_size);

        return static_cast<ssize_t>(entry.data_span.size_bytes());
    }
}

std::optional<std::unique_ptr<std::byte[]>> tinysockets::MultiplexedIOSocket::receiveBytes(const uint64_t tag,
                                                                                           std::span<std::byte> &data,
                                                                                           const bool no_wait) const {
    if (!(flags & MODE_RX)) {
        LOG(ERR) << "MultiplexedIOSocket::receiveBytes() called on a socket without RX mode";
        return std::nullopt;
    }
    while (true) {
        if (!running.load(std::memory_order_acquire) || socket_fd == 0) {
            return std::nullopt;
        }
        std::shared_lock lock{internal_state->receive_queues_mutex};
        if (!internal_state->receive_queues.contains(tag)) {
            continue;
        }
        const auto &queue = internal_state->receive_queues.at(tag);
        const auto entry_ptr = queue->front();
        if (entry_ptr == nullptr) {
            std::this_thread::yield();
            if (no_wait) {
                return nullptr;
            }
            continue;
        }
        auto entry = *entry_ptr;
        queue->pop();

        if (entry.tag != tag) {
            LOG(BUG) << "Obtained packet from SPMCQueue with unexpected tag " << entry.tag << "; expected " << tag;
            continue;
        }
        data = std::span(entry.data_span.data(), entry.data_span.size_bytes());

        auto data_ptr = std::unique_ptr<std::byte[]>(new std::byte[entry.data_span.size_bytes()]);
        std::memcpy(data_ptr.get(), entry.data_span.data(), entry.data_span.size_bytes());
        internal_state->rx_allocator.release(entry.data, entry.data_size);

        return std::move(data_ptr);
    }
}

void tinysockets::MultiplexedIOSocket::discardReceivedData(const uint64_t tag) const {
    std::unique_lock lock{internal_state->receive_queues_mutex};
    {
        const auto it = internal_state->receive_queues.find(tag);
        if (it == internal_state->receive_queues.end()) {
            return;
        }
        const auto &queue = it->second;
        while (queue->front() != nullptr) {
            queue->pop();
        }
    }
}

bool tinysockets::MultiplexedIOSocket::closeConnection() {
    if (socket_fd == 0) {
        return false;
    }

    // exchange socket fd & set running to false before actually closing it so loop reacts as early as possible
    running.store(false, std::memory_order_release);
    const int socket_fd = this->socket_fd;
    this->socket_fd = 0;

    if (internal_state->tx_park_handle != nullptr) {
        tparkWake(internal_state->tx_park_handle); // wake up tx thread such that it can exit
    }

    // Shut everything down.
    // This is needed to ensure recv() unblock and return an error.
    // Docker is pedantic about this.
    shutdown(socket_fd, SHUT_RDWR);

    // finally, close the socket
    closesocket(socket_fd);
    return true;
}

bool tinysockets::MultiplexedIOSocket::interrupt() {
    if (!running.load(std::memory_order_acquire)) {
        // already interrupted, either through discovery by io threads or external user
        return true;
    }

    running.store(false, std::memory_order_release);
    if (internal_state->tx_park_handle != nullptr) {
        tparkWake(internal_state->tx_park_handle); // wake up tx thread such that it can exit
    }

    // Shutdown both sides of the connection.
    // This is needed to ensure recv() unblocks and return an error.
    // Docker is pedantic about this.
    shutdown(socket_fd, SHUT_RDWR);

    // finally, close the socket
    closesocket(socket_fd);
    return true;
}

void tinysockets::MultiplexedIOSocket::join() {
    if (flags & MODE_RX) {
        if (recv_thread.joinable()) {
            recv_thread.join();
        }
    }
    if (flags & MODE_TX) {
        if (send_thread.joinable()) {
            send_thread.join();
        }
    }
}

bool tinysockets::MultiplexedIOSocket::isOpen() const { return running.load(std::memory_order_acquire); }

const ccoip_socket_address_t &tinysockets::MultiplexedIOSocket::getConnectSockAddr() const { return connect_sockaddr; }

tinysockets::MultiplexedIOSocket::~MultiplexedIOSocket() {
    if (!interrupt()) [[unlikely]] {
        // no way to react to failure in destructor
    }
    join();
    delete internal_state;
}
