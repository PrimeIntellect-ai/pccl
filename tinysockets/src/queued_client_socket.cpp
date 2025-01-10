#include <ccoip_utils.hpp>
#include <win_sock_bridge.h>

#include "tinysockets.hpp"
#include <cstring> // for std::strerror
#include <condition_variable>
#include <shared_mutex>

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

namespace tinysockets {
    struct ReceivedPacket {
        std::unique_ptr<uint8_t[]> data_ptr;
        std::span<uint8_t> data;
    };

    struct QueuedSocketInternalState {
        std::vector<ReceivedPacket> recv_queue;
        std::mutex mutex; // Protects condition variable and access coordination
        std::condition_variable cond_var;

        QueuedSocketInternalState() : recv_queue() {
        }
    };
}

tinysockets::QueuedSocket::QueuedSocket(const ccoip_socket_address_t &address) : socket_fd(0),
    connect_sockaddr(address),
    internal_state(new QueuedSocketInternalState),
    running(false) {
}

tinysockets::QueuedSocket::QueuedSocket(const int socket_fd) : socket_fd(socket_fd),
                                                               connect_sockaddr(),
                                                               internal_state(new QueuedSocketInternalState),
                                                               running(true) {
}

tinysockets::QueuedSocket::~QueuedSocket() {
    delete internal_state;
    if (running) {
        // ReSharper disable once CppDFAConstantConditions
        if (!interrupt()) [[unlikely]] {
            LOG(ERR) << "Failed to interrupt BlockingIOServerSocket from destructor";
        }
        // Notify all waiting threads to wake up and exit
        {
            std::lock_guard lock(internal_state->mutex);
            internal_state->cond_var.notify_all();
        }
        // ReSharper disable once CppNoDiscardExpression
        join();
    } else if (socket_fd != 0) {
        closesocket(socket_fd);
    }
}

bool tinysockets::QueuedSocket::run() {
    running = true;
    receive_thread = std::thread([this] {
        while (running && socket_fd != 0) {
            const auto length_opt = receivePacketLength();
            if (!length_opt) {
                if (running) {
                    if (isOpen()) {
                        LOG(ERR) << "Failed to receive packet length; closing connection & exiting receive loop...";
                    } else {
                        LOG(ERR) << "Connection was closed; exiting receive loop...";
                    }
                    if (!interrupt()) [[unlikely]] {
                        LOG(ERR) << "Failed to interrupt QueuedSocket";
                    }
                } else {
                    LOG(INFO) << "QueuedSocket::run() interrupted, exiting receive loop...";
                }
                // Notify all waiting threads to wake up and exit
                {
                    std::lock_guard lock(internal_state->mutex);
                    internal_state->cond_var.notify_all();
                }
                return;
            }
            const size_t length = *length_opt;
            if (length == 0) {
                LOG(ERR) << "Received packet with length 0; closing connection";
                if (!interrupt()) [[unlikely]] {
                    LOG(ERR) << "Failed to interrupt QueuedSocket";
                }
                return;
            }
            auto data_ptr = std::make_unique<uint8_t[]>(length);
            std::span data{data_ptr.get(), length};
            if (!receivePacketData(data)) {
                if (running) {
                    LOG(ERR) << "Failed to receive packet data for packet with length " << length;
                    if (!interrupt()) [[unlikely]] {
                        LOG(ERR) << "Failed to interrupt QueuedSocket";
                    }
                } else {
                    LOG(INFO) << "QueuedSocket::run() interrupted, exiting receive loop...";
                }
                return;
            }

            // Enqueue the received packet & notify waiting threads.
            // We lock the mutex during .push() because there is only one producer thread (this one).
            // Predicate-less consumers will use the thread safety of the MPMCQueue to access the queue.
            // Consumers with predicates will lock the mutex to access the queue to iterate over the elements
            // to check for a match.
            // For this to work, we have to ensure that no other thread is pushing to the queue while this is happening.
            {
                ReceivedPacket packet{std::move(data_ptr), data};
                std::lock_guard lock(internal_state->mutex);
                internal_state->recv_queue.push_back(std::move(packet));
                internal_state->cond_var.notify_all();
            }
        }

        // Notify all waiting threads in case the loop exits naturally
        {
            std::lock_guard lock(internal_state->mutex);
            internal_state->cond_var.notify_all();
        }
    });
    return true;
}

void tinysockets::QueuedSocket::join() {
    if (receive_thread.joinable()) {
        receive_thread.join();
    }
}

bool tinysockets::QueuedSocket::interrupt() {
    if (!running) {
        return false;
    }
    std::lock_guard lock(internal_state->mutex); // lock while setting running to false
    running = false;
    shutdown(socket_fd, SHUT_RDWR); // without shutdown, recv() may not get unblocked
    closesocket(socket_fd);
    socket_fd = 0;

    // Notify all waiting threads to wake up and handle interruption
    internal_state->cond_var.notify_all();
    return true;
}

const ccoip_socket_address_t &tinysockets::QueuedSocket::getConnectSockAddr() const {
    return connect_sockaddr;
}

bool tinysockets::QueuedSocket::isOpen() const {
    if (socket_fd == 0) [[unlikely]] {
        return false;
    }

#ifndef WIN32
    // Using MSG_PEEK with a small read: if it returns 0, the connection is closed.
    char buf;
    const ssize_t n = recv(socket_fd, &buf, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) {
        return false;
    }
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No data available, but socket may still be connected
            return true;
        }
        // An error occurred; some errors like ECONNRESET mean the socket is effectively closed
        return false;
    }
    // If we got here, there's data available to read, so the socket is still connected
    return true;
#else
    // set socket into non-blocking mode
    u_long mode = 1;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return false;
    }

    bool is_open = false;
    char buf;
    if (int n = recv(socket_fd, &buf, 1, MSG_PEEK); n == 0) {
        // The connection was closed gracefully by the peer.
        is_open = false;
    } else if (n == SOCKET_ERROR) {
        if (const int err = WSAGetLastError(); err == WSAEWOULDBLOCK) {
            // No data available now, but the socket is still considered open.
            is_open = true;
        } else {
            // Other errors (like WSAECONNRESET) mean the socket is effectively closed.
            is_open = false;
        }
    } else if (n > 0) {
        is_open = true;
    }
    // set socket back to blocking mode
    mode = 0;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return false;
    }
    return is_open;
#endif
}

bool tinysockets::QueuedSocket::establishConnection() {
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
        if (connect(socket_fd, reinterpret_cast<const sockaddr *>(&server_address_ipv6),
                    sizeof(server_address_ipv6)) < 0) {
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

bool tinysockets::QueuedSocket::closeConnection() {
    if (socket_fd == 0) {
        return false;
    }

    std::lock_guard lock(internal_state->mutex);

    shutdown(socket_fd, SHUT_RDWR); // without shutdown, recv() may not get unblocked
    closesocket(socket_fd);
    socket_fd = 0;

    running = false;

    // Notify all waiting threads to wake up and handle closure
    internal_state->cond_var.notify_all();

    return true;
}

std::optional<size_t> tinysockets::QueuedSocket::receivePacketLength() const {
    uint64_t length;
    size_t n_received = 0;
    do {
        const ssize_t i = recvvp(socket_fd, &length, sizeof(length), 0);
        if (i == -1 || i == 0 || !running) {
            std::string error_message = std::strerror(errno);
            if (!isOpen()) {
                error_message = "Connection closed";
            }
            if (running) {
                LOG(ERR) << "Failed to receive packet length with error: " << error_message;
            }
            return std::nullopt;
        }
        n_received += i;
    } while (n_received < sizeof(length));
    return net_u64_to_host(length);
}

bool tinysockets::QueuedSocket::receivePacketData(std::span<std::uint8_t> &dst) const {
    size_t n_received = 0;
    do {
        const ssize_t i = recvvp(socket_fd, dst.data() + n_received, dst.size_bytes() - n_received, 0);
        if (i == -1) {
            const std::string error_message = std::strerror(errno);
            return false;
        }
        n_received += i;
    } while (n_received < dst.size_bytes() && running && socket_fd != 0);
    return true;
}

bool tinysockets::QueuedSocket::sendLtvPacket(ccoip::packetId_t packet_id, const PacketWriteBuffer &buffer) const {
    PacketWriteBuffer tlv_buffer{};
    tlv_buffer.reserve(buffer.size() + sizeof(packet_id) + sizeof(uint64_t));
    tlv_buffer.write<uint64_t>(buffer.size() + sizeof(ccoip::packetId_t));
    tlv_buffer.write(packet_id);
    tlv_buffer.writeContents(buffer.data(), buffer.size());
    int flags = 0;
#ifndef WIN32
    flags |= MSG_NOSIGNAL;
#endif

    int max_buffer_size{};
    socklen_t optlen = sizeof(max_buffer_size);
    if (getsockoptvp(socket_fd, SOL_SOCKET, SO_SNDBUF, &max_buffer_size, &optlen) == -1) [[unlikely]] {
        return false;
    }

    size_t bytes_sent = 0;
    while (bytes_sent < tlv_buffer.size()) {
        const size_t bytes_remaining = tlv_buffer.size() - bytes_sent;
        const size_t to_send = std::min(bytes_remaining, static_cast<size_t>(max_buffer_size));
        const size_t bytes_sent_now = sendvp(socket_fd, tlv_buffer.data() + bytes_sent,
                                             to_send, MSG_NOSIGNAL);
        if (bytes_sent_now == -1) {
            const std::string error_message = std::strerror(errno);
            LOG(INFO) << "Failed to send packet with error: " << error_message;
            return false;
        }
        bytes_sent += bytes_sent_now;
    }
    return true;
}

std::optional<std::pair<std::unique_ptr<uint8_t[]>, std::span<uint8_t> > >
tinysockets::QueuedSocket::pollNextPacketBuffer(const ccoip::packetId_t packet_id, const bool no_wait) const {
    if (!running) {
        return std::nullopt;
    }

    std::unique_lock lock(internal_state->mutex);
    auto &packet_queue = internal_state->recv_queue;

    size_t idx = 0;
    while (true) {
        if (idx >= packet_queue.size()) {
            if (no_wait || !running) {
                return std::nullopt;
            }
            idx = 0; // reset index, as .wait() unlocks internal_state->mutex, where recv_queue can be modified again
            internal_state->cond_var.wait(lock);
            continue;
        }
        auto &packet = packet_queue[idx];

        std::span<uint8_t> data = packet.data;

        if (data.size_bytes() < sizeof(ccoip::packetId_t)) {
            LOG(FATAL) << "Received packet with insufficient length";
            idx++;
            continue;
        }

        PacketReadBuffer buffer{data.data(), data.size()};

        if (const auto received_packet_id = buffer.read<ccoip::packetId_t>(); packet_id != received_packet_id) {
            // Predicate did not match, continue to next packet
            idx++;
            continue;
        }

        // Predicate matched, return the packet and remove it from the queue
        std::unique_ptr<uint8_t[]> data_ptr = std::move(packet.data_ptr);
        packet_queue.erase(packet_queue.begin() + idx);
        return std::make_pair(std::move(data_ptr), data);
    }
}

std::optional<std::pair<std::unique_ptr<uint8_t[]>, std::span<uint8_t> > >
tinysockets::QueuedSocket::pollNextMatchingPacketBuffer(const ccoip::packetId_t packet_id,
                                                        const std::function<bool(const std::span<uint8_t> &)> &
                                                        predicate, const bool no_wait) const {
    if (!running) {
        return std::nullopt;
    }

    std::unique_lock lock(internal_state->mutex);
    auto &packet_queue = internal_state->recv_queue;

    size_t idx = 0;
    while (true) {
        if (idx >= packet_queue.size()) {
            if (no_wait || !running) {
                return std::nullopt;
            }
            idx = 0; // reset index, as .wait() unlocks internal_state->mutex, where recv_queue can be modified again
            internal_state->cond_var.wait(lock);
            continue;
        }
        auto &packet = packet_queue[idx];

        std::span<uint8_t> data = packet.data;

        if (data.size_bytes() < sizeof(ccoip::packetId_t)) {
            LOG(FATAL) << "Received packet with insufficient length";
            idx++;
            continue;
        }

        PacketReadBuffer buffer{data.data(), data.size()};

        if (const auto received_packet_id = buffer.read<ccoip::packetId_t>(); packet_id != received_packet_id) {
            // Predicate did not match, continue to next packet
            idx++;
            continue;
        }

        if (predicate(data)) {
            std::unique_ptr<uint8_t[]> data_ptr = std::move(packet.data_ptr);
            packet_queue.erase(packet_queue.begin() + idx);
            return std::make_pair(std::move(data_ptr), data);
        }

        // Predicate did not match, continue to next packet
        idx++;
    }
}
