#include "tinysockets.hpp"

#include <ccoip_utils.hpp>
#include <pccl_log.hpp>

#include "win_sock_bridge.h"
#include <cstring> // for std::strerror

static bool configure_socket_fd(const int socket_fd) {
    constexpr int opt = 1;

    // enable TCP_NODELAY
    if (setsockoptvp(socket_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) [[
        unlikely]] {
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


tinysockets::BlockingIOSocket::BlockingIOSocket(const ccoip_socket_address_t &address) : socket_fd(0),
    connect_sockaddr(address) {
}

tinysockets::BlockingIOSocket::BlockingIOSocket(const int socket_fd) : socket_fd(socket_fd), connect_sockaddr() {
}

bool tinysockets::BlockingIOSocket::establishConnection() {
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

bool tinysockets::BlockingIOSocket::closeConnection() {
    if (socket_fd == 0) {
        return false;
    }
    shutdown(socket_fd, SHUT_RDWR); // this is needed to ensure recv() unblock and return an error
    closesocket(socket_fd);
    socket_fd = 0;
    return true;
}

bool tinysockets::BlockingIOSocket::isOpen() const {
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

const ccoip_socket_address_t &tinysockets::BlockingIOSocket::getConnectSockAddr() const {
    return connect_sockaddr;
}

bool tinysockets::BlockingIOSocket::sendLtvPacket(const ccoip::packetId_t packet_id,
                                                  const PacketWriteBuffer &buffer) const {
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

std::optional<size_t> tinysockets::BlockingIOSocket::receivePacketLength(const bool no_wait) const {
    uint64_t length;
    size_t n_received = 0;
    do {
        ssize_t i = 0;
        if (no_wait) {
#ifdef WIN32
            // On Windows, MSG_DONTWAIT is not supported, so we have to use ioctlsocket to set the socket to non-blocking mode
            u_long mode = 1;
            if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
                LOG(ERR) << "Failed to set socket into non-blocking mode";
                return std::nullopt;
            }
            i = recvvp(socket_fd, &length, sizeof(length), 0);

            // set socket back to blocking mode
            mode = 0;
            if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
                LOG(ERR) << "Failed to set socket back to blocking mode";
                return std::nullopt;
            }
#elif __APPLE__
            // MacOS does not support MSG_DONTWAIT, so we have to use ioctl to set the socket to non-blocking mode
            int flags = fcntl(socket_fd, F_GETFL, 0);
            if (flags == -1) {
                LOG(ERR) << "Failed to get socket flags";
                return std::nullopt;
            }
            // set to non-blocking mode
            flags |= O_NONBLOCK;
            if (fcntl(socket_fd, F_SETFL, flags) == -1) {
                LOG(ERR) << "Failed to set socket into non-blocking mode";
                return std::nullopt;
            }

            // perform the non-blocking recv
            i = recvvp(socket_fd, &length, sizeof(length), 0);

            // set back to blocking mode
            flags &= ~O_NONBLOCK;
            if (fcntl(socket_fd, F_SETFL, flags) == -1) {
                LOG(ERR) << "Failed to set socket back to blocking mode";
                return std::nullopt;
            }
#else
            // Linux, FreeBSD and other Unix-like systems support MSG_DONTWAIT
            i = recvvp(socket_fd, &length, sizeof(length), MSG_DONTWAIT);
#endif
        } else {
            i = recvvp(socket_fd, &length, sizeof(length), 0);
        }
        if (no_wait) {
            if (i == -1) {
                return std::nullopt;
            }
            if (i == 0) {
                continue; // no data available, but repeat the loop to see if next call fails with EAGAIN/EWOULDBLOCK
            }
        }
        if (i == -1 || i == 0) {
            std::string error_message = std::strerror(errno);
            if (!isOpen()) {
                error_message = "Connection closed";
            }
            LOG(INFO) << "Failed to receive packet length with error: " << error_message;
            return std::nullopt;
        }
        n_received += i;
    } while (n_received < sizeof(length));
    return net_u64_to_host(length);
}

bool tinysockets::BlockingIOSocket::receivePacketData(std::span<std::uint8_t> &dst) const {
    size_t n_received = 0;
    do {
        const ssize_t i = recvvp(socket_fd, dst.data() + n_received, dst.size_bytes() - n_received, 0);
        if (i == -1) {
            const std::string error_message = std::strerror(errno);
            LOG(INFO) << "Failed to receive packet data with error: " << error_message;
            return false;
        }
        n_received += i;
    } while (n_received < dst.size_bytes());
    return true;
}
