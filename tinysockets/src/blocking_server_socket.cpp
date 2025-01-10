#include <ccoip_inet_utils.hpp>
#include <ccoip_utils.hpp>
#include <win_sock_bridge.h>

#include "tinysockets.hpp"

tinysockets::BlockingIOServerSocket::BlockingIOServerSocket(const ccoip_socket_address_t &listen_address)
    : listen_address(listen_address),
      bump_port_on_failure(false),
      socket_fd(0) {
}

tinysockets::BlockingIOServerSocket::BlockingIOServerSocket(const ccoip_inet_address_t &inet_address,
                                                            const uint16_t above_port)
    : listen_address({inet_address, above_port}),
      bump_port_on_failure(true),
      socket_fd(0) {
}

tinysockets::BlockingIOServerSocket::~BlockingIOServerSocket() {
    if (running) {
        if (!interrupt()) [[unlikely]] {
            LOG(ERR) << "Failed to interrupt BlockingIOServerSocket from destructor";
        }
        join();
    } else if (bound && socket_fd != 0) {
        closesocket(socket_fd);
    }
}

static bool configure_socket_fd(const int socket_fd) {
    constexpr int opt = 1;

    // enable TCP_NODELAY
    if (setsockoptvp(socket_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) [[
        unlikely]] {
        LOG(ERR) << "Failed to set TCP_NODELAY option on server socket";
        closesocket(socket_fd);
        return false;
    }

    // enable SO_REUSEADDR if available
#ifndef WIN32
#ifdef SO_REUSEADDR
    if (setsockoptvp(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) [[
        unlikely]] {
        LOG(ERR) << "Failed to set SO_REUSEADDR option on server socket";
        closesocket(socket_fd);
        return false;
    }
#endif
#endif

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

bool tinysockets::BlockingIOServerSocket::listen() {
    if (bound) {
        return false;
    }
    if (listen_address.port == 0) {
        return false;
    }

    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == -1) {
        return false;
    }

    if (!configure_socket_fd(socket_fd)) {
        return false;
    }

    // convert listen ccoip socket addr to sockaddr_in/sockaddr_in6 based on protocol
    sockaddr_in addr_ipv4{};
    sockaddr_in6 addr_ipv6{};
    if (listen_address.inet.protocol == inetIPv4) {
        if (convert_to_sockaddr_ipv4(listen_address, &addr_ipv4) != 0) {
            return false;
        }
    } else if (listen_address.inet.protocol == inetIPv6) {
        if (convert_to_sockaddr_ipv6(listen_address, &addr_ipv6) != 0) {
            return false;
        }
    } else [[unlikely]] {
        return false;
    }

    // bind to socket addr based on protocol
    bool failure = false;
    do {
        // If this isn't the first attempt, destroy the old handle (if any) and create a new one
        if (failure) {
            listen_address.port++;
        }

        // Update sockaddr with the new port
        const sockaddr *sock_addr = nullptr;
        if (listen_address.inet.protocol == inetIPv4) {
            addr_ipv4.sin_port = htons(listen_address.port);
            sock_addr = reinterpret_cast<const sockaddr *>(&addr_ipv4);
        } else {
            addr_ipv6.sin6_port = htons(listen_address.port);
            sock_addr = reinterpret_cast<const sockaddr *>(&addr_ipv6);
        }

        const int bind_result = bind(socket_fd, sock_addr, sizeof(sockaddr_in));
        failure = bind_result != 0;
        if (failure) {
            continue;
        }
        const int listen_result = ::listen(socket_fd, SOMAXCONN);
        failure = listen_result != 0;
        if (failure) {
            LOG(ERR) << "Failed to listen on port " << listen_address.port << " with error: " << std::strerror(errno) <<
                    " (" << errno << ")";
        }
    } while (bump_port_on_failure && failure);

    bound = true;
    return true;
}

bool tinysockets::BlockingIOServerSocket::runAsync() {
    if (running.load()) {
        return false; // Already running
    }
    running.store(true);
    server_thread = std::thread([this] {
        while (running.load()) {
            sockaddr_in client_address{};
            socklen_t client_address_len = sizeof(client_address);
            const int client_socket = accept(socket_fd, reinterpret_cast<sockaddr *>(&client_address),
                                             &client_address_len);
            if (client_socket == 0 || client_socket == -1) {
                if (!running.load() || socket_fd == 0) {
                    break; // Interrupted
                }
            }
            onNewConnection(client_socket, client_address);
        }
    });
    return true;
}

bool tinysockets::BlockingIOServerSocket::interrupt() {
    if (!running.load()) {
        return false; // Not running
    }
    running.store(false);

    shutdown(socket_fd, SHUT_RDWR); // without shutdown, accept() may not get unblocked
    closesocket(socket_fd);

    socket_fd = 0;
    return true;
}

void tinysockets::BlockingIOServerSocket::join() {
    server_thread.join();
}

void tinysockets::BlockingIOServerSocket::setJoinCallback(const BlockingServerSocketJoinCallback &callback) {
    join_callback = callback;
}

uint16_t tinysockets::BlockingIOServerSocket::getListenPort() const {
    if (!bound) {
        return 0;
    }
    return listen_address.port;
}

void tinysockets::BlockingIOServerSocket::onNewConnection(const int client_socket, sockaddr_in sockaddr_in) const {
    auto client_socket_wrapper = std::make_unique<BlockingIOSocket>(client_socket);
    if (join_callback) {
        ccoip_socket_address_t client_addr{};
        convert_from_sockaddr(reinterpret_cast<const sockaddr *>(&sockaddr_in), &client_addr);
        join_callback(client_addr, client_socket_wrapper);
    }
}
