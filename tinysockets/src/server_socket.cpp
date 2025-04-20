#include <ccoip_inet_utils.hpp>
#include <ccoip_types.hpp>

#include <uv.h>
#include "tinysockets.hpp"

#include <iostream>

#include "ccoip_utils.hpp"

void uv_err_check(const int status) {
    if (status < 0) {
        fprintf(stderr, "UV Error: %s\n", uv_strerror(status));
        exit(1);
    }
}

#define UV_ERR_CHECK(status) uv_err_check(status)

#define KEEP_ALIVE_SECONDS 60

namespace tinysockets {
    struct RecvBuffer {
        /// The expected length of the packet
        uint64_t expected_length = -1;

        /// The current buffer
        std::vector<uint8_t> buffer{};
    };

    struct ServerSocketState {
        std::unique_ptr<uv_loop_s> loop;
        std::unique_ptr<uv_tcp_s> tcp_server;
        std::unique_ptr<uv_async_s> async_handle;

        /// Maps the client socket addresses to the respective uv stream
        std::unordered_map<internal_inet_socket_address_t, uv_stream_s *> sockaddr_to_uvstream{};
        /// Maps the uv stream to the respective client socket addresses
        std::unordered_map<uv_handle_t *, internal_inet_socket_address_t> uvstream_to_sockaddr{};

        /// List of callbacks invoked on client read
        std::vector<ServerSocketReadCallback> read_callbacks{};

        /// List of callbacks invoked on client close
        std::vector<ServerSocketCloseCallback> close_callbacks{};

        /// List of callbacks invoked on client join
        std::vector<ServerSocketJoinCallback> join_callbacks{};

        /// Maps client socket addresses to the respective packet buffers.
        /// The ServerSocket asserts that the packet starts with a 64-bit length field
        /// and will concatenate the received data until the expected length is reached.
        /// This server is not fully LTV (length type value) aware, but it does assert
        /// that the first 8 bytes are a length field.
        std::unordered_map<internal_inet_socket_address_t, RecvBuffer> current_recv_buffers{};
    };
}

tinysockets::ServerSocket::ServerSocket(const ccoip_socket_address_t &listen_address)
    : listen_address(listen_address),
      bump_port_on_failure(false),
      server_socket_state(new ServerSocketState{}) {
}

tinysockets::ServerSocket::ServerSocket(const ccoip_inet_address_t &inet_address,
                                        const uint16_t above_port) : listen_address({inet_address, above_port}),
                                                                     bump_port_on_failure(true),
                                                                     server_socket_state(new ServerSocketState{}) {
}

bool tinysockets::ServerSocket::listen() {
    if (bound) {
        return false;
    }
    if (listen_address.port == 0) {
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

    server_socket_state->loop = std::make_unique<uv_loop_t>();
    UV_ERR_CHECK(uv_loop_init(server_socket_state->loop.get()));

    server_socket_state->tcp_server = std::make_unique<uv_tcp_t>();
    server_socket_state->tcp_server->data = this;
    UV_ERR_CHECK(uv_tcp_init(server_socket_state->loop.get(), server_socket_state->tcp_server.get()));

    // bind to socket addr based on protocol
    bool failure = false;
    do {
        // If this isn't the first attempt, destroy the old handle (if any) and create a new one
        if (failure) {
            listen_address.port++;
            LOG(INFO) << "Bind to port failed, bumping port to " << listen_address.port;

            uv_close(reinterpret_cast<uv_handle_t *>(server_socket_state->tcp_server.get()), nullptr);

            // Run the loop once to let the handle finish closing
            // This is necessary because uv_close is asynchronous and the handle is not guaranteed to be closed.
            // Otherwise, we would reinitialize the tcp_server too early, leading to its deletion
            // before uv has finished closing it.
            while (uv_run(server_socket_state->loop.get(), UV_RUN_NOWAIT) != 0) {
            }

            // Now it is safe to re-initialize the tcp handle
            server_socket_state->tcp_server = std::make_unique<uv_tcp_t>();
            server_socket_state->tcp_server->data = this;
            if (uv_tcp_init(server_socket_state->loop.get(), server_socket_state->tcp_server.get()) != 0) {
                LOG(ERR) << "Failed to re-initialize TCP handle.";
                return false;
            }
        }

        // Update sockaddr with the new port
        const sockaddr *sock_addr = nullptr;
        if (listen_address.inet.protocol == inetIPv4) {
            addr_ipv4.sin_port = htons(listen_address.port);
            sock_addr = reinterpret_cast<const sockaddr*>(&addr_ipv4);
        } else {
            addr_ipv6.sin6_port = htons(listen_address.port);
            sock_addr = reinterpret_cast<const sockaddr*>(&addr_ipv6);
        }

        const int bind_result = uv_tcp_bind(server_socket_state->tcp_server.get(), sock_addr, 0);
        failure = bind_result != 0;
        if (failure) {
            LOG(INFO) << "uv_tcp_bind failed with error: " << uv_strerror(bind_result) << " (" << bind_result << ")";
            continue;
        }

        const int listen_result = uv_listen(reinterpret_cast<uv_stream_t*>(server_socket_state->tcp_server.get()), SOMAXCONN,
            [](uv_stream_t *server, const int status) {
                auto *this_ptr = static_cast<ServerSocket *>(server->data);
                this_ptr->onNewConnection(reinterpret_cast<uv_server_stream_t *>(server), status);
            });
        failure = (listen_result != 0);
        if (failure) {
            LOG(INFO) << "uv_listen(port=" << listen_address.port << ", backlog=" << SOMAXCONN << ") failed with error: " << uv_strerror(listen_result) << " (" << listen_result << ")";
        }
    } while (bump_port_on_failure && failure);

    server_socket_state->async_handle = std::make_unique<uv_async_t>();
    server_socket_state->async_handle->data = this;
    UV_ERR_CHECK(
        uv_async_init(server_socket_state->loop.get(), server_socket_state->async_handle.get(), [](uv_async_t *handle) {
                const auto *this_ptr = static_cast<ServerSocket *>(handle->data);
                this_ptr->onAsyncSignal();
            }
        ));
    if (failure) {
        return false;
    }

    bound = true;
    return true;
}


bool tinysockets::ServerSocket::runAsync() {
    if (running) {
        return false;
    }
    server_thread = std::thread([this] {
        UV_ERR_CHECK(uv_run(server_socket_state->loop.get(), UV_RUN_DEFAULT));
        performLoopShutdown();
    });
    running = true;
    return true;
}

bool tinysockets::ServerSocket::interrupt() {
    if (!running) {
        return false;
    }
    if (interrupted) {
        return true;
    }
    uv_async_send(server_socket_state->async_handle.get());
    interrupted = true;
    return true;
}

void tinysockets::ServerSocket::join() {
    if (server_thread.joinable()) {
        server_thread.join();
    }
}

void tinysockets::ServerSocket::addReadCallback(const ServerSocketReadCallback &callback) const {
    server_socket_state->read_callbacks.push_back(callback);
}

void tinysockets::ServerSocket::addCloseCallback(const ServerSocketCloseCallback &callback) const {
    server_socket_state->close_callbacks.push_back(callback);
}

void tinysockets::ServerSocket::addJoinCallback(const ServerSocketJoinCallback &callback) const {
    server_socket_state->join_callbacks.push_back(callback);
}

bool tinysockets::ServerSocket::closeClientConnection(const ccoip_socket_address_t &client_address) const {
    if (!running) {
        return false;
    }
    const auto inet_internal = ccoip_socket_to_internal(client_address);
    if (const auto it = server_socket_state->sockaddr_to_uvstream.find(inet_internal);
        it != server_socket_state->sockaddr_to_uvstream.end()) {
        if (!uv_is_closing(reinterpret_cast<uv_handle_t *>(it->second))) {
            uv_close(reinterpret_cast<uv_handle_t *>(it->second), [](uv_handle_t *handle) {
                const auto *this_ptr = static_cast<ServerSocket *>(handle->data);
                this_ptr->onClientClose(handle);
            });
        }
        return true;
    }
    return false;
}

struct write_req_t {
    uv_write_t req;
    uv_buf_t buf;
};

bool tinysockets::ServerSocket::sendRawPacket(const ccoip_socket_address_t &client_address,
                                              const std::span<const std::byte> &buffer) {
    if (!running) {
        return false;
    }
    if (interrupted) {
        return false;
    }

    const auto inet_internal = ccoip_socket_to_internal(client_address);
    const auto it = server_socket_state->sockaddr_to_uvstream.find(inet_internal);
    if (it == server_socket_state->sockaddr_to_uvstream.end()) {
        return false;
    }

    auto *wr = new write_req_t;
    wr->buf = uv_buf_init(static_cast<char *>(std::malloc(buffer.size_bytes())), static_cast<unsigned int>(buffer.size_bytes()));
    std::memcpy(wr->buf.base, buffer.data(), buffer.size_bytes());
    wr->req.data = this;

    // free memory and delete write_req_t on write completion
    const int write_status = uv_write(&wr->req, it->second, &wr->buf, 1, [](uv_write_t *req, int) {
        const auto *write_req = reinterpret_cast<write_req_t *>(req);
        std::free(write_req->buf.base);
        delete write_req;
    });

    if (write_status != 0) {
        std::free(wr->buf.base);
        delete wr;
        return false;
    }

    return true;
}

bool tinysockets::ServerSocket::closeAllClientConnections() const {
    if (!running) {
        return false;
    }
    for (const auto &[_, stream]: server_socket_state->sockaddr_to_uvstream) {
        if (!uv_is_closing(reinterpret_cast<uv_handle_t *>(stream))) {
            uv_close(reinterpret_cast<uv_handle_t *>(stream), [](uv_handle_t *handle) {
                const auto *this_ptr = static_cast<ServerSocket *>(handle->data);
                this_ptr->onClientClose(handle);
            });
        }
    }
    return true;
}

std::thread::id tinysockets::ServerSocket::getServerThreadId() const {
    if (!running) {
        return std::thread::id{};
    }
    return server_thread.get_id();
}

uint16_t tinysockets::ServerSocket::getListenPort() const {
    if (!bound) {
        return 0;
    }
    return listen_address.port;
}

void tinysockets::ServerSocket::onAsyncSignal() const {
    uv_stop(server_socket_state->async_handle->loop);
}

static void createBuffer(uv_handle_t *, const size_t suggested_size, uv_buf_t *buf) {
    buf->base = new char[suggested_size];
    buf->len = suggested_size;
}

std::optional<ccoip_socket_address_t> tinysockets::ServerSocket::getUvStreamAddressCached(uv_stream_t *stream) const {
    const auto inet_internal = server_socket_state->uvstream_to_sockaddr.find(reinterpret_cast<uv_handle_t *>(stream));
    if (inet_internal == server_socket_state->uvstream_to_sockaddr.end()) {
        return std::nullopt;
    }
    return internal_to_ccoip_sockaddr(inet_internal->second);
}

void tinysockets::ServerSocket::performLoopShutdown() const {
    // Close the server and async handles
    uv_close(reinterpret_cast<uv_handle_t *>(server_socket_state->tcp_server.get()), nullptr);
    uv_close(reinterpret_cast<uv_handle_t *>(server_socket_state->async_handle.get()), nullptr);

    // If needed, close client connections as well before the loop is closed
    if (!closeAllClientConnections()) {
        LOG(ERR) << "Failed to close all clients connections";
    }

    // Run the loop until all pending close callbacks are processed
    while (uv_run(server_socket_state->loop.get(), UV_RUN_DEFAULT) != 0) {
        // This will spin until all handles are closed
    }

    // Now that all handles are closed, uv_loop_close should not return UV_EBUSY
    if (const int status = uv_loop_close(server_socket_state->loop.get()); status != 0) {
        LOG(ERR) << "uv_loop_close failed with: " << uv_err_name(status);
    }

    // Set the loop pointer to null after it's closed
    server_socket_state->loop = nullptr;
}

static std::optional<ccoip_socket_address_t> getUvStreamAddress(uv_stream_t *stream) {
    // get client address
    sockaddr_storage addr{};
    int addr_len = sizeof(addr);
    if (uv_tcp_getpeername(reinterpret_cast<uv_tcp_t *>(stream), reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) [
        [unlikely]] {
        return std::nullopt;
    }
    ccoip_socket_address_t client_addr{};
    convert_from_sockaddr(reinterpret_cast<const sockaddr *>(&addr), &client_addr);
    return client_addr;
}

void tinysockets::ServerSocket::onNewConnection(uv_server_stream_t *server, const int status) {
    if (status < 0) {
        return;
    }
    if (interrupted) {
        // don't accept more connections when the server has been interrupted
        return;
    }
    auto *client = new uv_tcp_t{};
    UV_ERR_CHECK(uv_tcp_init(server_socket_state->loop.get(), client));
    client->data = this;
    if (uv_accept(reinterpret_cast<uv_stream_t *>(server), reinterpret_cast<uv_stream_t *>(client)) == 0) {
        // the kernel should start probing if a connection is dead on our behalf after ~60 seconds
        uv_tcp_keepalive(client, 1, KEEP_ALIVE_SECONDS);

        LOG(INFO) << "New connection accepted";

        const auto client_addr = getUvStreamAddress(reinterpret_cast<uv_stream_t *>(client));

        if (!client_addr) [[unlikely]] {
            LOG(ERR) << "Failed to get client address";
            if (!uv_is_closing(reinterpret_cast<uv_handle_t *>(client))) {
                uv_close(reinterpret_cast<uv_handle_t *>(client), [](uv_handle_t *handle) {
                    const auto *uv_tcp = reinterpret_cast<uv_tcp_t *>(handle);
                    delete uv_tcp;
                });
            }
            return;
        }
        LOG(INFO) << "New connection client address: " << ccoip_sockaddr_to_str(*client_addr);

        const auto inet_internal = ccoip_socket_to_internal(*client_addr);
        server_socket_state->sockaddr_to_uvstream[inet_internal] = reinterpret_cast<uv_stream_t *>(client);
        server_socket_state->uvstream_to_sockaddr[reinterpret_cast<uv_handle_t *>(client)] = inet_internal;

        uv_read_start(reinterpret_cast<uv_stream_t *>(client), createBuffer,
                      [](uv_stream_t *stream, const ssize_t n_read, const uv_buf_t *buf) {
                          // Handle EOF or errors
                          if (n_read < 0) {
                              // Free the buffer
                              delete[] buf->base;

                              // Close the connection properly
                              uv_close(reinterpret_cast<uv_handle_t *>(stream), [](uv_handle_t *handle) {
                                  const auto *this_ptr = static_cast<ServerSocket *>(handle->data);
                                  this_ptr->onClientClose(handle);
                              });
                              return;
                          }

                          // invoke onClientRead
                          {
                              const auto *this_ptr = static_cast<ServerSocket *>(stream->data);
                              this_ptr->onClientRead(stream, n_read, buf);
                          }
                      });

        // invoke join callbacks
        for (const auto &callback: server_socket_state->join_callbacks) {
            callback(*client_addr);
        }
    } else {
        if (!uv_is_closing(reinterpret_cast<uv_handle_t *>(client))) {
            uv_close(reinterpret_cast<uv_handle_t *>(client), [](uv_handle_t *handle) {
                const auto *uv_tcp = reinterpret_cast<uv_tcp_t *>(handle);
                delete uv_tcp;
            });
        }
    }
}

void tinysockets::ServerSocket::onClientRead(uv_stream_t *stream, const ssize_t n_read, const uv_buf_t *buf) const {
    const auto client_addr = getUvStreamAddressCached(stream);
    if (!client_addr) [[unlikely]] {
        LOG(ERR) << "Failed to get client address";
        uv_close(reinterpret_cast<uv_handle_t *>(stream), nullptr);
        return;
    }

    // manage current recv buffer
    {
        const std::span data(reinterpret_cast<uint8_t *>(buf->base), n_read);
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);

        auto &current_recv_buffer = server_socket_state->current_recv_buffers[
            ccoip_socket_to_internal(*client_addr)];

        while (buffer.remaining() > 0) {
            if (current_recv_buffer.expected_length == -1) {
                // if the expected length is not set, read the length field
                if (buffer.remaining() < sizeof(uint64_t)) {
                    LOG(ERR) << "Expected 8-byte length field but got " << data.size_bytes() << " bytes";
                    if (!closeClientConnection(*client_addr)) [[unlikely]] {
                        LOG(ERR) << "Failed to close client connection";
                    }
                    return;
                }
                current_recv_buffer.expected_length = buffer.read<uint64_t>();
            }

            const size_t n_to_insert = std::min(buffer.remaining(),
                                                static_cast<size_t>(
                                                    current_recv_buffer.expected_length - current_recv_buffer.buffer.
                                                    size()));
            current_recv_buffer.buffer.resize(current_recv_buffer.buffer.size() + n_to_insert);

            buffer.readContents(
                current_recv_buffer.buffer.data() + current_recv_buffer.buffer.size() - n_to_insert,
                n_to_insert
            );

            if (current_recv_buffer.buffer.size() == current_recv_buffer.expected_length) {
                // if the buffer is full, invoke read callbacks
                for (const auto &callback: server_socket_state->read_callbacks) {
                    callback(*client_addr, current_recv_buffer.buffer);
                }
                current_recv_buffer.expected_length = -1;
                current_recv_buffer.buffer.clear();
            }
        }
    }
    delete[] buf->base;
}

void tinysockets::ServerSocket::onClientClose(uv_handle_t *handle) const {
    auto *client = reinterpret_cast<uv_tcp_t *>(handle);
    const auto *this_ptr = static_cast<ServerSocket *>(client->data);
    const auto client_addr = getUvStreamAddressCached(reinterpret_cast<uv_stream_t *>(client));

    if (!client_addr) [[unlikely]] {
        LOG(ERR) << "Failed to get client address";
        return;
    }

    // Remove from connections map
    const auto inet_internal = ccoip_socket_to_internal(*client_addr);
    this_ptr->server_socket_state->sockaddr_to_uvstream.erase(inet_internal);
    this_ptr->server_socket_state->uvstream_to_sockaddr.erase(handle);

    for (const auto &callback: this_ptr->server_socket_state->close_callbacks) {
        callback(*client_addr);
    }
    delete client;
}

tinysockets::ServerSocket::~ServerSocket() {
    if (!running && server_socket_state->loop != nullptr) {
        performLoopShutdown();
    }
    if (running && !interrupted) {
        // ReSharper disable once CppDFAConstantConditions
        if (!interrupt()) [[unlikely]] {
            LOG(ERR) << "Failed to interrupt ServerSocket from destructor";
        }
        join();
    }
    delete server_socket_state;
}
