#pragma once

#include <ccoip_inet.h>
#include <ccoip_packet.hpp>
#include <pccl_log.hpp>

#include <optional>
#include <functional>
#include <thread>
#include <span>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <threadpark.h>

#ifdef WIN32
typedef long long int ssize_t;
struct sockaddr_in;
#else
#include <sys/types.h>
#include <netinet/in.h>
#endif

struct uv_server_stream_t;
struct uv_stream_s;
struct uv_buf_t;
struct uv_handle_s;

namespace tinysockets {
    struct ServerSocketState;

    using ServerSocketReadCallback = std::function<void(ccoip_socket_address_t, const std::span<std::uint8_t> &)>;
    using ServerSocketJoinCallback = std::function<void(const ccoip_socket_address_t &)>;
    using ServerSocketCloseCallback = std::function<void(const ccoip_socket_address_t &)>;

    class ServerSocket final {
    private:
        ccoip_socket_address_t listen_address;
        bool bump_port_on_failure;

        ServerSocketState *server_socket_state;
        std::thread server_thread;

        bool bound = false;
        bool running = false;
        bool interrupted = false;

    public:
        /// binds to the specified socket address
        explicit ServerSocket(const ccoip_socket_address_t &listen_address);

        /// bind to any free port above the given port
        explicit ServerSocket(const ccoip_inet_address_t &inet_address, uint16_t above_port);

        ServerSocket(const ServerSocket &other) = delete;

        ServerSocket(ServerSocket &&other) = delete;

        ServerSocket &operator=(const ServerSocket &other) = delete;

        ServerSocket &operator=(ServerSocket &&other) = delete;

        /// Returns false if already bound or if listen_address is invalid
        [[nodiscard]] bool listen();

        /// Returns false if already running
        [[nodiscard]] bool runAsync();

        /// Returns false if not running
        [[nodiscard]] bool interrupt();

        /// Wait for the server thread to exit
        void join();

        /// Add a callback to be called when new data is received from a client
        void addReadCallback(const ServerSocketReadCallback &callback) const;

        /// Add a callback to be called when a client connection is closed
        void addCloseCallback(const ServerSocketCloseCallback &callback) const;

        /// Add a callback to be called when a client joins
        void addJoinCallback(const ServerSocketJoinCallback &callback) const;

        /// Closes the client connection associated with the given socket address
        /// Returns false if the client connection does not exist or if the server is not running
        [[nodiscard]] bool closeClientConnection(const ccoip_socket_address_t &client_address) const;

        /// Closes all client connections
        [[nodiscard]] bool closeAllClientConnections() const;

        /// Returns the thread ID of the server thread
        [[nodiscard]] std::thread::id getServerThreadId() const;

        /// Returns the port the server is listening on; returns 0 if not listening
        [[nodiscard]] uint16_t getListenPort() const;

        ~ServerSocket();

        // Packet decoding / encoding functions

        /// Receives a packet from the client associated with the given socket address
        /// Returns std::nullopt if the server is not running or packet reception fails
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receivePacket(PacketReadBuffer &buffer) {
            const ccoip::packetId_t id = T::packet_id;
            return receiveLtvPacket<T>(id, buffer);
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveLtvPacket(const ccoip::packetId_t packet_id, PacketReadBuffer &buffer) {
            if (const auto actual_packet_id = buffer.read<ccoip::packetId_t>(); actual_packet_id != packet_id) {
                LOG(ERR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            T packet{};
            if (!packet.deserialize(buffer)) {
                LOG(ERR) << "Failed to deserialize packet with ID " << packet_id;
                return std::nullopt;
            }
            return packet;
        }

        /// Sends a packet to the client associated with the given socket address
        /// Returns false if the server is not running or the client connection does not exist
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendPacket(const ccoip_socket_address_t &client_address, const T &packet) {
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendLtvPacket(client_address, id, buffer);
        }

        [[nodiscard]] bool sendLtvPacket(const ccoip_socket_address_t &client_address,
                                         const ccoip::packetId_t packet_id,
                                         const PacketWriteBuffer &buffer) {
            PacketWriteBuffer complete_packet{};
            complete_packet.write<uint64_t>(buffer.size() + sizeof(ccoip::packetId_t));
            complete_packet.write(packet_id);
            complete_packet.writeContents(buffer.data(), buffer.size());
            return sendRawPacket(client_address,
                                 std::span(reinterpret_cast<const std::byte *>(complete_packet.data()),
                                           complete_packet.size()));
        }

        [[nodiscard]] bool sendRawPacket(const ccoip_socket_address_t &client_address,
                                         const std::span<const std::byte> &buffer);

    private:
        void onAsyncSignal() const;

        void onNewConnection(uv_server_stream_t *server, int status);

        void onClientRead(uv_stream_s *stream, ssize_t n_read, const uv_buf_t *buf) const;

        void onClientClose(uv_handle_s *handle) const;

        [[nodiscard]] std::optional<ccoip_socket_address_t> getUvStreamAddressCached(uv_stream_s *stream) const;

        void performLoopShutdown() const;
    };

    class BlockingIOSocket final {
        int socket_fd;
        ccoip_socket_address_t connect_sockaddr;

        std::mutex send_mutex;
        std::mutex recv_mutex;

    public:
        explicit BlockingIOSocket(const ccoip_socket_address_t &address);

        explicit BlockingIOSocket(int socket_fd);

        BlockingIOSocket(const BlockingIOSocket &other) = delete;

        BlockingIOSocket(BlockingIOSocket &&other) = delete;

        BlockingIOSocket &operator=(const BlockingIOSocket &other) = delete;

        BlockingIOSocket &operator=(BlockingIOSocket &&other) = delete;

        [[nodiscard]] bool establishConnection();

        [[nodiscard]] bool enableReceiveTimout(int seconds) const;

        /// Closes the socket.
        /// @param allow_data_discard if true, will perform an instant shutdown without lingering
        [[nodiscard]] bool closeConnection(bool allow_data_discard = false);

        [[nodiscard]] bool isOpen();

        [[nodiscard]] const ccoip_socket_address_t &getConnectSockAddr() const;

        [[nodiscard]] int getSocketFd() const;

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendPacket(const T &packet) {
            std::lock_guard guard{send_mutex};
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendLtvPacket(id, buffer);
        }

        /// Receives a packet of the specified type
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receivePacket(const bool no_wait = false) {
            std::lock_guard guard{recv_mutex};
            const ccoip::packetId_t id = T::packet_id;
            return receiveLtvPacket<T>(id, no_wait);
        }

        // Packet decoding / encoding functions

        /// Sends a packet to the connected socket
        [[nodiscard]] bool sendLtvPacket(ccoip::packetId_t packet_id, const PacketWriteBuffer &buffer) const;

        /// Attempts to grow the send-buffer to the maximum size
        void maximizeSendBuffer() const;

        /// Attempts to grow the receive-buffer to the maximum size
        void maximizeReceiveBuffer() const;

        /// Receives n bytes from the socket and writes them into dst
        [[nodiscard]] ssize_t receiveRawData(std::span<std::byte> &dst, size_t n_bytes);

    private:
        // Packet decoding / encoding functions
        [[nodiscard]] std::optional<size_t> receivePacketLength(bool no_wait) const;

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveLtvPacket(const ccoip::packetId_t packet_id, const bool no_wait) {
            const std::optional<size_t> length_opt = receivePacketLength(no_wait);
            if (!length_opt) {
                return std::nullopt;
            }
            const size_t length = *length_opt;
            const std::unique_ptr<std::byte[]> data_ptr{new std::byte[length]};
            std::span data{data_ptr.get(), length};
            if (receiveRawData(data, data.size_bytes()) != data.size_bytes()) {
                LOG(ERR) << "Failed to receive packet data for packet ID " << packet_id << " with length " << data.
                        size();
                return std::nullopt;
            }
            PacketReadBuffer buffer{reinterpret_cast<uint8_t *>(data.data()), data.size()};
            if (const auto actual_packet_id = buffer.read<ccoip::packetId_t>(); actual_packet_id != packet_id) {
                LOG(ERR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            T packet{};
            if (!packet.deserialize(buffer)) {
                LOG(ERR) << "Failed to deserialize packet with ID " << packet_id;
                return std::nullopt;
            }
            return packet;
        }
    };

    struct QueuedSocketInternalState;

    /// A socket with a designated thread for sending packets and a receive-queue.
    /// Sending is threadsafe and can be performed from multiple threads, however sending results in locking.
    /// Receives occur on the designated thread and enqueued for the user to retrieve. Functions like @code receivePacket @endcode
    /// poll this queue.
    class QueuedSocket final {
        int socket_fd;
        ccoip_socket_address_t connect_sockaddr;
        std::mutex send_mutex;
        std::thread receive_thread;
        QueuedSocketInternalState *internal_state;
        volatile bool running;

    public:
        explicit QueuedSocket(const ccoip_socket_address_t &address);

        explicit QueuedSocket(int socket_fd);

        QueuedSocket(const QueuedSocket &other) = delete;

        QueuedSocket(QueuedSocket &&other) = delete;

        QueuedSocket &operator=(const QueuedSocket &other) = delete;

        QueuedSocket &operator=(QueuedSocket &&other) = delete;

        ~QueuedSocket();

        [[nodiscard]] bool establishConnection();

        [[nodiscard]] bool run();

        void join();

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool closeConnection();

        [[nodiscard]] bool isOpen() const;

        [[nodiscard]] const ccoip_socket_address_t &getConnectSockAddr() const;

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendPacket(const T &packet) {
            std::lock_guard guard{send_mutex};
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendLtvPacket(id, buffer);
        }

        /// Receives a packet of the specified type
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receivePacket(const bool no_wait = false) {
            const ccoip::packetId_t id = T::packet_id;
            return pollNextPacket<T>(id, no_wait);
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveMatchingPacket(const std::function<bool(const T &)> &predicate,
                                                             const bool no_wait = false) {
            const ccoip::packetId_t id = T::packet_id;
            return pollNextMatchingPacket<T>(id, predicate, no_wait);
        }

    private:
        // Packet decoding / encoding functions
        [[nodiscard]] bool sendLtvPacket(ccoip::packetId_t packet_id, const PacketWriteBuffer &buffer) const;

        [[nodiscard]] std::optional<size_t> receivePacketLength() const;

        [[nodiscard]] bool receivePacketData(std::span<std::uint8_t> &dst) const;

        [[nodiscard]] std::optional<std::pair<std::unique_ptr<uint8_t[]>, std::span<uint8_t>>> pollNextPacketBuffer(
            ccoip::packetId_t packet_id,
            bool no_wait) const;

        [[nodiscard]] std::optional<std::pair<std::unique_ptr<uint8_t[]>, std::span<uint8_t>>>
        pollNextMatchingPacketBuffer(
            ccoip::packetId_t packet_id, const std::function<bool(const std::span<uint8_t> &)> &predicate,
            bool no_wait) const;

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> pollNextPacket(const ccoip::packetId_t packet_id, const bool no_wait) {
            const auto pair = pollNextPacketBuffer(packet_id, no_wait);
            if (!pair) {
                return std::nullopt;
            }
            const auto &[data_ptr, data] = *pair;
            return parsePacket<T>(packet_id, data);
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> pollNextMatchingPacket(const ccoip::packetId_t packet_id,
                                                              const std::function<bool(const T &)> &predicate,
                                                              const bool no_wait) {
            const auto pair = pollNextMatchingPacketBuffer(
                packet_id, [packet_id, predicate](const std::span<uint8_t> &data) {
                    PacketReadBuffer buffer{data.data(), data.size()};
                    if (packet_id != buffer.read<ccoip::packetId_t>()) {
                        return false; // no need to further deserialize, packet ID does not match
                    }
                    T packet{};
                    if (!packet.deserialize(buffer)) {
                        LOG(ERR) << "Failed to deserialize packet with ID " << packet_id;
                        return false;
                    }
                    return predicate(packet);
                }, no_wait);
            if (!pair) {
                return std::nullopt;
            }
            const auto &[data_ptr, data] = *pair;
            return parsePacket<T>(packet_id, data);
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> parsePacket(const ccoip::packetId_t packet_id,
                                                   const std::span<std::uint8_t> &packet_data) {
            PacketReadBuffer buffer{packet_data.data(), packet_data.size()};
            if (const auto actual_packet_id = buffer.read<ccoip::packetId_t>(); actual_packet_id != packet_id) {
                LOG(ERR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            T packet{};
            if (!packet.deserialize(buffer)) {
                LOG(ERR) << "Failed to deserialize packet with ID " << packet_id;
                return std::nullopt;
            }
            return packet;
        }
    };

    using BlockingServerSocketJoinCallback = std::function<void(const ccoip_socket_address_t &,
                                                                std::unique_ptr<BlockingIOSocket> &)>;

    class BlockingIOServerSocket final {
    private:
        ccoip_socket_address_t listen_address;
        bool bump_port_on_failure;
        std::thread server_thread;

        bool bound = false;
        bool interrupted = false;
        std::atomic<bool> running{false};

        int socket_fd;

        BlockingServerSocketJoinCallback join_callback = nullptr;

    public:
        /// binds to the specified socket address
        explicit BlockingIOServerSocket(const ccoip_socket_address_t &listen_address);

        /// bind to any free port above the given port
        explicit BlockingIOServerSocket(const ccoip_inet_address_t &inet_address, uint16_t above_port);

        BlockingIOServerSocket(const BlockingIOServerSocket &other) = delete;

        BlockingIOServerSocket(BlockingIOServerSocket &&other) = delete;

        BlockingIOServerSocket &operator=(const BlockingIOServerSocket &other) = delete;

        BlockingIOServerSocket &operator=(BlockingIOServerSocket &&other) = delete;

        ~BlockingIOServerSocket();

        /// Returns false if already bound or if listen_address is invalid
        [[nodiscard]] bool listen();

        /// Returns false if already running
        [[nodiscard]] bool runAsync();

        /// Returns false if not running
        [[nodiscard]] bool interrupt();

        /// Wait for the server thread to exit
        void join();

        /// Sets the callback to be called when a client joins
        void setJoinCallback(const BlockingServerSocketJoinCallback &callback);

        /// Returns the port the server is listening on; returns 0 if not listening
        [[nodiscard]] uint16_t getListenPort() const;

    private:
        /// Called when a new connection is established
        void onNewConnection(int client_socket, sockaddr_in sockaddr_in) const;
    };

    struct MultiplexedIOSocketInternalState;

    enum ConnectionModeFlags {
        MODE_TX = 0x1,
        MODE_RX = 0x2,
    };

    class MultiplexedIOSocket final {
        std::atomic<bool> running = false;

        volatile int socket_fd;
        ccoip_socket_address_t connect_sockaddr;

        ConnectionModeFlags flags;

        std::thread recv_thread;
        std::thread send_thread;

        MultiplexedIOSocketInternalState *internal_state;

    public:
        explicit MultiplexedIOSocket(const ccoip_socket_address_t &address, ConnectionModeFlags flags);

        explicit MultiplexedIOSocket(int socket_fd, ConnectionModeFlags flags);

        explicit MultiplexedIOSocket(int socket_fd, const ccoip_socket_address_t &address, ConnectionModeFlags flags);

        MultiplexedIOSocket(const MultiplexedIOSocket &other) = delete;

        MultiplexedIOSocket(MultiplexedIOSocket &&other) = delete;

        MultiplexedIOSocket &operator=(const MultiplexedIOSocket &other) = delete;

        MultiplexedIOSocket &operator=(MultiplexedIOSocket &&other) = delete;

        [[nodiscard]] bool establishConnection();

        [[nodiscard]] bool run();

        [[nodiscard]] bool interrupt();

        // NOTE: User is responsible for destroying *pDoneHandleOut if pDoneHandleOut is populated
        [[nodiscard]] bool sendBytes(uint64_t tag, const std::span<const std::byte> &data, bool clone_memory = true, tpark_handle_t **pDoneHandleOut = nullptr) const;

        [[nodiscard]] std::optional<ssize_t> receiveBytesInplace(uint64_t tag, const std::span<std::byte> &data) const;

        [[nodiscard]] std::optional<std::unique_ptr<std::byte[]>> receiveBytes(
            uint64_t tag, std::span<std::byte> &data, bool no_wait) const;

        /// Sends a packet to the client associated with the given socket address
        /// Returns false if the server is not running or the client connection does not exist
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendPacket(const uint64_t tag,
                                      const T &packet) {
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendLtvPacket<T>(tag, id, buffer);
        }

        /// Receives a packet of the specified type
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receivePacket(const uint64_t tag, const bool no_wait) {
            const ccoip::packetId_t id = T::packet_id;
            return receiveNextPacket<T>(tag, id, no_wait);
        }

        /// Discards all received data that was not yet consumed by @code receiveBytes@endcode for the specified tag
        void discardReceivedData(uint64_t tag) const;

        void join();

        [[nodiscard]] bool isOpen() const;

        [[nodiscard]] const ccoip_socket_address_t &getConnectSockAddr() const;

        ~MultiplexedIOSocket();

    private:
        // Packet decoding / encoding functions
        [[nodiscard]] std::optional<size_t> receivePacketLength() const;

        [[nodiscard]] bool receivePacketData(std::span<std::uint8_t> &dst) const;

        /// Closes the socket.
        /// Note: this method will discard all data still queued for sending
        [[nodiscard]] bool closeConnection();

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveNextPacket(const uint64_t tag, const ccoip::packetId_t packet_id,
                                                         const bool no_wait) {
            std::span<std::byte> data_span{};
            auto data_uptr_opt = receiveBytes(tag, data_span, no_wait);
            if (!data_uptr_opt) {
                return std::nullopt;
            }
            if (data_uptr_opt == nullptr) {
                return std::nullopt;
            }
            const std::unique_ptr<std::byte[]> data_uptr = std::move(*data_uptr_opt);
            PacketReadBuffer buffer{reinterpret_cast<uint8_t *>(data_span.data()), data_span.size_bytes()};
            if (const auto actual_packet_id = buffer.read<ccoip::packetId_t>(); actual_packet_id != packet_id) {
                LOG(ERR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            T packet{};
            if (!packet.deserialize(buffer)) {
                LOG(ERR) << "Failed to deserialize packet with ID " << packet_id;
                return std::nullopt;
            }
            return packet;
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendLtvPacket(const uint64_t tag, const ccoip::packetId_t packet_id,
                                         const PacketWriteBuffer &buffer) const {
            PacketWriteBuffer complete_packet{};
            complete_packet.write(packet_id);
            complete_packet.writeContents(buffer.data(), buffer.size());
            return sendBytes(tag,
                             std::span(reinterpret_cast<const std::byte *>(complete_packet.data()),
                                       complete_packet.size()));
        }
    };

    namespace poll {
        enum PollEvent {
            POLL_INPUT = 1,
            POLL_OUTPUT = 2,
        };

        struct PollDescriptor {
            int socket_fd;
            PollEvent target_event{};
            PollEvent event_out{};

            [[nodiscard]] bool hasEvent(PollEvent event) const;
        };

        int poll(std::vector<PollDescriptor> &descriptors, int timeout);

        std::optional<size_t> send_nonblocking(const std::span<const std::byte> &data,
                                               const PollDescriptor &poll_descriptor);

        std::optional<size_t> recv_nonblocking(const std::span<std::byte> &data, const PollDescriptor &poll_descriptor);
    };
};
