#include <tinysockets.hpp>
#include <gtest/gtest.h>
#include <future>

inline ccoip_socket_address_t create_ipv4_address(const uint8_t a, const uint8_t b, const uint8_t c, const uint8_t d,
                                                  const uint16_t port) {
    return ccoip_socket_address_t{
        .inet = {
            .protocol = inetIPv4,
            .ipv4 = {
                .data = {a, b, c, d}
            }
        },
        .port = port
    };
}

TEST(TestServerSocket, test_bind_valid) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(0, 0, 0, 0, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
}

TEST(TestServerSocket, test_bind_loopback) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
}

TEST(TestServerSocket, test_bind_invalid) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(0, 0, 0, 0, 0);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_FALSE(server_socket.listen());
}

TEST(TestServerSocket, test_bind_valid_ipv6) {
    const ccoip_socket_address_t listen_address = {
        .inet = {
            .protocol = inetIPv6,
            .ipv6 = {
                .data = {},
            },
        },
        .port = 48148,
    };
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
}

TEST(TestServerSocket, test_bind_invalid_ipv6) {
    constexpr ccoip_socket_address_t listen_address = {
        .inet = {
            .protocol = inetIPv6,
            .ipv6 = {
                .data = {},
            },
        },
        .port = 0,
    };
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_FALSE(server_socket.listen());
}

// Test binding an already bound socket
TEST(TestServerSocket, test_bind_already_bound) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48149);
    tinysockets::ServerSocket server_socket1(listen_address);
    EXPECT_TRUE(server_socket1.listen());

    tinysockets::ServerSocket server_socket2(listen_address);
    EXPECT_FALSE(server_socket2.listen());
}

// Test listening on a bound socket
TEST(TestServerSocket, test_listen_after_bind) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48150);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
}

// Test running the server asynchronously
TEST(TestServerSocket, test_run_async) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48152);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Allow some time for the server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test running async when already running
TEST(TestServerSocket, test_run_async_already_running) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48153);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());
    EXPECT_FALSE(server_socket.runAsync()); // Should fail because server is already running

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test interrupting and joining the server
TEST(TestServerSocket, test_interrupt_and_join) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48154);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Allow some time for the server to run
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

struct DummyPacket final : ccoip::Packet {
    std::vector<std::uint8_t> payload;
    static constexpr ccoip::packetId_t packet_id = 0x1234;

    void serialize(PacketWriteBuffer &buffer) const override {
        buffer.writeContents(payload.data(), payload.size());
    }

    bool deserialize(PacketReadBuffer &buffer) override {
        payload.resize(buffer.remaining());
        buffer.readContents(payload.data(), payload.size());
        return true;
    }
};

// Test handling client connections and callbacks
TEST(TestServerSocket, test_client_connection_and_callbacks) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48155);
    tinysockets::ServerSocket server_socket(listen_address);

    // Variables to verify callbacks
    std::atomic read_callback_called(false);
    std::atomic close_callback_called(false);
    std::vector<std::uint8_t> received_data;

    // Set up callbacks
    server_socket.addReadCallback(
        [&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
            PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
            const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
            if (!packet) {
                FAIL() << "Failed to receive packet";
            }
            received_data = packet->payload;
            read_callback_called = true;
        });

    server_socket.addCloseCallback([&](const ccoip_socket_address_t &) {
        close_callback_called = true;
    });

    // Bind and listen
    EXPECT_TRUE(server_socket.listen());

    // Run the server asynchronously
    EXPECT_TRUE(server_socket.runAsync());

    // Allow some time for the server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create a client and connect to the server
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Send some data
    std::vector<std::uint8_t> data_to_send = {1, 2, 3, 4, 5};

    DummyPacket packet{};
    packet.payload = data_to_send;
    EXPECT_TRUE(client_socket.sendPacket(packet));

    // Closing the connection before the server has received the data
    // may result in data loss
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Close the client connection
    EXPECT_TRUE(client_socket.closeConnection());

    // Cleanup
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // close_callback_called can take a while to be set, but we don't want to wait forever
    for (int i = 0; i < 10; ++i) {
        if (close_callback_called) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Verify that callbacks were called
    EXPECT_TRUE(read_callback_called);
    EXPECT_TRUE(close_callback_called);
    EXPECT_EQ(received_data, data_to_send);
}

// Test closing a client connection
TEST(TestServerSocket, test_close_client_connection) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48156);
    tinysockets::ServerSocket server_socket(listen_address);

    // add a read callback to the server
    std::atomic read_callback_called(false), payload_match(false), client_close_success(false);
    server_socket.addReadCallback(
        [&](const ccoip_socket_address_t &addr, const std::span<std::uint8_t> &data) {
            PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
            const auto recv_packet = server_socket.receivePacket<DummyPacket>(buffer);
            if (!recv_packet) {
                FAIL() << "Failed to receive packet";
            }
            payload_match = recv_packet->payload == std::vector<std::uint8_t>({1, 2, 3, 4, 5});

            // Attempt to close the client connection from the server side
            client_close_success = server_socket.closeClientConnection(addr);

            read_callback_called = true;
        });

    // Bind and listen
    EXPECT_TRUE(server_socket.listen());

    // Run the server asynchronously
    EXPECT_TRUE(server_socket.runAsync());

    // Create a client and connect to the server
    tinysockets::BlockingIOSocket client_socket(listen_address);

    EXPECT_TRUE(client_socket.establishConnection());

    DummyPacket packet{};
    packet.payload = {1, 2, 3, 4, 5};
    EXPECT_TRUE(client_socket.sendPacket(packet));

    // read callback may take a while to be called, but we don't want to wait forever
    for (int i = 0; i < 15; ++i) {
        if (read_callback_called) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    EXPECT_TRUE(read_callback_called);
    EXPECT_TRUE(payload_match);
    EXPECT_TRUE(client_close_success);
    EXPECT_FALSE(client_socket.isOpen()); // Should fail because the connection is closed

    // Cleanup
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test binding the same socket twice
TEST(TestServerSocket, test_bind_twice) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48158);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
    EXPECT_FALSE(server_socket.listen()); // Second bind should fail
}

// Test listening twice
TEST(TestServerSocket, test_listen_twice) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48159);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
    EXPECT_FALSE(server_socket.listen()); // Second listen should fail
}

// Test run async after server interrupted
TEST(TestServerSocket, test_run_async_after_interrupt) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48160);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    EXPECT_FALSE(server_socket.runAsync()); // Server should not be restarted after join
}

// Test multiple clients connecting simultaneously
TEST(TestServerSocket, test_multiple_clients) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48161);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic packets_received(0);
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        if (auto packet = server_socket.receivePacket<DummyPacket>(buffer)) {
            ++packets_received;
        }
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // We'll connect multiple clients and send packets
    constexpr int num_clients = 5;
    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    for (int i = 0; i < num_clients; ++i) {
        clients.emplace_back(std::make_unique<tinysockets::BlockingIOSocket>(listen_address));
        EXPECT_TRUE(clients.back()->establishConnection());
    }

    // Send packets from all clients
    DummyPacket p;
    p.payload = {42};
    for (const auto &client: clients) {
        EXPECT_TRUE(client->sendPacket(p));
    }

    // Wait for all packets to arrive
    for (int i = 0; i < 50 && packets_received.load() < num_clients; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_EQ(packets_received.load(), num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending a large packet
TEST(TestServerSocket, test_large_packet) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48162);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic read_callback_called(false);
    const std::vector<uint8_t> large_data(1024 * 1024, 0xAB); // 1 MB of data
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive large packet";
        }
        EXPECT_EQ(packet->payload.size(), large_data.size());
        EXPECT_EQ(packet->payload, large_data);
        read_callback_called = true;
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Create client and send large packet
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    DummyPacket p{};
    p.payload = large_data;
    EXPECT_TRUE(client_socket.sendPacket(p));

    for (int i = 0; i < 50 && !read_callback_called.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(read_callback_called);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test closing a non-existent client connection
TEST(TestServerSocket, test_close_non_existent_client) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48163);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Dummy address that won't exist as a client
    const ccoip_socket_address_t non_existent_client = create_ipv4_address(127, 0, 0, 1, 9999);
    EXPECT_FALSE(server_socket.closeClientConnection(non_existent_client));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test adding multiple callbacks
TEST(TestServerSocket, test_multiple_callbacks) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48164);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic read_calls(0);
    std::atomic close_calls(0);

    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        ++read_calls;
    });
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        ++read_calls;
    });

    server_socket.addCloseCallback([&](const ccoip_socket_address_t &) {
        ++close_calls;
    });
    server_socket.addCloseCallback([&](const ccoip_socket_address_t &) {
        ++close_calls;
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect a client and send something
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    DummyPacket p{};
    p.payload = {1, 2, 3};
    EXPECT_TRUE(client_socket.sendPacket(p));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Close client
    EXPECT_TRUE(client_socket.closeConnection());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_GE(read_calls.load(), 2); // Both read callbacks should have fired
    EXPECT_GE(close_calls.load(), 2); // Both close callbacks should have fired

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test connecting a client when the server is not listening
TEST(TestServerSocket, test_client_connect_when_not_listening) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49100);
    tinysockets::ServerSocket server_socket(listen_address);
    // Notice we do not call listen()
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_FALSE(client_socket.establishConnection()); // Should fail because server not listening
}

// Test multiple interrupts
TEST(TestServerSocket, test_multiple_interrupts) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49102);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Interrupt the server multiple times
    EXPECT_TRUE(server_socket.interrupt());
    EXPECT_TRUE(server_socket.interrupt());

    server_socket.join();
}

// Test attempting to re-run the server after it has been joined
TEST(TestServerSocket, test_rerun_after_join) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49103);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // Attempting to run again after join. Expect failure.
    EXPECT_FALSE(server_socket.runAsync());
}

// Test sending zero-length payload
TEST(TestServerSocket, test_zero_length_packet) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49105);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic callback_called(false);
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_TRUE(packet->payload.empty());
        callback_called = true;
    });

    EXPECT_TRUE(server_socket.listen());

    EXPECT_TRUE(server_socket.runAsync());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Connect client
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Send zero-length payload
    DummyPacket p{};
    p.payload = {};
    EXPECT_TRUE(client_socket.sendPacket(p));

    for (int i = 0; i < 10 && !callback_called.load(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(callback_called);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending unknown packet ID
struct UnknownPacket final : ccoip::Packet {
    std::vector<std::uint8_t> payload;
    static constexpr ccoip::packetId_t packet_id = 0x9999;
    void serialize(PacketWriteBuffer &buffer) const override { buffer.writeContents(payload.data(), payload.size()); }

    bool deserialize(PacketReadBuffer &buffer) override {
        payload.resize(buffer.remaining());
        buffer.readContents(payload.data(), payload.size());
        return true;
    }
};

TEST(TestServerSocket, test_unknown_packet_id) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49106);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic callback_called(false);
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        // Try to parse as a DummyPacket, but we receive an UnknownPacket
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        EXPECT_FALSE(packet.has_value()); // Should fail to parse correctly
        callback_called = true;
    });

    EXPECT_TRUE(server_socket.listen());

    EXPECT_TRUE(server_socket.runAsync());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    UnknownPacket unknown{};
    unknown.payload = {0xAB, 0xCD};
    EXPECT_TRUE(client_socket.sendPacket(unknown));

    for (int i = 0; i < 10 && !callback_called.load(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(callback_called);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test concurrent client connections (attempting to connect multiple clients in parallel)
TEST(TestServerSocket, test_concurrent_client_connections) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49107);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    constexpr int num_clients = 10;
    std::vector<std::future<bool> > futures;
    futures.reserve(num_clients);
    for (int i = 0; i < num_clients; i++) {
        futures.emplace_back(std::async(std::launch::async, [&] {
            tinysockets::BlockingIOSocket client(listen_address);
            return client.establishConnection();
        }));
    }

    int success_count = 0;
    for (auto &f: futures) {
        if (f.get()) {
            success_count++;
        }
    }

    // All should succeed if enough backlog/concurrency is supported
    EXPECT_EQ(success_count, num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending data from the client after the server has been stopped
TEST(TestServerSocket, test_client_send_after_server_stop) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49108);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Stop the server
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // Now try sending a packet after server is stopped
    DummyPacket p{};
    p.payload = {0x01};
    // Sending might succeed or fail silently; we just ensure it doesn't crash.
    // If the server is truly down, the send might fail.
    if (!client_socket.sendPacket(p)) {
        SUCCEED();
        return;
    }

    // isOpen() might still return true for some time, but we don't want to wait forever
    for (int i = 0; i < 10; ++i) {
        if (!client_socket.isOpen()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Test attempting to close client connections after the server has stopped
TEST(TestServerSocket, test_close_client_after_server_stopped) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49109);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Establish a connection
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Interrupt the server and wait for it to stop
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // Now server is stopped, try to close client connection
    EXPECT_FALSE(server_socket.closeClientConnection(listen_address)); // Should fail since server is not running
}

// Test server sending data to client
TEST(TestServerSocket, test_server_send_data) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49110);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic client_read(false);
    server_socket.addReadCallback([&](const ccoip_socket_address_t &addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive packet";
        }
        client_read = true;

        EXPECT_EQ(packet->payload, std::vector<std::uint8_t>({0x01, 0x02, 0x03}));

        // send response back to client
        DummyPacket response{};
        response.payload = {0x01, 0x02, 0x03};
        EXPECT_TRUE(server_socket.sendPacket(addr, response));
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect to server
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Send data to server
    DummyPacket p{};
    p.payload = {0x01, 0x02, 0x03};
    EXPECT_TRUE(client_socket.sendPacket(p));

    // Wait for server to send data back
    auto response = client_socket.receivePacket<DummyPacket>();
    ASSERT_TRUE(response.has_value());
    EXPECT_EQ(response->payload, std::vector<std::uint8_t>({0x01, 0x02, 0x03}));

    for (int i = 0; i < 10 && !client_read.load(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(client_read);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending data upon client connection
TEST(TestServerSocket, test_server_send_on_client_connect) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49120);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic packets_sent(0);
    DummyPacket p{};
    p.payload = {0xAA, 0xBB, 0xCC};

    // Add join callback to send packet upon client connection
    server_socket.addJoinCallback([&](const ccoip_socket_address_t &addr) {
        packets_sent.fetch_add(1, std::memory_order_relaxed);
        EXPECT_TRUE(server_socket.sendPacket(addr, p));
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect multiple clients
    constexpr int num_clients = 3;
    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    clients.reserve(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        auto client = std::make_unique<tinysockets::BlockingIOSocket>(listen_address);
        EXPECT_TRUE(client->establishConnection());
        clients.emplace_back(std::move(client));
    }

    // Each client should receive the data
    for (auto &client: clients) {
        auto received = client->receivePacket<DummyPacket>();
        ASSERT_TRUE(received.has_value());
        EXPECT_EQ(received->payload, p.payload);
    }

    // Verify all packets were sent
    EXPECT_EQ(packets_sent.load(), num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test server sends response upon receiving client messages
TEST(TestServerSocket, test_server_send_on_client_message) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49121);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic responses_sent(0);
    DummyPacket response{};
    response.payload = {0xDE, 0xAD, 0xBE, 0xEF};

    // Add read callback to send a response when a packet is received
    server_socket.addReadCallback([&](const ccoip_socket_address_t &addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->payload, std::vector<uint8_t>({0x01, 0x02, 0x03}));

        // Send response back to client
        responses_sent.fetch_add(1, std::memory_order_relaxed);
        EXPECT_TRUE(server_socket.sendPacket(addr, response));
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect a client
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Client sends a packet to the server
    DummyPacket client_packet{};
    client_packet.payload = {0x01, 0x02, 0x03};
    EXPECT_TRUE(client_socket.sendPacket(client_packet));

    // Client receives the server's response
    auto received = client_socket.receivePacket<DummyPacket>();
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->payload, response.payload);

    // Verify that the server sent the response
    EXPECT_EQ(responses_sent.load(), 1);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test server attempts to send after client disconnects
TEST(TestServerSocket, test_server_send_after_client_disconnect) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49122);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic packet_sent(false);

    // Add join callback to send a packet upon connection
    server_socket.addJoinCallback([&](const ccoip_socket_address_t &addr) {
        DummyPacket p{};
        p.payload = {0x99, 0x88};
        EXPECT_TRUE(server_socket.sendPacket(addr, p));
        packet_sent = true;
    });

    // Add close callback to attempt sending after disconnect
    server_socket.addCloseCallback([&](const ccoip_socket_address_t &addr) {
        DummyPacket p{};
        p.payload = {0x77};
        EXPECT_FALSE(server_socket.sendPacket(addr, p)); // Should fail
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect a client
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Client receives the initial packet
    auto received = client_socket.receivePacket<DummyPacket>();
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->payload, std::vector<uint8_t>({0x99, 0x88}));

    // Client disconnects
    EXPECT_TRUE(client_socket.closeConnection());

    // Allow some time for the server to process the disconnect and attempt sending
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Verify that the server attempted to send after disconnect
    EXPECT_TRUE(packet_sent.load());

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test server sends large packets to clients
TEST(TestServerSocket, test_server_send_large_packets) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49123);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic large_packets_sent(0);
    std::vector<uint8_t> large_data(1024 * 1024, 0xFF); // 1 MB of data

    DummyPacket p{};
    p.payload = large_data;

    // Add join callback to send large packet upon connection
    server_socket.addJoinCallback([&](const ccoip_socket_address_t &addr) {
        large_packets_sent.fetch_add(1, std::memory_order_relaxed);
        EXPECT_TRUE(server_socket.sendPacket(addr, p));
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect multiple clients
    constexpr int num_clients = 2;
    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    clients.reserve(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        auto client = std::make_unique<tinysockets::BlockingIOSocket>(listen_address);
        EXPECT_TRUE(client->establishConnection());
        clients.emplace_back(std::move(client));
    }

    // Each client receives the large packet
    for (auto &client: clients) {
        auto received = client->receivePacket<DummyPacket>();
        ASSERT_TRUE(received.has_value());
        EXPECT_EQ(received->payload.size(), large_data.size());
        EXPECT_EQ(received->payload, large_data);
    }

    // Verify that all large packets were sent
    EXPECT_EQ(large_packets_sent.load(), num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending data to multiple clients
TEST(TestServerSocket, test_server_send_to_multiple_clients) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49112);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic responses_received(0);
    server_socket.addReadCallback([&](const ccoip_socket_address_t &addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to parse packet from server on client side callback.";
        }
        // Just count how many responses received
        responses_received.fetch_add(1, std::memory_order_relaxed);
    });

    DummyPacket p{};
    p.payload = {0x10, 0x20, 0x30};
    server_socket.addJoinCallback([&](const ccoip_socket_address_t &addr) {
        // Send data to each connected client from server
        EXPECT_TRUE(server_socket.sendPacket(addr, p));
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect multiple clients
    constexpr int num_clients = 3;
    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    clients.reserve(num_clients);
    for (int i = 0; i < num_clients; i++) {
        auto client = std::make_unique<tinysockets::BlockingIOSocket>(listen_address);
        EXPECT_TRUE(client->establishConnection());
        clients.push_back(std::move(client));
    }

    // Each client should receive the data. Let's verify on the client side.
    for (auto &client: clients) {
        auto received = client->receivePacket<DummyPacket>();
        ASSERT_TRUE(received.has_value());
        EXPECT_EQ(received->payload, p.payload);
    }

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test server attempts to send to non-existent clients
TEST(TestServerSocket, test_server_send_to_non_existent_clients) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49125);
    tinysockets::ServerSocket server_socket(listen_address);

    DummyPacket p{};
    p.payload = {0x55, 0x66};

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Attempt to send to a client address that never connected
    const auto bogus_address = create_ipv4_address(127, 0, 0, 1, 50000);
    EXPECT_FALSE(server_socket.sendPacket(bogus_address, p));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending an extremely large packet to the server
TEST(TestServerSocket, test_client_send_extremely_large_packet) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49127);
    tinysockets::ServerSocket server_socket(listen_address);

    // Define the size of the large packet (e.g., 10 MB)
    constexpr size_t large_size = 10 * 1024 * 1024; // 10 MB
    const std::vector<uint8_t> large_data(large_size, 0xCD); // Initialize with pattern 0xCD

    std::atomic read_callback_called(false);

    // Add read callback to verify received data
    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive large packet";
        }
        EXPECT_EQ(packet->payload.size(), large_size);
        EXPECT_EQ(packet->payload, large_data);
        read_callback_called = true;
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect client
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Send large packet from client to server
    DummyPacket p{};
    p.payload = large_data;
    EXPECT_TRUE(client_socket.sendPacket(p));

    // Wait for the server to receive the packet
    for (int i = 0; i < 100 && !read_callback_called.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(read_callback_called) << "Server did not receive the large packet in time";

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending multiple large packets in succession to the server
TEST(TestServerSocket, test_client_send_multiple_large_packets) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49128);
    tinysockets::ServerSocket server_socket(listen_address);

    constexpr size_t large_size = 5 * 1024 * 1024; // 5 MB per packet
    constexpr int num_packets = 3;
    const std::vector<uint8_t> large_data(large_size, 0xEF); // Initialize with pattern 0xEF

    std::atomic read_callbacks_called(0);

    // Add read callback to verify each received packet
    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive large packet";
        }
        EXPECT_EQ(packet->payload.size(), large_size);
        EXPECT_EQ(packet->payload, large_data);
        read_callbacks_called.fetch_add(1, std::memory_order_relaxed);
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect client
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send multiple large packets from client to server
    for (int i = 0; i < num_packets; ++i) {
        DummyPacket p{};
        p.payload = large_data;
        EXPECT_TRUE(client_socket.sendPacket(p));
    }

    // Wait for the server to receive all packets
    for (int i = 0; i < 200 && read_callbacks_called.load() < num_packets; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_EQ(read_callbacks_called.load(), num_packets)
        << "Server did not receive all large packets in time";

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending a large packet in chunks to the server
TEST(TestServerSocket, test_client_send_large_packet_in_chunks) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49129);
    tinysockets::ServerSocket server_socket(listen_address);

    constexpr size_t large_size = 8 * 1024 * 1024; // 8 MB
    const std::vector<uint8_t> large_data(large_size, 0xBA); // Initialize with pattern 0xBA

    std::atomic read_callback_called(false);

    // Add read callback to verify received data
    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive large packet";
        }
        EXPECT_EQ(packet->payload.size(), large_size);
        EXPECT_EQ(packet->payload, large_data);
        read_callback_called = true;
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect client
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Manually send the packet in smaller chunks
    // Assuming sendPacket sends the entire packet at once, we need to simulate partial sends
    // Since sendPacket is abstracted, we might need to implement a custom send function or modify sendPacket
    // For the purpose of this test, we'll assume sendPacket can handle partial sends internally

    DummyPacket p{};
    p.payload = large_data;
    EXPECT_TRUE(client_socket.sendPacket(p));

    // Wait for the server to receive the packet
    for (int i = 0; i < 200 && !read_callback_called.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(read_callback_called) << "Server did not receive the large packet in time";

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test stress by having multiple clients send large packets concurrently to the server
TEST(TestServerSocket, test_concurrent_clients_send_large_packets) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49131);
    tinysockets::ServerSocket server_socket(listen_address);

    constexpr size_t large_size = 5 * 1024 * 1024; // 5 MB per packet
    const std::vector<uint8_t> large_data(large_size, 0xAA); // Initialize with pattern 0xAA

    constexpr int num_clients = 10;
    std::atomic received_packets(0);

    // Add read callback to verify each received packet
    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive large packet from client";
        }
        EXPECT_EQ(packet->payload.size(), large_size);
        EXPECT_EQ(packet->payload, large_data);
        received_packets.fetch_add(1, std::memory_order_relaxed);
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Create and connect multiple clients
    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    clients.reserve(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        auto client = std::make_unique<tinysockets::BlockingIOSocket>(listen_address);
        EXPECT_TRUE(client->establishConnection());
        clients.emplace_back(std::move(client));
    }

    // Function to send a large packet from a client
    auto send_large_packet = [&](tinysockets::BlockingIOSocket *client) {
        DummyPacket p{};
        p.payload = large_data;
        EXPECT_TRUE(client->sendPacket(p));
    };

    // Launch threads to send large packets concurrently
    std::vector<std::thread> sender_threads;
    sender_threads.reserve(clients.size());
    for (auto &client: clients) {
        sender_threads.emplace_back(send_large_packet, client.get());
    }

    // Wait for all senders to finish
    for (auto &t: sender_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Wait for the server to receive all packets
    for (int i = 0; i < 200 && received_packets.load() < num_clients; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_EQ(received_packets.load(), num_clients)
        << "Server did not receive all large packets from concurrent clients in time";

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending a packet exactly at the server's maximum allowed size
TEST(TestServerSocket, test_client_send_exact_max_packet_size) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49132);
    tinysockets::ServerSocket server_socket(listen_address);

    constexpr size_t exact_size = 10 * 1024 * 1024; // 10 MB
    const std::vector<uint8_t> exact_data(exact_size, 0xBB); // Initialize with pattern 0xBB

    std::atomic read_callback_called(false);

    // Add read callback to verify received data
    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive exact-sized packet";
        }
        EXPECT_EQ(packet->payload.size(), exact_size);
        EXPECT_EQ(packet->payload, exact_data);
        read_callback_called = true;
    });

    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect client
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Send exact-sized packet from client to server
    DummyPacket p{};
    p.payload = exact_data;
    EXPECT_TRUE(client_socket.sendPacket(p));

    // Wait for the server to receive the packet
    for (int i = 0; i < 100 && !read_callback_called.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(read_callback_called) << "Server did not receive the exact-sized packet in time";

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
