#include <ccoip_inet.h>
#include <ccoip_packet.hpp>
#include <tinysockets.hpp>
#include <gtest/gtest.h>

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


TEST(TestQueuedSocket, test_recv_matching_packet_ignore_others) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        EXPECT_TRUE(packet.has_value());
        EXPECT_EQ(packet->payload.size(), 1);
        EXPECT_EQ(packet->payload[0], 42);

        DummyPacket response{};
        response.payload = {43};
        EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, response)); // send packet with payload 43
        EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, *packet)); // send packet with same payload back
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::QueuedSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    EXPECT_TRUE(client_socket.run());

    DummyPacket packet{};
    packet.payload = {42};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacket>(packet));

    auto received = client_socket.receiveMatchingPacket<DummyPacket>([](const DummyPacket &p) {
        return p.payload[0] == 42;
    });
    EXPECT_TRUE(received.has_value());
    EXPECT_EQ(received->payload.size(), 1);
    EXPECT_EQ(received->payload[0], 42);

    EXPECT_TRUE(client_socket.interrupt());
    client_socket.join();

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

struct DummyPacketUnique final : ccoip::Packet {
    std::unique_ptr<std::uint8_t[]> payload;
    size_t size = 0;
    static constexpr ccoip::packetId_t packet_id = 0x1235;

    void serialize(PacketWriteBuffer &buffer) const override {
        buffer.write<std::uint64_t>(size);
        buffer.writeContents(payload.get(), size);
    }

    bool deserialize(PacketReadBuffer &buffer) override {
        size = buffer.read<std::uint64_t>();
        payload = std::make_unique<std::uint8_t[]>(size);
        buffer.readContents(payload.get(), size);
        return true;
    }

    DummyPacketUnique(
        const std::initializer_list<std::uint8_t> &list) : payload(std::make_unique<std::uint8_t[]>(list.size())),
                                                           size(list.size()) {
        std::copy_n(list.begin(), size, payload.get());
    }

    DummyPacketUnique(DummyPacketUnique &&other) noexcept {
        size = other.size;
        payload = std::move(other.payload);
    }

    DummyPacketUnique(const DummyPacketUnique &other) = delete;
};

TEST(TestQueuedSocket, test_recv_matching_packet_none_lost__inorder) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacketUnique>(buffer);
        EXPECT_TRUE(packet.has_value());
        EXPECT_EQ(packet->size, 1);
        EXPECT_EQ(packet->payload[0], 42);

        const DummyPacketUnique response{std::initializer_list<uint8_t>{43}};
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, *packet));
        // send packet with same payload back
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, response)); // send packet with payload 43
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::QueuedSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    EXPECT_TRUE(client_socket.run());

    DummyPacketUnique packet{std::initializer_list<uint8_t>{42}};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacketUnique>(packet));

    auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
        return p.payload[0] == 42;
    });
    EXPECT_TRUE(received.has_value());
    EXPECT_EQ(received->size, 1);
    EXPECT_EQ(received->payload[0], 42);

    auto received2 = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
        return p.payload[0] == 43;
    });
    EXPECT_TRUE(received2.has_value());
    EXPECT_EQ(received2->size, 1);
    EXPECT_EQ(received2->payload[0], 43);

    EXPECT_TRUE(client_socket.interrupt());
    client_socket.join();

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}


TEST(TestQueuedSocket, test_recv_matching_packet_none_lost__outoforder) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacketUnique>(buffer);
        EXPECT_TRUE(packet.has_value());
        EXPECT_EQ(packet->size, 1);
        EXPECT_EQ(packet->payload[0], 42);

        const DummyPacketUnique response{std::initializer_list<uint8_t>{43}};
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, response)); // send packet with payload 43
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, *packet));
        // send packet with same payload back
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::QueuedSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    EXPECT_TRUE(client_socket.run());

    DummyPacketUnique packet{std::initializer_list<uint8_t>{42}};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacketUnique>(packet));

    auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
        return p.payload[0] == 42;
    });
    EXPECT_TRUE(received.has_value());
    EXPECT_EQ(received->size, 1);
    EXPECT_EQ(received->payload[0], 42);

    auto received2 = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
        return p.payload[0] == 43;
    });
    EXPECT_TRUE(received2.has_value());
    EXPECT_EQ(received2->size, 1);
    EXPECT_EQ(received2->payload[0], 43);

    // cleanup
    EXPECT_TRUE(client_socket.interrupt());
    client_socket.join();
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}


TEST(TestQueuedSocket, test_recv_matching_packet_none_lost__multiple_threads) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacketUnique>(buffer);
        EXPECT_TRUE(packet.has_value());
        EXPECT_EQ(packet->size, 1);
        EXPECT_EQ(packet->payload[0], 42);

        const DummyPacketUnique response{std::initializer_list<uint8_t>{43}};
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, response)); // send packet with payload 43
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, *packet));
        // send packet with same payload back
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::QueuedSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    EXPECT_TRUE(client_socket.run());

    const DummyPacketUnique packet{std::initializer_list<uint8_t>{42}};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacketUnique>(packet));

    // launch two threads to receive packets 42 and 43 respectively
    // synchronize threads precisely such that they will call receiveMatchingPacket roughly at the same time
    std::condition_variable cv{};
    std::mutex m{};
    std::atomic_int num_waiting = 0;
    std::thread t1([&client_socket, &cv, &m, &num_waiting] {
        std::unique_lock lock(m);
        ++num_waiting;
        cv.wait(lock);
        const auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
            return p.payload[0] == 42;
        });
        EXPECT_TRUE(received.has_value());
        EXPECT_EQ(received->size, 1);
        EXPECT_EQ(received->payload[0], 42);
    });
    std::thread t2([&client_socket, &cv, &m, &num_waiting] {
        std::unique_lock lock(m);
        ++num_waiting;
        cv.wait(lock);

        const auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
            return p.payload[0] == 43;
        });
        EXPECT_TRUE(received.has_value());
        EXPECT_EQ(received->size, 1);
        EXPECT_EQ(received->payload[0], 43);
    });
    while (num_waiting.load() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    cv.notify_all();

    t1.join();
    t2.join();

    EXPECT_TRUE(client_socket.interrupt());
    client_socket.join();

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestQueuedSocket, test_recv_matching_packet_none_lost__multiple_threads__many_packets) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    constexpr int num_packets = 2048;

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacketUnique>(buffer);
        EXPECT_TRUE(packet.has_value());
        EXPECT_EQ(packet->size, 1);
        EXPECT_EQ(packet->payload[0], 42);

        for (int i = 0; i < num_packets; ++i) {
            const DummyPacketUnique response{std::initializer_list<uint8_t>{43}};
            EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, response));
            // send packet with payload 43
            EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, *packet));
            // send packet with same payload back
        }
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::QueuedSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    EXPECT_TRUE(client_socket.run());

    const DummyPacketUnique packet{std::initializer_list<uint8_t>{42}};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacketUnique>(packet));

    // launch two threads to receive packets 42 and 43 respectively
    // synchronize threads precisely such that they will call receiveMatchingPacket roughly at the same time
    std::condition_variable cv{};
    std::mutex m{};
    std::atomic_int num_waiting = 0;
    std::thread t1([&client_socket, &cv, &m, &num_waiting] {
        {
            std::unique_lock lock(m);
            ++num_waiting;
            cv.wait(lock);
        }
        for (int i = 0; i < num_packets; ++i) {
            const auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>(
                [](const DummyPacketUnique &p) {
                    return p.payload[0] == 42;
                });
            EXPECT_TRUE(received.has_value());
            EXPECT_EQ(received->size, 1);
            EXPECT_EQ(received->payload[0], 42);
        }
    });
    std::thread t2([&client_socket, &cv, &m, &num_waiting] {
        {
            std::unique_lock lock(m);
            ++num_waiting;
            cv.wait(lock);
        }
        for (int i = 0; i < num_packets; ++i) {
            const auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>(
                [](const DummyPacketUnique &p) {
                    return p.payload[0] == 43;
                });
            EXPECT_TRUE(received.has_value());
            EXPECT_EQ(received->size, 1);
            EXPECT_EQ(received->payload[0], 43);
        }
    });
    while (num_waiting.load() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    cv.notify_all();

    t1.join();
    t2.join();

    EXPECT_TRUE(client_socket.interrupt());
    client_socket.join();

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

struct AnotherDummyPacket final : ccoip::Packet {
    static constexpr ccoip::packetId_t packet_id = 0x5678;
    std::vector<std::uint8_t> data;

    void serialize(PacketWriteBuffer &buffer) const override {
        buffer.writeContents(data.data(), data.size());
    }

    bool deserialize(PacketReadBuffer &buffer) override {
        data.resize(buffer.remaining());
        buffer.readContents(data.data(), data.size());
        return true;
    }
};

TEST(TestQueuedSocket, test_minimal_interleaved_two_packets) {
    // 1) Server setup
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    // We expect the client to send exactly 2 packets total:
    //   1 DummyPacket, 1 AnotherDummyPacket
    // For each inbound packet, we send back 2 responses:
    //   1 DummyPacket, 1 AnotherDummyPacket.

    // Let's track how many inbound packets the server sees, just as a sanity check.
    std::atomic inbound_count(0);

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        ++inbound_count;

        // Attempt first as DummyPacket
        {
            PacketReadBuffer buf = PacketReadBuffer::wrap(data);
            if (auto maybe_dummy = server_socket.receivePacket<DummyPacket>(buf); maybe_dummy.has_value()) {
                // Got a DummyPacket
                // Send back (DummyPacket + AnotherDummyPacket)
                DummyPacket resp1{};
                resp1.payload = {0xA1};
                EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, resp1));

                AnotherDummyPacket resp2{};
                resp2.data = {0xB1};
                EXPECT_TRUE(server_socket.sendPacket<AnotherDummyPacket>(client_addr, resp2));
                return;
            }
        }

        // Attempt second as AnotherDummyPacket
        {
            PacketReadBuffer buf = PacketReadBuffer::wrap(data);
            if (const auto maybe_another = server_socket.receivePacket<AnotherDummyPacket>(buf); maybe_another.has_value()) {
                // Got AnotherDummyPacket
                // Send back (DummyPacket + AnotherDummyPacket)
                DummyPacket resp1{};
                resp1.payload = {0xA2};
                EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, resp1));

                AnotherDummyPacket resp2{};
                resp2.data = {0xB2};
                EXPECT_TRUE(server_socket.sendPacket<AnotherDummyPacket>(client_addr, resp2));
                return;
            }
        }

        FAIL() << "Server: Could not parse inbound data as either DummyPacket or AnotherDummyPacket!";
    });
    EXPECT_TRUE(server_socket.runAsync());

    // 2) Client setup
    tinysockets::QueuedSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    EXPECT_TRUE(client_socket.run());

    // 3) Send 2 packets from client -> server
    {
        DummyPacket dp;
        dp.payload = {0x11};
        EXPECT_TRUE(client_socket.sendPacket(dp));

        AnotherDummyPacket ap;
        ap.data = {0x22};
        EXPECT_TRUE(client_socket.sendPacket(ap));
    }

    // The server should see exactly 2 inbound messages
    // (We can wait briefly to ensure the read callbacks fire)
    for (int i = 0; i < 5 && inbound_count.load() < 2; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    EXPECT_EQ(inbound_count.load(), 2) << "Server didn't receive the 2 packets we sent.";

    // 4) Now we expect 4 total inbound messages on the client side:
    //    Because for each inbound packet, the server sends back 2.
    //    We'll have 2 threads:
    //       Thread A reading DummyPacket (2 total)
    //       Thread B reading AnotherDummyPacket (2 total)

    constexpr int expected_dummy_count  = 2;
    constexpr int expected_another_count = 2;
    std::atomic received_dummy_count(0);
    std::atomic received_another_count(0);

    // Thread A: DummyPacket
    std::thread tA([&] {
        for (int i = 0; i < expected_dummy_count; ++i) {
            auto result = client_socket.receiveMatchingPacket<DummyPacket>(
                [](const DummyPacket &) {
                    // We'll accept all DummyPacket
                    return true;
                }
            );
            EXPECT_TRUE(result.has_value()) << "Failed to receive the next DummyPacket";
            ++received_dummy_count;
        }
    });

    // Thread B: AnotherDummyPacket
    std::thread tB([&] {
        for (int i = 0; i < expected_another_count; ++i) {
            auto result = client_socket.receiveMatchingPacket<AnotherDummyPacket>(
                [](const AnotherDummyPacket &) {
                    // Accept all AnotherDummyPacket
                    return true;
                }
            );
            EXPECT_TRUE(result.has_value()) << "Failed to receive the next AnotherDummyPacket";
            ++received_another_count;
        }
    });

    // Join the threads
    tA.join();
    tB.join();

    // Verify we got the total messages we expect
    EXPECT_EQ(received_dummy_count.load(), expected_dummy_count);
    EXPECT_EQ(received_another_count.load(), expected_another_count);

    // 5) Clean up
    EXPECT_TRUE(client_socket.interrupt());
    client_socket.join();

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestQueuedSocket, test_interleaved_packet_types) {
    // 1) Set up a server socket on localhost:48148
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    // 2) We'll send a mix of DummyPacket and AnotherDummyPacket:
    //    Let's do 5 of each, in random or alternating order.
    constexpr int countA = 5; // number of DummyPacket
    constexpr int countB = 5; // number of AnotherDummyPacket

    // 3) We'll track how many packets the server has received from the client,
    //    just as a sanity check (optional).
    std::atomic server_received_A(0);
    std::atomic server_received_B(0);

    // 4) Set up a read callback that attempts to parse either DummyPacket or AnotherDummyPacket
    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);

        // a) First, try to parse as DummyPacket
        if (auto packetA_opt = server_socket.receivePacket<DummyPacket>(buffer)) {
            // It's a DummyPacket
            ++server_received_A;

            // Let's send back 1 DummyPacket and 1 AnotherDummyPacket in *random* or alternating order.
            DummyPacket responseA{};
            responseA.payload = {0xAA}; // some small data
            EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, responseA));

            AnotherDummyPacket responseB{};
            responseB.data = {0xBB};
            EXPECT_TRUE(server_socket.sendPacket<AnotherDummyPacket>(client_addr, responseB));

            return;
        }

        // b) If not, try AnotherDummyPacket
        //    (We re-wrap the buffer, because the previous read might have advanced it or we can read from scratch.)
        PacketReadBuffer buffer2 = PacketReadBuffer::wrap(data);
        if (auto packetB_opt = server_socket.receivePacket<AnotherDummyPacket>(buffer2)) {
            ++server_received_B;
            // Send one AnotherDummyPacket followed by one DummyPacket
            AnotherDummyPacket responseB{};
            responseB.data = {0xCC};
            EXPECT_TRUE(server_socket.sendPacket<AnotherDummyPacket>(client_addr, responseB));

            DummyPacket responseA{};
            responseA.payload = {0xDD};
            EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, responseA));

            return;
        }

        // If neither parse succeeds, something’s wrong
        FAIL() << "Server could not parse incoming data as either DummyPacket or AnotherDummyPacket!";
    });

    // 5) Start the server
    EXPECT_TRUE(server_socket.runAsync());

    // 6) Create a client socket
    tinysockets::QueuedSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    EXPECT_TRUE(client_socket.run());

    // 7) Send 5 DummyPackets + 5 AnotherDummyPackets from client -> server in random order
    //    or just do an alternating pattern for simplicity.
    for (int i = 0; i < countA + countB; ++i) {
        if (i % 2 == 0) {
            // Send DummyPacket
            DummyPacket pA{};
            pA.payload = {static_cast<std::uint8_t>(0x10 + i)};
            EXPECT_TRUE(client_socket.sendPacket<DummyPacket>(pA));
        } else {
            // Send AnotherDummyPacket
            AnotherDummyPacket pB{};
            pB.data = {static_cast<std::uint8_t>(0x20 + i)};
            EXPECT_TRUE(client_socket.sendPacket<AnotherDummyPacket>(pB));
        }
    }

    // 8) Now we expect the server to send back a mix of DummyPacket and AnotherDummyPacket
    //    for each of the 10 messages we sent. So total 10 * 2 = 20 inbound messages, half each type.
    //    We'll split up the reading among 2 threads: one handles DummyPacket, the other AnotherDummyPacket.

    constexpr int expected_from_server_A = 10; // total DummyPackets from server
    constexpr int expected_from_server_B = 10; // total AnotherDummyPackets from server

    // We'll store how many we got in atomic counters
    std::atomic client_received_A(0);
    std::atomic client_received_B(0);

    // 9) Thread A: receives DummyPacket
    std::thread tA([&] {
        for (int i = 0; i < expected_from_server_A; ++i) {
            auto result = client_socket.receiveMatchingPacket<DummyPacket>([](const DummyPacket &p) {
                // We'll just accept all DummyPackets we see
                return true;
            });
            EXPECT_TRUE(result.has_value()); // We expect them all eventually
            ++client_received_A;
        }
    });

    // 10) Thread B: receives AnotherDummyPacket
    std::thread tB([&] {
        for (int i = 0; i < expected_from_server_B; ++i) {
            auto result = client_socket.receiveMatchingPacket<AnotherDummyPacket>([](const AnotherDummyPacket &p) {
                // We'll accept all AnotherDummyPackets
                return true;
            });
            EXPECT_TRUE(result.has_value());
            ++client_received_B;
        }
    });

    // 11) Wait for threads to finish
    tA.join();
    tB.join();

    // 12) Verify we received the counts we expected
    EXPECT_EQ(client_received_A.load(), expected_from_server_A);
    EXPECT_EQ(client_received_B.load(), expected_from_server_B);

    // 13) Optionally check how many the server received
    //     (We expect 5 of each type from the client)
    EXPECT_EQ(server_received_A.load(), countA);
    EXPECT_EQ(server_received_B.load(), countB);

    // 14) Clean up
    EXPECT_TRUE(client_socket.interrupt());
    client_socket.join();

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
