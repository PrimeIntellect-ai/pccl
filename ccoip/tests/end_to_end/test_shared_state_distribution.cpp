#include <ccoip.h>
#include <gtest/gtest.h>

#include <ccoip_client.hpp>
#include <ccoip_master.hpp>

#include <thread>
#include <cstring>
#include <port_guard.h>
#include <ranges>

#include "../../../include/pccl.h"

// Helper function to establish p2p connection between two clients
static void establishConnections(const std::vector<ccoip::CCoIPClient *> &clients) {
    size_t n_clients = clients.size();

    std::atomic_int clients_connected = 0;
    uint32_t target_n_clients = 0;
    std::vector<std::thread> client_threads{};
    for (const auto &client: clients) {
        std::thread client_thread([n_clients, &clients_connected, &client] {
            client->setMainThread(std::this_thread::get_id());
            EXPECT_TRUE(client->connect());
            ++clients_connected;
            while (clients_connected < n_clients) {
                EXPECT_TRUE(client->acceptNewPeers());
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }
        });
        client_threads.push_back(std::move(client_thread));
        target_n_clients++;
        while (clients_connected < target_n_clients) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    for (auto &client_thread: client_threads) {
        client_thread.join();
    }
}

// Basic shared state distribution test
// This is a two peer setup, where the two shared states differ.
// This is a tie in terms of content hash prevalence, so we simply assert that one transfer from any client to the other
// has occurred.
TEST(SharedStateDistribution, TestBasic) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);
    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));

        // expect either transmission from client1 to client2 or vice versa
        if (info.tx_bytes == value_size) {
            EXPECT_EQ(info.tx_bytes, value_size);
            EXPECT_EQ(info.rx_bytes, 0);
        } else {
            EXPECT_EQ(info.tx_bytes, 0);
            EXPECT_EQ(info.rx_bytes, value_size);
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 0x0);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client2.syncSharedState(shared_state, info));

        // expect either transmission from client1 to client2 or vice versa
        if (info.tx_bytes == value_size) {
            EXPECT_EQ(info.tx_bytes, value_size);
            EXPECT_EQ(info.rx_bytes, 0);
        } else {
            EXPECT_EQ(info.tx_bytes, 0);
            EXPECT_EQ(info.rx_bytes, value_size);
        }
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // assert the shared state of client 2 to be equal to that of client 1
    EXPECT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);

    // clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Identical shared state should not trigger data transfer
TEST(SharedStateDistribution, TestNoSyncIdenticalSharedState) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);
    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        // client 1 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));

        // for identical shared state, no data should be transferred
        EXPECT_EQ(info.rx_bytes, 0);
        EXPECT_EQ(info.tx_bytes, 0);
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 42);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client2.syncSharedState(shared_state, info));

        // for identical shared state, no data should be transferred
        EXPECT_EQ(info.rx_bytes, 0);
        EXPECT_EQ(info.tx_bytes, 0);
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // assert the shared state of client 2 to be equal to that of client 1
    EXPECT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);

    // clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Partially dirty shared state should only trigger retransmission of the keys that are dirty
TEST(SharedStateDistribution, TestPartialSyncPartiallyDirtyState) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;

    const std::unique_ptr<uint8_t[]> value1_p1(new uint8_t[value_size]);
    std::fill_n(value1_p1.get(), value_size, 42);

    const std::unique_ptr<uint8_t[]> value1_p2(new uint8_t[value_size]);
    std::fill_n(value1_p2.get(), value_size, 42);

    const std::unique_ptr<uint8_t[]> value2_p1(new uint8_t[value_size]);
    std::fill_n(value2_p1.get(), value_size, 43);

    const std::unique_ptr<uint8_t[]> value2_p2(new uint8_t[value_size]);
    std::fill_n(value2_p2.get(), value_size, 44);

    std::thread client1_sync_thread([&client1, &value1_p1, &value2_p1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        // client 1 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1_p1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key2",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2_p1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));

        // only one key is dirty, expect only one size of that key to be transmitted
        // Expect either transmission from client1 to client2 or vice versa
        if (value_size == info.tx_bytes) {
            EXPECT_EQ(info.tx_bytes, value_size);
            EXPECT_EQ(info.rx_bytes, 0);
        } else {
            EXPECT_EQ(info.tx_bytes, 0);
            EXPECT_EQ(info.rx_bytes, value_size);
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::thread client2_sync_thread([&client2, &value1_p2, &value2_p2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1_p2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key2",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2_p2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client2.syncSharedState(shared_state, info));

        // only one key is dirty, expect only one size of that key to be transmitted
        // Expect either transmission from client1 to client2 or vice versa
        if (value_size == info.tx_bytes) {
            EXPECT_EQ(info.tx_bytes, value_size);
            EXPECT_EQ(info.rx_bytes, 0);
        } else {
            EXPECT_EQ(info.tx_bytes, 0);
            EXPECT_EQ(info.rx_bytes, value_size);
        }
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // assert the shared state of client 2 to be equal to that of client 1
    EXPECT_EQ(std::memcmp(value1_p1.get(), value1_p2.get(), value_size), 0);
    EXPECT_EQ(std::memcmp(value2_p1.get(), value2_p2.get(), value_size), 0);

    // clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}

// Popular hash prevalence test. Three peers synchronize shared state, two of the three have the same shared state.
// This tests specifically tests whether client1 - which has the unpopular state, which hits the master >first< will
// NOT be determining the accepted shared state hash. Client1 MUST receive the more popular shared state from client2 or client3.
TEST(SharedStateDistribution, TestPopularHashPrevelance) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // client 3
    ccoip::CCoIPClient client3({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2, &client3});

    constexpr size_t value_size = 1024;


    const std::unique_ptr<uint8_t[]> value1_bak(new uint8_t[value_size]);
    std::fill_n(value1_bak.get(), value_size, 42);

    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 43);

    const std::unique_ptr<uint8_t[]> value3(new uint8_t[value_size]);
    std::fill_n(value3.get(), value_size, 43);

    // Client one has the incorrect shared state and will hit the master FIRST;
    // This test crucially tests whether despite the fact that client 1 is first to hit the master, the master will
    // chose later clients to be the determining peer for the accepted hashes.
    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        // client 1 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));
    });
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client2.syncSharedState(shared_state, info));
    });
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::thread client3_sync_thread([&client3, &value3, value_size] {
        client3.setMainThread(std::this_thread::get_id());
        // client 3 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value3.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client3.syncSharedState(shared_state, info));
    });

    client1_sync_thread.join();
    client2_sync_thread.join();
    client3_sync_thread.join();

    // assert that client1 no longer equals the bak and has in fact been modified
    EXPECT_NE(std::memcmp(value1_bak.get(), value1.get(), value_size), 0);

    // assert that client2 and client3 are equal
    EXPECT_EQ(std::memcmp(value2.get(), value3.get(), value_size), 0);

    // assert that client1 and client2 are equal
    EXPECT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);

    // clean shutdown
    EXPECT_TRUE(client3.interrupt());
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());
    EXPECT_TRUE(client3.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}

// Test of shared state distribution with many keys
TEST(SharedStateDistribution, TestPopularHashPrevalenceWithMultipleKeys) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    // Initialize the master with IPv4 settings
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch()) << "Failed to launch CCoIP master.";

    // Initialize three clients with IPv4 settings pointing to the master
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    ccoip::CCoIPClient client3({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    // Establish connections between the master and the clients
    establishConnections({&client1, &client2, &client3});

    constexpr size_t total_keys = 16;
    constexpr size_t value_size = 1024;

    // Prepare backup data for client1 to verify post-synchronization state
    std::vector<std::unique_ptr<uint8_t[]>> backup_values;
    for (size_t i = 0; i < total_keys; ++i) {
        auto backup = std::make_unique<uint8_t[]>(value_size);
        std::fill_n(backup.get(), value_size, 42); // Initialize with default value
        backup_values.push_back(std::move(backup));
    }

    // Prepare shared state values for all clients
    // Clients 2 and 3 share the same values for the dirty keys, client1 has different values
    std::vector<std::unique_ptr<uint8_t[]>> client1_values;
    std::vector<std::unique_ptr<uint8_t[]>> client2_values;
    std::vector<std::unique_ptr<uint8_t[]>> client3_values;

    for (size_t i = 0; i < total_keys; ++i) {
        // Initialize all keys with the same value
        client1_values.emplace_back(std::make_unique<uint8_t[]>(value_size));
        client2_values.emplace_back(std::make_unique<uint8_t[]>(value_size));
        client3_values.emplace_back(std::make_unique<uint8_t[]>(value_size));
        std::fill_n(client1_values.back().get(), value_size, 42);
        std::fill_n(client2_values.back().get(), value_size, 42);
        std::fill_n(client3_values.back().get(), value_size, 42);
    }

    // Define dirty keys indices (e.g., first 4 keys)
    std::vector<size_t> dirty_key_indices = {0, 1, 2, 3};

    // Modify dirty keys for client1 to make them different
    for (size_t idx: dirty_key_indices) {
        std::fill_n(client1_values[idx].get(), value_size, 50 + idx); // Unique values for client1
    }

    // Modify dirty keys for client2 and client3 to have the same values
    for (size_t idx: dirty_key_indices) {
        std::fill_n(client2_values[idx].get(), value_size, 100 + idx); // Common values for client2 & client3
        std::fill_n(client3_values[idx].get(), value_size, 100 + idx);
    }

    // Function to create shared state with multiple keys
    auto create_shared_state = [&](const std::vector<std::unique_ptr<uint8_t[]>> &values) -> ccoip_shared_state_t {
        ccoip_shared_state_t shared_state{};
        for (size_t i = 0; i < total_keys; ++i) {
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key" + std::to_string(i + 1),
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = values[i].get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
        }
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyEnforcePopular;
        return shared_state;
    };

    // Launch synchronization threads for each client
    std::thread client1_sync_thread([&client1, &client1_values, &create_shared_state] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state = create_shared_state(client1_values);
        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info)) << "Client1 failed to sync shared state.";
    });

    // Ensure client1 starts syncing first
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::thread client2_sync_thread([&client2, &client2_values, &create_shared_state] {
        client2.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state = create_shared_state(client2_values);
        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client2.syncSharedState(shared_state, info)) << "Client2 failed to sync shared state.";
    });

    // Slight delay before client3 starts syncing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::thread client3_sync_thread([&client3, &client3_values, &create_shared_state] {
        client3.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state = create_shared_state(client3_values);
        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client3.syncSharedState(shared_state, info)) << "Client3 failed to sync shared state.";
    });

    // Wait for all synchronization threads to complete
    client1_sync_thread.join();
    client2_sync_thread.join();
    client3_sync_thread.join();

    // Verify that only the dirty keys in client1 have been updated to the popular values
    for (size_t i = 0; i < total_keys; ++i) {
        if (std::ranges::find(dirty_key_indices, i) != dirty_key_indices.end()) {
            // Dirty key: client1 should have the popular value (from client2/client3)
            EXPECT_EQ(std::memcmp(client1_values[i].get(), client2_values[i].get(), value_size), 0)
                << "Dirty key " << i + 1 << " was not synchronized correctly.";
        } else {
            // Non-dirty key: client1's value should remain unchanged
            EXPECT_EQ(std::memcmp(backup_values[i].get(), client1_values[i].get(), value_size), 0)
                << "Non-dirty key " << i + 1 << " was incorrectly modified.";
        }
    }

    // Additionally, verify that client2 and client3 have identical shared states
    for (size_t i = 0; i < total_keys; ++i) {
        EXPECT_EQ(std::memcmp(client2_values[i].get(), client3_values[i].get(), value_size), 0)
            << "Client2 and Client3 differ on key " << i + 1 << ".";
    }

    // Clean shutdown of all clients
    EXPECT_TRUE(client3.interrupt()) << "Failed to interrupt Client3.";
    EXPECT_TRUE(client2.interrupt()) << "Failed to interrupt Client2.";
    EXPECT_TRUE(client1.interrupt()) << "Failed to interrupt Client1.";

    EXPECT_TRUE(client3.join()) << "Failed to join Client3 thread.";
    EXPECT_TRUE(client2.join()) << "Failed to join Client2 thread.";
    EXPECT_TRUE(client1.join()) << "Failed to join Client1 thread.";

    // Clean shutdown of the master
    EXPECT_TRUE(master.interrupt()) << "Failed to interrupt the master.";
    EXPECT_TRUE(master.join()) << "Failed to join the master thread.";
}

// Test of shared state distribution with multiple peer groups.
// Shared state synchronization should be local to each peer group.
// We test with two peer groups, each with two clients, where each client of a given peer group
// synchronizes shared state that is identical to the other client of the same peer group, but different
// from the shared state of the clients of the other peer group.
// The asserted behavior is that no data is transferred during the two shared state synchronization processes.
TEST(SharedStateDistribution, TestNoSyncIdenticalSharedStateMultiplePeerGroups) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // group 1, client 1
    ccoip::CCoIPClient g1client1({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 0, 1);
    // group 2, client 2
    ccoip::CCoIPClient g1client2({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 0, 1);

    // group 2, client 1
    ccoip::CCoIPClient g2client1({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 1, 1);
    // group 2, client 2
    ccoip::CCoIPClient g2client2({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 1, 1);
    establishConnections({&g1client1, &g1client2, &g2client1, &g2client2});

    constexpr size_t value_size = 1024;

    const auto launch_sync_func = [value_size](ccoip::CCoIPClient &client1, ccoip::CCoIPClient &client2,
                                               const int peer_group) {
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
        std::fill_n(value1.get(), value_size, 42 + peer_group);

        std::thread client1_sync_thread([&client1, value_size, &value1] {
            client1.setMainThread(std::this_thread::get_id());
            // client 1 distributes shared state
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key1",
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value1.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = 0;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client1.syncSharedState(shared_state, info));

            // for identical shared state, no data should be transferred
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);
        });
        std::this_thread::sleep_for(std::chrono::seconds(1));

        const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
        std::fill_n(value2.get(), value_size, 42 + peer_group);
        std::thread client2_sync_thread([&client2, value_size, &value2] {
            client2.setMainThread(std::this_thread::get_id());
            // client 2 requests shared state
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key1",
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value2.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = 0;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client2.syncSharedState(shared_state, info));

            // for identical shared state, no data should be transferred
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);
        });

        // wait for shared state sync to complete
        client1_sync_thread.join();
        client2_sync_thread.join();

        // assert the shared state of client 2 to be equal to that of client 1
        EXPECT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);
    };

    launch_sync_func(g1client1, g1client2, 1);
    launch_sync_func(g2client1, g2client2, 2);

    // clean shutdown
    EXPECT_TRUE(g1client1.interrupt());
    EXPECT_TRUE(g1client2.interrupt());
    EXPECT_TRUE(g2client1.interrupt());
    EXPECT_TRUE(g2client2.interrupt());

    EXPECT_TRUE(g1client1.join());
    EXPECT_TRUE(g1client2.join());
    EXPECT_TRUE(g2client1.join());
    EXPECT_TRUE(g2client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Same setup as TestNoSyncIdenticalSharedStateMultiplePeerGroups, but each peer group has a different shared state mask,
// meaning they shared different keys. Within each group, all peers still contain the same peers.
TEST(SharedStateDistribution, TestNoSyncIdenticalSharedStateMultiplePeerGroupsDifferentKeys) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // group 1, client 1
    ccoip::CCoIPClient g1client1({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 0, 1);
    // group 2, client 2
    ccoip::CCoIPClient g1client2({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 0, 1);

    // group 2, client 1
    ccoip::CCoIPClient g2client1({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 1, 1);
    // group 2, client 2
    ccoip::CCoIPClient g2client2({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 1, 1);
    establishConnections({&g1client1, &g1client2, &g2client1, &g2client2});

    constexpr size_t value_size = 1024;

    const auto launch_sync_func = [value_size](ccoip::CCoIPClient &client1, ccoip::CCoIPClient &client2,
                                               const int peer_group) {
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
        std::fill_n(value1.get(), value_size, 42 + peer_group);

        std::thread client1_sync_thread([&client1, value_size, &value1, peer_group] {
            client1.setMainThread(std::this_thread::get_id());
            // client 1 distributes shared state
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = peer_group == 1 ? "key1" : "key2",
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value1.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = 0;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client1.syncSharedState(shared_state, info));

            // for identical shared state, no data should be transferred
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);
        });
        std::this_thread::sleep_for(std::chrono::seconds(1));

        const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
        std::fill_n(value2.get(), value_size, 42 + peer_group);
        std::thread client2_sync_thread([&client2, value_size, &value2, peer_group] {
            client2.setMainThread(std::this_thread::get_id());

            // client 2 requests shared state
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = peer_group == 1 ? "key1" : "key2",
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value2.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = 0;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client2.syncSharedState(shared_state, info));

            // for identical shared state, no data should be transferred
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);
        });

        // wait for shared state sync to complete
        client1_sync_thread.join();
        client2_sync_thread.join();

        // assert the shared state of client 2 to be equal to that of client 1
        EXPECT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);
    };

    launch_sync_func(g1client1, g1client2, 1);
    launch_sync_func(g2client1, g2client2, 2);

    // clean shutdown
    EXPECT_TRUE(g1client1.interrupt());
    EXPECT_TRUE(g1client2.interrupt());
    EXPECT_TRUE(g2client1.interrupt());
    EXPECT_TRUE(g2client2.interrupt());

    EXPECT_TRUE(g1client1.join());
    EXPECT_TRUE(g1client2.join());
    EXPECT_TRUE(g2client1.join());
    EXPECT_TRUE(g2client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Same setup as TestNoSyncIdenticalSharedStateMultiplePeerGroupsDifferentKeys, but both sync processes are launched concurrently
TEST(SharedStateDistribution, TestNoSyncIdenticalSharedStateMultiplePeerGroupsDifferentKeysConcurrent) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // group 1, client 1
    ccoip::CCoIPClient g1client1({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 0, 1);
    // group 2, client 2
    ccoip::CCoIPClient g1client2({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 0, 1);

    // group 2, client 1
    ccoip::CCoIPClient g2client1({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 1, 1);
    // group 2, client 2
    ccoip::CCoIPClient g2client2({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 1, 1);
    establishConnections({&g1client1, &g1client2, &g2client1, &g2client2});

    constexpr size_t value_size = 1024;

    const auto launch_sync_func = [value_size, &g1client1, &g1client2, &g2client1, &g2client2] {
        const std::unique_ptr<uint8_t[]> value1_g1(new uint8_t[value_size]);
        std::fill_n(value1_g1.get(), value_size, 42 + 1);

        const std::unique_ptr<uint8_t[]> value1_g2(new uint8_t[value_size]);
        std::fill_n(value1_g2.get(), value_size, 42 + 2);

        std::atomic_int client1_sync_ctr = 0;

        std::thread client1_sync_thread(
            [&value1_g1, &value1_g2, value_size, &g1client1, &g2client1, &client1_sync_ctr] {
                // group 1 client 1 distributes shared state
                std::thread g1client1_sync_thread([&g1client1, value_size, &value1_g1, &client1_sync_ctr] {
                    g1client1.setMainThread(std::this_thread::get_id());
                    ccoip_shared_state_t shared_state{};
                    shared_state.entries.push_back(ccoip_shared_state_entry_t{
                        .key = "key1",
                        .data_type = ccoip::ccoipUint8,
                        .device_type = ccoip::ccoipDeviceCpu,
                        .data_ptr = value1_g1.get(),
                        .data_size = value_size,
                        .allow_content_inequality = false
                    });
                    shared_state.revision = 0;
                    ccoip_shared_state_sync_info_t info{};
                    ++client1_sync_ctr;
                    EXPECT_TRUE(g1client1.syncSharedState(shared_state, info));

                    // for identical shared state, no data should be transferred
                    EXPECT_EQ(info.rx_bytes, 0);
                    EXPECT_EQ(info.tx_bytes, 0);
                });

                // group 2 client 1 distributes shared state
                std::thread g2client1_sync_thread([&g2client1, value_size, &value1_g2, &client1_sync_ctr] {
                    g2client1.setMainThread(std::this_thread::get_id());
                    ccoip_shared_state_t shared_state{};
                    shared_state.entries.push_back(ccoip_shared_state_entry_t{
                        .key = "key2",
                        .data_type = ccoip::ccoipUint8,
                        .device_type = ccoip::ccoipDeviceCpu,
                        .data_ptr = value1_g2.get(),
                        .data_size = value_size,
                        .allow_content_inequality = false
                    });
                    shared_state.revision = 0;
                    ccoip_shared_state_sync_info_t info{};
                    ++client1_sync_ctr;
                    EXPECT_TRUE(g2client1.syncSharedState(shared_state, info));

                    // for identical shared state, no data should be transferred
                    EXPECT_EQ(info.rx_bytes, 0);
                    EXPECT_EQ(info.tx_bytes, 0);
                });

                // wait for shared state sync to complete
                g1client1_sync_thread.join();
                g2client1_sync_thread.join();
            });
        while (client1_sync_ctr < 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::thread client2_sync_thread([&value1_g1, &value1_g2, value_size, &g1client2, &g2client2] {
            // group 1 client 2 requests shared state
            std::thread g1client2_sync_thread([&value1_g1, &g1client2, value_size] {
                g1client2.setMainThread(std::this_thread::get_id());
                ccoip_shared_state_t shared_state{};
                shared_state.entries.push_back(ccoip_shared_state_entry_t{
                    .key = "key1",
                    .data_type = ccoip::ccoipUint8,
                    .device_type = ccoip::ccoipDeviceCpu,
                    .data_ptr = value1_g1.get(),
                    .data_size = value_size,
                    .allow_content_inequality = false
                });
                shared_state.revision = 0;
                ccoip_shared_state_sync_info_t info{};
                EXPECT_TRUE(g1client2.syncSharedState(shared_state, info));

                // for identical shared state, no data should be transferred
                EXPECT_EQ(info.rx_bytes, 0);
                EXPECT_EQ(info.tx_bytes, 0);
            });

            // group 2 client 2 requests shared state
            std::thread g2client2_sync_thread([&value1_g2, &g2client2, value_size] {
                g2client2.setMainThread(std::this_thread::get_id());
                ccoip_shared_state_t shared_state{};
                shared_state.entries.push_back(ccoip_shared_state_entry_t{
                    .key = "key2",
                    .data_type = ccoip::ccoipUint8,
                    .device_type = ccoip::ccoipDeviceCpu,
                    .data_ptr = value1_g2.get(),
                    .data_size = value_size,
                    .allow_content_inequality = false
                });
                shared_state.revision = 0;
                ccoip_shared_state_sync_info_t info{};
                EXPECT_TRUE(g2client2.syncSharedState(shared_state, info));

                // for identical shared state, no data should be transferred
                EXPECT_EQ(info.rx_bytes, 0);
                EXPECT_EQ(info.tx_bytes, 0);
            });

            // wait for shared state sync to complete
            g1client2_sync_thread.join();
            g2client2_sync_thread.join();
        });

        // wait for shared state sync to complete
        client1_sync_thread.join();
        client2_sync_thread.join();
    };

    launch_sync_func();

    // clean shutdown
    EXPECT_TRUE(g1client1.interrupt());
    EXPECT_TRUE(g1client2.interrupt());
    EXPECT_TRUE(g2client1.interrupt());
    EXPECT_TRUE(g2client2.interrupt());

    EXPECT_TRUE(g1client1.join());
    EXPECT_TRUE(g1client2.join());
    EXPECT_TRUE(g2client1.join());
    EXPECT_TRUE(g2client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Multistep advancement of shared state with identical updates
TEST(SharedStateDistribution, TestMultiStepAdvancement) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // Client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    constexpr int num_steps = 5;

    // Initialize shared state values
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 0);
    std::fill_n(value2.get(), value_size, 0);

    // Main loop for clients
    auto client_main = [](ccoip::CCoIPClient &client, uint8_t *value) {
        client.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value,
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        for (int step = 0; step < num_steps; ++step) {
            // Independently update shared state identically
            std::fill_n(value, value_size, static_cast<uint8_t>(42 + step));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client.syncSharedState(shared_state, info));

            // Since shared states are identical, no data should be transferred
            EXPECT_EQ(info.tx_bytes, 0);
            EXPECT_EQ(info.rx_bytes, 0);
        }
    };

    // Start client threads
    std::thread client1_main_thread([&client1, &value1, &client_main] {
        client_main(client1, value1.get());
    });
    std::thread client2_main_thread([&client2, &value2, &client_main] {
        client_main(client2, value2.get());
    });

    // Wait for both clients to finish`
    client1_main_thread.join();
    client2_main_thread.join();

    // Assert the shared states are identical
    EXPECT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);

    // Clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// "Drag-along" client scenario
// Two clients distribute the same shared state to establish hash prevalence, and the third client does not update its own shared state but calls syncSharedState.
// This tests whether the third client is "dragged along" by the other two clients and receives the correct shared state.
// This test focuses on just the shared state revision. The shared state revision is not incremented.
// However, after the first sync the shared state of the third peer will match. The test asserts that the shared state revision
// is being updated despite the fact that the peer did not do so out of its own volition post shared state sync,
// while also asserting that because the shared state contents are then equal, no unnecessary shared state transmissions
// occur beyond the initial sync in the first iteration.
TEST(SharedStateDistribution, TestDragAlongClientNoAdvancedStateContents) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // Client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    // Client 3
    ccoip::CCoIPClient client3({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2, &client3});

    constexpr size_t value_size = 1024;
    constexpr int num_steps = 5;

    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value3(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 42);
    std::fill_n(value2.get(), value_size, 42);
    std::fill_n(value3.get(), value_size, 0); // Client 3 does not update value

    // Client 1 continuously updates shared state
    std::thread client1_main_thread([&client1, &value1, value_size, num_steps] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 0; step < num_steps; ++step) {
            // Update value
            std::fill_n(value1.get(), value_size, static_cast<uint8_t>(42));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client1.syncSharedState(shared_state, info));

            // Either client 1 or client 2 may be chosen to distribute the shared state to Client 3;
            // What is certain is that client 3 needs to receive value_size bytes, which is asserted.
            if (info.tx_bytes != 0) {
                EXPECT_EQ(info.tx_bytes, value_size);
            } else {
                EXPECT_EQ(info.tx_bytes, 0);
            }
            EXPECT_EQ(info.rx_bytes, 0);
        }
    });

    // Client 2 continuously updates shared state and distributes the same shared state as client 1
    std::thread client2_main_thread([&client2, &value2, value_size, num_steps] {
        client2.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 0; step < num_steps; ++step) {
            // Update value
            std::fill_n(value2.get(), value_size, static_cast<uint8_t>(42));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client2.syncSharedState(shared_state, info));

            // Either client 1 or client 2 may be chosen to distribute the shared state to Client 3;
            // What is certain is that client 3 needs to receive value_size bytes, which is asserted.
            if (info.tx_bytes != 0) {
                EXPECT_EQ(info.tx_bytes, value_size);
            } else {
                EXPECT_EQ(info.tx_bytes, 0);
            }
            EXPECT_EQ(info.rx_bytes, 0);
        }
    });

    // Client 3 does not update its own shared state but calls syncSharedState
    std::thread client3_main_thread([&client3, &value3, value_size, num_steps] {
        client3.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value3.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0; // Client 2 does not update revision

        const std::unique_ptr<uint8_t[]> value1_inferred(new uint8_t[value_size]);
        for (int step = 0; step < num_steps; ++step) {
            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client3.syncSharedState(shared_state, info));

            // revision should be updated
            EXPECT_EQ(shared_state.revision, step);

            // Client 2 should receive data from client 1 in step 1
            // but after that, the contents are equal and NO FURTHER shared state transmissions should occur
            if (step == 0) {
                EXPECT_EQ(info.tx_bytes, 0);
                EXPECT_EQ(info.rx_bytes, value_size);
            } else {
                // beyond the initial sync, the shared state is not advanced on the other peers,
                // hence it should match going forward
                EXPECT_EQ(info.tx_bytes, 0);
                EXPECT_EQ(info.rx_bytes, 0);
            }

            // infer value1 from step
            std::fill_n(value1_inferred.get(), value_size, static_cast<uint8_t>(42));

            // Value3 should now be updated to match value1
            EXPECT_EQ(std::memcmp(value1_inferred.get(), value3.get(), value_size), 0);
        }
    });

    // Wait for both clients to finish
    client1_main_thread.join();
    client2_main_thread.join();
    client3_main_thread.join();

    // Clean shutdown
    EXPECT_TRUE(client3.interrupt());
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());
    EXPECT_TRUE(client3.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};


// "Drag-along" client scenario
// Two clients distribute the same shared state to establish hash prevalence, and the third client does not update its own shared state but calls syncSharedState.
// This tests whether the third client is "dragged along" by the other two clients and receives the correct shared state.
// This test also advances shared state for each step, which means that every step must result in complete retransmission
// of the shared state for the dragged along peer.
TEST(SharedStateDistribution, TestDragAlongClientWithAdvancedStateContents) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // Client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    // Client 3
    ccoip::CCoIPClient client3({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2, &client3});

    constexpr size_t value_size = 1024;
    constexpr int num_steps = 5;

    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value3(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 42);
    std::fill_n(value2.get(), value_size, 42);
    std::fill_n(value3.get(), value_size, 0); // Client 3 does not update value

    // Client 1 continuously updates shared state
    std::thread client1_main_thread([&client1, &value1, value_size, num_steps] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 0; step < num_steps; ++step) {
            // Update value
            std::fill_n(value1.get(), value_size, static_cast<uint8_t>(42 + step));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client1.syncSharedState(shared_state, info));

            // Either client 1 or client 2 may be chosen to distribute the shared state to Client 3;
            // What is certain is that client 3 needs to receive value_size bytes, which is asserted.
            if (info.tx_bytes != 0) {
                EXPECT_EQ(info.tx_bytes, value_size);
            } else {
                EXPECT_EQ(info.tx_bytes, 0);
            }
            EXPECT_EQ(info.rx_bytes, 0);
        }
    });

    // Client 2 continuously updates shared state and distributes the same shared state as client 1
    std::thread client2_main_thread([&client2, &value2, value_size, num_steps] {
        client2.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 0; step < num_steps; ++step) {
            // Update value
            std::fill_n(value2.get(), value_size, static_cast<uint8_t>(42 + step));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client2.syncSharedState(shared_state, info));

            // Either client 1 or client 2 may be chosen to distribute the shared state to Client 3;
            // What is certain is that client 3 needs to receive value_size bytes, which is asserted.
            if (info.tx_bytes != 0) {
                EXPECT_EQ(info.tx_bytes, value_size);
            } else {
                EXPECT_EQ(info.tx_bytes, 0);
            }
            EXPECT_EQ(info.rx_bytes, 0);
        }
    });

    // Client 3 does not update its own shared state but calls syncSharedState
    std::thread client3_main_thread([&client3, &value3, value_size, num_steps] {
        client3.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value3.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0; // Client 2 does not update revision

        const std::unique_ptr<uint8_t[]> value1_inferred(new uint8_t[value_size]);
        for (int step = 0; step < num_steps; ++step) {
            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client3.syncSharedState(shared_state, info));

            // revision should be updated
            EXPECT_EQ(shared_state.revision, step);

            EXPECT_EQ(info.tx_bytes, 0);
            EXPECT_EQ(info.rx_bytes, value_size);

            // infer value1 from step
            std::fill_n(value1_inferred.get(), value_size, static_cast<uint8_t>(42 + step));

            // Value3 should now be updated to match value1
            EXPECT_EQ(std::memcmp(value1_inferred.get(), value3.get(), value_size), 0);
        }
    });

    // Wait for both clients to finish
    client1_main_thread.join();
    client2_main_thread.join();
    client3_main_thread.join();

    // Clean shutdown
    EXPECT_TRUE(client3.interrupt());
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());
    EXPECT_TRUE(client3.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Test violation of the one-increment rule
// Clients must increment their shared state by exactly one before starting the next shared state synchronization
TEST(SharedStateDistribution, TestOneIncrementRuleViolationSimple) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // Client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    constexpr int num_steps = 2; // Only two steps, client will be kicked after the second step and no further steps

    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 42);
    std::fill_n(value2.get(), value_size, 0);

    // Client 1 continuously updates shared state, but violates the one-increment rule
    std::thread client1_main_thread([&client1, &value1, value_size, num_steps] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 0; step < num_steps; ++step) {
            // Update value
            std::fill_n(value1.get(), value_size, static_cast<uint8_t>(42 + step));
            shared_state.revision = step * 2; // increments by two each time; Illegal!

            ccoip_shared_state_sync_info_t info{};
            bool success = client1.syncSharedState(shared_state, info);
            if (step == 0) {
                // First step should succeed because revision is 0
                EXPECT_TRUE(success);
            } else {
                // Subsequent steps should fail
                EXPECT_FALSE(success);
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Client 2 does not update its own shared state but calls syncSharedState
    std::thread client2_main_thread([&client2, &value2, value_size, num_steps] {
        client2.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0; // Client 2 does not update revision

        const std::unique_ptr<uint8_t[]> value1_inferred(new uint8_t[value_size]);
        for (int step = 0; step < num_steps; ++step) {
            ccoip_shared_state_sync_info_t info{};
            bool success = client2.syncSharedState(shared_state, info);
            if (step == 0) {
                // First step should succeed because revision is 0
                EXPECT_TRUE(success);
            } else {
                EXPECT_FALSE(success);
            }
        }
    });

    // Wait for both clients to finish
    client1_main_thread.join();
    client2_main_thread.join();

    // Clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Calling the first ever sync shared state with revision 13 does not violate the one-increment rule because
// we allow "initialization" for logical resume
TEST(SharedStateDistribution, TestOneIncrementRuleViolationInitialization) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // Client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;

    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 42);
    std::fill_n(value2.get(), value_size, 0);

    // Client 1 violates the one-increment rule by starting at revision 1
    std::thread client1_main_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });

        // Update value
        std::fill_n(value1.get(), value_size, static_cast<uint8_t>(42));
        shared_state.revision = 13;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::thread client2_main_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0; // Client 2 should be updated to revision 13

        const std::unique_ptr<uint8_t[]> value1_inferred(new uint8_t[value_size]);
        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client2.syncSharedState(shared_state, info)); // this one should succeed because it uses revision 0

        EXPECT_EQ(info.tx_bytes, 0);
        EXPECT_EQ(info.rx_bytes, value_size);
        EXPECT_EQ(shared_state.revision, 13);
    });

    // Wait for both clients to finish
    client1_main_thread.join();
    client2_main_thread.join();

    // Clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};


// Client gets kicked due to shared state mask mismatch
// Client 1 and 2 establish popularity, leading to the mask of client 1 and client 2 to be elected,
// but client 3 attempts to sync shared state with mismatched keys that differ from those of client 1 and 2.
// Client 3 should be kicked.
// Subsequent calls to syncSharedState by the remaining clients 1 and 2 should succeed.
// Subsequent calls to syncSharedState by client 3 should also fail.
TEST(SharedStateDistribution, TestSharedStateMaskMismatchKick) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // Client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    // Client 3
    ccoip::CCoIPClient client3({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2, &client3});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value3(new uint8_t[value_size]);

    // Initialize values with same content for all peers
    std::fill_n(value1.get(), value_size, 42);
    std::fill_n(value2.get(), value_size, 42);
    std::fill_n(value3.get(), value_size, 42);

    // Client 1 synchronizes shared state
    std::thread client1_main_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));

        shared_state.revision = 1;
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));
    });

    // Client 2 synchronizes same shared state as client 1 with matching content as keys
    std::thread client2_main_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client2.syncSharedState(shared_state, info));

        shared_state.revision = 1;
        EXPECT_TRUE(client2.syncSharedState(shared_state, info));
    });

    // Client 3 attempts to sync shared state with mismatched keys; content is the same
    std::thread client3_main_thread([&client3, &value3, value_size] {
        client3.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key3", // Mismatched key
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value3.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0; // Same revision, but keyset is different

        ccoip_shared_state_sync_info_t info{};

        // Since the content does not match the mask, client2 should be kicked
        EXPECT_FALSE(client3.syncSharedState(shared_state, info));

        // Subsequent calls should fail
        EXPECT_FALSE(client3.syncSharedState(shared_state, info));
    });

    client1_main_thread.join();
    client2_main_thread.join();
    client3_main_thread.join();

    // Clean shutdown
    EXPECT_TRUE(client1.interrupt());
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client3.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());
    EXPECT_TRUE(client3.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Multiple peer groups advance the shared state for multiple steps; synchronization should be local to each peer group
TEST(SharedStateDistribution, TestConcurrentAdvancementWithinPeerGroups) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Peer group 1 clients
    ccoip::CCoIPClient g1client1({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 0, 1);
    ccoip::CCoIPClient g1client2({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 0, 1);

    // Peer group 2 clients
    ccoip::CCoIPClient g2client1({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 1, 1);
    ccoip::CCoIPClient g2client2({
                                     .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                     .port = CCOIP_PROTOCOL_PORT_MASTER
                                 }, 1, 1);

    establishConnections({&g1client1, &g1client2, &g2client1, &g2client2});

    constexpr size_t value_size = 1024;
    constexpr int num_steps = 5;

    // Function to run client's main loop within a peer group
    auto client_main = [](ccoip::CCoIPClient &client, uint8_t *value, const std::string &key,
                          const int delay_multiplier) {
        client.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = key,
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value,
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 0; step < num_steps; ++step) {
            // Independently update shared state identically within peer group
            std::fill_n(value, value_size, static_cast<uint8_t>(42 + step));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client.syncSharedState(shared_state, info));

            // Since shared states are identical within group, no data should be transferred
            EXPECT_EQ(info.tx_bytes, 0);
            EXPECT_EQ(info.rx_bytes, 0);

            // Sleep to simulate computation with varying delays
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * delay_multiplier));
        }
    };

    // Start clients in peer group 1
    const std::unique_ptr<uint8_t[]> g1value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> g1value2(new uint8_t[value_size]);
    std::fill_n(g1value1.get(), value_size, 0);
    std::fill_n(g1value2.get(), value_size, 0);

    std::thread g1client1_thread([&] {
        client_main(g1client1, g1value1.get(), "key1", 1);
    });
    std::thread g1client2_thread([&] {
        client_main(g1client2, g1value2.get(), "key1", 2); // Different delay
    });

    // Start clients in peer group 2
    const std::unique_ptr<uint8_t[]> g2value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> g2value2(new uint8_t[value_size]);
    std::fill_n(g2value1.get(), value_size, 0);
    std::fill_n(g2value2.get(), value_size, 0);

    std::thread g2client1_thread([&] {
        client_main(g2client1, g2value1.get(), "key2", 2); // Different delay
    });
    std::thread g2client2_thread([&] {
        client_main(g2client2, g2value2.get(), "key2", 1);
    });

    // Wait for all clients to finish
    g1client1_thread.join();
    g1client2_thread.join();
    g2client1_thread.join();
    g2client2_thread.join();

    // Assert the shared states are identical within each peer group
    EXPECT_EQ(std::memcmp(g1value1.get(), g1value2.get(), value_size), 0);
    EXPECT_EQ(std::memcmp(g2value1.get(), g2value2.get(), value_size), 0);

    // Clean shutdown
    EXPECT_TRUE(g1client1.interrupt());
    EXPECT_TRUE(g1client2.interrupt());
    EXPECT_TRUE(g2client1.interrupt());
    EXPECT_TRUE(g2client2.interrupt());

    EXPECT_TRUE(g1client1.join());
    EXPECT_TRUE(g1client2.join());
    EXPECT_TRUE(g2client1.join());
    EXPECT_TRUE(g2client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Concurrent "Drag-Along" operations in multiple peer groups.
// Per peer group two leaders and one follower are established.
// Leaders establish popularity for shared state mask & content and follower is dragged along.
TEST(SharedStateDistribution, TestConcurrentDragAlongAcrossPeerGroups) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Create multiple peer groups
    constexpr int num_peer_groups = 2;
    constexpr int clients_per_group = 3;
    constexpr size_t value_size = 512;
    constexpr int num_steps = 4;

    // Initialize clients for each peer group
    std::vector<std::vector<std::unique_ptr<ccoip::CCoIPClient>>> peer_groups;
    for (int group = 0; group < num_peer_groups; ++group) {
        std::vector<std::unique_ptr<ccoip::CCoIPClient>> group_clients;
        group_clients.reserve(clients_per_group);
        for (int client_id = 0; client_id < clients_per_group; ++client_id) {
            group_clients.emplace_back(std::make_unique<ccoip::CCoIPClient>(ccoip_socket_address_t{
                                                                                .inet = {
                                                                                    .protocol = inetIPv4,
                                                                                    .ipv4 = {.data = {127, 0, 0, 1}}
                                                                                },
                                                                                .port = CCOIP_PROTOCOL_PORT_MASTER
                                                                            }, group, 1));
        }
        peer_groups.emplace_back(std::move(group_clients));
    }

    // Establish connections within each peer group
    std::vector<ccoip::CCoIPClient *> clients_ptrs;
    for (int group = 0; group < num_peer_groups; ++group) {
        for (auto &client: peer_groups[group]) {
            clients_ptrs.push_back(client.get());
        }
    }
    establishConnections(clients_ptrs);

    // Function to perform "drag-along" synchronization within a peer group
    auto drag_along_sync = [&](int group) {
        const auto &clients = peer_groups[group];
        const std::unique_ptr<ccoip::CCoIPClient> &leader1 = clients[0]; // Leader 1 client
        const std::unique_ptr<ccoip::CCoIPClient> &leader2 = clients[1]; // Leader 2 client
        const std::unique_ptr<ccoip::CCoIPClient> &follower = clients[2]; // Follower client

        // Initialize shared state values
        const std::unique_ptr<uint8_t[]> leader1_value(new uint8_t[value_size]);
        const std::unique_ptr<uint8_t[]> leader2_value(new uint8_t[value_size]);
        const std::unique_ptr<uint8_t[]> follower_value(new uint8_t[value_size]);

        std::fill_n(leader1_value.get(), value_size, static_cast<uint8_t>(80 + group));
        std::fill_n(leader2_value.get(), value_size, static_cast<uint8_t>(85 + group));
        std::fill_n(follower_value.get(), value_size, 0); // Follower does not update

        // Leaders perform continuous updates
        std::vector<std::thread> leader_threads;

        auto launch_leader_thread = [&](const std::unique_ptr<ccoip::CCoIPClient> &leader,
                                        const std::unique_ptr<uint8_t[]> &leader_value) {
            std::thread leader_thread([&leader, &leader_value, value_size, num_steps, group] {
                leader->setMainThread(std::this_thread::get_id());
                ccoip_shared_state_t shared_state{};
                shared_state.entries.push_back(ccoip_shared_state_entry_t{
                    .key = "drag_key_" + std::to_string(group),
                    .data_type = ccoip::ccoipUint8,
                    .device_type = ccoip::ccoipDeviceCpu,
                    .data_ptr = leader_value.get(),
                    .data_size = value_size,
                    .allow_content_inequality = false
                });
                shared_state.revision = 0;

                for (int step = 0; step < num_steps; ++step) {
                    // Update leader's shared state
                    std::fill_n(leader_value.get(), value_size, static_cast<uint8_t>(90 + group + step));
                    shared_state.revision = step;

                    ccoip_shared_state_sync_info_t info{};
                    EXPECT_TRUE(leader->syncSharedState(shared_state, info));

                    // Expect one of the leaders to distribute the shared state to the follower
                    // What is certain is that the follower needs to receive value_size bytes, which is asserted.
                    if (info.tx_bytes != 0) {
                        EXPECT_EQ(info.tx_bytes, value_size);
                    } else {
                        EXPECT_EQ(info.tx_bytes, 0);
                    }
                    EXPECT_EQ(info.rx_bytes, 0);

                    // Simulate computation delay
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            });
            leader_threads.emplace_back(std::move(leader_thread));
        };
        launch_leader_thread(leader1, leader1_value);
        launch_leader_thread(leader2, leader2_value);

        // Follower requests synchronization without updating its own state
        std::thread follower_thread([&follower, &leader1_value, &follower_value, value_size, num_steps, group] {
            follower->setMainThread(std::this_thread::get_id());
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "drag_key_" + std::to_string(group),
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = follower_value.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = 0; // Does not update its own revision

            for (int step = 0; step < num_steps; ++step) {
                ccoip_shared_state_sync_info_t info{};
                EXPECT_TRUE(follower->syncSharedState(shared_state, info));

                // Expect follower to receive data from leader
                EXPECT_EQ(info.tx_bytes, 0);
                EXPECT_EQ(info.rx_bytes, value_size);

                // Verify synchronization
                EXPECT_EQ(std::memcmp(leader1_value.get(), follower_value.get(), value_size), 0);

                // Simulate computation delay
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
        for (auto &t: leader_threads) {
            t.join();
        }
        follower_thread.join();
    };

    // Launch "drag-along" synchronization for all peer groups concurrently
    std::vector<std::thread> drag_along_threads;
    drag_along_threads.reserve(num_peer_groups);
    for (int group = 0; group < num_peer_groups; ++group) {
        drag_along_threads.emplace_back(drag_along_sync, group);
    }

    for (auto &t: drag_along_threads) {
        t.join();
    }

    // Clean shutdown
    for (int group = 0; group < num_peer_groups; ++group) {
        for (const auto &client: peer_groups[group]) {
            EXPECT_TRUE(client->interrupt());
            EXPECT_TRUE(client->join());
        }
    }

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Overlapping keys across peer groups
TEST(SharedStateDistribution, TestConflictOverlappingKeysAcrossPeerGroups) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Create peer groups with overlapping keys
    constexpr int num_peer_groups = 2;
    constexpr int clients_per_group = 2;
    constexpr size_t value_size = 256;

    std::vector<std::vector<std::unique_ptr<ccoip::CCoIPClient>>> peer_groups;
    for (int group = 0; group < num_peer_groups; ++group) {
        std::vector<std::unique_ptr<ccoip::CCoIPClient>> group_clients;
        group_clients.reserve(clients_per_group);
        for (int client_id = 0; client_id < clients_per_group; ++client_id) {
            group_clients.push_back(std::make_unique<ccoip::CCoIPClient>(ccoip_socket_address_t{
                                                                             .inet = {
                                                                                 .protocol = inetIPv4,
                                                                                 .ipv4 = {.data = {127, 0, 0, 1}}
                                                                             },
                                                                             .port = CCOIP_PROTOCOL_PORT_MASTER
                                                                         }, group, 1));
        }
        peer_groups.emplace_back(std::move(group_clients));
    }

    // Establish connections within each peer group
    std::vector<ccoip::CCoIPClient *> clients_ptrs;
    for (int group = 0; group < num_peer_groups; ++group) {
        for (auto &client: peer_groups[group]) {
            clients_ptrs.push_back(client.get());
        }
    }
    establishConnections(clients_ptrs);

    // Function to synchronize peer groups with overlapping keys
    auto sync_overlapping_keys = [&](const int group) {
        const auto &clients = peer_groups[group];
        const std::unique_ptr<ccoip::CCoIPClient> &client1 = clients[0];
        const std::unique_ptr<ccoip::CCoIPClient> &client2 = clients[1];

        // Initialize shared state values
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
        const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

        // Fill with unique data per group
        std::fill_n(value1.get(), value_size, static_cast<uint8_t>(200 + group));
        std::fill_n(value2.get(), value_size, static_cast<uint8_t>(200 + group));

        // Key is overlapping across peer groups
        std::string key = "shared_key_overlapping";

        // Client 1 syncs
        std::thread client1_sync_thread([&client1, &value1, value_size, key] {
            client1->setMainThread(std::this_thread::get_id());
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = key,
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value1.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = 0;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client1->syncSharedState(shared_state, info));

            // For identical shared state, no data should be transferred
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);
        });

        // Client 2 syncs
        std::thread client2_sync_thread([&client2, &value1, &value2, value_size, key] {
            client2->setMainThread(std::this_thread::get_id());

            // Allow some time for client1 to sync first
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = key,
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value2.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = 0;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client2->syncSharedState(shared_state, info));

            // For identical shared state, no data should be transferred
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);

            // Verify that the shared states are identical
            EXPECT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);
        });

        client1_sync_thread.join();
        client2_sync_thread.join();
    };

    // Launch synchronization for all peer groups concurrently
    std::vector<std::thread> sync_threads;
    sync_threads.reserve(num_peer_groups);
    for (int group = 0; group < num_peer_groups; ++group) {
        sync_threads.emplace_back(sync_overlapping_keys, group);
    }

    for (auto &t: sync_threads) {
        t.join();
    }

    // Clean shutdown
    for (int group = 0; group < num_peer_groups; ++group) {
        for (const auto &client: peer_groups[group]) {
            EXPECT_TRUE(client->interrupt());
            EXPECT_TRUE(client->join());
        }
    }

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Changing peer group membership during synchronization
// Tests the behavior when a new client is dynamically added to a peer group during ongoing synchronization.
TEST(SharedStateDistribution, TestChangingPeerGroupMembershipBetweenSynchronizationSteps) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // Create a peer group
    constexpr int group = 0;
    constexpr int initial_clients = 2;
    constexpr size_t value_size = 256;
    constexpr int num_steps = 3;

    std::vector<std::unique_ptr<ccoip::CCoIPClient>> group_clients{};
    group_clients.reserve(initial_clients + 1);
    for (int client_id = 0; client_id < initial_clients; ++client_id) {
        group_clients.push_back(std::make_unique<ccoip::CCoIPClient>(ccoip_socket_address_t{
                                                                         .inet = {
                                                                             .protocol = inetIPv4,
                                                                             .ipv4 = {.data = {127, 0, 0, 1}}
                                                                         },
                                                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                                                     }, group, 1));
    }

    // Establish initial connections
    std::vector<ccoip::CCoIPClient *> initial_clients_ptrs;
    initial_clients_ptrs.reserve(group_clients.size());
    for (auto &client: group_clients) {
        initial_clients_ptrs.push_back(client.get());
    }
    establishConnections(initial_clients_ptrs);

    // Initialize shared state values
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 150);
    std::fill_n(value2.get(), value_size, 150);

    std::atomic_bool client3_joined = false;

    // Client 1 performs synchronization steps
    std::thread client1_thread([&group_clients, &value1, value_size, num_steps, group, &client3_joined] {
        const std::unique_ptr<ccoip::CCoIPClient> &client1 = group_clients[0];
        client1->setMainThread(std::this_thread::get_id());

        for (int step = 0; step < num_steps; ++step) {
            // Update shared state
            std::fill_n(value1.get(), value_size, static_cast<uint8_t>(160 + step));

            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key",
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value1.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client1->syncSharedState(shared_state, info));

            // Expect no data transfer for identical shared state
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);

            // Simulate computation delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // At step 2, dynamically add a new client to the peer group
            if (step == 2) {
                // Create and connect new client
                auto new_client = std::make_unique<ccoip::CCoIPClient>(
                    ccoip_socket_address_t{
                        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                        .port = CCOIP_PROTOCOL_PORT_MASTER
                    }, group, 1);
                const std::unique_ptr<ccoip::CCoIPClient> &added_client = group_clients.emplace_back(
                    std::move(new_client));
                std::thread added_client_thread([&added_client, &client3_joined] {
                    added_client->setMainThread(std::this_thread::get_id());
                    std::cout << "Added client connecting" << std::endl;
                    EXPECT_TRUE(added_client->connect());
                    client3_joined = true;
                });
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                EXPECT_TRUE(client1->acceptNewPeers());
                added_client_thread.join();
            }
        }
    });

    // Client 2 performs synchronization steps
    std::thread client2_thread([&group_clients, &value2, value_size, num_steps] {
        for (int step = 0; step < num_steps; ++step) {
            const std::unique_ptr<ccoip::CCoIPClient> &client2 = group_clients[1];
            client2->setMainThread(std::this_thread::get_id());

            // Update local shared state
            std::fill_n(value2.get(), value_size, static_cast<uint8_t>(160 + step));

            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key",
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value2.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client2->syncSharedState(shared_state, info));

            // Expect no data transfer for identical shared state
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);

            // Simulate computation delay
            std::this_thread::sleep_for(std::chrono::milliseconds(150));

            if (step == 2) {
                EXPECT_TRUE(client2->acceptNewPeers());
            }
        }
    });
    while (!client3_joined) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Client 3 performs last synchronization step (3)
    std::thread client3_thread([&group_clients, &value2, value_size, num_steps] {
        const std::unique_ptr<ccoip::CCoIPClient> &client3 = group_clients[2];
        client3->setMainThread(std::this_thread::get_id());

        for (int step = 3; step < num_steps; ++step) {
            // Update local shared state
            std::fill_n(value2.get(), value_size, static_cast<uint8_t>(160 + step));

            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key",
                .data_type = ccoip::ccoipUint8,
                .device_type = ccoip::ccoipDeviceCpu,
                .data_ptr = value2.get(),
                .data_size = value_size,
                .allow_content_inequality = false
            });
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            EXPECT_TRUE(client3->syncSharedState(shared_state, info));

            // Expect no data transfer for identical shared state
            EXPECT_EQ(info.rx_bytes, 0);
            EXPECT_EQ(info.tx_bytes, 0);

            // Simulate computation delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    client1_thread.join();
    client2_thread.join();
    client3_thread.join();

    // Clean shutdown
    for (const auto &client: group_clients) {
        EXPECT_TRUE(client->interrupt());
        EXPECT_TRUE(client->join());
    }

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
};

// Tests the behavior when two clients with different shared state content attempt to synchronize shared
// state while both using send only as a sync strategy.
// this is explicitly not allowed and one of the two peers should be kicked.
TEST(SharedStateDistribution, TestDifferentSharedStatetContentBothSendOnlyStrategyKick) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    std::atomic_bool any_kicked{false};

    std::thread client1_sync_thread([&client1, &value1, value_size, &any_kicked] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        const bool success = client1.syncSharedState(shared_state, info);
        if (!success) {
            any_kicked.store(true, std::memory_order_seq_cst);
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 0x0);
    std::thread client2_sync_thread([&client2, &value2, value_size, &any_kicked] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        const bool success = client2.syncSharedState(shared_state, info);
        if (!success) {
            any_kicked.store(true, std::memory_order_seq_cst);
        }
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // quick and dirty hack to avoid cross-thread visibility issues with any_kicked

    EXPECT_TRUE(any_kicked.load(std::memory_order_seq_cst));

    // clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}

// Tests the behavior when two clients both declare receive only as a sync strategy.
// In this variant both peers have different content, which should not matter as to the decision.
// In this case, no hash popularity election winner can be determined as no client has put its content up for election.
// In this case, both clients should be kicked.
TEST(SharedStateDistribution, TestBothReceiveOnlyStrategyKickDifferentContent) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyRxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 0x0);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyRxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client2.syncSharedState(shared_state, info));
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}


// Tests the behavior when two clients both declare receive only as a sync strategy.
// In this variant both peers have the same content, which should not matter as to the decision.
// In this case, no hash popularity election winner can be determined as no client has put its content up for election.
// In this case, both clients should be kicked.
TEST(SharedStateDistribution, TestBothReceiveOnlyStrategyKickSameContent) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyRxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 42);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyRxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client2.syncSharedState(shared_state, info));
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}

// This test tests that if one peer uses the "enforce popular" sync strategy,
// that all other peers must also use the "enforce popular" strategy.
// In this test two clients are created, one with "enforce popular" and one with "receive only".
// This is technically not a too problematic situation, but we still dis-allow it for reasons of caution
// and reducing unnecessary complexity.
// We expect the peer that did not declare "enforce popular" to be kicked.
TEST(SharedStateDistribution, TestEnforcePopluarSyncStrategyNoMixingWithReceiveOnly) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyEnforcePopular;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 42);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyRxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client2.syncSharedState(shared_state, info));
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}


// This test tests that if one peer uses the "enforce popular" sync strategy,
// that all other peers must also use the "enforce popular" strategy.
// In this test two clients are created, one with "enforce popular" and one with "send only".
// This is technically not a too problematic situation, but we still dis-allow it for reasons of caution
// and reducing unnecessary complexity.
// We expect the peer that did not declare "enforce popular" to be kicked.
TEST(SharedStateDistribution, TestEnforcePopluarSyncStrategyNoMixingWithSendOnly) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyEnforcePopular;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 42);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client2.syncSharedState(shared_state, info));
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // clean shutdown
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}

TEST(SharedStateDistribution, TestEnforcePopularSyncStrategyNoMixingWithSendOnlyWorldSize3) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // client 3
    ccoip::CCoIPClient client3({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2, &client3});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyEnforcePopular;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 42);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client2.syncSharedState(shared_state, info));
    });

    const std::unique_ptr<uint8_t[]> value3(new uint8_t[value_size]);
    std::fill_n(value3.get(), value_size, 42);
    std::thread client3_sync_thread([&client3, &value3, value_size] {
        client3.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value3.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client3.syncSharedState(shared_state, info));
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();
    client3_sync_thread.join();

    // clean shutdown
    EXPECT_TRUE(client3.interrupt());
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());
    EXPECT_TRUE(client3.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}


TEST(SharedStateDistribution, TestEnforcePopularSyncStrategyNoMixingWithSendOnlyAndReceiveOnlyWorldSize3) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // client 3
    ccoip::CCoIPClient client3({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);

    establishConnections({&client1, &client2, &client3});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyEnforcePopular;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 42);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client2.syncSharedState(shared_state, info));
    });

    const std::unique_ptr<uint8_t[]> value3(new uint8_t[value_size]);
    std::fill_n(value3.get(), value_size, 42);
    std::thread client3_sync_thread([&client3, &value3, value_size] {
        client3.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value3.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyRxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client3.syncSharedState(shared_state, info));
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();
    client3_sync_thread.join();

    // clean shutdown
    EXPECT_TRUE(client3.interrupt());
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());
    EXPECT_TRUE(client3.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}

// This peer tests the behavior when 4 peers are in 2 different peer groups of size 2 each,
// where one group has uses send-only on identical content (which is allowed) and the other group mixes send-only with enforce popular.
TEST(SharedStateDistribution, TestCrossPeerGroupLocalMixingLocalKickWorldSize4PeerGroups2) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    EXPECT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER,
                               }, 0, 1);
    // client 2
    ccoip::CCoIPClient client2({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 0, 1);
    // client 3
    ccoip::CCoIPClient client3({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 1, 1);
    // client 4
    ccoip::CCoIPClient client4({
                                   .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                   .port = CCOIP_PROTOCOL_PORT_MASTER
                               }, 1, 1);

    establishConnections({&client1, &client2, &client3, &client4});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);

    std::thread client1_sync_thread([&client1, &value1, value_size] {
        client1.setMainThread(std::this_thread::get_id());
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value1.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyEnforcePopular;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 42);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        client2.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value2.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_FALSE(client2.syncSharedState(shared_state, info));
    });

    const std::unique_ptr<uint8_t[]> value3(new uint8_t[value_size]);
    std::fill_n(value3.get(), value_size, 42);
    std::thread client3_sync_thread([&client3, &value3, value_size] {
        client3.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value3.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client3.syncSharedState(shared_state, info));
    });

    const std::unique_ptr<uint8_t[]> value4(new uint8_t[value_size]);
    std::fill_n(value4.get(), value_size, 42);
    std::thread client4_sync_thread([&client4, &value4, value_size] {
        client4.setMainThread(std::this_thread::get_id());
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .device_type = ccoip::ccoipDeviceCpu,
            .data_ptr = value4.get(),
            .data_size = value_size,
            .allow_content_inequality = false
        });
        shared_state.revision = 0;
        shared_state.sync_strategy = ccoipSyncStrategyTxOnly;

        ccoip_shared_state_sync_info_t info{};
        EXPECT_TRUE(client4.syncSharedState(shared_state, info));
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();
    client3_sync_thread.join();
    client4_sync_thread.join();

    // clean shutdown
    EXPECT_TRUE(client4.interrupt());
    EXPECT_TRUE(client3.interrupt());
    EXPECT_TRUE(client2.interrupt());
    EXPECT_TRUE(client1.interrupt());

    EXPECT_TRUE(client1.join());
    EXPECT_TRUE(client2.join());
    EXPECT_TRUE(client3.join());
    EXPECT_TRUE(client4.join());

    EXPECT_TRUE(master.interrupt());
    EXPECT_TRUE(master.join());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
