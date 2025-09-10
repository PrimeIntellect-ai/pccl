#include "benchmark_runner.hpp"

#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <tinysockets.hpp>
#include "win_sock_bridge.h"

#define BENCHMARK_LENGTH_SECONDS 2

#define DEFAULT_SEND_BUFFER_SIZE (8 * (1 << 20)) // 8 MB

ccoip::NetworkBenchmarkRunner::NetworkBenchmarkRunner(const ccoip_uuid_t self_peer_uuid,
                                                      const ccoip_socket_address_t &benchmark_endpoint) :
    self_peer_uuid(self_peer_uuid), benchmark_endpoint(benchmark_endpoint) {
    output_bandwidth_mbps.resize(num_benchmark_connections);
}


ccoip::NetworkBenchmarkRunner::BenchmarkResult ccoip::NetworkBenchmarkRunner::runBlocking() {
    LOG(DEBUG) << "NetworkBenchmarkRunner connecting to endpoint " << ccoip_sockaddr_to_str(benchmark_endpoint)
               << "...";
    for (int con_nr = 0; con_nr < num_benchmark_connections; ++con_nr) {
        if (const auto result = launchBenchmark(con_nr); result != BenchmarkResult::SUCCESS) {
            for (auto &thread: running_benchmark_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            return result;
        }
    }
    // await pending benchmarks
    for (auto &thread: running_benchmark_threads) {
        if (thread.joinable()) {
            thread.join();
        }
        if (has_send_failure.load(std::memory_order_relaxed)) {
            LOG(WARN) << "NetworkBenchmarkRunner encountered send failure in one of the connection threads";
            return BenchmarkResult::SEND_FAILURE;
        }
    }
    all_benchmarks_completed.store(true, std::memory_order_relaxed);
    return BenchmarkResult::SUCCESS;
}

double ccoip::NetworkBenchmarkRunner::getOutputBandwidthMbitsPerSecond() {
    if (!all_benchmarks_completed.load(std::memory_order_relaxed)) {
        LOG(WARN) << "NetworkBenchmarkRunner::getOutputBandwidthMbitsPerSecond() called before all benchmarks "
                     "completed after encountering a send failure";
        return 0;
    }
    output_bandwidth_mbps_mutex.lock();
    double total_bandwidth_mbits_per_second = 0;
    for (const auto &bandwidth: output_bandwidth_mbps) {
        total_bandwidth_mbits_per_second += bandwidth;
    }
    output_bandwidth_mbps_mutex.unlock();
    return total_bandwidth_mbits_per_second;
}

ccoip::NetworkBenchmarkRunner::BenchmarkResult
ccoip::NetworkBenchmarkRunner::launchBenchmark(const int connection_number) {
    LOG(INFO) << "Launching Benchmark Thread [" << (connection_number + 1) << "/" << num_benchmark_connections
              << "] for " << ccoip_sockaddr_to_str(benchmark_endpoint) << "...";
    auto socket = std::make_unique<tinysockets::BlockingIOSocket>(benchmark_endpoint);

    if (!socket->establishConnection()) {
        LOG(WARN) << "Failed to establish connection to benchmark endpoint "
                  << ccoip_sockaddr_to_str(benchmark_endpoint) << ".";
        return BenchmarkResult::CONNECTION_FAILURE;
    }
    C2BPacketHello packet{};
    packet.peer_uuid = self_peer_uuid;
    if (!socket->sendPacket(packet)) {
        return BenchmarkResult::SEND_FAILURE;
    }

    const auto packet_opt = socket->receivePacket<B2CPacketBenchmarkServerIsBusy>();
    if (!packet_opt) {
        LOG(ERR) << "Failed to receive B2CPacketBenchmarkServerIsBusy from "
                 << ccoip_sockaddr_to_str(benchmark_endpoint);
        return BenchmarkResult::CONNECTION_FAILURE;
    }
    if (packet_opt->is_busy) {
        if (!socket->closeConnection()) {
            LOG(WARN) << "Failed to close connection to " << ccoip_sockaddr_to_str(benchmark_endpoint)
                      << " after benchmark server indicated it was busy. Ignoring...";
        }
        return BenchmarkResult::BENCHMARK_SERVER_BUSY;
    }

    std::thread benchmark_thread([this, socket = std::move(socket), connection_number] {
        constexpr size_t send_buffer_size = DEFAULT_SEND_BUFFER_SIZE;

        // run benchmark for 10 seconds
        const int socket_fd = socket->getSocketFd();
        const std::unique_ptr<uint8_t[]> buffer(new uint8_t[send_buffer_size]);

        // fill random
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint16_t> dis(0, 255); // uint16_t to make MSVC happy
            for (size_t i = 0; i < send_buffer_size; ++i) {
                buffer[i] = static_cast<uint8_t>(dis(gen));
            }
        }

        size_t total_bytes_sent = 0;

        const auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            const auto now = std::chrono::high_resolution_clock::now();
            if (now - start_time >= std::chrono::seconds(BENCHMARK_LENGTH_SECONDS)) {
                break;
            }

            const auto remaining = std::chrono::seconds(BENCHMARK_LENGTH_SECONDS) - (now - start_time);
            const int remaining_ms =
                    static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count());

            std::vector descriptors = {
                    tinysockets::poll::PollDescriptor{socket_fd, tinysockets::poll::PollEvent::POLL_OUTPUT},
            };

            poll(descriptors, remaining_ms);
            auto &tx_descriptor = descriptors[0];
            if (!tx_descriptor.hasEvent(tinysockets::poll::PollEvent::POLL_OUTPUT)) {
                continue;
            }

            std::span<const std::byte> send_span(reinterpret_cast<const std::byte *>(buffer.get()), send_buffer_size);
            errno = 0;
            const auto n_sent = tinysockets::poll::send_nonblocking(send_span, tx_descriptor);
            if (!n_sent) {
                if (errno != 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                    LOG(ERR) << "Failed to send data to benchmark server: " << std::strerror(errno);
                    has_send_failure.store(true, std::memory_order_relaxed);
                    break;
                }
                continue;
            }
            total_bytes_sent += *n_sent;
        }

        const auto now = std::chrono::high_resolution_clock::now();
        if (!socket->closeConnection()) {
            LOG(WARN) << "Failed to close connection to " << ccoip_sockaddr_to_str(benchmark_endpoint)
                      << " after benchmark. Ignoring...";
        }
        const auto duration = now - start_time;
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        const auto duration_seconds = static_cast<double>(duration_us) / 1e6;
        const auto bandwidth_mbits_per_second =
                total_bytes_sent > 0 ? static_cast<double>(total_bytes_sent * 8) / 1e6 / duration_seconds : 0;

        output_bandwidth_mbps_mutex.lock();
        output_bandwidth_mbps[connection_number] = bandwidth_mbits_per_second;
        output_bandwidth_mbps_mutex.unlock();
    });
    running_benchmark_threads.push_back(std::move(benchmark_thread));
    return BenchmarkResult::SUCCESS;
}

bool ccoip::NetworkBenchmarkHandler::runBlocking(const int socket_fd) {
    size_t total_bytes_received = 0;
    constexpr size_t buffer_size = DEFAULT_SEND_BUFFER_SIZE;
    const std::unique_ptr<uint8_t[]> buffer(new uint8_t[buffer_size]);
    const auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        if (auto now = std::chrono::high_resolution_clock::now();
            now - start > std::chrono::seconds(BENCHMARK_LENGTH_SECONDS + 8)) { // grant an additional 8 seconds
            break;
        }

        std::vector descriptors = {
                tinysockets::poll::PollDescriptor{socket_fd, tinysockets::poll::PollEvent::POLL_INPUT},
        };
        auto &rx_descriptor = descriptors[0];
        poll(descriptors, 0);

        if (!rx_descriptor.hasEvent(tinysockets::poll::PollEvent::POLL_INPUT)) {
            continue;
        }

        std::span buffer_span(reinterpret_cast<std::byte *>(buffer.get()), buffer_size);
        const auto n_received = recv_nonblocking(buffer_span, rx_descriptor);
        if (!n_received) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            // check if closed
            if (errno == ECONNRESET || errno == ECONNABORTED || errno == ENOTCONN) {
                break;
            }

            LOG(ERR) << "Failed to receive data from benchmark client.";
            closesocket(socket_fd);
            return false;
        }
        if (*n_received == 0) {
            break;
        }
        total_bytes_received += *n_received;
    }
    shutdown(socket_fd, SHUT_RDWR);
    return true;
}
