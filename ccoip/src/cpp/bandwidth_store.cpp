#include "bandwidth_store.hpp"

#include <algorithm>
#include <iomanip>
#include <pccl_log.hpp>
#include <sstream>
#include <unordered_set>

using namespace ccoip;

bool BandwidthStore::registerPeer(const ccoip_uuid_t peer, const ccoip_inet_address_t &address, const uint32_t group) {
    auto [_, inserted] = registered_peers.emplace(peer, bw_peer_t{address, group});
    return inserted;
}

bool BandwidthStore::storeBandwidth(const ccoip_uuid_t from, const ccoip_uuid_t to, const double send_bandwidth_mpbs) {
    const auto from_it = registered_peers.find(from);
    if (from_it == registered_peers.end())
        return false;
    const auto to_it = registered_peers.find(to);
    if (to_it == registered_peers.end())
        return false;

    const auto &from_addr = from_it->second.address;
    const auto &to_addr = to_it->second.address;

    bandwidth_map[from_addr][to_addr] = send_bandwidth_mpbs;
    return true;
}

std::optional<double> BandwidthStore::getBandwidthMbps(const ccoip_uuid_t from, const ccoip_uuid_t to) const {
    const auto from_it_peer = registered_peers.find(from);
    if (from_it_peer == registered_peers.end())
        return std::nullopt;
    const auto to_it_peer = registered_peers.find(to);
    if (to_it_peer == registered_peers.end())
        return std::nullopt;

    const auto &from_addr = from_it_peer->second.address;
    const auto &to_addr = to_it_peer->second.address;

    const auto row_it = bandwidth_map.find(from_addr);
    if (row_it == bandwidth_map.end())
        return std::nullopt;

    const auto col_it = row_it->second.find(to_addr);
    if (col_it == row_it->second.end())
        return std::nullopt;

    return col_it->second;
}

std::unordered_map<ccoip_inet_address_t, ccoip_uuid_t, AddressHash, AddressEq>
BandwidthStore::addressRepresentativesForGroup(const uint32_t group) const {
    std::unordered_map<ccoip_inet_address_t, ccoip_uuid_t, AddressHash, AddressEq> reps;
    for (const auto &[uuid, info]: registered_peers) {
        if (info.group != group)
            continue;
        if (!reps.contains(info.address)) {
            reps.emplace(info.address, uuid); // first seen UUID in this group becomes representative
        }
    }
    return reps;
}
std::vector<bandwidth_entry> BandwidthStore::getMissingBandwidthEntries(const ccoip_uuid_t peer) const {
    std::vector<bandwidth_entry> missing_entries;

    const auto self_it = registered_peers.find(peer);
    if (self_it == registered_peers.end())
        return missing_entries;

    const auto &self_info = self_it->second;
    const auto &self_addr = self_info.address;
    const uint32_t group = self_info.group;

    const auto reps = addressRepresentativesForGroup(group); // distinct addresses in this group

    // If the group has only one distinct address, require a self-loop once (reused across groups).
    if (reps.size() == 1) {
        const bool have_self = (bandwidth_map.contains(self_addr)) && (bandwidth_map.at(self_addr).contains(self_addr));
        if (!have_self) {
            // Ask this peer to run the self-test. (If you want only one peer to request it,
            // you can gate this on being the group's representative.)
            missing_entries.push_back({peer, peer});
        }
        return missing_entries;
    }

    // Otherwise, same as before: ensure both directions to every other distinct address in the group.
    for (const auto &[other_addr, other_rep_uuid]: reps) {
        if (AddressEq{}(other_addr, self_addr))
            continue;

        // other -> self
        {
            auto row_it = bandwidth_map.find(other_addr);
            bool have = (row_it != bandwidth_map.end()) && row_it->second.contains(self_addr);
            if (!have) {
                missing_entries.push_back({other_rep_uuid, peer});
            }
        }

        // self -> other
        {
            auto row_it = bandwidth_map.find(self_addr);
            bool have = (row_it != bandwidth_map.end()) && row_it->second.contains(other_addr);
            if (!have) {
                missing_entries.push_back({peer, other_rep_uuid});
            }
        }
    }

    return missing_entries;
}
bool BandwidthStore::isBandwidthStoreFullyPopulated() const {
    // Build group -> set of distinct addresses
    std::unordered_map<uint32_t, std::unordered_set<ccoip_inet_address_t, AddressHash, AddressEq>> group_addrs;
    for (const auto &[uuid, info]: registered_peers) {
        group_addrs[info.group].insert(info.address);
    }

    for (const auto &[group, addrs]: group_addrs) {
        const std::size_t n = addrs.size();

        if (n == 1) {
            // Require a self-loop (addr -> addr); can be measured by any group.
            const auto &addr = *addrs.begin();
            const auto row_it = bandwidth_map.find(addr);
            if (row_it == bandwidth_map.end())
                return false;
            if (!row_it->second.contains(addr))
                return false;
            continue;
        }

        if (n > 1) {
            // Require edges from each addr to every other addr in the same group (no self-edges).
            for (const auto &addr: addrs) {
                const auto row_it = bandwidth_map.find(addr);
                if (row_it == bandwidth_map.end())
                    return false;

                std::size_t present = 0;
                for (const auto &dst: addrs) {
                    if (AddressEq{}(dst, addr))
                        continue;
                    if (row_it->second.contains(dst))
                        ++present;
                }
                if (present != n - 1)
                    return false;
            }
        }
    }

    return true;
}


size_t BandwidthStore::getNumberOfRegisteredPeers() const { return registered_peers.size(); }

bool BandwidthStore::unregisterPeer(const ccoip_uuid_t peer) {
    const auto it = registered_peers.find(peer);
    if (it == registered_peers.end())
        return false;

    const ccoip_inet_address_t addr_to_remove = it->second.address; // copy before erase
    registered_peers.erase(it);

    // If no other peer (in any group) uses this address, delete its row and inbound edges
    const bool address_still_used = std::ranges::any_of(
            registered_peers, [&](const auto &p) { return AddressEq{}(p.second.address, addr_to_remove); });

    if (!address_still_used) {
        bandwidth_map.erase(addr_to_remove); // remove row
        for (auto &row: bandwidth_map) {
            row.second.erase(addr_to_remove); // remove inbound edge
        }
    }
    return true;
}

void BandwidthStore::printBandwidthStore() const {
    // Still prints by UUIDs; values are reused transparently for peers sharing the same address.
    std::vector<ccoip_uuid_t> peers{};
    peers.reserve(registered_peers.size());
    for (const auto &[peer_uuid, _info]: registered_peers) {
        peers.push_back(peer_uuid);
    }

    std::stringstream ss{};
    ss << "Bandwidth store:\n";
    for (int i = 0; i < static_cast<int>(peers.size()); ++i) {
        ss << std::setw(5) << i << " ";
    }
    ss << '\n';

    for (size_t from_index = 0; from_index < peers.size(); ++from_index) {
        const auto from = peers[from_index];
        ss << std::to_string(from_index) << " ";
        for (const auto to: peers) {
            const auto bandwidth_opt = getBandwidthMbps(from, to);
            const auto bandwidth = bandwidth_opt.has_value() ? bandwidth_opt.value() / 1000.0 : -1.0;
            ss << std::setw(5) << std::setprecision(2) << std::fixed << bandwidth << " ";
        }
        ss << "\n";
    }
    LOG(DEBUG) << ss.str();
}
