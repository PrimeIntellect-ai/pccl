#pragma once

#include <ccoip_types.hpp>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <vector>

#include "ccoip_inet.h"

namespace ccoip {

    struct InetAddressHash {
        std::size_t operator()(const ccoip_inet_address_t &a) const noexcept {
            std::size_t h = std::hash<int>{}(a.protocol);
            if (a.protocol == inetIPv4) {
                for (uint8_t b: a.ipv4.data)
                    h = h * 131u + b;
            } else {
                for (uint8_t b: a.ipv6.data)
                    h = h * 131u + b;
            }
            return h;
        }
    };

    struct InetAddressEq {
        bool operator()(const ccoip_inet_address_t &lhs, const ccoip_inet_address_t &rhs) const noexcept {
            if (lhs.protocol != rhs.protocol)
                return false;
            if (lhs.protocol == inetIPv4) {
                return std::memcmp(lhs.ipv4.data, rhs.ipv4.data, sizeof(lhs.ipv4.data)) == 0;
            }
            return std::memcmp(lhs.ipv6.data, rhs.ipv6.data, sizeof(lhs.ipv6.data)) == 0;
        }
    };

    using AddressHash = InetAddressHash;
    using AddressEq = InetAddressEq;

    using AddressBandwidthRow = std::unordered_map<ccoip_inet_address_t, double, AddressHash, AddressEq>;
    using AddressBandwidthMap = std::unordered_map<ccoip_inet_address_t, AddressBandwidthRow, AddressHash, AddressEq>;

    struct bandwidth_entry {
        ccoip_uuid_t from_peer_uuid;
        ccoip_uuid_t to_peer_uuid;
    };

    struct bw_peer_t {
        ccoip_inet_address_t address;
        uint32_t group;
    };

    class BandwidthStore {
        /**
         * Stores the bandwidth matrix keyed by inet addresses (not UUIDs).
         * Multiple peers that share the same inet address reuse the same entries, even across groups.
         */
        AddressBandwidthMap bandwidth_map;

        /**
         * All registered peers (uuid -> {address, group}).
         */
        std::unordered_map<ccoip_uuid_t, bw_peer_t> registered_peers{};

        // Helpers: representatives by address within a group, and group address sets
        [[nodiscard]] std::unordered_map<ccoip_inet_address_t, ccoip_uuid_t, AddressHash, AddressEq>
        addressRepresentativesForGroup(uint32_t group) const;

    public:
        /**
         * Registers a peer with its address and group identity.
         * @param peer uuid of the peer
         * @param address inet address of the peer
         * @param group uint32_t group id for the peer
         * @return false if already registered, true otherwise
         */
        bool registerPeer(ccoip_uuid_t peer, const ccoip_inet_address_t &address, uint32_t group);

        /**
         * Stores the bandwidth between two peers (by their UUIDs), but persists by address pair.
         * Directional; A->B and B->A are independent. Groups do not restrict storing; edges are global by address.
         */
        [[nodiscard]] bool storeBandwidth(ccoip_uuid_t from, ccoip_uuid_t to, double send_bandwidth_mpbs);

        /**
         * Looks up bandwidth by the peers' addresses. Reused across peers/groups that share addresses.
         */
        [[nodiscard]] std::optional<double> getBandwidthMbps(ccoip_uuid_t from, ccoip_uuid_t to) const;

        /**
         * Determines missing bandwidth entries involving the specified peer, considering ONLY peers
         * in the same group. However, an edge is considered present if any group has already measured
         * that address pair.
         */
        [[nodiscard]] std::vector<bandwidth_entry> getMissingBandwidthEntries(ccoip_uuid_t peer) const;

        /**
         * Returns true iff, for every group, all address pairs within that group are populated
         * (possibly thanks to measurements from other groups).
         */
        [[nodiscard]] bool isBandwidthStoreFullyPopulated() const;

        /**
         * @return the number of registered peers in the bandwidth store.
         */
        [[nodiscard]] size_t getNumberOfRegisteredPeers() const;

        /**
         * Unregisters a peer. Address rows/columns are kept if any peer (in any group) still uses that address.
         */
        [[nodiscard]] bool unregisterPeer(ccoip_uuid_t peer);

        void printBandwidthStore() const;
    };
} // namespace ccoip
