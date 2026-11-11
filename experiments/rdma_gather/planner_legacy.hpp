// Eight logical ranks, four GPUs/CPU shards on each of two nodes.
// CPU-generated indices reproduce tests/engram/test_engram.py's stratification.
// Both methods use owner-major GPU staging followed by the same GPU reorder.
#pragma once
#include "rdma_common.hpp"

struct LegacyGroup { uint32_t count = 0, dst_row = 0; };
struct LegacyPeerPlan {
    std::vector<LegacyGroup> groups;
    std::vector<uint32_t> rows, positions;
};
struct LegacyPlan {
    std::vector<LegacyPeerPlan> peers = std::vector<LegacyPeerPlan>(8);
    std::vector<uint32_t> expected, positions;
    uint64_t histogram[25]{};
    uint64_t remote_rows = 0;
    double plan_ms = 0;
};
static LegacyPlan legacy_full(const Options& o, int round) {
    auto start = Clock::now();
    LegacyPlan p;
    const size_t total = frame_bytes(o) / ROW;
    p.expected.resize(total); p.positions.reserve(total);
    std::mt19937 rng(12345 + o.rank * 1009 + round * 97);
    uint32_t layer_base = 0;
    for (size_t layer = 0; layer < o.entries.size(); ++layer) {
        uint32_t e = o.entries[layer], global = e * o.world, interval = global / o.rows;
        for (int token = 0; token < o.groups; ++token) {
            std::vector<uint32_t> local[8], slots[8];
            for (int slot = 0; slot < o.rows; ++slot) {
                uint32_t index = o.distribution == "stratified" ? slot * interval + rng() % interval : rng() % global;
                uint32_t owner = index / e, row = layer_base + index % e;
                uint32_t output = (layer * o.groups + token) * o.rows + slot;
                local[owner].push_back(row); slots[owner].push_back(output);
                p.expected[output] = owner * o.table_rows + row;
                if (owner / 4 != uint32_t(o.rank / 4)) ++p.remote_rows;
            }
            for (int owner = 0; owner < 8; ++owner) {
                ++p.histogram[local[owner].size()];
                if (local[owner].empty()) continue;
                auto& peer = p.peers[owner];
                peer.groups.push_back({uint32_t(local[owner].size()), uint32_t(peer.rows.size())});
                peer.rows.insert(peer.rows.end(), local[owner].begin(), local[owner].end());
                peer.positions.insert(peer.positions.end(), slots[owner].begin(), slots[owner].end());
            }
        }
        layer_base += e;
    }
    uint32_t packed_base = 0;
    for (auto& peer : p.peers) {
        for (auto& group : peer.groups) group.dst_row += packed_base;
        p.positions.insert(p.positions.end(), peer.positions.begin(), peer.positions.end());
        packed_base += peer.rows.size();
    }
    check(p.positions.size() == total, "plan row count");
    p.plan_ms = us(start) / 1000;
    return p;
}
