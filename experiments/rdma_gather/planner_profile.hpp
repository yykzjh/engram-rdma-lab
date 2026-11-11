#pragma once
#include "planner.hpp"
#include "planner_legacy.hpp"
struct LegacyProfile {
    double total_ms = 0, temporary_ms = 0, append_ms = 0, destroy_ms = 0, merge_ms = 0;
    uint64_t allocations = 0, allocated_bytes = 0;
};
// Diagnostic-only per-token timers perturb runtime; uninstrumented legacy_full is
// recorded alongside this breakdown. Allocation counts observe actual vector growth.
static LegacyPlan legacy_profile(const Options& o, const uint32_t* input, LegacyProfile& stats) {
    auto begin = Clock::now(); LegacyPlan p;
    p.expected.resize(frame_bytes(o) / ROW); p.positions.reserve(p.expected.size());
    for (size_t layer = 0; layer < o.entries.size(); ++layer) {
        uint32_t e = o.entries[layer], base = layer ? o.entries[0] : 0;
        for (int token = 0; token < o.groups; ++token) {
            auto t = Clock::now(); Clock::time_point built, appended;
            {
                std::vector<uint32_t> local[8], slots[8];
                for (int slot = 0; slot < o.rows; ++slot) {
                    uint32_t output = (layer * o.groups + token) * o.rows + slot;
                    uint32_t index = input[output], owner = index / e, row = base + index % e;
                    auto push = [&](std::vector<uint32_t>& v, uint32_t x) {
                        size_t before = v.capacity(); v.push_back(x);
                        if (v.capacity() != before) { ++stats.allocations; stats.allocated_bytes += v.capacity() * sizeof(uint32_t); }
                    };
                    push(local[owner], row); push(slots[owner], output);
                    p.expected[output] = owner * o.table_rows + row;
                }
                built = Clock::now();
                for (int owner = 0; owner < 8; ++owner) {
                    ++p.histogram[local[owner].size()]; if (local[owner].empty()) continue;
                    auto& peer = p.peers[owner];
                    peer.groups.push_back({uint32_t(local[owner].size()), uint32_t(peer.rows.size())});
                    peer.rows.insert(peer.rows.end(), local[owner].begin(), local[owner].end());
                    peer.positions.insert(peer.positions.end(), slots[owner].begin(), slots[owner].end());
                }
                appended = Clock::now();
            }
            auto ended = Clock::now();
            stats.temporary_ms += std::chrono::duration<double, std::milli>(built - t).count();
            stats.append_ms += std::chrono::duration<double, std::milli>(appended - built).count();
            stats.destroy_ms += std::chrono::duration<double, std::milli>(ended - appended).count();
        }
    }
    auto t = Clock::now(); uint32_t base = 0;
    for (auto& peer : p.peers) {
        for (auto& g : peer.groups) g.dst_row += base;
        p.positions.insert(p.positions.end(), peer.positions.begin(), peer.positions.end()); base += peer.rows.size();
    }
    stats.merge_ms = us(t) / 1000; stats.total_ms = us(begin) / 1000; return p;
}
static void compare_plans(const Plan& fast, const LegacyPlan& old) {
    check(fast.positions == old.positions && fast.expected == old.expected, "planner permutation/reference equivalence");
    for (int peer = 0; peer < 8; ++peer) {
        const auto& a = fast.peers[peer]; const auto& b = old.peers[peer];
        check(a.rows.size() == b.rows.size() && a.groups.size() == b.groups.size() &&
              std::equal(a.rows.begin(), a.rows.end(), b.rows.begin()), "planner row equivalence");
        for (size_t i = 0; i < a.groups.size(); ++i)
            check(a.groups[i].count == b.groups[i].count && a.groups[i].dst_row == b.groups[i].dst_row,
                  "planner group equivalence");
    }
}
static void run_planner_profile(const Options& o) {
    pin_core(o, 0, false); Workers workers(o); Plan fast(o); Planner planner(o);
    for (int round = -1; round < o.rounds; ++round) {
        auto t = Clock::now(); auto input = generate_indices(o, std::max(round, 0)); double generate_ms = us(t) / 1000;
        expected_rows(fast, o, input.data());
        // Alternate measured methods to reduce ordering/cache bias.
        double old_ms = 0;
        auto run_old = [&] {
            auto start = Clock::now(); auto old = legacy_full(o, std::max(round, 0)); old_ms = us(start) / 1000;
            compare_plans(fast, old);
        };
        planner.route(input.data(), fast, workers);
        if (round % 2 == 0) { run_old(); planner.route(input.data(), fast, workers); }
        else run_old();
        LegacyProfile stats; auto diagnostic = legacy_profile(o, input.data(), stats); compare_plans(fast, diagnostic);
        std::printf("{\"kind\":\"planner\",\"rank\":%d,\"round\":%d,\"warmup\":%s,\"distribution\":\"%s\","
            "\"tokens\":%d,\"generate_ms\":%.6f,\"legacy_full_ms\":%.6f,\"diagnostic_route_ms\":%.6f,"
            "\"temporary_build_ms\":%.6f,\"peer_append_ms\":%.6f,\"temporary_destroy_ms\":%.6f,\"merge_ms\":%.6f,"
            "\"temporary_allocations\":%llu,\"temporary_allocation_bytes\":%llu,\"optimized_route_ms\":%.6f,"
            "\"decode_count_ms\":%.6f,\"prefix_ms\":%.6f,\"fill_ms\":%.6f,\"equivalent\":true}\n",
            o.rank, round, round < 0 ? "true" : "false", o.distribution.c_str(), o.groups, generate_ms, old_ms,
            stats.total_ms, stats.temporary_ms, stats.append_ms, stats.destroy_ms, stats.merge_ms,
            (unsigned long long)stats.allocations, (unsigned long long)stats.allocated_bytes,
            fast.plan_ms, fast.decode_ms, fast.prefix_ms, fast.fill_ms);
    }
    if (!o.indices_dir.empty()) load_indices(o); // Store the common unprocessed input, outside fetch timing.
}
