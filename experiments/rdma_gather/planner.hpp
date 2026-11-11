#pragma once
#include "rdma_common.hpp"

// A Buffer owns fixed-capacity storage derived only from max_tokens/world/EPT.
// No index-dependent grouping or address mapping is prepared at construction.
template<class T> struct View {
    T* ptr = nullptr; size_t length = 0;
    T* data() const { return ptr; }
    size_t size() const { return length; }
    T& operator[](size_t i) const { return ptr[i]; }
    T* begin() const { return ptr; }
    T* end() const { return ptr + length; }
};
struct Group { uint32_t count = 0, dst_row = 0; };
struct PeerPlan { View<Group> groups; View<uint32_t> rows; };
struct Plan {
    std::array<PeerPlan, 8> peers;
    std::unique_ptr<Group[]> group_storage;
    std::unique_ptr<uint32_t[]> row_storage;
    std::vector<uint32_t> positions, expected;
    uint64_t histogram[25]{};
    uint64_t remote_rows = 0;
    double plan_ms = 0, decode_ms = 0, prefix_ms = 0, fill_ms = 0;
    explicit Plan(const Options& o) {
        size_t tokens = size_t(o.groups) * o.entries.size(), rows = tokens * o.rows;
        size_t group_capacity = o.layout == "merged" ? (rows + o.merge_rows - 1) / o.merge_rows + o.world :
            tokens * (o.layout == "direct" ? o.rows : std::min(o.rows, o.world));
        group_storage.reset(new Group[group_capacity]);
        row_storage.reset(new uint32_t[rows]);
        if (o.layout != "direct") positions.resize(rows);
        expected.resize(rows); // expected is correctness-harness storage only.
    }
};
class Workers {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable work, done;
    std::function<void(int)> task;
    unsigned epoch = 0; int remaining = 0; bool stopping = false;
    std::exception_ptr error;
public:
    explicit Workers(const Options& o) {
        for (int i = 0; i < 8; ++i) threads.emplace_back([&, i, o] {
            pin_core(o, i, false); unsigned seen = 0;
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                work.wait(lock, [&] { return stopping || epoch != seen; });
                if (stopping) return;
                seen = epoch; auto f = task; lock.unlock();
                try { f(i); } catch (...) {
                    std::lock_guard<std::mutex> guard(mutex);
                    if (!error) error = std::current_exception();
                }
                lock.lock(); if (--remaining == 0) done.notify_one();
            }
        });
    }
    ~Workers() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        work.notify_all(); for (auto& t : threads) t.join();
    }
    template<class F> void run(F f) {
        std::unique_lock<std::mutex> lock(mutex);
        task = f; remaining = 8; error = nullptr; ++epoch; work.notify_all();
        done.wait(lock, [&] { return remaining == 0; });
        task = nullptr; if (error) std::rethrow_exception(error);
    }
};
static uint64_t frame_bytes(const Options& o) { return uint64_t(o.groups) * o.entries.size() * o.rows * ROW; }
static std::vector<uint32_t> generate_indices(const Options& o, int round = 0) {
    std::vector<uint32_t> result(frame_bytes(o) / ROW);
    std::mt19937 rng(12345 + o.rank * 1009 + round * 97);
    size_t i = 0;
    for (int e : o.entries) {
        uint32_t global = o.world * e, interval = global / o.rows;
        for (int token = 0; token < o.groups; ++token)
            for (int slot = 0; slot < o.rows; ++slot)
                result[i++] = o.distribution == "stratified" ? slot * interval + rng() % interval : rng() % global;
    }
    return result;
}
static std::vector<uint32_t> load_indices(const Options& o) {
    if (o.indices_dir.empty()) return generate_indices(o);
    std::string path = o.indices_dir + "/indices-r" + std::to_string(o.rank) + "-t" +
        std::to_string(o.groups) + "-" + o.distribution + ".bin";
    std::ifstream input(path, std::ios::binary);
    if (input) {
        std::vector<uint32_t> result(frame_bytes(o) / ROW);
        input.read(reinterpret_cast<char*>(result.data()), result.size() * sizeof(uint32_t));
        check(size_t(input.gcount()) == result.size() * sizeof(uint32_t), "input indices file size");
        return result;
    }
    auto result = generate_indices(o);
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.data()), result.size() * sizeof(uint32_t));
    check(output.good(), "write indices file"); return result;
}
static void expected_rows(Plan& p, const Options& o, const uint32_t* input) {
    size_t i = 0; uint32_t base = 0;
    for (int e : o.entries) {
        for (size_t j = 0; j < size_t(o.groups) * o.rows; ++j, ++i)
            p.expected[i] = (input[i] / e) * o.table_rows + base + input[i] % e;
        base += e;
    }
}
class Planner {
    struct Counts { uint64_t rows[8]{}, groups[8]{}, hist[25]{}, remote = 0; };
    // Keep writable counters separate even on 128-byte coherency lines,
    // instead of adjoining a neighbouring worker's counters.
    struct alignas(128) PaddedCounts { Counts value; };
    Options o;
    std::vector<uint8_t> owners, counts;
    std::vector<uint32_t> local_rows;
    std::array<Counts, 8> totals;
    std::array<PaddedCounts, 8> padded_totals;
    Counts& total(int worker) { return o.aligned_counts ? padded_totals[worker].value : totals[worker]; }
    bool direct;
    bool merged;
public:
    explicit Planner(const Options& opt) : o(opt), owners(frame_bytes(o) / ROW),
        counts(o.fast_plan ? 0 : size_t(o.groups) * o.entries.size() * 8), local_rows(owners.size()),
        direct(o.layout == "direct"), merged(o.layout == "merged") {}
    void route(const uint32_t* input, Plan& p, Workers& workers) {
        auto begin = Clock::now(); size_t tokens = size_t(o.groups) * o.entries.size();
        for (int worker = 0; worker < 8; ++worker) total(worker) = {};
        workers.run([&](int worker) {
            auto& sum = total(worker); size_t lo = tokens * worker / 8, hi = tokens * (worker + 1) / 8;
            if (o.fast_plan) {
                // Merged WRITE needs per-owner row counts, not per-token group
                // counts/histograms. Still decode every current input index.
                for (size_t layer = 0; layer < o.entries.size(); ++layer) {
                    uint32_t e = o.entries[layer], base = layer ? o.entries[0] : 0;
                    size_t begin = std::max(lo, layer * o.groups) * o.rows;
                    size_t end = std::min(hi, (layer + 1) * o.groups) * o.rows;
                    for (size_t at = begin; at < end; ++at) {
                        uint32_t index = input[at], owner = index / e;
                        if (owner >= 8) throw std::runtime_error("GPU index outside vocabulary");
                        owners[at] = owner; local_rows[at] = base + index % e;
                        ++sum.rows[owner]; sum.remote += owner / 4 != uint32_t(o.rank / 4);
                    }
                }
                return;
            }
            for (size_t token = lo; token < hi; ++token) {
                size_t layer = token / o.groups; uint32_t e = o.entries[layer];
                uint32_t base = layer ? o.entries[0] : 0;
                uint8_t* c = counts.data() + token * 8; std::memset(c, 0, 8);
                int previous_owner = -1;
                for (int slot = 0; slot < o.rows; ++slot) {
                    size_t at = token * o.rows + slot; uint32_t index = input[at], owner = index / e;
                    if (owner >= 8) throw std::runtime_error("GPU index outside vocabulary");
                    owners[at] = owner; local_rows[at] = base + index % e; ++c[owner];
                    ++sum.rows[owner]; sum.remote += owner / 4 != uint32_t(o.rank / 4);
                    if (direct && int(owner) != previous_owner) ++sum.groups[owner];
                    previous_owner = owner;
                }
                for (int owner = 0; owner < 8; ++owner) {
                    ++sum.hist[c[owner]];
                    if (!direct && !merged) sum.groups[owner] += c[owner] != 0;
                }
            }
        });
        p.decode_ms = us(begin) / 1000; auto t = Clock::now();
        uint32_t row_offsets[8][8], group_offsets[8][8], bases[8];
        uint32_t all_rows = 0, all_groups = 0;
        std::memset(p.histogram, 0, sizeof(p.histogram)); p.remote_rows = 0;
        for (int owner = 0; owner < 8; ++owner) {
            bases[owner] = all_rows; uint32_t rows = 0, groups = 0;
            for (int worker = 0; worker < 8; ++worker) {
                row_offsets[worker][owner] = rows; group_offsets[worker][owner] = groups;
                rows += total(worker).rows[owner]; groups += total(worker).groups[owner];
            }
            if (merged) groups = (rows + o.merge_rows - 1) / o.merge_rows;
            p.peers[owner].rows = {p.row_storage.get() + all_rows, rows};
            p.peers[owner].groups = {p.group_storage.get() + all_groups, groups};
            all_rows += rows; all_groups += groups;
        }
        for (int worker = 0; worker < 8; ++worker) {
            const auto& sum = total(worker);
            p.remote_rows += sum.remote;
            for (int n = 0; n <= 24; ++n) p.histogram[n] += sum.hist[n];
        }
        p.prefix_ms = us(t) / 1000; t = Clock::now();
        workers.run([&](int worker) {
            uint32_t rc[8], gc[8];
            std::memcpy(rc, row_offsets[worker], sizeof(rc)); std::memcpy(gc, group_offsets[worker], sizeof(gc));
            size_t lo = tokens * worker / 8, hi = tokens * (worker + 1) / 8;
            if (o.fast_plan) {
                for (size_t at = lo * o.rows; at < hi * o.rows; ++at) {
                    int owner = owners[at]; uint32_t row = rc[owner]++;
                    p.peers[owner].rows[row] = local_rows[at];
                    p.positions[bases[owner] + row] = at;
                }
                return;
            }
            for (size_t token = lo; token < hi; ++token) {
                const uint8_t* c = counts.data() + token * 8;
                if (direct) {
                    // One WR per contiguous owner run in THIS token. For the
                    // Engram stratified indices each owner has exactly one run.
                    for (int slot = 0; slot < o.rows;) {
                        size_t at = token * o.rows + slot; int owner = owners[at], end = slot + 1;
                        while (end < o.rows && owners[token * o.rows + end] == owner) ++end;
                        p.peers[owner].groups[gc[owner]++] = {uint32_t(end - slot), uint32_t(at)};
                        for (int i = slot; i < end; ++i)
                            p.peers[owner].rows[rc[owner]++] = local_rows[token * o.rows + i];
                        slot = end;
                    }
                } else {
                    if (!merged)
                        for (int owner = 0; owner < 8; ++owner) if (c[owner])
                            p.peers[owner].groups[gc[owner]++] = {c[owner], bases[owner] + rc[owner]};
                    for (int slot = 0; slot < o.rows; ++slot) {
                        size_t at = token * o.rows + slot; int owner = owners[at]; uint32_t row = rc[owner]++;
                        p.peers[owner].rows[row] = local_rows[at]; p.positions[bases[owner] + row] = at;
                    }
                }
            }
        });
        if (merged) workers.run([&](int owner) {
            auto& peer = p.peers[owner];
            for (size_t i = 0; i < peer.groups.size(); ++i) {
                uint32_t at = i * o.merge_rows;
                peer.groups[i] = {uint32_t(std::min(size_t(o.merge_rows), peer.rows.size() - at)), bases[owner] + at};
            }
        });
        p.fill_ms = us(t) / 1000; p.plan_ms = us(begin) / 1000;
    }
};
