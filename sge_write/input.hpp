#pragma once
#include "rdma_common.hpp"
struct Group{uint32_t count=0,dst_row=0;};
struct Plan{std::vector<uint32_t>expected;explicit Plan(const Options&o):expected(size_t(o.groups)*o.entries.size()*o.rows){}};
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
