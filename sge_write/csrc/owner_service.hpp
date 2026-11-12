#pragma once

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "rdma.hpp"

namespace engram_sge {

// 初始化后只读。CPU服务只需要邮箱、词表和已连接的数据通道，不依赖Python或CUDA stream。
struct OwnerServiceConfig {
    unsigned world, layers, capacity, pool, qps, row_bytes;
    uint64_t table_rows, slot_bytes;
    CUdeviceptr table, inbox;
};

// 一个RDMA requester对应一个服务线程，各层邮箱独立推进。
// Runtime必须先stop_and_join，再释放Channel、MR以及CPU词表/邮箱。
class OwnerServices {
    OwnerServiceConfig config_{};
    std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
    std::mutex error_mutex_;
    std::string error_;

    void serve(unsigned requester, Channel &channel) {
        try {
            const auto &c = config_;
            std::vector<uint64_t> completed(c.layers, 0);
            // 每个SGE描述一行；长度/lkey固定，addr由当前GPU请求的行号动态生成。
            // Channel将至多kMerge个离散源行合并为一个连续目标区间的RDMA WRITE。
            std::vector<ibv_sge> sges(size_t(c.pool) * kMerge, {0, c.row_bytes, channel.mr->lkey});
            while (!stop_.load(std::memory_order_relaxed)) {
                // 空owner/空层没有新请求，必须继续轮询其他层，不能阻塞在某一槽上。
                for (unsigned layer = 0; layer < c.layers; ++layer) {
                    auto *header = reinterpret_cast<RequestHeader *>(c.inbox + (size_t(requester) * c.layers + layer) *
                                                                                   c.slot_bytes);
                    // GPU先发布行号/元数据，最后signal ready；acquire后才读取请求内容。
                    const auto epoch = __atomic_load_n(&header->ready, __ATOMIC_ACQUIRE);
                    if (epoch == completed[layer])
                        continue;
                    ck(epoch == completed[layer] + 1, "mailbox sequence skipped");
                    const unsigned rows = header->rows, base = header->base, active = header->active_qps;
                    ck(rows <= c.capacity && base >= layer * c.capacity &&
                           uint64_t(base) + rows <= (uint64_t(layer) + 1) * c.capacity,
                       "mailbox bounds");
                    ck(active >= 1 && active <= c.qps, "mailbox QP count");
                    const auto *row_ids = reinterpret_cast<const unsigned *>(reinterpret_cast<const char *>(header) +
                                                                             sizeof(RequestHeader));

                    // SQ容量按allocated QPs分摊；仅使用部分QP时同比缩小提交窗口，
                    // 防止active=1的大请求把整个pool压到单个小SQ。
                    const unsigned window_groups = std::min(c.pool, ((c.pool + c.qps - 1) / c.qps) * active);
                    const unsigned window_rows = window_groups * kMerge;
                    for (unsigned offset = 0; offset < rows; offset += window_rows) {
                        const unsigned n = std::min(window_rows, rows - offset);
                        for (unsigned i = 0; i < n; ++i) {
                            ck(row_ids[offset + i] < c.table_rows, "vocabulary row bounds");
                            sges[i].addr = c.table + uint64_t(row_ids[offset + i]) * c.row_bytes;
                        }
                        channel.write(sges, n, base + offset, active, c.row_bytes);
                    }
                    // 每个使用过的QP分别发布DONE，同QP data WR在其DONE之前。
                    // GPU hook必须收齐本层各owner/各QP的epoch，不能仅检查QP0。
                    const unsigned used = std::max(1u, std::min(active, (rows + kMerge - 1) / kMerge));
                    channel.done(epoch, layer, used, c.world);
                    completed[layer] = epoch;
                }
            }
        } catch (const std::exception &error) {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (error_.empty())
                error_ = error.what();
            stop_.store(true, std::memory_order_relaxed);
            std::fprintf(stderr, "SGE owner: %s\n", error.what());
        }
    }

  public:
    OwnerServices() = default;
    OwnerServices(const OwnerServices &) = delete;
    OwnerServices &operator=(const OwnerServices &) = delete;
    ~OwnerServices() { stop_and_join(); }

    void start(OwnerServiceConfig config, const std::vector<std::unique_ptr<Channel>> &channels) {
        ck(threads_.empty(), "owner services already started");
        config_ = config;
        for (unsigned peer = 0; peer < channels.size(); ++peer) {
            if (channels[peer]) {
                auto *channel = channels[peer].get();
                threads_.emplace_back([this, peer, channel] { serve(peer, *channel); });
            }
        }
    }
    void check() {
        std::lock_guard<std::mutex> lock(error_mutex_);
        ck(error_.empty(), error_.c_str());
    }
    size_t thread_count() const { return threads_.size(); }
    void stop_and_join() noexcept {
        stop_.store(true, std::memory_order_relaxed);
        for (auto &thread : threads_)
            if (thread.joinable())
                thread.join();
        threads_.clear();
    }
};

} // namespace engram_sge
