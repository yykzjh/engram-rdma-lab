#pragma once

#include <memory>
#include <vector>
#include <pybind11/stl.h>

#include <deep_ep/common/math.cuh>
#include "engram_storage.hpp"
#include "../kernels/comm/api.hpp"

namespace deep_ep::engram {

// 只负责CPU词表的生命周期、EGM别名及GPU路由元数据；不发起任何通信kernel。
// Buffer在GPU barrier之后释放importer，再经barrier允许owner释放原始分配。
class EgmRouting {
    const bool enabled_;
    bool probed_ = false, ready_ = false, has_direct_ = false;
    std::shared_ptr<EngramHostMemory> owner_;
    std::vector<unsigned> routes_;
    std::vector<std::string> mapping_errors_;
    std::vector<void *> pointers_;
    std::vector<std::unique_ptr<EngramFabricAlias>> fabric_aliases_;
    void **storage_gpu_ = nullptr;
    unsigned *groups_gpu_ = nullptr;

  public:
    explicit EgmRouting(bool enabled) : enabled_(enabled) {}
    EgmRouting(const EgmRouting &) = delete;
    EgmRouting &operator=(const EgmRouting &) = delete;
    ~EgmRouting() { clear_imports(); }

    bool ready() const { return ready_; }
    bool enabled() const { return has_direct_; }
    void **storage() const { return storage_gpu_; }
    unsigned *groups() const { return groups_gpu_; }

    pybind11::dict bootstrap(const std::shared_ptr<EngramHostMemory> &owner, size_t bytes) {
        // 持有原始CUDA allocation handle；FABRIC只能由exporter的原始分配导出。
        owner_ = owner;
        auto blob = owner_ ? owner_->export_fabric() : std::vector<unsigned char>{};
        pybind11::dict result;
        result["table_bytes"] = bytes;
        result["table_fabric"] = pybind11::bytes(reinterpret_cast<const char *>(blob.data()), blob.size());
        result["fabric_error"] = owner_ ? owner_->fabric_error : "disabled";
        return result;
    }

    pybind11::dict probe(const comm::Context &context, const pybind11::list &peers,
                         const std::vector<unsigned> &candidates) {
        engram_route_check(!probed_ && !ready_ && peers.size() == context.num_ranks &&
                               candidates.size() == size_t(context.num_ranks),
                           "READ route probe state");
        int device;
        CUDA_RUNTIME_CHECK(cudaGetDevice(&device));
        const int local = context.gin_min_stride;
        routes_ = candidates;
        mapping_errors_.resize(context.num_ranks);
        pointers_.resize(context.num_ranks, nullptr);
        fabric_aliases_.resize(context.num_ranks);
        for (int peer = 0; peer < context.num_ranks; ++peer) {
            engram_route_check(candidates[peer] <= 1 && (!candidates[peer] || enabled_), "invalid READ EGM candidate");
            if (!candidates[peer])
                continue;
            try {
                auto metadata = peers[peer].cast<pybind11::dict>();
                engram_route_check(pybind11::cast<size_t>(metadata["table_bytes"]) ==
                                       size_t(context.num_rdma_storage_bytes),
                                   "READ EGM owner size mismatch");
                if (peer / local == context.rank_idx / local) {
                    // 本机词表由Context预映射；这里只探测该GPU对owner区间的访问权限，
                    // 回退时保留GIN仍需使用的基础映射，不为EGM另建CPU词表副本。
                    auto *pointer = math::advance_ptr(context.local_rdma_storage_ptr,
                                                      int64_t(peer % local - context.rank_idx % local) *
                                                          context.num_rdma_storage_bytes);
                    CUmemLocation location{CU_MEM_LOCATION_TYPE_DEVICE, device};
                    unsigned long long access = 0;
                    engram_route_cuda(
                        deep_ep::lazy_cuMemGetAccess(&access, &location, reinterpret_cast<CUdeviceptr>(pointer)),
                        "probe READ EGM GPU access");
                    engram_route_check(access & CU_MEM_ACCESS_FLAGS_PROT_READ, "READ EGM GPU read access denied");
                    pointers_[peer] = pointer;
                } else {
                    auto alias =
                        std::make_unique<EngramFabricAlias>(pybind11::cast<std::string>(metadata["table_fabric"]),
                                                            pybind11::cast<size_t>(metadata["table_bytes"]), device);
                    pointers_[peer] = reinterpret_cast<void *>(alias->ptr);
                    fabric_aliases_[peer] = std::move(alias);
                }
            } catch (const std::runtime_error &error) {
                // 仅捕获本条EGM映射/访问探测失败。全局内存分配、基础通信初始化
                // 等错误仍由调用方集体报错，不能伪装成可恢复的链路降级。
                routes_[peer] = 0;
                pointers_[peer] = nullptr;
                mapping_errors_[peer] = error.what();
            }
        }
        probed_ = true;
        pybind11::dict result;
        result["routes"] = routes_;
        result["errors"] = mapping_errors_;
        return result;
    }

    void configure(const comm::Context &context, const std::vector<unsigned> &selected) {
        engram_route_check(probed_ && !ready_ && selected.size() == routes_.size(), "READ route commit state");
        const int local = context.gin_min_stride;
        std::vector<unsigned> direct_groups(context.num_ranks / local, 1);
        has_direct_ = false;
        for (int peer = 0; peer < context.num_ranks; ++peer) {
            // 协商只能把探测成功的链路降级，不能把失败的映射重新升级为EGM。
            engram_route_check(selected[peer] <= routes_[peer], "READ route exceeds probed capability");
            if (!selected[peer]) {
                pointers_[peer] = nullptr;
                fabric_aliases_[peer].reset();
                direct_groups[peer / local] = 0;
            } else {
                has_direct_ = true;
            }
        }
        routes_ = selected;
        // 部分owner回退时，GIN分组仍保留flush/wait；只有全直读组才跳过描述符。
        const auto &pointers = pointers_;
        CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void **>(&storage_gpu_), pointers.size() * sizeof(void *)));
        CUDA_RUNTIME_CHECK(
            cudaMemcpy(storage_gpu_, pointers.data(), pointers.size() * sizeof(void *), cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&groups_gpu_), direct_groups.size() * sizeof(unsigned)));
        CUDA_RUNTIME_CHECK(cudaMemcpy(groups_gpu_, direct_groups.data(), direct_groups.size() * sizeof(unsigned),
                                      cudaMemcpyHostToDevice));
        ready_ = true;
    }

    pybind11::dict info() const {
        pybind11::dict result;
        result["egm_enabled"] = enabled_;
        result["routes"] = routes_;
        result["egm_mapping_errors"] = mapping_errors_;
        result["cpu_vocabulary_allocations"] = owner_ ? 1 : 0;
        result["cpu_vocabulary_allocated_bytes"] = owner_ ? owner_->bytes : 0;
        result["fabric_error"] = owner_ ? owner_->fabric_error : "disabled";
        return result;
    }

    // 调用方保证没有在途kernel；析构清理不抛异常。
    void clear_imports() noexcept {
        if (storage_gpu_) {
            cudaFree(storage_gpu_);
            storage_gpu_ = nullptr;
        }
        if (groups_gpu_) {
            cudaFree(groups_gpu_);
            groups_gpu_ = nullptr;
        }
        fabric_aliases_.clear();
        pointers_.clear();
        probed_ = ready_ = has_direct_ = false;
    }
    void release_owner() { owner_.reset(); }
};

} // namespace deep_ep::engram
