#pragma once
#include <cuda.h>
#include "../utils/lazy_driver.hpp"
#include <unistd.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>

inline void engram_route_check(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
inline void engram_route_cuda(CUresult rc, const char *what) {
    if (rc != CUDA_SUCCESS) {
        const char *s = nullptr;
        deep_ep::lazy_cuGetErrorString(rc, &s);
        throw std::runtime_error(std::string(what) + ": " + (s ? s : "CUDA error"));
    }
}
inline size_t engram_route_aligned(size_t n, size_t a = 4096) {
    return (n + a - 1) / a * a;
}

// 独立CPU NUMA词表/邮箱分配。导出的fd在本节点所有进程完成import前保持有效。
struct EngramHostMemory {
    CUdeviceptr ptr = 0;
    CUmemGenericAllocationHandle handle = 0;
    size_t bytes = 0;
    int fd = -1, numa = -1;
    bool fabric_enabled = false;
    std::string fabric_error = "disabled";
    EngramHostMemory(size_t requested, int device, bool use_fabric = false) {
        engram_route_cuda(deep_ep::lazy_cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device),
                          "host NUMA");

        CUmemAllocationProp p{};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location = {CU_MEM_LOCATION_TYPE_HOST_NUMA, numa};
        p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        // FABRIC handle可跨进程/主机传递；POSIX fd只用于同一主机的映射。
        // 不支持HOST_NUMA跨节点IPC或没有IMEX权限时保留普通EGM/RDMA分配，
        // bootstrap会公布空fabric handle，使该owner的跨节点访问明确走RDMA。
        if (use_fabric) {
            int supported = 0, host_ipc = 0;
#if CUDA_VERSION >= 13000
            engram_route_cuda(deep_ep::lazy_cuDeviceGetAttribute(
                                  &supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device),
                              "fabric attribute");
            engram_route_cuda(deep_ep::lazy_cuDeviceGetAttribute(
                                  &host_ipc, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, device),
                              "host fabric attribute");
#endif
            fabric_enabled = supported && host_ipc;
            fabric_error = fabric_enabled ? "" : "HOST_NUMA fabric IPC unavailable";
            if (fabric_enabled)
                p.requestedHandleTypes =
                    static_cast<CUmemAllocationHandleType>(p.requestedHandleTypes | CU_MEM_HANDLE_TYPE_FABRIC);
        }
        size_t granularity;
        engram_route_cuda(
            deep_ep::lazy_cuMemGetAllocationGranularity(&granularity, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
            "host granularity");
        bytes = engram_route_aligned(requested, granularity);
        auto rc = deep_ep::lazy_cuMemCreate(&handle, bytes, &p, 0);
        if (fabric_enabled && (rc == CUDA_ERROR_NOT_SUPPORTED || rc == CUDA_ERROR_NOT_PERMITTED)) {
            fabric_enabled = false;
            fabric_error = "HOST_NUMA fabric allocation denied or unsupported";
            p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
            engram_route_cuda(
                deep_ep::lazy_cuMemGetAllocationGranularity(&granularity, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                "host fallback granularity");
            bytes = engram_route_aligned(requested, granularity);
            rc = deep_ep::lazy_cuMemCreate(&handle, bytes, &p, 0);
        }
        engram_route_cuda(rc, "allocate host memory");
        try {
            engram_route_cuda(deep_ep::lazy_cuMemAddressReserve(&ptr, bytes, granularity, 0, 0), "reserve host memory");
            engram_route_cuda(deep_ep::lazy_cuMemMap(ptr, bytes, 0, handle, 0), "map host memory");
            CUmemAccessDesc a[2]{};
            a[0].location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
            a[1].location = p.location;
            a[0].flags = a[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            engram_route_cuda(deep_ep::lazy_cuMemSetAccess(ptr, bytes, a, 2), "host access");
            engram_route_cuda(
                deep_ep::lazy_cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
                "export host memory");
            std::memset(reinterpret_cast<void *>(ptr), 0, bytes);
        } catch (...) {
            if (fd >= 0) {
                ::close(fd);
                fd = -1;
            }
            if (ptr) {
                deep_ep::lazy_cuMemUnmap(ptr, bytes);
                deep_ep::lazy_cuMemAddressFree(ptr, bytes);
                ptr = 0;
            }
            if (handle) {
                deep_ep::lazy_cuMemRelease(handle);
                handle = 0;
            }
            throw;
        }
    }
    std::vector<unsigned char> export_fabric() {
        if (!fabric_enabled)
            return {};
        CUmemFabricHandle h{};
        auto rc = deep_ep::lazy_cuMemExportToShareableHandle(&h, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
        if (rc != CUDA_SUCCESS) {
            fabric_enabled = false;
            fabric_error = "HOST_NUMA fabric export failed: " + std::to_string(rc);
            return {};
        }
        return {h.data, h.data + sizeof(h.data)};
    }

    ~EngramHostMemory() {
        if (fd >= 0)
            ::close(fd);
        if (ptr) {
            deep_ep::lazy_cuMemUnmap(ptr, bytes);
            deep_ep::lazy_cuMemAddressFree(ptr, bytes);
        }
        if (handle)
            deep_ep::lazy_cuMemRelease(handle);
    }
};

// 跨主机EGM只在requester GPU页表中建立映射，不添加本机CPU NUMA权限。
// 导入handle前，Python bootstrap已核对cluster UUID、clique及双方导出能力。
struct EngramFabricAlias {
    CUdeviceptr ptr = 0;
    CUmemGenericAllocationHandle handle = 0;
    size_t bytes = 0;
    EngramFabricAlias(const std::string &blob, size_t n, int device) : bytes(n) {
        engram_route_check(blob.size() == sizeof(CUmemFabricHandle), "fabric handle ABI");
        CUmemFabricHandle h{};
        std::memcpy(&h, blob.data(), sizeof(h));
        try {
            engram_route_cuda(deep_ep::lazy_cuMemImportFromShareableHandle(&handle, &h, CU_MEM_HANDLE_TYPE_FABRIC),
                              "import CPU fabric allocation");
            engram_route_cuda(deep_ep::lazy_cuMemAddressReserve(&ptr, bytes, 0, 0, 0), "reserve CPU fabric alias");
            engram_route_cuda(deep_ep::lazy_cuMemMap(ptr, bytes, 0, handle, 0), "map CPU fabric alias");
            CUmemAccessDesc a{};
            a.location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
            a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            engram_route_cuda(deep_ep::lazy_cuMemSetAccess(ptr, bytes, &a, 1), "CPU fabric GPU access");
        } catch (...) {
            if (ptr) {
                deep_ep::lazy_cuMemUnmap(ptr, bytes);
                deep_ep::lazy_cuMemAddressFree(ptr, bytes);
                ptr = 0;
            }
            if (handle) {
                deep_ep::lazy_cuMemRelease(handle);
                handle = 0;
            }
            throw;
        }
    }
    ~EngramFabricAlias() {
        if (ptr) {
            deep_ep::lazy_cuMemUnmap(ptr, bytes);
            deep_ep::lazy_cuMemAddressFree(ptr, bytes);
        }
        if (handle)
            deep_ep::lazy_cuMemRelease(handle);
    }
};
