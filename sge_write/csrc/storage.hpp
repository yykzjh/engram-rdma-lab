#pragma once
#include <cuda.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>

inline void ck(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
inline void cu(CUresult rc, const char *what) {
    if (rc != CUDA_SUCCESS) {
        const char *s = nullptr;
        cuGetErrorString(rc, &s);
        throw std::runtime_error(std::string(what) + ": " + (s ? s : "CUDA error"));
    }
}
inline size_t aligned(size_t n, size_t a = 4096) {
    return (n + a - 1) / a * a;
}

// 独立CPU NUMA词表/邮箱分配。导出的fd在本节点所有进程完成import前保持有效。
struct HostMemory {
    CUdeviceptr ptr = 0;
    CUmemGenericAllocationHandle handle = 0;
    size_t bytes = 0;
    int fd = -1, numa = -1;
    bool fabric_enabled = false;
    std::string fabric_error = "disabled";
    HostMemory(size_t requested, int device, bool use_fabric = false) {
        cu(cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device), "host NUMA");

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
            cu(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device),
               "fabric attribute");
            cu(cuDeviceGetAttribute(&host_ipc, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, device),
               "host fabric attribute");
#endif
            fabric_enabled = supported && host_ipc;
            fabric_error = fabric_enabled ? "" : "HOST_NUMA fabric IPC unavailable";
            if (fabric_enabled)
                p.requestedHandleTypes =
                    static_cast<CUmemAllocationHandleType>(p.requestedHandleTypes | CU_MEM_HANDLE_TYPE_FABRIC);
        }
        size_t granularity;
        cu(cuMemGetAllocationGranularity(&granularity, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "host granularity");
        bytes = aligned(requested, granularity);
        auto rc = cuMemCreate(&handle, bytes, &p, 0);
        if (fabric_enabled && (rc == CUDA_ERROR_NOT_SUPPORTED || rc == CUDA_ERROR_NOT_PERMITTED)) {
            fabric_enabled = false;
            fabric_error = "HOST_NUMA fabric allocation denied or unsupported";
            p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
            cu(cuMemGetAllocationGranularity(&granularity, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
               "host fallback granularity");
            bytes = aligned(requested, granularity);
            rc = cuMemCreate(&handle, bytes, &p, 0);
        }
        cu(rc, "allocate host memory");
        try {
            cu(cuMemAddressReserve(&ptr, bytes, granularity, 0, 0), "reserve host memory");
            cu(cuMemMap(ptr, bytes, 0, handle, 0), "map host memory");
            CUmemAccessDesc a[2]{};
            a[0].location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
            a[1].location = p.location;
            a[0].flags = a[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cu(cuMemSetAccess(ptr, bytes, a, 2), "host access");
            cu(cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
               "export host memory");
            std::memset(reinterpret_cast<void *>(ptr), 0, bytes);
        } catch (...) {
            if (fd >= 0) {
                ::close(fd);
                fd = -1;
            }
            if (ptr) {
                cuMemUnmap(ptr, bytes);
                cuMemAddressFree(ptr, bytes);
                ptr = 0;
            }
            if (handle) {
                cuMemRelease(handle);
                handle = 0;
            }
            throw;
        }
    }
    std::vector<unsigned char> export_fabric() {
        if (!fabric_enabled)
            return {};
        CUmemFabricHandle h{};
        auto rc = cuMemExportToShareableHandle(&h, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
        if (rc != CUDA_SUCCESS) {
            fabric_enabled = false;
            fabric_error = "HOST_NUMA fabric export failed: " + std::to_string(rc);
            return {};
        }
        return {h.data, h.data + sizeof(h.data)};
    }

    ~HostMemory() {
        if (fd >= 0)
            ::close(fd);
        if (ptr) {
            cuMemUnmap(ptr, bytes);
            cuMemAddressFree(ptr, bytes);
        }
        if (handle)
            cuMemRelease(handle);
    }
};

// 跨主机EGM只在requester GPU页表中建立映射，不添加本机CPU NUMA权限。
// 导入handle前，Python bootstrap已核对cluster UUID、clique及双方导出能力。
struct FabricAlias {
    CUdeviceptr ptr = 0;
    CUmemGenericAllocationHandle handle = 0;
    size_t bytes = 0;
    FabricAlias(const std::string &blob, size_t n, int device) : bytes(n) {
        ck(blob.size() == sizeof(CUmemFabricHandle), "fabric handle ABI");
        CUmemFabricHandle h{};
        std::memcpy(&h, blob.data(), sizeof(h));
        try {
            cu(cuMemImportFromShareableHandle(&handle, &h, CU_MEM_HANDLE_TYPE_FABRIC), "import CPU fabric allocation");
            cu(cuMemAddressReserve(&ptr, bytes, 0, 0, 0), "reserve CPU fabric alias");
            cu(cuMemMap(ptr, bytes, 0, handle, 0), "map CPU fabric alias");
            CUmemAccessDesc a{};
            a.location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
            a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cu(cuMemSetAccess(ptr, bytes, &a, 1), "CPU fabric GPU access");
        } catch (...) {
            if (ptr) {
                cuMemUnmap(ptr, bytes);
                cuMemAddressFree(ptr, bytes);
                ptr = 0;
            }
            if (handle) {
                cuMemRelease(handle);
                handle = 0;
            }
            throw;
        }
    }
    ~FabricAlias() {
        if (ptr) {
            cuMemUnmap(ptr, bytes);
            cuMemAddressFree(ptr, bytes);
        }
        if (handle)
            cuMemRelease(handle);
    }
};

struct HostAlias {
    CUdeviceptr ptr = 0;
    CUmemGenericAllocationHandle handle = 0;
    size_t bytes = 0;
    HostAlias(int pid, int fd, size_t n, int device, int numa) : bytes(n) {
        int pidfd = syscall(SYS_pidfd_open, pid, 0);
        ck(pidfd >= 0, "open exporter pidfd");
        int duplicate = syscall(SYS_pidfd_getfd, pidfd, fd, 0);
        ::close(pidfd);
        ck(duplicate >= 0, "duplicate host fd");

        CUresult status = cuMemImportFromShareableHandle(&handle, reinterpret_cast<void *>(intptr_t(duplicate)),
                                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        ::close(duplicate);
        cu(status, "import host allocation");
        try {
            cu(cuMemAddressReserve(&ptr, bytes, 0, 0, 0), "reserve host alias");
            cu(cuMemMap(ptr, bytes, 0, handle, 0), "map host alias");
            CUmemAccessDesc a[2]{};
            a[0].location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
            a[1].location = {CU_MEM_LOCATION_TYPE_HOST_NUMA, numa};
            a[0].flags = a[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cu(cuMemSetAccess(ptr, bytes, a, 2), "alias access");
        } catch (...) {
            if (ptr) {
                cuMemUnmap(ptr, bytes);
                cuMemAddressFree(ptr, bytes);
                ptr = 0;
            }
            if (handle) {
                cuMemRelease(handle);
                handle = 0;
            }
            throw;
        }
    }
    ~HostAlias() {
        if (ptr) {
            cuMemUnmap(ptr, bytes);
            cuMemAddressFree(ptr, bytes);
        }
        if (handle)
            cuMemRelease(handle);
    }
};
