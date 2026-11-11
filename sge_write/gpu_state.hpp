#pragma once
#include "input.hpp"
struct Gpu {
    CUdevice device = -1; CUcontext ctx = nullptr;
    CUdeviceptr allocation = 0, packed = 0, output = 0, positions = 0;
    CUdeviceptr input = 0, done_flags = 0;
    CUfunction wait_done = nullptr;
    CUdeviceptr sf_shard = 0, sf_output = 0;
    CUdeviceptr sf_peers[4]{};
    CUfunction gather_sf = nullptr;
    CUstream sf_stream = nullptr;
    CUevent sf_start = nullptr, sf_end = nullptr;
    uint32_t* host_input = nullptr;
    CUmodule module = nullptr; CUfunction scatter = nullptr;
    size_t bytes = 0;
    size_t allocation_bytes = 0;
    int rdma_write_ordering = 0;
    bool adaptive_grid = false;
    bool profile_routes = true;
    void (*allocation_release)(CUdeviceptr)=nullptr;
    ~Gpu() {
        if (module) cuModuleUnload(module);
        if (positions) cuMemFree(positions);
        if (input) cuMemFree(input);
        if (sf_stream) cuStreamDestroy(sf_stream);
        if (sf_start) cuEventDestroy(sf_start);
        if (sf_end) cuEventDestroy(sf_end);
        if (sf_output) cuMemFree(sf_output);
        if (sf_shard) cuMemFree(sf_shard);
        if (host_input) cuMemFreeHost(host_input);
        if (allocation) {if(allocation_release)allocation_release(allocation);else cuMemFree(allocation);}
        if (ctx) cuDevicePrimaryCtxRelease(device);
    }
    void init(const Options& o,void (*allocate)(CUdeviceptr*,size_t)=nullptr,void (*release)(CUdeviceptr)=nullptr) {
        cuda_check(cuInit(0), "cuInit"); cuda_check(cuDeviceGet(&device, o.gpu), "CUDA device");
        cuda_check(cuDevicePrimaryCtxRetain(&ctx, device), "CUDA context");
        cuda_check(cuCtxSetCurrent(ctx), "set CUDA context");
        cuda_check(cuDeviceGetAttribute(&rdma_write_ordering, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                                       device), "query GPUDirect write ordering");
        std::fprintf(stderr, "GPU_RDMA_WRITE_ORDERING=%d\n", rdma_write_ordering);
        bytes = frame_bytes(o);
        adaptive_grid = o.adaptive_grid;
        profile_routes = o.profile_routes;
        allocation_bytes = false ? bytes + 2 * GUARD : 2 * bytes + 3 * GUARD;
        if (o.done_gpu) allocation_bytes += 2 * GUARD;
        allocation_release=release;
        if(allocate)allocate(&allocation,allocation_bytes);
        else cuda_check(cuMemAlloc(&allocation, allocation_bytes), "GPU output allocation");
        packed = allocation + GUARD;
        output = false ? packed : packed + bytes + GUARD;
        if (o.done_gpu) {
            check(rdma_write_ordering >= CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER, "GPU DONE requires native owner ordering");
            done_flags = allocation + allocation_bytes - 2 * GUARD;
        }
        unsigned one = 1;
        // Raw-verbs legacy allocations need SYNC_MEMOPS. NCCL/VMM buffers do
        // not support that attribute and use GIN's explicit completion ordering.
        if(!allocate)cuda_check(cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, allocation), "SYNC_MEMOPS");
        if (true)
            cuda_check(cuMemAlloc(&positions, bytes / ROW * sizeof(uint32_t)), "GPU position map");
        cuda_check(cuMemAlloc(&input, bytes / ROW * sizeof(uint32_t)), "GPU raw indices");
        cuda_check(cuMemHostAlloc(reinterpret_cast<void**>(&host_input), bytes / ROW * sizeof(uint32_t),
                                 CU_MEMHOSTALLOC_PORTABLE), "host index transfer workspace");
        // Exactly the same scale-table partition as DeepEP: 8 owners / 4 local GPU shards.
        cuda_check(cuMemAlloc(&sf_shard, uint64_t(o.table_rows) * 2 * 8), "GPU scale shard");
        cuda_check(cuMemAlloc(&sf_output, bytes / ROW * 8), "GPU scale output");
        std::vector<uint64_t> scales(size_t(o.table_rows) * 2);
        size_t sf_at = 0;
        for (size_t layer = 0; layer < o.entries.size(); ++layer)
            for (uint32_t row = 0; row < uint32_t(2 * o.entries[layer]); ++row) {
                uint32_t global = (o.rank % 4) * 2 * o.entries[layer] + row;
                uint32_t a = 0x3f800000U | ((global + layer * 17) & 0x7fffffU);
                uint32_t b = 0x40000000U | ((global * 3 + layer * 29) & 0x7fffffU);
                scales[sf_at++] = uint64_t(a) | (uint64_t(b) << 32);
            }
        cuda_check(cuMemcpyHtoD(sf_shard, scales.data(), scales.size() * 8), "initialize scale shard");
        cuda_check(cuStreamCreate(&sf_stream, CU_STREAM_NON_BLOCKING), "scale stream");
        cuda_check(cuEventCreate(&sf_start, 0), "scale start event");
        cuda_check(cuEventCreate(&sf_end, 0), "scale end event");
        const char* source = R"(
extern "C" __global__ void scatter_rows(const unsigned long long* input,
        unsigned long long* output, const unsigned* positions, unsigned long long words) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < words; i += (unsigned long long)gridDim.x * blockDim.x)
        output[(unsigned long long)positions[i / 32] * 32 + i % 32] = input[i];
}
extern "C" __global__ void wait_remote_done(const unsigned long long* flags, unsigned qps,
        unsigned long long epoch,const unsigned* totals,unsigned merge_rows) {
    unsigned index = threadIdx.x;
    if (index < 8 * qps) {
        unsigned owner = index / qps, qp = index % qps;
        unsigned active=totals?max(1u,min(qps,(totals[owner]+merge_rows-1)/merge_rows)):qps;
        if(qp<active){
        const unsigned long long* address = flags + owner * 16 + qp;
        unsigned long long value; unsigned long long begin = clock64();
        do {
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
            if (value != epoch) __nanosleep(128);
            if (clock64() - begin > 100000000000ULL) asm volatile("trap;");
        } while (value != epoch);
        }
    }
    __syncthreads();
}
extern "C" __global__ void gather_scales(const unsigned* indices, unsigned* output,
        const unsigned long long* s0, const unsigned long long* s1,
        const unsigned long long* s2, const unsigned long long* s3, unsigned tokens) {
    unsigned count = tokens * 24;
    const unsigned long long* shards[] = {s0, s1, s2, s3};
    for (unsigned at = blockIdx.x * blockDim.x + threadIdx.x; at < count * 2;
         at += gridDim.x * blockDim.x) {
        unsigned layer = at / count, x = indices[at], n = layer ? 524309 : 524288;
        unsigned shard = x / (2 * n), row = (layer ? 2 * 524288 : 0) + x % (2 * n);
        unsigned long long value = __ldg(shards[shard] + row);
        unsigned token = (at % count) / 24, slot = at % 24;
        output[layer * tokens * 48 + token + slot * 2 * tokens] = (unsigned)value;
        output[layer * tokens * 48 + token + (slot * 2 + 1) * tokens] = (unsigned)(value >> 32);
    }
})";
        nvrtcProgram prog;
        check(nvrtcCreateProgram(&prog, source, "scatter.cu", 0, nullptr, nullptr) == NVRTC_SUCCESS, "NVRTC create");
        int major, minor;
        cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "compute major");
        cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "compute minor");
        std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
        const char* opts[] = {arch.c_str(), "--std=c++11"};
        nvrtcResult rc = nvrtcCompileProgram(prog, 2, opts);
        if (rc != NVRTC_SUCCESS) {
            size_t n; nvrtcGetProgramLogSize(prog, &n); std::vector<char> log(n);
            nvrtcGetProgramLog(prog, log.data()); nvrtcDestroyProgram(&prog);
            throw std::runtime_error(std::string("NVRTC: ") + log.data());
        }
        size_t n; check(nvrtcGetPTXSize(prog, &n) == NVRTC_SUCCESS, "NVRTC PTX size");
        std::vector<char> ptx(n); nvrtcGetPTX(prog, ptx.data()); nvrtcDestroyProgram(&prog);
        cuda_check(cuModuleLoadData(&module, ptx.data()), "load scatter kernel");
        cuda_check(cuModuleGetFunction(&scatter, module, "scatter_rows"), "find scatter kernel");
        cuda_check(cuModuleGetFunction(&gather_sf, module, "gather_scales"), "find scale kernel");
        cuda_check(cuModuleGetFunction(&wait_done, module, "wait_remote_done"), "find GPU DONE kernel");
    }
    void clear_for_validation() {
        cuda_check(cuMemsetD8(allocation, 0xa5, allocation_bytes), "reset guards");
        if (packed != output) cuda_check(cuMemsetD8(packed, 0, bytes), "reset staging");
        cuda_check(cuMemsetD8(output, 0, bytes), "reset output");
        cuda_check(cuMemsetD8(sf_output, 0, bytes / ROW * 8), "reset scale output");
        cuda_check(cuCtxSynchronize(), "reset synchronize");
    }
    double download_indices() {
        auto t = Clock::now();
        cuda_check(cuMemcpyDtoH(host_input, input, bytes / ROW * sizeof(uint32_t)), "download original GPU indices");
        return us(t) / 1000;
    }
    void start_scales(const Options& o) {
        unsigned tokens = o.groups;
        void* args[] = {&input, &sf_output, &sf_peers[0], &sf_peers[1], &sf_peers[2], &sf_peers[3], &tokens};
        if(profile_routes)cuda_check(cuEventRecord(sf_start, sf_stream), "record scale start");
        unsigned grid=adaptive_grid?std::min(512U,(tokens*48+255)/256):512;
        cuda_check(cuLaunchKernel(gather_sf, grid, 1, 1, 256, 1, 1, 0, sf_stream, args, nullptr), "gather scales");
        if(profile_routes)cuda_check(cuEventRecord(sf_end, sf_stream), "record scale end");
    }
    void close_scale_peers(const Options& o) {
        for (int i = 0; i < 4; ++i) if (i != o.rank % 4 && sf_peers[i]) {
            cuda_check(cuIpcCloseMemHandle(sf_peers[i]), "close scale IPC"); sf_peers[i] = 0;
        }
    }
    double reorder(bool scatter_needed = true, unsigned qps = 0, uint64_t epoch = 0,CUdeviceptr totals=0,unsigned merge_rows=30) {
        auto t = Clock::now();
        if (qps) {
            void* wait_args[] = {&done_flags, &qps, &epoch,&totals,&merge_rows};
            cuda_check(cuLaunchKernel(wait_done, 1,1,1, 256,1,1, 0,nullptr,wait_args,nullptr), "GPU wait all QP epochs");
        } uint64_t words = bytes / 8;
        if (rdma_write_ordering < CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER)
            cuda_check(cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                                CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER), "flush incoming GPU writes");
        void* args[] = {&packed, &output, &positions, &words};
        if (packed != output && scatter_needed)
            cuda_check(cuLaunchKernel(scatter, adaptive_grid?unsigned(std::min(uint64_t(4096),(words+255)/256)):4096, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr), "scatter rows");
        cuda_check(cuCtxSynchronize(), "scatter synchronize"); return us(t) / 1000;
    }
    void validate(const Plan& p) {
        std::vector<uint64_t> data(bytes / 8);
        cuda_check(cuMemcpyDtoH(data.data(), output, bytes), "validation GPU copy");
        for (size_t i = 0; i < p.expected.size(); ++i)
            for (unsigned w = 0; w < ROW / 8; ++w)
                if (data[i * (ROW / 8) + w] != pattern(p.expected[i], w))
                    throw std::runtime_error("payload mismatch: output row " + std::to_string(i) + " word " + std::to_string(w));
        std::vector<uint8_t> guard(GUARD);
        std::vector<CUdeviceptr> guards{allocation, packed + bytes, output + bytes};
        if (done_flags) guards.push_back(done_flags + GUARD);
        for (auto ptr : guards) {
            cuda_check(cuMemcpyDtoH(guard.data(), ptr, GUARD), "guard copy");
            check(std::all_of(guard.begin(), guard.end(), [](uint8_t x) { return x == 0xa5; }), "GPU guard");
        }
    }
    void validate_scales(const Options& o) {
        std::vector<uint32_t> values(bytes / ROW * 2);
        cuda_check(cuMemcpyDtoH(values.data(), sf_output, values.size() * 4), "copy scale validation");
        size_t count = size_t(o.groups) * 24;
        for (size_t at = 0; at < bytes / ROW; ++at) {
            uint32_t layer = at / count, x = host_input[at], token = (at % count) / 24, slot = at % 24;
            uint32_t a = 0x3f800000U | ((x + layer * 17) & 0x7fffffU);
            uint32_t b = 0x40000000U | ((x * 3 + layer * 29) & 0x7fffffU);
            check(values[layer * count * 2 + token + slot * 2 * o.groups] == a &&
                  values[layer * count * 2 + token + (slot * 2 + 1) * o.groups] == b, "FP8 scale output mismatch");
        }
    }
};
