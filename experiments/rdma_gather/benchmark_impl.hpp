#pragma once
// Eight logical ranks, four GPUs/CPU vocabulary shards on each of two nodes.
// Every fetch starts from original GPU indices and rebuilds current-input routing.
// WRITE supports per-token direct, per-token staged, and cross-token merged layouts.
#include "rdma_common.hpp"

#include "planner.hpp"
#include "planner_profile.hpp"

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
        allocation_bytes = o.layout == "direct" || o.target_umr == "strict" ? bytes + 2 * GUARD : 2 * bytes + 3 * GUARD;
        if (o.done_gpu) allocation_bytes += 2 * GUARD;
        allocation_release=release;
        if(allocate)allocate(&allocation,allocation_bytes);
        else cuda_check(cuMemAlloc(&allocation, allocation_bytes), "GPU output allocation");
        packed = allocation + GUARD;
        output = o.layout == "direct" || o.target_umr == "strict" ? packed : packed + bytes + GUARD;
        if (o.done_gpu) {
            check(rdma_write_ordering >= CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER, "GPU DONE requires native owner ordering");
            done_flags = allocation + allocation_bytes - 2 * GUARD;
        }
        unsigned one = 1;
        // Raw-verbs legacy allocations need SYNC_MEMOPS. NCCL/VMM buffers do
        // not support that attribute and use GIN's explicit completion ordering.
        if(!allocate)cuda_check(cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, allocation), "SYNC_MEMOPS");
        if (o.layout != "direct" && o.target_umr != "strict")
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
    double upload_map(const Plan& p) {
        if (p.positions.empty()) return 0;
        auto t = Clock::now();
        cuda_check(cuMemcpyHtoD(positions, p.positions.data(), p.positions.size() * sizeof(uint32_t)), "upload positions");
        cuda_check(cuCtxSynchronize(), "prepare synchronize");
        return us(t) / 1000;
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
static void configure(Resources& r, const std::vector<Group>& groups, const std::vector<ibv_sge>& sges,
                      int n, Stats& stats) {
    auto t = Clock::now();
    auto* x = ibv_qp_to_qp_ex(r.qps[0]); auto* d = mlx5dv_qp_ex_from_ibv_qp_ex(x);
    ibv_wr_start(x); size_t offset = 0;
    for (int i = 0; i < n; ++i) {
        x->wr_id = i; x->wr_flags = IBV_SEND_INLINE | (i == n - 1 ? IBV_SEND_SIGNALED : 0);
        mlx5dv_mkey_conf_attr attr{}; mlx5dv_wr_mkey_configure(d, r.keys[i], 2, &attr);
        mlx5dv_wr_set_mkey_access_flags(d, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
        mlx5dv_wr_set_mkey_layout_list(d, groups[i].count, sges.data() + offset);
        offset += groups[i].count;
    }
    verbs_check(ibv_wr_complete(x), "post UMR batch"); stats.umr_post_us = us(t);
    r.poll(1); stats.umr_us = us(t);
}
// Only capacity and invariant metadata may be reused. Source addresses, destination
// offsets and current-input SGE counts are filled again inside every timed fetch.
struct TransferWorkspace {
    std::vector<ibv_send_wr> wr;
    std::vector<ibv_sge> dst;
    explicit TransferWorkspace(int n) : wr(n), dst(n) {}
};
static int transfer(Resources& r, const Options& o, bool read, const Group* groups, int n,
                     uint64_t gpu, uint32_t rkey, const std::vector<ibv_sge>& source,
                     const std::vector<uint32_t>& keys, Stats& stats, TransferWorkspace& workspace, bool defer_completion=false) {
    auto t = Clock::now();
    if (!read && o.engine == "ex") {
        int completions = 0; size_t offset = 0;
        for (int q = 0; q < o.qps && q < n; ++q) ibv_wr_start(ibv_qp_to_qp_ex(r.qps[q]));
        for (int i = 0; i < n; ++i) {
            auto* x = ibv_qp_to_qp_ex(r.qps[i % o.qps]);
            x->wr_id = i; x->wr_flags = i + o.qps >= n ? IBV_SEND_SIGNALED : 0;
            ibv_wr_rdma_write(x, rkey, gpu + uint64_t(groups[i].dst_row) * ROW);
            ibv_wr_set_sge_list(x, groups[i].count, source.data() + offset);
            offset += groups[i].count;
        }
        for (int q = 0; q < o.qps && q < n; ++q) {
            verbs_check(ibv_wr_complete(ibv_qp_to_qp_ex(r.qps[q])), "complete extended WRITE"); ++completions;
        }
        // Extended verbs directly constructs provider WQEs: its build/post time
        // cannot be separated into the classic user-WR and provider-post phases.
        stats.verbs_post_us = stats.data_post_us = us(t);
        auto poll_start = Clock::now(); if(!defer_completion)r.poll(completions);
        stats.cq_wait_us = defer_completion?0:us(poll_start); stats.data_us = us(t); return completions;
    }
    std::vector<ibv_send_wr> fresh_wr;
    std::vector<ibv_sge> fresh_dst;
    if (o.engine == "baseline") { fresh_wr.resize(n); if (read) fresh_dst.resize(n); }
    auto& wr = o.engine == "baseline" ? fresh_wr : workspace.wr;
    auto& dst = o.engine == "baseline" ? fresh_dst : workspace.dst;
    std::vector<int> head(o.qps, -1), tail(o.qps, -1); size_t offset = 0;
    for (int i = 0; i < n; ++i) {
        auto& w = wr[i]; int q = i % o.qps; w.wr_id = i; w.next = nullptr; w.send_flags = 0;
        w.opcode = read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
        uint64_t addr = gpu + uint64_t(groups[i].dst_row) * ROW;
        if (read) {
            dst[i] = {addr, uint32_t(groups[i].count * ROW), r.mr->lkey};
            w.sg_list = &dst[i]; w.num_sge = 1;
            w.wr.rdma.remote_addr = 0; w.wr.rdma.rkey = keys[i];
        } else {
            w.sg_list = const_cast<ibv_sge*>(source.data() + offset); w.num_sge = groups[i].count;
            w.wr.rdma.remote_addr = addr; w.wr.rdma.rkey = rkey;
        }
        offset += groups[i].count;
        if (tail[q] >= 0) wr[tail[q]].next = &w; else head[q] = i;
        tail[q] = i;
    }
    stats.wr_build_us = us(t); auto post_start = Clock::now();
    int completions = 0;
    for (int q = 0; q < o.qps; ++q) if (head[q] >= 0) {
        wr[tail[q]].send_flags = IBV_SEND_SIGNALED; ibv_send_wr* bad = nullptr;
        verbs_check(ibv_post_send(r.qps[q], &wr[head[q]], &bad), "post data batch"); ++completions;
    }
    stats.verbs_post_us = us(post_start); stats.data_post_us = us(t);
    auto poll_start = Clock::now(); if(!defer_completion)r.poll(completions);
    stats.cq_wait_us = defer_completion?0:us(poll_start); stats.data_us = us(t);
    return completions;
}
static void add(Stats& a, const Stats& b) {
    a.owner_receive_us += b.owner_receive_us; a.layout_us += b.layout_us;
    a.umr_us += b.umr_us; a.umr_post_us += b.umr_post_us;
    a.data_us += b.data_us; a.data_post_us += b.data_post_us;
    a.wr_build_us += b.wr_build_us; a.verbs_post_us += b.verbs_post_us;
    a.cq_wait_us += b.cq_wait_us;
    a.owner_ack_us += b.owner_ack_us; a.owner_notify_us += b.owner_notify_us;
}
struct Channel {
    Options o; Resources r; Socket sock; Hello peer{};
    std::unique_ptr<TransferWorkspace> workspace;
    uint64_t gpu_target = 0;
    uint32_t registered_target_rkey = 0;
    void init(const Options& base, int other, void* table, const Gpu* gpu) {
        o = base;
        int owner = o.owner ? o.rank : other, requester = o.owner ? other : o.rank;
        o.port += owner * 8 + requester;
        o.host = node_ip(o, owner);
        open_hca(r, o); create_qps(r, o);
        workspace = std::make_unique<TransferWorkspace>(o.pool);
        if (o.owner) {
            r.cpu = table; r.borrowed_cpu = true;
            r.mr = ibv_reg_mr(r.pd, table, size_t(o.table_rows) * ROW,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
            check(r.mr != nullptr, "register owner CPU shard");
            auto t = Clock::now();
            for (int i = 0; o.methods != "write" && i < o.pool; ++i) {
                mlx5dv_mkey_init_attr a{}; a.pd = r.pd;
                a.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
                a.max_entries = o.layout == "merged" ? o.merge_rows : o.rows;
                auto* k = mlx5dv_create_mkey(&a); check(k != nullptr, "create pooled MKey"); r.keys.push_back(k);
            }
            std::fprintf(stderr, "POOL owner=%d requester=%d count=%zu create_ms=%.3f\n", owner, requester, r.keys.size(), us(t)/1000);
        } else {
            cuda_check(cuCtxSetCurrent(gpu->ctx), "thread CUDA context");
            r.mr = ibv_reg_mr(r.pd, reinterpret_cast<void*>(gpu->allocation),
                             gpu->allocation_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            check(r.mr != nullptr, "register requester GPU memory");
        }
        connect(sock, o);
        Hello h{}; h.pool = o.pool; h.qps = o.qps; h.rows = o.rows;
        h.groups = o.groups; h.table_rows = o.table_rows;
        h.reserved = o.layout == "direct" ? 1 : o.layout == "merged" ? 2 : 0;
        h.pad = o.merge_rows;
        h.engine = o.engine == "baseline" ? 0 : o.engine == "reuse" ? 1 : 2; h.pipeline = o.pipeline; h.protocol = o.protocol == "frame"; h.done_gpu = o.done_gpu;
        h.compact_request = o.compact_request; h.gpu_request = o.gpu_request;
        h.target_umr = o.target_umr != "none";
        h.owner_windows = o.owner_windows;
        h.compact_done=o.compact_done;h.hybrid_local=o.hybrid_local;
        if (!o.owner) { gpu_target = o.layout == "direct" ? gpu->output : gpu->packed;
            h.dst_addr = gpu_target; h.dst_rkey = r.mr->rkey;
            if (o.done_gpu) { h.done_addr = gpu->done_flags + owner * 16 * sizeof(uint64_t); h.done_rkey = r.mr->rkey; }
        }
        sock.send(h); sock.recv(peer);
        check(peer.magic == MAGIC && peer.version == h.version && peer.bytes == sizeof(Hello) &&
              peer.pool == h.pool && peer.qps == h.qps && peer.rows == h.rows &&
              peer.groups == h.groups && peer.table_rows == h.table_rows &&
              peer.reserved == h.reserved && peer.pad == h.pad && peer.engine == h.engine && peer.pipeline == h.pipeline && peer.protocol == h.protocol && peer.done_gpu == h.done_gpu &&
              peer.compact_request == h.compact_request && peer.gpu_request == h.gpu_request && peer.target_umr == h.target_umr && peer.owner_windows==h.owner_windows && peer.compact_done==h.compact_done && peer.hybrid_local==h.hybrid_local, "hello mismatch");
        activate(r, o, sock);
    }
};
#include "target_umr.hpp"

struct WorkHeader { uint32_t groups = 0, rows = 0; };
// One logical frame request, with bounded internal RDMA batches. TCP applies
// backpressure while the owner drains QPs; no per-window DONE/ACK is exchanged.
struct FrameHeader { uint64_t rows = 0, fetch_id = 0; uint32_t base_row = 0, indirect = 0; };
static void receive_target_root(Channel& c) {
    if(c.o.target_umr=="none")return;
    c.sock.recv(c.registered_target_rkey);
    check(c.registered_target_rkey!=0,"registered target root rkey");
    // Its logical address starts at zero. Only mappings change at fetch time.
}
static void publish_gpu_done(Channel& c, uint64_t epoch,unsigned active_qps=0) {
    // Each data QP publishes its own tail flag after its data. GPU waits ALL
    // flags; ordering of unrelated QPs is never inferred from one flag.
    unsigned count=active_qps?active_qps:unsigned(c.o.qps);
    for (unsigned q = 0; q < count; ++q) {
        ibv_sge sg{reinterpret_cast<uintptr_t>(&epoch), 8, 0}; ibv_send_wr wr{}, *bad = nullptr;
        wr.wr_id = UINT64_MAX; wr.opcode = IBV_WR_RDMA_WRITE;
        wr.sg_list = &sg; wr.num_sge = 1; wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = c.peer.done_addr + q * 8; wr.wr.rdma.rkey = c.peer.done_rkey;
        verbs_check(ibv_post_send(c.r.qps[q], &wr, &bad), "post GPU DONE epoch");
    }
    c.r.poll(count);
}
static void serve_frame(Channel& c) {
    auto& r = c.r; const auto& o = c.o;
    const size_t max_rows = frame_bytes(o) / ROW;
    const size_t max_groups = o.layout == "merged" ? (max_rows + o.merge_rows - 1) / o.merge_rows + 8 : max_rows;
    const size_t group_rows = o.layout == "merged" ? o.merge_rows : o.rows;
    std::vector<Group> groups(max_groups);
    std::vector<uint32_t> rows(size_t(o.pool) * group_rows), keys;
    std::vector<ibv_sge> sges(rows.size(), {0, uint32_t(ROW), r.mr->lkey});
    while (true) {
        Command cmd{}; c.sock.recv(cmd); if (!cmd.mode) break;
        auto receive_start = Clock::now(); FrameHeader h{}; c.sock.recv(h);
        check(cmd.mode == 2 && cmd.groups <= max_groups && h.rows <= max_rows, "frame header bounds");
        if (!o.compact_request) c.sock.recv(groups.data(), size_t(cmd.groups) * sizeof(Group));
        else {
            check(cmd.groups == (h.rows + o.merge_rows - 1) / o.merge_rows, "compact group count");
            for (size_t i = 0; i < cmd.groups; ++i) {
                uint32_t at = i * o.merge_rows;
                groups[i] = {uint32_t(std::min(uint64_t(o.merge_rows), h.rows - at)), h.base_row + at};
            }
        }
        Response response{}; response.groups = cmd.groups; response.fetch_id = h.fetch_id;
        response.stats.owner_receive_us = us(receive_start); size_t received_rows = 0;
        for (size_t at = 0; at < cmd.groups; at += o.pool) {
            size_t n = std::min(size_t(o.pool), size_t(cmd.groups) - at), nr = 0;
            auto t = Clock::now();
            for (size_t i = 0; i < n; ++i) {
                auto& g = groups[at + i];
                check(g.count > 0 && g.count <= group_rows && uint64_t(g.dst_row) + g.count <= max_rows, "frame group bounds");
                if (h.indirect) {
                    check(c.registered_target_rkey!=0,"indirect request without registered root");
                    check(g.dst_row >= h.base_row && uint64_t(g.dst_row - h.base_row) + g.count <= h.rows, "indirect offset bounds");
                    g.dst_row -= h.base_row;
                }
                nr += g.count;
            }
            check(nr <= rows.size() && received_rows + nr <= h.rows, "frame rows bounds");
            c.sock.recv(rows.data(), nr * sizeof(uint32_t));
            response.stats.owner_receive_us += us(t); t = Clock::now();
            for (size_t i = 0; i < nr; ++i) {
                check(rows[i] < uint32_t(o.table_rows), "frame shard index");
                sges[i].addr = reinterpret_cast<uintptr_t>(r.cpu) + size_t(rows[i]) * ROW;
            }
            response.stats.layout_us += us(t); Stats batch{};
            transfer(r, o, false, groups.data() + at, n, h.indirect ? 0 : c.peer.dst_addr,
                     h.indirect ? c.registered_target_rkey : c.peer.dst_rkey, sges, keys, batch, *c.workspace);
            add(response.stats, batch); received_rows += nr;
        }
        check(received_rows == h.rows, "complete frame row count");
        // Telemetry goes over TCP AFTER the GPU-visible completion. The GPU
        // does not wait for this response; the requester reads it outside e2e.
        if (o.done_gpu) publish_gpu_done(c, h.fetch_id);
        c.sock.send(response);
    }
}
static void serve(Channel& c) {
    receive_target_root(c);
    if (c.o.protocol == "frame") { serve_frame(c); return; }
    auto& r = c.r; const auto& o = c.o;
    const uint32_t max_group_rows = o.layout == "merged" ? o.merge_rows : o.rows;
    std::vector<Group> groups(o.pool); std::vector<uint32_t> rows(size_t(o.pool) * max_group_rows), keys(o.pool);
    std::vector<ibv_sge> sges;
    if (o.engine == "baseline") sges.reserve(rows.size());
    else sges.resize(rows.size(), {0, uint32_t(ROW), r.mr->lkey});
    while (true) {
        Command cmd{}; c.sock.recv(cmd); if (cmd.mode == 0) break;
        check(cmd.mode == 1 || cmd.mode == 2, "mode");
        double previous_ack_us = 0, previous_notify_us = 0;
        for (uint32_t at = 0; at < cmd.groups;) {
            auto t = Clock::now(); WorkHeader h{}; c.sock.recv(h);
            check(h.groups > 0 && h.groups <= uint32_t(o.pool) && h.groups <= cmd.groups - at &&
                  h.rows <= h.groups * max_group_rows, "batch bounds");
            c.sock.recv(groups.data(), h.groups * sizeof(Group)); c.sock.recv(rows.data(), h.rows * sizeof(uint32_t));
            Response response{}; response.groups = h.groups; response.stats.owner_receive_us = us(t);
            // Return the previous window's post-response timings in this DONE.
            // Last-window ACK is not awaited by the requester, hence outside its
            // critical fetch boundary; no extra footer/round trip is introduced.
            response.stats.owner_ack_us = previous_ack_us;
            response.stats.owner_notify_us = previous_notify_us;
            t = Clock::now(); if (o.engine == "baseline") sges.resize(h.rows); size_t count = 0;
            for (uint32_t i = 0; i < h.groups; ++i) {
                check(groups[i].count > 0 && groups[i].count <= max_group_rows &&
                      uint64_t(groups[i].dst_row) + groups[i].count <= frame_bytes(o) / ROW, "group bounds");
                count += groups[i].count;
            }
            check(count == h.rows, "group row count");
            for (uint32_t i = 0; i < h.rows; ++i) {
                check(rows[i] < uint32_t(o.table_rows), "shard index");
                auto addr = reinterpret_cast<uintptr_t>(r.cpu) + size_t(rows[i]) * ROW;
                if (o.engine == "baseline") sges[i] = {addr, uint32_t(ROW), r.mr->lkey};
                else sges[i].addr = addr;
            }
            response.stats.layout_us = us(t);
            if (cmd.mode == 1) {
                configure(r, groups, sges, h.groups, response.stats);
                for (uint32_t i = 0; i < h.groups; ++i) keys[i] = r.keys[i]->rkey;
                auto notify_start = Clock::now();
                c.sock.send(response); c.sock.send(keys.data(), h.groups * sizeof(uint32_t));
                previous_notify_us = us(notify_start);
            } else {
                transfer(r, o, false, groups.data(), h.groups, c.peer.dst_addr, c.peer.dst_rkey, sges, keys, response.stats, *c.workspace);
                auto notify_start = Clock::now(); c.sock.send(response);
                previous_notify_us = us(notify_start);
            }
            if (cmd.mode == 1 || !o.pipeline) {
                auto ack_start = Clock::now(); uint32_t ack; c.sock.recv(ack);
                previous_ack_us = us(ack_start); check(ack == at + h.groups, "completion ACK");
            }
            // READ needs ACK before remapping MKeys. WRITE has already completed
            // all QPs before DONE; no requester ACK is needed for source reuse.
            at += h.groups;
        }
    }
}
struct PeerStats { Stats stats{}; double send_us = 0, wait_us = 0, ack_us = 0, elapsed_us = 0; };
static PeerStats fetch(Channel& c, const PeerPlan& plan, uint32_t mode, int round, const TargetMkey* target = nullptr) {
    PeerStats result; auto start = Clock::now(); auto t = Clock::now();
    Command cmd{mode, uint32_t(round), 0, uint32_t(plan.groups.size())}; c.sock.send(cmd);
    result.send_us += us(t);
    if (c.o.protocol == "frame") {
        check(mode == 2, "frame WRITE mode");
        FrameHeader h{plan.rows.size(), uint64_t(round) + 1};
        h.base_row = plan.groups.size() ? plan.groups[0].dst_row : 0;
        if (target) { h.indirect = 1; h.base_row = target->base_row; }
        t = Clock::now(); c.sock.send(h);
        if (!c.o.compact_request) c.sock.send(plan.groups.data(), plan.groups.size() * sizeof(Group));
        c.sock.send(plan.rows.data(), plan.rows.size() * sizeof(uint32_t));
        result.send_us += us(t);
        if (c.o.done_gpu) { result.elapsed_us = us(start); return result; }
        Response response{}; t = Clock::now(); c.sock.recv(response);
        check(response.status == 0 && response.groups == plan.groups.size() && response.fetch_id == h.fetch_id, "final frame DONE");
        result.wait_us += us(t); add(result.stats, response.stats);
        result.elapsed_us = us(start); return result;
    }
    if (mode == 2 && c.o.pipeline) {
        size_t sent = 0, received = 0, row_offset = 0; int in_flight = 0;
        while (received < plan.groups.size()) {
            while (sent < plan.groups.size() && in_flight < c.o.pipeline) {
                uint32_t n = std::min(size_t(c.o.pool), plan.groups.size() - sent), nr = 0;
                for (uint32_t i = 0; i < n; ++i) nr += plan.groups[sent + i].count;
                WorkHeader h{n, nr}; t = Clock::now(); c.sock.send(&h, sizeof(h), c.o.coalesce_control ? MSG_MORE : 0);
                c.sock.send(plan.groups.data() + sent, n * sizeof(Group), c.o.coalesce_control ? MSG_MORE : 0);
                c.sock.send(plan.rows.data() + row_offset, nr * sizeof(uint32_t));
                result.send_us += us(t); sent += n; row_offset += nr; ++in_flight;
            }
            Response response{}; t = Clock::now(); c.sock.recv(response);
            uint32_t n = std::min(size_t(c.o.pool), plan.groups.size() - received);
            check(response.groups == n && response.status == 0, "pipelined DONE");
            result.wait_us += us(t); add(result.stats, response.stats);
            received += n; --in_flight;
        }
        result.elapsed_us = us(start); return result;
    }
    std::vector<uint32_t> keys(c.o.pool); std::vector<ibv_sge> empty;
    size_t row_offset = 0;
    for (size_t at = 0; at < plan.groups.size(); at += c.o.pool) {
        uint32_t n = std::min(size_t(c.o.pool), plan.groups.size() - at), nr = 0;
        for (uint32_t i = 0; i < n; ++i) nr += plan.groups[at + i].count;
        WorkHeader header{n, nr}; t = Clock::now(); c.sock.send(&header, sizeof(header), c.o.coalesce_control ? MSG_MORE : 0);
        c.sock.send(plan.groups.data() + at, n * sizeof(Group), c.o.coalesce_control ? MSG_MORE : 0);
        c.sock.send(plan.rows.data() + row_offset, nr * sizeof(uint32_t));
        result.send_us += us(t);
        Response response{}; t = Clock::now(); c.sock.recv(response);
        check(response.groups == n && response.status == 0, "response");
        if (mode == 1) c.sock.recv(keys.data(), n * sizeof(uint32_t));
        result.wait_us += us(t); add(result.stats, response.stats);
        if (mode == 1) {
            Stats local{};
            transfer(c.r, c.o, true, plan.groups.data() + at, n,
                c.gpu_target, 0, empty, keys, local, *c.workspace);
            add(result.stats, local);
        }
        uint32_t ack = at + n; t = Clock::now();
        c.sock.send(&ack, sizeof(ack), c.o.coalesce_control && at + n < plan.groups.size() ? MSG_MORE : 0); result.ack_us += us(t);
        row_offset += nr;
    }
    result.elapsed_us = us(start); return result;
}
class Barrier {
    std::vector<std::unique_ptr<Socket>> sockets;
    bool leader;
public:
    template<class T> std::vector<T> allgather(const T& local) {
        std::vector<T> values(8);
        if (leader) {
            values[0] = local;
            for (auto& s : sockets) { T x{}; s->recv(x); check(x.rank > 0 && x.rank < 8, "allgather rank"); values[x.rank] = x; }
            for (auto& s : sockets) s->send(values.data(), values.size() * sizeof(T));
        } else { sockets[0]->send(local); sockets[0]->recv(values.data(), values.size() * sizeof(T)); }
        return values;
    }
    explicit Barrier(const Options& o) : leader(o.rank == 0) {
        if (leader) {
            Socket listener; listener.fd = socket(AF_INET, SOCK_STREAM, 0); check(listener.fd >= 0, "barrier socket");
            int one = 1; setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(o.port + 200);
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
            check(!bind(listener.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), "barrier bind");
            check(!listen(listener.fd, 8), "barrier listen");
            for (int i = 1; i < 8; ++i) {
                auto p = std::make_unique<Socket>(); p->fd = accept(listener.fd, nullptr, nullptr);
                check(p->fd >= 0, "barrier accept"); p->tune(); sockets.push_back(std::move(p));
            }
        } else {
            Options b = o; b.owner = false; b.port += 200; b.host = node_ip(o, 0);
            auto p = std::make_unique<Socket>(); connect(*p, b); sockets.push_back(std::move(p));
        }
    }
    void wait(uint32_t phase) {
        if (leader) {
            for (auto& s : sockets) { uint32_t x; s->recv(x); check(x == phase, "barrier phase"); }
            for (auto& s : sockets) s->send(phase);
        } else { sockets[0]->send(phase); uint32_t x; sockets[0]->recv(x); check(x == phase, "barrier phase"); }
    }
};
// Propagate failures from native worker threads instead of terminating the process silently.
template<class F> static void parallel(F f) {
    std::exception_ptr error; std::mutex mutex; std::vector<std::thread> threads;
    for (int peer = 0; peer < 8; ++peer) threads.emplace_back([&, peer] {
        try { f(peer); } catch (const std::exception& e) {
            std::fprintf(stderr, "peer %d: %s\n", peer, e.what());
            std::lock_guard<std::mutex> lock(mutex);
            if (!error) error = std::current_exception();
        }
    });
    for (auto& t : threads) t.join();
    if (error) std::rethrow_exception(error);
}
static void run_owner(const Options& o) {
    pin_core(o, 0, true);
    void* table; check(!posix_memalign(&table, 4096, size_t(o.table_rows) * ROW), "allocate CPU shard");
    std::unique_ptr<void, decltype(&free)> allocation(table, &free);
    for (int row = 0; row < o.table_rows; ++row)
        for (unsigned w = 0; w < ROW / 8; ++w)
            static_cast<uint64_t*>(table)[size_t(row) * ROW / 8 + w] = pattern(o.rank * o.table_rows + row, w);
    parallel([&](int peer) {
        pin_core(o, peer, true); Channel c; c.init(o, peer, table, nullptr); serve(c);
    });
}
static void run_requester(const Options& o) {
    pin_core(o, 0, false); Gpu gpu; gpu.init(o);
    Workers workers(o); Plan p(o); Planner planner(o);
    auto raw_input = load_indices(o);
    expected_rows(p, o, raw_input.data());
    cuda_check(cuMemcpyHtoD(gpu.input, raw_input.data(), raw_input.size() * sizeof(uint32_t)), "initialize GPU input indices");
    std::vector<std::unique_ptr<Channel>> channels(8);
    workers.run([&](int peer) {
        pin_core(o, peer, false); channels[peer] = std::make_unique<Channel>();
        channels[peer]->init(o, peer, nullptr, &gpu);
    });
    std::vector<std::unique_ptr<TargetMkey>> targets(8);
    if (o.target_umr != "none") workers.run([&](int peer) {
        targets[peer] = std::make_unique<TargetMkey>(*channels[peer], o, gpu.output);
        channels[peer]->sock.send(targets[peer]->rkey());
    });
    Barrier barrier(o); uint32_t phase = 0;
    struct ScaleShare { uint32_t rank; CUipcMemHandle handle; } share{};
    share.rank = o.rank;
    cuda_check(cuIpcGetMemHandle(&share.handle, gpu.sf_shard), "export scale shard");
    auto shares = barrier.allgather(share);
    for (int local = 0; local < 4; ++local) {
        if (local == o.rank % 4) gpu.sf_peers[local] = gpu.sf_shard;
        else cuda_check(cuIpcOpenMemHandle(&gpu.sf_peers[local], shares[(o.rank / 4) * 4 + local].handle,
                                         CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS), "import peer scale shard");
    }
    for (int round = -o.warmups; round < o.rounds; ++round) {
        for (int pos = 0; pos < (o.methods == "write" ? 1 : 2); ++pos) {
            uint32_t mode = o.methods == "write" ? 2 : (round + o.warmups + pos) % 2 + 1;
            if (o.vary_indices) {
                raw_input = generate_indices(o, round + o.warmups + 1);
                expected_rows(p, o, raw_input.data());
                cuda_check(cuMemcpyHtoD(gpu.input, raw_input.data(), raw_input.size() * sizeof(uint32_t)), "upstream current indices");
            }
            gpu.clear_for_validation(); barrier.wait(++phase);
            auto begin = Clock::now(); std::vector<PeerStats> stats(8);
            gpu.start_scales(o);
            double index_download_ms = gpu.download_indices();
            planner.route(gpu.host_input, p, workers);
            bool target_active = false; double target_build_ms = 0, target_config_ms = 0;
            if (o.target_umr != "none") {
                auto t = Clock::now(); workers.run([&](int peer) { targets[peer]->prepare(p, peer); });
                target_build_ms = us(t)/1000; target_active = true;
                for (auto& target : targets) target_active &= target->fits;
                check(target_active || o.target_umr != "strict", "current mapping exceeds target MKey pool; use adaptive fallback");
                if (target_active) {
                    t = Clock::now(); workers.run([&](int peer) { targets[peer]->configure(); });
                    target_config_ms = us(t)/1000;
                }
            }
            double map_upload_ms = target_active ? 0 : gpu.upload_map(p);
            auto network_begin = Clock::now();
            workers.run([&](int peer) {
                pin_core(o, peer, false); cuda_check(cuCtxSetCurrent(gpu.ctx), "fetch CUDA context");
                stats[peer] = fetch(*channels[peer], p.peers[peer], mode, round + o.warmups, target_active ? targets[peer].get() : nullptr);
            });
            double fetch_ms = us(network_begin) / 1000;
            double reorder_ms = gpu.reorder(!target_active, o.done_gpu ? o.qps : 0, uint64_t(round + o.warmups) + 1);
            double elapsed = us(begin) / 1000;
            if (o.done_gpu) workers.run([&](int peer) {
                Response telemetry{}; channels[peer]->sock.recv(telemetry);
                check(telemetry.status == 0 && telemetry.fetch_id == uint64_t(round + o.warmups) + 1 &&
                      telemetry.groups == p.peers[peer].groups.size(), "post-completion telemetry");
                add(stats[peer].stats, telemetry.stats);
            });
            gpu.validate(p); // Full payload and all guards, outside timed region, every sample.
            gpu.validate_scales(o);
            std::printf("{\"kind\":\"sample\",\"rank\":%d,\"mode\":\"%s\",\"round\":%d,\"warmup\":%s,"
                "\"tokens\":%d,\"layers\":2,\"ept\":24,\"row_bytes\":256,\"bytes\":%llu,\"remote_bytes\":%llu,"
                "\"pool_per_peer\":%d,\"qps_per_peer\":%d,\"distribution\":\"%s\",\"layout\":\"%s\",\"merge_rows\":%d,\"protocol\":\"%s\",\"engine\":\"%s\",\"pipeline\":%d,\"compact_qp\":%s,\"aligned_counts\":%s,\"histogram_recorded\":%s,\"quick_ack\":%s,\"coalesce_control\":%s,\"target_umr\":\"%s\",\"target_active\":%s,\"target_build_ms\":%.6f,\"target_config_ms\":%.6f,\"vary_indices\":%s,\"done_gpu\":%s,\"compact_request\":%s,\"plan_ms\":%.6f,"
                "\"map_upload_ms\":%.6f,\"index_download_ms\":%.6f,\"decode_count_ms\":%.6f,\"prefix_ms\":%.6f,\"fill_ms\":%.6f,"
                "\"including_plan_ms\":%.6f,\"fetch_ms\":%.6f,\"reorder_ms\":%.6f,"
                "\"e2e_ms\":%.6f,\"gbps\":%.6f,\"data_ok\":true,\"scale_ok\":true,\"histogram\":[",
                o.rank, mode == 1 ? "read_umr" : "write_sge", round, round < 0 ? "true" : "false", o.groups,
                (unsigned long long)frame_bytes(o), (unsigned long long)(p.remote_rows * ROW), o.pool, o.qps,
                o.distribution.c_str(), o.layout.c_str(), o.merge_rows, o.protocol.c_str(), o.engine.c_str(), o.pipeline, o.compact_qp ? "true" : "false", o.aligned_counts ? "true" : "false", o.fast_plan ? "false" : "true", o.quick_ack ? "true" : "false", o.coalesce_control ? "true" : "false", o.target_umr.c_str(), target_active ? "true" : "false", target_build_ms, target_config_ms, o.vary_indices ? "true" : "false", o.done_gpu ? "true" : "false", o.compact_request ? "true" : "false", p.plan_ms, map_upload_ms, index_download_ms, p.decode_ms, p.prefix_ms, p.fill_ms, elapsed,
                fetch_ms, reorder_ms, elapsed, frame_bytes(o) / elapsed / 1e6);
            for (int i = 0; i <= 24; ++i) std::printf("%s%llu", i ? "," : "", (unsigned long long)p.histogram[i]);
            std::printf("],\"peers\":[");
            for (int peer = 0; peer < 8; ++peer) {
                const auto& s = stats[peer]; const auto& x = s.stats;
                std::printf("%s{\"owner\":%d,\"target_keys\":%zu,\"target_runs\":%zu,\"groups\":%zu,\"rows\":%zu,\"elapsed_ms\":%.6f,"
                    "\"request_send_ms\":%.6f,\"response_wait_ms\":%.6f,\"ack_send_ms\":%.6f,"
                    "\"owner_receive_ms\":%.6f,\"layout_ms\":%.6f,\"umr_ms\":%.6f,\"umr_post_ms\":%.6f,"
                    "\"data_ms\":%.6f,\"data_post_ms\":%.6f,\"wr_build_ms\":%.6f,\"verbs_post_ms\":%.6f,\"cq_wait_ms\":%.6f,\"owner_ack_ms\":%.6f,\"owner_notify_ms\":%.6f}", peer ? "," : "", peer,
                    targets[peer] ? targets[peer]->used_keys : size_t(0), targets[peer] ? targets[peer]->input_runs : size_t(0), p.peers[peer].groups.size(),
                    p.peers[peer].rows.size(), s.elapsed_us/1000, s.send_us/1000, s.wait_us/1000, s.ack_us/1000,
                    x.owner_receive_us/1000, x.layout_us/1000, x.umr_us/1000, x.umr_post_us/1000,
                    x.data_us/1000, x.data_post_us/1000, x.wr_build_us/1000, x.verbs_post_us/1000, x.cq_wait_us/1000, x.owner_ack_us/1000, x.owner_notify_us/1000);
            }
            std::puts("]}");
        }
    }
    barrier.wait(++phase); gpu.close_scale_peers(o); barrier.wait(++phase);
    for (auto& c : channels) { Command stop{}; c->sock.send(stop); }
}
