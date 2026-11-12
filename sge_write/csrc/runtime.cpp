#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nccl.h>
#include <nccl_device/host.h>
#include <optional>
#include <unordered_map>
#include <climits>
#include <pybind11/stl.h>
#include "owner_service.hpp"

namespace py = pybind11;
static void nc(ncclResult_t rc, const char *what) {
    if (rc != ncclSuccess)
        throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(rc));
}
static int env_int(const char *name, int fallback) {
    const char *p = std::getenv(name);
    return p ? std::stoi(p) : fallback;
}

namespace engram_sge {
class Runtime : public std::enable_shared_from_this<Runtime> {
    unsigned rank, world, local_rank, local_size, device, layers, hidden, element_bytes, max_tokens, ept, packs, qps,
        pool;
    unsigned capacity, row_bytes, tiles_max, tokens = 0, active_qps = 0, egm_blocks = 1, finish_blocks = 1;
    unsigned sf_token_stride = 0, sf_hidden_stride = 0, sf_layer_stride = 0, owners_per_sf_shard = 0;
    unsigned sl_index = 1;
    bool egm_enabled = false, local_sf = false;
    std::vector<unsigned> routes;
    std::vector<std::string> mapping_errors;
    std::vector<CUdeviceptr> table_ptrs;
    bool routes_probed = false, routes_ready = false;
    CUdeviceptr routes_gpu = 0;
    bool has_egm_routes = false, connected = false, closed = false;
    uint64_t timeout_cycles, slot_bytes, table_rows = 0, allocation_bytes = 0;
    std::vector<unsigned> entries, table_bases, sf_bases;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    std::vector<CUdeviceptr> device_allocations;
    CUdeviceptr allocation = 0, packed = 0, output = 0, done = 0, positions = 0, counts = 0, offsets = 0, totals = 0,
                bases = 0, epoch = 0;
    CUdeviceptr entries_gpu = 0, table_bases_gpu = 0, sf_bases_gpu = 0, table_aliases_gpu = 0, inbox_aliases_gpu = 0,
                sf_aliases_gpu = 0, targets_gpu = 0;
    CUdeviceptr pack_counters = 0, dev_comm_gpu = 0;
    CUdeviceptr sf_shard = 0, sf_output = 0;
    std::vector<CUdeviceptr> sf_aliases, inbox_ptrs;
    CUevent issued = nullptr, capture_issued = nullptr;
    CUstream issue_stream = nullptr;
    bool issue_captured = false;
    std::vector<CUevent> finished;
    std::unique_ptr<HostMemory> table, inbox;
    std::vector<std::unique_ptr<HostAlias>> aliases, table_aliases;
    std::vector<std::unique_ptr<FabricAlias>> fabric_aliases;
    std::vector<std::unique_ptr<Channel>> owners, receivers;
    OwnerServices services;
    ncclComm_t communicator = nullptr;
    ncclDevComm_t dev_comm{};
    ncclUniqueId unique_id{};
    ncclWindow_t source_window = nullptr;
    std::vector<ncclWindow_t> targets;
    void *request_rows = nullptr;
    std::unordered_map<std::string, CUfunction> functions;

    CUdeviceptr allocate(size_t bytes) {
        CUdeviceptr p;
        cu(cuMemAlloc(&p, std::max(size_t(8), bytes)), "allocate WRITE workspace");
        device_allocations.push_back(p);
        return p;
    }
    template <class T> CUdeviceptr upload(const std::vector<T> &values) {
        auto p = allocate(values.size() * sizeof(T));
        cu(cuMemcpyHtoD(p, values.data(), values.size() * sizeof(T)), "upload WRITE metadata");
        return p;
    }
    void launch(const char *name, unsigned x, unsigned y, unsigned block, void **args, CUstream stream) {
        cu(cuLaunchKernel(functions.at(name), std::max(1u, x), std::max(1u, y), 1, block, 1, 1, 0, stream, args,
                          nullptr),
           name);
    }
    bool serves(unsigned peer) const { return routes.at(peer) == 0; }
    void healthy() {
        services.check();
        ck(connected && !closed, "WRITE runtime is not connected");
    }
    CUstream stream() { return at::cuda::getCurrentCUDAStream(device).stream(); }

    py::bytes endpoint_bytes(const std::unique_ptr<Channel> &channel) {
        if (!channel)
            return py::bytes();
        return py::bytes(reinterpret_cast<const char *>(channel->endpoints.data()),
                         channel->endpoints.size() * sizeof(Endpoint));
    }
    std::vector<Endpoint> parse_endpoints(py::handle b) {
        std::string s = py::cast<std::string>(b);
        ck(s.size() == qps * sizeof(Endpoint), "endpoint ABI");
        std::vector<Endpoint> v(qps);
        std::memcpy(v.data(), s.data(), s.size());
        return v;
    }

  public:
    unsigned gin_min_stride = 0;
    Runtime(unsigned rank_, unsigned world_, unsigned local_rank_, unsigned local_size_, unsigned device_,
            const std::vector<unsigned> &entries_, unsigned hidden_, unsigned elem_, unsigned max_tokens_,
            unsigned ept_, unsigned packs_, unsigned qps_, unsigned depth, unsigned egm_, unsigned timeout, unsigned sl,
            const std::string &package_ptx)
        : rank(rank_), world(world_), local_rank(local_rank_), local_size(local_size_), device(device_),
          layers(entries_.size()), hidden(hidden_), element_bytes(elem_), max_tokens(max_tokens_), ept(ept_),
          packs(packs_), qps(qps_), egm_enabled(egm_ != 0), entries(entries_) {
        try {
            sl_index = sl;
            ck(egm_ <= 1, "ENGRAM_EGM_ENABLE must be 0 or 1");
            int sf_mode = env_int("ENGRAM_WRITE_LOCAL_SF", egm_enabled ? 1 : 0);
            ck(sf_mode == 0 || sf_mode == 1, "ENGRAM_WRITE_LOCAL_SF must be 0 or 1");
            local_sf = sf_mode;
            ck(world >= 1 && world <= 64 && layers >= 1 && layers <= 32 && local_size >= 1 && world % local_size == 0,
               "WRITE rank/layer limits");
            ck(local_rank == rank % local_size && rank < world && qps >= 1 && qps <= 16 && depth >= 32 && timeout >= 1,
               "WRITE configuration");
            owners_per_sf_shard = local_sf ? world : world / local_size;
            ck(uint64_t(max_tokens) * ept * layers <= UINT32_MAX && max_tokens >= 1 && ept >= 1, "WRITE capacity");
            capacity = max_tokens * ept;
            row_bytes = hidden * element_bytes;
            ck(row_bytes > 0 && row_bytes % 16 == 0, "WRITE row alignment");
            pool = std::min(1024u, (depth - 16) * qps);
            tiles_max = (capacity + kTile - 1) / kTile;
            cu(cuInit(0), "CUDA init");
            cu(cuDevicePrimaryCtxRetain(&context, device), "retain CUDA context");
            cu(cuCtxSetCurrent(context), "set CUDA context");
            int ordering, clock;
            cu(cuDeviceGetAttribute(&ordering, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, device),
               "RDMA ordering");
            ck(ordering >= CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER, "GPU DONE requires native OWNER ordering");
            cu(cuDeviceGetAttribute(&clock, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device), "GPU clock");
            timeout_cycles = uint64_t(clock) * 1000 * timeout;
            unsigned sf_at = 0;
            for (unsigned n : entries) {
                ck(n > 0 && uint64_t(n) * world <= INT32_MAX, "vocabulary index limit");
                table_bases.push_back(table_rows);
                sf_bases.push_back(sf_at);
                table_rows += n;
                sf_at += n * owners_per_sf_shard * packs;
            }
            ck(table_rows <= UINT32_MAX, "local vocabulary size");
            // 一份CUDA VMM CPU词表同时服务RDMA和EGM，更新时只写这一份内存。
            // 开启EGM时尝试额外导出FABRIC句柄；能力不足由bootstrap选择RDMA。
            table = std::make_unique<HostMemory>(table_rows * row_bytes, device, egm_enabled);
            slot_bytes = aligned(uint64_t(capacity) * 4 + 64);
            inbox = std::make_unique<HostMemory>(slot_bytes * world * layers, device);
            size_t data_bytes = uint64_t(layers) * capacity * row_bytes;
            allocation_bytes = aligned(data_bytes) * 2 + aligned(uint64_t(layers) * world * qps * 8);
            allocation = allocate(allocation_bytes);
            packed = allocation;
            output = allocation + aligned(data_bytes);
            done = output + aligned(data_bytes);
            unsigned sync = 1;
            cu(cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, allocation), "SYNC_MEMOPS");
            cu(cuMemsetD8(done, 0, uint64_t(layers) * world * qps * 8), "clear DONE epochs");
            positions = allocate(uint64_t(layers) * capacity * 4);
            counts = allocate(uint64_t(layers) * world * tiles_max * 4);
            offsets = allocate(uint64_t(layers) * world * tiles_max * 4);
            totals = allocate(layers * world * 4);
            bases = allocate(layers * world * 4);
            epoch = allocate(layers * world * 8);
            cu(cuMemsetD8(epoch, 0, layers * world * 8), "initialize owner epochs");
            pack_counters = allocate(layers * 4);
            entries_gpu = upload(entries);
            table_bases_gpu = upload(table_bases);
            sf_bases_gpu = upload(sf_bases);
            sf_shard = allocate(std::max(8u, sf_at * 4));
            sf_output = allocate(uint64_t(layers) * aligned(max_tokens, 4) * ept * packs * 4);
            nc(ncclMemAlloc(&request_rows, uint64_t(layers) * capacity * 4), "allocate GIN request rows");
            cu(cuEventCreate(&issued, CU_EVENT_DISABLE_TIMING), "issue event");
            cu(cuEventCreate(&capture_issued, CU_EVENT_DISABLE_TIMING), "capture issue dependency");
            for (unsigned i = 0; i < layers; ++i) {
                CUevent e;
                cu(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING), "layer completion event");
                finished.push_back(e);
            }
            const char *override_ptx = std::getenv("ENGRAM_GPU_PTX");
            std::string ptx = override_ptx && *override_ptx ? override_ptx : package_ptx;
            cu(cuModuleLoad(&module, ptx.c_str()), "load independent WRITE kernels");
            for (const char *name : {"route_counts", "route_prefix", "route_pack", "send_requests", "gather_local",
                                     "gather_scales", "issue_egm", "pack_send_egm", "finish_layer"}) {
                CUfunction fn;
                cu(cuModuleGetFunction(&fn, module, name), name);
                functions[name] = fn;
            }
            int sms;
            cu(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "SM count");
            finish_blocks = sms;
            egm_blocks = sms * 2;
            owners.resize(world);
            receivers.resize(world);
            if (!rank)
                nc(ncclGetUniqueId(&unique_id), "GIN unique ID");
        } catch (...) {
            release_resources();
            throw;
        }
    }

    py::dict bootstrap() {
        py::dict d;
        d["pid"] = getpid();
        d["table_fd"] = table->fd;
        d["table_bytes"] = table->bytes;
        d["table_numa"] = table->numa;
        auto fabric = table->export_fabric();
        d["table_fabric"] = py::bytes(reinterpret_cast<const char *>(fabric.data()), fabric.size());
        d["fabric_error"] = table->fabric_error;
        d["inbox_fd"] = inbox->fd;
        d["inbox_bytes"] = inbox->bytes;
        d["unique_id"] = py::bytes(reinterpret_cast<char *>(&unique_id), sizeof(unique_id));
        CUipcMemHandle h;
        cu(cuIpcGetMemHandle(&h, sf_shard), "export scale shard");
        d["scale_handle"] = py::bytes(reinterpret_cast<char *>(&h), sizeof(h));
        py::list owner_eps, receiver_eps, receive_info;
        for (unsigned i = 0; i < world; ++i) {
            owner_eps.append(endpoint_bytes(owners[i]));
            receiver_eps.append(endpoint_bytes(receivers[i]));
            receive_info.append(receivers[i]
                                    ? py::make_tuple(packed, done + uint64_t(i) * qps * 8, receivers[i]->mr->rkey)
                                    : py::make_tuple(0, 0, 0));
        }
        d["owners"] = owner_eps;
        d["receivers"] = receiver_eps;
        d["receive_info"] = receive_info;
        d["config"] = py::make_tuple(world, local_size, entries, hidden, element_bytes, max_tokens, ept, packs, qps,
                                     egm_enabled, local_sf);
        return d;
    }
    py::dict probe_routes(const py::list &peers, const std::vector<unsigned> &candidates) {
        ck(!routes_probed && !connected && !closed && peers.size() == world && candidates.size() == world,
           "WRITE route probe state");
        routes = candidates;
        mapping_errors.resize(world);
        table_ptrs.resize(world, 0);
        table_aliases.resize(world);
        fabric_aliases.resize(world);
        for (unsigned peer = 0; peer < world; ++peer) {
            ck(candidates[peer] <= 1 && (!candidates[peer] || egm_enabled), "invalid WRITE EGM candidate");
            if (!candidates[peer])
                continue;
            try {
                auto d = peers[peer].cast<py::dict>();
                if (peer / local_size == rank / local_size) {
                    if (peer == rank)
                        table_ptrs[peer] = table->ptr;
                    else {
                        auto alias = std::make_unique<HostAlias>(py::cast<int>(d["pid"]), py::cast<int>(d["table_fd"]),
                                                                 py::cast<size_t>(d["table_bytes"]), device,
                                                                 py::cast<int>(d["table_numa"]));
                        table_ptrs[peer] = alias->ptr;
                        table_aliases[peer] = std::move(alias);
                    }
                } else {
                    auto alias = std::make_unique<FabricAlias>(py::cast<std::string>(d["table_fabric"]),
                                                               py::cast<size_t>(d["table_bytes"]), device);
                    table_ptrs[peer] = alias->ptr;
                    fabric_aliases[peer] = std::move(alias);
                }
            } catch (const std::runtime_error &error) {
                routes[peer] = 0;
                table_ptrs[peer] = 0;
                mapping_errors[peer] = error.what();
            }
        }
        routes_probed = true;
        py::dict result;
        result["routes"] = routes;
        result["errors"] = mapping_errors;
        return result;
    }

    py::dict prepare_routes(const py::list &peers, const std::vector<unsigned> &selected) {
        ck(routes_probed && !routes_ready && !connected && !closed && peers.size() == world && selected.size() == world,
           "WRITE route commit state");
        has_egm_routes = false;
        for (unsigned peer = 0; peer < world; ++peer) {
            ck(selected[peer] <= routes[peer], "WRITE route exceeds probed capability");
            if (!selected[peer]) {
                table_ptrs[peer] = 0;
                table_aliases[peer].reset();
                fabric_aliases[peer].reset();
            } else
                has_egm_routes = true;
        }
        routes = selected;
        // 直到两端协商完成才发布GPU路由、创建QP；反向映射失败也只降级当前rank对。
        inbox_ptrs.clear();
        for (unsigned peer = 0; peer < world; ++peer) {
            auto d = peers[peer].cast<py::dict>();
            bool same_node = peer / local_size == rank / local_size;
            if (same_node) {
                if (peer == rank) {
                    inbox_ptrs.push_back(inbox->ptr);
                    if (!local_sf)
                        sf_aliases.push_back(sf_shard);
                } else {
                    auto b = std::make_unique<HostAlias>(py::cast<int>(d["pid"]), py::cast<int>(d["inbox_fd"]),
                                                         py::cast<size_t>(d["inbox_bytes"]), device, inbox->numa);
                    inbox_ptrs.push_back(b->ptr);
                    aliases.push_back(std::move(b));
                    if (!local_sf) {
                        std::string blob = py::cast<std::string>(d["scale_handle"]);
                        ck(blob.size() == sizeof(CUipcMemHandle), "scale IPC ABI");
                        CUipcMemHandle h;
                        std::memcpy(&h, blob.data(), sizeof(h));
                        CUdeviceptr p;
                        cu(cuIpcOpenMemHandle(&p, h, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS), "import scale shard");
                        sf_aliases.push_back(p);
                    }
                }
            }
        }
        if (local_sf)
            sf_aliases.push_back(sf_shard);
        routes_gpu = upload(routes);
        table_aliases_gpu = upload(table_ptrs);
        inbox_aliases_gpu = upload(inbox_ptrs);
        sf_aliases_gpu = upload(sf_aliases);
        std::string hca =
            std::getenv("ENGRAM_WRITE_HCA") ? std::getenv("ENGRAM_WRITE_HCA") : "mlx5_bond_" + std::to_string(device);
        int gid = env_int("NCCL_IB_GID_INDEX", 3);
        // QP和CPU服务线程只为RDMA边建立。直接读owner的totals恒为0，既不发请求也不等DONE。
        for (unsigned peer = 0; peer < world; ++peer)
            if (serves(peer)) {
                owners[peer] = std::make_unique<Channel>(hca, qps, pool, true, reinterpret_cast<void *>(table->ptr),
                                                         table_rows * row_bytes, gid, sl_index);
                receivers[peer] = std::make_unique<Channel>(hca, qps, pool, false, reinterpret_cast<void *>(allocation),
                                                            allocation_bytes, gid, sl_index);
            }
        routes_ready = true;
        return bootstrap();
    }
    void connect(const py::list &peers) {
        ck(routes_ready && !connected && !closed && peers.size() == world, "bootstrap state");
        for (unsigned peer = 0; peer < world; ++peer)
            if (serves(peer)) {
                auto d = peers[peer].cast<py::dict>();
                owners[peer]->connect(parse_endpoints(d["receivers"].cast<py::list>()[rank]));
                receivers[peer]->connect(parse_endpoints(d["owners"].cast<py::list>()[rank]));
                auto info = d["receive_info"].cast<py::list>()[rank].cast<py::tuple>();
                owners[peer]->peer = {py::cast<uint64_t>(info[0]), py::cast<uint64_t>(info[1]),
                                      py::cast<uint32_t>(info[2])};
            }
        auto root = peers[0].cast<py::dict>();
        std::string uid = py::cast<std::string>(root["unique_id"]);
        ck(uid.size() == sizeof(unique_id), "GIN ID ABI");
        std::memcpy(&unique_id, uid.data(), uid.size());
        // 仅初始化阶段释放GIL并建立GIN。forward从不调用进程组collective或主机同步。
        {
            py::gil_scoped_release release;
            nc(ncclCommInitRank(&communicator, world, unique_id, rank), "initialize independent GIN");
            ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
            nc(ncclCommQueryProperties(communicator, &props), "GIN properties");
            gin_min_stride = props.ginMinStride;
            ck(gin_min_stride == local_size, "detected GIN stride must match node-local group size");
            ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
            req.ginType = NCCL_GIN_TYPE_GDAKI;
            req.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
            req.ginCustomStride = props.ginMinStride;
            req.ginContextCount = std::max(1u, local_size);
            req.ginQueueDepth = 1024;
            req.ginExclusiveContexts = true;
            req.ginSignalCount = world * layers;
            req.useRuntimeVersion = true;
            ck(props.devCommRuntimeVersionSize == sizeof(dev_comm), "NCCL host/device ABI");
            nc(ncclDevCommCreate(communicator, &req, &dev_comm), "GIN device communicator");
            nc(ncclCommWindowRegister(communicator, request_rows, uint64_t(layers) * capacity * 4, &source_window,
                                      NCCL_WIN_GIN_ONLY | NCCL_WIN_STRICT_ORDERING),
               "register GPU request rows");
            targets.resize(local_size);
            for (unsigned i = 0; i < local_size; ++i)
                nc(ncclCommWindowRegister(communicator, reinterpret_cast<void *>(inbox_ptrs[i]), inbox->bytes,
                                          &targets[i], NCCL_WIN_GIN_ONLY | NCCL_WIN_STRICT_ORDERING),
                   "register mailbox alias");
        }
        targets_gpu = upload(targets);
        dev_comm_gpu = upload(std::vector<ncclDevComm_t>{dev_comm});
        connected = true;
        services.start({world, layers, capacity, pool, qps, row_bytes, table_rows, slot_bytes, table->ptr, inbox->ptr},
                       owners);
    }
    void write(unsigned layer, const torch::Tensor &storage, const std::optional<torch::Tensor> &scales) {
        healthy();
        ck(layer < layers, "write layer");
        c10::cuda::CUDAGuard guard(device);
        auto dst = table->ptr + uint64_t(table_bases[layer]) * row_bytes;
        size_t bytes = uint64_t(entries[layer]) * row_bytes;
        if (storage.is_cuda())
            cu(cuMemcpyDtoH(reinterpret_cast<void *>(dst), reinterpret_cast<CUdeviceptr>(storage.data_ptr()), bytes),
               "write GPU vocabulary to CPU owner");
        else
            std::memcpy(reinterpret_cast<void *>(dst), storage.data_ptr(), bytes);
        if (packs) {
            ck(scales.has_value(), "missing scales");
            size_t rows = size_t(entries[layer]) * owners_per_sf_shard, count = rows * packs;
            auto *src =
                reinterpret_cast<const char *>(scales->data_ptr()) + (local_sf ? 0 : size_t(local_rank) * count * 4);
            auto target = sf_shard + uint64_t(sf_bases[layer]) * 4;
            if (scales->is_cuda())
                cu(cuMemcpyDtoD(target, reinterpret_cast<CUdeviceptr>(src), count * 4), "write scale shard");
            else
                cu(cuMemcpyHtoD(target, src, count * 4), "write CPU scale shard");
        }
    }
    py::list fetch(const torch::Tensor &indices, unsigned requested_qps, bool col_major) {
        healthy();
        c10::cuda::CUDAGuard guard(device);
        auto s = stream();
        tokens = indices.size(1);
        active_qps = requested_qps;
        // eager路径等上一轮scatter完成，支持跨stream复用。捕获从已静止的Buffer开始，
        // 不把捕获外旧事件作为内部依赖（CUDA会拒绝），Graph内由stream/DAG保证顺序。
        CUstreamCaptureStatus capture_status;
        cu(cuStreamIsCapturing(s, &capture_status), "query issue capture");
        issue_captured = capture_status != CU_STREAM_CAPTURE_STATUS_NONE;
        issue_stream = s;
        if (!issue_captured)
            for (auto e : finished)
                cu(cuStreamWaitEvent(s, e, 0), "workspace reuse dependency");
        unsigned rows = tokens * ept, tiles = std::max(1u, (rows + kTile - 1) / kTile);
        CUdeviceptr input = reinterpret_cast<CUdeviceptr>(indices.data_ptr()),
                    request = reinterpret_cast<CUdeviceptr>(request_rows);
        sf_token_stride = col_major ? 1 : ept * packs;
        sf_hidden_stride = col_major ? aligned(tokens, 4) : 1;
        sf_layer_stride = col_major ? sf_hidden_stride * ept * packs : tokens * ept * packs;
        if (has_egm_routes) {
            unsigned vectors = row_bytes / 16;
            // 普通网格没有全GPU屏障；小批次只启动足够覆盖行数的CTA。
            unsigned blocks = std::min(egm_blocks, std::max(1u, (rows + 27) / 28));
            void *args[] = {&input,
                            &output,
                            &table_aliases_gpu,
                            &sf_output,
                            &sf_aliases_gpu,
                            &entries_gpu,
                            &table_bases_gpu,
                            &sf_bases_gpu,
                            &rows,
                            &capacity,
                            &vectors,
                            &layers,
                            &ept,
                            &packs,
                            &owners_per_sf_shard,
                            &sf_token_stride,
                            &sf_hidden_stride,
                            &sf_layer_stride,
                            &rank,
                            &world,
                            &local_size,
                            &tiles,
                            &pack_counters,
                            &counts,
                            &totals,
                            &routes_gpu};
            launch("issue_egm", blocks, 1, 512, args, s);
            // 前一kernel结束是counts和本地输出的完成依赖；不把GPU判断读回CPU。
            // 每层最后一个打包CTA发布邮箱，纯机内时该节点只做只读检查并立即返回。
            unsigned pack_blocks = std::min(finish_blocks, tiles);
            void *send[] = {
                &input,         &request,       &positions,         &counts,          &totals,     &bases,
                &epoch,         &pack_counters, &entries_gpu,       &table_bases_gpu, &rows,       &capacity,
                &rank,          &world,         &local_size,        &layers,          &tiles,      &dev_comm_gpu,
                &source_window, &targets_gpu,   &inbox_aliases_gpu, &slot_bytes,      &active_qps, &routes_gpu};
            launch("pack_send_egm", pack_blocks, layers, 256, send, s);
        } else {
            // 以下launch都排在调用方当前stream上；外部capture会原样记录为Graph节点，
            // 不再由WRITE内部构建/启动一个包含wait的完整Graph。
            // counts -> prefix -> pack -> send_requests：
            // prefix在GPU端按[layer,owner]递增非空请求序号，Graph每次replay自动推进。
            // counts统计[layer,owner,tile]；prefix生成每owner的总数、打包起点及tile偏移。
            void *count[] = {&input, &counts, &rows, &world, &entries_gpu, &rank, &local_size, &routes_gpu};
            launch("route_counts", tiles, layers, 256, count, s);
            void *prefix[] = {&counts, &offsets, &totals, &bases, &epoch, &world, &tiles};
            launch("route_prefix", layers, 1, 64, prefix, s);
            // pack按owner收拢本地词表行号，同时保留positions用于最终恢复token/entry顺序。
            // CPU只看到行号列表；embedding本体不会从GPU搬到CPU来打包。
            void *pack[] = {&input, &request,  &positions, &offsets, &bases,      &entries_gpu, &table_bases_gpu,
                            &rows,  &capacity, &world,     &rank,    &local_size, &routes_gpu};
            launch("route_pack", tiles, layers, 256, pack, s);
            void *send[] = {&dev_comm_gpu, &source_window, &targets_gpu, &inbox_aliases_gpu,
                            &request,      &totals,        &bases,       &rank,
                            &world,        &local_size,    &layers,      &capacity,
                            &slot_bytes,   &routes_gpu,    &active_qps,  &epoch};
            launch("send_requests", world, layers, 256, send, s);
            // send只负责发布邮箱，不等待CPU服务或DONE；随后可与CPU/网卡传输重叠地
            // 收集本地数据及scales。issued记录在这些本地准备工作之后。
            unsigned words = row_bytes / 8, grid = std::max(1u, std::min(4096u, (rows * words + 255) / 256));
            void *local[] = {&input, &output, &table_aliases_gpu, &entries_gpu, &table_bases_gpu, &rows, &capacity,
                             &words, &rank,   &local_size,        &routes_gpu};
            launch("gather_local", grid, layers, 256, local, s);
            if (packs) {
                unsigned owners_per_shard = owners_per_sf_shard;
                void *sf[] = {
                    &input, &sf_output, &sf_aliases_gpu,   &entries_gpu,     &sf_bases_gpu,     &tokens,
                    &ept,   &packs,     &owners_per_shard, &sf_token_stride, &sf_hidden_stride, &sf_layer_stride};
                launch("gather_scales", std::max(1u, std::min(512u, (rows + 255) / 256)), layers, 256, sf, s);
            }
        }
        if (issue_captured) {
            // 内部事件负责让另一个hook stream加入同一capture；external事件保留实际
            // GPU完成通知，以便在Graph之外调用hook。不能用同一个普通捕获事件混淆两者。
            cu(cuEventRecord(capture_issued, s), "record capture issue dependency");
            cu(cuEventRecordWithFlags(issued, s, CU_EVENT_RECORD_EXTERNAL), "record external issue completion");
        } else
            cu(cuEventRecord(issued, s), "record issue completion");
        // 此处没有wait DONE或scatter。真正的层完成依赖只在对应hook中提交。
        py::list result;
        auto keep = shared_from_this();
        auto type = packs ? torch::kFloat8_e4m3fn : torch::kBFloat16;
        auto data_options = torch::TensorOptions().dtype(type).device(torch::Device(torch::kCUDA, device));
        for (unsigned l = 0; l < layers; ++l) {
            auto data = torch::from_blob(
                reinterpret_cast<void *>(output + uint64_t(l) * capacity * row_bytes),
                {int64_t(tokens), int64_t(hidden) * ept}, [keep](void *) {}, data_options);
            if (packs) {
                auto sf = torch::from_blob(
                              reinterpret_cast<void *>(sf_output + uint64_t(l) * sf_layer_stride * 4),
                              {int64_t(sf_layer_stride)}, [keep](void *) {},
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, device)))
                              .as_strided({int64_t(tokens), int64_t(ept) * packs},
                                          {int64_t(sf_token_stride), int64_t(sf_hidden_stride)});
                result.append(py::make_tuple(data, sf));
            } else
                result.append(data);
        }
        return result;
    }
    void wait_layer(unsigned layer) {
        healthy();
        ck(layer < layers, "wait layer");
        c10::cuda::CUDAGuard guard(device);
        auto s = stream();
        // 同一capture中的跨stream hook通过内部事件加入DAG；捕获外使用真实完成事件。
        CUstreamCaptureStatus source_status = CU_STREAM_CAPTURE_STATUS_NONE, hook_status;
        if (issue_captured)
            cu(cuStreamIsCapturing(issue_stream, &source_status), "query source capture");
        if (source_status == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
            cu(cuStreamWaitEvent(s, capture_issued, 0), "capture issue to hook dependency");
        } else {
            cu(cuStreamIsCapturing(s, &hook_status), "query hook capture");
            cu(cuStreamWaitEvent(s, issued,
                                 hook_status == CU_STREAM_CAPTURE_STATUS_ACTIVE ? CU_EVENT_WAIT_EXTERNAL : 0),
               "issue to hook dependency");
        }
        // hook只提交GPU工作，不在CPU轮询完成：issued依赖 -> 本层finish。
        // wait只检查[layer,owner,QP]对应epoch；其他层是否已调用hook不影响这一层。
        // scatter消费packed与positions，finished供下一轮eager fetch保护工作区复用。
        unsigned words = row_bytes / 8, grid = std::max(1u, std::min(4096u, (tokens * ept * words + 255) / 256));
        grid = std::min(grid, finish_blocks);
        void *finish[] = {&done,  &epoch,    &totals, &packed, &output,     &positions,     &layer,
                          &world, &capacity, &words,  &qps,    &active_qps, &timeout_cycles};
        launch("finish_layer", grid, 1, 256, finish, s);
        cu(cuStreamIsCapturing(s, &hook_status), "query layer completion capture");
        cu(cuEventRecordWithFlags(finished[layer], s,
                                  hook_status == CU_STREAM_CAPTURE_STATUS_ACTIVE ? CU_EVENT_RECORD_EXTERNAL : 0),
           "record layer completion");
    }
    py::dict memory_usage() {
        py::dict d;
        d["rank"] = rank;
        d["data_allocation_bytes"] = allocation_bytes;
        d["staging_bytes"] = uint64_t(layers) * capacity * row_bytes;
        d["cpu_vocabulary_bytes"] = table_rows * row_bytes;
        d["cpu_mailbox_bytes"] = inbox->bytes;
        d["owner_threads"] = services.thread_count();
        d["egm_enabled"] = egm_enabled;
        d["has_egm_routes"] = has_egm_routes;
        d["egm_blocks"] = egm_blocks;
        d["routes"] = routes;
        d["egm_mapping_errors"] = mapping_errors;
        d["local_sf"] = local_sf;
        d["table_numa"] = table->numa;
        d["cpu_vocabulary_allocations"] = 1;
        d["cpu_vocabulary_allocated_bytes"] = table->bytes;
        d["fabric_error"] = table->fabric_error;
        return d;
    }
    void close() {
        if (closed)
            return;
        cu(cuCtxSetCurrent(context), "close CUDA context");
        cu(cuCtxSynchronize(), "finish WRITE before shutdown");
        {
            py::gil_scoped_release release;
            services.stop_and_join();
        }
        owners.clear();
        receivers.clear();
        if (communicator) {
            if (source_window)
                nc(ncclCommWindowDeregister(communicator, source_window), "deregister requests");
            for (auto w : targets)
                nc(ncclCommWindowDeregister(communicator, w), "deregister mailbox");
            nc(ncclDevCommDestroy(communicator, &dev_comm), "destroy GIN device");
            nc(ncclCommDestroy(communicator), "destroy GIN");
            communicator = nullptr;
        }
        // collective destroy后的Python barrier保证所有importer先释放别名，
        // 再允许exporter随输出Tensor的生命周期释放原始CPU/GPU分配。
        if (!local_sf)
            for (unsigned i = 0; i < sf_aliases.size(); ++i)
                if (i != local_rank)
                    cu(cuIpcCloseMemHandle(sf_aliases[i]), "close scale alias");
        sf_aliases.clear();
        aliases.clear();
        table_aliases.clear();
        fabric_aliases.clear();
        // 输出Tensor持有Runtime共享所有权；close只关闭通信，内存到最后一个Tensor释放才回收。
        closed = true;
        connected = false;
    }
    void release_resources() noexcept {
        if (context)
            cuCtxSetCurrent(context);
        services.stop_and_join();
        owners.clear();
        receivers.clear();
        if (communicator) {
            if (source_window)
                ncclCommWindowDeregister(communicator, source_window);
            for (auto w : targets)
                ncclCommWindowDeregister(communicator, w);
            ncclDevCommDestroy(communicator, &dev_comm);
            ncclCommDestroy(communicator);
        }
        if (issued)
            cuEventDestroy(issued);
        if (capture_issued)
            cuEventDestroy(capture_issued);
        for (auto e : finished)
            cuEventDestroy(e);
        if (!local_sf)
            for (unsigned i = 0; i < sf_aliases.size(); ++i)
                if (i != local_rank)
                    cuIpcCloseMemHandle(sf_aliases[i]);
        aliases.clear();
        table_aliases.clear();
        fabric_aliases.clear();
        inbox.reset();
        table.reset();
        if (request_rows)
            ncclMemFree(request_rows);
        for (auto p : device_allocations)
            cuMemFree(p);
        if (module)
            cuModuleUnload(module);
        if (context)
            cuDevicePrimaryCtxRelease(device);
    }
    ~Runtime() { release_resources(); }
};
} // namespace engram_sge

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using Runtime = engram_sge::Runtime;
    py::class_<Runtime, std::shared_ptr<Runtime>>(m, "Runtime")
        .def(py::init<unsigned, unsigned, unsigned, unsigned, unsigned, const std::vector<unsigned> &, unsigned,
                      unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                      const std::string &>())
        .def("probe_routes", &Runtime::probe_routes)
        .def("prepare_routes", &Runtime::prepare_routes)
        .def("bootstrap", &Runtime::bootstrap)
        .def("connect", &Runtime::connect)
        .def("write", &Runtime::write)
        .def("fetch", &Runtime::fetch)
        .def("wait_layer", &Runtime::wait_layer)
        .def("close", &Runtime::close)
        .def("memory_usage", &Runtime::memory_usage)
        .def_readonly("gin_min_stride", &Runtime::gin_min_stride);
}
