#include <cuda_runtime.h>
#include <nccl_device.h>
#include <cuda/atomic>
#include "protocol.hpp"

// 所有shape来自set_config/fetch，不再写死8 ranks、两层或256B行宽。
// counts/offsets:[layer,owner,tile]；totals/bases:[layer,owner]。
__device__ void count_tile(const int *input, unsigned *counts, unsigned rows, unsigned world, const unsigned *entries,
                           unsigned rank, unsigned local, const unsigned *routes, unsigned tile, unsigned layer,
                           unsigned tiles) {
    __shared__ unsigned bins[64];
    for (unsigned i = threadIdx.x; i < world; i += blockDim.x)
        bins[i] = 0;
    __syncthreads();
    unsigned lo = tile * kTile, hi = min(lo + kTile, rows), n = entries[layer];
    for (unsigned at = lo + threadIdx.x; at < hi; at += blockDim.x) {
        int x = input[size_t(layer) * rows + at];
        if (x < 0)
            continue;
        unsigned owner = unsigned(x) / n;
        if (owner >= world)
            asm volatile("trap;");
        if (!routes[owner])
            atomicAdd(bins + owner, 1);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < world; i += blockDim.x)
        counts[(layer * world + i) * tiles + tile] = bins[i];
    __syncthreads(); // 同一个CTA可能继续处理下一tile，先完成本tile读取。
}
extern "C" __global__ void route_counts(const int *input, unsigned *counts, unsigned rows, unsigned world,
                                        const unsigned *entries, unsigned rank, unsigned local,
                                        const unsigned *routes) {
    count_tile(input, counts, rows, world, entries, rank, local, routes, blockIdx.x, blockIdx.y, gridDim.x);
}
__device__ void prefix_layer(const unsigned *counts, unsigned *offsets, unsigned *totals, unsigned *bases,
                             unsigned long long *epochs, unsigned world, unsigned tiles, unsigned layer) {
    unsigned owner = threadIdx.x;
    if (owner < world) {
        unsigned sum = 0;
        for (unsigned b = 0; b < tiles; ++b) {
            unsigned p = (layer * world + owner) * tiles + b;
            offsets[p] = sum;
            sum += counts[p];
        }
        totals[layer * world + owner] = sum;
        // 每个[layer,owner]独立递增，只给非空请求分配序号。不能使用全局fetch序号：
        // 纯机内/空层不发送邮箱，之后再次访问远端时GIN signal仍只会增加一次。
        if (sum)
            ++epochs[layer * world + owner];
    }
    __syncthreads();
    if (!owner) {
        unsigned sum = 0;
        for (unsigned o = 0; o < world; ++o) {
            bases[layer * world + o] = sum;
            sum += totals[layer * world + o];
        }
    }
}
extern "C" __global__ void route_prefix(const unsigned *counts, unsigned *offsets, unsigned *totals, unsigned *bases,
                                        unsigned long long *epochs, unsigned world, unsigned tiles) {
    prefix_layer(counts, offsets, totals, bases, epochs, world, tiles, blockIdx.x);
}
template <unsigned Warps, bool LocalPrefix = false>
__device__ void pack_tile(const int *input, unsigned *packed, unsigned *positions, const unsigned *offsets,
                          const unsigned *bases, const unsigned *entries, const unsigned *table_bases, unsigned rows,
                          unsigned capacity, unsigned world, unsigned rank, unsigned local, const unsigned *routes,
                          unsigned tile, unsigned layer, unsigned tiles) {
    __shared__ unsigned counts[64 * Warps], warp_bases[64 * Warps], consumed[64];
    for (unsigned i = threadIdx.x; i < world; i += blockDim.x)
        consumed[i] = 0;
    __syncthreads();
    unsigned lo = tile * kTile, hi = min(lo + kTile, rows), lane = threadIdx.x % 32, warp = threadIdx.x / 32,
             n = entries[layer];
    for (unsigned step = 0; step < hi - lo; step += blockDim.x) {
        for (unsigned i = threadIdx.x; i < world * Warps; i += blockDim.x)
            counts[i] = 0;
        __syncthreads();
        unsigned at = lo + step + threadIdx.x;
        int x = at < hi ? input[size_t(layer) * rows + at] : -1;
        unsigned owner = x >= 0 ? unsigned(x) / n : world;
        bool active = at < hi && x >= 0 && (!routes[owner]);
        if (!active)
            owner = world;
        unsigned mask = __match_any_sync(0xffffffff, owner);
        if (active && lane == unsigned(__ffs(mask) - 1))
            counts[owner * Warps + warp] = __popc(mask);
        __syncthreads();
        for (unsigned o = threadIdx.x; o < world; o += blockDim.x) {
            unsigned sum = consumed[o];
            for (unsigned w = 0; w < Warps; ++w) {
                warp_bases[o * Warps + w] = sum;
                sum += counts[o * Warps + w];
            }
            consumed[o] = sum;
        }
        __syncthreads();
        if (active) {
            unsigned p = layer * capacity + (LocalPrefix ? bases[owner] : bases[layer * world + owner]) +
                         (LocalPrefix ? offsets[owner] : offsets[(layer * world + owner) * tiles + tile]) +
                         warp_bases[owner * Warps + warp] + __popc(mask & ((1u << lane) - 1));
            packed[p] = table_bases[layer] + unsigned(x) % n;
            positions[p] = at;
        }
        __syncthreads();
    }
}
extern "C" __global__ void route_pack(const int *input, unsigned *packed, unsigned *positions, const unsigned *offsets,
                                      const unsigned *bases, const unsigned *entries, const unsigned *table_bases,
                                      unsigned rows, unsigned capacity, unsigned world, unsigned rank, unsigned local,
                                      const unsigned *routes) {
    pack_tile<8>(input, packed, positions, offsets, bases, entries, table_bases, rows, capacity, world, rank, local,
                 routes, blockIdx.x, blockIdx.y, gridDim.x);
}

// GIN提交包含较多寄存器状态。保持独立device函数，并通过设备指针访问comm，
// 避免把整个通信描述符及GIN临时变量带入纯机内gather的寄存器活跃区间。
__device__ __noinline__ void send_request(const ncclDevComm_t *comm_ptr, ncclWindow_t source,
                                          const ncclWindow_t *targets, const unsigned long long *aliases,
                                          const unsigned *packed, const unsigned *totals, const unsigned *bases,
                                          unsigned rank, unsigned world, unsigned local, unsigned layers,
                                          unsigned capacity, unsigned long long slot_bytes, const unsigned *routes,
                                          unsigned qps, const unsigned long long *epochs, unsigned owner,
                                          unsigned layer) {
    if (routes[owner])
        return;
    unsigned nr = totals[layer * world + owner], base = layer * capacity + bases[layer * world + owner];
    if (!nr)
        return; // 空owner不需要CPU服务、数据WRITE或DONE，也不推进邮箱序号。
    size_t slot = (size_t(rank) * layers + layer) * slot_bytes;
    if (owner / local == rank / local) {
        auto *dst = reinterpret_cast<unsigned *>(aliases[owner % local] + slot + 64);
        for (unsigned i = threadIdx.x; i < nr; i += blockDim.x)
            dst[i] = packed[base + i];
        __threadfence_system();
        __syncthreads();
        if (!threadIdx.x) {
            auto *h = reinterpret_cast<RequestHeader *>(aliases[owner % local] + slot);
            h->rows = nr;
            h->base = base;
            h->active_qps = qps;
            // 请求发布点：CPU acquire看到epoch后，才允许读取metadata和行号。
            asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(&h->ready), "l"(epochs[layer * world + owner])
                         : "memory");
        }
    } else if (!threadIdx.x) {
        const auto &comm = *comm_ptr;
        // 走GIN检测出的rail连接；target窗口选实际owner，代表rank无需CPU转发。
        unsigned representative = owner / local * local + rank % local;
        ncclGin gin(comm, owner % comm.ginContextCount, NCCL_GIN_RESOURCE_SHARING_GPU);
        auto team = ncclTeamWorld(comm);
        auto target = targets[owner % local];
        gin.putValue(team, representative, target, slot + 8, uint64_t(nr) | (uint64_t(base) << 32));
        gin.putValue(team, representative, target, slot + 16, qps);
        // metadata先发、行号payload后发；STRICT_ORDERING窗口配合StrongVASignalAdd，
        // 使CPU看到ready增加时能读取本轮完整请求。每个[layer,owner]的序号独立；
        // 空请求不发signal，CPU按槽轮询，GPU hook也跳过该槽。
        gin.put(team, representative, target, slot + 64, source, size_t(base) * 4, size_t(nr) * 4,
                ncclGin_StrongVASignalAdd{target, slot, 1}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
                cuda::thread_scope_thread, cuda::thread_scope_system, ncclGinOptFlagsDefault, ncclGin_SegmentMixed{});
    }
}
extern "C" __global__ void send_requests(const ncclDevComm_t *comm, ncclWindow_t source, const ncclWindow_t *targets,
                                         const unsigned long long *aliases, const unsigned *packed,
                                         const unsigned *totals, const unsigned *bases, unsigned rank, unsigned world,
                                         unsigned local, unsigned layers, unsigned capacity,
                                         unsigned long long slot_bytes, const unsigned *routes, unsigned qps,
                                         const unsigned long long *epochs) {
    send_request(comm, source, targets, aliases, packed, totals, bases, rank, world, local, layers, capacity,
                 slot_bytes, routes, qps, epochs, blockIdx.x, blockIdx.y);
}

extern "C" __global__ void gather_local(const int *input, unsigned long long *output, const unsigned long long *aliases,
                                        const unsigned *entries, const unsigned *table_bases, unsigned rows,
                                        unsigned capacity, unsigned row_words, unsigned rank, unsigned local,
                                        const unsigned *routes) {
    unsigned layer = blockIdx.y, n = entries[layer];
    size_t words = size_t(rows) * row_words;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) {
        unsigned at = i / row_words;
        int x = input[size_t(layer) * rows + at];
        size_t dst = size_t(layer) * capacity * row_words + i;
        if (x < 0)
            output[dst] = 0;
        else if (routes[unsigned(x) / n]) {
            auto *table = reinterpret_cast<const unsigned long long *>(aliases[unsigned(x) / n]);
            output[dst] = __ldg(table + size_t(table_bases[layer] + unsigned(x) % n) * row_words + i % row_words);
        }
    }
}
extern "C" __global__ void gather_scales(const int *input, float *output, const unsigned long long *shards,
                                         const unsigned *entries, const unsigned *sf_bases, unsigned tokens,
                                         unsigned ept, unsigned packs, unsigned owners_per_shard, unsigned token_stride,
                                         unsigned hidden_stride, unsigned layer_stride) {
    unsigned layer = blockIdx.y, rows = tokens * ept, n = entries[layer];
    for (unsigned at = blockIdx.x * blockDim.x + threadIdx.x; at < rows; at += gridDim.x * blockDim.x) {
        int x = input[size_t(layer) * rows + at];
        unsigned token = at / ept, slot = at % ept;
        for (unsigned p = 0; p < packs; ++p) {
            float value = 0;
            if (x >= 0) {
                auto *shard = reinterpret_cast<const float *>(shards[unsigned(x) / (n * owners_per_shard)]);
                value = __ldg(shard + sf_bases[layer] + size_t(unsigned(x) % (n * owners_per_shard)) * packs + p);
            }
            output[size_t(layer) * layer_stride + token * token_stride + (slot * packs + p) * hidden_stride] = value;
        }
    }
}

// 普通kernel融合本地gather/scales和远端行数统计，不使用全网格屏障。
// Graph中：issue_egm -> pack_send_egm -> 调用方独立计算 -> 各层finish。
// 前两个节点仅发布请求，不等待CPU服务或DONE；纯机内时第二个节点快速返回。
extern "C" __global__ __launch_bounds__(512, 2) void issue_egm(
    const int *input, int4 *output, const unsigned long long *table_aliases, float *sf_output,
    const unsigned long long *sf_aliases, const unsigned *entries, const unsigned *table_bases,
    const unsigned *sf_bases, unsigned rows, unsigned capacity, unsigned vectors, unsigned layers, unsigned ept,
    unsigned packs, unsigned owners_per_sf_shard, unsigned sf_token_stride, unsigned sf_hidden_stride,
    unsigned sf_layer_stride, unsigned rank, unsigned world, unsigned local, unsigned tiles, unsigned *counters,
    unsigned *counts, unsigned *totals, const unsigned *routes) {
    if (!blockIdx.x) {
        for (unsigned i = threadIdx.x; i < layers; i += blockDim.x)
            counters[i] = 0;
        for (unsigned i = threadIdx.x; i < layers * world; i += blockDim.x)
            totals[i] = 0;
    }
    for (unsigned task = blockIdx.x; task < layers * tiles; task += gridDim.x)
        count_tile(input, counts, rows, world, entries, rank, local, routes, task % tiles, task / tiles, tiles);

    // 与READ相同：16个lane合作搬一行，每lane一次16B加载，直接写最终output。
    // 词表仍在owner的CPU NUMA；SM通过CUDA VMM别名读取，数据经owner C2C及GPU NVLink传入。
    // FP8 scales默认读取GPU shard的IPC别名；LOCAL_SF=1时读取本GPU完整副本。
    // FP8让最后2个warp独立收集scales。不能让搬词表的half-warp先读CPU行，
    // 再等一次GPU peer scales读取后才处理下一行，否则两种访存延迟会串行叠加。
    unsigned copy_threads = packs ? 448 : 512, lane = threadIdx.x % 16;
    for (unsigned layer = 0; layer < layers; ++layer) {
        unsigned n = entries[layer];
        if (threadIdx.x < copy_threads)
            for (unsigned at = (blockIdx.x * copy_threads + threadIdx.x) / 16; at < rows;
                 at += gridDim.x * (copy_threads / 16)) {
                int x = input[size_t(layer) * rows + at];
                unsigned owner = x >= 0 ? unsigned(x) / n : world;
                if (x >= 0 && owner >= world)
                    asm volatile("trap;");
                if (x < 0 || routes[owner]) {
                    auto *src = x < 0 ? nullptr
                                      : reinterpret_cast<const int4 *>(table_aliases[owner]) +
                                            size_t(table_bases[layer] + unsigned(x) % n) * vectors;
                    auto *dst = output + (size_t(layer) * capacity + at) * vectors;
                    for (unsigned v = lane; v < vectors; v += 16)
                        dst[v] = src ? __ldg(src + v) : make_int4(0, 0, 0, 0);
                }
            }
        else if (packs)
            for (unsigned at = blockIdx.x * (512 - copy_threads) + threadIdx.x - copy_threads; at < rows;
                 at += gridDim.x * (512 - copy_threads)) {
                int x = input[size_t(layer) * rows + at];
                for (unsigned p = 0; p < packs; ++p) {
                    float value = 0;
                    if (x >= 0) {
                        auto *sf = reinterpret_cast<const float *>(sf_aliases[unsigned(x) / (n * owners_per_sf_shard)]);
                        value =
                            __ldg(sf + sf_bases[layer] + size_t(unsigned(x) % (n * owners_per_sf_shard)) * packs + p);
                    }
                    sf_output[size_t(layer) * sf_layer_stride + (at / ept) * sf_token_stride +
                              ((at % ept) * packs + p) * sf_hidden_stride] = value;
                }
            }
    }
}

// 每个CTA从只读counts独立计算前缀，写入各自tile的packed/positions。
// 末尾使用“最后一个CTA”发布请求：其他CTA做完就退出，不会占着SM自旋等未调度CTA。
// 与合作网格不同，这个普通kernel不要求全部CTA同时驻留。
extern "C" __global__ __launch_bounds__(256, 1) void pack_send_egm(
    const int *input, unsigned *request, unsigned *positions, const unsigned *counts, unsigned *totals, unsigned *bases,
    unsigned long long *epochs, unsigned *counters, const unsigned *entries, const unsigned *table_bases, unsigned rows,
    unsigned capacity, unsigned rank, unsigned world, unsigned local, unsigned layers, unsigned tiles,
    const ncclDevComm_t *comm, ncclWindow_t source, const ncclWindow_t *targets,
    const unsigned long long *inbox_aliases, unsigned long long slot_bytes, unsigned qps, const unsigned *routes) {
    __shared__ unsigned sums[64], local_bases[64], tile_offsets[64], remote_rows;
    __shared__ bool last;
    unsigned layer = blockIdx.y, owner = threadIdx.x;
    if (owner < world) {
        unsigned sum = 0;
        for (unsigned b = 0; b < tiles; ++b)
            sum += counts[(layer * world + owner) * tiles + b];
        sums[owner] = sum;
    }
    __syncthreads();
    if (!owner) {
        unsigned sum = 0;
        for (unsigned o = 0; o < world; ++o) {
            local_bases[o] = sum;
            sum += sums[o];
        }
        remote_rows = sum;
    }
    __syncthreads();
    if (!remote_rows)
        return; // 纯机内：totals由前一节点清0，hook直接完成。
    if (!blockIdx.x && owner < world) {
        totals[layer * world + owner] = sums[owner];
        bases[layer * world + owner] = local_bases[owner];
    }
    for (unsigned tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        if (owner < world) {
            unsigned sum = 0;
            for (unsigned b = 0; b < tile; ++b)
                sum += counts[(layer * world + owner) * tiles + b];
            tile_offsets[owner] = sum;
        }
        __syncthreads();
        pack_tile<8, true>(input, request, positions, tile_offsets, local_bases, entries, table_bases, rows, capacity,
                           world, rank, local, routes, tile, layer, tiles);
    }
    __threadfence();
    __syncthreads(); // 所有搬运行号/位置线程先release本CTA的写入。
    if (!owner) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(counters[layer]);
        // acq_rel的RMW形成release序列，最后一个CTA acquire其他CTA的全部写入。
        last = counter.fetch_add(1, cuda::memory_order_acq_rel) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last)
        return;
    if (owner < world && sums[owner])
        ++epochs[layer * world + owner];
    __syncthreads(); // 邮箱发布前，本层所有非空owner序号均已就绪。
    for (unsigned o = 0; o < world; ++o)
        send_request(comm, source, targets, inbox_aliases, request, totals, bases, rank, world, local, layers, capacity,
                     slot_bytes, routes, qps, epochs, o, layer);
}

// 每个CTA先独立acquire本层所需QP的DONE，再搬自己负责的结果片段。
// 纯机内totals全0，hook仍维持stream依赖，但无需等待远端或执行空scatter网格。
extern "C" __global__ void finish_layer(const unsigned long long *flags, const unsigned long long *epochs,
                                        const unsigned *totals, const unsigned long long *packed,
                                        unsigned long long *output, const unsigned *positions, unsigned layer,
                                        unsigned world, unsigned capacity, unsigned row_words, unsigned allocated_qps,
                                        unsigned active_qps, unsigned long long timeout) {
    __shared__ unsigned rows;
    if (!threadIdx.x) {
        rows = 0;
        for (unsigned owner = 0; owner < world; ++owner)
            rows += totals[layer * world + owner];
    }
    __syncthreads();
    if (!rows)
        return;
    for (unsigned i = threadIdx.x; i < world * allocated_qps; i += blockDim.x) {
        unsigned owner = i / allocated_qps, q = i % allocated_qps;
        unsigned nr = totals[layer * world + owner], used = min(active_qps, (nr + kMerge - 1) / kMerge);
        if (q >= used)
            continue;
        auto *p = flags + (size_t(layer) * world + owner) * allocated_qps + q;
        unsigned long long expected = epochs[layer * world + owner], start = clock64(), value;
        do {
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(p) : "memory");
            if (value != expected)
                __nanosleep(128);
            if (clock64() - start > timeout)
                asm volatile("trap;");
        } while (value != expected);
    }
    __syncthreads(); // 每个搬运线程都必须在本CTA完成DONE acquire之后读取RDMA输出。
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(rows) * row_words;
         i += size_t(gridDim.x) * blockDim.x) {
        size_t src = size_t(layer) * capacity * row_words + i;
        unsigned pos = positions[size_t(layer) * capacity + i / row_words];
        output[(size_t(layer) * capacity + pos) * row_words + i % row_words] = packed[src];
    }
}
