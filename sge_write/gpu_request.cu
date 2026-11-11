#include <cuda_runtime.h>
#include <nccl_device.h>
#include <stdint.h>
constexpr unsigned TILE = 4096;
extern "C" __global__ void debug_gin(ncclDevComm_t comm) {
    if (threadIdx.x || blockIdx.x) return;
    printf("GIN_DEBUG rank=%d ranks=%d stride=%d connections=%u contexts=%u qp_size=%llu\n",
        comm.rank,comm.nRanks,comm.ginConnectionStride,unsigned(comm.ginConnectionCount),
        comm.ginContextCount,(unsigned long long)sizeof(doca_gpu_dev_verbs_qp));
    for(unsigned i=0;i<comm.ginConnectionCount;++i){
        auto* gdaki=static_cast<ncclGinGdakiGPUContext*>(comm.ginHandles[i]);
        auto* qps=gdaki[0].gdqp;
        for(int p=0;p<comm.nRanks/comm.ginConnectionStride;++p){auto& q=qps[p];
            printf("GIN_QP connection=%u peer=%d count=%u reserved=%llu wq=%p cq=%p\n",
                i,p,unsigned(q.sq_wqe_num),(unsigned long long)q.sq_rsvd_index,
                q.sq_wqe_daddr,q.cq_sq.cqe_daddr);
        }
    }
}
extern "C" __global__ void route_counts(const unsigned* input, unsigned* counts, unsigned rows, unsigned tokens,unsigned rank,unsigned hybrid) {
    // 输入按 [layer, token, slot] 展平。每个 block 处理最多 TILE=4096 行，
    // 输出 counts[owner, block]。C2C 本地行直接 gather，不进入远端请求列表。
    __shared__ unsigned bins[8];
    if (threadIdx.x < 8) bins[threadIdx.x] = 0; __syncthreads();
    unsigned lo = blockIdx.x * TILE, hi = min(lo + TILE, rows);
    for (unsigned at = lo + threadIdx.x; at < hi; at += blockDim.x) {
        unsigned n = at < tokens * 24 ? 524288 : 524309, owner = input[at] / n;
        if (owner >= 8) asm volatile("trap;");
        if(hybrid && owner/4==rank/4)continue;
        atomicAdd(bins + owner, 1);
    }
    __syncthreads(); if (threadIdx.x < 8) counts[threadIdx.x * gridDim.x + blockIdx.x] = bins[threadIdx.x];
}
extern "C" __global__ void route_prefix(const unsigned* counts, unsigned* offsets, unsigned* totals,
                                        unsigned* bases, unsigned blocks) {
    // offsets[owner, block]：该 block 在本 owner 行列表中的起点；
    // totals[owner]：该 owner 本轮总行数；bases[owner]：owner-major staging 起点。
    // 这些前缀只依赖当前输入，必须每次 fetch 重新计算。
    unsigned owner = threadIdx.x;
    if (owner < 8) {
        unsigned sum = 0;
        for (unsigned block = 0; block < blocks; ++block) {
            offsets[owner * blocks + block] = sum; sum += counts[owner * blocks + block];
        }
        totals[owner] = sum;
    }
    __syncthreads();
    if (!owner) { unsigned sum = 0; for (unsigned i=0;i<8;++i) { bases[i]=sum; sum+=totals[i]; } }
}
extern "C" __global__ void route_pack(const unsigned* input, unsigned* local_rows, unsigned* positions,
        const unsigned* offsets, const unsigned* bases, unsigned rows, unsigned tokens,unsigned rank,unsigned hybrid) {
    // warp 内先按 owner 聚合，再结合 block/owner 前缀压紧，无需 CPU 下载索引。
    // local_rows 是发给 owner 的本地行号；positions 留在 GPU 用于最后重排。
    __shared__ unsigned counts[8*8], warp_bases[8*8], consumed[8];
    if (threadIdx.x < 8) consumed[threadIdx.x]=0; __syncthreads();
    unsigned lo=blockIdx.x*TILE, hi=min(lo+TILE,rows), lane=threadIdx.x%32, warp=threadIdx.x/32;
    for(unsigned step=0;step<hi-lo;step+=blockDim.x){
        if(threadIdx.x<64)counts[threadIdx.x]=0; __syncthreads();
        unsigned at=lo+step+threadIdx.x, layer=at>=tokens*24, n=layer?524309:524288;
        bool active=at<hi;unsigned x=active?input[at]:0,owner=active?x/n:8;
        if(active&&owner>=8)asm volatile("trap;");
        active=active && (!hybrid || owner/4!=rank/4);
        if(!active)owner=8;
        unsigned mask=__match_any_sync(0xffffffff,owner);
        if(active&&lane==unsigned(__ffs(mask)-1))counts[owner*8+warp]=__popc(mask);
        __syncthreads();
        if(threadIdx.x<8){unsigned owner_id=threadIdx.x,sum=consumed[owner_id];
            for(unsigned w=0;w<8;++w){warp_bases[owner_id*8+w]=sum;sum+=counts[owner_id*8+w];}consumed[owner_id]=sum;}
        __syncthreads();
        if(active){unsigned at_owner=warp_bases[owner*8+warp]+__popc(mask&((1u<<lane)-1));
            unsigned packed=bases[owner]+offsets[owner*gridDim.x+blockIdx.x]+at_owner;
            // owner 的两层词表连续存储；第二层行号加上第一层 524288 行。
            // positions[packed]=at 把 owner-major 接收位置对应回 token/slot 顺序。
            local_rows[packed]=(layer?524288:0)+x%n;positions[packed]=at;}
        __syncthreads();
    }
}
__device__ __forceinline__ void send_requests_body(ncclDevComm_t comm, ncclWindow_t source,
        const ncclWindow_t* targets, const unsigned long long* local_aliases,
        const unsigned* local_rows, const unsigned* totals, const unsigned* bases,
        unsigned rank, unsigned long long slot_bytes,unsigned hybrid, unsigned long long epoch) {
    // 8 个 block 各负责一个 owner。每个 owner 中只访问 requester=rank 的槽。
    // 请求格式：slot+0 ready，slot+8 {base:32,nr:32}，slot+64 uint32行号列表。
    // base 是 GPU staging 的行偏移，不是最终输出偏移；目的地址/rkey 已在初始化交换。
    unsigned owner = blockIdx.x;
    if(hybrid && owner/4==rank/4)return;
    unsigned long long slot = rank * slot_bytes;
    if (owner / 4 == rank / 4) {
        // C2C 关闭时，同机也由 CPU owner 服务，但请求直接写 CPU 邮箱映射。
        // 所有线程先写行号并 system fence，block 同步后才由线程0发布 metadata/ready。
        auto* destination = reinterpret_cast<volatile unsigned*>(local_aliases[owner%4] + slot + 64);
        for (unsigned i=threadIdx.x;i<totals[owner];i+=blockDim.x) destination[i] = local_rows[bases[owner]+i];
        __threadfence_system(); __syncthreads();
        if (!threadIdx.x) {
            auto* metadata = reinterpret_cast<volatile unsigned long long*>(local_aliases[owner%4] + slot + 8);
            *metadata = uint64_t(totals[owner]) | (uint64_t(bases[owner]) << 32);
            auto* ready = reinterpret_cast<unsigned long long*>(local_aliases[owner%4] + slot);
            // release 是“请求发布点”；CPU acquire 观察到本轮 epoch 后才能读请求。
            asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(ready), "l"(epoch) : "memory");
        }
    } else if (!threadIdx.x) {
        // GIN supports rail connectivity on this system. Every rank registers
        // all four local owner inboxes, so a rail peer can target ANY owner.
        ncclWindow_t target = targets[owner%4];
        unsigned representative = owner/4*4 + rank%4;
        // representative 保持 requester 的 local-rank，即走已建立的 rail 连接；
        // targets[owner%4] 则选择实际 owner 的邮箱。representative CPU 不转发请求。
        ncclGin gin(comm, owner % comm.ginContextCount, NCCL_GIN_RESOURCE_SHARING_GPU);
        ncclTeam world = ncclTeamWorld(comm);
        uint64_t metadata = uint64_t(totals[owner]) | (uint64_t(bases[owner]) << 32);
        gin.putValue(world, representative, target, slot + 8, metadata);
        if (totals[owner]) {
            // 行号 payload 完成后用 strong VA signal 将 owner ready 累加1。
            // 同一 requester 每轮只发一次请求，远端 ready 从0累加为1、2、3...；
            // 与 GPU epoch、owner 服务线程的预期 epoch 一一对应。
            // metadata 的 putValue 与后续 strong signal 使用同一 GIN context。
            gin.put(world, representative, target, slot + 64, source, size_t(bases[owner])*4,
                    size_t(totals[owner])*4, ncclGin_StrongVASignalAdd{target, size_t(slot), 1},
                    ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
                    cuda::thread_scope_system, ncclGinOptFlagsDefault, ncclGin_SegmentMixed{});
        // 空组也发布 ready：owner 仍回写 DONE，避免双方轮次失配或 GPU 永久等待。
        } else gin.signal(world, representative, ncclGin_StrongVASignalAdd{target, size_t(slot), 1});
    }
}

extern "C" __global__ void send_gpu_requests(ncclDevComm_t comm,ncclWindow_t source,
        const ncclWindow_t* targets,const unsigned long long* aliases,
        const unsigned* rows,const unsigned* totals,const unsigned* bases,unsigned rank,
        unsigned long long stride,unsigned hybrid,unsigned long long epoch){
    send_requests_body(comm,source,targets,aliases,rows,totals,bases,rank,stride,hybrid,epoch);
}
extern "C" __global__ void begin_fetch_epoch(unsigned long long* epoch){
    // 序号留在 GPU，每次 Graph 执行自增，无需主机为下一帧修改 Graph 参数。
    if(!threadIdx.x && !blockIdx.x)++*epoch;
}
extern "C" __global__ void send_gpu_requests_epoch(ncclDevComm_t comm,ncclWindow_t source,
        const ncclWindow_t* targets,const unsigned long long* aliases,
        const unsigned* rows,const unsigned* totals,const unsigned* bases,unsigned rank,
        unsigned long long stride,unsigned hybrid,const unsigned long long* epoch){
    send_requests_body(comm,source,targets,aliases,rows,totals,bases,rank,stride,hybrid,*epoch);
}
extern "C" __global__ void wait_gpu_epoch(const unsigned long long* flags,unsigned qps,
        const unsigned long long* epoch,const unsigned* totals,unsigned merge_rows,unsigned compact,unsigned rank,unsigned hybrid){
    // 一个 block 等所有 owner/QP。线程 i 对应 owner=i/qps、qp=i%qps。
    // 每个 owner 实际只用前 min(qps,ceil(nr/merge_rows)) 个 QP；空请求至少等QP0。
    // C2C 本地 owner 不走 WRITE，也不等待它们的 DONE。
    unsigned i=threadIdx.x;
    if(i<8*qps && (!hybrid || (i/qps)/4!=rank/4)){
        unsigned active=compact?max(1u,min(qps,(totals[i/qps]+merge_rows-1)/merge_rows)):qps;
        if(i%qps<active){
        const unsigned long long* p=flags+(i/qps)*16+i%qps;
        unsigned long long value,expected=*epoch,start=clock64();
        do{
            // 同时保证“读到本轮序号”与 system acquire。数据可见性的另一前提是
            // 初始化已检查 GPUDirect 原生 OWNER ordering；仅 volatile 轮询不够。
            // GIN 的请求 ready 不覆盖 verbs 数据完成，必须独立等这些 per-QP DONE。
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];":"=l"(value):"l"(p):"memory");
            if(value!=expected)__nanosleep(128);
            if(clock64()-start>100000000000ULL)asm volatile("trap;");
        }while(value!=expected);
        }
    }
    // 所有等待线程退出后 kernel 才能结束；Graph 的重排节点依赖此节点完成。
    __syncthreads();
}
extern "C" __global__ void gather_local_cpu_rows(const unsigned* input,
        unsigned long long* output,const unsigned long long* aliases,unsigned rows,unsigned tokens,unsigned rank){
    // C2C 路径直接读同机 owner CPU 词表，写最终 token-major 输出，跳过 staging。
    // 本地槽位与远端 scatter 写入槽位互斥，因此两者可在 Graph 中并行。
    unsigned long long words=(unsigned long long)rows*32;
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=(unsigned long long)gridDim.x*blockDim.x){
        unsigned at=i/32,layer=at>=tokens*24,n=layer?524309:524288,x=input[at],owner=x/n;
        if(owner/4==rank/4){
            auto* table=reinterpret_cast<const unsigned long long*>(aliases[owner%4]);
            output[i]=__ldg(table+(size_t(layer?524288:0)+x%n)*32+i%32);
        }
    }
}
extern "C" __global__ void scatter_remote_rows(const unsigned long long* input,
        unsigned long long* output,const unsigned* positions,const unsigned* totals){
    // 仅搬运本轮远端总行数，而非整个 staging 容量。每行 32 个uint64=256B，
    // positions 将合并接收的行恢复到原始 [layer,token,slot] 位置。
    __shared__ unsigned long long words;
    if(!threadIdx.x){words=0;for(int i=0;i<8;++i)words+=size_t(totals[i])*32;}
    __syncthreads();
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=(unsigned long long)gridDim.x*blockDim.x)
        output[size_t(positions[i/32])*32+i%32]=input[i];
}

// Application-side GPU producer/consumer used ONLY by the asynchronous test.
// They deliberately change every frame's raw indices; no CPU input download or
// per-frame host fence participates in this producer -> fetch -> consumer chain.
extern "C" __global__ void async_producer(unsigned* input,unsigned tokens,unsigned rank,unsigned epoch,unsigned uniform) {
    unsigned rows=tokens*48;
    for(unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<rows;at+=gridDim.x*blockDim.x){
        unsigned n=at<tokens*24?524288:524309,global=n*8,interval=global/24;
        unsigned x=at+rank*1009+epoch*97531+12345;x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;
        input[at]=uniform?x%global:(at%24)*interval+x%interval;
    }
}
extern "C" __global__ void async_consumer(const unsigned* indices,const unsigned long long* output,
        const unsigned* scales,unsigned tokens,unsigned epoch,unsigned long long* verdicts,unsigned frame) {
    unsigned rows=tokens*48;
    for(unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<rows;at+=gridDim.x*blockDim.x){
        unsigned layer=at>=tokens*24,n=layer?524309:524288,x=indices[at];
        unsigned tag=(x/n)*(524288+524309)+(layer?524288:0)+x%n;
        unsigned bad=0;
        for(unsigned word=0;word<32;++word){unsigned long long expected=0x9e3779b97f4a7c15ULL*(tag+1ULL)^0xd1b54a32d192ed03ULL*(word+1ULL);bad+=output[size_t(at)*32+word]!=expected;}
        unsigned token=(at%(tokens*24))/24,slot=at%24;
        unsigned a=0x3f800000U|((x+layer*17)&0x7fffffU),b=0x40000000U|((x*3+layer*29)&0x7fffffU);
        bad+=scales[layer*tokens*48+token+slot*2*tokens]!=a;
        bad+=scales[layer*tokens*48+token+(slot*2+1)*tokens]!=b;
        if(bad)atomicAdd(verdicts+frame*2,(unsigned long long)bad);
    }
    if(!blockIdx.x&&!threadIdx.x)verdicts[frame*2+1]=epoch;
}
