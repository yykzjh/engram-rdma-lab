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
            local_rows[packed]=(layer?524288:0)+x%n;positions[packed]=at;}
        __syncthreads();
    }
}
__device__ __forceinline__ void send_requests_body(ncclDevComm_t comm, ncclWindow_t source,
        const ncclWindow_t* targets, const unsigned long long* local_aliases,
        const unsigned* local_rows, const unsigned* totals, const unsigned* bases,
        unsigned rank, unsigned long long slot_bytes,unsigned hybrid, unsigned long long epoch) {
    unsigned owner = blockIdx.x;
    if(hybrid && owner/4==rank/4)return;
    unsigned long long slot = rank * slot_bytes;
    if (owner / 4 == rank / 4) {
        auto* destination = reinterpret_cast<volatile unsigned*>(local_aliases[owner%4] + slot + 64);
        for (unsigned i=threadIdx.x;i<totals[owner];i+=blockDim.x) destination[i] = local_rows[bases[owner]+i];
        __threadfence_system(); __syncthreads();
        if (!threadIdx.x) {
            auto* metadata = reinterpret_cast<volatile unsigned long long*>(local_aliases[owner%4] + slot + 8);
            *metadata = uint64_t(totals[owner]) | (uint64_t(bases[owner]) << 32);
            auto* ready = reinterpret_cast<unsigned long long*>(local_aliases[owner%4] + slot);
            asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(ready), "l"(epoch) : "memory");
        }
    } else if (!threadIdx.x) {
        // GIN supports rail connectivity on this system. Every rank registers
        // all four local owner inboxes, so a rail peer can target ANY owner.
        ncclWindow_t target = targets[owner%4];
        unsigned representative = owner/4*4 + rank%4;
        ncclGin gin(comm, owner % comm.ginContextCount, NCCL_GIN_RESOURCE_SHARING_GPU);
        ncclTeam world = ncclTeamWorld(comm);
        uint64_t metadata = uint64_t(totals[owner]) | (uint64_t(bases[owner]) << 32);
        gin.putValue(world, representative, target, slot + 8, metadata);
        if (totals[owner]) {
            gin.put(world, representative, target, slot + 64, source, size_t(bases[owner])*4,
                    size_t(totals[owner])*4, ncclGin_StrongVASignalAdd{target, size_t(slot), 1},
                    ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
                    cuda::thread_scope_system, ncclGinOptFlagsDefault, ncclGin_SegmentMixed{});
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
    unsigned i=threadIdx.x;
    if(i<8*qps && (!hybrid || (i/qps)/4!=rank/4)){
        unsigned active=compact?max(1u,min(qps,(totals[i/qps]+merge_rows-1)/merge_rows)):qps;
        if(i%qps<active){
        const unsigned long long* p=flags+(i/qps)*16+i%qps;
        unsigned long long value,expected=*epoch,start=clock64();
        do{
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];":"=l"(value):"l"(p):"memory");
            if(value!=expected)__nanosleep(128);
            if(clock64()-start>100000000000ULL)asm volatile("trap;");
        }while(value!=expected);
        }
    }
    __syncthreads();
}
extern "C" __global__ void gather_local_cpu_rows(const unsigned* input,
        unsigned long long* output,const unsigned long long* aliases,unsigned rows,unsigned tokens,unsigned rank){
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
    __shared__ unsigned long long words;
    if(!threadIdx.x){words=0;for(int i=0;i<8;++i)words+=size_t(totals[i])*32;}
    __syncthreads();
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=(unsigned long long)gridDim.x*blockDim.x)
        output[size_t(positions[i/32])*32+i%32]=input[i];
}
