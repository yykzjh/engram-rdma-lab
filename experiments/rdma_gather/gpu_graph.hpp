#pragma once
// Buffer-owned, shape-dependent graph. All routing still runs on THIS fetch's
// raw GPU input; no index-dependent requests or permutations are captured.
class GpuFetchGraph {
    CUgraph graph=nullptr;
    CUgraphExec executable=nullptr;
    CUstream stream=nullptr;
    CUdeviceptr epoch=0;
    CUgraphNode kernel(CUfunction fn,unsigned grid,unsigned block,void** args,
                       std::initializer_list<CUgraphNode> dependencies={}) {
        CUDA_KERNEL_NODE_PARAMS p{};p.func=fn;p.gridDimX=grid;p.gridDimY=p.gridDimZ=1;
        p.blockDimX=block;p.blockDimY=p.blockDimZ=1;p.kernelParams=args;
        CUgraphNode node;
        cuda_check(cuGraphAddKernelNode(&node,graph,dependencies.begin(),dependencies.size(),&p),"add fetch graph kernel");
        return node;
    }
public:
    GpuFetchGraph(Gpu& gpu,GpuRouter& router,Inbox& inbox,const Options& o,CUevent route_start,CUevent route_end,CUdeviceptr table_aliases){
        cuda_check(cuMemAlloc(&epoch,8),"graph fetch epoch");
        cuda_check(cuMemsetD8(epoch,0,8),"initialize graph epoch");
        cuda_check(cuGraphCreate(&graph,0),"create fetch graph");
        cuda_check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"fetch graph stream");
        if(o.gpu_get){
            unsigned rows=frame_bytes(o)/ROW,tokens=o.groups,rank=o.rank;
            CUfunction issue,finish,local;
            cuda_check(cuModuleGetFunction(&issue,router.module,o.get_batch?"issue_batched_gets":"issue_direct_gets"),"GET issue kernel");
            cuda_check(cuModuleGetFunction(&finish,router.module,"finish_direct_gets"),"GET completion kernel");
            cuda_check(cuModuleGetFunction(&local,router.module,"gather_local_cpu_rows"),"GET local gather kernel");
            void* args[]={&router.dev,&gpu.input,&router.get_sources,&router.get_output,&rows,&tokens,&rank};
            auto in=kernel(issue,((o.get_batch?router.dev.ginContextCount:rows)+127)/128,128,args);
            void* finish_args[]={&router.dev};kernel(finish,1,128,finish_args,{in});
            uint64_t words=gpu.bytes/8;
            void* local_args[]={&gpu.input,&gpu.output,&table_aliases,&rows,&tokens,&rank};
            kernel(local,unsigned(std::min(uint64_t(4096),(words+255)/256)),256,local_args);
            void* scale_args[]={&gpu.input,&gpu.sf_output,&gpu.sf_peers[0],&gpu.sf_peers[1],&gpu.sf_peers[2],&gpu.sf_peers[3],&tokens};
            kernel(gpu.gather_sf,std::min(512U,(tokens*48+255)/256),256,scale_args);
            cuda_check(cuGraphInstantiateWithFlags(&executable,graph,0),"instantiate direct GET graph");
            return;
        }
        CUfunction begin,send,wait;
        cuda_check(cuModuleGetFunction(&begin,router.module,"begin_fetch_epoch"),"graph epoch kernel");
        cuda_check(cuModuleGetFunction(&send,router.module,"send_gpu_requests_epoch"),"graph send kernel");
        cuda_check(cuModuleGetFunction(&wait,router.module,"wait_gpu_epoch"),"graph wait kernel");
        void* begin_args[]={&epoch};auto en=kernel(begin,1,1,begin_args);
        CUgraphNode a,b;
        if(o.profile_routes)cuda_check(cuGraphAddEventRecordNode(&a,graph,nullptr,0,route_start),"graph route start event");
        unsigned rows=frame_bytes(o)/ROW,tokens=o.groups,blocks=router.blocks,rank=o.rank,qps=o.qps,hybrid=o.hybrid_local,compact=o.compact_done,merge=o.merge_rows;
        CUdeviceptr packed=reinterpret_cast<CUdeviceptr>(router.packed);uint64_t stride=inbox.slot_bytes;
        void* count_args[]={&gpu.input,&router.counts,&rows,&tokens,&rank,&hybrid};
        auto cn=o.profile_routes?kernel(router.count,blocks,256,count_args,{a}):kernel(router.count,blocks,256,count_args);
        void* prefix_args[]={&router.counts,&router.offsets,&router.totals,&router.bases,&blocks};
        auto pn=kernel(router.prefix,1,32,prefix_args,{cn});
        void* pack_args[]={&gpu.input,&packed,&gpu.positions,&router.offsets,&router.bases,&rows,&tokens,&rank,&hybrid};
        auto kn=kernel(router.pack,blocks,256,pack_args,{pn});
        if(o.profile_routes)cuda_check(cuGraphAddEventRecordNode(&b,graph,&kn,1,route_end),"graph route end event");
        else b=kn;
        void* send_args[]={&router.dev,&router.source,&router.target_windows,&router.alias_pointers,
            &router.indirect_mode,&packed,&router.totals,&router.bases,&rank,&stride,&hybrid,&epoch};
        auto sn=kernel(send,8,256,send_args,{b,en});
        void* wait_args[]={&gpu.done_flags,&qps,&epoch,&router.totals,&merge,&compact,&rank,&hybrid};auto wn=kernel(wait,1,256,wait_args,{sn});
        uint64_t words=gpu.bytes/8;
        void* scatter_args[]={&gpu.packed,&gpu.output,&gpu.positions,&words};
        unsigned scatter_grid=o.adaptive_grid?unsigned(std::min(uint64_t(4096),(words+255)/256)):4096;
        if(o.hybrid_local){
            CUfunction local,remote;
            cuda_check(cuModuleGetFunction(&local,router.module,"gather_local_cpu_rows"),"local CPU gather kernel");
            cuda_check(cuModuleGetFunction(&remote,router.module,"scatter_remote_rows"),"remote scatter kernel");
            void* remote_args[]={&gpu.packed,&gpu.output,&gpu.positions,&router.totals};
            kernel(remote,scatter_grid,256,remote_args,{wn});
            void* local_args[]={&gpu.input,&gpu.output,&table_aliases,&rows,&tokens,&rank};
            kernel(local,scatter_grid,256,local_args,{sn});
        }else kernel(gpu.scatter,scatter_grid,256,scatter_args,{wn});
        void* scale_args[]={&gpu.input,&gpu.sf_output,&gpu.sf_peers[0],&gpu.sf_peers[1],&gpu.sf_peers[2],&gpu.sf_peers[3],&tokens};
        kernel(gpu.gather_sf,o.adaptive_grid?std::min(512U,(tokens*48+255)/256):512,256,scale_args);
        cuda_check(cuGraphInstantiateWithFlags(&executable,graph,0),"instantiate fetch graph");
    }
    double run(){
        auto t=Clock::now();
        cuda_check(cuGraphLaunch(executable,stream),"launch fetch graph");
        cuda_check(cuStreamSynchronize(stream),"complete fetch graph");
        return us(t)/1000;
    }
    ~GpuFetchGraph(){
        if(executable)cuGraphExecDestroy(executable);
        if(graph)cuGraphDestroy(graph);
        if(stream)cuStreamDestroy(stream);
        if(epoch)cuMemFree(epoch);
    }
};
