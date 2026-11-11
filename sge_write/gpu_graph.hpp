#pragma once
// 按Buffer容量/shape建立一次Graph。捕获的是kernel、固定地址与节点依赖，
// 不是某轮索引内容；每次执行仍读取当前gpu.input，重新路由/打包/计算位置映射。
//
// 依赖图（-> 表示 CUDA Graph 边，虚线语义由 wait kernel 连接外部CPU/NIC）：
//   epoch++ ------------------------------+
//   count -> prefix -> pack --------------+-> send -> wait DONE -> scatter
//                                               +-> C2C local gather（可选）
//   gather scales（独立根节点）
//
// send仅提交/发布请求，并不等于数据返回。owner CPU观察邮箱ready后构造SGE WRITE，
// NIC写数据及各QP DONE；wait轮询这些DONE，将外部通信完成接回Graph依赖链。
// 调用stream上的后续kernel在整个Graph完成后才执行，包括local/scales独立分支。
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
    GpuFetchGraph(Gpu& gpu,GpuRouter& router,Inbox& inbox,const Options& o,CUdeviceptr table_aliases){
        cuda_check(cuMemAlloc(&epoch,8),"graph fetch epoch");
        cuda_check(cuMemsetD8(epoch,0,8),"initialize graph epoch");
        cuda_check(cuGraphCreate(&graph,0),"create fetch graph");
        cuda_check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"fetch graph stream");
        CUfunction begin,send,wait;
        cuda_check(cuModuleGetFunction(&begin,router.module,"begin_fetch_epoch"),"graph epoch kernel");
        cuda_check(cuModuleGetFunction(&send,router.module,"send_gpu_requests_epoch"),"graph send kernel");
        cuda_check(cuModuleGetFunction(&wait,router.module,"wait_gpu_epoch"),"graph wait kernel");
        // en和cn都是根节点，可以并行；发送请求前必须同时拿到本轮epoch和打包结果。
        void* begin_args[]={&epoch};auto en=kernel(begin,1,1,begin_args);

        unsigned rows=frame_bytes(o)/ROW,tokens=o.groups,blocks=router.blocks,rank=o.rank,qps=o.qps,hybrid=o.hybrid_local,compact=o.compact_done,merge=o.merge_rows;
        CUdeviceptr packed=reinterpret_cast<CUdeviceptr>(router.packed);uint64_t stride=inbox.slot_bytes;
        void* count_args[]={&gpu.input,&router.counts,&rows,&tokens,&rank,&hybrid};
        auto cn=kernel(router.count,blocks,256,count_args);
        // pn依赖cn：必须先收齐各tile的owner计数，才能计算owner/tile前缀。
        void* prefix_args[]={&router.counts,&router.offsets,&router.totals,&router.bases,&blocks};
        auto pn=kernel(router.prefix,1,32,prefix_args,{cn});
        // kn依赖pn：以本轮前缀确定local_rows和positions的写入位置。
        void* pack_args[]={&gpu.input,&packed,&gpu.positions,&router.offsets,&router.bases,&rows,&tokens,&rank,&hybrid};
        auto kn=kernel(router.pack,blocks,256,pack_args,{pn});
        void* send_args[]={&router.dev,&router.source,&router.target_windows,&router.alias_pointers,
            &packed,&router.totals,&router.bases,&rank,&stride,&hybrid,&epoch};
        // sn同时依赖kn/en：请求中行号与metadata完整，并使用本轮序号。
        auto sn=kernel(send,8,256,send_args,{kn,en});
        // wn依赖sn：先发请求，再用单个256线程block等待所需owner/QP的DONE。
        // CPU owner常驻线程并非CUDA节点；它们通过邮箱和DONE与这两个节点衔接。
        void* wait_args[]={&gpu.done_flags,&qps,&epoch,&router.totals,&merge,&compact,&rank,&hybrid};auto wn=kernel(wait,1,256,wait_args,{sn});
        uint64_t words=gpu.bytes/8;
        void* scatter_args[]={&gpu.packed,&gpu.output,&gpu.positions,&words};
        unsigned scatter_grid=o.adaptive_grid?unsigned(std::min(uint64_t(4096),(words+255)/256)):4096;
        if(o.hybrid_local){
            CUfunction local,remote;
            cuda_check(cuModuleGetFunction(&local,router.module,"gather_local_cpu_rows"),"local CPU gather kernel");
            cuda_check(cuModuleGetFunction(&remote,router.module,"scatter_remote_rows"),"remote scatter kernel");
            void* remote_args[]={&gpu.packed,&gpu.output,&gpu.positions,&router.totals};
            // 远端scatter必须依赖wn，不能只依赖sn，否则可能读到尚未写完的staging。
            kernel(remote,scatter_grid,256,remote_args,{wn});
            void* local_args[]={&gpu.input,&gpu.output,&table_aliases,&rows,&tokens,&rank};
            // 本地gather只依赖sn，刻意让请求先发出，再与远端wait/WRITE并行。
            // 它直接写本地owner对应的最终槽位，与remote scatter的槽位互斥。
            kernel(local,scatter_grid,256,local_args,{sn});
        }else kernel(gpu.scatter,scatter_grid,256,scatter_args,{wn});
        // scales只读当前原始indices和节点内GPU分片，无须等待embedding数据。
        // 不传dependencies即为独立根节点；Graph完成仍会等待此分支执行结束。
        void* scale_args[]={&gpu.input,&gpu.sf_output,&gpu.sf_peers[0],&gpu.sf_peers[1],&gpu.sf_peers[2],&gpu.sf_peers[3],&tokens};
        kernel(gpu.gather_sf,o.adaptive_grid?std::min(512U,(tokens*48+255)/256):512,256,scale_args);
        cuda_check(cuGraphInstantiateWithFlags(&executable,graph,0),"instantiate fetch graph");
        // instantiate/upload/首次同步都在初始化阶段，不能放进每轮enqueue。
        cuda_check(cuGraphUpload(executable,stream),"prepare fetch graph on device");
        cuda_check(cuStreamSynchronize(stream),"finish Buffer graph initialization");
    }
    // 异步forward入口：只向调用方的推理stream提交Graph，无主机等待/D2H/TCP/分配。
    // 同stream的上游producer先写input，Graph再读取；下游consumer等整个Graph完成。
    // 同一Buffer允许在同stream连续排队，不能在无依赖的stream中并发覆盖共享工作区。
    // 跨stream消费需record/wait event；真正并发fetch需要独立Buffer或显式串行依赖。
    void enqueue(CUstream inference_stream) {
        cuda_check(cuGraphLaunch(executable,inference_stream),"enqueue Engram fetch");
    }
    CUstream benchmark_stream() const {return stream;}
    double run(){
        // 同步测试包装：等待是为了测完整fetch，与上面的异步enqueue接口分开。
        auto t=Clock::now();
        enqueue(stream);
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
