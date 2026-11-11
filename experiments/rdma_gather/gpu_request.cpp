#include "benchmark_impl.hpp"
#include <mpi.h>
#include <nccl.h>
#include <nccl_device/host.h>
#include <atomic>
#include <sys/syscall.h>
static void nccl_check(ncclResult_t r, const char* where) {
    if (r != ncclSuccess) { std::fprintf(stderr,"NCCL ERROR %s: %s\n",where,ncclGetErrorString(r)); throw std::runtime_error(std::string(where)+": "+ncclGetErrorString(r)); }
}
static void allocate_nccl_output(CUdeviceptr* p,size_t bytes){
    void* out=nullptr;nccl_check(ncclMemAlloc(&out,bytes),"allocate GIN-compatible GPU output");*p=reinterpret_cast<CUdeviceptr>(out);
}
static void release_nccl_output(CUdeviceptr p){ncclMemFree(reinterpret_cast<void*>(p));}
struct Inbox {
    CUdeviceptr ptr = 0; CUmemGenericAllocationHandle handle{}; size_t bytes = 0, slot_bytes = 0;
    explicit Inbox(const Options& o,size_t capacity=0) {
        int numa; cuda_check(cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, o.gpu), "host NUMA");
        CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; prop.location.id = numa;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        size_t granularity; cuda_check(cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "mailbox granularity");
        slot_bytes = ((frame_bytes(o)/ROW*4 + 64 + 4095)/4096)*4096;
        bytes = (((capacity?capacity:slot_bytes*8) + granularity-1)/granularity)*granularity;
        cuda_check(cuMemCreate(&handle,bytes,&prop,0), "create CPU mailbox VMM");
        cuda_check(cuMemAddressReserve(&ptr,bytes,granularity,0,0), "reserve mailbox");
        cuda_check(cuMemMap(ptr,bytes,0,handle,0), "map mailbox");
        CUmemAccessDesc access[2]{};
        access[0].location.type=CU_MEM_LOCATION_TYPE_DEVICE;access[0].location.id=o.gpu;access[0].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        access[1].location=prop.location;access[1].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuda_check(cuMemSetAccess(ptr,bytes,access,2), "mailbox access");
        std::memset(reinterpret_cast<void*>(ptr),0,bytes);
    }
    ~Inbox() { if(ptr){cuMemUnmap(ptr,bytes);cuMemAddressFree(ptr,bytes);cuMemRelease(handle);} }
};
struct InboxAliases {
    CUdeviceptr pointers[4]{}; CUmemGenericAllocationHandle handles[4]{};
    size_t bytes;
    InboxAliases(const Options& o, Inbox& inbox) : bytes(inbox.bytes) {
        struct Share { int pid, fd; } local{}, shares[8]{};
        local.pid=getpid();cuda_check(cuMemExportToShareableHandle(&local.fd,inbox.handle,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,0),"export owner inbox");
        MPI_Allgather(&local,sizeof(local),MPI_BYTE,shares,sizeof(local),MPI_BYTE,MPI_COMM_WORLD);
        int numa;cuda_check(cuDeviceGetAttribute(&numa,CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,o.gpu),"alias NUMA");
        for(int i=0;i<4;++i) {
            int rank=o.rank/4*4+i;
            if(rank==o.rank){pointers[i]=inbox.ptr;continue;}
            int pidfd=syscall(SYS_pidfd_open,shares[rank].pid,0);check(pidfd>=0,"open inbox exporter pidfd");
            int fd=syscall(SYS_pidfd_getfd,pidfd,shares[rank].fd,0);close(pidfd);check(fd>=0,"duplicate inbox handle");
            cuda_check(cuMemImportFromShareableHandle(&handles[i],reinterpret_cast<void*>(intptr_t(fd)),CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),"import owner inbox");close(fd);
            cuda_check(cuMemAddressReserve(&pointers[i],bytes,0,0,0),"reserve inbox alias");cuda_check(cuMemMap(pointers[i],bytes,0,handles[i],0),"map inbox alias");
            CUmemAccessDesc access[2]{};access[0].location={CU_MEM_LOCATION_TYPE_DEVICE,o.gpu};access[0].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            access[1].location={CU_MEM_LOCATION_TYPE_HOST_NUMA,numa};access[1].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuda_check(cuMemSetAccess(pointers[i],bytes,access,2),"alias access");
        }
        MPI_Barrier(MPI_COMM_WORLD);close(local.fd);
    }
    ~InboxAliases(){for(int i=0;i<4;++i)if(handles[i]){cuMemUnmap(pointers[i],bytes);cuMemAddressFree(pointers[i],bytes);cuMemRelease(handles[i]);}}
};
static void serve_gpu(Channel& c, Inbox& inbox, int requester, std::atomic<bool>& stop) {
    receive_target_root(c);
    auto& r=c.r; const auto& o=c.o; uint64_t epoch=1;
    if(o.gpu_get || (o.hybrid_local && requester/4==o.rank/4)){while(!stop.load(std::memory_order_relaxed))usleep(1000);return;}
    auto* slot=reinterpret_cast<uint64_t*>(inbox.ptr+requester*inbox.slot_bytes);
    auto* rows=reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(slot)+64);
    std::vector<Group> groups(o.pool);
    std::vector<ibv_sge> sges(size_t(o.pool)*o.merge_rows,{0,uint32_t(ROW),r.mr->lkey});
    std::vector<uint32_t> keys;
    while (!stop.load(std::memory_order_relaxed)) {
        unsigned spins=0;
        while (__atomic_load_n(slot,__ATOMIC_ACQUIRE) < epoch) {
            if (stop.load(std::memory_order_relaxed)) return;
            if (++spins==1024){if(o.owner_poll_us)usleep(o.owner_poll_us);spins=0;}
        }
        uint64_t meta=__atomic_load_n(slot+1,__ATOMIC_ACQUIRE);
        uint32_t nr=uint32_t(meta),base=uint32_t(meta>>32);
        bool indirect=slot[2]!=0;
        check(!indirect || c.registered_target_rkey,"GPU request without registered root");
        check(uint64_t(base)+nr<=frame_bytes(o)/ROW,"GPU request bounds");
        uint32_t ng=(nr+o.merge_rows-1)/o.merge_rows;
        Response result{};result.groups=ng;result.fetch_id=epoch;
        size_t offset=0;int pending=0,windows=0;
        for (uint32_t at=0;at<ng;at+=o.pool) {
            auto started=Clock::now();uint32_t n=std::min(uint32_t(o.pool),ng-at),batch_rows=0;
            for(uint32_t i=0;i<n;++i){uint32_t row=(at+i)*o.merge_rows,count=std::min(uint32_t(o.merge_rows),nr-row);groups[i]={count,(indirect?0:base)+row};batch_rows+=count;}
            for(uint32_t i=0;i<batch_rows;++i){uint32_t row=rows[offset+i];check(row<uint32_t(o.table_rows),"GPU source row");sges[i].addr=reinterpret_cast<uintptr_t>(r.cpu)+size_t(row)*ROW;}
            result.stats.layout_us+=us(started);Stats batch{};
            pending+=transfer(r,o,false,groups.data(),n,indirect?0:c.peer.dst_addr,indirect?c.registered_target_rkey:c.peer.dst_rkey,sges,keys,batch,*c.workspace,true);
            add(result.stats,batch);offset+=batch_rows;
            // Every wave drains ALL expected CQEs before SQ credits are reused;
            // counting only one completed window would be unsafe across QPs.
            if(++windows==o.owner_windows || at+n==ng){
                auto t=Clock::now();r.poll(pending);double wait=us(t);
                result.stats.cq_wait_us+=wait;result.stats.data_us+=wait;
                windows=0;pending=0;
            }
        }
        publish_gpu_done(c,epoch,o.compact_done?std::max(1U,std::min(unsigned(o.qps),ng)):0); c.sock.send(result); ++epoch;
    }
}
struct GpuRouter {
    CUmodule module=nullptr;CUfunction count=nullptr,prefix=nullptr,pack=nullptr,send=nullptr;
    CUdeviceptr counts=0,offsets=0,totals=0,bases=0;void* packed=nullptr;
    ncclComm_t comm=nullptr;ncclDevComm_t dev{};ncclWindow_t source=nullptr;std::array<ncclWindow_t,4> targets{};
    std::array<ncclWindow_t,4> models{};ncclWindow_t get_output=nullptr;CUdeviceptr get_sources=0;
    std::unique_ptr<InboxAliases> aliases;CUdeviceptr alias_pointers=0,target_windows=0,indirect_mode=0;
    unsigned blocks=0; size_t packed_bytes=0;
    GpuRouter(const Options& o, Inbox& inbox) {
        const char* override_path=std::getenv("ENGRAM_GPU_PTX");
        char executable[4096]; ssize_t n=readlink("/proc/self/exe",executable,sizeof(executable)-1);
        check(n>0 && size_t(n)<sizeof(executable)-1,"locate GPU requester binary");
        executable[n]='\0';std::string path(executable);
        path=path.substr(0,path.find_last_of('/')+1)+"gpu_request.ptx";
        if(override_path && *override_path)path=override_path;
        cuda_check(cuModuleLoad(&module,path.c_str()),"load GPU router PTX");
        for(auto pair: {std::pair<CUfunction*,const char*>{&count,"route_counts"},{&prefix,"route_prefix"},{&pack,"route_pack"},{&send,"send_gpu_requests"}}) cuda_check(cuModuleGetFunction(pair.first,module,pair.second),pair.second);
        blocks=(frame_bytes(o)/ROW+4095)/4096;packed_bytes=frame_bytes(o)/ROW*4;
        cuda_check(cuMemAlloc(&counts,blocks*8*4),"GPU tile counts");cuda_check(cuMemAlloc(&offsets,blocks*8*4),"GPU tile offsets");
        cuda_check(cuMemAlloc(&totals,8*4),"GPU owner counts");cuda_check(cuMemAlloc(&bases,8*4),"GPU owner bases");
        nccl_check(ncclMemAlloc(&packed,packed_bytes),"allocate GIN source");
        if(o.gpu_request){
            aliases=std::make_unique<InboxAliases>(o,inbox);
            ncclUniqueId id{};if(!o.rank)nccl_check(ncclGetUniqueId(&id),"GIN unique ID");MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);
            nccl_check(ncclCommInitRank(&comm,8,id,o.rank),"init GIN communicator");
            ncclCommProperties_t props=NCCL_COMM_PROPERTIES_INITIALIZER;nccl_check(ncclCommQueryProperties(comm,&props),"query GIN properties");
            ncclDevCommRequirements_t req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
            check(props.ginMinStride==4,"this eight-rank prototype requires detected G=4");
            req.ginType=NCCL_GIN_TYPE_GDAKI;
            // Match the verified DeepEP host-storage path. The stride comes
            // from NCCL detection; there is no workload/user stride override.
            req.ginConnectionType=NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
            req.ginCustomStride=props.ginMinStride;
            req.ginContextCount=o.gin_contexts;req.ginQueueDepth=1024;req.ginExclusiveContexts=true;req.ginSignalCount=8;
            req.useRuntimeVersion=true;
            check(props.devCommRuntimeVersionSize==sizeof(dev),"GIN runtime/header device communicator size");
            nccl_check(ncclDevCommCreate(comm,&req,&dev),"create GDAKI device communicator");
            nccl_check(ncclCommWindowRegister(comm,packed,packed_bytes,&source,NCCL_WIN_GIN_ONLY|NCCL_WIN_STRICT_ORDERING),"register packed GPU rows");
            for(int i=0;i<4;++i)nccl_check(ncclCommWindowRegister(comm,reinterpret_cast<void*>(aliases->pointers[i]),inbox.bytes,&targets[i],NCCL_WIN_GIN_ONLY|NCCL_WIN_STRICT_ORDERING),"register owner CPU inbox alias");
            cuda_check(cuMemAlloc(&alias_pointers,sizeof(aliases->pointers)),"alias GPU pointers");cuda_check(cuMemcpyHtoD(alias_pointers,aliases->pointers,sizeof(aliases->pointers)),"copy alias pointers");
            cuda_check(cuMemAlloc(&target_windows,sizeof(targets)),"window GPU handles");cuda_check(cuMemcpyHtoD(target_windows,targets.data(),sizeof(targets)),"copy target windows");
            cuda_check(cuMemAlloc(&indirect_mode,4),"GPU destination mode");
            unsigned fixed_mode=o.target_umr=="strict";
            cuda_check(cuMemcpyHtoD(indirect_mode,&fixed_mode,4),"initial GPU destination mode");
            std::fprintf(stderr,"GPU_REQUEST GDAKI contexts=%u detected_min_stride=%d backend_type=%d\n",unsigned(dev.ginContextCount),props.ginMinStride,int(req.ginType));
            const char* debug=std::getenv("ENGRAM_GPU_DEBUG");
            if(debug && std::string(debug)=="1"){
                CUfunction inspect;
                cuda_check(cuModuleGetFunction(&inspect,module,"debug_gin"),"GIN inspector");
                void* args[]={&dev};
                cuda_check(cuLaunchKernel(inspect,1,1,1,1,1,1,0,nullptr,args,nullptr),"inspect GIN QPs");
                cuda_check(cuCtxSynchronize(),"GIN inspection complete");
            }
        }
    }
    void route(Gpu& gpu,const Options& o) {
        unsigned rows=frame_bytes(o)/ROW,tokens=o.groups,rank=o.rank,hybrid=o.hybrid_local;CUdeviceptr data=reinterpret_cast<CUdeviceptr>(packed);
        void* a[]={&gpu.input,&counts,&rows,&tokens,&rank,&hybrid};cuda_check(cuLaunchKernel(count,blocks,1,1,256,1,1,0,nullptr,a,nullptr),"GPU count owners");
        void* b[]={&counts,&offsets,&totals,&bases,&blocks};cuda_check(cuLaunchKernel(prefix,1,1,1,32,1,1,0,nullptr,b,nullptr),"GPU prefix owners");
        void* c[]={&gpu.input,&data,&gpu.positions,&offsets,&bases,&rows,&tokens,&rank,&hybrid};cuda_check(cuLaunchKernel(pack,blocks,1,1,256,1,1,0,nullptr,c,nullptr),"GPU pack rows and positions");
    }
    void register_get(Gpu& gpu,const InboxAliases& aliases){
        for(int i=0;i<4;++i)nccl_check(ncclCommWindowRegister(comm,reinterpret_cast<void*>(aliases.pointers[i]),aliases.bytes,&models[i],NCCL_WIN_GIN_ONLY|NCCL_WIN_STRICT_ORDERING),"register CPU model for GET");
        nccl_check(ncclCommWindowRegister(comm,reinterpret_cast<void*>(gpu.output),gpu.bytes,&get_output,NCCL_WIN_GIN_ONLY|NCCL_WIN_STRICT_ORDERING),"register final GPU GET output");
        cuda_check(cuMemAlloc(&get_sources,sizeof(models)),"GET model handles");
        cuda_check(cuMemcpyHtoD(get_sources,models.data(),sizeof(models)),"publish GET handles");
    }
    void close_get(){
        if(get_output){ncclCommWindowDeregister(comm,get_output);get_output=nullptr;}
        for(auto& m:models)if(m){ncclCommWindowDeregister(comm,m);m=nullptr;}
        if(get_sources){cuMemFree(get_sources);get_sources=0;}
    }
    void send_requests(const Options& o,Inbox& inbox,uint64_t epoch) {
        unsigned rank=o.rank,hybrid=o.hybrid_local;uint64_t stride=inbox.slot_bytes;CUdeviceptr rows=reinterpret_cast<CUdeviceptr>(packed);
        void* args[]={&dev,&source,&target_windows,&alias_pointers,&indirect_mode,&rows,&totals,&bases,&rank,&stride,&hybrid,&epoch};
        cuda_check(cuLaunchKernel(send,8,1,1,256,1,1,0,nullptr,args,nullptr),"GPU send index requests");
    }
    ~GpuRouter(){
        close_get();
        if(comm){if(source)ncclCommWindowDeregister(comm,source);for(auto target:targets)if(target)ncclCommWindowDeregister(comm,target);ncclDevCommDestroy(comm,&dev);ncclCommDestroy(comm);}
        if(alias_pointers)cuMemFree(alias_pointers);
        if(target_windows)cuMemFree(target_windows);
        if(indirect_mode)cuMemFree(indirect_mode);
        if(packed)ncclMemFree(packed);
        if(counts)cuMemFree(counts);
        if(offsets)cuMemFree(offsets);
        if(totals)cuMemFree(totals);
        if(bases)cuMemFree(bases);
        if(module)cuModuleUnload(module);
    }
};
#include "gpu_graph.hpp"
static void run_gpu_rank(Options o) {
    check(o.protocol=="frame"&&o.layout=="merged"&&o.done_gpu,"GPU requester requires frame+merged+GPU DONE");
    pin_core(o,0,false);Gpu gpu;gpu.init(o,o.gpu_get?allocate_nccl_output:nullptr,o.gpu_get?release_nccl_output:nullptr);Inbox inbox(o);Plan plan(o);Workers workers(o);
    if(!gpu.positions)cuda_check(cuMemAlloc(&gpu.positions,frame_bytes(o)/ROW*4),"GPU routing positions for dynamic UMR");
    void* table=nullptr;std::unique_ptr<Inbox> vmm_table;
    std::unique_ptr<void,decltype(&free)> owner_table(nullptr,&free);
    if(o.host_vmm_table){vmm_table=std::make_unique<Inbox>(o,size_t(o.table_rows)*ROW);table=reinterpret_cast<void*>(vmm_table->ptr);}
    else{check(!posix_memalign(&table,4096,size_t(o.table_rows)*ROW),"integrated owner table");owner_table.reset(table);}
    for(int row=0;row<o.table_rows;++row)for(unsigned word=0;word<ROW/8;++word)static_cast<uint64_t*>(table)[size_t(row)*ROW/8+word]=pattern(o.rank*o.table_rows+row,word);
    std::atomic<bool> stop{false};std::vector<std::thread> owners;
    try {
    for(int peer=0;!o.gpu_get && peer<8;++peer)owners.emplace_back([&,peer]{
        try{Options owner=o;owner.owner=true;pin_core(owner,peer,true);Channel c;c.init(owner,peer,table,nullptr);
            if(o.gpu_request)serve_gpu(c,inbox,peer,stop);else serve(c);
        }catch(const std::exception& e){std::fprintf(stderr,"owner peer=%d %s\n",peer,e.what());std::abort();}
    });
    std::vector<std::unique_ptr<Channel>> channels(8);
    if(!o.gpu_get)workers.run([&](int peer){cuda_check(cuCtxSetCurrent(gpu.ctx),"requester context");channels[peer]=std::make_unique<Channel>();channels[peer]->init(o,peer,nullptr,&gpu);});
    std::vector<std::unique_ptr<TargetMkey>> targets(8);
    if(o.target_umr!="none")workers.run([&](int peer){
        targets[peer]=std::make_unique<TargetMkey>(*channels[peer],o,gpu.output);
        channels[peer]->sock.send(targets[peer]->rkey());
    });
    GpuRouter router(o,inbox);
    std::unique_ptr<InboxAliases> table_aliases;CUdeviceptr table_gpu_pointers=0;
    if(o.hybrid_local){
        table_aliases=std::make_unique<InboxAliases>(o,*vmm_table);
        cuda_check(cuMemAlloc(&table_gpu_pointers,sizeof(table_aliases->pointers)),"local CPU table GPU aliases");
        cuda_check(cuMemcpyHtoD(table_gpu_pointers,table_aliases->pointers,sizeof(table_aliases->pointers)),"publish local CPU table aliases");
        if(o.gpu_get)router.register_get(gpu,*table_aliases);
    }
    CUipcMemHandle handle{};cuda_check(cuIpcGetMemHandle(&handle,gpu.sf_shard),"scale IPC export");std::array<CUipcMemHandle,8> handles;
    MPI_Allgather(&handle,sizeof(handle),MPI_BYTE,handles.data(),sizeof(handle),MPI_BYTE,MPI_COMM_WORLD);
    for(int i=0;i<4;++i){if(i==o.rank%4)gpu.sf_peers[i]=gpu.sf_shard;else cuda_check(cuIpcOpenMemHandle(&gpu.sf_peers[i],handles[o.rank/4*4+i],CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),"scale IPC import");}
    const char* debug_env=std::getenv("ENGRAM_GPU_DEBUG");
    const bool debug=debug_env && std::string(debug_env)=="1";
    auto raw=load_indices(o); CUevent a,b;cuda_check(cuEventCreate(&a,0),"route event");cuda_check(cuEventCreate(&b,0),"route event");
    std::unique_ptr<GpuFetchGraph> graph;
    if(o.gpu_graph)graph=std::make_unique<GpuFetchGraph>(gpu,router,inbox,o,a,b,table_gpu_pointers);
    for(int round=-o.warmups;round<o.rounds;++round){
        if(o.vary_indices)raw=generate_indices(o,round+o.warmups+1);
        expected_rows(plan,o,raw.data());
        cuda_check(cuMemcpyHtoD(gpu.input,raw.data(),raw.size()*4),"upstream indices");gpu.clear_for_validation();MPI_Barrier(MPI_COMM_WORLD);
        uint64_t epoch=round+o.warmups+1;auto started=Clock::now();
        double gpu_tail_ms=0;
        std::vector<PeerStats> peer_stats(8);uint32_t counts[8]{},bases[8]{};double cpu_bridge_ms=0,target_build_ms=0,target_config_ms=0;bool target_active=false;
        if(graph)gpu_tail_ms=graph->run();
        else{
        gpu.start_scales(o);
        if(o.profile_routes)cuda_check(cuEventRecord(a,nullptr),"route start");
        router.route(gpu,o);
        if(o.profile_routes)cuda_check(cuEventRecord(b,nullptr),"route end");
        if(debug) cuda_check(cuCtxSynchronize(),"debug route complete");
        if(!o.gpu_request || o.target_umr!="none"){
            auto t=Clock::now();cuda_check(cuMemcpyDtoH(counts,router.totals,32),"download counts");cuda_check(cuMemcpyDtoH(bases,router.bases,32),"download bases");
            if(!o.gpu_request)cuda_check(cuMemcpyDtoH(gpu.host_input,reinterpret_cast<CUdeviceptr>(router.packed),router.packed_bytes),"download packed rows");
            size_t groups=0;
            for(int peer=0;peer<8;++peer){size_t n=(counts[peer]+o.merge_rows-1)/o.merge_rows;plan.peers[peer].groups={plan.group_storage.get()+groups,n};plan.peers[peer].rows={gpu.host_input+bases[peer],counts[peer]};
                for(size_t i=0;i<n;++i){uint32_t at=i*o.merge_rows;plan.peers[peer].groups[i]={uint32_t(std::min(size_t(o.merge_rows),size_t(counts[peer])-at)),bases[peer]+at};}groups+=n;}
            cpu_bridge_ms=us(t)/1000;
            if(o.target_umr!="none"){
                t=Clock::now();cuda_check(cuMemcpyDtoH(plan.positions.data(),gpu.positions,router.packed_bytes),"download actual GPU destination map");
                workers.run([&](int peer){targets[peer]->prepare(plan,peer);});target_build_ms=us(t)/1000;
                target_active=true;for(auto& target:targets)target_active &= target->fits;
                check(target_active || o.target_umr!="strict","GPU routing mapping exceeds UMR pool");
                if(target_active){t=Clock::now();workers.run([&](int peer){targets[peer]->configure();});target_config_ms=us(t)/1000;}
                if(o.gpu_request && o.target_umr=="adaptive"){
                    unsigned mode=target_active;
                    cuda_check(cuMemcpyHtoD(router.indirect_mode,&mode,sizeof(mode)),"publish indirect/fallback mode");
                }
            }
            if(!o.gpu_request)workers.run([&](int peer){cuda_check(cuCtxSetCurrent(gpu.ctx),"CPU transport context");peer_stats[peer]=fetch(*channels[peer],plan.peers[peer],2,epoch-1,target_active?targets[peer].get():nullptr);});
        }
        if(o.gpu_request){
            router.send_requests(o,inbox,epoch);
            if(debug) cuda_check(cuCtxSynchronize(),"debug GIN send complete");
        }
        gpu_tail_ms=gpu.reorder(!target_active,o.qps,epoch,o.compact_done?router.totals:0,o.merge_rows);
        }
        double e2e_ms=us(started)/1000;
        float route_ms=-1;if(o.profile_routes)cuda_check(cuEventElapsedTime(&route_ms,a,b),"route elapsed");
        // Validation and telemetry are after GPU outputs are complete, outside e2e.
        for(int peer=0;peer<8;++peer){if(o.gpu_get || (o.hybrid_local && peer/4==o.rank/4))continue;Response response{};channels[peer]->sock.recv(response);check(response.fetch_id==epoch&&response.status==0,"GPU completion telemetry");add(peer_stats[peer].stats,response.stats);}
        cuda_check(cuMemcpyDtoH(gpu.host_input,gpu.input,raw.size()*4),"validation raw indices");gpu.validate(plan);gpu.validate_scales(o);
        std::printf("{\"kind\":\"sample\",\"mode\":\"%s\",\"rank\":%d,\"round\":%d,\"warmup\":%s,\"gpu_request\":%s,\"tokens\":%d,\"bytes\":%llu,\"e2e_ms\":%.6f,\"gbps\":%.6f,\"gpu_route_ms\":%.6f,\"cpu_bridge_ms\":%.6f,\"gpu_tail_ms\":%.6f,\"data_ok\":true,\"scale_ok\":true,\"peers\":[",o.gpu_get?"gpu_direct_get":"gpu_route_write",o.rank,round,round<0?"true":"false",o.gpu_request?"true":"false",o.groups,(unsigned long long)frame_bytes(o),e2e_ms,frame_bytes(o)/e2e_ms/1e6,double(route_ms),cpu_bridge_ms,gpu_tail_ms);
        for(int peer=0;peer<8;++peer){auto& v=peer_stats[peer].stats;std::printf("%s{\"owner\":%d,\"layout_ms\":%.6f,\"data_ms\":%.6f,\"verbs_post_ms\":%.6f,\"cq_wait_ms\":%.6f}",peer?",":"",peer,v.layout_us/1000,v.data_us/1000,v.verbs_post_us/1000,v.cq_wait_us/1000);}
        size_t max_keys=0,max_runs=0;
        for(auto& target:targets)if(target){max_keys=std::max(max_keys,target->used_keys);max_runs=std::max(max_runs,target->input_runs);}
        std::printf("],\"target_active\":%s,\"target_build_ms\":%.6f,\"target_config_ms\":%.6f,\"target_keys_max\":%zu,\"target_runs_max\":%zu,\"gpu_graph\":%s,\"adaptive_grid\":%s,\"owner_windows\":%d,\"owner_poll_us\":%d,\"compact_done\":%s,\"hybrid_local\":%s,\"host_vmm_table\":%s,\"input_epoch\":%llu,\"gpu_get\":%s,\"gin_contexts\":%u,\"get_batch\":%s}\n",target_active?"true":"false",target_build_ms,target_config_ms,max_keys,max_runs,o.gpu_graph?"true":"false",o.adaptive_grid?"true":"false",o.owner_windows,o.owner_poll_us,o.compact_done?"true":"false",o.hybrid_local?"true":"false",o.host_vmm_table?"true":"false",(unsigned long long)epoch,o.gpu_get?"true":"false",unsigned(router.dev.ginContextCount),o.get_batch?"true":"false");
    }
    MPI_Barrier(MPI_COMM_WORLD); // No peer may still READ a model window we deregister.
    graph.reset();router.close_get();if(table_gpu_pointers)cuMemFree(table_gpu_pointers);
    cuEventDestroy(a);cuEventDestroy(b);MPI_Barrier(MPI_COMM_WORLD);
    if(o.gpu_request)stop.store(true);else for(auto& c:channels){Command cmd{};c->sock.send(cmd);}
    for(auto& t:owners)t.join();
    MPI_Barrier(MPI_COMM_WORLD);gpu.close_scale_peers(o);MPI_Barrier(MPI_COMM_WORLD);
    } catch(const std::exception& e) {
        // Abort the owned MPI job before joinable service threads unwind. A
        // setup/device error must retain its diagnostic instead of terminate().
        std::fprintf(stderr,"ERROR GPU rank %d: %s\n",o.rank,e.what());
        MPI_Abort(MPI_COMM_WORLD,1);
        throw;
    }
}
int main(int argc,char** argv){
    int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    try{setbuf(stdout,nullptr);Options o=options(argc,argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);check(size==8,"eight MPI ranks required");o.rank=rank;o.gpu=rank%4;o.device="mlx5_bond_"+std::to_string(o.gpu);run_gpu_rank(o);MPI_Finalize();return 0;}
    catch(const std::exception& e){std::fprintf(stderr,"ERROR GPU requester: %s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
