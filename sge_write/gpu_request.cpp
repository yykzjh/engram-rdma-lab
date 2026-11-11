#include "rdma.hpp"
#include <mpi.h>
#include <nccl.h>
#include <nccl_device/host.h>
#include <atomic>
#include <sys/syscall.h>
static void nccl_check(ncclResult_t r,const char*where){if(r!=ncclSuccess)throw std::runtime_error(std::string(where)+": "+ncclGetErrorString(r));}
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
static void serve_gpu(Channel&c,Inbox&inbox,int requester,std::atomic<bool>&stop){
    auto&r=c.r;auto&o=c.o;uint64_t epoch=1;auto*slot=reinterpret_cast<uint64_t*>(inbox.ptr+requester*inbox.slot_bytes);
    auto*rows=reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(slot)+64);
    std::vector<Group>groups(o.pool);std::vector<ibv_sge>sges(size_t(o.pool)*o.merge_rows,{0,uint32_t(ROW),r.mr->lkey});
    while(!stop.load()){
        unsigned spins=0;while(__atomic_load_n(slot,__ATOMIC_ACQUIRE)<epoch){if(stop.load())return;if(++spins==1024){if(o.owner_poll_us)usleep(o.owner_poll_us);spins=0;}}
        uint64_t meta=__atomic_load_n(slot+1,__ATOMIC_ACQUIRE);uint32_t nr=uint32_t(meta),base=uint32_t(meta>>32);
        check(uint64_t(base)+nr<=frame_bytes(o)/ROW,"request bounds");unsigned ng=(nr+o.merge_rows-1)/o.merge_rows;
        size_t offset=0;int pending=0,windows=0;
        for(unsigned at=0;at<ng;at+=o.pool){unsigned n=std::min(unsigned(o.pool),ng-at),batch_rows=0;
            for(unsigned i=0;i<n;++i){unsigned row=(at+i)*o.merge_rows,count=std::min(unsigned(o.merge_rows),nr-row);groups[i]={count,base+row};batch_rows+=count;}
            for(unsigned i=0;i<batch_rows;++i){uint32_t row=rows[offset+i];check(row<unsigned(o.table_rows),"source row");sges[i].addr=reinterpret_cast<uintptr_t>(r.cpu)+size_t(row)*ROW;}
            pending+=post_write(c,groups.data(),n,sges);offset+=batch_rows;
            if(++windows==o.owner_windows||at+n==ng){r.poll(pending);pending=windows=0;}
        }
        publish_gpu_done(c,epoch,std::max(1U,std::min(unsigned(o.qps),ng)));++epoch;
    }
}
struct GpuRouter {
    CUmodule module=nullptr;CUfunction count=nullptr,prefix=nullptr,pack=nullptr,send=nullptr;
    CUdeviceptr counts=0,offsets=0,totals=0,bases=0;void* packed=nullptr;
    ncclComm_t comm=nullptr;ncclDevComm_t dev{};ncclWindow_t source=nullptr;std::array<ncclWindow_t,4> targets{};
    std::unique_ptr<InboxAliases> aliases;CUdeviceptr alias_pointers=0,target_windows=0;
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
    ~GpuRouter(){
        if(comm){if(source)ncclCommWindowDeregister(comm,source);for(auto target:targets)if(target)ncclCommWindowDeregister(comm,target);ncclDevCommDestroy(comm,&dev);ncclCommDestroy(comm);}
        if(alias_pointers)cuMemFree(alias_pointers);if(target_windows)cuMemFree(target_windows);if(packed)ncclMemFree(packed);
        if(counts)cuMemFree(counts);if(offsets)cuMemFree(offsets);if(totals)cuMemFree(totals);if(bases)cuMemFree(bases);if(module)cuModuleUnload(module);
    }
};
#include "gpu_graph.hpp"
#include "async_validation.hpp"
static void run_rank(Options o){
    pin_core(o,0,false);Gpu gpu;gpu.init(o);Inbox inbox(o);Plan plan(o);Workers workers(o);
    std::printf("{\"kind\":\"metadata\",\"rank\":%d,\"c2c\":%s,\"gpu_allocation_bytes\":%zu,\"staging_bytes\":%zu,\"owner_threads\":%d}\n",o.rank,o.hybrid_local?"true":"false",gpu.allocation_bytes,gpu.bytes,o.hybrid_local?4:8);
    Inbox table(o,size_t(o.table_rows)*ROW);
    for(int row=0;row<o.table_rows;++row)for(unsigned w=0;w<ROW/8;++w)
        reinterpret_cast<uint64_t*>(table.ptr)[size_t(row)*ROW/8+w]=pattern(o.rank*o.table_rows+row,w);
    std::atomic<bool>stop{false};std::vector<std::thread>owners;
    try{
        for(int peer=0;peer<8;++peer)if(!o.hybrid_local||peer/4!=o.rank/4)owners.emplace_back([&,peer]{
            try{Options owner=o;owner.owner=true;pin_core(owner,peer,true);Channel c;c.init(owner,peer,reinterpret_cast<void*>(table.ptr),nullptr);serve_gpu(c,inbox,peer,stop);}
            catch(const std::exception&e){std::fprintf(stderr,"owner error: %s\n",e.what());std::abort();}
        });
        std::vector<std::unique_ptr<Channel>>channels(8);
        workers.run([&](int peer){if(o.hybrid_local&&peer/4==o.rank/4)return;cuda_check(cuCtxSetCurrent(gpu.ctx),"channel context");channels[peer]=std::make_unique<Channel>();channels[peer]->init(o,peer,nullptr,&gpu);});
        GpuRouter router(o,inbox);InboxAliases table_aliases(o,table);CUdeviceptr pointers=0;
        cuda_check(cuMemAlloc(&pointers,sizeof(table_aliases.pointers)),"CPU table pointers");cuda_check(cuMemcpyHtoD(pointers,table_aliases.pointers,sizeof(table_aliases.pointers)),"publish CPU table pointers");
        CUipcMemHandle handle{};cuda_check(cuIpcGetMemHandle(&handle,gpu.sf_shard),"export scales");std::array<CUipcMemHandle,8>handles;
        MPI_Allgather(&handle,sizeof(handle),MPI_BYTE,handles.data(),sizeof(handle),MPI_BYTE,MPI_COMM_WORLD);
        for(int i=0;i<4;++i){if(i==o.rank%4)gpu.sf_peers[i]=gpu.sf_shard;else cuda_check(cuIpcOpenMemHandle(&gpu.sf_peers[i],handles[o.rank/4*4+i],CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),"import scales");}
        auto graph=std::make_unique<GpuFetchGraph>(gpu,router,inbox,o,pointers);
        if(o.async_check)validate_async_forward(gpu,router,*graph,o);
        else for(int round=-o.warmups;round<o.rounds;++round){
            uint64_t epoch=round+o.warmups+1;auto raw=generate_indices(o,epoch);expected_rows(plan,o,raw.data());
            cuda_check(cuMemcpyHtoD(gpu.input,raw.data(),raw.size()*4),"upstream indices");gpu.clear_for_validation();MPI_Barrier(MPI_COMM_WORLD);
            auto start=Clock::now();graph->run();double elapsed=us(start)/1000;
            cuda_check(cuMemcpyDtoH(gpu.host_input,gpu.input,raw.size()*4),"validation indices");gpu.validate(plan);gpu.validate_scales(o);
            std::printf("{\"kind\":\"sample\",\"mode\":\"write_sge\",\"rank\":%d,\"round\":%d,\"warmup\":%s,\"tokens\":%d,\"bytes\":%llu,\"e2e_ms\":%.6f,\"gbps\":%.6f,\"data_ok\":true,\"scale_ok\":true,\"input_epoch\":%llu,\"c2c\":%s}\n",o.rank,round,round<0?"true":"false",o.groups,(unsigned long long)frame_bytes(o),elapsed,frame_bytes(o)/elapsed/1e6,(unsigned long long)epoch,o.hybrid_local?"true":"false");
        }
        MPI_Barrier(MPI_COMM_WORLD);graph.reset();cuMemFree(pointers);stop.store(true);
        for(auto&t:owners)t.join();MPI_Barrier(MPI_COMM_WORLD);gpu.close_scale_peers(o);MPI_Barrier(MPI_COMM_WORLD);
    }catch(const std::exception&e){std::fprintf(stderr,"requester rank %d: %s\n",o.rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);throw;}
}
int main(int argc,char**argv){
    int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    try{setbuf(stdout,nullptr);Options o=options(argc,argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);check(size==8,"eight ranks required");o.rank=rank;o.gpu=rank%4;o.device="mlx5_bond_"+std::to_string(o.gpu);run_rank(o);MPI_Finalize();return 0;}
    catch(const std::exception&e){std::fprintf(stderr,"SGE WRITE: %s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
