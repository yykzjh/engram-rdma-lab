#pragma once
// A test harness, not part of enqueue(): upstream and dependent application
// kernels surround fetch on the SAME externally supplied stream. The host queues
// every forward before its single final stream synchronization.
static void validate_async_forward(Gpu& gpu,GpuRouter& router,GpuFetchGraph& graph,const Options& o){
    CUstream inference=nullptr;cuda_check(cuStreamCreate(&inference,CU_STREAM_NON_BLOCKING),"inference stream");
    CUfunction producer,consumer;
    cuda_check(cuModuleGetFunction(&producer,router.module,"async_producer"),"producer function");
    cuda_check(cuModuleGetFunction(&consumer,router.module,"async_consumer"),"consumer function");
    CUdeviceptr verdict=0;cuda_check(cuMemAlloc(&verdict,size_t(o.rounds)*16),"async verdicts");
    cuda_check(cuMemsetD8(verdict,0,size_t(o.rounds)*16),"clear async verdicts");
    std::vector<CUevent>begins(o.rounds),ends(o.rounds);
    for(int i=0;i<o.rounds;++i){cuda_check(cuEventCreate(&begins[i],0),"fetch start event");cuda_check(cuEventCreate(&ends[i],0),"fetch end event");}
    gpu.clear_for_validation();MPI_Barrier(MPI_COMM_WORLD);
    const auto sends_before=control_socket_sends.load(),recvs_before=control_socket_recvs.load();
    std::vector<double>enqueue_us;auto queued_start=Clock::now();
    unsigned tokens=o.groups,rank=o.rank,uniform=o.distribution=="uniform";
    for(unsigned frame=0;frame<unsigned(o.rounds);++frame){
        unsigned epoch=frame+1;void*pa[]={&gpu.input,&tokens,&rank,&epoch,&uniform};
        cuda_check(cuLaunchKernel(producer,std::min(512U,(tokens*48+255)/256),1,1,256,1,1,0,inference,pa,nullptr),"upstream GPU indices");
        cuda_check(cuEventRecord(begins[frame],inference),"start async fetch timing");
        auto at=Clock::now();graph.enqueue(inference);enqueue_us.push_back(us(at));
        cuda_check(cuEventRecord(ends[frame],inference),"end async fetch timing");
        void*ca[]={&gpu.input,&gpu.output,&gpu.sf_output,&tokens,&epoch,&verdict,&frame};
        cuda_check(cuLaunchKernel(consumer,std::min(4096U,(tokens*48+255)/256),1,1,256,1,1,0,inference,ca,nullptr),"dependent GPU consumer");
    }
    double queue_us=us(queued_start);CUresult pending=cuEventQuery(ends.back());check(pending==CUDA_SUCCESS||pending==CUDA_ERROR_NOT_READY,"query queued completion");
    cuda_check(cuStreamSynchronize(inference),"single final forward synchronization");
    const auto socket_messages=control_socket_sends.load()+control_socket_recvs.load()-sends_before-recvs_before;
    check(socket_messages==0,"TCP protocol entered asynchronous forward");
    std::vector<uint64_t>host(o.rounds*2);cuda_check(cuMemcpyDtoH(host.data(),verdict,host.size()*8),"small validation verdicts after all forwards");
    for(int i=0;i<o.rounds;++i){check(host[i*2]==0&&host[i*2+1]==uint64_t(i+1),"GPU consumer observed wrong fetch/epoch");float ms;cuda_check(cuEventElapsedTime(&ms,begins[i],ends[i]),"async elapsed time");
        std::printf("{\"kind\":\"async_sample\",\"rank\":%d,\"tokens\":%d,\"round\":%d,\"enqueue_us\":%.6f,\"fetch_gpu_ms\":%.6f,\"data_ok\":true,\"scale_ok\":true,\"epoch\":%d}\n",o.rank,o.groups,i,enqueue_us[i],ms,i+1);
        cuEventDestroy(begins[i]);cuEventDestroy(ends[i]);}
    std::printf("{\"kind\":\"async_summary\",\"rank\":%d,\"tokens\":%d,\"forwards_queued\":%d,\"queue_us\":%.6f,\"last_fetch_incomplete_after_enqueue\":%s,\"per_forward_host_syncs\":0,\"measured_forward_tcp_messages\":%llu,\"requester_index_d2h_bytes\":0}\n",o.rank,o.groups,o.rounds,queue_us,pending==CUDA_ERROR_NOT_READY?"true":"false",(unsigned long long)socket_messages);
    cuMemFree(verdict);cuStreamDestroy(inference);
}
