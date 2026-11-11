#pragma once
#include "gpu_state.hpp"
struct Channel{
    Options o;Resources r;Socket sock;Hello peer{};
    void init(const Options&base,int other,void*table,const Gpu*gpu){
        o=base;int owner=o.owner?o.rank:other,requester=o.owner?other:o.rank;
        o.port+=owner*8+requester;o.host=node_ip(o,owner);open_hca(r,o);create_qps(r,o);
        if(o.owner){r.cpu=table;r.borrowed_cpu=true;r.mr=ibv_reg_mr(r.pd,table,size_t(o.table_rows)*ROW,IBV_ACCESS_LOCAL_WRITE);check(r.mr,"register CPU vocabulary");}
        else{cuda_check(cuCtxSetCurrent(gpu->ctx),"requester CUDA context");r.mr=ibv_reg_mr(r.pd,reinterpret_cast<void*>(gpu->allocation),gpu->allocation_bytes,IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);check(r.mr,"register GPU destination");}
        connect(sock,o);Hello h{};h.pool=o.pool;h.qps=o.qps;h.groups=o.groups;h.table_rows=o.table_rows;h.merge_rows=o.merge_rows;
        if(!o.owner){h.dst_addr=gpu->packed;h.dst_rkey=r.mr->rkey;h.done_addr=gpu->done_flags+owner*16*sizeof(uint64_t);h.done_rkey=r.mr->rkey;}
        sock.send(h);sock.recv(peer);check(peer.magic==MAGIC&&peer.version==h.version&&peer.bytes==sizeof(Hello)&&peer.pool==h.pool&&peer.qps==h.qps&&peer.groups==h.groups&&peer.table_rows==h.table_rows&&peer.merge_rows==h.merge_rows,"WRITE handshake");activate(r,o,sock);
    }
};
// Invariant lengths/lkeys are initialized once. Current row addresses, SGE
// counts and destination offsets are constructed for every timed fetch.
static int post_write(Channel&c,const Group*groups,unsigned n,const std::vector<ibv_sge>&sges){
    unsigned offset=0;
    for(unsigned q=0;q<unsigned(c.o.qps)&&q<n;++q)ibv_wr_start(ibv_qp_to_qp_ex(c.r.qps[q]));
    for(unsigned i=0;i<n;++i){auto*x=ibv_qp_to_qp_ex(c.r.qps[i%c.o.qps]);x->wr_id=i;x->wr_flags=i+c.o.qps>=n?IBV_SEND_SIGNALED:0;
        ibv_wr_rdma_write(x,c.peer.dst_rkey,c.peer.dst_addr+uint64_t(groups[i].dst_row)*ROW);ibv_wr_set_sge_list(x,groups[i].count,sges.data()+offset);offset+=groups[i].count;}
    int completions=0;
    for(unsigned q=0;q<unsigned(c.o.qps)&&q<n;++q){verbs_check(ibv_wr_complete(ibv_qp_to_qp_ex(c.r.qps[q])),"submit WRITE WQEs");++completions;}
    return completions;
}
static void publish_gpu_done(Channel&c,uint64_t epoch,unsigned count){
    for(unsigned q=0;q<count;++q){ibv_sge sg{reinterpret_cast<uintptr_t>(&epoch),8,0};ibv_send_wr wr{},*bad=nullptr;
        wr.wr_id=UINT64_MAX;wr.opcode=IBV_WR_RDMA_WRITE;wr.sg_list=&sg;wr.num_sge=1;wr.send_flags=IBV_SEND_INLINE|IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr=c.peer.done_addr+q*8;wr.wr.rdma.rkey=c.peer.done_rkey;verbs_check(ibv_post_send(c.r.qps[q],&wr,&bad),"publish GPU DONE");}
    c.r.poll(count);
}
