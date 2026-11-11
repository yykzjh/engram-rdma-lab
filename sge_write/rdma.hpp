#pragma once
#include "gpu_state.hpp"
struct Channel{
    // 一条 Channel 对应一个固定的 (owner,requester) 对，内含多条 RC 数据QP。
    // TCP socket 只用于初始化交换 Hello/Endpoint；每轮 fetch 不调用它。
    Options o;Resources r;Socket sock;Hello peer{};
    void init(const Options&base,int other,void*table,const Gpu*gpu){
        o=base;int owner=o.owner?o.rank:other,requester=o.owner?other:o.rank;
        o.port+=owner*8+requester;o.host=node_ip(o,owner);open_hca(r,o);create_qps(r,o);
        if(o.owner){r.cpu=table;r.borrowed_cpu=true;r.mr=ibv_reg_mr(r.pd,table,size_t(o.table_rows)*ROW,IBV_ACCESS_LOCAL_WRITE);check(r.mr,"register CPU vocabulary");}
        else{cuda_check(cuCtxSetCurrent(gpu->ctx),"requester CUDA context");r.mr=ibv_reg_mr(r.pd,reinterpret_cast<void*>(gpu->allocation),gpu->allocation_bytes,IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);check(r.mr,"register GPU destination");}
        connect(sock,o);Hello h{};h.pool=o.pool;h.qps=o.qps;h.groups=o.groups;h.table_rows=o.table_rows;h.merge_rows=o.merge_rows;
        // requester 将 staging 基地址和该 owner 专属的 DONE 子区发给 owner。
        // DONE 布局为 uint64_t done_flags[8][16]，16是每owner预留的最大QP数，
        // 与本次实际 qps（默认4）分开；数据和DONE都在同一块已注册GPU分配内。
        if(!o.owner){h.dst_addr=gpu->packed;h.dst_rkey=r.mr->rkey;h.done_addr=gpu->done_flags+owner*16*sizeof(uint64_t);h.done_rkey=r.mr->rkey;}
        sock.send(h);sock.recv(peer);check(peer.magic==MAGIC&&peer.version==h.version&&peer.bytes==sizeof(Hello)&&peer.pool==h.pool&&peer.qps==h.qps&&peer.groups==h.groups&&peer.table_rows==h.table_rows&&peer.merge_rows==h.merge_rows,"WRITE handshake");activate(r,o,sock);
    }
};
// 提交一个内部窗口。SGE 的 length/lkey 是固定工作区属性；addr/count/dst_row
// 来自本轮实际请求。groups[i] 对应一个 WR，sges 中按组顺序存放该 WR 的各源行。
// 一个多SGE WRITE = NIC gather多个本地离散CPU行 -> 连续远端GPU地址，
// 不是多个WRITE，也不是CPU先将数据复制到连续缓冲区。
static int post_write(Channel&c,const Group*groups,unsigned n,const std::vector<ibv_sge>&sges){
    unsigned offset=0;
    for(unsigned q=0;q<unsigned(c.o.qps)&&q<n;++q)ibv_wr_start(ibv_qp_to_qp_ex(c.r.qps[q]));
    // 在本窗口内轮转分配：WR0->QP0，WR1->QP1，...，WR4->QP0（默认4QP）。
    // 每QP只标记最后一条WR为SIGNALED。该尾部CQE用于回收本QP之前的窗口数据。
    for(unsigned i=0;i<n;++i){auto*x=ibv_qp_to_qp_ex(c.r.qps[i%c.o.qps]);x->wr_id=i;x->wr_flags=i+c.o.qps>=n?IBV_SEND_SIGNALED:0;
        // offset 索引本窗口 SGE 数组；dst_row 是相对 staging 基址的行号。
        // CPU 调用 verbs 构造 WQE，真正的数据读取和发送由 NIC 完成。
        ibv_wr_rdma_write(x,c.peer.dst_rkey,c.peer.dst_addr+uint64_t(groups[i].dst_row)*ROW);ibv_wr_set_sge_list(x,groups[i].count,sges.data()+offset);offset+=groups[i].count;}
    int completions=0;
    for(unsigned q=0;q<unsigned(c.o.qps)&&q<n;++q){verbs_check(ibv_wr_complete(ibv_qp_to_qp_ex(c.r.qps[q])),"submit WRITE WQEs");++completions;}
    return completions;
}
static void publish_gpu_done(Channel&c,uint64_t epoch,unsigned count){
    // 调用者已经 poll 完本轮所有数据窗口；这里仍在每个所需数据QP各发一个DONE。
    // 每条QP的提交顺序是：data WR... -> 最后一个data WR -> DONE(epoch)。
    // QP之间没有这种隐含顺序，所以不能拿QP0的DONE代替QP1/2/3的完成。
    // requester GPU 必须等齐本轮所需QP的DONE，才允许重排/消费数据。
    for(unsigned q=0;q<count;++q){ibv_sge sg{reinterpret_cast<uintptr_t>(&epoch),8,0};ibv_send_wr wr{},*bad=nullptr;
        // INLINE 将8B epoch复制进WQE，网卡不异步引用栈上的epoch，lkey也无需注册。
        // 这是普通 RDMA WRITE 到 GPU 标志区，并不是 TCP response 或 WRITE_WITH_IMM。
        wr.wr_id=UINT64_MAX;wr.opcode=IBV_WR_RDMA_WRITE;wr.sg_list=&sg;wr.num_sge=1;wr.send_flags=IBV_SEND_INLINE|IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr=c.peer.done_addr+q*8;wr.wr.rdma.rkey=c.peer.done_rkey;verbs_check(ibv_post_send(c.r.qps[q],&wr,&bad),"publish GPU DONE");}
    // owner 回收DONE自身的CQE后才处理此通道下一epoch；GPU在看到DONE后即可前进，
    // 不需要回发网络ACK。同一Buffer顺序复用的约束由调用方CUDA stream维护。
    c.r.poll(count);
}
