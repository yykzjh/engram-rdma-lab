#pragma once
#include "storage.hpp"
#include "protocol.hpp"
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <algorithm>
#include <chrono>
#include <array>

struct Endpoint {
    uint32_t qpn = 0;
    std::array<uint8_t, 16> gid{};
};
struct Receiver {
    uint64_t data = 0, done = 0;
    uint32_t key = 0;
};

// 每个(owner,requester)对独立RC通道。bootstrap由PyTorch进程组交换，热路径没有TCP。
struct Channel {
    ibv_context *context = nullptr;
    ibv_pd *pd = nullptr;
    ibv_cq *cq = nullptr;
    ibv_mr *mr = nullptr;
    std::vector<ibv_qp *> qps;
    std::vector<Endpoint> endpoints;
    Receiver peer;
    int gid_index = 3, sl = 1;
    unsigned pool = 1024;
    Channel(const std::string &hca, int qcount, unsigned window, bool owner, void *memory, size_t bytes, int gid,
            int service_level)
        : gid_index(gid), sl(service_level), pool(window) {
        int count;
        auto devices = ibv_get_device_list(&count);
        ck(devices, "RDMA devices");
        for (int i = 0; i < count; ++i)
            if (hca == ibv_get_device_name(devices[i])) {
                mlx5dv_context_attr a{};
                a.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
                context = mlx5dv_open_device(devices[i], &a);
                break;
            }
        ibv_free_device_list(devices);
        ck(context, "open RDMA HCA");
        ibv_device_attr caps{};
        ck(!ibv_query_device(context, &caps), "query HCA");
        ck(!owner || caps.max_sge >= int(kMerge), "30 SGE support required");
        pd = ibv_alloc_pd(context);
        ck(pd, "allocate PD");
        cq = ibv_create_cq(context, pool + qcount + 32, nullptr, nullptr, 0);
        ck(cq, "create CQ");
        mr = ibv_reg_mr(pd, memory, bytes, IBV_ACCESS_LOCAL_WRITE | (owner ? 0 : IBV_ACCESS_REMOTE_WRITE));
        ck(mr, "register vocabulary/output");
        union ibv_gid gid_value{};
        ck(!ibv_query_gid(context, 1, gid_index, &gid_value), "query GID");
        for (int q = 0; q < qcount; ++q) {
            ibv_qp_init_attr_ex a{};
            a.pd = pd;
            a.send_cq = a.recv_cq = cq;
            a.qp_type = IBV_QPT_RC;
            a.cap.max_send_wr = (pool + qcount - 1) / qcount + 16;
            a.cap.max_recv_wr = 1;
            a.cap.max_send_sge = owner ? kMerge : 1;
            a.cap.max_recv_sge = 1;
            a.cap.max_inline_data = owner ? 8 : 0;
            a.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
            a.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
            mlx5dv_qp_init_attr d{};
            d.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
            d.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
            auto *qp = mlx5dv_create_qp(context, &a, &d);
            ck(qp, "create RC QP");
            qps.push_back(qp);
            Endpoint e;
            e.qpn = qp->qp_num;
            std::memcpy(e.gid.data(), &gid_value, 16);
            endpoints.push_back(e);
        }
    }
    void connect(const std::vector<Endpoint> &remote) {
        ck(remote.size() == qps.size(), "QP bootstrap count");
        for (size_t i = 0; i < qps.size(); ++i) {
            ibv_qp_attr a{};
            a.qp_state = IBV_QPS_INIT;
            a.port_num = 1;
            a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
            ck(!ibv_modify_qp(qps[i], &a, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS),
               "QP INIT");
            a = {};
            a.qp_state = IBV_QPS_RTR;
            a.path_mtu = IBV_MTU_4096;
            a.dest_qp_num = remote[i].qpn;
            a.rq_psn = 0x1234;
            a.max_dest_rd_atomic = 16;
            a.min_rnr_timer = 12;
            a.ah_attr.port_num = 1;
            a.ah_attr.is_global = 1;
            a.ah_attr.sl = sl;
            std::memcpy(&a.ah_attr.grh.dgid, remote[i].gid.data(), 16);
            a.ah_attr.grh.sgid_index = gid_index;
            a.ah_attr.grh.hop_limit = 64;
            ck(!ibv_modify_qp(qps[i], &a,
                              IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                  IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
               "QP RTR");
            a = {};
            a.qp_state = IBV_QPS_RTS;
            a.sq_psn = 0x1234;
            a.timeout = 18;
            a.retry_cnt = 5;
            a.rnr_retry = 5;
            a.max_rd_atomic = 16;
            ck(!ibv_modify_qp(qps[i], &a,
                              IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                  IBV_QP_MAX_QP_RD_ATOMIC),
               "QP RTS");
        }
    }
    void poll(unsigned count) {
        auto start = std::chrono::steady_clock::now();
        while (count) {
            ibv_wc wc[16];
            int n = ibv_poll_cq(cq, std::min(count, 16u), wc);
            ck(n >= 0, "CQ poll");
            for (int i = 0; i < n; ++i)
                if (wc[i].status != IBV_WC_SUCCESS)
                    throw std::runtime_error(std::string("WRITE CQE: ") + ibv_wc_status_str(wc[i].status));
            count -= n;
            ck(std::chrono::steady_clock::now() - start < std::chrono::seconds(30), "WRITE CQ timeout");
        }
    }
    // 一个窗口按active QPs轮转；每QP只有尾WR请求CQE，CPU回收后才复用本窗口。
    void write(const std::vector<ibv_sge> &sges, unsigned rows, unsigned base, unsigned active, unsigned row_bytes) {
        unsigned groups = (rows + kMerge - 1) / kMerge, used = std::min(active, groups), offset = 0;
        for (unsigned q = 0; q < used; ++q)
            ibv_wr_start(ibv_qp_to_qp_ex(qps[q]));
        for (unsigned i = 0; i < groups; ++i) {
            auto *x = ibv_qp_to_qp_ex(qps[i % active]);
            unsigned n = std::min(kMerge, rows - offset);
            x->wr_id = i;
            x->wr_flags = i + active >= groups ? IBV_SEND_SIGNALED : 0;
            ibv_wr_rdma_write(x, peer.key, peer.data + uint64_t(base + offset) * row_bytes);
            ibv_wr_set_sge_list(x, n, sges.data() + offset);
            offset += n;
        }
        for (unsigned q = 0; q < used; ++q)
            ck(!ibv_wr_complete(ibv_qp_to_qp_ex(qps[q])), "submit SGE WRITE");
        // CQE用于CPU回收本窗口的WR/SGE；GPU不读CQ，也不能把一个QP的CQE
        // 当成所有QP均可见。GPU侧完成依据仍是后续每QP独立写入的DONE。
        poll(used);
    }
    void done(uint64_t epoch, unsigned layer, unsigned used, unsigned world) {
        // 同一个QP上：所有该层data WR -> DONE(layer,owner,qp)。不同QP各有自己的DONE，
        // 不用QP0替代其他QP；层间也独立，layer0就绪无需等待layer1的传输/重排。
        for (unsigned q = 0; q < used; ++q) {
            ibv_sge s{reinterpret_cast<uintptr_t>(&epoch), 8, 0};
            ibv_send_wr w{}, *bad = nullptr;
            w.opcode = IBV_WR_RDMA_WRITE;
            w.sg_list = &s;
            w.num_sge = 1;
            w.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
            w.wr.rdma.remote_addr = peer.done + (uint64_t(layer) * world * qps.size() + q) * 8;
            w.wr.rdma.rkey = peer.key;
            ck(!ibv_post_send(qps[q], &w, &bad), "submit per-layer per-QP DONE");
        }
        poll(used);
    }
    ~Channel() {
        for (auto *q : qps)
            ibv_destroy_qp(q);
        if (mr)
            ibv_dereg_mr(mr);
        if (cq)
            ibv_destroy_cq(cq);
        if (pd)
            ibv_dealloc_pd(pd);
        if (context)
            ibv_close_device(context);
    }
};
