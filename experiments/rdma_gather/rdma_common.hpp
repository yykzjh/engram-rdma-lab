#pragma once
// RDMA resources and homogeneous-host control transport for the standalone benchmark.
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda.h>
#include <nvrtc.h>
#include <thread>
#include <mutex>
#include <exception>
#include <sched.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <condition_variable>
#include <functional>
#include <fstream>

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a) {
    return std::chrono::duration<double, std::micro>(Clock::now() - a).count();
}
static void check(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what + ": " + std::strerror(errno));
}
static void cuda_check(CUresult rc, const char* what) {
    if (rc != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(rc, &s);
        std::fprintf(stderr, "CUDA ERROR %s: %s (%d)\n", what, s ? s : "unknown", int(rc));
        throw std::runtime_error(std::string(what) + ": " + (s ? s : "CUDA error"));
    }
}
static void verbs_check(int rc, const char* what) {
    if (rc) throw std::runtime_error(std::string(what) + ": rc=" + std::to_string(rc));
}
constexpr uint64_t MAGIC = 0x554d525347453031ULL;
constexpr size_t ROW = 256, GUARD = 4096;
struct Options {
    bool owner = false;
    std::string host, device = "mlx5_bond_0", ips = "127.0.0.1,127.0.0.1", distribution = "stratified";
    int port = 39300, gid = 3, gpu = 0, pool = 256, qps = 4, rank = 0, world = 8;
    int rows = 24, groups = 512, rounds = 5, warmups = 1, table_rows = 524288 + 524309;
    std::vector<int> entries{524288, 524309};
    std::string indices_dir;
    bool planner_profile = false;
    std::string methods = "both";
    std::string layout = "staged";
    int merge_rows = 30;
    std::string engine = "baseline";
    int pipeline = 0;
    bool fast_plan = false;
    bool compact_qp = false;
    bool aligned_counts = false;
    bool quick_ack = false, coalesce_control = false;
    std::string protocol = "window";
    std::string target_umr = "none";
    int umr_keys = 128;
    bool vary_indices = false;
    bool done_gpu = false;
    bool compact_request = false, gpu_request = false;
    bool extended_sge = false;
    bool gpu_graph = false, adaptive_grid = false;
    bool profile_routes = true;
    int owner_windows = 1;
    int owner_poll_us = 1;
    bool compact_done = false, hybrid_local = false, host_vmm_table = false;
    bool gpu_get = false, get_batch = false;
    int gin_contexts = 4;
    std::string dump_indices;
};
static Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--owner") { o.owner = true; continue; }
        if (k == "--planner-profile") { o.planner_profile = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + k);
        std::string v = argv[++i];
        if (k == "--ips") o.ips = v;
        else if (k == "--distribution") o.distribution = v;
        else if (k == "--indices-dir") o.indices_dir = v;
        else if (k == "--methods") o.methods = v;
        else if (k == "--layout") o.layout = v;
        else if (k == "--merge-rows") o.merge_rows = std::stoi(v);
        else if (k == "--engine") o.engine = v;
        else if (k == "--pipeline") o.pipeline = std::stoi(v);
        else if (k == "--fast-plan") o.fast_plan = std::stoi(v) != 0;
        else if (k == "--compact-qp") o.compact_qp = std::stoi(v) != 0;
        else if (k == "--aligned-counts") o.aligned_counts = std::stoi(v) != 0;
        else if (k == "--quick-ack") o.quick_ack = std::stoi(v) != 0;
        else if (k == "--coalesce-control") o.coalesce_control = std::stoi(v) != 0;
        else if (k == "--protocol") o.protocol = v;
        else if (k == "--target-umr") o.target_umr = v;
        else if (k == "--umr-keys") o.umr_keys = std::stoi(v);
        else if (k == "--vary-indices") o.vary_indices = std::stoi(v) != 0;
        else if (k == "--done-gpu") o.done_gpu = std::stoi(v) != 0;
        else if (k == "--compact-request") o.compact_request = std::stoi(v) != 0;
        else if (k == "--gpu-request") o.gpu_request = std::stoi(v) != 0;
        else if (k == "--extended-sge") o.extended_sge = std::stoi(v) != 0;
        else if (k == "--gpu-graph") o.gpu_graph = std::stoi(v) != 0;
        else if (k == "--adaptive-grid") o.adaptive_grid = std::stoi(v) != 0;
        else if (k == "--profile-routes") o.profile_routes = std::stoi(v) != 0;
        else if (k == "--owner-windows") o.owner_windows = std::stoi(v);
        else if (k == "--owner-poll-us") o.owner_poll_us = std::stoi(v);
        else if (k == "--compact-done") o.compact_done = std::stoi(v) != 0;
        else if (k == "--hybrid-local") o.hybrid_local = std::stoi(v) != 0;
        else if (k == "--host-vmm-table") o.host_vmm_table = std::stoi(v) != 0;
        else if (k == "--gpu-get") o.gpu_get = std::stoi(v) != 0;
        else if (k == "--get-batch") o.get_batch = std::stoi(v) != 0;
        else if (k == "--gin-contexts") o.gin_contexts = std::stoi(v);
        else if (k == "--dump-indices") o.dump_indices = v;
        else if (k == "--port") o.port = std::stoi(v);
        else if (k == "--gid") o.gid = std::stoi(v);
        else if (k == "--pool") o.pool = std::stoi(v);
        else if (k == "--qps") o.qps = std::stoi(v);
        else if (k == "--rank") o.rank = std::stoi(v);
        else if (k == "--tokens") o.groups = std::stoi(v);
        else if (k == "--rounds") o.rounds = std::stoi(v);
        else if (k == "--warmups") o.warmups = std::stoi(v);
        else throw std::runtime_error("Unknown option " + k);
    }
    check(o.pool >= 1 && o.pool <= 4096 && o.qps >= 1 && o.qps <= 16, "invalid pool/QPs");
    check(o.rank >= 0 && o.rank < 8 && o.groups >= 1 && o.groups <= 262144 &&
          o.rounds >= 1 && o.rounds <= 30 && o.warmups >= 0 && o.warmups <= 3, "invalid workload");
    check(o.distribution == "stratified" || o.distribution == "uniform", "invalid distribution");
    check(o.methods == "both" || o.methods == "write", "invalid methods");
    check(o.layout == "staged" || o.layout == "direct" || o.layout == "merged", "invalid output layout");
    check(o.merge_rows >= 1 && o.merge_rows <= (o.extended_sge ? 52 : 30), "merge rows exceeds selected SGE capability");
    check(o.engine == "baseline" || o.engine == "reuse" || o.engine == "ex", "invalid WRITE engine");
    check(o.pipeline >= 0 && o.pipeline <= 16 && (!o.pipeline || o.methods == "write"), "invalid WRITE pipeline");
    check(!o.fast_plan || o.layout == "merged", "flat planner requires merged layout");
    check(o.protocol == "window" || o.protocol == "frame", "invalid control protocol");
    check(o.protocol != "frame" || (o.methods == "write" && !o.pipeline), "frame protocol is WRITE-only without window pipeline");
    check(o.target_umr == "none" || o.target_umr == "adaptive" || o.target_umr == "strict", "invalid target UMR mode");
    check(o.target_umr == "none" || (o.protocol == "frame" && o.layout == "merged"), "target UMR requires merged frame WRITE");
    check(o.umr_keys >= 8 && o.umr_keys <= 16384, "target MKey pool bounds");
    check(!o.done_gpu || o.protocol == "frame", "GPU DONE requires frame protocol");
    check(!o.compact_request || (o.protocol == "frame" && o.layout == "merged"), "compact requests require merged frame protocol");
    check(o.owner_windows>=1 && o.owner_windows<=16 &&
          (o.owner_windows==1 || (o.gpu_request && o.engine=="ex" && o.protocol=="frame" && o.methods=="write")),"invalid owner WRITE wave");
    check(size_t((o.pool+o.qps-1)/o.qps)*o.owner_windows<=16384,"owner SQ capacity bound");
    check(o.owner_poll_us>=0 && o.owner_poll_us<=100,"owner polling sleep bound");
    check(!o.gpu_graph || (o.gpu_request && o.target_umr=="none"),"graph requires GPU requests without CPU UMR bridge");
    check(!o.compact_done || o.gpu_request,"compact DONE requires GPU routing");
    check(!o.hybrid_local || o.gpu_graph,"hybrid local CPU gather requires graph");
    check(o.gin_contexts>=1 && o.gin_contexts<=256,"GIN context bound");
    check(!o.gpu_get || (o.hybrid_local && o.gpu_graph && o.groups<=512),"direct GET experiment supports hybrid graph up to512 tokens");
    check(!o.get_batch || o.gpu_get,"batched GET requires GPU GET");
    check(!o.gpu_get || uint64_t(o.groups)*o.rows*o.entries.size()<=uint64_t(o.gin_contexts)*1024,
          "GET worst-case requests per context exceed queue depth");
    if(o.gpu_get)o.profile_routes=false;
    if(o.hybrid_local)o.host_vmm_table=true;
    o.gpu = o.rank % 4; o.device = "mlx5_bond_" + std::to_string(o.gpu);
    return o;
}
static std::string node_ip(const Options& o, int rank) {
    size_t comma = o.ips.find(','); check(comma != std::string::npos, "two IPs required");
    return rank < 4 ? o.ips.substr(0, comma) : o.ips.substr(comma + 1);
}
static void pin_core(const Options& o, int peer, bool owner) {
    const int local = o.rank % 4;
    const int bases[] = {0, 16, 72, 88};
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(bases[local] + (owner ? 0 : 32) + peer, &set);
    check(!pthread_setaffinity_np(pthread_self(), sizeof(set), &set), "pin CPU");
}
struct Socket {
    int fd = -1;
    bool quick_ack = false;
    ~Socket() { if (fd >= 0) close(fd); }
    void tune() {
        int one = 1;
        check(!setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)), "TCP_NODELAY");
        timeval t{60, 0};
        check(!setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t, sizeof(t)), "SO_RCVTIMEO");
        check(!setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &t, sizeof(t)), "SO_SNDTIMEO");
    }
    void send(const void* p, size_t n, int flags = 0) {
        const char* b = static_cast<const char*>(p);
        while (n) {
            ssize_t s = ::send(fd, b, n, MSG_NOSIGNAL | flags);
            if (s < 0 && errno == EINTR) continue;
            check(s > 0, "control send"); b += s; n -= s;
        }
    }
    void recv(void* p, size_t n) {
        char* b = static_cast<char*>(p);
        while (n) {
            ssize_t s = ::recv(fd, b, n, 0);
            if (s < 0 && errno == EINTR) continue;
            check(s > 0, "control recv"); b += s; n -= s;
        }
        if (quick_ack) {
            int one = 1;
            check(!setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one)), "TCP_QUICKACK");
        }
    }
    template<class T> void send(const T& v) { send(&v, sizeof(v)); }
    template<class T> void recv(T& v) { recv(&v, sizeof(v)); }
};
static void connect(Socket& s, const Options& o) {
    s.quick_ack = o.quick_ack;
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(o.port);
    if (o.owner) {
        Socket listen;
        listen.fd = socket(AF_INET, SOCK_STREAM, 0); check(listen.fd >= 0, "socket");
        int one = 1;
        setsockopt(listen.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        check(!bind(listen.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), "bind");
        check(!::listen(listen.fd, 1), "listen");
        std::fprintf(stderr, "LISTEN port=%d\n", o.port); std::fflush(stderr);
        s.fd = accept(listen.fd, nullptr, nullptr); check(s.fd >= 0, "accept");
    } else {
        s.fd = socket(AF_INET, SOCK_STREAM, 0); check(s.fd >= 0, "socket");
        check(inet_pton(AF_INET, o.host.c_str(), &addr.sin_addr) == 1, "IPv4 host");
        auto start = Clock::now();
        while (::connect(s.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr))) {
            check(errno == ECONNREFUSED || errno == EINTR, "connect");
            check(us(start) < 60000000, "connect timeout");
            close(s.fd); usleep(10000); s.fd = socket(AF_INET, SOCK_STREAM, 0);
        }
    }
    s.tune();
}
// This private control protocol is for homogeneous little-endian hosts only.
// Explicit version/ABI sizes are checked before any buffer lengths are trusted.
struct Hello {
    uint64_t magic = MAGIC;
    uint32_t version = 8, bytes = sizeof(Hello);
    uint32_t pool, qps, rows, groups, table_rows;
    uint32_t reserved = 0;
    uint64_t dst_addr = 0;
    uint32_t dst_rkey = 0, pad = 0;
    uint32_t engine = 0, pipeline = 0;
    uint32_t protocol = 0;
    uint32_t done_gpu = 0, done_rkey = 0;
    uint64_t done_addr = 0;
    uint32_t compact_request = 0, gpu_request = 0;
    uint32_t target_umr = 0;
    uint32_t owner_windows = 1;
    uint32_t compact_done = 0, hybrid_local = 0;
};
struct Endpoint { uint32_t qpn = 0, psn = 0x1234; uint8_t gid[16]{}; };
struct Command { uint32_t mode, round, warmup, groups; };
struct Stats {
    double owner_receive_us = 0, layout_us = 0, umr_us = 0;
    double umr_post_us = 0, data_us = 0, data_post_us = 0;
    double wr_build_us = 0, verbs_post_us = 0, cq_wait_us = 0;
    double owner_ack_us = 0, owner_notify_us = 0;
};
struct Response { uint32_t groups = 0, status = 0; Stats stats; uint64_t fetch_id = 0; };
static uint64_t pattern(uint32_t row, unsigned word) {
    return (0x9e3779b97f4a7c15ULL * (uint64_t(row) + 1)) ^
           (0xd1b54a32d192ed03ULL * (uint64_t(word) + 1));
}
struct Resources {
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    std::vector<ibv_qp*> qps;
    std::vector<mlx5dv_mkey*> keys;
    ibv_mr* mr = nullptr;
    void* cpu = nullptr;
    bool borrowed_cpu = false;
    CUcontext cuda = nullptr;
    CUdevice cuda_device = -1;
    CUdeviceptr gpu = 0;
    ~Resources() {
        for (auto* q : qps) ibv_destroy_qp(q);
        for (auto* k : keys) mlx5dv_destroy_mkey(k);
        if (mr) ibv_dereg_mr(mr);
        if (gpu) cuMemFree(gpu);
        if (cuda) cuDevicePrimaryCtxRelease(cuda_device);
        if (!borrowed_cpu) free(cpu);
        if (cq) ibv_destroy_cq(cq);
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
    void poll(int count) {
        auto start = Clock::now();
        while (count) {
            ibv_wc wc[16];
            int n = ibv_poll_cq(cq, std::min(count, 16), wc);
            check(n >= 0, "ibv_poll_cq");
            for (int i = 0; i < n; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS)
                    throw std::runtime_error("CQE: " + std::string(ibv_wc_status_str(wc[i].status)) +
                        " vendor=" + std::to_string(wc[i].vendor_err) + " wr_id=" + std::to_string(wc[i].wr_id));
            }
            count -= n;
            if (us(start) > 30000000) throw std::runtime_error("CQ timeout");
        }
    }
};
static void open_hca(Resources& r, const Options& o) {
    int n = 0; auto** devices = ibv_get_device_list(&n); check(devices != nullptr, "device list");
    for (int i = 0; i < n; ++i) {
        if (o.device == ibv_get_device_name(devices[i])) {
            mlx5dv_context_attr attr{}; attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
            r.ctx = mlx5dv_open_device(devices[i], &attr); break;
        }
    }
    ibv_free_device_list(devices); check(r.ctx != nullptr, "open " + o.device);
    ibv_port_attr port{}; ibv_device_attr device{};
    verbs_check(ibv_query_port(r.ctx, 1, &port), "query port MTU");
    verbs_check(ibv_query_device(r.ctx, &device), "query device limits");
    std::fprintf(stderr, "HCA device=%s active_mtu=%u max_mtu=%u max_sge=%d path_mtu=4096\n",
                 o.device.c_str(), 128U << unsigned(port.active_mtu),
                 128U << unsigned(port.max_mtu), device.max_sge);
    check(port.active_mtu >= IBV_MTU_4096 && (!o.owner ||
          o.extended_sge || device.max_sge >= (o.layout == "merged" ? o.merge_rows : o.rows)), "MTU/SGE capability");
    r.pd = ibv_alloc_pd(r.ctx); check(r.pd != nullptr, "alloc pd");
    r.cq = ibv_create_cq(r.ctx, o.pool * o.qps + 32, nullptr, nullptr, 0);
    check(r.cq != nullptr, "create cq");
}
static void create_qps(Resources& r, const Options& o) {
    for (int i = 0; i < o.qps; ++i) {
        const bool needs_umr = i == 0 && ((o.owner && (!o.compact_qp || o.methods != "write")) ||
                                        (!o.owner && o.target_umr != "none"));
        ibv_qp_init_attr_ex a{};
        a.pd = r.pd; a.send_cq = r.cq; a.recv_cq = r.cq; a.qp_type = IBV_QPT_RC;
        // Every data window is drained before the next is posted. Data WRs are
        // distributed round-robin; UMR configuration alone posts all on QP0.
        a.cap.max_send_wr = (o.compact_qp && !needs_umr ? (o.pool + o.qps - 1) / o.qps : o.pool) *
                           (o.owner ? o.owner_windows : 1) + 16;
        a.cap.max_recv_wr = 1;
        a.cap.max_send_sge = o.owner ? std::max(o.rows, o.layout == "merged" ? o.merge_rows : o.rows) : 1;
        a.cap.max_recv_sge = 1;
        a.cap.max_inline_data = needs_umr ? 828 : 0;
        if (o.owner && o.done_gpu) a.cap.max_inline_data = std::max(a.cap.max_inline_data, 8U);
        a.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        a.send_ops_flags = IBV_QP_EX_WITH_RDMA_READ | IBV_QP_EX_WITH_RDMA_WRITE;
        mlx5dv_qp_init_attr d{};
        d.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
        d.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
        if (needs_umr) {
            d.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
            d.send_ops_flags = MLX5DV_QP_EX_WITH_MKEY_CONFIGURE;
        }
        auto* q = mlx5dv_create_qp(r.ctx, &a, &d); check(q != nullptr, "create qp");
        check(a.cap.max_send_sge >= unsigned(o.owner ? std::max(o.rows, o.layout == "merged" ? o.merge_rows : o.rows) : 1), "returned QP SGE capacity");
        r.qps.push_back(q);
    }
}
static void activate(Resources& r, const Options& o, Socket& sock) {
    union ibv_gid gid{};
    verbs_check(ibv_query_gid(r.ctx, 1, o.gid, &gid), "query gid");
    for (auto* q : r.qps) {
        Endpoint local{}, remote{};
        local.qpn = q->qp_num; std::memcpy(local.gid, &gid, 16);
        sock.send(local); sock.recv(remote);
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_INIT; a.port_num = 1; a.pkey_index = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        verbs_check(ibv_modify_qp(q, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                    IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "qp INIT");
        a = {}; a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_4096;
        a.dest_qp_num = remote.qpn; a.rq_psn = remote.psn;
        a.max_dest_rd_atomic = 16; a.min_rnr_timer = 12;
        a.ah_attr.port_num = 1; a.ah_attr.is_global = 1;
        std::memcpy(&a.ah_attr.grh.dgid, remote.gid, 16);
        a.ah_attr.grh.sgid_index = o.gid; a.ah_attr.grh.hop_limit = 64;
        verbs_check(ibv_modify_qp(q, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                    IBV_QP_MIN_RNR_TIMER), "qp RTR");
        a = {}; a.qp_state = IBV_QPS_RTS; a.sq_psn = local.psn;
        a.timeout = 18; a.retry_cnt = 5; a.rnr_retry = 5; a.max_rd_atomic = 16;
        verbs_check(ibv_modify_qp(q, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                    IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC), "qp RTS");
    }
    uint32_t ready = 1, peer = 0; sock.send(ready); sock.recv(peer); check(peer == 1, "peer ready");
}
