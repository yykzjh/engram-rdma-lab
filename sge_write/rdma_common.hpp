#pragma once
// RDMA resources and homogeneous-host control transport for the standalone benchmark.
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda.h>
#include <nvrtc.h>
#include <thread>
#include <atomic>
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
constexpr uint64_t MAGIC = 0x5347455752495445ULL;
constexpr size_t ROW = 256, GUARD = 4096;
struct Options {
    bool owner=false;
    bool async_check=false;
    std::string host,device="mlx5_bond_0",ips="127.0.0.1,127.0.0.1",distribution="stratified";
    int port=26700,gid=3,gpu=0,pool=1024,qps=4,rank=0,world=8,rows=24,groups=512,rounds=20,warmups=1;
    int table_rows=524288+524309,merge_rows=30,owner_windows=1,owner_poll_us=0,gin_contexts=4;
    std::vector<int> entries{524288,524309};
    std::string layout="merged";
    bool done_gpu=true,adaptive_grid=true,profile_routes=false,hybrid_local=true,compact_done=true,gpu_request=true,vary_indices=true;
};
static Options options(int argc,char**argv){
    Options o;
    auto env_flag=[](const char* name){const char* v=std::getenv(name);if(!v)return false;check(std::string(v)=="0"||std::string(v)=="1",std::string(name)+" must be 0 or 1");return *v=='1';};
    o.hybrid_local=env_flag("ENGRAM_WRITE_C2C");
    o.async_check=env_flag("ENGRAM_WRITE_ASYNC_CHECK");
    for(int i=1;i<argc;++i){
        std::string k=argv[i];check(i+1<argc,"missing option value");std::string v=argv[++i];
        if(k=="--ips")o.ips=v; else if(k=="--distribution")o.distribution=v;
        else if(k=="--tokens")o.groups=std::stoi(v);else if(k=="--rounds")o.rounds=std::stoi(v);
        else if(k=="--warmups")o.warmups=std::stoi(v);else if(k=="--rank")o.rank=std::stoi(v);
        else if(k=="--port")o.port=std::stoi(v);else if(k=="--gid")o.gid=std::stoi(v);
        else if(k=="--pool")o.pool=std::stoi(v);else if(k=="--qps")o.qps=std::stoi(v);
        else if(k=="--merge-rows")o.merge_rows=std::stoi(v);
        else if(k=="--owner-windows")o.owner_windows=std::stoi(v);
        else if(k=="--owner-poll-us")o.owner_poll_us=std::stoi(v);
        else if(k=="--methods")check(v=="write","WRITE only");
        else if(k=="--layout")check(v=="merged","merged output required");
        else if(k=="--engine")check(v=="ex","extended verbs required");
        else if(k=="--protocol")check(v=="frame","whole-fetch protocol required");
        else if(k=="--profile-routes")check(v=="0","primary timing has no extra events");
        else if(k=="--compact-qp"||k=="--compact-request"||k=="--done-gpu"||k=="--gpu-request"||k=="--vary-indices"||k=="--gpu-graph"||k=="--adaptive-grid"||k=="--compact-done")check(v=="1","required WRITE path feature");
        else throw std::runtime_error("unknown SGE WRITE option "+k);
    }
    check(o.rank>=0&&o.rank<8&&o.groups>=1&&o.groups<=262144,"invalid rank/tokens");
    check(o.rounds>=1&&o.rounds<=30&&o.warmups>=0&&o.warmups<=3,"invalid rounds");
    check(o.pool>=1&&o.pool<=4096&&o.qps>=1&&o.qps<=16&&o.merge_rows>=1&&o.merge_rows<=30,"invalid QP/SGE capacity");
    check(o.owner_windows>=1&&o.owner_windows<=16&&o.owner_poll_us>=0&&o.owner_poll_us<=100,"invalid owner settings");
    check(size_t((o.pool+o.qps-1)/o.qps)*o.owner_windows<=16384,"SQ capacity bound");
    check(o.distribution=="stratified"||o.distribution=="uniform","invalid distribution");
    o.gpu=o.rank%4;o.device="mlx5_bond_"+std::to_string(o.gpu);return o;
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
inline std::atomic<uint64_t> control_socket_sends{0},control_socket_recvs{0};
struct Socket {
    int fd = -1;
    ~Socket() { if (fd >= 0) close(fd); }
    void tune() {
        int one = 1;
        check(!setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)), "TCP_NODELAY");
        timeval t{60, 0};
        check(!setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t, sizeof(t)), "SO_RCVTIMEO");
        check(!setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &t, sizeof(t)), "SO_SNDTIMEO");
    }
    void send(const void* p, size_t n, int flags = 0) {
        ++control_socket_sends;
        const char* b = static_cast<const char*>(p);
        while (n) {
            ssize_t s = ::send(fd, b, n, MSG_NOSIGNAL | flags);
            if (s < 0 && errno == EINTR) continue;
            check(s > 0, "control send"); b += s; n -= s;
        }
    }
    void recv(void* p, size_t n) {
        ++control_socket_recvs;
        char* b = static_cast<char*>(p);
        while (n) {
            ssize_t s = ::recv(fd, b, n, 0);
            if (s < 0 && errno == EINTR) continue;
            check(s > 0, "control recv"); b += s; n -= s;
        }

    }
    template<class T> void send(const T& v) { send(&v, sizeof(v)); }
    template<class T> void recv(T& v) { recv(&v, sizeof(v)); }
};
static void connect(Socket& s, const Options& o) {
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
    uint64_t magic=MAGIC;uint32_t version=11,bytes=sizeof(Hello);
    uint32_t pool=0,qps=0,groups=0,table_rows=0,merge_rows=0;
    uint64_t dst_addr=0,done_addr=0;uint32_t dst_rkey=0,done_rkey=0;
};
struct Endpoint {uint32_t qpn=0,psn=0x1234;uint8_t gid[16]{};};
static uint64_t pattern(uint32_t row, unsigned word) {
    return (0x9e3779b97f4a7c15ULL * (uint64_t(row) + 1)) ^
           (0xd1b54a32d192ed03ULL * (uint64_t(word) + 1));
}
struct Resources {
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    std::vector<ibv_qp*> qps;
    ibv_mr* mr = nullptr;
    void* cpu = nullptr;
    bool borrowed_cpu = false;
    CUcontext cuda = nullptr;
    CUdevice cuda_device = -1;
    CUdeviceptr gpu = 0;
    ~Resources() {
        for (auto* q : qps) ibv_destroy_qp(q);
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
          device.max_sge >= o.merge_rows), "MTU/SGE capability");
    r.pd = ibv_alloc_pd(r.ctx); check(r.pd != nullptr, "alloc pd");
    r.cq = ibv_create_cq(r.ctx, o.pool * o.qps + 32, nullptr, nullptr, 0);
    check(r.cq != nullptr, "create cq");
}
static void create_qps(Resources& r,const Options&o){
    for(int i=0;i<o.qps;++i){
        ibv_qp_init_attr_ex a{};a.pd=r.pd;a.send_cq=r.cq;a.recv_cq=r.cq;a.qp_type=IBV_QPT_RC;
        a.cap.max_send_wr=((o.pool+o.qps-1)/o.qps)*(o.owner?o.owner_windows:1)+16;
        a.cap.max_recv_wr=1;a.cap.max_send_sge=o.owner?std::max(o.rows,o.merge_rows):1;a.cap.max_recv_sge=1;
        a.cap.max_inline_data=o.owner?8:0;a.comp_mask=IBV_QP_INIT_ATTR_PD|IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        a.send_ops_flags=IBV_QP_EX_WITH_RDMA_WRITE;
        mlx5dv_qp_init_attr d{};d.comp_mask=MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
        d.create_flags=MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
        auto*q=mlx5dv_create_qp(r.ctx,&a,&d);check(q!=nullptr,"create WRITE QP");
        check(a.cap.max_send_sge>=unsigned(o.owner?o.merge_rows:1),"QP SGE capacity");r.qps.push_back(q);
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
