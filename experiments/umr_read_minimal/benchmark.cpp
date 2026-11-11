// Standalone CPU-table -> remote GPU gather microbenchmark.
// Build: g++ -std=c++17 -O3 -Wall -Wextra -I/usr/local/cuda/include
//        benchmark.cpp -o rdma_gather_bench -libverbs -lmlx5 -lcuda
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda.h>
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
    std::string host, device = "mlx5_bond_0";
    int port = 39143, gid = 3, gpu = 0, pool = 1024, qps = 4;
    int rows = 24, groups = 4096, rounds = 8, warmups = 2, table_rows = 1 << 20;
};
static Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--owner") { o.owner = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + k);
        std::string v = argv[++i];
        if (k == "--host") o.host = v;
        else if (k == "--device") o.device = v;
        else if (k == "--port") o.port = std::stoi(v);
        else if (k == "--gid") o.gid = std::stoi(v);
        else if (k == "--gpu") o.gpu = std::stoi(v);
        else if (k == "--pool") o.pool = std::stoi(v);
        else if (k == "--qps") o.qps = std::stoi(v);
        else if (k == "--rows") o.rows = std::stoi(v);
        else if (k == "--groups") o.groups = std::stoi(v);
        else if (k == "--rounds") o.rounds = std::stoi(v);
        else if (k == "--warmups") o.warmups = std::stoi(v);
        else if (k == "--table-rows") o.table_rows = std::stoi(v);
        else throw std::runtime_error("Unknown option " + k);
    }
    // Keep bounded allocations and remain inside advertised SGE contract.
    check(o.pool >= 1 && o.pool <= 4096 && o.qps >= 1 && o.qps <= 16,
          "pool must be 1..4096, qps 1..16");
    check(o.rows >= 1 && o.rows <= 30 && o.groups >= 1 && o.groups <= (1 << 19),
          "rows must be 1..30, groups 1..524288");
    check(o.table_rows >= 64 && o.table_rows <= (1 << 24) && o.rounds >= 1 &&
          o.rounds <= 100 && o.warmups >= 0 && o.warmups <= 10, "invalid workload size");
    return o;
}
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
    void send(const void* p, size_t n) {
        const char* b = static_cast<const char*>(p);
        while (n) {
            ssize_t s = ::send(fd, b, n, MSG_NOSIGNAL);
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
        check(!::connect(s.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), "connect");
    }
    s.tune();
}
// This private control protocol is for homogeneous little-endian hosts only.
// Explicit version/ABI sizes are checked before any buffer lengths are trusted.
struct Hello {
    uint64_t magic = MAGIC;
    uint32_t version = 1, bytes = sizeof(Hello);
    uint32_t pool, qps, rows, groups, table_rows;
    uint32_t reserved = 0;
    uint64_t dst_addr = 0;
    uint32_t dst_rkey = 0, pad = 0;
};
struct Endpoint { uint32_t qpn = 0, psn = 0x1234; uint8_t gid[16]{}; };
struct Command { uint32_t mode, round, warmup, groups; };
struct Stats {
    double owner_receive_us = 0, layout_us = 0, umr_us = 0;
    double umr_post_us = 0, data_us = 0, data_post_us = 0;
};
struct Response { uint32_t groups = 0, status = 0; Stats stats; };
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
    CUcontext cuda = nullptr;
    CUdevice cuda_device = -1;
    CUdeviceptr gpu = 0;
    ~Resources() {
        for (auto* q : qps) ibv_destroy_qp(q);
        for (auto* k : keys) mlx5dv_destroy_mkey(k);
        if (mr) ibv_dereg_mr(mr);
        if (gpu) cuMemFree(gpu);
        if (cuda) cuDevicePrimaryCtxRelease(cuda_device);
        free(cpu);
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
    r.pd = ibv_alloc_pd(r.ctx); check(r.pd != nullptr, "alloc pd");
    r.cq = ibv_create_cq(r.ctx, o.pool * o.qps + 32, nullptr, nullptr, 0);
    check(r.cq != nullptr, "create cq");
}
static void create_qps(Resources& r, const Options& o) {
    for (int i = 0; i < o.qps; ++i) {
        ibv_qp_init_attr_ex a{};
        a.pd = r.pd; a.send_cq = r.cq; a.recv_cq = r.cq; a.qp_type = IBV_QPT_RC;
        a.cap.max_send_wr = o.pool + 16; a.cap.max_recv_wr = 1;
        a.cap.max_send_sge = o.owner ? o.rows : 1; a.cap.max_recv_sge = 1;
        a.cap.max_inline_data = o.owner && i == 0 ? 828 : 0;
        a.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        a.send_ops_flags = IBV_QP_EX_WITH_RDMA_READ | IBV_QP_EX_WITH_RDMA_WRITE;
        mlx5dv_qp_init_attr d{};
        d.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
        d.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
        if (o.owner && i == 0) {
            d.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
            d.send_ops_flags = MLX5DV_QP_EX_WITH_MKEY_CONFIGURE;
        }
        auto* q = mlx5dv_create_qp(r.ctx, &a, &d); check(q != nullptr, "create qp");
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
static void memory(Resources& r, const Options& o) {
    auto t = Clock::now();
    if (o.owner) {
        size_t bytes = size_t(o.table_rows) * ROW;
        check(!posix_memalign(&r.cpu, 4096, bytes), "CPU table allocation");
        for (int row = 0; row < o.table_rows; ++row)
            for (unsigned w = 0; w < ROW / 8; ++w)
                static_cast<uint64_t*>(r.cpu)[size_t(row) * (ROW / 8) + w] = pattern(row, w);
        r.mr = ibv_reg_mr(r.pd, r.cpu, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
        check(r.mr != nullptr, "register CPU table");
        auto kt = Clock::now();
        for (int i = 0; i < o.pool; ++i) {
            mlx5dv_mkey_init_attr a{}; a.pd = r.pd;
            a.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT; a.max_entries = o.rows;
            auto* k = mlx5dv_create_mkey(&a); check(k != nullptr, "create pooled MKey");
            r.keys.push_back(k);
        }
        std::printf("{\"kind\":\"pool\",\"mkeys\":%d,\"entries_each\":%d,\"create_ms\":%.6f}\n",
                    o.pool, o.rows, us(kt) / 1000);
    } else {
        cuda_check(cuInit(0), "cuInit");
        cuda_check(cuDeviceGet(&r.cuda_device, o.gpu), "cuDeviceGet");
        cuda_check(cuDevicePrimaryCtxRetain(&r.cuda, r.cuda_device), "retain primary context");
        cuda_check(cuCtxSetCurrent(r.cuda), "set current context");
        size_t bytes = size_t(o.groups) * o.rows * ROW + 2 * GUARD;
        cuda_check(cuMemAlloc(&r.gpu, bytes), "cuMemAlloc");
        unsigned one = 1;
        cuda_check(cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, r.gpu), "SYNC_MEMOPS");
        r.mr = ibv_reg_mr(r.pd, reinterpret_cast<void*>(r.gpu), bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        check(r.mr != nullptr, "register GPU memory (requires GPUDirect RDMA)");
    }
    std::fprintf(stderr, "memory setup %.3f ms\n", us(t) / 1000);
}
static void layouts(const Resources& r, const Options& o, const uint32_t* indices,
                    int n, std::vector<ibv_sge>& sges) {
    sges.resize(size_t(n) * o.rows);
    for (size_t i = 0; i < sges.size(); ++i) {
        check(indices[i] < uint32_t(o.table_rows), "row index out of range");
        sges[i].addr = reinterpret_cast<uintptr_t>(r.cpu) + size_t(indices[i]) * ROW;
        sges[i].length = ROW; sges[i].lkey = r.mr->lkey;
    }
}
static void configure(Resources& r, const Options& o, const std::vector<ibv_sge>& sges,
                      int n, Stats& stats) {
    auto t = Clock::now();
    auto* x = ibv_qp_to_qp_ex(r.qps[0]); auto* d = mlx5dv_qp_ex_from_ibv_qp_ex(x);
    ibv_wr_start(x);
    for (int i = 0; i < n; ++i) {
        x->wr_id = i; x->wr_flags = IBV_SEND_INLINE | (i == n - 1 ? IBV_SEND_SIGNALED : 0);
        mlx5dv_mkey_conf_attr a{};
        mlx5dv_wr_mkey_configure(d, r.keys[i], 2, &a);
        mlx5dv_wr_set_mkey_access_flags(d, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
        mlx5dv_wr_set_mkey_layout_list(d, o.rows, sges.data() + size_t(i) * o.rows);
    }
    verbs_check(ibv_wr_complete(x), "submit UMR configurations");
    stats.umr_post_us = us(t);
    r.poll(1); stats.umr_us = us(t);
}
// Post linked lists with one signalled tail per QP, then drain all QPs.
// The WRITE source uses o.rows SGEs; READ uses a single GPU SGE and remote indirect rkey.
static void transfer(Resources& r, const Options& o, bool read, int n, uint64_t dst,
                     uint32_t dst_rkey, const std::vector<ibv_sge>& layout,
                     const std::vector<uint32_t>& rkeys, Stats& stats) {
    auto t = Clock::now();
    std::vector<ibv_send_wr> wr(n);
    std::vector<ibv_sge> gpu_sges(read ? n : 0);
    std::vector<int> head(o.qps, -1), tail(o.qps, -1);
    const uint32_t bytes = o.rows * ROW;
    for (int i = 0; i < n; ++i) {
        auto& w = wr[i]; int q = i % o.qps; w.wr_id = i;
        w.opcode = read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
        if (read) {
            gpu_sges[i].addr = dst + size_t(i) * bytes;
            gpu_sges[i].length = bytes; gpu_sges[i].lkey = r.mr->lkey;
            w.sg_list = &gpu_sges[i]; w.num_sge = 1;
            w.wr.rdma.remote_addr = 0; w.wr.rdma.rkey = rkeys[i];
        } else {
            w.sg_list = const_cast<ibv_sge*>(layout.data() + size_t(i) * o.rows);
            w.num_sge = o.rows;
            w.wr.rdma.remote_addr = dst + size_t(i) * bytes; w.wr.rdma.rkey = dst_rkey;
        }
        if (tail[q] >= 0) wr[tail[q]].next = &w;
        else head[q] = i;
        tail[q] = i;
    }
    int completions = 0;
    for (int q = 0; q < o.qps; ++q) if (head[q] >= 0) {
        wr[tail[q]].send_flags = IBV_SEND_SIGNALED;
        ibv_send_wr* bad = nullptr;
        verbs_check(ibv_post_send(r.qps[q], &wr[head[q]], &bad), "submit data WRs");
        ++completions;
    }
    stats.data_post_us = us(t); // Includes native data-WR construction.
    r.poll(completions); stats.data_us = us(t);
}
static Hello hello(const Options& o, const Resources& r) {
    Hello h{};
    h.pool = o.pool; h.qps = o.qps; h.rows = o.rows; h.groups = o.groups; h.table_rows = o.table_rows;
    if (!o.owner) { h.dst_addr = r.gpu + GUARD; h.dst_rkey = r.mr->rkey; }
    return h;
}
static void owner(Resources& r, const Options& o, Socket& sock, const Hello& peer) {
    std::vector<uint32_t> indices(size_t(o.pool) * o.rows), rkeys(o.pool);
    std::vector<ibv_sge> sges; sges.reserve(size_t(o.pool) * o.rows);
    while (true) {
        Command cmd{}; sock.recv(cmd); if (cmd.mode == 0) break;
        check((cmd.mode == 1 || cmd.mode == 2) && cmd.groups == uint32_t(o.groups), "bad command");
        for (int at = 0; at < o.groups; at += o.pool) {
            int n = std::min(o.pool, o.groups - at);
            Response response{}; response.groups = n;
            auto t = Clock::now();
            sock.recv(indices.data(), size_t(n) * o.rows * sizeof(uint32_t));
            response.stats.owner_receive_us = us(t);
            t = Clock::now(); layouts(r, o, indices.data(), n, sges);
            response.stats.layout_us = us(t);
            if (cmd.mode == 1) {
                configure(r, o, sges, n, response.stats);
                for (int i = 0; i < n; ++i) rkeys[i] = r.keys[i]->rkey;
                sock.send(response); sock.send(rkeys.data(), size_t(n) * sizeof(uint32_t));
            } else {
                transfer(r, o, false, n, peer.dst_addr + size_t(at) * o.rows * ROW,
                         peer.dst_rkey, sges, rkeys, response.stats);
                sock.send(response);
            }
            // READ requester sends ACK only after all READ CQEs. Never remap an in-flight key.
            uint32_t ack = 0; sock.recv(ack); check(ack == uint32_t(at + n), "bad completion ACK");
        }
    }
}
static void add(Stats& a, const Stats& b) {
    a.owner_receive_us += b.owner_receive_us; a.layout_us += b.layout_us;
    a.umr_us += b.umr_us; a.umr_post_us += b.umr_post_us;
    a.data_us += b.data_us; a.data_post_us += b.data_post_us;
}
static void validate(Resources& r, const Options& o, const std::vector<uint32_t>& indices) {
    size_t bytes = size_t(o.groups) * o.rows * ROW;
    std::vector<uint64_t> data(bytes / 8);
    cuda_check(cuMemcpyDtoH(data.data(), r.gpu + GUARD, bytes), "copy validation data");
    for (size_t i = 0; i < indices.size(); ++i)
        for (unsigned w = 0; w < ROW / 8; ++w)
            if (data[i * (ROW / 8) + w] != pattern(indices[i], w))
                throw std::runtime_error("payload mismatch at row=" + std::to_string(i) + " word=" + std::to_string(w));
    std::vector<uint8_t> guard(GUARD);
    for (CUdeviceptr ptr : {r.gpu, r.gpu + GUARD + bytes}) {
        cuda_check(cuMemcpyDtoH(guard.data(), ptr, GUARD), "copy guard");
        check(std::all_of(guard.begin(), guard.end(), [](uint8_t x) { return x == 0xa5; }), "GPU guard corrupted");
    }
}
static void requester(Resources& r, const Options& o, Socket& sock) {
    const size_t bytes = size_t(o.groups) * o.rows * ROW;
    std::vector<uint32_t> indices(size_t(o.groups) * o.rows), rkeys(o.pool);
    std::vector<ibv_sge> empty;
    for (int round = -o.warmups; round < o.rounds; ++round) {
        std::mt19937 rng(12345 + round + o.warmups);
        for (auto& i : indices) i = rng() % o.table_rows;
        // Alternate order to reduce drift; both methods receive exactly the same indices.
        for (int pos = 0; pos < 2; ++pos) {
            uint32_t mode = ((round + o.warmups + pos) % 2) + 1;
            cuda_check(cuMemsetD8(r.gpu, 0xa5, bytes + 2 * GUARD), "reset GPU allocation");
            cuda_check(cuMemsetD8(r.gpu + GUARD, 0, bytes), "clear GPU destination");
            cuda_check(cuCtxSynchronize(), "synchronize before timing");
            Stats sum{}; double send_us = 0, wait_us = 0, ack_us = 0;
            auto begin = Clock::now();
            Command command{mode, uint32_t(round + o.warmups), uint32_t(round < 0), uint32_t(o.groups)};
            auto t = Clock::now(); sock.send(command); send_us += us(t);
            for (int at = 0; at < o.groups; at += o.pool) {
                int n = std::min(o.pool, o.groups - at);
                t = Clock::now();
                sock.send(indices.data() + size_t(at) * o.rows, size_t(n) * o.rows * sizeof(uint32_t));
                send_us += us(t);
                Response response{}; t = Clock::now(); sock.recv(response);
                check(response.groups == uint32_t(n) && response.status == 0, "bad response");
                if (mode == 1) sock.recv(rkeys.data(), size_t(n) * sizeof(uint32_t));
                wait_us += us(t); add(sum, response.stats);
                if (mode == 1) {
                    Stats local{};
                    transfer(r, o, true, n, r.gpu + GUARD + size_t(at) * o.rows * ROW,
                             0, empty, rkeys, local);
                    sum.data_us += local.data_us; sum.data_post_us += local.data_post_us;
                }
                uint32_t ack = at + n; t = Clock::now(); sock.send(ack); ack_us += us(t);
            }
            // GPU validation/copy is outside timing. End-to-end ends at NIC completion/notification.
            double elapsed = us(begin);
            cuda_check(cuCtxSynchronize(), "synchronize after RDMA");
            validate(r, o, indices); // Full payload and both guards, every sample (including warmup).
            std::printf("{\"kind\":\"sample\",\"mode\":\"%s\",\"round\":%d,\"warmup\":%s,"
                "\"device\":\"%s\",\"gpu\":%d,\"groups\":%d,\"rows\":%d,\"bytes\":%zu,"
                "\"pool\":%d,\"qps\":%d,\"e2e_ms\":%.6f,\"gbps\":%.6f,"
                "\"control_send_ms\":%.6f,\"response_wait_ms\":%.6f,\"ack_send_ms\":%.6f,"
                "\"owner_receive_ms\":%.6f,\"layout_ms\":%.6f,\"umr_ms\":%.6f,\"umr_post_ms\":%.6f,"
                "\"data_ms\":%.6f,\"data_post_ms\":%.6f,\"data_gbps\":%.6f,\"data_ok\":true}\n",
                mode == 1 ? "read_umr" : "write_sge", round, round < 0 ? "true" : "false",
                o.device.c_str(), o.gpu, o.groups, o.rows, bytes, o.pool, o.qps,
                elapsed / 1000, bytes / elapsed / 1000,
                send_us / 1000, wait_us / 1000, ack_us / 1000,
                sum.owner_receive_us / 1000, sum.layout_us / 1000, sum.umr_us / 1000,
                sum.umr_post_us / 1000, sum.data_us / 1000, sum.data_post_us / 1000,
                bytes / sum.data_us / 1000);
            std::fflush(stdout);
        }
    }
    Command stop{}; sock.send(stop);
}
int main(int argc, char** argv) {
    try {
        setbuf(stdout, nullptr);
        Options o = options(argc, argv);
        uint16_t endian = 1; check(*reinterpret_cast<uint8_t*>(&endian) == 1, "little-endian required");
        Socket sock; Resources r;
        open_hca(r, o); create_qps(r, o); memory(r, o); connect(sock, o);
        Hello local = hello(o, r), peer{};
        sock.send(local); sock.recv(peer);
        check(peer.magic == MAGIC && peer.version == 1 && peer.bytes == sizeof(Hello) &&
              peer.pool == local.pool && peer.qps == local.qps && peer.rows == local.rows &&
              peer.groups == local.groups && peer.table_rows == local.table_rows, "hello mismatch");
        activate(r, o, sock);
        if (o.owner) owner(r, o, sock, peer);
        else requester(r, o, sock);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1;
    }
}
