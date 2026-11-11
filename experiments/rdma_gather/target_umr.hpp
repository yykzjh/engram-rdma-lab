#pragma once
// Every owner gets one logical root MKey. Children describe only the CURRENT
// mapping; regular runs are detected from actual positions, never from slot IDs.
// A bounded, shape-independent pool prevents unbounded metadata/resource growth.
class TargetMkey {
    struct Recipe {
        int key;
        bool strided = false;
        uint32_t repeats = 0;
        mlx5dv_mr_interleaved stride{};
        std::vector<ibv_sge> entries;
    };
    Channel& c;
    const Options& o;
    CUdeviceptr output;
    std::vector<Recipe> recipes;
    unsigned next_key = 1;
    uint64_t length = 0;
    int take_key() { return next_key < c.r.keys.size() ? int(next_key++) : -1; }
    static uint32_t byte_sum(const std::vector<ibv_sge>& list) {
        uint64_t bytes = 0; for (const auto& e : list) bytes += e.length;
        check(bytes <= UINT32_MAX, "indirect length"); return uint32_t(bytes);
    }
public:
    uint32_t base_row = 0;
    bool fits = false;
    double build_ms = 0, config_ms = 0;
    size_t used_keys = 0, input_runs = 0;
    TargetMkey(Channel& ch, const Options& opt, CUdeviceptr out) : c(ch), o(opt), output(out) {
        for (int i = 0; i < o.umr_keys; ++i) {
            mlx5dv_mkey_init_attr a{}; a.pd = c.r.pd;
            a.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT; a.max_entries = 52;
            auto* key = mlx5dv_create_mkey(&a); check(key, "create target MKey pool"); c.r.keys.push_back(key);
        }
    }
    uint32_t rkey() const { return c.r.keys[0]->rkey; }
    bool prepare(const Plan& plan, int owner, bool allow_stride = true) {
        auto started = Clock::now(); fits = false; recipes.clear(); next_key = 1; input_runs = 0;
        base_row = 0; for (int i = 0; i < owner; ++i) base_row += plan.peers[i].rows.size();
        size_t n = plan.peers[owner].rows.size(); length = n * ROW;
        struct Run { uint32_t start, count; size_t next; };
        auto run = [&](size_t at) {
            uint32_t start = plan.positions[base_row + at]; size_t end = at + 1;
            while (end < n && plan.positions[base_row + end] == start + end - at) ++end;
            return Run{start, uint32_t(end - at), end};
        };
        std::vector<ibv_sge> pieces; pieces.reserve(size_t(o.umr_keys) * 52);
        auto failed = [&] { used_keys = next_key; build_ms = us(started)/1000; return false; };
        for (size_t at = 0; at < n;) {
            auto a = run(at); size_t end = a.next; uint32_t repeats = 1; int64_t delta = 0;
            if (end < n) {
                auto b = run(end); delta = int64_t(b.start) - a.start;
                if (b.count == a.count && delta > int64_t(a.count)) {
                    uint32_t previous = a.start;
                    while (end < n) {
                        auto b2 = run(end);
                        if (b2.count != a.count || int64_t(b2.start) - previous != delta) break;
                        previous = b2.start; end = b2.next; ++repeats;
                    }
                }
            }
            if (repeats >= 4 && allow_stride) {
                int key = take_key(); if (key < 0) return failed();
                Recipe rec{}; rec.key = key; rec.strided = true; rec.repeats = repeats;
                rec.stride.addr = output + uint64_t(a.start) * ROW;
                rec.stride.bytes_count = a.count * ROW;
                rec.stride.bytes_skip = uint32_t(delta - a.count) * ROW;
                rec.stride.lkey = c.r.mr->lkey; recipes.push_back(std::move(rec));
                pieces.push_back({0, uint32_t(uint64_t(a.count) * repeats * ROW), c.r.keys[key]->lkey});
                input_runs += repeats; at = end;
            } else {
                pieces.push_back({output + uint64_t(a.start) * ROW, uint32_t(a.count * ROW), c.r.mr->lkey});
                ++input_runs; at = a.next;
            }
            if (pieces.size() > size_t(o.umr_keys) * 52) return failed();
        }
        // Fold arbitrary remaining pieces into <=52-entry list nodes. Root is
        // always key0; child recipes precede all recipes that reference them.
        int depth = 1;
        while (pieces.size() > 52) {
            if (++depth > 4) return failed();
            std::vector<ibv_sge> parents; parents.reserve((pieces.size()+51)/52);
            for (size_t at = 0; at < pieces.size(); at += 52) {
                int key = take_key(); if (key < 0) return failed();
                Recipe rec{}; rec.key = key;
                rec.entries.assign(pieces.begin()+at, pieces.begin()+std::min(at+52, pieces.size()));
                uint32_t bytes = byte_sum(rec.entries); recipes.push_back(std::move(rec));
                parents.push_back({0, bytes, c.r.keys[key]->lkey});
            }
            pieces.swap(parents);
        }
        // A strided child adds one extra indirection level.
        bool strided = false; for (const auto& rec : recipes) strided |= rec.strided;
        // A few accidental equal-stride runs in a large arbitrary mapping
        // should not force an unnecessary fifth level. Retry as plain KLMs.
        if (depth + int(strided) > 4) return prepare(plan, owner, false);
        if (n) { Recipe root{}; root.key = 0; root.entries = std::move(pieces); recipes.push_back(std::move(root)); }
        used_keys = next_key; fits = true; build_ms = us(started)/1000; return true;
    }
    void configure() {
        auto started = Clock::now();
        if (!length) { config_ms = 0; return; }
        auto* qp = ibv_qp_to_qp_ex(c.r.qps[0]); auto* dv = mlx5dv_qp_ex_from_ibv_qp_ex(qp);
        // A whole-fetch mapping can have many children. Bound each submission
        // by SQ capacity while configuring the COMPLETE tree before requests.
        for(size_t first=0;first<recipes.size();first+=o.pool){
        size_t end=std::min(first+size_t(o.pool),recipes.size());
        ibv_wr_start(qp);
        for (size_t i = first; i < end; ++i) {
            const auto& rec = recipes[i]; qp->wr_id = i;
            qp->wr_flags = IBV_SEND_INLINE | (i + 1 == end ? IBV_SEND_SIGNALED : 0);
            mlx5dv_mkey_conf_attr attr{};
            mlx5dv_wr_mkey_configure(dv, c.r.keys[rec.key], 2, &attr);
            mlx5dv_wr_set_mkey_access_flags(dv, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            if (rec.strided) mlx5dv_wr_set_mkey_layout_interleaved(dv, rec.repeats, 1, &rec.stride);
            else mlx5dv_wr_set_mkey_layout_list(dv, rec.entries.size(), rec.entries.data());
        }
        verbs_check(ibv_wr_complete(qp), "configure target UMR tree"); c.r.poll(1);
        }
        config_ms = us(started)/1000;
    }
};
