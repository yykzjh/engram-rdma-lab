"""READ 唯一测试入口：两机八 rank、动态索引、完整 fetch 与数据/scales 校验。

每个节点运行一次，RANK 为节点编号0/1，WORLD_SIZE=2；本程序各启动4个GPU进程。
输入按统一MT19937规则生成；支持全requester/owner以及单requester/owner两种流量。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from rank_startup import assigned_devices, bind_rank

ENTRIES = [524288, 524309]
HIDDEN, EPT = 256, 24
A = 0x9E3779B97F4A7C15 - (1 << 64)
B = 0xD1B54A32D192ED03 - (1 << 64)


def generate_indices(rank, tokens, epoch, distribution, owner=None):
    """与WRITE入口matched-read模式使用相同的种子和采样规则。

    使用 RandomState 的原始uint32输出，而非default_rng或uniform_int_distribution，
    避免不同播种/区间采样算法改变输入。每个epoch重新生成，计时起点仍是原始GPU索引。
    """
    rng = np.random.RandomState(12345 + rank * 1009 + epoch * 97)
    raw = rng.randint(0, 1 << 32, size=(2, tokens, EPT), dtype=np.uint32)
    for layer, entries in enumerate(ENTRIES):
        global_rows = entries * 8
        if owner is not None:
            raw[layer] %= entries
            raw[layer] += owner * entries
        elif distribution == "uniform":
            raw[layer] %= global_rows
        else:
            interval = global_rows // EPT
            raw[layer] %= interval
            raw[layer] += np.arange(EPT, dtype=np.uint32) * interval
    return raw


def pattern(tags):
    words = torch.arange(1, 33, dtype=torch.int64, device=tags.device) * B
    return ((tags + 1) * A)[:, None].bitwise_xor(words[None, :]).contiguous().view(torch.uint8)


def scales(global_indices, layer):
    x = global_indices.to(torch.int64)
    a = ((x + layer * 17) & 0x7FFFFF) | 0x3F800000
    b = ((x * 3 + layer * 29) & 0x7FFFFF) | 0x40000000
    return torch.stack([a, b], dim=-1).to(torch.int32).view(torch.float32)


def validate(outputs, indices):
    base = 0
    for layer, n in enumerate(ENTRIES):
        x = indices[layer].flatten().to(torch.int64)
        tags = (x // n) * sum(ENTRIES) + base + x % n
        expected = pattern(tags)
        actual, sf = outputs[layer]
        assert torch.equal(actual.view(torch.uint8).reshape(-1, HIDDEN), expected), (layer, "embedding bytes")
        expected_sf = scales(x, layer).reshape(indices.shape[1], EPT * 2)
        assert torch.equal(sf, expected_sf), (layer, "scale values")
        base += n


def parse_rank_pair(value):
    try:
        requester, owner = map(int, value.split(":"))
        if min(requester, owner) < 0 or requester == owner:
            raise ValueError
        return requester, owner
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected distinct nonnegative requester:owner ranks") from error


def inject_egm_failures(rank, pairs):
    """仅测试使用：破坏指定方向的映射元数据，不给生产实现增加故障开关。"""
    if not pairs:
        return
    import deep_ep
    native = deep_ep._C.EngramBuffer
    original = native.probe_routes

    def probe(runtime, peers, candidates):
        altered = [dict(peer) for peer in peers]
        for requester, owner in pairs:
            if requester != rank:
                continue
            if peers[owner]["node"] == peers[rank]["node"]:
                # READ本机映射由Context持有，破坏EGM范围校验而保留RDMA窗口。
                altered[owner]["table_bytes"] = 0
            else:
                # 保持FABRIC句柄ABI长度，触发真实CUDA import失败。
                altered[owner]["table_fabric"] = bytes(len(peers[owner]["table_fabric"]))
        return original(runtime, altered, candidates)

    native.probe_routes = probe


@torch.inference_mode()
def worker(local_rank, args):
    device, placement = bind_rank(local_rank, args.assigned_devices)
    # DeepEP initializes the CUDA driver at import time. Import only after the
    # fresh worker's device enumeration is fixed, before constructing any Buffer.
    import deep_ep
    from deep_ep.utils.envs import init_dist
    bases = [32, 48, 104, 120]
    os.sched_setaffinity(0, range(bases[local_rank], bases[local_rank] + args.affinity_cores))
    rank, world, group = init_dist(local_rank, 4, seed=12345, cuda_device=device)
    assert world == 8
    if not (0 <= args.requester_rank < world and 0 <= args.owner_rank < world):
        raise ValueError("requester/owner rank is outside the process group")
    active = args.traffic == "all-to-all" or rank == args.requester_rank
    owner = args.owner_rank if args.traffic == "single" else None
    forward_submissions = 0
    control = dist.new_group(backend="gloo")
    if any(max(pair) >= world for pair in args.egm_fail_pair):
        raise ValueError("Injected EGM pair is outside the process group")
    if args.egm_fail_pair and os.environ.get("ENGRAM_EGM_ENABLE", "0") != "1":
        raise ValueError("--egm-fail-pair requires ENGRAM_EGM_ENABLE=1")
    inject_egm_failures(rank, args.egm_fail_pair)
    root = Path(args.output_dir)
    root.mkdir(exist_ok=True, parents=True)
    raw = (
        generate_indices(rank, args.tokens, 1, args.distribution, owner)
        if active
        else np.empty((2, 0, EPT), dtype=np.uint32)
    )
    indices = torch.from_numpy(raw.view(np.int32).copy()).to(device="cuda")
    recommended = deep_ep.EngramBuffer.get_theoretical_config(group, 2, args.tokens, EPT, args.config_read_requests)
    qps = recommended[0] if args.qps == 0 else args.qps
    # Queue sizing is transport-specific. Never lower QPs below the framework's
    # minimum for a full-frame launch just to match an unrelated verbs parameter.
    assert qps >= recommended[0], (qps, recommended)
    gpu_bytes, cpu_bytes = deep_ep.EngramBuffer.get_storage_size_hint(
        group, ENTRIES, HIDDEN, args.tokens, EPT, torch.float8_e4m3fn, 2
    )
    restrict = bool(recommended[2]) and os.environ.get("NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC") == "8"
    qp_depth = args.qp_depth or recommended[1]
    buffer = deep_ep.EngramBuffer(
        group,
        gpu_bytes,
        cpu_bytes,
        True,
        qps,
        qp_depth,
        restrict,
        allow_hybrid_mode=True,
        num_gpu_timeout_secs=180,
        explicitly_destroy=True,
    )
    assert buffer.context.gin_min_stride == 4
    buffer.set_config(ENTRIES, HIDDEN, args.tokens, EPT, torch.float8_e4m3fn, 2)
    storages, sfs = [], []
    base = 0
    for layer, n in enumerate(ENTRIES):
        tags = torch.arange(n, dtype=torch.int64, device="cuda") + rank * sum(ENTRIES) + base
        storages.append(pattern(tags).view(torch.float8_e4m3fn))
        sfs.append(scales(torch.arange(world * n, dtype=torch.int64, device="cuda"), layer))
        base += n
    buffer.write(storages, sfs=sfs)
    torch.cuda.synchronize()
    del storages, sfs
    include = Path(deep_ep.__file__).parent / "include/deep_ep/impls/engram"
    routing = buffer.get_route_info()
    metadata = {
        "placement": placement,
        "egm_enabled": routing["egm_enabled"],
        "injected_egm_failures": args.egm_fail_pair,
        "routing": routing,
        "local_copy_lanes": 16,
        "kernel_sha256": {
            n: hashlib.sha256((include / n).read_bytes()).hexdigest()
            for n in ["engram_fetch.cuh", "engram_fetch_wait.cuh", "egm_config.cuh"]
        },
        "rank": rank,
        "indices_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "traffic": args.traffic,
        "requester_active": active,
        "requester_rank": args.requester_rank,
        "owner_rank": owner,
        "queried_owners": [owner] if owner is not None else list(range(world)),
        "initial_owner_counts": [
            np.bincount((raw[l] // n).reshape(-1), minlength=world).tolist() for l, n in enumerate(ENTRIES)
        ],
        "extension_sha256": hashlib.sha256(Path(deep_ep._C.__file__).read_bytes()).hexdigest(),
        "qps": qps,
        "recommended": recommended,
        "qp_depth": qp_depth,
        "cuda_graph": args.cuda_graph,
        "vary_indices": True,
        "profile_cuda": args.profile_cuda,
        "config_read_requests": args.config_read_requests,
        "restrict_rd_atomic": restrict,
        "gin_min_stride": buffer.context.gin_min_stride,
        "cpu_table": True,
        "rd_atomic": os.environ.get("NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC"),
        "dest_rd_atomic": os.environ.get("NCCL_GIN_GDAKI_MAX_DEST_RD_ATOMIC"),
        "torch": torch.__version__,
        "module": deep_ep.__file__,
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    graph = None
    if args.cuda_graph and active:
        # Compile and initialize any lazy framework state before capture.
        for _ in range(2):
            hooks = buffer.fetch(indices)
            outputs = [hook() for hook in hooks]
            torch.cuda.synchronize()
            forward_submissions += 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hooks = buffer.fetch(indices)
            outputs = [hook() for hook in hooks]
        graph.replay()
        forward_submissions += 1
        torch.cuda.synchronize()
        validate(outputs, indices)
    records = []
    cuda_start = torch.cuda.Event(enable_timing=True) if args.profile_cuda and active else None
    cuda_end = torch.cuda.Event(enable_timing=True) if args.profile_cuda and active else None
    for iteration in range(-args.warmups, args.rounds):
        epoch = iteration + args.warmups + 1
        # 上层输入生成/上传放在fetch计时之外，与WRITE测试保持相同边界。
        # 原始索引每轮变化；fetch内仍实际执行当前索引的地址计算与通信。
        input_hash = None
        if active:
            raw = generate_indices(rank, args.tokens, epoch, args.distribution, owner)
            if owner is not None:
                assert all(np.all(raw[l] // n == owner) for l, n in enumerate(ENTRIES))
            input_hash = hashlib.sha256(raw.tobytes()).hexdigest()
            indices.copy_(torch.from_numpy(raw.view(np.int32).copy()), non_blocking=False)
        dist.barrier(group=control)
        torch.cuda.synchronize()
        if not active:
            # 空闲rank不启动任何fetch/hook；在CPU控制组等待，保持owner注册内存存活。
            records.append(
                dict(
                    rank=rank,
                    round=iteration,
                    warmup=iteration < 0,
                    requester_active=False,
                    bytes=0,
                    scale_bytes=0,
                    e2e_ms=None,
                    gbps=None,
                    indices_sha256=None,
                    data_ok=None,
                    scale_ok=None,
                )
            )
            dist.barrier(group=control)
            continue
        start = time.perf_counter_ns()
        if args.profile_cuda:
            cuda_start.record()
        if graph is None:
            hooks = buffer.fetch(indices)
            issued = time.perf_counter_ns()
            outputs = [hook() for hook in hooks]
            hooked = time.perf_counter_ns()
        else:
            graph.replay()
            issued = hooked = time.perf_counter_ns()
        if args.profile_cuda:
            cuda_end.record()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        elapsed_ms = (end - start) / 1e6
        forward_submissions += 1
        validate(outputs, indices)  # Full embedding and scale validation outside the timer.
        v = dict(
            kind="sample",
            mode="deepep",
            rank=rank,
            round=iteration,
            warmup=iteration < 0,
            tokens=args.tokens,
            layers=2,
            ept=EPT,
            row_bytes=HIDDEN,
            bytes=2 * args.tokens * EPT * HIDDEN,
            scale_bytes=2 * args.tokens * EPT * 8,
            e2e_ms=elapsed_ms,
            gbps=(2 * args.tokens * EPT * HIDDEN) / elapsed_ms / 1e6,
            cuda_ms=cuda_start.elapsed_time(cuda_end) if args.profile_cuda else None,
            cuda_graph=bool(graph),
            input_epoch=epoch,
            indices_sha256=input_hash,
            requester_active=True,
            python_issue_ms=(issued - start) / 1e6,
            python_hooks_ms=(hooked - issued) / 1e6,
            synchronize_ms=(end - hooked) / 1e6,
            data_ok=True,
            scale_ok=True,
        )
        records.append(v)
        print("SAMPLE " + json.dumps(v), flush=True)
        if args.traffic != "all-to-all":
            # 在计时/校验之后会合，防止空闲rank提前进入GPU销毁collective干扰最后一轮。
            dist.barrier(group=control)
    metadata["forward_submissions"] = forward_submissions
    (root / f"deepep-r{rank}.json").write_text(json.dumps({"metadata": metadata, "samples": records}, indent=2))
    dist.barrier(group=control)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=131072)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--distribution", choices=["stratified", "uniform"], default="stratified")
    p.add_argument(
        "--egm-fail-pair",
        type=parse_rank_pair,
        action="append",
        default=[],
        help="Inject one-way EGM mapping failure for requester:owner (repeatable)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--qps", type=int, default=0)
    p.add_argument("--traffic", choices=["all-to-all", "single", "single-requester"], default="all-to-all")
    p.add_argument("--requester-rank", type=int, default=0)
    p.add_argument("--owner-rank", type=int, default=4)
    p.add_argument(
        "--affinity-cores",
        type=int,
        choices=[8, 16],
        default=8,
        help="CPU cores per rank; use the same value for both benchmark entries",
    )
    p.add_argument("--qp-depth", type=int, default=0, help="0 uses the framework recommendation")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument(
        "--profile-cuda", action="store_true", help="Adds timed profiling events; keep off for latency comparisons"
    )
    p.add_argument(
        "--config-read-requests", type=int, default=16, help="Read window assumption passed to framework recommendation"
    )
    args = p.parse_args()
    if os.environ.get("WORLD_SIZE") != "2" or os.environ.get("RANK") not in {"0", "1"}:
        p.error("WORLD_SIZE must be 2 (nodes), RANK must be 0 or 1")
    if not 1 <= args.tokens <= 262144 or not 1 <= args.rounds <= 30 or not 0 <= args.warmups <= 3:
        p.error("tokens: 1..262144; rounds: 1..30; warmups: 0..3")
    args.assigned_devices = assigned_devices(4)
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=4)
