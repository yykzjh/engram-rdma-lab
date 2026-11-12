"""WRITE 唯一测试入口：独立EngramBuffer API、逐层hook、Graph和异步forward。"""

import argparse
import contextlib
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from rank_startup import assigned_devices, bind_rank

A, B = 0x9E3779B97F4A7C15 - (1 << 64), 0xD1B54A32D192ED03 - (1 << 64)


def pattern(tags, row_bytes):
    words = torch.arange(1, row_bytes // 8 + 1, dtype=torch.int64, device=tags.device) * B
    return ((tags + 1)[:, None] * A).bitwise_xor(words[None, :]).contiguous().view(torch.uint8)


def scale_values(indices, layer, packs, matched_read=False):
    if matched_read:
        x = indices.to(torch.int64)
        a = ((x + layer * 17) & 0x7FFFFF) | 0x3F800000
        b = ((x * 3 + layer * 29) & 0x7FFFFF) | 0x40000000
        return torch.stack([a, b], dim=-1).to(torch.int32).view(torch.float32)
    return torch.stack(
        [
            (((indices.to(torch.int64) * (p + 1) + layer * 17) & 0x7FFFFF) | 0x3F800000)
            .to(torch.int32)
            .view(torch.float32)
            for p in range(packs)
        ],
        dim=-1,
    )


def expected(indices, entries, hidden, dtype, packs, model_offset=0, matched_read=False):
    result, base = [], 0
    for layer, n in enumerate(entries):
        x = indices[layer].flatten().to(torch.int64)
        valid = x >= 0
        safe = x.clamp_min(0)
        tags = safe // n * sum(entries) + base + safe % n
        data = pattern(tags + model_offset, hidden * dtype.itemsize)
        data.mul_(valid[:, None])
        sf = None
        if packs:
            sf = scale_values(safe, layer + model_offset, packs, matched_read)
            sf.mul_(valid[:, None])
            sf = sf.reshape(indices.shape[1], indices.shape[2] * packs)
        result.append((data.reshape(indices.shape[1], indices.shape[2] * hidden * dtype.itemsize), sf))
        base += n
    return result


def generate(rank, tokens, entries, ept, epoch, distribution, padding):
    generator = torch.Generator(device="cuda").manual_seed(12345 + rank * 1009 + epoch * 97)
    layers = []
    for n in entries:
        if distribution == "uniform":
            x = torch.randint(
                0, dist.get_world_size() * n, (tokens, ept), dtype=torch.int32, device="cuda", generator=generator
            )
        else:
            interval = dist.get_world_size() * n // ept
            x = torch.randint(0, interval, (tokens, ept), dtype=torch.int32, device="cuda", generator=generator)
            x += torch.arange(ept, dtype=torch.int32, device="cuda") * interval
        if padding and tokens:
            x.view(-1)[::7] = -1
        layers.append(x)
    return torch.stack(layers)


def matched_indices(rank, tokens, entries, ept, epoch, distribution, world, owner=None):
    """性能对照专用：逐字节匹配READ入口，输入生成和上传均在计时之外。

    不共享READ的实现代码，避免两个独立测试入口互相导入。
    """
    rng = np.random.RandomState(12345 + rank * 1009 + epoch * 97)
    raw = rng.randint(0, 1 << 32, size=(len(entries), tokens, ept), dtype=np.uint32)
    for layer, n in enumerate(entries):
        if owner is not None:
            raw[layer] %= n
            raw[layer] += owner * n
        elif distribution == "uniform":
            raw[layer] %= world * n
        else:
            interval = world * n // ept
            raw[layer] %= interval
            raw[layer] += np.arange(ept, dtype=np.uint32) * interval
    return raw


def error_count(outputs, reference, packs):
    count = torch.zeros((), dtype=torch.int64, device="cuda")
    for output, (data, sf) in zip(outputs, reference):
        actual, actual_sf = output if packs else (output, None)
        count += (actual.view(torch.uint8).reshape(data.shape) != data).sum()
        if packs:
            count += (actual_sf != sf).sum()
    return count


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
    import engram_sge_write
    native = engram_sge_write._C.Runtime
    original = native.probe_routes

    def probe(runtime, peers, candidates):
        altered = [dict(peer) for peer in peers]
        for requester, owner in pairs:
            if requester != rank:
                continue
            if peers[owner]["node"] == peers[rank]["node"]:
                # 只破坏词表fd，邮箱/GIN必需的共享映射仍保持正常。
                altered[owner]["table_fd"] = -1
            else:
                # 保持FABRIC句柄ABI长度，触发真实CUDA import失败。
                altered[owner]["table_fabric"] = bytes(len(peers[owner]["table_fabric"]))
        return original(runtime, altered, candidates)

    native.probe_routes = probe


@torch.inference_mode()
def worker(local_rank, args):
    _, placement = bind_rank(local_rank, args.assigned_devices, write=True)
    import engram_sge_write
    if args.matched_read:
        bases = [32, 48, 104, 120]
        os.sched_setaffinity(0, range(bases[local_rank], bases[local_rank] + args.affinity_cores))
    node_rank, nodes = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.getenv('MASTER_PORT', '23890')}",
        rank=node_rank * args.local_ranks + local_rank,
        world_size=nodes * args.local_ranks,
    )
    group = dist.group.WORLD
    rank, world = group.rank(), group.size()
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
    entries = [int(n) for n in args.entries.split(",")]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float8_e4m3fn
    packs = 0 if args.dtype == "bf16" else args.sf_packs
    capacity = args.capacity or max(1, args.tokens)
    sizes = engram_sge_write.EngramBuffer.get_storage_size_hint(
        group, entries, args.hidden, capacity, args.ept, dtype, packs
    )
    qps, depth, restrict = engram_sge_write.EngramBuffer.get_theoretical_config(group, len(entries), capacity, args.ept)
    buffer = engram_sge_write.EngramBuffer(
        group, *sizes, True, qps, depth, restrict, explicitly_destroy=True, num_gpu_timeout_secs=120
    )
    buffer.set_config(entries, args.hidden, capacity, args.ept, dtype, packs)

    def load_weights(model_offset):
        storages, scales, base = [], [], 0
        for layer, n in enumerate(entries):
            tags = torch.arange(n, dtype=torch.int64, device="cuda") + rank * sum(entries) + base + model_offset
            storage = pattern(tags, args.hidden * dtype.itemsize).view(dtype).reshape(n, args.hidden)
            storages.append(storage.cpu() if args.cpu_write else storage)
            if packs:
                sf = scale_values(
                    torch.arange(world * n, device="cuda"), layer + model_offset, packs, args.matched_read
                )
                scales.append(sf.cpu() if args.cpu_write else sf)
            base += n
        buffer.write(storages, scales if packs else None)

    load_weights(0)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    order = list(range(len(entries)))
    if args.hook_order == "reverse":
        order.reverse()
    consume_stream = torch.cuda.Stream() if args.cross_stream else None
    locality_epoch = torch.zeros((), dtype=torch.int32, device="cuda")

    def advance_locality(indices):
        # 同一个Buffer/Graph轮流遇到纯机内、空层、全padding、跨机和混合请求。
        # phase留在GPU上，避免CPU提前知道路由；专门检查空请求跳过后的邮箱/DONE序号。
        locality_epoch.add_(1)
        phase = locality_epoch.remainder(6)
        local_owner = rank // args.local_ranks * args.local_ranks + (rank + 1) % args.local_ranks
        remote_owner = ((rank // args.local_ranks + 1) % nodes) * args.local_ranks + (rank + 1) % args.local_ranks
        mixed = torch.arange(args.ept, device="cuda", dtype=torch.int32).remainder(world)
        for layer, n in enumerate(entries):
            target = torch.where(
                phase == 4,
                mixed,
                torch.where(
                    (phase == 3) | ((phase == 1) & (layer > 0)) | ((phase == 5) & (layer == 0)),
                    remote_owner,
                    local_owner,
                ),
            )
            indices[layer].add_(1).remainder_(n).add_(target * n)
            indices[layer].masked_fill_((phase == 2) | ((phase == 1) & (layer == 0)), -1)

    def fetch_and_wait(indices):
        if args.locality_cycle:
            advance_locality(indices)
        hooks = buffer.fetch(indices, num_qps=args.qps, use_tma_aligned_col_major_sf=not args.row_major_sf)
        # 模拟fetch与hook之间不依赖embedding的计算，等待并未被提前放入fetch。
        if not args.matched_read:
            scratch.add_(1)
        outputs = [None] * len(entries)
        with torch.cuda.stream(consume_stream) if consume_stream is not None else contextlib.nullcontext():
            for layer in order:
                outputs[layer] = hooks[layer]()
        if args.cross_stream:
            torch.cuda.current_stream().wait_stream(consume_stream)
        return outputs

    def check_layout(outputs, tokens):
        for item in outputs:
            data, sf = item if packs else (item, None)
            assert data.shape == (tokens, args.hidden * args.ept) and data.dtype == dtype
            if packs:
                assert sf.shape == (tokens, args.ept * packs) and sf.dtype == torch.float32
                expected_stride = (args.ept * packs, 1) if args.row_major_sf else (1, (tokens + 3) // 4 * 4)
                assert sf.stride() == expected_stride

    scratch = torch.zeros(64, device="cuda")
    if args.matched_read:
        raw = (
            matched_indices(rank, args.tokens, entries, args.ept, 1, args.distribution, world, owner)
            if active
            else np.empty((len(entries), 0, args.ept), dtype=np.uint32)
        )
        indices = torch.from_numpy(raw.view(np.int32).copy()).to(device="cuda")
    else:
        indices = generate(rank, args.tokens, entries, args.ept, 1, args.distribution, args.padding)
    initial_owner_counts = (
        [np.bincount((raw[l] // n).reshape(-1), minlength=world).tolist() for l, n in enumerate(entries)]
        if args.matched_read
        else None
    )
    for _ in range(2 if active else 0):
        outputs = fetch_and_wait(indices)
        forward_submissions += 1
    torch.cuda.synchronize()
    if active:
        check_layout(outputs, args.tokens)
        assert (
            error_count(
                outputs, expected(indices, entries, args.hidden, dtype, packs, matched_read=args.matched_read), packs
            ).item()
            == 0
        )
    if args.check_errors:
        hooks = buffer.fetch(indices)
        try:
            buffer.fetch(indices)
        except RuntimeError:
            pass
        else:
            raise AssertionError("A second fetch overwrote pending hooks")
        for hook in hooks:
            hook()
        newer = buffer.fetch(indices)
        try:
            hooks[0]()
        except RuntimeError:
            pass
        else:
            raise AssertionError("A stale hook was accepted")
        for hook in newer:
            hook()
        forward_submissions += 2
    model_offset = 17 if args.rewrite else 0
    if args.rewrite:
        load_weights(model_offset)
    graph = None
    verdict = torch.zeros(args.rounds, dtype=torch.int64, device="cuda")
    if active and (args.cuda_graph or (args.async_check and not args.async_eager)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # producer也在Graph里，每次重放都改变GPU原始索引。
            if not args.matched_read:
                for layer, n in enumerate(entries):
                    indices[layer].add_(1).remainder_(world * n)
            outputs = fetch_and_wait(indices)
        graph.replay()
        torch.cuda.synchronize()
        forward_submissions += 1
    records = []
    if args.profile:
        if not 0 <= args.profile_rank < world:
            raise ValueError("--profile-rank must identify a rank in this process group")
        selected = rank == args.profile_rank
        scope = torch.profiler.record_function if selected else lambda name: contextlib.nullcontext()
        profiler = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
                profile_memory=False,
            )
            if selected
            else contextlib.nullcontext()
        )
        dist.barrier(group=control)
        torch.cuda.synchronize()
        # 初始化、写表、预热均已完成。只捕获连续forward及最后一次同步，
        # 把完整数据校验放在profiler外，避免校验kernel遮住通信时间线。
        with profiler:
            for frame in range(args.rounds):
                with scope(f"SGE_WRITE/forward_{frame}"):
                    if graph is not None:
                        with scope("SGE_WRITE/graph_replay"):
                            graph.replay()
                    else:
                        with scope("SGE_WRITE/issue_requests"):
                            hooks = buffer.fetch(
                                indices, num_qps=args.qps, use_tma_aligned_col_major_sf=not args.row_major_sf
                            )
                        with scope("SGE_WRITE/independent_compute"):
                            scratch.add_(1)
                        outputs = [None] * len(entries)
                        with (
                            torch.cuda.stream(consume_stream)
                            if consume_stream is not None
                            else contextlib.nullcontext()
                        ):
                            for layer in order:
                                with scope(f"SGE_WRITE/layer_{layer}/wait_DONE_and_scatter"):
                                    outputs[layer] = hooks[layer]()
                        if consume_stream is not None:
                            torch.cuda.current_stream().wait_stream(consume_stream)
                    forward_submissions += 1
            with scope("SGE_WRITE/final_synchronize"):
                torch.cuda.synchronize()
        trace = root / f"write-sge-r{rank}.trace.json"
        if selected:
            profiler.export_chrome_trace(str(trace))
        check_layout(outputs, indices.shape[1])
        assert (
            error_count(
                outputs, expected(indices, entries, args.hidden, dtype, packs, model_offset, args.matched_read), packs
            ).item()
            == 0
        )
        records.append(
            dict(
                rank=rank,
                profile=True,
                profiled_forwards=args.rounds,
                validation="final_forward_after_profile",
                data_ok=True,
                scale_ok=True,
            )
        )
        metadata = dict(
            async_check=False,
            profile=True,
            profile_rank=args.profile_rank,
            cuda_graph=graph is not None,
            trace=str(trace) if selected else None,
        )
        # 其余rank保持owner服务存活，直到采集rank完成CUPTI flush和trace导出。
        dist.barrier(group=control)
    elif args.async_check:
        dist.barrier(group=control)
        for frame in range(args.rounds):
            start = time.perf_counter_ns()
            if graph is None:
                for layer, n in enumerate(entries):
                    indices[layer].add_(1).remainder_(world * n)
                outputs = fetch_and_wait(indices)
            else:
                graph.replay()
            enqueue_us = (time.perf_counter_ns() - start) / 1000
            forward_submissions += 1
            # GPU consumer逐帧校验，循环内不.item()/D2H/主机同步；最后一次同步后读取verdict。
            verdict[frame] = error_count(
                outputs, expected(indices, entries, args.hidden, dtype, packs, model_offset, args.matched_read), packs
            )
            records.append(dict(rank=rank, round=frame, enqueue_us=enqueue_us))
        done = torch.cuda.Event()
        done.record()
        pending = not done.query()
        torch.cuda.synchronize()
        assert verdict.sum().item() == 0
        for record in records:
            record.update(data_ok=True, scale_ok=True, async_check=True)
        metadata = dict(
            async_check=True,
            cuda_graph=graph is not None,
            forwards=args.rounds,
            pending_after_enqueue=pending,
            per_forward_host_syncs=0,
            requester_indices_d2h=0,
        )
    else:
        for iteration in range(-args.warmups, args.rounds):
            input_hash = None
            if args.matched_read and active:
                epoch = iteration + args.warmups + 1
                raw = matched_indices(rank, args.tokens, entries, args.ept, epoch, args.distribution, world, owner)
                if owner is not None:
                    assert all(np.all(raw[l] // n == owner) for l, n in enumerate(entries))
                input_hash = hashlib.sha256(raw.tobytes()).hexdigest()
                indices.copy_(torch.from_numpy(raw.view(np.int32).copy()), non_blocking=False)
            elif graph is None and active:
                count = [args.tokens, args.tokens // 2, 0][iteration % 3] if args.vary_tokens else args.tokens
                indices = generate(
                    rank, count, entries, args.ept, iteration + args.warmups + 1, args.distribution, args.padding
                )
            dist.barrier(group=control)
            torch.cuda.synchronize()
            if not active:
                # 不用padding伪造单requester：其余rank完全跳过GPU请求和hook。
                records.append(
                    dict(
                        rank=rank,
                        round=iteration,
                        warmup=iteration < 0,
                        requester_active=False,
                        bytes=0,
                        scale_bytes=0,
                        e2e_ms=None,
                        indices_sha256=None,
                        data_ok=None,
                        scale_ok=None,
                    )
                )
                dist.barrier(group=control)
                continue
            start = time.perf_counter_ns()
            if graph is None:
                outputs = fetch_and_wait(indices)
            else:
                graph.replay()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            forward_submissions += 1
            check_layout(outputs, indices.shape[1])
            assert (
                error_count(
                    outputs,
                    expected(indices, entries, args.hidden, dtype, packs, model_offset, args.matched_read),
                    packs,
                ).item()
                == 0
            )
            records.append(
                dict(
                    rank=rank,
                    round=iteration,
                    warmup=iteration < 0,
                    e2e_ms=elapsed,
                    indices_sha256=input_hash,
                    requester_active=True,
                    bytes=len(entries) * indices.shape[1] * args.ept * args.hidden * dtype.itemsize,
                    data_ok=True,
                    scale_ok=True,
                )
            )
            if args.traffic != "all-to-all":
                dist.barrier(group=control)
        metadata = dict(async_check=False)
    metadata.update(
        placement=placement,
        egm_enabled=buffer.egm.enabled,
        injected_egm_failures=args.egm_fail_pair,
        routing=buffer.get_route_info(),
        dtype=args.dtype,
        entries=entries,
        tokens=args.tokens,
        capacity=capacity,
        hook_order=args.hook_order,
        cross_stream=args.cross_stream,
        col_major_sf=not args.row_major_sf,
        gin_min_stride=buffer.context.gin_min_stride,
        matched_read=args.matched_read,
        affinity=sorted(os.sched_getaffinity(0)),
        traffic=args.traffic,
        requester_active=active,
        requester_rank=args.requester_rank,
        owner_rank=owner,
        queried_owners=[owner] if owner is not None else list(range(world)),
        forward_submissions=forward_submissions,
        initial_owner_counts=initial_owner_counts,
        memory=buffer.runtime.memory_usage(),
    )
    (root / f"write-r{rank}.json").write_text(json.dumps(dict(metadata=metadata, samples=records), indent=2))
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--capacity", type=int, default=0)
    p.add_argument("--entries", default="524288,524309")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--ept", type=int, default=24)
    p.add_argument("--dtype", choices=["fp8", "bf16"], default="fp8")
    p.add_argument("--sf-packs", type=int, default=2)
    p.add_argument("--qps", type=int, default=0)
    p.add_argument("--traffic", choices=["all-to-all", "single", "single-requester"], default="all-to-all")
    p.add_argument("--requester-rank", type=int, default=0)
    p.add_argument("--owner-rank", type=int, default=4)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--local-ranks", type=int, default=4)
    p.add_argument("--distribution", choices=["uniform", "stratified"], default="uniform")
    p.add_argument("--hook-order", choices=["forward", "reverse"], default="forward")
    p.add_argument("--row-major-sf", action="store_true")
    p.add_argument("--cross-stream", action="store_true")
    p.add_argument("--padding", action="store_true")
    p.add_argument("--vary-tokens", action="store_true")
    p.add_argument(
        "--locality-cycle",
        action="store_true",
        help="GPU changes local/remote/empty-layer traffic each forward, including Graph replay",
    )
    p.add_argument("--cpu-write", action="store_true")
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--check-errors", action="store_true")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--async-check", action="store_true")
    p.add_argument(
        "--async-eager", action="store_true", help="With --async-check, submit eager fetch/hooks instead of a Graph"
    )
    p.add_argument("--profile", action="store_true", help="Export a CPU/CUDA torch.profiler Chrome trace after warmup")
    p.add_argument("--profile-rank", type=int, default=0, help="Global rank to profile; other ranks still participate")
    p.add_argument(
        "--matched-read", action="store_true", help="Match READ input, weights, CPU affinity, and fetch-only timing"
    )
    p.add_argument(
        "--affinity-cores",
        type=int,
        choices=[8, 16],
        default=8,
        help="CPU cores per rank in --matched-read mode; match the READ setting",
    )
    p.add_argument(
        "--egm-fail-pair",
        type=parse_rank_pair,
        action="append",
        default=[],
        help="Inject one-way EGM mapping failure for requester:owner (repeatable)",
    )
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    if a.tokens < 0 or a.capacity and a.capacity < max(1, a.tokens) or a.rounds < 1:
        p.error("Invalid token capacity or rounds")
    if a.vary_tokens and (a.cuda_graph or a.async_check):
        p.error("--vary-tokens is an ordinary API test; Graph replay has fixed shape")
    if a.async_eager and (not a.async_check or a.cuda_graph):
        p.error("--async-eager requires --async-check and excludes --cuda-graph")
    if a.profile and (a.async_check or a.vary_tokens):
        p.error("--profile uses a fixed-shape forward capture; exclude --async-check and --vary-tokens")
    if a.traffic != "all-to-all" and (not a.matched_read or a.profile or a.async_check or a.check_errors or a.rewrite):
        p.error("Single-requester traffic requires --matched-read and the ordinary/Graph benchmark mode")
    if a.matched_read and (
        a.entries != "524288,524309"
        or a.hidden != 256
        or a.ept != 24
        or a.dtype != "fp8"
        or a.sf_packs != 2
        or a.local_ranks != 4
        or a.async_check
        or a.padding
        or a.vary_tokens
        or a.rewrite
        or a.locality_cycle
    ):
        p.error("--matched-read requires the default FP8 shape and ordinary or Graph benchmark mode")
    a.assigned_devices = assigned_devices(a.local_ranks)
    torch.multiprocessing.spawn(worker, args=(a,), nprocs=a.local_ranks)
