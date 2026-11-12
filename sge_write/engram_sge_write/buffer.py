import math
import functools
import warnings
from pathlib import Path

import torch
import torch.distributed as dist

from . import _C
from .routing import (
    EgmConfig,
    env_flag,
    node_identity,
    node_layout,
    fabric_identity,
    owner_routes,
    agree_owner_routes,
    route_names,
    local_control_domain,
)


def _align(n, alignment=4096):
    return (n + alignment - 1) // alignment * alignment


def _legacy_storage_size(init):
    @functools.wraps(init)
    def wrapped(self, *args, **kwargs):
        if "num_rdma_storage_size" in kwargs:
            if "num_rdma_storage_bytes" in kwargs:
                raise TypeError("Specify only one storage-size keyword")
            kwargs["num_rdma_storage_bytes"] = kwargs.pop("num_rdma_storage_size")
        return init(self, *args, **kwargs)

    return wrapped


class EngramBuffer:
    """独立 WRITE 后端，保留 DeepEP 风格的配置、写表、fetch 和逐层 hook。

    fetch/hook 均只向当前 CUDA stream 提交工作。一个 Buffer 同时保留一轮输出；
    重新 fetch 前须调用上一轮所有 hook，消费方还须遵守共享输出缓冲区的生命周期。
    初始化、set_config、write、destroy 是所有 group ranks 参与的集体操作。
    """

    @_legacy_storage_size
    def __init__(
        self,
        group,
        num_gpu_bytes,
        num_rdma_storage_bytes,
        use_cpu_rdma_storage,
        num_allocated_qps,
        qp_depth,
        restrict_rd_atomic=False,
        allow_hybrid_mode=True,
        sl_idx=None,
        num_gpu_timeout_secs=100,
        explicitly_destroy=False,
        **legacy,
    ):
        if legacy:
            raise TypeError(f"Unknown arguments: {', '.join(legacy)}")
        if not use_cpu_rdma_storage:
            raise NotImplementedError("SGE WRITE currently serves CPU vocabulary shards")
        if not allow_hybrid_mode:
            raise NotImplementedError("This backend requires node-local shared CPU mappings")
        if not 1 <= num_allocated_qps <= 16 or qp_depth < 32:
            raise ValueError("WRITE requires 1..16 data QPs and qp_depth >= 32")
        self.egm = EgmConfig.from_env()
        self.local_sf = env_flag("ENGRAM_WRITE_LOCAL_SF", default=self.egm.enabled)
        self.group, self.rank_idx, self.num_ranks = group, group.rank(), group.size()
        self.device = torch.cuda.current_device()
        self.num_gpu_bytes, self.num_rdma_storage_bytes = num_gpu_bytes, num_rdma_storage_bytes
        self.num_allocated_qps, self.qp_depth = num_allocated_qps, qp_depth
        self.use_cpu_rdma_storage, self.allow_hybrid_mode = True, True
        self.explicitly_destroy = explicitly_destroy
        self.num_gpu_timeout_secs, self.sl_idx = num_gpu_timeout_secs, sl_idx
        self.runtime = None
        self.context = None
        self._config = None
        self._written = False
        self._generation = 0
        self._pending = set()
        self._input = None
        topology = [None] * self.num_ranks
        self.node_id = node_identity()
        dist.all_gather_object(topology, self.node_id, group=group)
        self.scaleup_rank_idx, local = node_layout(topology, self.rank_idx)
        self.num_scaleup_ranks = local
        self.num_scaleout_ranks = self.num_ranks // local
        self.scaleout_rank_idx = self.rank_idx // local
        self.num_nvlink_ranks, self.num_rdma_ranks = local, self.num_scaleout_ranks
        self.fabric_domain, self.fabric_error = fabric_identity(
            torch.cuda.get_device_properties(self.device).uuid, self.egm.enabled
        )
        # 保留签名以兼容调用方。WRITE 数据面没有 READ credit 限制，不能拿它作优化开关。
        self.restrict_rd_atomic = restrict_rd_atomic

    @staticmethod
    def get_storage_size_hint(
        group,
        num_entries_per_layer,
        hidden,
        num_max_tokens,
        num_entries_per_token,
        dtype=torch.bfloat16,
        num_sf_packs=0,
    ):
        layers = len(num_entries_per_layer)
        if (
            layers < 1
            or min(num_entries_per_layer) < 1
            or hidden < 1
            or num_max_tokens < 1
            or num_entries_per_token < 1
        ):
            raise ValueError("Storage dimensions must be positive")
        row_bytes = hidden * dtype.itemsize
        rows = layers * num_max_tokens * num_entries_per_token
        # 双份数据区（staging/output）和动态行号/位置图；另外预留路由、每层每QP DONE及scales。
        route = layers * group.size() * math.ceil(num_max_tokens * num_entries_per_token / 4096) * 8
        gpu = (
            2 * rows * row_bytes
            + rows * 8
            + route
            + layers * group.size() * 16 * 8
            + layers * _align(num_max_tokens, 4) * num_entries_per_token * num_sf_packs * 4
            + sum(num_entries_per_layer) * group.size() * num_sf_packs * 4
            + (1 << 20)
        )
        return _align(gpu), _align(sum(num_entries_per_layer) * row_bytes)

    @staticmethod
    def get_theoretical_config(
        group, num_layers, num_max_tokens, num_entries_per_token, num_max_read_requests_per_qp=16
    ):
        # 返回相同的三元组，但取值属于 WRITE 数据QP/SQ；不复用 READ 的QP估算公式。
        return 4, 1024, False

    @property
    def main_context(self):
        return self.runtime

    def set_config(
        self, num_entries_per_layer, hidden, num_max_tokens, num_entries_per_token, dtype=torch.bfloat16, num_sf_packs=0
    ):
        if self.runtime is not None:
            raise RuntimeError("set_config is initialization-only; create a new Buffer to reconfigure")
        if dtype not in {torch.bfloat16, torch.float8_e4m3fn}:
            raise ValueError("Supported vocabulary dtypes: bfloat16 and float8_e4m3fn")
        if (dtype == torch.float8_e4m3fn) != (num_sf_packs > 0):
            raise ValueError("FP8 requires scales; BF16 requires num_sf_packs=0")
        if hidden * dtype.itemsize % 16:
            raise ValueError("A vocabulary row must be a multiple of 16 bytes")
        needed = self.get_storage_size_hint(
            self.group, num_entries_per_layer, hidden, num_max_tokens, num_entries_per_token, dtype, num_sf_packs
        )
        if needed[0] > self.num_gpu_bytes or needed[1] > self.num_rdma_storage_bytes:
            raise ValueError("Insufficient capacity: use this backend's get_storage_size_hint")
        self._config = (list(num_entries_per_layer), hidden, num_max_tokens, num_entries_per_token, dtype, num_sf_packs)
        with torch.cuda.device(self.device):
            native = None
            try:
                native = _C.Runtime(
                    self.rank_idx,
                    self.num_ranks,
                    self.scaleup_rank_idx,
                    self.num_scaleup_ranks,
                    self.device,
                    list(num_entries_per_layer),
                    hidden,
                    dtype.itemsize,
                    num_max_tokens,
                    num_entries_per_token,
                    num_sf_packs,
                    self.num_allocated_qps,
                    self.qp_depth,
                    self.egm.enabled,
                    self.num_gpu_timeout_secs,
                    self.sl_idx if self.sl_idx is not None else 1,
                    str(Path(__file__).with_name("gpu_request.ptx")),
                )
                bootstrap = dict(
                    native.bootstrap(),
                    node=self.node_id,
                    fabric_domain=self.fabric_domain,
                    fabric_probe_error=self.fabric_error,
                )
            except Exception as error:
                bootstrap = {"error": str(error)}
            peers = [None] * self.num_ranks
            dist.all_gather_object(peers, bootstrap, group=self.group)
            failures = [p["error"] for p in peers if "error" in p]
            if failures:
                raise RuntimeError(f"WRITE initialization failed: {failures[0]}")
            if any(p["config"] != bootstrap["config"] for p in peers):
                raise ValueError("All ranks must use identical WRITE configurations")
            try:
                candidates = owner_routes(peers, self.rank_idx, self.egm)
                try:
                    report = dict(native.probe_routes(peers, candidates))
                except Exception as error:
                    report = {"error": str(error)}
                reports = self._exchange_route_reports(report, "probe")
                self.routes, self.owner_fallback_reasons = agree_owner_routes(reports, self.rank_idx)
                try:
                    report = dict(native.prepare_routes(peers, self.routes))
                except Exception as error:
                    report = {"error": str(error)}
                peers = self._exchange_route_reports(report, "commit")
            except Exception:
                native.close()
                raise
            with local_control_domain():
                native.connect(peers)
            self.runtime = self.context = native
            dist.barrier(group=self.group)

    def write(self, storages, sfs=None):
        self._ready()
        if self._pending:
            raise RuntimeError("Call all outstanding hooks before replacing vocabulary contents")
        entries, hidden, _, _, dtype, packs = self._config
        if len(storages) != len(entries) or (packs and (sfs is None or len(sfs) != len(entries))):
            raise ValueError("One storage/scales tensor is required per layer")
        # 与 DeepEP 一样，这是初始化/权重更新边界，不在每次 fetch 中发生。
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for layer, (storage, n) in enumerate(zip(storages, entries)):
            if storage.shape != (n, hidden) or storage.dtype != dtype or not storage.is_contiguous():
                raise ValueError(f"Invalid vocabulary tensor for layer {layer}")
            if storage.is_cuda and storage.device.index != self.device:
                raise ValueError("Vocabulary CUDA tensor is on the wrong device")
            sf = None if not packs else sfs[layer]
            if sf is not None and (
                sf.shape != (self.num_ranks * n, packs) or sf.dtype != torch.float32 or not sf.is_contiguous()
            ):
                raise ValueError(f"Invalid global scale tensor for layer {layer}")
            if sf is not None and sf.is_cuda and sf.device.index != self.device:
                raise ValueError("Scale CUDA tensor is on the wrong device")
            self.runtime.write(layer, storage, sf)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._written = True

    def fetch(self, indices, num_qps=0, use_tma_aligned_col_major_sf=True):
        """提交所有层的请求，立即返回逐层hook；调用hook后返回该层输出视图。

        hook也只提交GPU等待/重排，并不让主机等数据。调用方可在当前stream直接
        接续依赖embedding的计算；跨stream消费应通过stream/event建立依赖。
        capture前先使Buffer静止，并在Graph中连接hook stream后再复用工作区。
        """
        self._ready()
        if not self._written:
            raise RuntimeError("write() must initialize the vocabulary before fetch()")
        if self._pending:
            raise RuntimeError("Call every previous layer hook before the next fetch")
        entries, _, capacity, ept, _, _ = self._config
        if (
            indices.ndim != 3
            or indices.shape[0] != len(entries)
            or indices.shape[2] != ept
            or not 0 <= indices.shape[1] <= capacity
            or indices.dtype != torch.int32
            or not indices.is_cuda
            or indices.device.index != self.device
            or not indices.is_contiguous()
        ):
            raise ValueError("indices must be contiguous CUDA int32 [layers,tokens,entries_per_token] within capacity")
        qps = num_qps or self.num_allocated_qps
        if not 1 <= qps <= self.num_allocated_qps:
            raise ValueError("num_qps exceeds the allocated WRITE data QPs")
        self._generation += 1
        generation = self._generation
        outputs = self.runtime.fetch(indices, qps, use_tma_aligned_col_major_sf)
        self._pending = set(range(len(entries)))
        self._input = indices  # 保留异步kernel仍在使用的原始Tensor。
        # 指针被交给Driver API，显式告诉PyTorch allocator它在哪条stream上使用。
        indices.record_stream(torch.cuda.current_stream(self.device))

        def make_hook(layer):
            def wait_hook():
                self._ready()
                if generation != self._generation:
                    raise RuntimeError("This hook belongs to a superseded fetch")
                self.runtime.wait_layer(layer)
                self._pending.discard(layer)
                return outputs[layer]

            return wait_hook

        return [make_hook(layer) for layer in range(len(entries))]

    def barrier(self):
        # 与模型forward分离；显式barrier用于初始化/调试，不作为fetch推进条件。
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    def get_comm_stream(self):
        return torch.cuda.current_stream(self.device)

    def get_physical_domain_size(self):
        return self.num_rdma_ranks, self.num_nvlink_ranks

    def get_logical_domain_size(self):
        return self.num_scaleout_ranks, self.num_scaleup_ranks

    def _exchange_route_reports(self, report, phase):
        reports = [None] * self.num_ranks
        dist.all_gather_object(reports, report, group=self.group)
        failures = [f"rank {rank}: {item['error']}" for rank, item in enumerate(reports) if "error" in item]
        if failures:
            raise RuntimeError(f"WRITE route {phase} failed: {failures[0]}")
        return reports

    def get_route_info(self):
        self._ready()
        return dict(
            egm_enabled=self.egm.enabled,
            owner_routes=route_names(self.routes),
            owner_fallback_reasons=list(self.owner_fallback_reasons),
            fabric_domain=self.fabric_domain,
            fabric_probe_error=self.fabric_error,
            memory=self.runtime.memory_usage(),
        )

    def print_memory_usage(self):
        self._ready()
        print(self.runtime.memory_usage(), flush=True)

    def _ready(self):
        if self.runtime is None:
            raise RuntimeError("Buffer is unconfigured or destroyed")

    def destroy(self):
        if not self.explicitly_destroy:
            raise RuntimeError("Set explicitly_destroy=True before calling destroy()")
        if self.runtime is not None:
            if self._pending:
                raise RuntimeError("Outstanding layer hooks must be called before destroy()")
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            self.runtime.close()
            dist.barrier(group=self.group)
            self.runtime = self.context = None
            self._input = None

    def __del__(self):
        if getattr(self, "runtime", None) is not None and self.explicitly_destroy:
            warnings.warn("EngramBuffer was not collectively destroyed before process shutdown", ResourceWarning)
