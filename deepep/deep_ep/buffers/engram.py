import functools
import os
import weakref
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from .base import BufferBase
from .. import comm
from ..utils.envs import EnvOverride, check_nvlink_connections, check_fast_rdma_atomic_support
from ..utils.math import align
from ..utils.engram_routing import (
    EgmConfig,
    node_identity,
    node_layout,
    fabric_identity,
    owner_routes,
    agree_owner_routes,
    route_names,
    local_control_domain,
)


# TODO: remove this
def _accept_legacy_storage_size(init):
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        # TODO: Remove the legacy keyword alias once callers migrate to num_rdma_storage_bytes.
        if "num_rdma_storage_size" in kwargs:
            if "num_rdma_storage_bytes" in kwargs:
                raise TypeError("Specify only one of num_rdma_storage_bytes and num_rdma_storage_size")
            kwargs["num_rdma_storage_bytes"] = kwargs.pop("num_rdma_storage_size")
        return init(self, *args, **kwargs)

    return wrapper


class EngramBuffer(BufferBase):
    """GPU fetch buffer with separate CPU or GPU storage shards registered for RDMA."""

    # 生命周期由comm的managed缓存管理；弱引用不妨碍destroy_all_managed_nccl_comm()释放资源。
    _control_handles = weakref.WeakValueDictionary()

    @staticmethod
    def _control_handle(group):
        # 不复用可能已启用NCCL MNNVL的用户/PyTorch communicator。
        if group not in EngramBuffer._control_handles:
            with local_control_domain():
                EngramBuffer._control_handles[group] = comm.get_nccl_comm_handle(group, force_new_comm=True)
        return EngramBuffer._control_handles[group].get()

    # Common communication functions; refer to `deep_ep.comm` for usage.
    barrier = comm.barrier
    get_comm_stream = comm.get_comm_stream
    get_physical_domain_size = comm.get_physical_domain_size
    get_logical_domain_size = comm.get_logical_domain_size

    @_accept_legacy_storage_size
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_gpu_bytes: int,
        num_rdma_storage_bytes: int,
        use_cpu_rdma_storage: bool,
        num_allocated_qps: int,
        qp_depth: int,
        restrict_rd_atomic: bool = False,
        allow_hybrid_mode: bool = True,
        sl_idx: Optional[int] = None,
        num_gpu_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
    ):
        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.num_gpu_bytes = num_gpu_bytes
        self.num_rdma_storage_bytes = num_rdma_storage_bytes
        self.use_cpu_rdma_storage = use_cpu_rdma_storage
        self.allow_hybrid_mode = allow_hybrid_mode

        self.egm = EgmConfig.from_env()
        if not use_cpu_rdma_storage:
            self.egm = EgmConfig()
        if self.egm.enabled and not allow_hybrid_mode:
            raise ValueError("CPU EGM routing requires allow_hybrid_mode=True")
        self.node_id = node_identity()
        nodes = [None] * self.num_ranks
        dist.all_gather_object(nodes, self.node_id, group)
        _, node_size = node_layout(nodes, self.rank_idx)
        self.fabric_domain, self.fabric_error = fabric_identity(
            torch.cuda.get_device_properties(torch.cuda.current_device()).uuid, self.egm.enabled
        )
        options = (
            self.egm.enabled,
            num_gpu_bytes,
            num_rdma_storage_bytes,
            use_cpu_rdma_storage,
            allow_hybrid_mode,
            num_allocated_qps,
            qp_depth,
        )
        configurations = [None] * self.num_ranks
        dist.all_gather_object(configurations, options, group)
        if any(config != options for config in configurations):
            raise ValueError("All ranks must use identical READ configurations")

        num_max_local_ranks = int(os.getenv("EP_NUM_MAX_LOCAL_RANKS", 16)) if allow_hybrid_mode else 1
        num_registered_bytes = num_gpu_bytes + num_rdma_storage_bytes * num_max_local_ranks + (1 << 32)
        num_total_gpu_bytes = torch.cuda.get_device_properties("cuda").total_memory
        if num_registered_bytes > num_total_gpu_bytes:
            os.environ["NCCL_WIN_STRIDE"] = str(align(num_registered_bytes, 1 << 32))

        check_nvlink_connections(group)

        assert (
            num_allocated_qps > 0 and qp_depth > 0
        ), "pass `num_allocated_qps` and `qp_depth` from `get_theoretical_config`"
        self.num_allocated_qps = num_allocated_qps

        shared_comm = []
        self._cpu_storage = None
        if allow_hybrid_mode:
            if use_cpu_rdma_storage:
                # 原始handle在Buffer生命周期内存活；Context接收独立dup的POSIX fd。
                # FABRIC在原始handle上导出，跨节点importer不触碰pidfd或本机NUMA权限。
                self._cpu_storage = _C.EngramBuffer.create_cpu_storage(num_rdma_storage_bytes, self.egm.enabled)
                pid, fd = self._cpu_storage.shared_handle()
            else:
                pid, fd = _C.EngramBuffer.create_shared_handle(num_rdma_storage_bytes, use_cpu_rdma_storage)
            shared_comm = [None] * self.num_ranks
            dist.all_gather_object(shared_comm, (pid, fd), group)

        super().__init__(explicitly_destroy)
        # register only GDAKI backend to accelerate init
        with local_control_domain(), EnvOverride(
            [
                ("NCCL_GIN_TYPE", "3", True),
                # TODO: Configure RD atomics through NCCL devComm config when supported.
                ("NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC", "8", restrict_rd_atomic),
            ]
        ):
            self.nccl_comm_handle = comm.get_nccl_comm_handle(group, force_new_comm=True)
            self.runtime = _C.EngramBuffer(
                self.rank_idx,
                self.num_ranks,
                self.nccl_comm_handle.get(),
                shared_comm,
                num_gpu_bytes,
                num_rdma_storage_bytes,
                use_cpu_rdma_storage,
                allow_hybrid_mode,
                sl_idx,
                num_allocated_qps,
                qp_depth,
                num_gpu_timeout_secs,
                explicitly_destroy,
            )
        self.context = self.runtime.context
        if allow_hybrid_mode and self.context.gin_min_stride != node_size:
            raise RuntimeError("READ GIN stride must match physical node rank count")
        bootstrap = dict(
            self.runtime.route_bootstrap(self._cpu_storage),
            node=self.node_id,
            fabric_domain=self.fabric_domain,
            fabric_probe_error=self.fabric_error,
        )
        peers = [None] * self.num_ranks
        dist.all_gather_object(peers, bootstrap, group)
        try:
            candidates = owner_routes(peers, self.rank_idx, self.egm)
            try:
                report = dict(self.runtime.probe_routes(peers, candidates))
            except Exception as error:
                report = {"error": str(error)}
            reports = self._exchange_route_reports(report, "probe")
            self.routes, self.owner_fallback_reasons = agree_owner_routes(reports, self.rank_idx)
            try:
                self.runtime.configure_routes(self.routes)
                report = {}
            except Exception as error:
                report = {"error": str(error)}
            self._exchange_route_reports(report, "commit")
        except Exception:
            self.runtime.destroy()
            self.runtime = self.context = None
            self._cpu_storage = None
            raise

        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()

        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()

    def _exchange_route_reports(self, report, phase):
        reports = [None] * self.num_ranks
        dist.all_gather_object(reports, report, group=self.group)
        failures = [f"rank {rank}: {item['error']}" for rank, item in enumerate(reports) if "error" in item]
        if failures:
            raise RuntimeError(f"READ route {phase} failed: {failures[0]}")
        return reports

    def get_route_info(self):
        return dict(
            egm_enabled=self.egm.enabled,
            owner_routes=route_names(self.routes),
            owner_fallback_reasons=list(self.owner_fallback_reasons),
            fabric_domain=self.fabric_domain,
            fabric_probe_error=self.fabric_error,
            memory=self.runtime.route_info(),
        )

    def destroy(self) -> None:
        """Destroy the runtime. Requires ``explicitly_destroy=True`` at construction."""
        super().destroy()
        self.context = None
        self._cpu_storage = None
        self.nccl_comm_handle = None

    @staticmethod
    def get_storage_size_hint(
        group: dist.ProcessGroup,
        num_entries_per_layer: List[int],
        hidden: int,
        num_max_tokens: int,
        num_entries_per_token: int,
        dtype: torch.dtype = torch.bfloat16,
        num_sf_packs: int = 0,
    ) -> Tuple[int, int]:
        """
        Return the aligned main GPU buffer and per-rank RDMA storage sizes.

        The GPU size covers the fetch receive area plus this rank's shard of the FP8
        scaling-factor table; pass ``num_sf_packs=0`` for BF16 storage.
        """
        handle = EngramBuffer._control_handle(group)
        return _C.EngramBuffer.calculate_storage_size(
            handle, num_entries_per_layer, hidden, dtype.itemsize, num_max_tokens, num_entries_per_token, num_sf_packs
        )

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_theoretical_config(
        group: dist.ProcessGroup,
        num_layers: int,
        num_max_tokens: int,
        num_entries_per_token: int,
        num_max_read_requests_per_qp: int = 16,
    ) -> Tuple[int, int, bool]:
        """
        Return the recommended `(num_qps, qp_depth, restrict_rd_atomic)`.
        Pass `restrict_rd_atomic` to the constructor to limit requester read atomics.
        """
        nic_read_mpps = 90 if check_fast_rdma_atomic_support() else 45
        # 返回 (num_qps, qp_depth, restrict_rd_atomic)
        handle = EngramBuffer._control_handle(group)
        return _C.EngramBuffer.get_theoretical_config(
            handle, num_layers, num_max_tokens, num_entries_per_token, nic_read_mpps, num_max_read_requests_per_qp
        )

    def set_config(
        self,
        num_entries_per_layer: List[int],
        hidden: int,
        num_max_tokens: int,
        num_entries_per_token: int,
        dtype: torch.dtype = torch.bfloat16,
        num_sf_packs: int = 0,
    ) -> None:
        self.runtime.set_config(
            num_entries_per_layer, hidden, dtype.itemsize, num_max_tokens, num_entries_per_token, num_sf_packs
        )

    def write(self, storages: List[torch.Tensor], sfs: Optional[List[torch.Tensor]] = None) -> None:
        self.runtime.write(storages, sfs)

    def fetch(
        self, indices: torch.Tensor, num_qps: int = 0, use_tma_aligned_col_major_sf: bool = True
    ) -> List[Callable]:
        """
        Issue the RDMA gets for all layers and return one wait hook per layer.

        Arguments:
            indices: `[num_layers, num_tokens, num_entries_per_token]` entry indices to fetch.
            num_qps: the number of GIN contexts (QPs) to drive, 0 for all of the allocated ones.
            use_tma_aligned_col_major_sf: gather the fetched factors column-major and TMA-aligned.
        """
        num_qps = self.num_allocated_qps if num_qps == 0 else num_qps
        assert num_qps <= self.num_allocated_qps, "Allocated QPs are not enough"
        return self.runtime.fetch(indices, num_qps, use_tma_aligned_col_major_sf)
