"""CPU词表的主机/fabric域识别与逐owner路由；只在初始化阶段运行。"""

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from contextlib import contextmanager
import os
from pathlib import Path


def env_flag(name, default=False):
    """配置入口只接受明确的0/1，避免字符串或多值模式被隐式转换。"""
    value = os.getenv(name, "1" if default else "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


@dataclass(frozen=True)
class EgmConfig:
    """Buffer生命周期内固定的EGM开关，同时控制机内和同fabric域跨机访问。"""

    enabled: bool = False

    @classmethod
    def from_env(cls):
        return cls(env_flag("ENGRAM_EGM_ENABLE"))


class OwnerRoute(IntEnum):
    RDMA = 0
    EGM = 1


def node_identity():
    # boot_id标识物理主机，避免容器hostname不同或不同机器hostname相同造成误判。
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def fabric_identity(cuda_uuid, enabled):
    if not enabled:
        return None, "disabled"

    # 使用当前CUDA设备的UUID查NVML，不能把CUDA可见序号当成物理GPU序号。
    class Info(ctypes.Structure):
        _fields_ = [
            ("cluster", ctypes.c_ubyte * 16),
            ("status", ctypes.c_int),
            ("clique", ctypes.c_uint),
            ("state", ctypes.c_ubyte),
        ]

    lib = None
    initialized = False
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        rc = lib.nvmlInit_v2()
        if rc:
            raise RuntimeError(f"NVML init error {rc}")
        initialized = True
        handle = ctypes.c_void_p()
        uuid = str(cuda_uuid).encode()
        lib.nvmlDeviceGetHandleByUUID.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        rc = lib.nvmlDeviceGetHandleByUUID(uuid, ctypes.byref(handle))
        if rc:
            raise RuntimeError(f"NVML UUID lookup error {rc}")
        lib.nvmlDeviceGetGpuFabricInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
        info = Info()
        rc = lib.nvmlDeviceGetGpuFabricInfo(handle, ctypes.byref(info))
        # 完成注册、成功状态、非零cluster UUID同时成立才确认属于可直连fabric。
        if rc or info.state != 3 or info.status != 0 or not any(info.cluster):
            raise RuntimeError(f"fabric unavailable: rc={rc}, state={info.state}, status={info.status}")
        return (bytes(info.cluster).hex(), info.clique), None
    except (OSError, AttributeError, RuntimeError) as error:
        return None, str(error)
    finally:
        if initialized:
            lib.nvmlShutdown()


def node_layout(nodes, rank):
    local = nodes.count(nodes[rank])
    if not local or len(nodes) % local or any(nodes[i] != nodes[(i // local) * local] for i in range(len(nodes))):
        raise ValueError("Ranks must be grouped contiguously by physical node, with equal GPUs per node")
    if any(nodes.count(node) != local for node in set(nodes)):
        raise ValueError("Every physical node must contribute the same number of ranks")
    return rank % local, local


def owner_routes(peers, rank, config):
    """开启EGM后机内直接映射；机间还须同fabric域且双方可导出。"""
    me = peers[rank]
    routes = []
    for peer in peers:
        if peer["node"] == me["node"]:
            direct = config.enabled
        else:
            direct = (
                config.enabled
                and me.get("fabric_domain") is not None
                and peer.get("fabric_domain") == me["fabric_domain"]
                and bool(me.get("table_fabric"))
                and bool(peer.get("table_fabric"))
            )
        routes.append(OwnerRoute.EGM if direct else OwnerRoute.RDMA)
    return routes


def agree_owner_routes(reports, rank):
    """逐rank对取双向探测结果的交集；全体先交换结果，再创建RDMA通道。

    一端可映射不代表反向也能映射。统一使用对称路由，使WRITE的owner服务、
    requester QP与对端端点严格匹配；不会把单条链路失败扩大为整节点回退。
    """
    world = len(reports)
    if not 0 <= rank < world:
        raise ValueError("Invalid route negotiation rank")
    for report in reports:
        if (
            len(report["routes"]) != world
            or len(report["errors"]) != world
            or any(route not in (OwnerRoute.RDMA, OwnerRoute.EGM) for route in report["routes"])
        ):
            raise ValueError("Invalid EGM mapping report")
    routes, reasons = [], []
    for peer in range(world):
        direct = reports[rank]["routes"][peer] == reports[peer]["routes"][rank] == OwnerRoute.EGM
        routes.append(OwnerRoute.EGM if direct else OwnerRoute.RDMA)
        failures = []
        for requester, owner in dict.fromkeys([(rank, peer), (peer, rank)]):
            error = reports[requester]["errors"][owner]
            if error:
                failures.append(f"{requester}->{owner}: {error}")
        reasons.append("; ".join(failures) if not direct else "")
    return routes, reasons


def route_names(routes):
    return [OwnerRoute(route).name.lower() for route in routes]


@contextmanager
def local_control_domain():
    # GIN rail/邮箱与FP8 scales继续使用节点内LSA域。机间EGM数据面由独立
    # FABRIC映射实现，不依赖把整个机柜合并成NCCL的本机地址共享域。
    keys = {"NCCL_MNNVL_ENABLE": "0", "EP_REUSE_NCCL_COMM": "0"}
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
