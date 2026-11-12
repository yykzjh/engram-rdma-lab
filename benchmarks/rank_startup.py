"""Fresh-process GPU placement shared by the two benchmark launchers.

Device discovery runs in the parent. Workers use multiprocessing *spawn*, so
their CUDA enumeration can still be configured before their first CUDA call.
Bind before importing a backend that initializes the CUDA driver (e.g. DeepEP).
This module shares no backend state or communication implementation.
"""

import os
import subprocess


def assigned_devices(local_ranks):
    import torch

    count = torch.cuda.device_count()
    if count < local_ranks:
        raise ValueError(f"Need {local_ranks} assigned GPUs, found {count}")
    # Resolve through CUDA, respecting the parent's visibility and device order.
    # nvidia-smi is only used to find physical indices for the default HCA names.
    physical = {}
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"], text=True
        )
        for row in rows.splitlines():
            uuid, index = row.split(",")
            physical[uuid.strip()] = int(index)
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    devices = []
    for ordinal in range(count):
        uuid = str(torch.cuda.get_device_properties(ordinal).uuid)
        if not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        devices.append({"uuid": uuid, "physical_index": physical.get(uuid)})
    if len({device["uuid"] for device in devices}) != count:
        raise ValueError("Assigned GPU UUIDs must be unique")
    return devices


def bind_rank(local_rank, devices, *, write=False):
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Bind the rank before its first CUDA initialization")
    if not 0 <= local_rank < len(devices):
        raise ValueError("Invalid local rank")
    owner_first = os.environ.get("ENGRAM_EGM_ENABLE", "0") == "1"
    if owner_first:
        order = [local_rank] + [i for i in range(len(devices)) if i != local_rank]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[i]["uuid"] for i in order)
    ordinal = 0 if owner_first else local_rank
    if write and "ENGRAM_WRITE_HCA" not in os.environ:
        physical = devices[local_rank]["physical_index"]
        if physical is None:
            raise RuntimeError("Set ENGRAM_WRITE_HCA explicitly when physical GPU indices are unavailable")
        os.environ["ENGRAM_WRITE_HCA"] = f"mlx5_bond_{physical}"
    torch.cuda.set_device(ordinal)
    actual = str(torch.cuda.get_device_properties(ordinal).uuid).removeprefix("GPU-")
    if actual != devices[local_rank]["uuid"].removeprefix("GPU-"):
        raise RuntimeError("GPU enumeration changed the rank's physical GPU assignment")
    return ordinal, {
        "owner_first": owner_first,
        "logical_local_rank": local_rank,
        "cuda_ordinal": ordinal,
        "physical_gpu_uuid": devices[local_rank]["uuid"],
        "physical_gpu_index": devices[local_rank]["physical_index"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "write_hca": os.environ.get("ENGRAM_WRITE_HCA") if write else None,
    }
