from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional
from datetime import timedelta


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    """
    Initialize torch.distributed if launched under torchrun/srun with env:// variables.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)

    import torch
    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1")

    if not dist.is_initialized():
        timeout_s = int(os.environ.get("TORCH_DDP_TIMEOUT_SECONDS", "1800"))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=max(60, timeout_s)),
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def broadcast_object(value: Any, *, src: int, dist_info: DistInfo) -> Any:
    """
    Broadcast a picklable Python object from src to all ranks.
    No-op when not distributed.
    """
    if not dist_info.distributed:
        return value

    import torch.distributed as dist

    obj_list: List[Any] = [value] if dist_info.rank == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def all_reduce_mean(value: float, *, dist_info: DistInfo) -> float:
    """
    All-reduce a scalar (mean) across ranks.
    No-op when not distributed.
    """
    if not dist_info.distributed:
        return float(value)

    import torch
    import torch.distributed as dist

    t = torch.tensor([float(value)], device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(dist_info.world_size)
    return float(t.item())


def barrier(*, dist_info: DistInfo) -> None:
    if not dist_info.distributed:
        return
    import torch.distributed as dist

    dist.barrier()
