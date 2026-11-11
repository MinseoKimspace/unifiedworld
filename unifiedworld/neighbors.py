"""Deterministic capped radius neighborhoods, with spatial hashing on CUDA."""

import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

os.environ.setdefault("WARP_CACHE_PATH", str(Path(__file__).resolve().parents[1] / ".warp-cache"))

import warp as wp  # noqa: E402

# Keep Warp-owned allocations alive across asynchronous launches. Separate grids
# prevent two Torch streams from rebuilding the same index concurrently.
_GRIDS: dict[tuple[int, int, torch.dtype, int], wp.HashGrid] = {}


@wp.kernel(enable_backward=False)
def _query_neighbors(
    grid: wp.uint64,
    points: wp.array[Any],
    radius: Any,
    particles_per_batch: int,
    max_neighbors: int,
    indices: wp.array2d[wp.int64],
    distances: wp.array2d[Any],
    counts: wp.array[wp.int64],
):
    i = wp.tid()
    group = i // particles_per_batch
    kept = int(0)
    count = wp.int64(0)
    for j in wp.hash_grid_query(grid, points[i], radius, group):
        offset = points[j] - points[i]
        distance = wp.dot(offset, offset)
        if j != i and distance <= radius * radius:
            count += wp.int64(1)
            local_j = wp.int64(j % particles_per_batch)
            last = max_neighbors - 1
            if (
                kept < max_neighbors
                or distance < distances[i, last]
                or (distance == distances[i, last] and local_j < indices[i, last])
            ):
                slot = wp.min(kept, last)
                while slot > 0:
                    previous_distance = distances[i, slot - 1]
                    previous_id = indices[i, slot - 1]
                    if distance > previous_distance or (
                        distance == previous_distance and local_j > previous_id
                    ):
                        break
                    distances[i, slot] = previous_distance
                    indices[i, slot] = previous_id
                    slot -= 1
                distances[i, slot] = distance
                indices[i, slot] = local_j
                kept = wp.min(kept + 1, max_neighbors)
    counts[i] = count


def _cpu_neighbors(x: Tensor, radius: float, max_neighbors: int) -> tuple[Tensor, Tensor, Tensor]:
    """Small-input CPU fallback: O(N²) work, with no dense N×N allocation."""
    batch_size, n, _ = x.shape
    indices = torch.full((batch_size, n, max_neighbors), -1, dtype=torch.long)
    counts = torch.zeros((batch_size, n), dtype=torch.long)
    for batch in range(batch_size):
        for i in range(n):
            distances = (x[batch] - x[batch, i]).square().sum(-1)
            distances[i] = torch.inf
            within = distances <= radius**2
            counts[batch, i] = within.sum()
            # Stable sorting breaks equal-distance ties by the original ID.
            ordered = distances.masked_fill(~within, torch.inf).argsort(stable=True)
            selected = ordered[:max_neighbors]
            valid = within[selected]
            indices[batch, i, : selected.numel()] = selected.masked_fill(~valid, -1)
    return indices, indices >= 0, counts


def _cuda_neighbors(x: Tensor, radius: float, max_neighbors: int) -> tuple[Tensor, Tensor, Tensor]:
    wp.init()
    batch_size, n, _ = x.shape
    # Keep double precision when requested; low precision search is promoted.
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    points = x.to(dtype=dtype).reshape(-1, 3)
    indices = torch.full((batch_size * n, max_neighbors), -1, dtype=torch.long, device=x.device)
    distances = torch.full(indices.shape, torch.inf, dtype=dtype, device=x.device)
    counts = torch.empty(batch_size * n, dtype=torch.long, device=x.device)
    groups = torch.arange(batch_size, dtype=torch.int32, device=x.device).repeat_interleave(n)
    scalar_type, vector_type = (
        (wp.float64, wp.vec3d) if dtype == torch.float64 else (wp.float32, wp.vec3)
    )
    stream = torch.cuda.current_stream(x.device)
    # Bucket resolution affects performance only; every candidate is distance-filtered.
    resolution = max(1, math.ceil((batch_size * n) ** (1 / 3)))
    key = (x.device.index, stream.cuda_stream, dtype, resolution)
    with wp.ScopedStream(wp.stream_from_torch(stream)):
        if key not in _GRIDS:
            _GRIDS[key] = wp.HashGrid(
                resolution, resolution, resolution, device=str(x.device), dtype=scalar_type
            )
        grid = _GRIDS[key]
        warp_points = wp.from_torch(points, dtype=vector_type)
        grid.build(warp_points, radius, groups=wp.from_torch(groups))
        wp.launch(
            _query_neighbors,
            dim=batch_size * n,
            inputs=[
                wp.uint64(grid.id),
                warp_points,
                scalar_type(radius),
                n,
                max_neighbors,
                wp.from_torch(indices),
                wp.from_torch(distances),
                wp.from_torch(counts),
            ],
            device=str(x.device),
        )
    indices = indices.reshape(batch_size, n, max_neighbors)
    return indices, indices >= 0, counts.reshape(batch_size, n)


@torch.no_grad()
def radius_neighbors(
    x: Tensor, radius: float, max_neighbors: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Return nearest neighbors sorted by (squared distance, particle ID).

    Input positions have shape [B, N, 3]. Returned indices (local to each batch)
    and validity masks have shape [B, N, K]; invalid slots contain -1. Counts
    [B, N] include all neighbors inside the inclusive radius before the K cap.
    Self edges are excluded, including for particles at identical positions.
    CUDA uses a grouped Warp HashGrid; CPU uses a small-input brute-force path.
    Search is detached; gather original tensors with these indices to retain
    gradients through the resulting messages.
    """
    if x.ndim != 3 or x.shape[-1] != 3 or not x.is_floating_point():
        raise ValueError("x must be a floating point tensor with shape [B, N, 3]")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if isinstance(max_neighbors, bool) or not isinstance(max_neighbors, int) or max_neighbors <= 0:
        raise ValueError("max_neighbors must be a positive integer")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("radius_neighbors supports CPU and CUDA tensors")
    if not torch.isfinite(x).all():
        raise ValueError("x must contain only finite positions")
    batch_size, n, _ = x.shape
    if batch_size == 0 or n == 0:
        indices = torch.full(
            (batch_size, n, max_neighbors), -1, dtype=torch.long, device=x.device
        )
        counts = torch.zeros((batch_size, n), dtype=torch.long, device=x.device)
        return indices, indices >= 0, counts
    detached = x.detach().contiguous()
    if x.is_cuda:
        return _cuda_neighbors(detached, radius, max_neighbors)
    return _cpu_neighbors(detached, radius, max_neighbors)
