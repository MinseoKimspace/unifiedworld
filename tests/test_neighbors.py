import pytest
import torch

from unifiedworld.neighbors import radius_neighbors


def dense_reference(x, radius, max_neighbors):
    distances = (x[:, :, None] - x[:, None, :]).square().sum(-1)
    n = x.shape[1]
    within = (distances <= radius**2) & ~torch.eye(n, dtype=torch.bool, device=x.device)[None]
    counts = within.sum(-1)
    order = distances.masked_fill(~within, torch.inf).argsort(dim=-1, stable=True)
    selected = order[..., :max_neighbors]
    valid = within.gather(-1, selected)
    selected = selected.masked_fill(~valid, -1)
    indices = torch.full(
        (*x.shape[:2], max_neighbors), -1, dtype=torch.long, device=x.device
    )
    indices[..., : selected.shape[-1]] = selected
    return indices, indices >= 0, counts


def assert_matches_reference(x, radius, max_neighbors):
    result = radius_neighbors(x, radius, max_neighbors)
    reference = dense_reference(x, radius, max_neighbors)
    for actual, expected in zip(result, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.device == x.device
        assert not actual.requires_grad
    assert result[0].dtype == result[2].dtype == torch.long
    assert result[1].dtype == torch.bool
    return result


@pytest.mark.parametrize("shape", [(2, 0, 3), (0, 4, 3), (2, 1, 3)])
def test_empty_and_singleton(shape):
    indices, valid, counts = assert_matches_reference(torch.zeros(shape), 0.25, 4)
    assert not valid.any()
    assert (indices == -1).all()
    assert not counts.any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_random_positions_match_dense_reference(dtype):
    x = torch.randn(3, 37, 3, generator=torch.Generator().manual_seed(17), dtype=dtype)
    assert_matches_reference(x, 1.3, 6)


def test_inclusive_radius_and_isolated_batches():
    x = torch.tensor([[[0.0, 0, 0], [0.5, 0, 0], [2, 0, 0]]]).expand(2, -1, -1)
    _, _, counts = assert_matches_reference(x, 0.5, 5)
    torch.testing.assert_close(counts, torch.tensor([[1, 1, 0], [1, 1, 0]]))


def test_overflow_counts_duplicate_coordinates_and_id_ties():
    x = torch.tensor(
        [[[0.0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]]
    )
    indices, _, counts = assert_matches_reference(x, 2.0, 3)
    assert counts.tolist() == [[5, 5, 5, 5, 5, 5]]
    assert indices[0, 0].tolist() == [1, 2, 3]


def test_detached_search_preserves_gather_gradients():
    x = torch.tensor([[[0.0, 0, 0], [0.1, 0, 0], [0.3, 0, 0]]], requires_grad=True)
    indices, valid, _ = radius_neighbors(x, 0.2, 2)
    gathered = x[torch.arange(1)[:, None, None], indices.clamp_min(0)]
    gathered.masked_fill(~valid[..., None], 0).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        radius_neighbors(torch.zeros(1, 2, 3), radius, 2)


@pytest.mark.parametrize("cap", [0, -1, True, 2.5])
def test_rejects_invalid_cap(cap):
    with pytest.raises(ValueError, match="max_neighbors"):
        radius_neighbors(torch.zeros(1, 2, 3), 1, cap)


def test_rejects_nonfinite_and_malformed_positions():
    for x in [torch.zeros(2, 3), torch.zeros(1, 2, 2), torch.zeros(1, 2, 3, dtype=torch.long)]:
        with pytest.raises(ValueError, match="shape"):
            radius_neighbors(x, 1, 2)
    with pytest.raises(ValueError, match="finite positions"):
        radius_neighbors(torch.full((1, 2, 3), float("nan")), 1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_hash_grid_on_torch_current_stream(dtype):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(2, 31, 3, generator=torch.Generator().manual_seed(29)).to(
            device="cuda", dtype=dtype
        )
        assert_matches_reference(x, 1.4, 5)
        tied = torch.zeros(2, 9, 3, device="cuda", dtype=dtype)
        indices, _, counts = assert_matches_reference(tied, 0.25, 3)
        assert indices[0, 0].tolist() == [1, 2, 3]
        assert counts.eq(8).all()
        boundary = torch.tensor(
            [[[0.0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0]]],
            device="cuda",
            dtype=dtype,
        )
        assert_matches_reference(boundary, 0.5, 2)
        assert_matches_reference(x, 0.01, 40)
    stream.synchronize()
