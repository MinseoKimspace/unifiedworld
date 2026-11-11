import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib")
pytest.importorskip("imageio_ffmpeg")
import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg

from unifiedworld.data import canonicalize_scene
from unifiedworld.render import render_video
from unifiedworld.schema import schema_metadata


def positions(frames=3):
    initial = torch.tensor(
        [
            [0.2, 0.7, 0.2],
            [0.35, 0.6, 0.25],
            [0.4, 0.7, 0.6],
            [0.7, 0.4, 0.7],
            [0.6, 0.5, 0.4],
            [0.3, 0.4, 0.7],
        ]
    )
    motion = torch.arange(frames)[:, None, None] * torch.tensor([0.08, -0.1, 0.0])
    return initial[None] + motion


def test_fixed_box_camera_culls_escaped_particles_without_changing_prediction(tmp_path, monkeypatch):
    predicted = positions()
    predicted[:, 0] = 1e20
    original_prediction = predicted.clone()
    snapshots = []
    draw = FigureCanvasAgg.draw

    def inspect(canvas):
        axis = canvas.figure.axes[0]
        snapshots.append((axis.get_xlim(), len(axis.collections[0]._offsets3d[0])))
        return draw(canvas)

    monkeypatch.setattr(FigureCanvasAgg, "draw", inspect)
    output = render_video(
        {"Concat": predicted}, torch.arange(3).repeat_interleave(2),
        torch.tensor([[0., 0, 0], [1., 1, 1]]), tmp_path / "fixed.mp4",
        size=64, fixed_box=True,
    )
    assert output.stat().st_size > 0
    assert snapshots
    for limits, count in snapshots:
        assert limits == pytest.approx((-0.05, 1.05))
        assert count == predicted.shape[1] - 1
    torch.testing.assert_close(predicted, original_prediction, rtol=0, atol=0)


def scene():
    material = torch.arange(3).repeat_interleave(2)
    params = torch.tensor(
        [[0, 0, 1e15, 0, 1], [100, 0, 1e10, 300, 1], [0, 0.5, 1e15, 0, 0]]
    )[material]
    masks = torch.ones_like(params)
    return canonicalize_scene(
        {
            "x": positions(6),
            "v": torch.full((6, 6, 3), 0.1),
            "mass": torch.ones(6, 1),
            "params": params,
            "applicable": masks,
            "known": masks,
            "material": material,
            "acceleration": torch.zeros(3),
            "bounds": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            "dt": 0.02,
            "spacing": 0.028,
            "schema": schema_metadata(),
            "boundary": "box",
            "radius": 0.014,
            "density": 1000.0,
        }
    )


def test_mp4_has_moving_particles_and_only_method_titles(tmp_path, monkeypatch):
    snapshots = []
    original = FigureCanvasAgg.draw

    def inspect(canvas):
        figure = canvas.figure
        snapshots.append(
            [
                (
                    axis.get_title(),
                    axis.axison,
                    axis.get_xlabel(),
                    axis.get_ylabel(),
                    axis.get_zlabel(),
                    axis.get_legend(),
                    tuple(axis.get_xlim()),
                    tuple(axis.get_ylim()),
                    tuple(axis.get_zlim()),
                )
                for axis in figure.axes
            ]
        )
        assert not figure.texts
        assert all(not axis.texts for axis in figure.axes)
        return original(canvas)

    monkeypatch.setattr(FigureCanvasAgg, "draw", inspect)
    sample = scene()
    output = render_video(
        {"Concat": positions(), "Score": positions() + 0.025},
        sample["material"],
        sample["bounds"],
        tmp_path / "comparison.mp4",
        fps=12,
        size=128,
    )
    with imageio.get_reader(output, format="FFMPEG") as reader:
        assert reader.count_frames() == 3
        assert reader.get_meta_data()["fps"] == pytest.approx(12)
        decoded = [frame for frame in reader]
    assert decoded[0].shape == (128, 256, 3)
    assert np.count_nonzero(np.abs(decoded[0].astype(int) - decoded[-1].astype(int)) > 20) > 50
    for panel in (decoded[0][:, :128], decoded[0][:, 128:]):
        assert np.count_nonzero(panel[35:].max(-1) - panel[35:].min(-1) > 30) > 20
    assert len(snapshots) == 3
    for snapshot in snapshots:
        assert [axis[0] for axis in snapshot] == ["Concat", "Score"]
        assert all(axis[1:6] == (False, "", "", "", None) for axis in snapshot)
        assert snapshot[0][6:] == snapshot[1][6:]
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    assert list(tmp_path.glob(".*.mp4")) == []


@pytest.mark.parametrize("kind", ["fps", "shape", "nonfinite", "existing"])
def test_invalid_render_preserves_existing_output(tmp_path, kind):
    sample = scene()
    output, value, fps = tmp_path / "result.mp4", positions(), 30
    expected_error = ValueError
    if kind == "fps":
        fps = float("nan")
    elif kind == "shape":
        value = value[..., :2]
    elif kind == "nonfinite":
        value[1, 1, 1] = float("inf")
    else:
        output.write_bytes(b"existing video")
        expected_error = FileExistsError
    with pytest.raises(expected_error):
        render_video({"Concat": value}, sample["material"], sample["bounds"], output, fps=fps)
    if kind == "existing":
        assert output.read_bytes() == b"existing video"
    else:
        assert not output.exists()
