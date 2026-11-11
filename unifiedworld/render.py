from itertools import combinations, product
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile


def render_video(trajectories, material, bounds, output, fps=30, size=512, fixed_box=False):
    import imageio.v2 as imageio
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def array(value):
        return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)

    if not trajectories or any(
        not isinstance(title, str) or not title.strip() for title in trajectories
    ):
        raise ValueError("Provide at least one trajectory with a nonempty method title")
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(size, int) or size < 64 or size % 16:
        raise ValueError("size must be a multiple of 16 and at least 64 pixels")
    positions = {title: array(value) for title, value in trajectories.items()}
    shape = next(iter(positions.values())).shape
    if len(shape) != 3 or shape[-1] != 3 or not shape[0] or not shape[1]:
        raise ValueError("Trajectories must have shape [T>0,N>0,3]")
    if any(value.shape != shape or not np.isfinite(value).all() for value in positions.values()):
        raise ValueError("All trajectories must have matching finite [T,N,3] values")
    material, bounds = array(material), array(bounds)
    if material.shape != (shape[1],) or not np.isin(material, [0, 1, 2]).all():
        raise ValueError("material must contain one integer ID 0, 1, or 2 per particle")
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
        raise ValueError("bounds must contain finite box corners with positive side lengths")
    output = Path(output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Video output must use the .mp4 extension")
    if output.exists():
        raise FileExistsError(f"Video already exists: {output}")

    low, high = bounds.astype(float)
    if not fixed_box:
        low = np.minimum(low, np.min([value.min((0, 1)) for value in positions.values()], axis=0))
        high = np.maximum(high, np.max([value.max((0, 1)) for value in positions.values()], axis=0))
    margin = (high - low) * 0.05
    low, high = low - margin, high + margin
    colors = np.array(["#3b82f6", "#f97316", "#d6a51c"])[material.astype(int)]
    figure = Figure(figsize=(size * len(positions) / 100, size / 100), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(positions), subplot_kw={"projection": "3d"}, squeeze=False)[0]
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.02)
    artists = []
    vertices = np.array(list(product(*zip(bounds[0], bounds[1]))))
    for axis, title in zip(axes, positions):
        axis.set_axis_off()
        axis.set_proj_type("ortho")
        axis.view_init(elev=20, azim=-55)
        axis.set(xlim=(low[0], high[0]), ylim=(low[2], high[2]), zlim=(low[1], high[1]))
        axis.set_box_aspect((high - low)[[0, 2, 1]])
        axis.set_title(title, fontsize=16, color="#252525", pad=10)
        for first, second in combinations(vertices, 2):
            if np.count_nonzero(first != second) == 1:
                edge = np.array([first, second])[:, [0, 2, 1]]
                axis.plot(*edge.T, color="#cbd1d8", linewidth=0.7, alpha=0.8)
        initial = positions[title][0][:, [0, 2, 1]]
        artists.append(axis.scatter(*initial.T, c=colors, s=35, linewidths=0, depthshade=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with imageio.get_writer(
            temporary, format="FFMPEG", fps=fps, codec="libx264", quality=8, macro_block_size=16
        ) as writer:
            for frame in range(shape[0]):
                for artist, trajectory in zip(artists, positions.values()):
                    points = trajectory[frame]
                    if fixed_box:
                        # Camera culling only: stored trajectories retain every escaped particle.
                        visible = ((points >= low) & (points <= high)).all(-1)
                        points = points[visible]
                        artist.set_color(colors[visible])
                    artist._offsets3d = tuple(points[:, [0, 2, 1]].T)
                canvas.draw()
                writer.append_data(np.asarray(canvas.buffer_rgba())[..., :3])
        if temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce a video")
        if output.exists():
            raise FileExistsError(f"Video already exists: {output}")
        temporary.rename(output)
    finally:
        figure.clear()
        temporary.unlink(missing_ok=True)
    return output
