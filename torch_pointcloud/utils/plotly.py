from typing import Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import torch


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_points(
    points: Union[np.ndarray, torch.Tensor],
    colors: Union[np.ndarray, torch.Tensor, List[str], str] = "blue",
    labels: Optional[Union[np.ndarray, List[Union[str, None]], str]] = None,
    colorscale: str | None = None,
    opacity: float = 0.8,
    size: int = 2,
    max_points: int = 10_000,
    name: str | None = "point",
    fig: go.Figure | None = None,
) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    N, _ = points.shape
    num_points = min(max_points, N)
    sampled_idxs = np.random.choice(N, num_points, replace=False)

    if isinstance(labels, (list, np.ndarray)):
        labels = np.array(labels)[sampled_idxs]
    else:
        labels = np.array([labels] * num_points)
    assert isinstance(labels, np.ndarray)  # for typing

    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    if isinstance(colors, np.ndarray) and colors.shape == (N, 3):
        if colors.dtype == float and colors.max() <= 1:
            colors = (colors * 255).astype(np.uint8)
        assert colors.dtype == np.uint8, "colors must be in uint8 format"
        colors = colors[sampled_idxs]
        colors = [rgb_to_hex(*c) for c in colors]
    elif isinstance(colors, list) and len(colors) == N:
        colors = np.array(colors)[sampled_idxs]
    elif isinstance(colors, str):
        colors = [colors] * num_points

    points = points[sampled_idxs]
    # Above the above to add a legend for each label
    for label in set(labels):
        label_idxs = np.where(labels == label)[0]
        label_points = points[label_idxs]
        label = str(label)

        fig.add_trace(
            go.Scatter3d(
                name=f"{name} {label}",
                x=label_points[:, 0],
                y=label_points[:, 1],
                z=label_points[:, 2],
                mode="markers",
                marker=dict(
                    size=size,
                    color=colors[label_idxs],
                    colorscale=colorscale,
                    opacity=0.1 if label == "-1" else opacity,
                ),
            )
        )

    return fig


def plot_bboxes(
    bboxes: np.ndarray,
    colors: Optional[Union[str, List[Union[str, None]]]] = None,
    labels: Optional[Union[str, List[Union[str, None]]]] = None,
    width: int = 2,
    fig: go.Figure | None = None,
) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    labels = labels if isinstance(labels, list) else [labels] * len(bboxes)
    colors = colors if isinstance(colors, list) else [colors] * len(bboxes)
    if not len(labels) == len(colors) == len(bboxes):
        raise ValueError(
            "bboxes, labels and colors must have the same length, "
            f"got {len(bboxes)}, {len(labels)} and {len(colors)}"
        )

    label_to_color: Dict[Union[str, None], str] = {}

    for bbox, color, label in zip(bboxes, colors, labels):
        cx, cy, cz, dx, dy, dz, *_ = bbox

        corners = [
            (-dx / 2, -dy / 2, -dz / 2),
            (-dx / 2, dy / 2, -dz / 2),
            (dx / 2, dy / 2, -dz / 2),
            (dx / 2, -dy / 2, -dz / 2),
            (-dx / 2, -dy / 2, dz / 2),
            (-dx / 2, dy / 2, dz / 2),
            (dx / 2, dy / 2, dz / 2),
            (dx / 2, -dy / 2, dz / 2),
        ]
        corners = [(cx + x, cy + y, cz + z) for x, y, z in corners]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

        # Check if we have previously seen this label or not
        showlegend = label not in label_to_color

        # Get associated color if it is None
        if color is None:
            color = label_to_color.get(label)
        if color is None:
            color = rgb_to_hex(*np.random.choice(range(256), size=3))
            label_to_color[label] = color

        for i, (s, e) in enumerate(edges):
            fig.add_trace(
                go.Scatter3d(
                    name=f"bbox {label}",
                    x=[corners[s][0], corners[e][0]],
                    y=[corners[s][1], corners[e][1]],
                    z=[corners[s][2], corners[e][2]],
                    mode="lines",
                    line=dict(color=color, width=width),
                    showlegend=i == 0 and showlegend,
                    legendgroup=f"bbox_{label}",
                )
            )

    return fig
