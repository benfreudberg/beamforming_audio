from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

Coord3 = Tuple[float, float, float]

def _collect_points(mics: Iterable[Coord3], srcs: Iterable[Coord3],
                    left: Coord3 | None, right: Coord3 | None) -> np.ndarray:
    pts: List[Coord3] = []
    if mics:
        pts.extend(mics)
    if srcs:
        pts.extend(srcs)
    if left:
        pts.append(left)
    if right:
        pts.append(right)
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.array(pts, dtype=float)

def _set_equal_aspect_3d(ax, pts: np.ndarray, margin: float = 0.05) -> None:
    """
    Make 3D axes have equal scale. `margin` is a fractional padding of the data range.
    """
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    spans = (maxs - mins)
    max_span = float(spans.max())
    # Pad by a small fraction of the span (or a small absolute default if span ~ 0)
    pad = max(max_span * margin, 1e-3)
    radii = (max_span / 2.0) + pad
    x0, y0, z0 = centers
    ax.set_xlim(x0 - radii, x0 + radii)
    ax.set_ylim(y0 - radii, y0 + radii)
    ax.set_zlim(z0 - radii, z0 + radii)
    # For newer Matplotlib, this also helps
    try:
        ax.set_box_aspect([1, 1, 1])  # equal aspect if available (mpl >= 3.3)
    except Exception:
        pass

def plot_scene(mic_xyz, src_xyz, left_xyz, right_xyz) -> None:
    """
    Render a 3D scene:
      - Microphones (list of (x,y,z)) in shades of blue
      - Sound sources          ''     in shades of orange
      - Left ear in red, right ear in green
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Microphone Array Simulation Scene")

    # Microphones
    if mic_xyz:
        xs, ys, zs = zip(*mic_xyz)
        cmap_b = plt.cm.Blues
        colors_b = cmap_b(np.linspace(0.4, 0.9, len(xs)))
        ax.scatter(xs, ys, zs, s=60, c=colors_b, marker='o', label="Microphones")

    # Sources
    if src_xyz:
        xs, ys, zs = zip(*src_xyz)
        cmap_o = plt.cm.Oranges
        colors_o = cmap_o(np.linspace(0.4, 0.9, len(xs)))
        ax.scatter(xs, ys, zs, s=60, c=colors_o, marker='^', label="Sound Sources")

    # Ears
    if left_xyz:
        ax.scatter([left_xyz[0]], [left_xyz[1]], [left_xyz[2]],
                   s=80, c='red', marker='s', label='left_ear')
    if right_xyz:
        ax.scatter([right_xyz[0]], [right_xyz[1]], [right_xyz[2]],
                   s=80, c='green', marker='s', label='right_ear')

    # Equal aspect
    pts = _collect_points(mic_xyz, src_xyz, left_xyz, right_xyz)
    _set_equal_aspect_3d(ax, pts, margin=0.05)

    ax.legend(loc="best")
    plt.show()
