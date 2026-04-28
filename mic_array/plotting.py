from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim

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


# ---- Source-direction plotting ----

Coord3 = Tuple[float, float, float]


def _color_for_id(sid: Optional[int]) -> Tuple[float, float, float]:
    if sid is None:
        return (0.4, 0.4, 0.4)
    cmap = plt.cm.tab20
    return cmap(int(sid) % 20)[:3]


# Fixed plot bounds for source-direction plots.
AZ_LIM = (-180.0, 180.0)
EL_LIM = (-50.0, 50.0)


def _setup_az_el_axes(ax, title: str) -> None:
    # Mirrored: 180 on the left, -180 on the right.
    ax.set_xlim(AZ_LIM[1], AZ_LIM[0])
    ax.set_ylim(*EL_LIM)
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-50, 51, 10))
    ax.grid(True, alpha=0.3)
    ax.set_title(title)


# 16:9 camera with 105 deg diagonal FOV, looking along +y.
#   half_diag = 52.5 deg
#   half_h    = 52.5 * 16 / sqrt(16^2 + 9^2) ~= 45.76 deg
#   half_v    = 52.5 *  9 / sqrt(16^2 + 9^2) ~= 25.74 deg
# +y direction is at azimuth 90 deg, elevation 0 deg.
_CAM_DIAG_DEG = 105.0
_CAM_ASPECT = (16.0, 9.0)
_CAM_DIAG_NORM = float(np.hypot(*_CAM_ASPECT))
CAMERA_FOV_AZ_CENTER = 90.0
CAMERA_FOV_EL_CENTER = 0.0
CAMERA_FOV_HALF_H = 0.5 * _CAM_DIAG_DEG * _CAM_ASPECT[0] / _CAM_DIAG_NORM
CAMERA_FOV_HALF_V = 0.5 * _CAM_DIAG_DEG * _CAM_ASPECT[1] / _CAM_DIAG_NORM


def _draw_camera_fov(ax) -> None:
    """Outline the field of view of a 16:9 camera with 90 deg diagonal FOV
    pointing along +y on the az/el plot."""
    from matplotlib.patches import Rectangle
    az_min = CAMERA_FOV_AZ_CENTER - CAMERA_FOV_HALF_H
    el_min = CAMERA_FOV_EL_CENTER - CAMERA_FOV_HALF_V
    width = 2 * CAMERA_FOV_HALF_H
    height = 2 * CAMERA_FOV_HALF_V
    rect = Rectangle((az_min, el_min), width, height,
                     fill=False, edgecolor="#1f78b4", linewidth=1.5,
                     linestyle="--", alpha=0.8, zorder=1.5,
                     label="Camera FOV (16:9, 105 deg diag)")
    ax.add_patch(rect)


def _gt_color(i: int) -> Tuple[float, float, float]:
    cmap = plt.cm.tab10
    return cmap(i % 10)[:3]


def _audio_duration_s(audio_path: Optional[Path]) -> Optional[float]:
    if audio_path is None or not audio_path.is_file():
        return None
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(audio_path))
        return float(data.shape[0]) / float(sr)
    except Exception:
        return None


def plot_sources_static(
    sources_per_frame: List[List["object"]],
    times_s: Sequence[float],
    ground_truth: Sequence[Tuple[str, float, float, "np.ndarray"]],
    out_path: Path,
) -> None:
    """Static figure: az-vs-el scatter with named/coloured GT stars + legend,
    and az-vs-time scatter coloured by source_id.

    `ground_truth` items are (name, az_deg, el_deg, envelope) tuples.
    The envelope is unused in the static plot but kept for API symmetry.
    """
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    _setup_az_el_axes(ax1, "Detected source directions")
    _draw_camera_fov(ax1)

    for srcs in sources_per_frame:
        for s in srcs:
            ax1.plot(s.azimuth_deg, s.elevation_deg, "o",
                     color=_color_for_id(getattr(s, "source_id", None)),
                     alpha=0.35, markersize=4)
    for i, gt in enumerate(ground_truth):
        name, az, el = gt[0], gt[1], gt[2]
        ax1.plot(az, el, marker="*", color="black", markersize=18,
                 markerfacecolor=_gt_color(i), markeredgewidth=1.2,
                 label=name)
    if ground_truth:
        ax1.legend(loc="lower left", fontsize=8, framealpha=0.85, title="Sources")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Source azimuth over time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Azimuth (deg)")
    ax2.set_ylim(AZ_LIM[1], AZ_LIM[0])
    ax2.set_yticks(np.arange(-180, 181, 30))
    ax2.grid(True, alpha=0.3)
    for t, srcs in zip(times_s, sources_per_frame):
        for s in srcs:
            ax2.plot(t, s.azimuth_deg, "o",
                     color=_color_for_id(getattr(s, "source_id", None)),
                     markersize=4, alpha=0.7)
    for i, gt in enumerate(ground_truth):
        ax2.axhline(gt[1], color=_gt_color(i), linestyle="--", alpha=0.5, linewidth=1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sources_animation(
    sources_per_frame: List[List["object"]],
    times_s: Sequence[float],
    ground_truth: Sequence[Tuple[str, float, float, "np.ndarray"]],
    out_path: Path,
    trail_frames: int = 20,
    audio_path: Optional[Path] = None,
) -> None:
    """Animated rectangular az-vs-el plot.

    - GT stars are colour-coded with a legend, and pulse in size/alpha based
      on the per-source RMS envelope (sourced from the original audio files).
    - Animation duration is forced to match the muxed audio file's duration,
      so playback stays in sync.
    """
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    _setup_az_el_axes(ax, "Detected sources (animated)")
    _draw_camera_fov(ax)

    times_arr = np.asarray(times_s, dtype=float)
    n_total = len(sources_per_frame)

    # Cap frames to the audio duration so video and audio durations match.
    audio_dur = _audio_duration_s(audio_path)
    if audio_dur is not None and times_arr.size:
        keep = times_arr <= audio_dur + 1e-6
        n_used = int(np.sum(keep))
    else:
        n_used = n_total
        audio_dur = float(times_arr[-1]) if times_arr.size else 1.0

    n_used = max(n_used, 1)
    # Frame rate chosen so video duration ≈ audio_dur (real-time playback).
    output_fps = max(1.0, n_used / max(audio_dur, 1e-3))

    # Pre-create one persistent star artist per ground-truth source so we can
    # animate its size/alpha based on the envelope at the current time.
    gt_artists = []
    for i, gt in enumerate(ground_truth):
        name, az, el = gt[0], gt[1], gt[2]
        art, = ax.plot(az, el, marker="*", color="black",
                       markersize=18, markerfacecolor=_gt_color(i),
                       markeredgewidth=1.2, label=name, linestyle="None")
        gt_artists.append(art)
    if ground_truth:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85, title="Sources")

    time_text = ax.text(0.01, 1.02, "", transform=ax.transAxes, fontsize=10)
    detection_artists: List = []

    def update(i: int):
        for art in detection_artists:
            art.remove()
        detection_artists.clear()

        lo = max(0, i - trail_frames + 1)
        for j in range(lo, i + 1):
            age = i - j
            alpha = max(0.05, 1.0 - age / trail_frames)
            size = 4 + (1.0 - age / max(trail_frames, 1)) * 4
            for s in sources_per_frame[j]:
                col = _color_for_id(getattr(s, "source_id", None))
                line, = ax.plot(s.azimuth_deg, s.elevation_deg, "o",
                                color=col, alpha=alpha,
                                markersize=size, markeredgecolor="white",
                                markeredgewidth=0.5)
                detection_artists.append(line)

        # Modulate ground-truth stars by their per-source envelope.
        for art, gt in zip(gt_artists, ground_truth):
            env = gt[3] if len(gt) > 3 else None
            if env is not None and len(env) > 0:
                v = float(env[min(i, len(env) - 1)])
            else:
                v = 0.0
            v = max(0.0, min(1.0, v))
            art.set_markersize(12 + 18 * v)
            art.set_alpha(0.30 + 0.70 * v)

        time_text.set_text(f"t = {times_arr[i]:.2f} s")
        return detection_artists + gt_artists + [time_text]

    interval_ms = 1000.0 / output_fps
    anim = manim.FuncAnimation(fig, update, frames=n_used, interval=interval_ms, blit=False)

    silent_path = out_path.with_suffix(".silent.mp4")
    have_ffmpeg = shutil.which("ffmpeg") is not None
    try:
        if have_ffmpeg:
            writer = manim.FFMpegWriter(fps=output_fps, bitrate=1800)
            anim.save(str(silent_path), writer=writer)
        else:
            gif_path = out_path.with_suffix(".gif")
            anim.save(str(gif_path), writer=manim.PillowWriter(fps=int(round(output_fps))))
            print(f"[plot_sources_animation] ffmpeg not found; wrote GIF: {gif_path}")
            plt.close(fig)
            return
    finally:
        plt.close(fig)

    if audio_path is not None and audio_path.is_file() and have_ffmpeg:
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(silent_path),
                "-i", str(audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                # Re-encode video with web/QuickTime-friendly settings so the
                # audio track plays in any player.
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "160k",
                "-movflags", "+faststart",
                "-shortest",
                str(out_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg returned {res.returncode}:\n{res.stderr[-2000:]}")
            silent_path.unlink(missing_ok=True)
            print(f"[plot_sources_animation] muxed audio from {audio_path}")
        except Exception as e:
            print(f"[plot_sources_animation] audio mux failed: {e}; keeping silent video at {silent_path}")
            silent_path.replace(out_path)
    else:
        if audio_path is not None and not audio_path.is_file():
            print(f"[plot_sources_animation] audio file not found: {audio_path}")
        silent_path.replace(out_path)
