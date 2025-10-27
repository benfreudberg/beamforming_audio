from pathlib import Path
import numpy as np
from headphone_mic_array.simulator import (
    MicrophoneArraySimulation,
    SoundSource,
    Microphone,
)
from headphone_mic_array.geometry import Node

# todo move these functions to another file
def _x_at_K(A, s0, r, K, include_center: bool):
    """
    Half-aperture position of the K-th positive-side mic for given s0 (center spacing),
    growth r, and whether a center mic exists.

    include_center=False (no mic at 0):
        x_K = s0/2 + s0*sum_{i=1}^{K-1} r^i
             = s0/2 + s0 * r * (1 - r^{K-1}) / (1 - r)   (r != 1)
             = s0/2 + s0*(K-1)                           (r == 1)

    include_center=True (mic at 0, first gap is s0):
        x_K = s0 + s0*sum_{i=1}^{K-1} r^i
             = s0 + s0 * r * (1 - r^{K-1}) / (1 - r)     (r != 1)
             = s0 * K                                    (r == 1)
    """
    if K < 1:
        return 0.0
    if np.isclose(r, 1.0):
        if include_center:
            return s0 * K
        else:
            return s0/2.0 + s0*(K-1)
    # r != 1
    if include_center:
        return s0 + s0 * r * (1.0 - r**(K-1)) / (1.0 - r)
    else:
        return s0/2.0 + s0 * r * (1.0 - r**(K-1)) / (1.0 - r)

def _solve_r_for_A(A, s0, K, include_center: bool, r_min=1.0):
    """
    Solve for r >= r_min such that _x_at_K(A,s0,r,K,include_center) == A.
    Uses monotonicity of x_K in r for K>=2; bisection between [lo, hi].
    """
    lo = max(1.0, r_min)
    x_lo = _x_at_K(A, s0, lo, K, include_center)
    if x_lo > A + 1e-12:
        return None  # even r=1 already overshoots (shouldn’t happen if K chosen right)

    # expand upper bound until we hit/exceed A
    hi = lo * 1.5
    for _ in range(60):
        x_hi = _x_at_K(A, s0, hi, K, include_center)
        if x_hi >= A:
            break
        hi *= 1.8
    else:
        return None

    # bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        x_mid = _x_at_K(A, s0, mid, K, include_center)
        if x_mid < A:
            lo = mid
        else:
            hi = mid
        if abs(x_mid - A) <= 1e-12:
            break
    return 0.5 * (lo + hi)

def geometric_linear_positions_coerced_growth(total_length: float,
                                              center_spacing: float,
                                              r_target: float = 1.0,
                                              include_center: bool = False) -> tuple[np.ndarray, float]:
    """
    Return symmetric x-positions (meters) and the solved growth r (>=1), meeting the aperture exactly.

    Parameters
    ----------
    total_length : tip-to-tip aperture (meters)
    center_spacing : nearest-neighbor spacing at the center:
        - include_center=False: spacing between the two innermost mics
        - include_center=True : spacing from x=0 to the first mic on either side
    r_target : preferred growth factor (>=1); we choose K whose solved r is closest to this
    include_center : whether to place a mic at x=0

    Returns
    -------
    xs : np.ndarray of positions (meters), symmetric about 0
    r_solved : float growth factor actually used (>=1)
    """
    A = float(total_length) / 2.0
    s0 = float(center_spacing)
    if A <= 0 or s0 <= 0:
        raise ValueError("total_length and center_spacing must be positive.")
    if r_target < 1.0:
        r_target = 1.0

    # feasibility check for the first positive mic
    min_first = s0 if include_center else (s0 / 2.0)
    if min_first > A + 1e-12:
        raise ValueError("center_spacing too large for the given total_length.")

    # For r=1, minimal half-aperture to place K positive-side mics:
    # include_center=False: A_min(K) = s0/2 + s0*(K-1) = s0*(K - 0.5)
    # include_center=True : A_min(K) = s0*K
    if include_center:
        K_max = int(np.floor(A / s0))  # require s0*K <= A
    else:
        K_max = int(np.floor(1.0 + (A - s0/2.0) / s0))  # require s0*(K - 0.5) <= A

    if K_max < 1:
        # Only possible if A equals the first position exactly
        if np.isclose(A, min_first, atol=1e-12):
            r_solved = 1.0
            pos_plus = np.array([min_first])
        else:
            raise ValueError("Aperture too small for even one positive-side mic with given center_spacing.")
    else:
        K_candidates = [K for K in range(1, K_max + 1)]  # allow K=1..K_max
        best = None  # (distance_to_target, r_solved, K)
        r_solutions = {}
        for K in K_candidates:
            # check feasibility at r=1
            if _x_at_K(A, s0, 1.0, K, include_center) > A + 1e-12:
                continue
            rK = _solve_r_for_A(A, s0, K, include_center, r_min=1.0)
            if rK is None:
                continue
            r_solutions[K] = rK
            # compare in log-space for scale-invariant closeness
            dist = abs(np.log(rK) - np.log(r_target))
            if (best is None) or (dist < best[0]):
                best = (dist, rK, K)
        if best is None:
            raise ValueError("No feasible (K, r) pair found; adjust total_length or center_spacing.")

        _, r_solved, K_best = best

        # Build exact positive-side positions using r_solved
        pos_plus = []
        for k in range(1, K_best + 1):
            if np.isclose(r_solved, 1.0):
                xk = (s0 * k) if include_center else (s0/2.0 + s0*(k-1))
            else:
                if include_center:
                    xk = s0 + s0 * r_solved * (1.0 - r_solved**(k-1)) / (1.0 - r_solved)
                else:
                    xk = s0/2.0 + s0 * r_solved * (1.0 - r_solved**(k-1)) / (1.0 - r_solved)
            pos_plus.append(xk)
        pos_plus = np.array(pos_plus)
        # exact endpoint
        pos_plus[-1] = A

    # mirror
    neg = -pos_plus[::-1]
    xs = np.concatenate([neg, np.array([0.0]), pos_plus]) if include_center else np.concatenate([neg, pos_plus])
    xs = np.unique(np.round(xs, 15))
    xs.sort()
    return xs, float(r_solved)

def circular_array_xy(
    diameter: float,
    num_elements: int,
    first_angle: float = 0.0,
    degrees: bool = False,
) -> np.ndarray:
    """
    Return x–y coordinates for mics placed uniformly on a circle.

    Parameters
    ----------
    diameter : float
        Circle diameter in meters.
    num_elements : int
        Number of microphones (>= 1).
    first_angle : float, optional
        Angular offset for the first element (default 0).
        Measured from +x axis, counterclockwise.
    degrees : bool, optional
        If True, 'first_angle' is in degrees; otherwise radians.

    Returns
    -------
    np.ndarray
        Array of shape (num_elements, 2) with columns [x, y] (meters).
    """
    if num_elements < 1:
        raise ValueError("num_elements must be >= 1")
    if diameter < 0:
        raise ValueError("diameter must be non-negative")

    r = diameter * 0.5
    theta0 = np.deg2rad(first_angle) if degrees else first_angle
    step = 2.0 * np.pi / max(1, num_elements)  # ok for num_elements==1

    k = np.arange(num_elements, dtype=float)
    theta = theta0 + k * step

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

# project_root is where this script lives
base_dir = Path(__file__).resolve().parent
audio_dir = base_dir / "input_audio_files"

descriptive_voice_file = audio_dir / "descriptive.wav"
informative_voice_file = audio_dir / "informative.wav"
dialog_voice_a = audio_dir / "dialog_voice_a.wav"
dialog_voice_b = audio_dir / "dialog_voice_b.wav"
food_hall_L = audio_dir / "food_hall_L.wav"
food_hall_R = audio_dir / "food_hall_R.wav"

sim = MicrophoneArraySimulation()
sim.add_sound_source(SoundSource(4, 1, 0, descriptive_voice_file))
sim.add_sound_source(SoundSource(0, 2, 0, informative_voice_file))
sim.add_sound_source(SoundSource(1.5, 2, 0, dialog_voice_a))
sim.add_sound_source(SoundSource(-1.5, 2, 0, dialog_voice_b))
sim.add_sound_source(SoundSource(-4, -5, 0, food_hall_L))
sim.add_sound_source(SoundSource(4, -5, 0, food_hall_R))
sim.add_target("dialog_voice_a", 1.5, 2, 0)
sim.add_target("dialog_voice_b", -1.5, 2, 0)

circular = True
if not circular:
    total_length = .24      # meters
    center_spacing = 0.018    # meters between the two innermost mics
    r_target = 1.3            # outward spacing ratio (>1 grows, 1.0 = uniform)
    include_center = False   # set True to place a mic at x=0

    xs, r_used = geometric_linear_positions_coerced_growth(
        total_length, center_spacing, r_target, include_center
    )

    print(f"Using growth factor r = {r_used:.6f} with {len(xs)} microphones.")
    center_mic = None
    origin = Node(0, 0, 0)
    for x in xs:
        z = 0.0
        name = f"mic_x{int(round(x*100)):02d}cm"
        mic = Microphone(name, float(x), 0.0, float(z))
        sim.add_microphone(mic)
        if center_mic is None or center_mic.distance_to(origin) > mic.distance_to(origin):
            center_mic = mic
else:
    xy_inner = circular_array_xy(diameter=0.06, num_elements=8, first_angle=2*np.pi/16)
    xy_mid = circular_array_xy(diameter=0.14, num_elements=6, first_angle=0)
    xy_outer = circular_array_xy(diameter=0.3, num_elements=6, first_angle=-2*np.pi/18)
    xy_outer2 = circular_array_xy(diameter=0.45, num_elements=6, first_angle=2*np.pi/18)

    for i, (x, y) in enumerate(xy_inner):
        name = f"mic_ring_{i:02d}"
        mic = Microphone(name, float(x), float(y), 0.0)
        sim.add_microphone(mic)
    for i, (x, y) in enumerate(xy_mid):
        name = f"mic_ring_{i:02d}"
        mic = Microphone(name, float(x), float(y), 0.0)
        sim.add_microphone(mic)
    for i, (x, y) in enumerate(xy_outer):
        name = f"mic_ring_{i:02d}"
        mic = Microphone(name, float(x), float(y), 0.0)
        sim.add_microphone(mic)
    for i, (x, y) in enumerate(xy_outer2):
        name = f"mic_ring_{i:02d}"
        mic = Microphone(name, float(x), float(y), 0.0)
        sim.add_microphone(mic)
    center_mic = Microphone("center", 0.0, 0.0, 0.0)
    sim.add_microphone(center_mic)

# Ambient (all sources → ears)
sim.apply_ambient_audio_to_ears()
sim.export_ears_stereo("raw_ambient.wav")
sim.reset_recordings()

# Record with the mic grid → build target → apply to ears
sim.run_recording()

# Export target tracks
sim.create_target_tracks_DS()
sim.export_target_tracks("DS")
sim.create_target_tracks_MVDR()
sim.export_target_tracks("MVDR")
sim.apply_targets_to_ears()
sim.export_ears_stereo("targets_applied_to_ears.wav")

# Export center mic recording (if found)
if center_mic is not None:
    center_mic.export()

# Visualize setup
sim.show_scene_3d()
