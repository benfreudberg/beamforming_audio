from pathlib import Path
import numpy as np
from headphone_mic_array.simulator import (
    MicrophoneArraySimulation,
    SoundSource,
    Microphone,
)

# todo move to another file

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

# project_root is where this script lives
base_dir = Path(__file__).resolve().parent
audio_dir = base_dir / "input_audio_files"

descriptive_voice_file = audio_dir / "descriptive.wav"
informative_voice_file = audio_dir / "informative.wav"

sim = MicrophoneArraySimulation()
sim.add_sound_source(SoundSource(10, 20, 0, descriptive_voice_file))
sim.add_sound_source(SoundSource(0, 20, 0, informative_voice_file))
sim.add_target("descriptive", 10, 20, 0)
sim.add_target("informative", 0, 20, 0)


total_length = 2.0      # meters
center_spacing = 0.02    # meters between the two innermost mics
r_target = 1.3            # outward spacing ratio (>1 grows, 1.0 = uniform)
include_center = False   # set True to place a mic at x=0

xs, r_used = geometric_linear_positions_coerced_growth(
    total_length, center_spacing, r_target, include_center
)
print(r_used)
print(xs)

print(f"Using growth factor r = {r_used:.6f} with {len(xs)} microphones.")
center_mic = None
for x in xs:
    z = 0.0
    name = f"mic_x{int(round(x*100)):02d}cm_z{int(round(z*100)):02d}cm"
    mic = Microphone(name, float(x), 0.0, float(z))
    sim.add_microphone(mic)
    if np.isclose(x, 0.0) and np.isclose(z, 0.0):
        center_mic = mic

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
# sim.apply_targets_to_ears()
# sim.export_ears_stereo("targets_applied_to_ears.wav")

# Export center mic recording (if found)
if center_mic is not None:
    center_mic.export()

# Visualize setup
sim.show_scene_3d()







def geometric_linear_positions(total_length: float,
                               center_spacing: float,
                               growth: float,
                               include_center: bool = False) -> np.ndarray:
    """
    Return symmetric x-positions (meters) for a geometrically spaced linear array.
    - total_length: full aperture in meters (distance from leftmost to rightmost mic).
    - center_spacing: spacing between the two innermost mics (meters).
    - growth: geometric ratio (>0). =1 gives uniform spacing; >1 grows outward.
    - include_center: if True, include a mic at x=0 as well.

    Spacings outward from the center pair are: s0=center_spacing (between ±1),
    then s1=s0*growth (between +1↔+2 and -1↔-2), then s2=s1*growth, etc.
    The last spacing is adjusted (only that one) so the outermost mic hits ±total_length/2.
    """
    A = float(total_length) / 2.0  # half-aperture
    s0 = float(center_spacing)
    r  = float(growth)

    if A <= 0 or s0 <= 0 or r <= 0:
        raise ValueError("total_length, center_spacing, and growth must be positive.")
    if s0 > 2*A:
        raise ValueError("center_spacing too large for the given total_length.")

    # Handle r == 1 (uniform outward spacing)
    pos_plus = []
    if np.isclose(r, 1.0):
        x = s0 / 2.0
        if x > A:
            raise ValueError("center_spacing too large for the given total_length.")
        # keep adding s0 until we would exceed A
        while x + s0 <= A + 1e-12:
            pos_plus.append(x)
            x += s0
        # adjust last step to hit A exactly (only if we don't already land on A)
        if not np.isclose(x, A):
            pos_plus.append(A)
        else:
            pos_plus.append(x)
    else:
        # Geometrically increasing gaps: x_k = s0/2 + s0 * sum_{i=1}^{k-1} r^i
        # Keep adding until next would exceed A, then adjust final to A.
        k = 1
        # first candidate
        xk = s0 / 2.0
        if xk > A + 1e-12:
            raise ValueError("center_spacing too large for the given total_length.")

        # Add as many as fit strictly within A
        while True:
            if xk <= A + 1e-12:
                pos_plus.append(xk)
            # propose next x_{k+1}
            k += 1
            # sum_{i=1}^{k-1} r^i = r * (1 - r^{k-1}) / (1 - r)
            geom_sum = r * (1.0 - r**(k-1)) / (1.0 - r)
            x_next = s0 / 2.0 + s0 * geom_sum
            if x_next > A + 1e-12:
                break
            xk = x_next

        # If we didn't already land on A, append A as the last (adjusting only the last spacing)
        if not np.isclose(pos_plus[-1], A, atol=1e-12):
            pos_plus.append(A)

    # Build symmetric array: (optional center), then negative side, then positive side
    pos_plus = np.array(sorted(set(np.round(pos_plus, 15))))  # unique & stable
    neg = -pos_plus[::-1]
    if include_center:
        xs = np.concatenate([neg, np.array([0.0]), pos_plus])
    else:
        xs = np.concatenate([neg, pos_plus])
    return xs