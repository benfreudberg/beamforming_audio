"""
bird_finder.py
==============
Bird-finding pipeline that orchestrates the general-purpose analysis methods
on MicrophoneArraySimulation to locate bird sound sources.

The heavy lifting (rolling FFT, peak picking, direction-of-arrival, clustering)
lives as methods on MicrophoneArraySimulation and is therefore available to any
microphone-array application. This module provides only the bird-specific
orchestration layer: BirdFinderConfig and BirdFinderPipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .simulator import (
    FrequencyFrame,
    MicrophoneArraySimulation,
    PeakEntry,
    PeakTracker,
    SourceDirection,
)


@dataclass
class BirdFinderConfig:
    """Tuning knobs for the BirdFinderPipeline."""
    window_s: float = 4.0
    hop_ms: float = 100.0
    n_peaks: int = 20
    min_freq_hz: float = 1000.0
    max_freq_hz: float = 20000.0
    freq_tolerance_hz: float = 50.0
    max_miss_frames: int = 5
    high_freq_bias: float = 0.01
    angle_threshold_deg: float = 15.0
    reference_mic_index: int = 0   # index into sim._microphones used for peak picking


class BirdFinderPipeline:
    """
    Orchestrates the full bird-finding analysis on a completed simulation.

    Usage
    -----
    ::

        pipeline = BirdFinderPipeline(sim, config=BirdFinderConfig())
        pipeline.run()
        pipeline.plot_sources()
        pipeline.export_target_tracks()

    Parameters
    ----------
    sim : MicrophoneArraySimulation
        A simulation instance where ``run_recording()`` has already been called.
    config : BirdFinderConfig
        Algorithm parameters.
    """

    def __init__(self, sim: MicrophoneArraySimulation, config: Optional[BirdFinderConfig] = None) -> None:
        self.sim = sim
        self.config = config or BirdFinderConfig()
        # Results populated by run()
        self.frames: List[FrequencyFrame] = []
        self.tracked_peaks_per_frame: List[List[PeakEntry]] = []
        self.sources_per_frame: List[List[SourceDirection]] = []

    def run(self) -> None:
        """
        Execute the full pipeline:
          1. sim.compute_rolling_fft() on the reference microphone.
          2. sim.find_peak_frequencies_in_frame() per frame.
          3. PeakTracker.update() across frames.
          4. sim.localize_frequency() for each active frequency track.
          5. sim.cluster_directions() to merge nearby DOAs into sources.
        """
        raise NotImplementedError("BirdFinderPipeline.run is not yet implemented.")

    def plot_sources(self, animate: bool = False) -> None:
        """
        Visualise source directions on a polar/azimuth plot.
        If *animate* is True, produce a time-evolving animation.
        """
        raise NotImplementedError("BirdFinderPipeline.plot_sources is not yet implemented.")

    def export_target_tracks(self) -> None:
        """
        For each identified source, beamform a target track and export WAV.
        """
        raise NotImplementedError("BirdFinderPipeline.export_target_tracks is not yet implemented.")
