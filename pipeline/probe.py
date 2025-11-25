"""
Core data objects for neural recording pipeline.
"""
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
from pathlib import Path
from loguru import logger
from typing import Optional
from dataclasses import dataclass

from .utils import log_recording, log_timestamps, log_samples, log_intervals, format_duration, format_size


@dataclass
class SessionTimestamps:
    event_ts: np.ndarray
    cont_ts: np.ndarray
    states: np.ndarray
    event_samples: Optional[np.ndarray] = None
    cont_samples: Optional[np.ndarray] = None


class Probe:
    """
    Neural probe with multi-session recordings and global timestamp management.
    Use sync_to() to create a mapping to a target probe's timeline.
    """
    
    def __init__(self, name: str, fs: float = 30000.0):
        self.name = name
        self.fs = fs
        self.dt = 1 / fs
        
        self.recordings: list[si.BaseRecording] = []
        self.session_timestamps: list[SessionTimestamps] = []
        
        self.global_event_ts: list[np.ndarray] = []
        self.intervals: list[tuple[float, float]] = []
        self.session_durations: list[float] = []
        self.starting_states: list[int] = []
        
        self.timestamps_map: Optional[np.ndarray] = None
        self.concatenated: Optional[si.BaseRecording] = None
        self._concat_path: Optional[Path] = None
    
    def load_session(self, session_path: str, stream_id: str) -> si.BaseRecording:
        rec = se.read_openephys(session_path, stream_id=stream_id)
        self.recordings.append(rec)
        return rec
    
    def load_timestamps(self, event_path: str, cont_path: str) -> SessionTimestamps:
        event_ts = np.load(event_path, mmap_mode='r')
        cont_ts = np.load(cont_path, mmap_mode='r')
        states = np.load(event_path.replace('timestamps.npy', 'states.npy'), mmap_mode='r')
        
        ts = SessionTimestamps(event_ts=event_ts, cont_ts=cont_ts, states=states)
        self.session_timestamps.append(ts)
        return ts
    
    def build_global_timestamps(self) -> None:
        logger.info(f"{self.name}: Building global timestamps")
        
        self.global_event_ts = []
        self.intervals = []
        self.session_durations = []
        self.starting_states = []
        t_offset = 0.0
        
        for idx, ts in enumerate(self.session_timestamps):
            session_duration = ts.cont_ts[-1] - ts.cont_ts[0]
            self.session_durations.append(session_duration)
            
            local_event = ts.event_ts - ts.cont_ts[0]
            global_event = local_event + t_offset + self.dt
            
            self.global_event_ts.append(global_event)
            self.intervals.append((t_offset, t_offset + session_duration))
            self.starting_states.append(int(ts.states[0]))
            
            logger.info(f"  Segment {idx}:")
            log_timestamps(ts.cont_ts, "    Continuous")
            log_timestamps(ts.event_ts, "    Local Events")
            log_timestamps(global_event, "    Global Events")
            
            t_offset += session_duration
        
        log_intervals(self.intervals, f"{self.name} global")
        log_timestamps(self.get_global_timestamps(), f"{self.name} events")
    
    def get_global_timestamps(self) -> np.ndarray:
        return np.concatenate(self.global_event_ts)
    
    def get_local_event_timestamps(self, session_idx: int) -> np.ndarray:
        ts = self.session_timestamps[session_idx]
        return ts.event_ts - ts.cont_ts[0]
    
    def sync_to(self, target: 'Probe') -> np.ndarray:
        """Build timestamps_map to target probe's timeline."""
        logger.info(f"{self.name}: Syncing to {target.name}")
        
        self_times = []
        target_times = []
        target_offset = 0.0
        
        for session_idx in range(len(self.intervals)):
            self_local = self.get_local_event_timestamps(session_idx)
            target_local = target.get_local_event_timestamps(session_idx)
            
            if self.starting_states[session_idx] != target.starting_states[session_idx]:
                logger.debug(f"  Session {session_idx}: Aligning chirp edges (state mismatch)")
                self_local, target_local = align_chirp_edges(self_local, target_local)
            
            overlap = min(self.session_durations[session_idx], target.session_durations[session_idx])
            
            if self.session_durations[session_idx] != target.session_durations[session_idx]:
                logger.debug(f"  Session {session_idx}: Duration mismatch ({format_duration(self.session_durations[session_idx])} vs {format_duration(target.session_durations[session_idx])}), using {format_duration(overlap)} overlap")
            
            self_local = self_local[self_local <= overlap]
            target_local = target_local[target_local <= overlap]
            
            min_len = min(len(self_local), len(target_local))
            if min_len == 0:
                logger.warning(f"  Session {session_idx}: No overlapping events")
                target_offset += self.session_durations[session_idx]
                continue
            
            self_local = self_local[:min_len]
            target_local = target_local[:min_len]
            
            self_global = self_local + self.intervals[session_idx][0]
            target_global = target_local + target_offset
            
            self_times.append(self_global)
            target_times.append(target_global)
            
            logger.debug(f"  Session {session_idx}: {min_len} anchor points")
            
            target_offset += self.session_durations[session_idx]
        
        if self_times:
            self.timestamps_map = np.column_stack([
                np.concatenate(self_times),
                np.concatenate(target_times)
            ])
        else:
            self.timestamps_map = np.empty((0, 2))
        
        logger.info(f"  Result: {len(self.timestamps_map)} anchor points")
        return self.timestamps_map
    
    def concatenate(self, output_path: Path, save_kwargs: dict, force: bool = False) -> si.BaseRecording:
        concat_dir = output_path / self.name / 'concat'
        bin_path = concat_dir / 'traces_cached_seg0.raw'
        
        if bin_path.exists() and not force:
            logger.info(f"{self.name}: Loading existing concatenated data from {concat_dir}")
            self.concatenated = si.load(concat_dir)
            self._concat_path = concat_dir
            log_recording(self.concatenated, self.name)
            return self.concatenated
        
        if len(self.recordings) == 0:
            raise ValueError(f"{self.name}: No recordings loaded")
        
        logger.info(f"{self.name}: Concatenating {len(self.recordings)} sessions")
        
        for idx, rec in enumerate(self.recordings):
            log_recording(rec, f"Session {idx}")
        
        concat_rec = (
            si.concatenate_recordings(self.recordings) 
            if len(self.recordings) > 1 
            else self.recordings[0]
        )
        
        concat_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"{self.name}: Saving to {concat_dir}")
        self.concatenated = concat_rec.save(folder=concat_dir, **save_kwargs)
        self._concat_path = concat_dir
        
        log_recording(self.concatenated, f"{self.name} concatenated")
        return self.concatenated
    
    def __repr__(self):
        return f"Probe('{self.name}', sessions={len(self.recordings)}, concatenated={self.concatenated is not None})"


class ADC(Probe):
    """
    ADC recorder with global sample number tracking.
    Global samples computed like timestamps: local = event - cont[0], then offset.
    """
    
    def __init__(self, name: str = 'OneBox-ADC', fs: float = 30300.5):
        super().__init__(name, fs)
        self.global_event_samples: list[np.ndarray] = []
        self.sample_intervals: list[tuple[int, int]] = []
    
    def load_timestamps(self, event_path: str, cont_path: str) -> SessionTimestamps:
        ts = super().load_timestamps(event_path, cont_path)
        
        event_samples = np.load(event_path.replace('timestamps.npy', 'sample_numbers.npy'), mmap_mode='r')
        cont_samples = np.load(cont_path.replace('timestamps.npy', 'sample_numbers.npy'), mmap_mode='r')
        ts.event_samples = event_samples
        ts.cont_samples = cont_samples
        
        return ts
    
    def build_global_timestamps(self) -> None:
        super().build_global_timestamps()
        
        logger.info(f"{self.name}: Building global sample numbers")
        
        self.global_event_samples = []
        self.sample_intervals = []
        sample_offset = 0
        
        for idx, ts in enumerate(self.session_timestamps):
            local_samples = ts.event_samples - ts.cont_samples[0]
            global_samples = local_samples + sample_offset
            self.global_event_samples.append(global_samples)
            
            session_samples = ts.cont_samples[-1] - ts.cont_samples[0]
            self.sample_intervals.append((sample_offset, sample_offset + session_samples))
            
            logger.debug(f"  Segment {idx}: samples [{sample_offset} - {sample_offset + session_samples}], {len(local_samples)} events")
            
            sample_offset += session_samples + 1
        
        log_samples(self.get_global_samples(), f"{self.name} samples")
        logger.info(f"  Total: {sample_offset} samples")
    
    def get_global_samples(self) -> np.ndarray:
        return np.concatenate(self.global_event_samples)


def detect_chirp_reset(event_ts: np.ndarray, limit: int = 500) -> int:
    diffs = np.diff(event_ts[:limit])
    minima = find_peaks(-diffs)[0]
    return minima[0] + 1 if len(minima) > 0 else 0


def align_chirp_edges(ts1: np.ndarray, ts2: np.ndarray, limit: int = 200) -> tuple[np.ndarray, np.ndarray]:
    reset1 = detect_chirp_reset(ts1, limit)
    reset2 = detect_chirp_reset(ts2, limit)
    
    if reset1 == reset2:
        return ts1, ts2
    if reset1 < reset2:
        return ts1, ts2[reset2 - reset1:]
    return ts1[reset1 - reset2:], ts2


def interpolate_to_target(source_times: np.ndarray, timestamps_map: np.ndarray) -> np.ndarray:
    """Interpolate source times to target timebase using timestamps_map [N, 2]."""
    if len(timestamps_map) == 0:
        return np.array([])
    spl = make_interp_spline(timestamps_map[:, 0], timestamps_map[:, 1], k=1)
    return spl(source_times)


def samples_to_timestamps(
    sample_indices: np.ndarray,
    global_samples: np.ndarray,
    global_timestamps: np.ndarray
) -> np.ndarray:
    """Interpolate sample indices to timestamps."""
    spl = make_interp_spline(global_samples, global_timestamps, k=1)
    return spl(sample_indices)