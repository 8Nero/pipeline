import re
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
from pathlib import Path
from loguru import logger
from typing import Optional, Literal
from dataclasses import dataclass

from .utils import log_recording, log_intervals, log_samples, log_timestamps


@dataclass
class SessionData:
    event_ts: np.ndarray
    cont_ts: np.ndarray
    states: np.ndarray
    event_samples: np.ndarray
    cont_samples: np.ndarray


ReferenceMode = Literal['samples', 'timestamps']


class Probe:
    """Probe with multi-session recordings and synchronization."""
    
    def __init__(self, name: str, fs: float = 30000.0):
        self.name = name
        self.fs = fs
        
        self.recordings: list[si.BaseRecording] = []
        self.sessions: list[SessionData] = []
        
        # Store global references for both modes: 'samples' and 'timestamps'
        self.global_events: dict[ReferenceMode, np.ndarray] = {}
        self.intervals: dict[ReferenceMode, list[tuple]] = {}
        self.starting_states: dict[ReferenceMode, list[int]] = {}
        
        self.sync_map: Optional[np.ndarray] = None
        self.concatenated: Optional[si.BaseRecording] = None
    
    def load_from_sessions(self, session_paths: list[str]) -> None:
        """Load recordings and timestamps from session paths."""
        logger.info(f"{self.name}: Loading from {len(session_paths)} sessions")

        for session_path in session_paths:
            path = Path(session_path)
            logger.debug(f"Session: {path}")
            
            # Find and load recording stream
            stream_names, stream_ids = se.get_neo_streams('openephysbinary', session_path)
            for stream_name, stream_id in zip(stream_names, stream_ids):
                if self.name in stream_name and "SYNC" not in stream_name:
                    rec = se.read_openephys(session_path, stream_id=stream_id)
                    self.recordings.append(rec)
                    log_recording(rec, f"  {path.name}")
                    break
            
            # Find and load timestamps
            event_paths = self._find_timestamp_file(path, 'events')
            cont_paths = self._find_timestamp_file(path, 'continuous')
            for event_path, cont_path in zip(event_paths, cont_paths):
                logger.debug(f"    Loading timestamps from {event_path.relative_to(path)} and {cont_path.relative_to(path)}")
                self._load_session_data(str(event_path), str(cont_path))
    
    def _find_timestamp_file(self, session_path: Path, folder: str) -> Optional[str]:
        """Find timestamps.npy for this probe in given folder (sorted by Recording number)."""
        def extract_rec_num(p):
            match = re.search(r'[Rr]ecording(\d+)', str(p))
            return int(match.group(1)) if match else 0
        
        ts_files = sorted(session_path.glob(f'**/{folder}/**/timestamps.npy'), key=extract_rec_num)
        ts_files = [p for p in ts_files if self.name in str(p)]
        return ts_files
    
    def _load_session_data(self, event_path: str, cont_path: str) -> SessionData:
        """Load session timestamps and sample numbers."""
        event_ts = np.load(event_path, mmap_mode='r')
        cont_ts = np.load(cont_path, mmap_mode='r')
        states = np.load(event_path.replace('timestamps.npy', 'states.npy'), mmap_mode='r')
        event_samples = np.load(event_path.replace('timestamps.npy', 'sample_numbers.npy'), mmap_mode='r')
        cont_samples = np.load(cont_path.replace('timestamps.npy', 'sample_numbers.npy'), mmap_mode='r')
        
        session = SessionData(
            event_ts=event_ts,
            cont_ts=cont_ts,
            states=states,
            event_samples=event_samples,
            cont_samples=cont_samples
        )
        self.sessions.append(session)
        return session
    
    def build_global_references(self, mode: ReferenceMode = 'samples') -> np.ndarray:
        """Build global event references across sessions. Returns concatenated global events."""
        logger.info(f"{self.name}: Building global references using {mode}")
        
        global_events = []
        intervals = []
        starting_states = []
        offset = 0
        
        for idx, session in enumerate(self.sessions):
            if mode == 'samples':
                local = session.event_samples - session.cont_samples[0]
                session_length = int(session.cont_samples[-1] - session.cont_samples[0])
                log_samples(session.event_samples, f"  Session {idx} event_samples")
                log_samples(session.cont_samples, f"  Session {idx} cont_samples")
                log_samples(local, f"  Session {idx} local")
            else:
                local = session.event_ts - session.cont_ts[0]
                session_length = session.cont_ts[-1] - session.cont_ts[0]
                log_timestamps(session.event_ts, f"  Session {idx} event_ts")
                log_timestamps(session.cont_ts, f"  Session {idx} cont_ts")
                log_timestamps(local, f"  Session {idx} local")
            
            global_ev = local + offset
            global_events.append(global_ev)
            intervals.append((offset, offset + session_length))
            starting_states.append(int(session.states[0]))
            
            if mode == 'samples':
                log_samples(global_ev, f"  Session {idx} global")
                logger.debug(f"  Session {idx}: interval=[{offset}, {offset + session_length}], length={session_length}")
            else:
                log_timestamps(global_ev, f"  Session {idx} global")
                logger.debug(f"  Session {idx}: interval=[{offset:.6f}, {offset + session_length:.6f}]s, length={session_length:.6f}s")
            
            offset += session_length + (1 if mode == 'samples' else 1/self.fs)
        
        # Store in dict
        self.global_events[mode] = np.concatenate(global_events)
        self.intervals[mode] = intervals
        self.starting_states[mode] = starting_states
        
        log_intervals(intervals, f"{self.name} ({mode})")
        return self.global_events[mode]
    
    def get_global_events(self, mode: ReferenceMode) -> np.ndarray:
        """Get concatenated global events in specified mode."""
        if mode not in self.global_events:
            raise ValueError(f"Global {mode} not built. Call build_global_references('{mode}') first.")
        return self.global_events[mode]
    
    def _get_local_events(self, session_idx: int, mode: ReferenceMode) -> np.ndarray:
        """Get local events for a session in specified mode."""
        session = self.sessions[session_idx]
        if mode == 'samples':
            return session.event_samples - session.cont_samples[0]
        elif mode == 'timestamps':
            return session.event_ts - session.cont_ts[0]
    
    def _get_session_length(self, session_idx: int, mode: ReferenceMode):
        """Get session length in specified mode."""
        session = self.sessions[session_idx]
        if mode == 'samples':
            return int(session.cont_samples[-1] - session.cont_samples[0])
        elif mode == 'timestamps':
            return session.cont_ts[-1] - session.cont_ts[0]
    
    def sync_to(self, target: 'Probe', mode: ReferenceMode = 'samples') -> np.ndarray:
        """Build sync_map to target probe's reference frame."""
        logger.info(f"{self.name}: Syncing to {target.name} (using {mode})")
        
        self_values = []
        target_values = []
        log_fn = log_samples if mode == 'samples' else log_timestamps
        
        for session_idx in range(len(self.intervals[mode])):
            self_local = self._get_local_events(session_idx, mode)
            target_local = target._get_local_events(session_idx, mode)
            
            log_fn(self_local, f"  Session {session_idx} self_local")
            log_fn(target_local, f"  Session {session_idx} target_local")
            
            if self.starting_states[mode][session_idx] != target.starting_states[mode][session_idx]:
                logger.debug(f"  Session {session_idx}: Aligning edges (self_state={self.starting_states[mode][session_idx]}, target_state={target.starting_states[mode][session_idx]})")
                self_local, target_local = align_edges(self_local, target_local)
                log_fn(self_local, f"  Session {session_idx} self_local (aligned)")
                log_fn(target_local, f"  Session {session_idx} target_local (aligned)")
            
            min_len = min(len(self_local), len(target_local))
            if min_len == 0:
                logger.warning(f"  Session {session_idx}: No overlapping events")
                continue
            
            logger.debug(f"  Truncating past {min_len}")
            self_local = self_local[:min_len]
            target_local = target_local[:min_len]
            
            self_global = self_local + self.intervals[mode][session_idx][0]
            target_global = target_local + target.intervals[mode][session_idx][0]
            
            self_values.append(self_global)
            target_values.append(target_global)
            
            log_fn(self_global, f"  Session {session_idx} self_global")
            log_fn(target_global, f"  Session {session_idx} target_global")
        
        self.sync_map = np.column_stack([np.concatenate(self_values), np.concatenate(target_values)])
        logger.debug(f"  sync_map: {self.sync_map.shape}")
        log_fn(self.sync_map[:, 0], f"  sync_map self")
        log_fn(self.sync_map[:, 1], f"  sync_map target")
        return self.sync_map
    
    def concatenate(self, output_path: Path, save_kwargs: dict, force: bool = False) -> si.BaseRecording:
        concat_dir = output_path / self.name / 'concat'
        bin_path = concat_dir / 'traces_cached_seg0.raw'
        
        if bin_path.exists() and not force:
            logger.info(f"{self.name}: Loading existing from {concat_dir}")
            self.concatenated = si.load(concat_dir)
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
        
        log_recording(self.concatenated, f"{self.name} concatenated")
        return self.concatenated
    
    def __repr__(self):
        return f"Probe('{self.name}', sessions={len(self.sessions)})"


def detect_reset(events: np.ndarray, limit: int = 500) -> int:
    """Detect chirp reset point from interval minima."""
    diffs = np.diff(events[:limit])
    minima = find_peaks(-diffs)[0]
    return minima[0] + 1 if len(minima) > 0 else 0


def align_edges(arr1: np.ndarray, arr2: np.ndarray, limit: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Align two event arrays by trimming to matching chirp phase."""
    reset1 = detect_reset(arr1, limit)
    reset2 = detect_reset(arr2, limit)
    
    if reset1 == reset2:
        return arr1, arr2
    if reset1 < reset2:
        return arr1, arr2[reset2 - reset1:]
    return arr1[reset1 - reset2:], arr2


def interpolate(source: np.ndarray, sync_map: np.ndarray) -> np.ndarray:
    """Interpolate source values to target reference using sync_map [N, 2]."""
    if len(sync_map) == 0:
        return np.array([])
    spl = make_interp_spline(sync_map[:, 0], sync_map[:, 1], k=1)
    return spl(source)