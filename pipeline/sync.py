import numpy as np
import pynapple as nap

from pathlib import Path
from collections import defaultdict
from typing import Optional
from .utils import log_timestamps, load_events
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
from loguru import logger


def detect_reset(event: np.ndarray, limit: int = 500) -> int:
    """
    Find chirp cycle reset point by detecting first local minimum in timestamp diffs.
    
    Returns 0 if no minimum found (chirp started at cycle beginning).
    """
    event_ts = np.diff(event[:limit])
    minima = find_peaks(-event_ts)[0]
    if len(minima) == 0:
        logger.warning("No reset point detected in chirp signal. Using index 0.")
        return 0
    reset_idx = minima[0] + 1
    return reset_idx


def match_chirp_edges(
    probe_ts: np.ndarray, 
    adc_ts: np.ndarray, 
    limit: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align probe and ADC chirp signals by trimming mismatched pre-reset edges.
    
    Returns (probe_ts, adc_ts) - either original or trimmed arrays.
    """
    # Detect reset points
    reset_idx1 = detect_reset(probe_ts, limit=limit)
    reset_idx2 = detect_reset(adc_ts, limit=limit)

    num1 = reset_idx1  # Number of edges before reset in ts1
    num2 = reset_idx2  # Number of edges before reset in ts2

    if num1 == num2:
        logger.debug("Chirp edges already aligned")
        return probe_ts, adc_ts
    
    if num1 < num2:
        # Trim adc_ts
        trim_size = num2 - num1
        adc_ts_matched = adc_ts[trim_size:]
        logger.info(f"  Trimmed ADC sync timestamps by {trim_size} samples to align edges")
        return probe_ts, adc_ts_matched
    else:
        # Trim probe_ts
        trim_size = num1 - num2
        probe_ts_matched = probe_ts[trim_size:]
        logger.info(f"  Trimmed Probe sync timestamps by {trim_size} samples to align edges")
        return probe_ts_matched, adc_ts

def get_kilosort_spikes(output_path, probe_filter=None):
    """
    Load Kilosort spike times from output directory.
    
    Returns memmap dictionary: {probe_name: spike_times}
    """
    spike_times_dict = {}

    kilosort_files = list(Path(output_path).glob('*/kilosort/spike_times.npy'))
    if not kilosort_files:
        logger.error("No Kilosort output found. Run Kilosort first.")
        raise FileNotFoundError(f"No spike_times.npy files in {output_path}")
    
    for spike_file in kilosort_files:
        probe_name = spike_file.parent.parent.name

        # Filter probes if requested
        if probe_filter and probe_name not in probe_filter:
            continue

        spike_times_dict[probe_name] = np.load(spike_file, mmap_mode='r')
        logger.info(f"Loaded {len(spike_times_dict[probe_name])} spikes from {probe_name}")

    return spike_times_dict

class Timestamps:
    """
    Manages global timestamps across multi-session recordings.
    
    Concatenates 'local' timestamps from individual sessions into a unified 'global'
    timeline by tracking cumulative offsets.

    Attributes
    ----------
    name : str
        Probe identifier (e.g., 'ProbeA', 'OneBox-ADC')
    fs : float
        Sampling frequency in Hz
    global_timestamps : list[np.ndarray]
        Per-session global timestamps (local_ts + cumulative_offset)
    intervals : list[tuple[float, float]]
        (start, end) times for each session in global time
    """
    
    def __init__(self, name: str, fs: float, t_start: float = 0.0):
        self.name = name
        self.fs = fs
        self.dt = 1 / fs
        
        self.global_timestamps = []
        self.intervals = []
        self.t_offset = t_start
        self.starting_states = []

    def update(self, local_ts: np.ndarray, t_end: float, starting_state: Optional[int] = None) -> None:
        """Add new session to global timeline."""
        # Update global timestamps
        self.global_timestamps.append(local_ts + self.t_offset + self.dt)

        # Update intervals
        self.intervals.append((self.t_offset, self.t_offset + t_end))
        
        # Update offset for next segment
        self.t_offset += t_end

        if starting_state:
            self.starting_states.append(starting_state)
        
    def __repr__(self):
        return f"Timestamps(name='{self.name}', @ {self.fs:.1f} Hz)"

def synchronize(
    output_path: Path, 
    probe_filter: list[str], 
    timestamps: dict[str, dict[str, list[str]]]
) -> None:
    """
    Synchronize probe spike times to ADC global timebase using TTL chirp (frequency sweep) signals.
    
    1. Builds global timestamps for ADC and each probe by concatenating multi-session
    recordings. 
    2. Match the edges if initial states differ by tracking first reset point.
    3. Interpolates Kilosort spike times from probe-timebase to ADC-timebase using linear splines.
    
    Saves per-probe outputs:
    - timestamps_map.npy: [N x 2] array of [probe_time, adc_time] pairs (Used for interpolation)
    - adc_spikes.npy: spike times interpolated to ADC timebase
    - timestamps.npy: continuous ADC timestamps (saved to OneBox-ADC/ directory)
    
    Parameters
    ----------
    output_path : Path
        Directory containing probe subdirectories with kilosort/ folders
    probe_filter : list[str]
        Probe names to synchronize. Must include 'OneBox-ADC' for synchronization.
    timestamps : dict[str, dict[str, list[str]]]
        Nested dict of {probe: {'event': [...], 'cont': [...]}} with paths to
        timestamps.npy files from OpenEphys sessions (from parse_timestamps())
    """
    logger.info("RUNNING SYNCHRONIZATION")
    
    logger.info("Computing global ADC timestamps")

    # Compute ADC global timestamps first
    ADC = Timestamps(name='OneBox-ADC', fs=30300.5, t_start=0.0)
    adc_event_paths = timestamps["OneBox-ADC"]['event']
    adc_cont_paths = timestamps["OneBox-ADC"]['cont']

    adc_timestamps_file = output_path / 'OneBox-ADC' / 'timestamps.npy'
    if adc_timestamps_file.exists():
        logger.info(f"ADC timestamps already exist at {adc_timestamps_file}.")
        adc_timestamps = None
    else:
        adc_timestamps = []
        logger.info("Will save ADC continuous timestamps")
    t_last = 0.0

    for idx, (event_path, cont_path) in enumerate(zip(adc_event_paths, adc_cont_paths)):
        event_ts, cont_ts, states = load_events(event_path, cont_path)
        # Subtract the offset of continuous recording
        ADC.update(event_ts - cont_ts[0], cont_ts[-1] - cont_ts[0], starting_state=states[0])

        ########## LOGGING #################################
        log_timestamps(event_ts, "ADC event")
        log_timestamps(cont_ts, "ADC cont")
        logger.info(f"ADC interval: {ADC.intervals[-1][0]:.4f} ... {ADC.intervals[-1][1]:.4f} s")
        log_timestamps(ADC.global_timestamps[-1], "ADC global segment")
        log_timestamps(np.concatenate(ADC.global_timestamps), "ADC global")
        logger.info("-"*60)
        ####################################################
        
        if adc_timestamps is not None:
            # Accumulate timestamps
            cont_ts = cont_ts - cont_ts[0] + t_last
            t_last += cont_ts[-1]
            adc_timestamps.append(cont_ts)

    # Save ADC recording timestamps
    if adc_timestamps is not None:
        adc_timestamps_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(adc_timestamps_file, np.concatenate(adc_timestamps))
        logger.success(f"Saved ADC timestamps to {adc_timestamps_file}")

    # Load Kilosort spike times
    logger.info("Loading Kilosort spike times")
    ks_spikes = get_kilosort_spikes(output_path=output_path, probe_filter=probe_filter)

    logger.info("Interpolating spikes to ADC global timebase")
    probe_timestamps = {k:d for k,d in timestamps.items() if k != "OneBox-ADC" and k in probe_filter}
    
    for probe, paths in probe_timestamps.items():
        PRB = Timestamps(name=probe, fs=30000.0, t_start=0.0)
        adc_global_timestamps = []
        logger.info(f"Processing probe: {probe}")

        logger.info("Extracting kilosort spikes")
        kilosort_spikes = ks_spikes[probe] / PRB.fs
        log_timestamps(kilosort_spikes, f"{probe} spikes")
        total_spikes_left = kilosort_spikes.size
        logger.info('='*60)

        save_dir = Path(output_path / probe)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to: {save_dir}")

        # masks           = []
        synced_spikes   = []

        for idx, (ev_path, cont_path) in enumerate(zip(paths['event'], paths['cont'])):
            event_ts, cont_ts, states = load_events(ev_path, cont_path)

            # Update probe timestamps, starting state is not needed here
            PRB.update(event_ts - cont_ts[0], cont_ts[-1] - cont_ts[0])

            probe_times = PRB.global_timestamps[-1]
            ########## LOGGING #################################
            log_timestamps(event_ts, f"Event timestamps")
            log_timestamps(cont_ts, f"Continuous timestamps")
            log_timestamps(probe_times, f"Global segment")
            log_timestamps(np.concatenate(PRB.global_timestamps), f"Global")
            ########## LOGGING #################################

            adc_times = ADC.global_timestamps[idx]
            
            # Align both arrays using reset detection
            # Find reset indices for both probe and ADC to ensure same physical event at index 0
            idx_probe = detect_reset(probe_times)
            idx_adc = detect_reset(adc_times)
            
            logger.info(f"Detected reset indices - Probe: {idx_probe}, ADC: {idx_adc}")
            
            # Slice both arrays from their respective reset indices
            probe_times_aligned = probe_times[idx_probe:]
            adc_times_aligned = adc_times[idx_adc:]
            
            # Check if aligned arrays are empty
            if len(probe_times_aligned) == 0 or len(adc_times_aligned) == 0:
                logger.error(f"Aligned arrays are empty after reset detection. Skipping segment {idx}.")
                logger.error(f"  Probe aligned length: {len(probe_times_aligned)}, ADC aligned length: {len(adc_times_aligned)}")
                continue
            
            # Extract spikes based on aligned start time
            # Use the first aligned timestamp as the start, and continuous end as the end
            cont_start_aligned = probe_times_aligned[0]
            cont_end = PRB.intervals[idx][1]
            logger.info(f"Extracting spikes in aligned interval: {cont_start_aligned:.5f} ... {cont_end:.5f} s")
            spike_mask = (kilosort_spikes >= cont_start_aligned) & (kilosort_spikes <= cont_end)
            # masks.append(spike_mask) # DEBUG purpose
            
            probe_spikes = kilosort_spikes[spike_mask]
            log_timestamps(probe_spikes, f"Extracted spikes")
            
            # Truncate aligned arrays to minimum common length
            min_length = min(len(probe_times_aligned), len(adc_times_aligned))
            if min_length < len(adc_times_aligned):
                logger.warning(f"  Truncating aligned ADC timestamps. ADC timestamps: {len(adc_times_aligned)} -> {min_length}.")
                adc_times_aligned = adc_times_aligned[:min_length]
            elif min_length < len(probe_times_aligned):
                logger.warning(f"  Truncating aligned Probe timestamps. Probe timestamps: {len(probe_times_aligned)} -> {min_length}.")
                probe_times_aligned = probe_times_aligned[:min_length]
            
            # Adjust end boundary for spike extraction after truncation
            cont_end_aligned = probe_times_aligned[-1]
            if cont_end_aligned < cont_end:
                logger.info(f"Adjusting end boundary from {cont_end:.5f} to {cont_end_aligned:.5f} due to alignment")
                # Re-filter spikes to exclude those beyond the aligned end time
                aligned_mask = (probe_spikes >= cont_start_aligned) & (probe_spikes <= cont_end_aligned)
                probe_spikes = probe_spikes[aligned_mask]
                log_timestamps(probe_spikes, f"Re-filtered spikes")
            
            # Store aligned timestamps for output
            PRB.global_timestamps[idx] = probe_times_aligned
            adc_global_timestamps.append(adc_times_aligned)

            # Interpolate/extrapolate to ADC time using aligned arrays
            spl = make_interp_spline(x=probe_times_aligned, y=adc_times_aligned, k=1)
            adc_spikes = spl(probe_spikes)
            synced_spikes.append(adc_spikes)
            total_spikes_left -= adc_spikes.size

            log_timestamps(adc_spikes, "ADC interpolated spikes")
            logger.info(f"Synced spikes: {adc_spikes.size}/{kilosort_spikes.size}. Remaining spikes: {total_spikes_left}")
            logger.info("-"*60)
        
        # Create and save timestamps map Probe Global timestamps <-> ADC Global timestamps
        probe_times = np.concatenate(PRB.global_timestamps)
        adc_times = np.concatenate(adc_global_timestamps)
        timestamps_map = np.vstack((probe_times, adc_times)).T

        np.save(save_dir / "timestamps_map.npy", timestamps_map)
        np.save(save_dir / "adc_spikes.npy", np.concatenate(synced_spikes))
        
        # np.save(save_dir / "masks.npy", np.concatenate(masks))
        # np.save(save_dir / "intervals.npy", PRB.intervals)
        
        logger.success(f"Completed synchronization for probe: {probe}")
    logger.success("SYNCHRONIZATION COMPLETED")
