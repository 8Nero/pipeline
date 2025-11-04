import numpy as np
import pynapple as nap

from pathlib import Path
from collections import defaultdict
from .utils import log_timestamps, load_events
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
from loguru import logger


def detect_reset(event, limit=500):
    """Detect chirp signal reset point by finding first local minimum in timestamp differences.
    
    Parameters
    ----------
    event : np.ndarray
        Event timestamps (e.g., TTL pulse times)
    limit : int, optional
        Number of initial samples to search (default: 500)
        
    Returns
    -------
    int
        Index of reset point (first sample after reset)
    """
    event_ts = np.diff(event[:limit])
    minima = find_peaks(-event_ts)[0]
    if len(minima) == 0:
        logger.warning("No reset point detected in chirp signal. Using index 0.")
        return 0
    reset_idx = minima[0] + 1
    return reset_idx


def match_chirp_edges(chirp_ts1, chirp_ts2, limit=200):
    """Align two chirp signals based on their reset points.
    
    Parameters
    ----------
    chirp_ts1, chirp_ts2 : np.ndarray
        Chirp signal timestamps to align
    limit : int, optional
        Number of samples to search for reset (default: 200)
        
    Returns
    -------
    tuple of np.ndarray
        (aligned_ts1, aligned_ts2) - Trimmed arrays with matching edge counts
        
    Notes
    -----
    Trims the signal with more edges before reset to match the other.
    This handles cases where acquisition started mid-chirp cycle.
    """
    # Detect reset points
    reset_idx1 = detect_reset(chirp_ts1, limit=limit)
    reset_idx2 = detect_reset(chirp_ts2, limit=limit)

    num1 = reset_idx1  # Number of edges before reset in ts1
    num2 = reset_idx2  # Number of edges before reset in ts2

    if num1 == num2:
        logger.debug("Chirp edges already aligned")
        return chirp_ts1, chirp_ts2
    
    if num1 < num2:
        # Trim chirp_ts2
        trim_size = num2 - num1
        chirp_ts2_matched = chirp_ts2[trim_size:]
        logger.info(f"  Trimmed chirp_ts2 by {trim_size} samples to align edges")
        return chirp_ts1, chirp_ts2_matched
    else:
        # Trim chirp_ts1
        trim_size = num1 - num2
        chirp_ts1_matched = chirp_ts1[trim_size:]
        logger.info(f"  Trimmed chirp_ts1 by {trim_size} samples to align edges")
        return chirp_ts1_matched, chirp_ts2

def get_kilosort_spikes(output_path, probe_filter=None):
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
    def __init__(self, name, fs, t_start=0.0):
        self.name = name
        self.fs = fs
        self.dt = 1 / fs
        
        self.global_timestamps = []
        self.intervals = []
        self.t_offset = t_start
        self.starting_states = []

    def update(self, local_ts, t_end, starting_state=None):
        """ Update global timestamps with a new segment."""
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

def synchronize(output_path, probe_filter, timestamps):
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
        np.save(adc_timestamps_file, np.concatenate(adc_timestamps))

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

            # Handle state mismatches
            if states[0] != ADC.starting_states[idx]:
                logger.warning(f"State mismatch between {probe} and ADC")
                logger.info("Matching edges")
                event_ts, _ = match_chirp_edges(event_ts, ADC.global_timestamps[idx])

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
            
            # Extract spikes based on continuous range
            cont_start, cont_end = PRB.intervals[idx]
            logger.info(f"Extracting spikes in interval: {cont_start:.5f} ... {cont_end:.5f} s")
            mask = (kilosort_spikes > cont_start) & (kilosort_spikes <= cont_end)
            # masks.append(mask) # DEBUG purpose
            
            probe_spikes = kilosort_spikes[mask]
            log_timestamps(probe_spikes, f"Extracted spikes")
            
            # Handle length mismatches
            min_length = min(len(probe_times), len(adc_times))
            if min_length < len(adc_times):
                logger.warning(f"  Truncating ADC timestamps. ADC timestamps: {len(adc_times)} -> {min_length}.")
                adc_times = adc_times[:min_length]
            elif min_length < len(probe_times):
                logger.warning(f"  Truncating Probe timestamps. Probe timestamps: {len(probe_times)} -> {min_length}.")
                PRB.global_timestamps[idx] = probe_times[:min_length]
            
            adc_global_timestamps.append(adc_times)

            # Interpolate/extrapolate to ADC time
            spl = make_interp_spline(x=probe_times, y=adc_times, k=1)
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
