import glob
import re
import numpy as np
import pynapple as nap

from pathlib import Path
from collections import defaultdict
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
from loguru import logger

def parse_sessions(sessions):
    """Parse OpenEphys folders grouped by session and recording.
    
    Parameters
    ----------
    sessions : list of str or Path
        Paths to OpenEphys session directories
        
    Returns
    -------
    dict
        Nested dict: {session_idx: {recording_num: {probe_name: {'events': path, 'continuous': path}}}}
        
    Notes
    -----
    - session_idx is 1-indexed (session 1, 2, 3, ...)
    - Searches for OneBox-* folders with timestamps.npy files
    - Extracts probe names from stream names (e.g., 'OneBox-0.ProbeA' -> 'ProbeA')
    """
    # Regex pattern to extract: recording number, data type (events/continuous), probe name
    pattern = r'recording(\d+)/(events|continuous)/(OneBox-\d+\.[\w-]+)'
    parsed = {}
    
    for session_idx, session_path in enumerate(sessions):
        session_path = Path(session_path)
        # Get all OneBox folders (events and continuous)
        onebox_paths = glob.glob('**/OneBox**/**/timestamps.npy', root_dir=session_path, recursive=True)
        
        logger.debug(f"Session {session_idx + 1}: Found {len(onebox_paths)} timestamp files in {session_path.name}")
        
        for path in onebox_paths:
            path = Path(path).as_posix() # \\ -> /
            match = re.search(pattern, path)
            if match:
                recording_num = int(match.group(1))
                data_type = match.group(2)                      # 'events' or 'continuous'
                probe_name = match.group(3).split('.')[1]       # select 'ProbeA' from 'OneBox-0.ProbeA'
                
                # Initialize nested dicts on first access
                session_key = session_idx + 1
                parsed.setdefault(session_key, {}).setdefault(recording_num, {}).setdefault(probe_name, {})[data_type] = path
    
    return parsed


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


def compute_global_timestamps(recording_paths, parsed_probes, fs=30000.0):
    """Compute global timestamps for all probes across concatenated sessions.
    
    Parameters
    ----------
    recording_paths : list of Path
        Ordered list of session directories (concatenation order)
    parsed_probes : dict
        Output from parse_sessions()
    fs : float, optional
        Sampling frequency in Hz (default: 30000.0)
        
    Returns
    -------
    dict
        {probe_name: [global_timestamps_session1, global_timestamps_session2, ...]}
        
    Notes
    -----
    - Aligns probe events to ADC reference using chirp signal matching
    - Handles temporal concatenation by tracking cumulative time offset
    - Checks initial state matching between probe and ADC events
    """
    global_events = defaultdict(list)
    dt = 1 / fs
    
    # Initialize last timestamp tracker for each probe
    first_session_probes = next(iter(parsed_probes.values()))
    first_recording_probes = next(iter(first_session_probes.values()))
    last_ts = {probe: 0.0 for probe in first_recording_probes.keys()}

    logger.info("Computing global timestamps")
    
    for session_idx, session in parsed_probes.items():
        logger.info(f"Session {session_idx}")
        session_path = Path(recording_paths[session_idx - 1])
        
        for rec_idx, rec in session.items():
            logger.info(f"  Recording {rec_idx}:")
            adc = rec['OneBox-ADC']
            
            for probe in sorted(rec):
                paths = rec[probe]
                logger.info(f"    Probe: {probe}")

                # Load probe timestamps (from TTL folder)
                event = np.load(session_path / paths['events'], mmap_mode='r')
                cont = np.load(session_path / paths['continuous'], mmap_mode='r')

                if "ADC" not in probe:
                    # Check first edges
                    event_state = np.load(session_path / paths['events'].replace('timestamps.npy', 'states.npy'), mmap_mode='r')
                    adc_state = np.load(session_path / adc['events'].replace('timestamps.npy', 'states.npy'), mmap_mode='r')

                    if event_state[0] != adc_state[0]:
                        logger.info(f"      Initial states differ (probe={event_state[0]}, ADC={adc_state[0]})")
                        logger.info("      Aligning to ADC reference...")

                        # Load ADC timestamps for matching
                        adc_events = np.load(session_path / adc['events'], mmap_mode='r')

                        # Match ADC and probe timestamps
                        event, _ = match_chirp_edges(event, adc_events)
                    else:
                        logger.debug(f"      Initial states match (state={event_state[0]})")
            
                # Compute global timestamps
                global_ts = event - cont[0] + last_ts[probe]

                logger.debug(f"      Event range: {event[0]} ... {event[-1]}")
                logger.debug(f"      Continuous range: {cont[0]} ... {cont[-1]}")
                logger.info(f"      Global timestamps: {global_ts[0]:.2f} ... {global_ts[-1]:.2f} s")

                # Update last_ts for next session
                last_ts[probe] += cont[-1] - cont[0] + dt

                # Store global events
                global_events[probe].append(global_ts)

    return dict(global_events)


def sync_spikes_to_adc(spike_times_dict, global_events, fs=30000.0, output_path=None):
    """Synchronize Kilosort spike times to ADC reference frame.
    
    Parameters
    ----------
    spike_times_dict : dict
        {probe_name: spike_times_array} from Kilosort output
    global_events : dict
        {probe_name: [timestamps_per_session]} from compute_global_timestamps()
    fs : float, optional
        Sampling frequency in Hz (default: 30000.0)
    output_path : Path, optional
        If provided, saves synced spikes to {probe}/sync/spike_times_synced.npy
        
    Returns
    -------
    dict
        {probe_name: synced_spike_times} aligned to ADC reference
        
    """
    logger.info("Synchronizing spike times to ADC reference")
    
    # Extract ADC reference timestamps
    adc_events = global_events.pop('OneBox-ADC')
    synced_spikes = {}
    
    for probe_name, spike_times in spike_times_dict.items():
        logger.info(f"Processing {probe_name}")
        
        if probe_name not in global_events:
            logger.warning(f"  No global timestamps found for {probe_name}. Skipping.")
            continue
        
        # Convert spike times to seconds
        spike_times_sec = nap.Ts(t=spike_times / fs, time_units='s')
        adc_spikes = []
        
        probe_events = global_events[probe_name]
        for session_idx, (probe_times, adc_times) in enumerate(zip(probe_events, adc_events), 1):
            # Define overlapping epochs
            pr_epoch = nap.IntervalSet(start=probe_times[0], end=probe_times[-1])
            ad_epoch = nap.IntervalSet(start=adc_times[0], end=adc_times[-1])
            overlap = pr_epoch.intersect(ad_epoch)
            
            # Restrict spikes to overlap
            pr_spikes = spike_times_sec.restrict(overlap)
            
            logger.debug(f"  Session {session_idx}: {len(pr_spikes)} spikes in overlap region")
            
            # Linear interpolation to ADC time base
            spl = make_interp_spline(probe_times, adc_times, k=1)
            adc_spikes.append(spl(pr_spikes.to_numpy()))
        
        # Concatenate all sessions
        synced_spikes[probe_name] = np.concatenate(adc_spikes)
        logger.success(f"  Synced {len(synced_spikes[probe_name])} spikes for {probe_name}")
        
        # Save if output path provided
        if output_path:
            sync_dir = output_path / probe_name / "sync"
            sync_dir.mkdir(parents=True, exist_ok=True)
            sync_file = sync_dir / "spike_times_synced.npy"
            np.save(sync_file, synced_spikes[probe_name])
            logger.info(f"  Saved to: {sync_file}")
    
    return synced_spikes


def run_synchronization(protocol):
    """Run full synchronization workflow after Kilosort.
    
    Parameters
    ----------
    protocol : dict
        Pipeline configuration with keys:
        - recording_paths: list of session paths
        - output_path: Path to output directory with Kilosort results
        - probe_filter: optional list of probes to process
        
    Returns
    -------
    dict
        {probe_name: synced_spike_times} aligned to ADC reference
        
    Notes
    -----
    Expects Kilosort output structure:
        {output_path}/{probe_name}/kilosort/spike_times.npy
    
    Creates sync outputs:
        {output_path}/{probe_name}/sync/spike_times_synced.npy
    """
    logger.info("RUNNING SYNCHRONIZATION")
    
    # Parse OpenEphys sessions
    parsed_probes = parse_sessions(protocol['recording_paths'])
    
    # Compute global timestamps
    global_events = compute_global_timestamps(
        recording_paths=protocol['recording_paths'],
        parsed_probes=parsed_probes
    )
    
    # Load Kilosort spike times
    spike_times_dict = {}
    output_path = protocol['local_output']
    
    kilosort_files = list(output_path.glob('*/kilosort/spike_times.npy'))
    if not kilosort_files:
        logger.error("No Kilosort output found. Run Kilosort first.")
        raise FileNotFoundError(f"No spike_times.npy files in {output_path}")
    
    for spike_file in kilosort_files:
        probe_name = spike_file.parent.parent.name
        
        # Filter probes if requested
        if protocol.get('probe_filter') and probe_name not in protocol['probe_filter']:
            continue
        
        spike_times_dict[probe_name] = np.load(spike_file)
        logger.info(f"Loaded {len(spike_times_dict[probe_name])} spikes from {probe_name}")
    
    # Synchronize to ADC
    _ = sync_spikes_to_adc(
        spike_times_dict=spike_times_dict,
        global_events=global_events,
        output_path=output_path
    )
    
    logger.success("SYNCHRONIZATION COMPLETED")
