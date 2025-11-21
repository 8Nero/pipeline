import sys
import re
import yaml
import shutil
import numpy as np

from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Literal

# Pre-compile regex pattern for better performance
_RECORDING_NUM_PATTERN = re.compile(r'recording(\d+)')

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string (hours, minutes, or seconds)."""
    if seconds >= 3600:
        return f"{seconds/3600:.2f} h"
    elif seconds >= 60:
        return f"{seconds/60:.2f} min"
    else:
        return f"{seconds:.1f} s"

def validate_probe_filter(
    probe_filter: list[str] | None, 
    rec_paths: list[str]
) -> list[str]:
    """Convert ADC variants to OneBox-ADC and auto-detect probes if probe_filter is None."""
    if probe_filter is None:
        # Auto-detect available probes from timestamp files
        available_probes = set()
        probe_keywords = {'ProbeA', 'ProbeB', 'ProbeC', 'ProbeD', 'OneBox-ADC'}
        for session_path in rec_paths:
            ts_files = Path(session_path).glob('**/timestamps.npy')
            for ts_file in ts_files:
                ts_str = str(ts_file)
                # Extract probe names from paths - exit early when found
                for keyword in probe_keywords:
                    if keyword in ts_str:
                        available_probes.add(keyword)
                        break  # Each file typically only matches one probe
        probe_filter = sorted(available_probes)
        logger.info(f"Found probes: {probe_filter}")
        return probe_filter
    
    # Normalize ADC variants to OneBox-ADC and remove duplicates in one pass
    seen = set()
    normalized = []
    for probe in probe_filter:
        probe_upper = probe.upper()
        normalized_probe = 'OneBox-ADC' if probe_upper in ('ADC', 'ONEBOX-ADC') else probe
        if normalized_probe not in seen:
            seen.add(normalized_probe)
            normalized.append(normalized_probe)
    
    return normalized

def setup_logger(debug: bool = False):
    """Setup console logger with INFO or DEBUG level."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG" if debug else "INFO",
        colorize=True
    )

def load_config(config_path):
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)        
        logger.success(f"Loaded configuration from: {config_path}")
        return config
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

def load_events(events_path, cont_path):
    event_ts  = np.load(events_path, mmap_mode='r')
    cont_ts   = np.load(cont_path, mmap_mode='r')
    states    = np.load(events_path.replace('timestamps.npy', 'states.npy'), mmap_mode='r')
    return event_ts, cont_ts, states

def validate_config(config):
    for k in ['recording_paths', 'remote_output', 'local_output', 'save_kwargs']:
        if k not in config:
            raise ValueError(f"Missing required configuration parameter: {k}")
    
    config.setdefault('session_name', 'default_session')
    config.setdefault('probe_filter', None)
    config.setdefault('target_fs', None)

    if not config['recording_paths']:
        raise ValueError("No recording paths specified in configuration.")
    
    # Define keywords that shouldn't appear in parent folder paths
    # These keywords are used for stream/probe identification
    reserved_keywords = ['ADC', 'Adc', 'ProbeA', 'ProbeB', 'ProbeC', 'ProbeD']
    
    # Validate recording paths
    for path in config['recording_paths']:
        if not Path(path).exists():
            raise FileNotFoundError(f"Recording path not found: {path}")
        
        # Check parent folders for reserved keywords
        path_obj = Path(path).resolve()
        parent_parts = path_obj.parts
        
        for keyword in reserved_keywords:
            for part in parent_parts:
                if keyword in part:
                    raise ValueError(
                        f"Recording path contains reserved keyword '{keyword}' in parent folder: {path}\n")

    if not Path(config['local_output']).exists():
        raise FileNotFoundError(f"Local output directory not found: {config['local_output']}")

    if not Path(config['remote_output']).exists():
        raise FileNotFoundError(f"Remote output directory not found: {config['remote_output']}")

    return config

def setup(config_path: str = 'config.yaml', debug: bool = False) -> dict:
    """
    Initialize pipeline: load config, validate paths, setup logging.

    Creates session directories and log file at {local_output}/{session_name}/.
    """
    # Load and validate config
    setup_logger(debug=debug)
    protocol = validate_config(load_config(config_path))
    
    # Extract config parameters
    session_name = protocol['session_name']
    session_name = session_name.strip().replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    protocol['session_name'] = session_name
    
    # Setup output paths
    remote_output = Path(protocol['remote_output']).resolve()
    local_output = Path(protocol['local_output']).resolve()

    protocol['remote_output'] = remote_output / session_name
    protocol['local_output'] = local_output / session_name

    # Setup logging file in session folder root
    protocol['local_output'].mkdir(parents=True, exist_ok=True)
    log_path = protocol['local_output'] / f'{session_name}_{timestamp}.log'

    logger.add(
        log_path,
        rotation="500 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    logger.success(f"Logger configured at {log_path}")

    logger.info(f"Pipeline configuration:")
    logger.info(f"Session: {session_name}")
    logger.info(f"Recording sessions: {len(protocol['recording_paths'])}")
    logger.info(f"Local output: {protocol['local_output']}")
    logger.info(f"Remote output: {protocol['remote_output']}")
    logger.debug(f"  Parallel jobs: {protocol['save_kwargs']['n_jobs']}")
    logger.debug(f"  EEG downsampling frequency: {protocol['target_fs']}")
    return protocol

def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def log_recording(rec, name="Recording"):
    """Log SpikeInterface Recording objects."""
    n_channels  = rec.get_num_channels()
    duration    = rec.get_total_duration()
    fs          = rec.get_sampling_frequency()
    file_size   = rec.get_total_memory_size()
    dtype       = rec.get_dtype()
    duration_str = format_duration(duration)
    logger.info(f"{name}: {n_channels} ch, {duration:.1f}s ({duration_str}) @ {fs/1000:.1f} kHz {dtype} ({format_file_size(file_size)})")
    # Log filepath if available
    if hasattr(rec, '_kwargs') and 'folder_path' in rec._kwargs:
        filepath = rec._kwargs['folder_path']
        logger.debug(f"Filepath: {filepath}")

def log_timestamps(timestamps, name):
    """Log timestamps basic information."""
    start_ts = timestamps[0]
    end_ts = timestamps[-1]
    logger.info(f"{name}: {start_ts:.4f} ... {end_ts:.4f} s ({timestamps.size} samples)")

def parse_timestamps(rec_paths: list[str], probe_filter: list[str]) -> dict[str, dict[str, list[str]]]:
    """
    Extract OpenEphys timestamp file paths from multi-session recordings, grouped by probe and type (event/continuous).
    
    Sorts files in each session by recording number.
    
    Returns Nested dict: {probe: {'event': [paths...], 'cont': [paths...]}}
    """
    timestamps = {probe: {'event': [], 'cont': []} for probe in probe_filter}
    # Pre-compile string patterns for faster matching
    probe_filter_set = set(probe_filter)

    for session_idx, session_path in enumerate(rec_paths, 1):
        session_path = Path(session_path)
        logger.debug(f"Session {session_idx}/{len(rec_paths)}: {session_path.name}") 
        # Recursively get all timestamps.npy files
        ts_files = list(session_path.glob('**/timestamps.npy'))
        # Sort by recording number extracted from filename using pre-compiled pattern
        def extract_recording_num(p):
            match = _RECORDING_NUM_PATTERN.search(str(p))
            return int(match.group(1)) if match else 0
        ts_files = sorted(ts_files, key=extract_recording_num)
        for ts_file in ts_files:
            ts_file_str = str(ts_file)
            # Check event/continuous type once
            is_event = 'events' in ts_file_str
            is_cont = 'continuous' in ts_file_str
            if not (is_event or is_cont):
                continue
            
            # Match probe and add to appropriate list
            for probe in probe_filter_set:
                if probe in ts_file_str:
                    if is_event:
                        timestamps[probe]['event'].append(ts_file_str)
                    elif is_cont:
                        timestamps[probe]['cont'].append(ts_file_str)
                    break  # Each file only belongs to one probe
    return timestamps

def copy_to_remote(
    local_path: Path, 
    remote_path: Path, 
    overwrite_mode: Literal['prompt', 'all', 'skip-all'] = 'prompt'
) -> None:
    """
    Recursively copy session directory to remote/network storage.
    
    Parameters
    ----------
    local_path : Path
        Source directory
    remote_path : Path
        Destination directory
    overwrite_mode : {'prompt', 'all', 'skip-all'}, default='prompt'
        File conflict resolution options
    """
    logger.info("COPYING TO REMOTE STORAGE")
    logger.info(f"  From: {local_path}")
    logger.info(f"  To: {remote_path}")
    
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    
    copied_size = 0
    skipped_count = 0
    
    if overwrite_mode == 'all':
        logger.info("Overwrite mode: Overwriting all existing files")
    elif overwrite_mode == 'skip-all':
        logger.info("Overwrite mode: Skipping all existing files")
    
    # Pre-collect all files to copy for better performance
    files_to_process = [f for f in local_path.rglob('*') if f.is_file()]
    
    for local_file in files_to_process:
        relative_path = local_file.relative_to(local_path)
        remote_file = remote_path / relative_path
        
        if remote_file.exists():
            if overwrite_mode == 'prompt':
                logger.warning(f"File already exists: {relative_path}")
                response = input(f"Overwrite '{relative_path}'? (y/n/all/skip-all) [default: n]: ").strip().lower()
                if response == 'all':
                    overwrite_mode = 'all'
                    logger.info("Overwriting all existing files")
                elif response == 'skip-all':
                    overwrite_mode = 'skip-all'
                    logger.info("Skipping all existing files")
                    skipped_count += 1
                    continue
                elif response != 'y':
                    logger.info(f"Skipping: {relative_path}")
                    skipped_count += 1
                    continue
            elif overwrite_mode == 'skip-all':
                skipped_count += 1
                continue
        
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, remote_file)
        copied_size += local_file.stat().st_size
    
    logger.success(f"  Copied {format_file_size(copied_size)} to remote")
    if skipped_count > 0:
        logger.debug(f"  Skipped {skipped_count} file(s)")