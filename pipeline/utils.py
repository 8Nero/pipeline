import sys
import re
import yaml
import shutil
import numpy as np

from pathlib import Path
from datetime import datetime
from loguru import logger

def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
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

def setup(config_path='config.yaml'):
    # Load and validate config
    setup_logger()
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
    """Log recordings."""
    n_channels  = rec.get_num_channels()
    duration    = rec.get_total_duration()
    fs          = rec.get_sampling_frequency()
    file_size   = rec.get_total_memory_size()
    dtype       = rec.get_dtype()
    logger.info(f"{name}:{n_channels} ch, {duration:.1f}s @ {fs/1000:.1f} kHz {dtype} ({format_file_size(file_size)})")
    # Log filepath if available
    if hasattr(rec, '_kwargs') and 'folder_path' in rec._kwargs:
        filepath = rec._kwargs['folder_path']
        logger.debug(f"Filepath: {filepath}")

def log_timestamps(timestamps, name):
    """Log timestamps."""
    start_ts = timestamps[0]
    end_ts = timestamps[-1]
    logger.info(f"{name}: {start_ts:.4f} ... {end_ts:.4f} s ({timestamps.size} samples)")

def parse_timestamps(rec_paths, probe_filter):
    """Recursively parse timestamps.npy files from recording paths."""
    timestamps = {probe: {'event': [], 'cont': []} for probe in probe_filter}

    for session_idx, session_path in enumerate(rec_paths, 1):
        session_path = Path(session_path)
        logger.debug(f"Session {session_idx}/{len(rec_paths)}: {session_path.name}") 
        # Recursively get all timestamps.npy files
        ts_files = list(session_path.glob('**/timestamps.npy'))
        # Sort by recording number extracted from filename
        ts_files = sorted(ts_files, key=lambda p: int(re.search(r'recording(\d+)', str(p)).group(1)))
        for ts_file in ts_files:
            ts_file = str(ts_file)
            for probe in probe_filter:
                if probe in ts_file:
                    if 'events' in ts_file:
                        timestamps[probe]['event'].append(ts_file)
                    elif 'continuous' in ts_file:
                        timestamps[probe]['cont'].append(ts_file)
    return timestamps

def copy_to_remote(local_path, remote_path, overwrite_mode='prompt'):
    """Copy session directory to remote storage."""
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
    
    for local_file in local_path.rglob('*'):
        if local_file.is_file():
            relative_path = local_file.relative_to(local_path)
            remote_file = remote_path / relative_path
            
            if remote_file.exists():
                if overwrite_mode == 'prompt':
                    logger.warning(f"File already exists: {relative_path}")
                    response = input(f"Overwrite '{relative_path}'? (y/n/all/skip-all): ").strip().lower()
                    
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