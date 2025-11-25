"""
Utility functions for pipeline configuration and file handling.
"""
import sys
import re
import yaml
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Literal


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds/3600:.2f}h"
    elif seconds >= 60:
        return f"{seconds/60:.2f}min"
    return f"{seconds:.1f}s"


def format_size(size_bytes: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"


def log_recording(rec, name: str = "Recording") -> None:
    """Log recording: channels, duration, sampling rate, size."""
    n_ch = rec.get_num_channels()
    duration = rec.get_total_duration()
    fs = rec.get_sampling_frequency()
    size = rec.get_total_memory_size()
    logger.info(f"  {name}: {n_ch}ch, {format_duration(duration)} @ {fs/1000:.1f}kHz, {format_size(size)}")


def log_timestamps(ts: np.ndarray, name: str = "Timestamps") -> None:
    """Log timestamp array: range, count, duration."""
    if len(ts) == 0:
        logger.info(f"  {name}: empty")
        return
    duration = ts[-1] - ts[0]
    logger.info(f"  {name}: [{ts[0]:.3f} - {ts[-1]:.3f}]s, {len(ts)} events, {format_duration(duration)}")


def log_samples(samples: np.ndarray, name: str = "Samples") -> None:
    """Log sample array: range, count."""
    if len(samples) == 0:
        logger.info(f"  {name}: empty")
        return
    logger.info(f"  {name}: [{samples[0]} - {samples[-1]}], {len(samples)} events")


def log_intervals(intervals: list, name: str = "Intervals") -> None:
    """Log intervals: count and total duration."""
    if not intervals:
        logger.info(f"  {name}: none")
        return
    total = sum(end - start for start, end in intervals)
    logger.info(f"  {name}: {len(intervals)} segments, {format_duration(total)} total")


def setup_logger(debug: bool = False, log_path: Path = None):
    logger.remove()
    
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG" if debug else "INFO",
        colorize=True
    )
    
    if log_path:
        logger.add(
            log_path,
            rotation="500 MB",
            retention="10 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: dict) -> dict:
    required = ['recording_paths', 'remote_output', 'local_output', 'save_kwargs']
    for k in required:
        if k not in config:
            raise ValueError(f"Missing config parameter: {k}")
    
    config.setdefault('session_name', 'default_session')
    config.setdefault('probe_filter', None)
    config.setdefault('target_fs', None)
    
    if not config['recording_paths']:
        raise ValueError("No recording paths specified")
    
    for path in config['recording_paths']:
        if not Path(path).exists():
            raise FileNotFoundError(f"Recording not found: {path}")
    
    if not Path(config['local_output']).exists():
        raise FileNotFoundError(f"Local output not found: {config['local_output']}")
    
    return config


def normalize_probe_filter(probe_filter: list[str] | None, session_paths: list[str]) -> list[str]:
    if probe_filter is None:
        available = set()
        for path in session_paths:
            for ts_file in Path(path).glob('**/timestamps.npy'):
                ts_str = str(ts_file)
                for keyword in ['ProbeA', 'ProbeB', 'ProbeC', 'ProbeD', 'OneBox-ADC']:
                    if keyword in ts_str:
                        available.add(keyword)
        probe_filter = sorted(list(available))
        logger.info(f"Auto-detected probes: {probe_filter}")
        return probe_filter
    
    normalized = []
    for probe in probe_filter:
        if probe.upper() in ['ADC', 'ONEBOX-ADC']:
            normalized.append('OneBox-ADC')
        else:
            normalized.append(probe)
    
    return list(dict.fromkeys(normalized))


def parse_timestamps(session_paths: list[str], probe_filter: list[str]) -> dict[str, dict[str, list[str]]]:
    timestamps = {probe: {'event': [], 'cont': []} for probe in probe_filter}
    
    def extract_rec_num(p):
        match = re.search(r'recording(\d+)', str(p))
        return int(match.group(1)) if match else 0
    
    for session_path in session_paths:
        ts_files = sorted(Path(session_path).glob('**/timestamps.npy'), key=extract_rec_num)
        
        for ts_file in ts_files:
            ts_str = str(ts_file)
            for probe in probe_filter:
                if probe in ts_str:
                    if 'events' in ts_str:
                        timestamps[probe]['event'].append(ts_str)
                    elif 'continuous' in ts_str:
                        timestamps[probe]['cont'].append(ts_str)
    
    return timestamps


def setup_pipeline(config_path: str, debug: bool = False) -> dict:
    setup_logger(debug=debug)
    
    config = validate_config(load_config(config_path))
    
    session_name = config['session_name'].strip().replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    remote = Path(config['remote_output']).resolve() / session_name
    local = Path(config['local_output']).resolve() / session_name
    local.mkdir(parents=True, exist_ok=True)
    
    config['session_name'] = session_name
    config['remote_output'] = remote
    config['local_output'] = local
    
    log_path = local / f'{session_name}_{timestamp}.log'
    setup_logger(debug=debug, log_path=log_path)
    
    logger.info("=" * 60)
    logger.info("PIPELINE INITIALIZED")
    logger.info(f"  Session: {session_name}")
    logger.info(f"  Input: {len(config['recording_paths'])} recording sessions")
    logger.info(f"  Local: {local}")
    logger.info(f"  Remote: {remote}")
    logger.info("=" * 60)
    
    return config


def copy_to_remote(
    local_path: Path,
    remote_path: Path,
    overwrite_mode: Literal['prompt', 'all', 'skip-all'] = 'prompt'
) -> None:
    logger.info("COPYING TO REMOTE")
    logger.info(f"  From: {local_path}")
    logger.info(f"  To: {remote_path}")
    
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    
    copied_size = 0
    copied_count = 0
    skipped = 0
    
    for local_file in local_path.rglob('*'):
        if not local_file.is_file():
            continue
        
        relative = local_file.relative_to(local_path)
        remote_file = remote_path / relative
        
        if remote_file.exists():
            if overwrite_mode == 'skip-all':
                skipped += 1
                continue
            elif overwrite_mode == 'prompt':
                response = input(f"Overwrite '{relative}'? (y/n/all/skip-all): ").strip().lower()
                if response == 'all':
                    overwrite_mode = 'all'
                elif response == 'skip-all':
                    overwrite_mode = 'skip-all'
                    skipped += 1
                    continue
                elif response != 'y':
                    skipped += 1
                    continue
        
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, remote_file)
        copied_size += local_file.stat().st_size
        copied_count += 1
    
    logger.info(f"  Copied: {copied_count} files, {format_size(copied_size)}")
    if skipped > 0:
        logger.info(f"  Skipped: {skipped} files")