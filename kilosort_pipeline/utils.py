import sys
import re
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from loguru import logger
import spikeinterface.extractors as se

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


def validate_config(config):
    for k in ['recording_paths', 'remote_output', 'local_output', 'save_kwargs']:
        if k not in config:
            raise ValueError(f"Missing required configuration parameter: {k}")
    
    config.setdefault('session_name', 'default_session')
    config.setdefault('probe_filter', None)
    config.setdefault('target_fs', None)

    if not config['recording_paths']:
        raise ValueError("No recording paths specified in configuration.")
    
    # Validate recording paths
    for path in config['recording_paths']:
        if not Path(path).exists():
            raise FileNotFoundError(f"Recording path not found: {path}")

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
    """Log recording information."""
    n_channels  = rec.get_num_channels()
    duration    = rec.get_total_duration()
    fs          = rec.get_sampling_frequency()
    file_size   = rec.get_total_memory_size()
    logger.info(f"{name}:{n_channels} ch, {duration:.1f}s @ {fs/1000:.1f} kHz ({format_file_size(file_size)})")
    # Log filepath if available
    if hasattr(rec, '_kwargs') and 'folder_path' in rec._kwargs:
        filepath = rec._kwargs['folder_path']
        logger.debug(f"Filepath: {filepath}")

def parse_openephys_folders(recording_paths, probe_filter=None):
    """Parse OpenEphys folders and return Recording objects with timestamp paths.

    Parameters
    ----------
    recording_paths : list of str or Path
        Paths to OpenEphys session directories
    probe_filter : list of str, optional
        Filter to include only certain probe names (e.g., ['ProbeA', 'ProbeB'])

    Returns
    -------
    dict
        {'segments': {probe_name: [Recording1, Recording2, ...], ...},
         'timestamps': {
             stream_name: {
                 'event': [path1, path2, ...],
                 'cont': [path1, path2, ...]
             },
             ...
         }
        }
    """
    segments = defaultdict(list)
    timestamps = defaultdict(lambda: {'event': [], 'cont': []})

    logger.info("Parsing OpenEphys folders")

    for session_idx, rec_path in enumerate(recording_paths, 1):
        session_path = Path(rec_path).resolve()
        logger.debug(f"Session {session_idx}/{len(recording_paths)}: {session_path.name}")

        try:
            # Discover probes and ADC streams
            stream_names, stream_ids = se.get_neo_streams('openephysbinary', session_path)

            for stream_name, stream_id in zip(stream_names, stream_ids):
                # "OneBox-0.ProbeA" -> "ProbeA"
                if ".Probe" in stream_name:
                    clean_name = stream_name.split(".")[-1]
                elif "ADC" in stream_name:
                    clean_name = stream_name.split(".")[-1]  # "OneBox-ADC"
                else:
                    continue

                # Apply probe filter if specified
                if probe_filter and clean_name not in probe_filter:
                    continue

                # Load recordings as OpenEphysBinaryExtractor objects
                rec = se.read_openephys(session_path, stream_id=stream_id)
                segments[clean_name].append(rec)

                # Recursively search timestamp files
                event_ts = session_path.glob(f"**/events/{stream_name.split('#')[-1]}/TTL/timestamps.npy")
                cont_ts = session_path.glob(f"**/continuous/{stream_name.split('#')[-1]}/timestamps.npy")

                # Sort by recording number
                rec_n = lambda p: int(re.search(r'recording(\d+)', str(p)).group(1))
                event_ts = sorted(event_ts, key=rec_n)
                cont_ts = sorted(cont_ts, key=rec_n)

                # Extend timestamp lists
                timestamps[clean_name]['event'].extend([str(p) for p in event_ts])
                timestamps[clean_name]['cont'].extend([str(p) for p in cont_ts])

                logger.debug(f"  {clean_name}: {rec.get_num_segments()} segment(s), "
                           f"{len(event_ts)} event files, {len(cont_ts)} cont files")

        except Exception as e:
            logger.error(f"  Failed to parse session {session_path.name}: {e}")
            logger.exception("  Full traceback:")
            continue

    logger.success(f"Parsed {len(segments)} stream(s) across {len(recording_paths)} session(s)")

    return {
        'segments': dict(segments),
        'timestamps': dict(timestamps)
    }


def copy_to_remote(local_path, remote_path):
    """Copy session directory to remote storage."""
    logger.info("COPYING TO REMOTE STORAGE")
    logger.info(f"  From: {local_path}")
    logger.info(f"  To: {remote_path}")
    
    try:
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_path, remote_path, dirs_exist_ok=True)
        
        local_size = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file())
        logger.success(f"  Copied {format_file_size(local_size)} to remote")
        
    except Exception as e:
        logger.error(f"Failed to copy to remote server: {e}")
        raise