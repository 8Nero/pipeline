import sys
import yaml
import shutil
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


def validate_config(config):
    for k in ['recording_paths', 'base_output', 'local_output', 'save_kwargs']:
        if k not in config:
            raise ValueError(f"Missing required configuration key: {k}")
    
    config.setdefault('session_name', 'default_session')
    config.setdefault('probe_filter', None)
    config.setdefault('target_fs', None)

    if not config['recording_paths']:
        raise ValueError("No recording paths specified in configuration.")
    
    # Validate paths exist
    for path in config['recording_paths']:
        if not Path(path).exists():
            raise FileNotFoundError(f"Recording path not found: {path}")
        
    if not Path(config['local_output']).parent.exists():
        raise FileNotFoundError(f"Local output directory not found: {config['local_output']}")

    if not Path(config['base_output']).parent.exists():
        raise FileNotFoundError(f"Base output directory not found: {config['base_output']}")

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
    base_output = Path(protocol['base_output']).resolve()
    local_output = Path(protocol['local_output']).resolve()

    protocol['base_output'] = base_output / session_name
    protocol['local_output'] = local_output / session_name

    # Configure logging file in session folder root
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
    logger.info(f"Remote output: {protocol['base_output']}")
    logger.debug(f"  Parallel jobs: {protocol['save_kwargs']['n_jobs']}")
    logger.debug(f"  Target FS: {protocol['target_fs']}")
    return protocol

def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


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
        logger.error(f"Failed to copy to remote storage: {e}")
        raise