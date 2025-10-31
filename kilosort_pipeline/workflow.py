from pathlib import Path
from loguru import logger

from kilosort_pipeline.sync import run_synchronization

from .concat import concat
from .sorting import run_kilosort4
from .utils import copy_to_remote, parse_openephys_folders


def run_full_pipeline(protocol):
    """Execute complete pipeline: load → concat → save → sort → sync → copy to remote.
    
    Parameters
    ----------
    protocol : dict
        Configuration dict from utils.setup() with keys:
        - recording_paths: list of Path to OpenEphys sessions
        - output_path: Path to session output directory
        - save_kwargs: dict for recording.save()
        - target_fs: float or None for EEG downsampling
        - probe_filter: list of str or None
        - local_output: Path to local session directory
        - base_output: Path to remote session directory
        
    Returns
    -------
    dict[str, Recording]
        Probe name -> saved recording object
    """
    logger.info("STARTING PIPELINE")

    # Parse OpenEphys folders to get recordings and timestamps
    parsed = parse_openephys_folders(
        recording_paths=protocol['recording_paths'],
        probe_filter=protocol['probe_filter']
    )

    probe_recordings = parsed['segments']

    if not probe_recordings:
        logger.error("No recordings loaded. Pipeline aborted.")
        raise RuntimeError("No recordings loaded.")

    probe_concat = concat(
        probe_recordings=probe_recordings,
        output_path=protocol['local_output'],
        save_kwargs=protocol['save_kwargs'],
        target_fs=protocol['target_fs']
    )

    if not probe_concat:
        logger.error("No recordings saved. Pipeline aborted.")
        raise RuntimeError("No recordings saved.")

    run_kilosort4(
        probe_concat=probe_concat,
        output_path=protocol['local_output']
    )

    run_synchronization(protocol, timestamps=parsed['timestamps'])
    
    logger.info(f"Copying to: {protocol['base_output']}")
    copy_to_remote(
        local_path=protocol['local_output'],
        remote_path=protocol['base_output']
    )
    logger.success(f"Copying completed")
    logger.success("PIPELINE COMPLETED SUCCESSFULLY")
    
    return probe_concat