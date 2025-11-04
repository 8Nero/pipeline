import os
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['NUM_THREADS'] = '16'

from pathlib import Path
import numpy as np
from typing import Optional
import spikeinterface as si

from kilosort import run_kilosort, DEFAULT_SETTINGS
from loguru import logger

from .utils import log_recording


def probe_to_kilosort(probe):
    """Convert SpikeInterface probe to Kilosort probe dictionary."""
    return {
        'chanMap': np.arange(probe.get_contact_count(), dtype=int),
        'xc': probe.contact_positions[:, 0].astype('float32'),
        'yc': probe.contact_positions[:, 1].astype('float32'),
        'kcoords': probe.shank_ids.astype('float32'),
        'n_chan': probe.get_contact_count(),
    }


def run_kilosort4(
    probe_concat: dict[str, si.BaseRecording], 
    output_path: Path, 
    device: str = 'cuda', 
    custom_settings: Optional[dict] = None
) -> None:
    """
    Run Kilosort4 spike sorting on concatenated probe recordings.
    """
    logger.info("RUNNING KILOSORT4")

    for probe_idx, (probe_name, rec) in enumerate(probe_concat.items(), 1):
        logger.info(f"[{probe_idx}/{len(probe_concat)}] Spike sorting {probe_name}")

        try:
            probe_folder    = output_path / probe_name
            concat_folder   = probe_folder / "concat"
            kilosort_output = probe_folder / "kilosort"
            kilosort_output.mkdir(parents=True, exist_ok=True)

            # Check if Kilosort output already exists
            spike_times_file = kilosort_output / 'spike_times.npy'
            if spike_times_file.exists():
                logger.info(f"Kilosort output already exists at {spike_times_file}")
                logger.success(f"Skipped {probe_name}")
                continue

            # Extract probe geometry
            prb = rec.get_probe()
            probe_dict = probe_to_kilosort(prb)

            # Validate binary file exists
            binary_file = concat_folder / 'traces_cached_seg0.raw'
            if not binary_file.exists():
                raise FileNotFoundError(f"Binary file not found: {binary_file}")

            # Log recording info
            log_recording(rec)
            logger.info(f"Output directory: {kilosort_output}")
            logger.info("Starting Kilosort4...")

            # Configure Kilosort4 settings
            settings = DEFAULT_SETTINGS.copy()
            if custom_settings:
                settings.update(custom_settings)
            
            settings['n_chan_bin'] = rec.get_num_channels()
            settings['fs']         = rec.get_sampling_frequency()
            # Run Kilosort4
            _ = run_kilosort(
                settings=settings,
                probe=probe_dict,
                data_dtype='int16',
                filename=str(binary_file),
                results_dir=str(kilosort_output),
                device=device,
                verbose_console=True
            )
            
            logger.success(f"Kilosort4 completed for {probe_name}")
        
        except Exception as e:
            logger.error(f" :( Kilosort4 failed for {probe_name}: {e}")
            logger.exception("Full traceback:")
            continue

    logger.success("Spike sorting completed")