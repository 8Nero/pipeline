from pathlib import Path
from collections import defaultdict

import spikeinterface as si
import spikeinterface.extractors as se
from loguru import logger

from .utils import format_file_size


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


def load_sessions(rec_paths, probe_filter=None, load_sync=False):
    """Load OpenEphys recording sessions grouped by probe. Concatenates multi-recordings within a session.

    Returns
    -------
    dict[str, list[Recording]]
        Probe name -> list of recordings (one per session)
    """
    logger.info("LOADING RECORDING SESSIONS")

    probe_recordings = defaultdict(list)
    total_probes = set()
    
    for i, rec_path in enumerate(rec_paths, 1):
        rec_dir = Path(rec_path)
        logger.info(f"Session {i}/{len(rec_paths)}: {rec_dir.name}")
        
        try:
            stream_names, stream_ids = se.get_neo_streams('openephysbinary', rec_dir)
            for stream_name, stream_id in zip(stream_names, stream_ids):
                if ".Probe" not in stream_name:
                    continue
                    
                probe_name = stream_name.split(".")[-1]
                total_probes.add(probe_name)

                if probe_filter and probe_name not in probe_filter:
                    logger.info(f"  {probe_name}: SKIPPED (filtered)")
                    continue
                
                rec = se.read_openephys(rec_dir, stream_id=stream_id, load_sync_channel=load_sync)
                
                # Handle multiple recordings 
                num_segments = rec.get_num_segments()
                if num_segments > 1:
                    logger.info(f"  {probe_name}: {num_segments} segments found, concatenating...")
                    rec = si.concatenate_recordings([rec])
                    logger.success(f"  {probe_name}: Concatenated segments successfully")
                
                probe_recordings[probe_name].append(rec)
                log_recording(rec)
                
        except Exception as e:
            logger.error(f"  :( Failed to load session: {e}")
            logger.exception("  Full traceback:")
            continue
    
    # Summary
    logger.success("FINISHED LOADING SESSIONS")
    logger.info(f"Probes loaded: {sorted(probe_recordings.keys())} {len(probe_recordings)}/{len(total_probes)}")

    return dict(probe_recordings)


def downsample_eeg(eeg_folder, rec, target_fs, chunk_duration=60):
    """Downsample recording to target_fs and save as binary."""
    eeg_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downsampling EEG to: {eeg_folder}")
    
    fs = rec.get_sampling_frequency()
    decimation_factor = int(fs / target_fs)
    
    logger.info(f"Decimation factor: {decimation_factor}")
    logger.info(f"Output rate: {fs/decimation_factor:.2f} Hz")

    chunk_samples = int(chunk_duration * fs)
    total_samples = rec.get_num_frames()
    output_file = eeg_folder / 'eeg_data.dat'
    
    with open(output_file, 'wb') as f:
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk_data = rec.get_traces(start_frame=start, end_frame=end)
            
            # Pick every Nth sample
            decimated = chunk_data[:, ::decimation_factor].astype('int16')
            decimated.tofile(f)
            
            start = end
            # logger.debug(f"Processed {end}/{total_samples} samples")
    logger.success(f"EEG data saved: {output_file}")


def concat(probe_recordings, output_path, save_kwargs, target_fs=None):
    """Concatenate recordings across sessions and save as binary. Optional EEG downsampling."""
    logger.info("CONCATENATING RECORDINGS")
    probe_concat = {}
    
    for probe_idx, (probe_name, rec_list) in enumerate(probe_recordings.items(), 1):
        logger.info(f"[{probe_idx}/{len(probe_recordings)}] Processing {probe_name}")
        
        try:
            probe_folder = output_path / probe_name
            concat_folder = probe_folder / "concat"
            
            # Check if concatenation already exists
            binary_file = concat_folder / 'traces_cached_seg0.raw'
            if binary_file.exists():
                logger.info(f"Concatenation already exists, loading from disk...")

                saved_rec = si.load_extractor(concat_folder)
                log_recording(saved_rec)

                logger.success(f"Loaded existing concatenated recording")
            else:
                # Perform concatenation across sessions
                logger.info(f"Concatenating {len(rec_list)} session(s)...")
                concat_rec = si.concatenate_recordings(rec_list) if len(rec_list) > 1 else rec_list[0]
                
                log_recording(concat_rec)
                logger.info(f"Data type: {concat_rec.get_dtype()}")
                logger.info(f"Sessions concatenated: {len(rec_list)}")
                
                concat_folder.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving to: {concat_folder}")
                saved_rec = concat_rec.save(
                    folder=concat_folder,
                    **save_kwargs
                )
                logger.success(f"Saved concatenated recording")
            
            # Optional EEG downsampling (independent of main concat)
            if target_fs:
                eeg_folder = probe_folder / "eeg"
                eeg_file = eeg_folder / 'eeg_data.bin'
                
                if eeg_file.exists():
                    logger.info(f"EEG data already exists, skipping downsampling")
                else:
                    downsample_eeg(eeg_folder, rec=saved_rec, target_fs=target_fs)

            probe_concat[probe_name] = saved_rec

        except Exception as e:
            logger.error(f"Failed to concatenate {probe_name}: {e}")
            logger.exception("Full traceback:")
            continue

    logger.success(f"Successfully processed {len(probe_concat)}/{len(probe_recordings)} probe(s)")
    return probe_concat