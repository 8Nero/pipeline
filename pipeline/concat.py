import spikeinterface as si
import spikeinterface.extractors as se
from pathlib import Path
from loguru import logger

from .utils import log_recording

def downsample_eeg(eeg_folder, rec, target_fs, chunk_duration=60):
    """Downsample recording to target_fs and save as binary."""
    eeg_file = eeg_folder / 'eeg_data.dat'
    if eeg_file.exists():
        logger.info(f"EEG data already exists, skipping downsampling")
        return
    
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
            decimated = chunk_data[::decimation_factor, :].astype('int16')
            decimated.tofile(f)
            
            start = end
            # logger.debug(f"Processed {end}/{total_samples} samples")
    logger.success(f"EEG data saved: {output_file}")

def concat(rec_paths, probe_filter, output_path, save_kwargs, target_fs=None):
    """Load and concatenate recordings across sessions. Optional EEG downsampling."""
    logger.info("CONCATENATING RECORDINGS")
    probe_recs = {prb: [] for prb in probe_filter}
    # # Check if concatenated probe data exists
    # for probe_idx, probe in enumerate(probe_recs.keys(), 1):
    #     probe_path = output_path / probe / 'concat'
    #     bin_path = probe_path / 'traces_cached_seg0.raw'
    #     if bin_path.exists():
    #         logger.info(f"Found concatenated data at {bin_path}.")
    #         logger.info(f"Skipping concatenation for {probe}.")
    #         saved_rec = si.load_extractor(bin_path.parent)
    #         probe_recs[probe].append(saved_rec)
    #         log_recording(saved_rec)
    #         continue

    # Load recordings for each session, group them by probe
    for session_idx, session_path in enumerate(rec_paths, 1):
        logger.info(f"Loading session {session_idx}/{len(rec_paths)}: {session_path}")
        
        # Discover probes and ADC streams
        stream_names, stream_ids = se.get_neo_streams('openephysbinary', session_path)
        for stream_name, stream_id in zip(stream_names, stream_ids):
            # Extract probe name (e.g., "OneBox-0.ProbeA" -> "ProbeA")
            probe = stream_name.split(".")[-1]

            # Skip if: SYNC channel, not in filter
            if "SYNC" in stream_name or probe not in probe_filter:
                continue

            # Load recordings as OpenEphysBinaryExtractor objects
            rec = se.read_openephys(session_path, stream_id=stream_id)
            probe_recs[probe].append(rec)
        
        logger.success("Session loaded.")

    probe_concat = {}
    # Concatenate, downsample, and save
    for probe_idx, (probe, recs) in enumerate(probe_recs.items(), 1):
        logger.info(f"[{probe_idx}/{len(probe_recs)}]")

        # Check if concatenated data already exists
        probe_dir = output_path / probe
        concat_dir = probe_dir / 'concat'
        bin_path = concat_dir / 'traces_cached_seg0.raw'
        
        if bin_path.exists():
            logger.info(f"Found concatenated {probe} data at {bin_path}")
            logger.info(f"Skipping concatenation for {probe}")
            saved_rec = si.load(concat_dir)
            if probe != 'OneBox-ADC':
                probe_concat[probe] = saved_rec
            continue

        logger.info(f"Concatenating {len(recs)} session(s)...")
        concat_rec = si.concatenate_recordings(recs) if len(recs) > 1 else recs[0]
        log_recording(concat_rec)
        logger.info(f"Sessions concatenated: {len(recs)}")
        
        concat_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to: {concat_dir}")
        saved_rec = concat_rec.save(
            folder=concat_dir,
            **save_kwargs
        )
        logger.success(f"Saved concatenated recording")

        if probe == 'OneBox-ADC':
            continue  # Skip EEG downsampling for ADC

        probe_concat[probe] = saved_rec
        if target_fs:
            downsample_eeg(probe_dir, rec=saved_rec, target_fs=target_fs)

    logger.success(f"Successfully concatenated {len(probe_concat)}/{len(probe_recs)} probe(s)")
    logger.success(f"CONCATENATION FINISHED.")
    return probe_concat