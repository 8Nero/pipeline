import spikeinterface as si
from loguru import logger

from .utils import format_file_size, log_recording

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

def save_adc(adc_folder, rec, save_kwargs):
    

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
                logger.info(f"Sessions concatenated: {len(rec_list)}")
                
                concat_folder.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving to: {concat_folder}")
                saved_rec = concat_rec.save(
                    folder=concat_folder,
                    **save_kwargs
                )
                logger.success(f"Saved concatenated recording")
            
            if "ADC" in probe_name:
                continue

            # EEG downsampling
            if target_fs:
                downsample_eeg(probe_folder, rec=saved_rec, target_fs=target_fs)

            probe_concat[probe_name] = saved_rec

        except Exception as e:
            logger.error(f"Failed to concatenate {probe_name}: {e}")
            logger.exception("Full traceback:")
            continue

    logger.success(f"Successfully processed {len(probe_concat)}/{len(probe_recordings)} probe(s)")
    return probe_concat