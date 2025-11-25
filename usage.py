#!/usr/bin/env python3
"""
Example usage of the pipeline for experimentation.
"""
import numpy as np
from pathlib import Path
from loguru import logger

from pipeline import (
    Probe, ADC,
    interpolate_to_target,
    samples_to_timestamps,
    setup_pipeline,
    parse_timestamps,
    DecimatedRecording
)
from pipeline.utils import setup_logger


# --- Single probe workflow ---

def example_single_probe():
    setup_logger(debug=True)
    
    session_paths = ["/path/to/session1", "/path/to/session2"]
    
    probe = Probe(name='ProbeA', fs=30000.0)
    
    import spikeinterface.extractors as se
    for session_path in session_paths:
        stream_names, stream_ids = se.get_neo_streams('openephysbinary', session_path)
        for name, sid in zip(stream_names, stream_ids):
            if 'ProbeA' in name:
                probe.load_session(session_path, sid)
    
    output_path = Path("./output")
    probe.concatenate(output_path, save_kwargs={'n_jobs': 4, 'chunk_duration': '1s'})
    
    return probe


# --- ADC sample to timestamp conversion ---

def example_camera_ttl():
    setup_logger(debug=True)
    
    adc = ADC(name='OneBox-ADC', fs=30300.5)
    
    # Load timestamps (assumes files exist)
    event_paths = [
        "/path/to/session1/events/OneBox-ADC/timestamps.npy",
        "/path/to/session2/events/OneBox-ADC/timestamps.npy",
    ]
    cont_paths = [
        "/path/to/session1/continuous/OneBox-ADC/timestamps.npy",
        "/path/to/session2/continuous/OneBox-ADC/timestamps.npy",
    ]
    
    for ev, cont in zip(event_paths, cont_paths):
        adc.load_timestamps(ev, cont)
    
    adc.build_global_timestamps()
    
    # Convert camera TTL sample indices to timestamps
    camera_sample_indices = np.array([1000, 2000, 3000, 4000])
    
    camera_timestamps = samples_to_timestamps(
        camera_sample_indices,
        adc.get_global_samples(),
        adc.get_global_timestamps()
    )
    
    logger.info(f"Camera frame times: {camera_timestamps}")
    return adc, camera_timestamps


# --- Manual sync workflow ---

def example_manual_sync():
    setup_logger(debug=True)
    
    probe = Probe(name='ProbeA', fs=30000.0)
    adc = ADC(name='OneBox-ADC', fs=30300.5)
    
    # ... load timestamps for both ...
    
    probe.build_global_timestamps()
    adc.build_global_timestamps()
    
    # Create mapping from probe timeline to ADC timeline
    probe.sync_to(adc)
    
    # Use the map for spike interpolation
    spike_times_probe = np.array([10.0, 20.0, 30.0])
    spike_times_adc = interpolate_to_target(spike_times_probe, probe.timestamps_map)
    
    logger.info(f"Probe times: {spike_times_probe}")
    logger.info(f"ADC times: {spike_times_adc}")


# --- EEG downsampling ---

def example_downsampling():
    import spikeinterface as si
    
    concat_rec = si.load("/path/to/ProbeA/concat")
    
    dec_rec = DecimatedRecording(concat_rec, decimation_factor=30)
    
    logger.info(f"Original: {concat_rec.get_sampling_frequency():.0f} Hz")
    logger.info(f"Decimated: {dec_rec.get_sampling_frequency():.0f} Hz")
    
    dec_rec.save(
        format='binary',
        folder=Path("/path/to/output/lfp"),
        n_jobs=16,
        chunk_duration='2s'
    )


# --- Quick config-based workflow ---

def example_quick_script():
    from pipeline import load_probes, concatenate_probes
    
    config = setup_pipeline('config.yaml')
    
    probe_names = ['ProbeA', 'OneBox-ADC']
    timestamp_map = parse_timestamps(config['recording_paths'], probe_names)
    
    probes = load_probes(config['recording_paths'], probe_names, timestamp_map)
    neural_probes = concatenate_probes(probes, config['local_output'], config['save_kwargs'])
    
    probe_a = probes['ProbeA']
    logger.info(f"ProbeA: {probe_a}")


if __name__ == '__main__':
    pass