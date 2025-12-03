#!/usr/bin/env python3
"""
Example usage of the pipeline.
"""
import numpy as np
from pathlib import Path
from loguru import logger

from pipeline import (
    Probe,
    interpolate,
    setup_pipeline,
    DecimatedRecording
)
from pipeline.utils import setup_logger


# --- Single probe workflow ---

def example_single_probe():
    setup_logger(debug=True)
    
    session_paths = ["/path/to/session1", "/path/to/session2"]
    
    probe = Probe(name='ProbeA', fs=30000.0)
    probe.load_from_sessions(session_paths)
    
    output_path = Path("./output")
    probe.concatenate(output_path, save_kwargs={'n_jobs': 4, 'chunk_duration': '1s'})
    
    return probe


# --- ADC sample to timestamp conversion ---

def example_adc_conversion():
    setup_logger(debug=True)
    
    session_paths = ["/path/to/session1", "/path/to/session2"]
    
    adc = Probe(name='OneBox-ADC', fs=30300.5)
    adc.load_from_sessions(session_paths)
    
    # Build both sample and timestamp references
    adc.build_global_references(mode='samples')
    global_samples = adc.get_global_events()
    
    adc.build_global_references(mode='timestamps')
    global_timestamps = adc.get_global_events()
    
    # For any ADC sample, convert to timestamp by dividing by fs
    camera_samples = np.array([1000, 2000, 3000, 4000])
    camera_timestamps = camera_samples / adc.fs
    
    logger.info(f"Camera frame times: {camera_timestamps}")
    return adc


# --- Probe to ADC sync workflow ---

def example_sync():
    setup_logger(debug=True)
    
    session_paths = ["/path/to/session1", "/path/to/session2"]
    
    probe = Probe(name='ProbeA', fs=30000.0)
    probe.load_from_sessions(session_paths)
    
    adc = Probe(name='OneBox-ADC', fs=30300.5)
    adc.load_from_sessions(session_paths)
    
    probe.build_global_references(mode='samples')
    adc.build_global_references(mode='samples')
    
    probe.sync_to(adc)
    
    # Convert spikes: probe samples -> ADC samples -> timestamps
    spike_samples = np.array([10000, 20000, 30000])
    adc_samples = interpolate(spike_samples, probe.sync_map)
    adc_times = adc_samples / adc.fs
    
    logger.info(f"Probe samples: {spike_samples}")
    logger.info(f"ADC samples: {adc_samples}")
    logger.info(f"ADC times: {adc_times}")


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
    probes = load_probes(config['recording_paths'], probe_names)
    neural_probes = concatenate_probes(probes, config['local_output'], config['save_kwargs'])
    
    probe_a = probes['ProbeA']
    logger.info(f"ProbeA: {probe_a}")


if __name__ == '__main__':
    pass