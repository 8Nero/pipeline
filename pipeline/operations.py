"""
Pipeline operations: loading, spike sorting, and synchronization.
"""
import os
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
from pathlib import Path
from loguru import logger
from typing import Optional

try:
    from kilosort import run_kilosort, DEFAULT_SETTINGS
    KILOSORT_AVAILABLE = True
except ImportError:
    KILOSORT_AVAILABLE = False
    DEFAULT_SETTINGS = {}
    run_kilosort = None

from .probe import Probe, interpolate
from .decimation import DecimatedRecording
from .utils import log_recording

def load_probes(session_paths: list[str], probe_names: list[str]) -> dict[str, Probe]:
    """Load probes from session paths."""
    logger.info("LOADING PROBES")
    logger.info(f"  Probes: {probe_names}")
    logger.info(f"  Sessions: {len(session_paths)}")
    
    probes = {}
    for name in probe_names:
        fs = 30300.5 if name == 'OneBox-ADC' else 30000.0
        probe = Probe(name=name, fs=fs)
        probe.load_from_sessions(session_paths)
        probes[name] = probe
    
    logger.info(f"Loaded {len(probes)} probes")
    return probes


def concatenate_probes(
    probes: dict[str, Probe],
    output_path: Path,
    save_kwargs: dict,
    target_fs: Optional[int] = None
) -> dict[str, Probe]:
    """Concatenate all probe recordings and optionally downsample EEG."""
    logger.info("=" * 60)
    logger.info("CONCATENATING RECORDINGS")
    
    neural_probes = {}
    
    for idx, (name, probe) in enumerate(probes.items(), 1):
        logger.info(f"[{idx}/{len(probes)}] {name}")
        
        probe.concatenate(output_path, save_kwargs)
        
        if name == 'OneBox-ADC':
            continue
        
        neural_probes[name] = probe
        
        if target_fs and probe.concatenated is not None:
            downsample_eeg(output_path / name, probe.concatenated, target_fs=target_fs, **save_kwargs)
    
    logger.info(f"Concatenated {len(neural_probes)} neural probes")
    logger.info("=" * 60)
    return neural_probes


def downsample_eeg(
    output_folder: Path,
    rec: si.BaseRecording,
    target_fs: int = 1250,
    verbose: bool = True,
    overwrite: bool = False,
    **job_kwargs
) -> Optional[si.BaseRecording]:
    eeg_file = output_folder / 'eeg.dat'
    
    if eeg_file.exists():
        if overwrite:
            logger.info(f"  {eeg_file} file exists, deleting")
            os.remove(eeg_file)
        else:
            logger.info(f"  {eeg_file} file exists, loading")
            return si.load(eeg_file)
    
    fs = rec.get_sampling_frequency()
    decimation_factor = int(fs / target_fs)
    actual_fs = fs / decimation_factor
    
    logger.info(f"  Downsampling: {fs:.0f}Hz -> {actual_fs:.0f}Hz (factor={decimation_factor})")
    dec_rec = DecimatedRecording(rec, decimation_factor)
    log_recording(dec_rec, "  EEG")
    
    si.write_binary_recording(
        recording=dec_rec,
        file_paths=eeg_file,
        add_file_extension=False,
        verbose=verbose,
        **job_kwargs)
    
    return dec_rec


def probe_to_kilosort(probe) -> dict:
    return {
        'chanMap': np.arange(probe.get_contact_count(), dtype=int),
        'xc': probe.contact_positions[:, 0].astype('float32'),
        'yc': probe.contact_positions[:, 1].astype('float32'),
        'kcoords': probe.shank_ids.astype('float32'),
        'n_chan': probe.get_contact_count(),
    }


def run_kilosort4(
    probes: dict[str, Probe],
    output_path: Path,
    device: str = 'cuda',
    custom_settings: Optional[dict] = None
) -> dict[str, np.ndarray]:
    """Run Kilosort4 on concatenated probe recordings."""
    if not KILOSORT_AVAILABLE:
        logger.error("Kilosort4 is not installed. Cannot run spike sorting.")
        logger.error("Please install using `uv pip install pipeline[full]` or manually install kilosort and torch.")
        raise ImportError("Kilosort4 not installed")

    logger.info("=" * 60)
    logger.info("RUNNING KILOSORT4")
    
    spike_times = {}
    
    for idx, (name, probe) in enumerate(probes.items(), 1):
        logger.info(f"[{idx}/{len(probes)}] {name}")
        
        if probe.concatenated is None:
            logger.warning(f"  No concatenated recording, skipping")
            continue
        
        kilosort_dir = output_path / name / 'kilosort'
        spike_file = kilosort_dir / 'spike_times.npy'
        
        if spike_file.exists():
            spike_times[name] = np.load(spike_file, mmap_mode='r')
            logger.info(f"  Loaded existing: {len(spike_times[name])} spikes")
            continue
        
        kilosort_dir.mkdir(parents=True, exist_ok=True)
        
        rec = probe.concatenated
        prb = rec.get_probe()
        probe_dict = probe_to_kilosort(prb)
        
        binary_file = output_path / name / 'concat' / 'traces_cached_seg0.raw'
        
        settings = DEFAULT_SETTINGS.copy()
        if custom_settings:
            settings.update(custom_settings)
        settings['n_chan_bin'] = rec.get_num_channels()
        settings['fs'] = rec.get_sampling_frequency()
        
        logger.info(f"  Input: {binary_file}")
        logger.info(f"  Output: {kilosort_dir}")
        
        try:
            _ = run_kilosort(
                settings=settings,
                probe=probe_dict,
                data_dtype='int16',
                filename=str(binary_file),
                results_dir=str(kilosort_dir),
                device=device,
                verbose_console=True
            )
            spike_times[name] = np.load(spike_file, mmap_mode='r')
            logger.info(f"  Sorted: {len(spike_times[name])} spikes")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue
    
    logger.info("=" * 60)
    return spike_times


def save_adc_references(adc: Probe, output_path: Path) -> None:
    """Save ADC sample-to-timestamp mapping."""
    logger.info("SAVING ADC REFERENCES")
    
    adc.build_global_references(mode='samples')
    adc.build_global_references(mode='timestamps')
    
    adc_global_samples = adc.get_global_events('samples')
    adc_global_timestamps = adc.get_global_events('timestamps')
    
    adc_dir = output_path / adc.name
    adc_dir.mkdir(parents=True, exist_ok=True)
    np.save(adc_dir / "global_samples.npy", adc_global_samples)
    np.save(adc_dir / "global_timestamps.npy", adc_global_timestamps)
    logger.info(f"  Saved ADC references to {adc_dir}")


def synchronize_probes(
    probes: dict[str, Probe],
    target: str,
    spike_times: dict[str, np.ndarray],
    output_path: Path
) -> dict[str, np.ndarray]:
    """Synchronize probe spikes to target (ADC) reference frame."""
    logger.info("=" * 60)
    logger.info("SYNCHRONIZING")
    
    target_probe = probes.get(target)
    if target_probe is None:
        raise ValueError(f"Target probe '{target}' not found in probes dictionary.")
    
    target_probe.build_global_references(mode='samples')
    synced_spikes = {}
    
    for name, probe in probes.items():
        if name == target:
            continue
        
        if name not in spike_times:
            logger.warning(f"{name}: No spikes, skipping")
            continue
        
        logger.info(f"{name}:")
        
        probe.build_global_references(mode='samples')
        sync_map = probe.sync_to(target_probe, mode='samples')
        
        ks_spike_samples = spike_times[name].flatten()
        logger.info(f"  Input: {len(ks_spike_samples)} spikes, samples [{ks_spike_samples[0]} - {ks_spike_samples[-1]}]")
        
        # Probe samples -> ADC samples -> ADC timestamps
        adc_spike_samples = interpolate(ks_spike_samples, sync_map)
        
        # Check if adc_spike_samples are sorted
        unsorted_mask = adc_spike_samples[:-1] > adc_spike_samples[1:]
        if np.any(unsorted_mask):
            unsorted_indices = np.where(unsorted_mask)[0]
            logger.warning(f"  adc_spike_samples not sorted at {len(unsorted_indices)} indices: {unsorted_indices[:10]}{'...' if len(unsorted_indices) > 10 else ''}")
        
        adc_spike_times = adc_spike_samples / target_probe.fs
        logger.info(f"  Output: [{adc_spike_times[0]:.3f} - {adc_spike_times[-1]:.3f}]s")
        
        save_dir = output_path / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(save_dir / "sync_map.npy", sync_map)
        np.save(save_dir / "adc_spike_samples.npy", adc_spike_samples)
        np.save(save_dir / "adc_spike_times.npy", adc_spike_times)
        
        synced_spikes[name] = adc_spike_times
        logger.info(f"  Saved to {save_dir}")

    logger.info("SYNCHRONIZATION COMPLETED")
    logger.info("=" * 60)
    return synced_spikes