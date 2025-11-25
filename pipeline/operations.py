"""
Pipeline operations: loading, spike sorting, and synchronization.
"""
import numpy as np
import re
import spikeinterface as si
import spikeinterface.extractors as se
from pathlib import Path
from loguru import logger
from typing import Optional

from kilosort import run_kilosort, DEFAULT_SETTINGS

from .probe import Probe, ADC, interpolate_to_target
from .decimation import DecimatedRecording
from .utils import log_recording, log_timestamps, format_duration, format_size


def load_probes(
    session_paths: list[str],
    probe_names: list[str],
    timestamp_map: dict[str, dict[str, list[str]]]
) -> dict[str, Probe]:
    """Load probe recordings and timestamps from multiple sessions."""
    logger.info("LOADING PROBES")
    logger.info(f"  Probes: {probe_names}")
    logger.info(f"  Sessions: {len(session_paths)}")
    
    probes = {}
    for name in probe_names:
        if name == 'OneBox-ADC':
            probes[name] = ADC(name=name)
        else:
            probes[name] = Probe(name=name)
    
    # 1. Load Recordings (per session)
    for session_idx, session_path in enumerate(session_paths):
        session_name = Path(session_path).name
        logger.info(f"Session {session_idx + 1}/{len(session_paths)}: {session_name}")
        
        stream_names, stream_ids = se.get_neo_streams('openephysbinary', session_path)
        
        for stream_name, stream_id in zip(stream_names, stream_ids):
            probe_name = stream_name.split(".")[-1]
            
            if "SYNC" in stream_name or probe_name not in probe_names:
                continue
            
            rec = probes[probe_name].load_session(session_path, stream_id)
            log_recording(rec, f"  {probe_name}")
            
    # 2. Load Timestamps (per recording segment)
    logger.info("LOADING TIMESTAMPS")
    for name, probe in probes.items():
        event_paths = timestamp_map[name]['event']
        cont_paths = timestamp_map[name]['cont']
        
        logger.info(f"{name}: Loading {len(event_paths)} timestamp segments")
        
        for i, (ep, cp) in enumerate(zip(event_paths, cont_paths)):
            # Extract debug info
            ep_path = Path(ep)
            rec_match = re.search(r'recording(\d+)', str(ep_path))
            rec_num = rec_match.group(1) if rec_match else "?"
            
            # Find session name from path
            session_name = "Unknown"
            for sp in session_paths:
                if sp in str(ep_path):
                    session_name = Path(sp).name
                    break
            
            ts = probe.load_timestamps(ep, cp)
            
            logger.info(f"  Segment {i+1} [{session_name}/rec{rec_num}]:")
            log_timestamps(ts.cont_ts, "    Continuous")
            log_timestamps(ts.event_ts, "    Events")
    
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
            downsample_eeg(output_path / name, probe.concatenated, target_fs=target_fs)
    
    logger.info(f"Concatenated {len(neural_probes)} neural probes")
    logger.info("=" * 60)
    return neural_probes


def downsample_eeg(
    probe_folder: Path,
    rec: si.BaseRecording,
    target_fs: int = 1250,
    n_jobs: int = 8,
    chunk_duration: str = '1s'
) -> Optional[si.BaseRecording]:
    eeg_dir = probe_folder / 'eeg'
    eeg_file = eeg_dir / 'traces_cached_seg0.raw'
    
    if eeg_file.exists():
        logger.info(f"  EEG exists, skipping")
        return si.load(eeg_dir)
    
    fs = rec.get_sampling_frequency()
    decimation_factor = int(fs / target_fs)
    
    logger.info(f"  Downsampling: {fs:.0f}Hz -> {fs/decimation_factor:.0f}Hz (factor={decimation_factor})")
    
    dec_rec = DecimatedRecording(rec, decimation_factor)
    saved = dec_rec.save(
        format='binary',
        folder=eeg_dir,
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        overwrite=True
    )
    
    log_recording(saved, "  EEG")
    return saved


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


def synchronize_probes(
    probes: dict[str, Probe],
    adc: ADC,
    spike_times: dict[str, np.ndarray],
    output_path: Path
) -> dict[str, np.ndarray]:
    """Synchronize probe spike times to ADC global timebase."""
    logger.info("=" * 60)
    logger.info("SYNCHRONIZING TO ADC")
    
    adc.build_global_timestamps()
    # adc.save_timestamps(output_path)
    
    synced_spikes = {}
    
    for name, probe in probes.items():
        if name == 'OneBox-ADC':
            continue
        
        if name not in spike_times:
            logger.warning(f"{name}: No spike times, skipping")
            continue
        
        logger.info(f"{name}:")
        
        probe.build_global_timestamps()
        probe.sync_to(adc)
        
        ks_spikes = spike_times[name] / probe.fs
        log_timestamps(ks_spikes, f"  Input spikes")
        
        adc_spikes = interpolate_to_target(ks_spikes, probe.timestamps_map)
        log_timestamps(adc_spikes, f"  Output spikes")
        
        save_dir = output_path / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(save_dir / "timestamps_map.npy", probe.timestamps_map)
        np.save(save_dir / "adc_spikes.npy", adc_spikes)
        
        synced_spikes[name] = adc_spikes
        logger.info(f"  Saved to {save_dir}")
    
    logger.info("SYNCHRONIZATION COMPLETED")
    logger.info("=" * 60)
    return synced_spikes