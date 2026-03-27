# Copilot Instructions for Kilosort Pipeline

## Project Overview

Automated spike sorting pipeline for OpenEphys multi-probe electrophysiology recordings using Kilosort4 and SpikeInterface. Processes raw recordings → concatenates → spike sorts → synchronizes to ADC reference frame → copies to remote storage.

## Architecture

### Pipeline Flow (`run_pipeline.py`)
```
load_probes → concatenate_probes → run_kilosort4 → synchronize_probes → copy_to_remote
```

### Core Components
- **`Probe` class** (`probe.py`): Central abstraction for neural probes AND ADC. Handles loading, timestamps, concatenation, and synchronization.
- **`operations.py`**: High-level pipeline operations (load, concat, sort, sync)
- **`decimation.py`**: On-the-fly downsampling via SpikeInterface's `BaseRecording` extension
- **`utils.py`**: Config loading, logging setup, file transfer

### Key Design Decisions
- ADC (`OneBox-ADC`) uses same `Probe` class but with different sampling rate (30300.5 Hz vs 30000.0 Hz)
- Synchronization uses chirp signal matching between probe and ADC event timestamps
- All outputs saved as `.npy` files for easy numpy loading
- Resume capability: checks for existing outputs before reprocessing

## Development

### Setup
```bash
uv sync && source .venv/bin/activate
```

### Running
```bash
python -m run_pipeline                           # All probes, local config.yaml
python -m run_pipeline --probe ProbeA ProbeB     # Specific probes
python -m run_pipeline --config /path/to/config.yaml --debug
```

## Code Patterns

### Probe Loading
Probes identified by stream names containing `.Probe` substring or `OneBox-ADC`:
```python
probe = Probe(name='ProbeA', fs=30000.0)  # Neural probe
adc = Probe(name='OneBox-ADC', fs=30300.5)  # ADC has different sample rate
```

### Synchronization Pattern
```python
probe.build_global_references(mode='samples')
adc.build_global_references(mode='samples')
sync_map = probe.sync_to(adc, mode='samples')
adc_samples = interpolate(spike_samples, sync_map)
adc_times = adc_samples / adc.fs
```

### Error Handling
Pipeline continues processing remaining probes if one fails:
```python
for name, probe in probes.items():
    try:
        # process
    except Exception as e:
        logger.error(f"Failed for {name}: {e}")
        continue
```

### Logging
Uses `loguru` with structured output. Helper functions in `utils.py`:
- `log_recording()`: channels, duration, size
- `log_timestamps()`: range and count
- `log_samples()`: sample range and count

## File Structure

**Output per session:**
```
{local_output}/{session_name}/
├── {probe}/
│   ├── concat/traces_cached_seg0.raw    # Concatenated binary (int16)
│   ├── eeg/traces_cached_seg0.raw       # Downsampled EEG
│   ├── kilosort/spike_times.npy         # Raw spike times
│   ├── sync_map.npy                     # Probe→ADC mapping
│   └── adc_spike_times.npy              # Synchronized spikes
└── OneBox-ADC/
    ├── global_samples.npy
    └── global_timestamps.npy
```

## Dependencies

- **SpikeInterface**: Recording I/O, preprocessing (`se.read_openephys`)
- **Kilosort4**: Spike sorting (`run_kilosort`)
- **PyTorch**: GPU backend (CUDA 12.8)
- **pynapple**: Time series (used in sync)

## Configuration (`config.yaml`)

Required fields: `recording_paths`, `remote_output`, `local_output`, `save_kwargs`

`mp_context`: Use `'spawn'` on Windows, `'fork'` on macOS/Linux
