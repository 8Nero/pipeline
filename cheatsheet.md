# Neural Recording Pipeline

Object-oriented pipeline for processing OpenEphys neural recordings.

## Structure

```
pipeline/
├── probe.py          # Probe class (also used for ADC)
├── operations.py     # load, concat, sort, sync
├── decimation.py     # DecimatedRecording
└── utils.py          # Config, logging, file handling
```

## Core Classes

### Probe

```python
probe = Probe(name='ProbeA', fs=30000.0)
probe.load_from_sessions(session_paths)  # loads recordings + timestamps
probe.build_global_references(mode='samples')  # or mode='timestamps'
probe.concatenate(output_path, save_kwargs)

# Sync to target
probe.sync_to(adc)  # creates probe.sync_map
```

### ADC (using Probe)

```python
adc = Probe(name='OneBox-ADC', fs=30300.5)
adc.load_from_sessions(session_paths)

# Build references in both modes
adc.build_global_references(mode='samples')
global_samples = adc.get_global_events()

adc.build_global_references(mode='timestamps')
global_timestamps = adc.get_global_events()

# Convert samples to timestamps (same clock)
timestamps = samples / adc.fs
```

## Synchronization

Sample-based sync with direct timestamp conversion:

```python
probe.build_global_references(mode='samples')
adc.build_global_references(mode='samples')

probe.sync_to(adc)  # probe.sync_map is [N, 2] (samples)

# Convert spike samples to ADC reference
adc_samples = interpolate(spike_samples, probe.sync_map)
adc_times = adc_samples / adc.fs
```

### Workflow

```
1. Probe spike samples -> ADC samples (via sync_map interpolation)
2. ADC samples -> ADC timestamps (divide by adc.fs)
```

## Usage

### Full pipeline

```bash
python run_pipeline.py --config config.yaml
python run_pipeline.py --probe ProbeA ProbeB --debug
```

### Scripting

```python
from pipeline import Probe, interpolate

probe = Probe('ProbeA')
adc = Probe('OneBox-ADC', fs=30300.5)

probe.load_from_sessions(session_paths)
adc.load_from_sessions(session_paths)

probe.build_global_references(mode='samples')
adc.build_global_references(mode='samples')

probe.sync_to(adc)

adc_samples = interpolate(spike_samples, probe.sync_map)
adc_times = adc_samples / adc.fs
```