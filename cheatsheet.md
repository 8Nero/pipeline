# Neural Recording Pipeline

Object-oriented pipeline for processing OpenEphys neural recordings.

## Structure

```
pipeline/
├── probe.py          # Probe, ADC classes
├── operations.py     # load, concat, sort, sync
├── decimation.py     # DecimatedRecording
└── utils.py          # Config, logging, file handling
```

## Core Classes

### Probe

```python
probe = Probe(name='ProbeA', fs=30000.0)
probe.load_session(session_path, stream_id)
probe.load_timestamps(event_path, cont_path)
probe.build_global_timestamps()
probe.concatenate(output_path, save_kwargs)

# Sync to target timeline
probe.sync_to(adc)  # creates probe.timestamps_map
```

### ADC

```python
adc = ADC(name='OneBox-ADC', fs=30300.5)
adc.load_timestamps(event_path, cont_path)  # loads sample_numbers too
adc.build_global_timestamps()

# Access global samples/timestamps for interpolation
adc.get_global_samples()
adc.get_global_timestamps()
```

## Synchronization

Each probe builds its own timeline. Use `sync_to()` to create a mapping to the target:

```python
probe.build_global_timestamps()
adc.build_global_timestamps()

probe.sync_to(adc)  # probe.timestamps_map is now [N, 2]

# Interpolate spike times
adc_spikes = interpolate_to_target(probe_spikes, probe.timestamps_map)
```

### Handling mismatched durations

```
Session 2: ProbeA=50s, ADC=100s

probe.sync_to(adc) uses only the 50s overlap.
Next session offset advances by probe's duration (50s), not ADC's.
```

## ADC Sample Interpolation

Convert TTL sample indices to timestamps:

```python
adc.build_global_timestamps()

camera_timestamps = samples_to_timestamps(
    camera_sample_indices,
    adc.get_global_samples(),
    adc.get_global_timestamps()
)
```

## Usage

### Full pipeline

```bash
python run_pipeline.py --config config.yaml
python run_pipeline.py --probe ProbeA ProbeB --debug
```

### Scripting

```python
from pipeline import Probe, ADC, interpolate_to_target, samples_to_timestamps

probe = Probe('ProbeA')
adc = ADC()

# ... load and build timestamps ...

probe.sync_to(adc)
synced_spikes = interpolate_to_target(spikes, probe.timestamps_map)
```