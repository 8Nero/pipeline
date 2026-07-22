# Pipeline

Automated spike sorting pipeline for OpenEphys sessions with [Kilosort4](https://github.com/MouseLand/Kilosort) and [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

The pipeline runs the following stages in sequence:
```
load OpenEphys sessions → concatenate → run kilosort → downsample to EEG → synchronize → copy to remote
```

## Installation

<details>
<summary>instructions for installing uv</summary>

**Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

*Exit and reopen your terminal after installation.*
</details>

Use `uv tool` to install the pipeline system wide

```bash
uv tool install "pipeline@git+https://github.com/8Nero/pipeline.git"
```

To update to the latest version:
```bash
uv tool upgrade pipeline
```

## Usage

### In the terminal

```bash
# Run pipeline
pipe /path/to/config.yaml
pipe /path/to/config.yaml --debug
```

### Configuration

The pipeline takes path to a `.yaml` file as input:

```yaml
run_name: "my_session"
session_paths:                      # OpenEphys session folders containing the Record Node folder
  - "/path/to/session1"
  - "/path/to/session2"
# probe_filter:                       # Optional; probe names to skip
#   - "ProbeA"
#   - "OneBox-ADC"
local_output: "/local/disk"         # Local output directory
remote_output: "/remote/storage"    # Optional; Remote storage path to copy
copy_mode: 'newer'                  # 'newer', 'prompt', 'all', 'skip-all'
target_fs: 1250.0                   # EEG downsampling (Hz)
verbose: True
overwrite: False
# SpikeInterface arguments for concatenation, EEG downsampling
job_kwargs:
  n_jobs: 4
  chunk_duration: '2s'
  progress_bar: True
  mp_context: 'spawn'               # For Windows; use 'fork' for macOS
# Kilosort arguments
per_shank: False
#openblas_threads: 24
```

The `copy_mode` option controls how existing files are handled when copying to remote storage:
- `newer` — Only overwrite if local file is newer (default)
- `prompt` — Prompt before overwriting each file
- `skip-all` — Skip all existing files
- `all` — Overwrite all existing files


## Output Structure

```
{local_output}/{run_name}/
├── logs/
│   └── {run_name}_{timestamp}.log
├── ProbeA/
│   ├── concat/
│   │   ├── traces_cached_seg0.raw  # Concatenated binary file (int16)
│   │   └── si_folder.json          # SpikeInterface metadata
│   ├── eeg.dat                     # Downsampled EEG (if target_fs set)
│   ├── probe_geometry.png          # Configurated probe plot
│   ├── sync_map.npy                # [N x 2] array of [probe_time, adc_time] pairs
│   ├── sync_drift.png              # Clock drift visualization
│   ├── adc_spike_times.npy         # Spike times interpolated to ADC timebase
│   └── kilosort/                   # Kilosort4 outputs
│       ├── spike_times.npy
│       ├── spike_clusters.npy
│       └── ...
├── ProbeB/
│   └── ...
└── OneBox-ADC/
    ├── sync_map.npy                # ADC samples → timestamps mapping
    ├── periods_samples.npy         # Session start, end time in sample numbers
    └── periods_timestamps.npy      # Session start, end time in seconds
    
```
---
## How the Pipeline Works

The main script in `pipeline.run_script` runs the following stages:

### 1. Load probes

Auto-discovers all probes and the ADC from OpenEphys session folders. Each probe loads its recordings .dat file, TTL sync events, and timestamps.

```python
from pipeline.operations import load_probes

probes = load_probes(["/path/to/session1", "/path/to/session2"])
# probes = {'ProbeA': Probe(...), 'OneBox-ADC': Probe(...)}
```

### 2. Concatenate recordings

Multi-session recordings are concatenated into a single binary file per probe using SpikeInterface.

```python
from pipeline.operations import concatenate_probes

concatenate_probes(probes, config)
# Writes: {local_output}/{probe}/concat/traces_cached_seg0.raw
```

### 3. Sorting

Runs Kilosort4 on each probe's concatenated recording.

```python
from pipeline.operations import sort_probes

# probe_paths: Dictionary of paths to concat 
# {'ProbeA': {local_output}/{probe}/concat}, ...}
sort_probes(probe_paths, config)
```

### 4. Downsample EEG

Decimates the concatenated recording to `target_fs`. Skip if `target_fs` isn't configured.

```python
from pipeline.operations import downsample_probes

downsample_probes(probe_paths, config)
```

### 5. Synchronize to ADC

Interpolate spike timestamps from probe time to ADC time.

```python
from pipeline.operations import synchronize_probes

synchronize_probes(probes, config)
# Writes: {local_output}/{probe}/sync_map.npy        — [probe_time, adc_time] pairs
#         {local_output}/{probe}/adc_spike_times.npy  — spike times in ADC timebase
#         {local_output}/{probe}/sync_drift.png       — clock drift plot
```

### 6. Copy to remote

```python
from pipeline.utils import copy_to_remote

copy_to_remote(local_path=config['local_output'],
               remote_path=config['remote_output'],
               overwrite_mode=config['copy_mode'])
```