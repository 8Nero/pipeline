# Pipeline

Automated spike-sorting pipeline for Open Ephys sessions using [Kilosort4](https://github.com/MouseLand/Kilosort) and [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

The pipeline runs the following stages in sequence:

```text
load Open Ephys sessions → concatenate recordings → run Kilosort4 → decimate for EEG (optional) → synchronize time bases → copy outputs to remote storage (optional)
```

## Installation

<details>
<summary>Install uv</summary>

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

Use `uv tool` to install the pipeline as a globally available command:

```bash
uv tool install "pipeline@git+https://github.com/8Nero/pipeline.git"
```

To update to the latest version:

```bash
uv tool upgrade pipeline
```

## Usage

### In the Terminal

```bash
# Run pipeline
pipe /path/to/config.yaml
pipe /path/to/config.yaml --debug
```

### Configuration

The pipeline accepts the path to a `.yaml` configuration file:

```yaml
run_name: "my_session"
session_paths:                      # Open Ephys session directories containing a `Record Node` subdirectory
  - "/path/to/session1"
  - "/path/to/session2"
# probe_filter:                       # Optional; probe names to skip
#   - "ProbeA"
local_output: "/local/disk"         # Local output directory
remote_output: "/remote/storage"    # Optional; local or mounted filesystem destination
copy_mode: 'newer'                  # 'newer', 'prompt', 'all', 'skip-all'
target_fs: 1250.0                   # Target EEG sampling frequency (Hz)
verbose: True
overwrite: False
# SpikeInterface concatenation arguments
job_kwargs:
  n_jobs: 4
  chunk_duration: '2s'
  progress_bar: True                # Also controls EEG downsampling progress
  mp_context: 'spawn'               # For Windows; use 'fork' for macOS
# Kilosort4 arguments
per_shank: False
# openblas_threads: 24
```

## Output Structure

The following tree shows the main outputs from a typical run; SpikeInterface and Kilosort4 may create additional files.

```text
{local_output}/{run_name}/
├── logs/
│   ├── {run_name}_{timestamp}.log
│   └── kilosort_{probe}.log
├── ProbeA/
│   ├── concat/
│   │   ├── traces_cached_seg0.raw  # Concatenated binary data
│   │   └── si_folder.json          # SpikeInterface metadata
│   ├── eeg.dat                     # Decimated EEG (if target_fs is set)
│   ├── probe_geometry.png          # Probe geometry plot
│   ├── sync_map.npy                # [N × 2] array of [probe_time_s, adc_time_s] pairs
│   ├── sync_drift.png              # Clock drift visualization
│   ├── adc_spike_times.npy         # Spike times interpolated to the ADC time base
│   └── kilosort/                   # Kilosort4 outputs
├── ProbeB/
│   └── ...
└── OneBox-ADC/
    ├── concat/
    │   ├── traces_cached_seg0.raw  # Concatenated ADC binary data
    │   └── si_folder.json          # SpikeInterface metadata
    ├── sync_map.npy                # Global ADC sample indices → timestamps (s)
    ├── periods_samples.npy         # Global session start/end sample indices
    └── periods_timestamps.npy      # Global session start/end timestamps (s)
```

With `per_shank: True`, Kilosort4 results are stored in `kilosort/shank_<id>/`, and interpolated spike times are written to `adc_spike_times_<id>.npy`.

## Running on Quest

To run the pipeline on Quest compute nodes using Slurm, see the [Quest guide](slurm_guide.md).

## How the Pipeline Works

See the [tutorial notebook](tutorial.ipynb) for a detailed walkthrough.
