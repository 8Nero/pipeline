# Kilosort Pipeline

Automated spike sorting pipeline for OpenEphys multi-probe recordings with [Kilosort4](https://github.com/MouseLand/Kilosort) and [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).


## Installation

### 1. Install uv

**Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

*Exit and reopen your terminal after installation.*

### 2. Clone and install
```bash
git clone https://github.com/8Nero/pipeline.git
cd pipeline
uv sync --extra full
```

### 3. Activate environment

**Linux:**
```bash
source .venv/bin/activate
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate
```
---

## Using the pipeline

1. **Create `config.yaml`:**

```yaml
run_name: "my_session"
session_paths:                      # OpenEphys session folders containing structure.oebin file
  - "/path/to/session1"
  - "/path/to/session2"
local_output: "/local/disk"         # Path to local disk
remote_output: "/remote/storage"    # Path to fsmresfiles
per_shank: True                     # Run Kilosort per shank
copy_mode: 'newer'                  # 'newer', 'prompt', 'all', 'skip-all'
# Default parameters
target_fs: 1250.0                   # EEG downsampling (Hz)
verbose: True
overwrite: False
job_kwargs:
  n_jobs: 4
  chunk_duration: '2s'
  progress_bar: True
  mp_context: 'spawn'               # For Windows; use 'fork' for macOS/Linux
```

2. **Run pipeline:**

```bash
# Process all probes with default config.yaml
python run_script.py

# Use a different config file
python run_script.py --config /path/to/config.yaml

# Enable debug logging
python run_script.py --debug
```

The `copy_mode` config option controls how existing files are handled when copying to remote storage:
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
    └── sync_map.npy                # ADC samples → timestamps mapping
```