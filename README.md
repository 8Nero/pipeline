# Kilosort Pipeline

Automated spike sorting pipeline for OpenEphys recordings with [Kilosort4](https://github.com/MouseLand/Kilosort) and [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

## Installation

[uv](https://github.com/astral-sh/uv) seems to be simpler and faster alternative to conda. ([SpikeInterface's tips on uv](https://github.com/SpikeInterface/spikeinterface/blob/main/installation_tips/README.md))

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
session_name: "session_name"
recording_paths:                    # OpenEphys session folders containing structure.oebin file
  - "/path/to/session1"
  - "/path/to/session2"
local_output: "/local/disk"         # path to local disk
# Default parameters
remote_output: "/remote/storage"    # path to fsmresfiles
fs: 30000.0
target_fs: 1250.0                   # EEG downsampling (Hz)
save_kwargs:
  n_jobs: 16
  chunk_duration: '2s'
  mp_context: 'spawn'               # for Windows; use 'fork' for macOS/Linux
  overwrite: true
```

2. **Run pipeline:**

```bash
# Process all probes
python -m run_pipeline

# Process from common config
python -m run_pipeline --config R:/remote_path/config.yaml

# Process specific probe(s)
python -m run_pipeline --probe ProbeA
python -m run_pipeline --probe ProbeA ProbeB

# If the config file is in fsmresfiles
python -m run_pipeline --probe ProbeA --config_remote session1.yaml 

# Concatenate ADC data (for synchronization ADC timestamps are always used)
python -m run_pipeline --probe ProbeA ADC           

# Overwrite options when copying to remote server
python -m run_pipeline --overwrite newer            # Default: Only overwrite if local file is newer
python -m run_pipeline --overwrite prompt           # Prompt before overwriting files
python -m run_pipeline --overwrite skip-all         # Skip all existing files when copying to remote storage
python -m run_pipeline --overwrite all              # Overwrite all existing files when copying to remote storage
```


## Output Structure

```
{local_output}/{session_name}/
├── {session_name}_{timestamp}.log  # Timestamped log file
├── ProbeA/
│   ├── concat/
│   │   ├── traces_cached_seg0.raw  # Concatenated binary file (int16)
│   │   └── si_folder.json          # SpikeInterface metadata
│   ├── eeg_data.dat                # Downsampled EEG (if target_fs set)
│   ├── adc_spikes.npy              # Spike times in ADC timebase
│   ├── timestamps_map.npy          # [N x 2] array of [probe_time, adc_time] pairs
│   └── kilosort/                   # Kilosort4 outputs
│       ├── spike_times.npy
│       ├── spike_clusters.npy
│       └── ...
└── ProbeB/
    └── ...
```
