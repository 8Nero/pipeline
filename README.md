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

### 2. Clone and install in one step
```bash
git clone https://github.com/Senzai-Lab/kilosort_pipeline.git
cd kilosort_pipeline
uv sync
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

1. **Edit `config.yaml`:**

```yaml
session_name: "session_name"      # timestamp auto-appended
recording_paths:                  # OpenEphys session folders containing structure.oebin file
  - "/path/to/session1"
  - "/path/to/session2"
local_output: "/local/disk"       # path to local SSD
base_output: "/remote/storage"    # path to fsmresfiles

# Optional parameters
target_fs: 1250                   # EEG downsampling (Hz)
save_kwargs:
  n_jobs: 16
  chunk_duration: '2s'
  mp_context: 'spawn'             # for Windows; use 'fork' for macOS/Linux
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
```


## Output Structure

```
{local_output}/{session_name}/
├── {session_name}_{timestamp}.log  # Timestamped log file
├── ProbeA/
│   ├── concat/
│   │   └── traces_cached_seg0.raw  # Concatenated binary file
│   ├── eeg/
│   │   └── eeg_data.bin            # Downsampled EEG (if target_fs set)
│   ├── kilosort/                   # Kilosort4 outputs
│   │   ├── spike_times.npy
│   │   ├── spike_clusters.npy
│   │   └── ...
│   └── sync/                       # Synchronized spikes
│       └── spike_times_synced.npy  # Spike times in ADC time scale
└── ProbeB/
    ├── ...
```