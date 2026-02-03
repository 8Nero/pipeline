from pipeline import downsample_eeg
from pathlib import Path
import spikeinterface as si

if __name__ == '__main__':
    probe_folder = Path(r"")
    probe = si.load(probe_folder / "concat")
    print(probe)
    downsample_eeg(probe_folder = probe_folder, rec = probe)
               