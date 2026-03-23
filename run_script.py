import sys

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '24')
os.environ.setdefault('OMP_NUM_THREADS', '24')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from pipeline.operations import downsample_probes, load_probes, sort_probes, concatenate_probes, synchronize_probes
from pipeline.utils import setup_pipeline, copy_to_remote

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = setup_pipeline(config_path, debug=True)
    
    probes = load_probes(config['session_paths'])
    
    concatenate_probes(probes, config)
    
    probe_paths = {name: Path(config['local_output']) / name / 'concat' for name in probes.keys() if name != 'OneBox-ADC'}
    
    sort_probes(probe_paths, config)
    
    downsample_probes(probe_paths, config)
    
    synchronize_probes(probes, config)
    copy_to_remote(local_path=config['local_output'],
                   remote_path=config['remote_output'],
                   overwrite_mode=config['transfer_mode'])
