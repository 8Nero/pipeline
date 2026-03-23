import sys
import matplotlib
matplotlib.use('Agg')

from pipeline.operations import load_probes, sort_probes, concatenate_probes, synchronize_probes
from pipeline.utils import setup_pipeline, copy_to_remote

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = setup_pipeline(config_path, debug=True)
    probes = load_probes(config['session_paths'])
    concatenate_probes(probes, config)
    sort_probes(probes, config)
    synchronize_probes(probes, config)
    copy_to_remote(local_path=config['local_output'],
                   remote_path=config['remote_output'],
                   overwrite_mode=config['transfer_mode'])
