import argparse
from pathlib import Path
from loguru import logger

import matplotlib
matplotlib.use('Agg') # Switch non GUI backend for kilosort internal plotting

from . import (
    Probe,
    load_probes,
    concatenate_probes,
    run_kilosort4,
    synchronize_probes,
    save_adc_references,
    setup_pipeline,
    normalize_probe_filter,
    copy_to_remote
)


def main():
    parser = argparse.ArgumentParser(description='Neural recording pipeline')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--config_remote', type=str, default=None)
    parser.add_argument('--probe', type=str, nargs='+', default=None,
                        choices=['ProbeA', 'ProbeB', 'ProbeC', 'ProbeD', 'ADC', 'adc', 'OneBox-ADC'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overwrite', type=str, default='newer',
                        choices=['all', 'skip-all', 'prompt', 'newer'])
    
    args = parser.parse_args()
    
    if args.config_remote:
        config_path = Path(r"R:\Basic_Sciences\Phys\SenzaiLab\pipeline_output\configs") / args.config_remote
    else:
        config_path = args.config
    
    config = setup_pipeline(str(config_path), debug=args.debug)
    
    probe_names = normalize_probe_filter(args.probe, config['recording_paths'])
    if 'OneBox-ADC' not in probe_names:
        probe_names.append('OneBox-ADC')
    
    logger.info(f"Probes: {probe_names}")
    
    probes = load_probes(
        session_paths=config['recording_paths'],
        probe_names=probe_names
    )
    
    neural_probes = concatenate_probes(
        probes=probes,
        output_path=config['local_output'],
        save_kwargs=config['save_kwargs'],
        target_fs=config.get('target_fs')
    )
    
    spike_times = run_kilosort4(
        probes=neural_probes,
        output_path=config['local_output']
    )
    
    # Save ADC samples -> timestamps mapping
    save_adc_references(adc=probes.get('OneBox-ADC'), output_path=config['local_output'])
    
    synchronize_probes(
        probes=probes,
        target='OneBox-ADC',
        spike_times=spike_times,
        output_path=config['local_output']
    )
    
    copy_to_remote(
        local_path=config['local_output'],
        remote_path=config['remote_output'],
        overwrite_mode=args.overwrite
    )
    
    logger.info("=" * 60)
    logger.success("PIPELINE COMPLETED")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
