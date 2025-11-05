#!/usr/bin/env python3
from pipeline.sync import synchronize
from pipeline.concat import concat
from pipeline.sorting import run_kilosort4
from pipeline.utils import copy_to_remote, parse_timestamps, setup, validate_probe_filter

import argparse
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser(
        description='Run Kilosort pipeline on OpenEphys recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m run_pipeline                              # Default: config.yaml, all available probes
  python -m run_pipeline --config my_config.yaml      # Custom config file (local)
  python -m run_pipeline --config_remote session1.yaml # Remote config from network path
  python -m run_pipeline --probe ProbeA               # Process single probe
  python -m run_pipeline --probe ProbeA ProbeB        # Process multiple probes
  python -m run_pipeline --probe ProbeA ADC           # Include ADC concatenated data
  python -m run_pipeline --debug                      # Enable debug logging to console
  python -m run_pipeline --overwrite prompt           # Prompt before overwriting files
  python -m run_pipeline --overwrite skip-all         # Skip all existing files when copying to remote storage
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--config_remote',
        type=str,
        default=None,
        help='Config filename in remote configs directory (R:/Basic_Sciences/Phys/SenzaiLab/pipeline_output/configs/)'
    )
    parser.add_argument(
        '--probe',
        type=str,
        nargs='+',
        choices=['ProbeA', 'ProbeB', 'ProbeC', 'ProbeD', 'ADC', 'adc', 'OneBox-ADC'],
        default=None,
        help='Probe(s) to process (e.g., --probe ProbeA ProbeB ADC). If not specified, processes all available probes.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging to console'
    )
    parser.add_argument(
        '--overwrite',
        type=str,
        choices=['all', 'skip-all', 'prompt'],
        default='all',
        help='Overwrite existing files in remote (default: all)'
    )
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config_remote:
        config_path = Path(r"R:\Basic_Sciences\Phys\SenzaiLab\pipeline_output\configs") / args.config_remote
    else:
        config_path = args.config
    
    # Setup pipeline configuration
    protocol = setup(config_path=str(config_path), debug=args.debug)
    
    # Normalize probe filter (converts ADC variants, auto-detects if None)
    protocol['probe_filter'] = validate_probe_filter(args.probe, protocol['recording_paths'])
    logger.info(f"Probe filter: {protocol['probe_filter']}")

    logger.info("STARTING PIPELINE")
    # Load and concatenate recordings
    probe_concat = concat(
        rec_paths=protocol['recording_paths'],
        probe_filter=protocol['probe_filter'],
        output_path=protocol['local_output'],
        save_kwargs=protocol['save_kwargs'],
        target_fs=protocol['target_fs']
    )

    # Spike sorting
    run_kilosort4(
        probe_concat=probe_concat,
        output_path=protocol['local_output']
    )

    # Synchronization requires ADC timestamps
    if 'OneBox-ADC' not in protocol['probe_filter']:
        protocol['probe_filter'].append('OneBox-ADC')
        logger.debug("Added OneBox-ADC to probe filter for synchronization")

    # Load timestamps and synchronize
    timestamps = parse_timestamps(protocol['recording_paths'], protocol['probe_filter'])
    synchronize(
        output_path=protocol['local_output'], 
        probe_filter=protocol['probe_filter'], 
        timestamps=timestamps
    )

    logger.info(f"Copying to: {protocol['remote_output']}")
    copy_to_remote(
        local_path=protocol['local_output'],
        remote_path=protocol['remote_output'],
        overwrite_mode=args.overwrite
    )
    logger.success(f"Copying completed")
    logger.success("PIPELINE COMPLETED SUCCESSFULLY")
if __name__ == '__main__':
    main()

