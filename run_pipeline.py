#!/usr/bin/env python3
import argparse
from pathlib import Path
from kilosort_pipeline.utils import setup, copy_to_remote
from kilosort_pipeline.kilosort_pipeline import kilosort_pipeline
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description='Run Kilosort pipeline on OpenEphys recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m run_pipeline                              # Default: config.yaml, all probes
  python -m run_pipeline --config my_config.yaml      # Custom config file
  python -m run_pipeline --probe ProbeA               # Process single probe
  python -m run_pipeline --probe ProbeA ProbeB        # Process multiple probes
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--probe',
        type=str,
        nargs='+',
        choices=['ProbeA', 'ProbeB', 'ProbeC', 'ProbeD'],
        help='Probe(s) to process (e.g., --probe ProbeA ProbeB).'
    )
    
    args = parser.parse_args()
    
    protocol = setup(config_path=args.config, probe_filter=args.probe)
    probe_concat = kilosort_pipeline(protocol)

    if probe_concat:
        logger.info(f"Copying to remote path {protocol['base_output']}")
        copy_to_remote(Path(protocol['local_output']), protocol['base_output'])

if __name__ == '__main__':
    main()
