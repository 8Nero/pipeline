#!/usr/bin/env python3
import argparse
from pathlib import Path
from kilosort_pipeline.utils import setup
from kilosort_pipeline.workflow import run_full_pipeline
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
    
    # Setup pipeline configuration
    protocol = setup(config_path=args.config)
    
    # Add runtime arguments to protocol
    protocol['probe_filter'] = args.probe
    if protocol['probe_filter']:
        logger.info(f"Probe filter: {protocol['probe_filter']}")

    run_full_pipeline(protocol)

if __name__ == '__main__':
    main()

