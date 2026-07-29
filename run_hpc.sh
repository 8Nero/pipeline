#!/bin/bash
#SBATCH --account=<account-name>
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --job-name=pipeline
#SBATCH --output=pipe_%A.out
#SBATCH --error=pipe_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=<email>

CONFIG_PATH='</path/to/config.yaml>'

# Load environment
module purge all
module load uv/0.8.18

source .venv/bin/activate
uv pip install -e .

uv run pipe "$CONFIG_PATH"