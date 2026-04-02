#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

./env_setup.sh
eval "$(conda shell.bash hook)"
conda activate dllm

./pull_hf_assets.sh

mkdir -p results/baseline
python eval/batch_eval.py \
  --input_file data/baseline.jsonl \
  --output_dir results \
  --run_name baseline/baseline

python rl.py \
  config=configs/rl_dream_py_multitask.yaml \
