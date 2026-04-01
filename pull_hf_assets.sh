#!/usr/bin/env bash
set -e

# Run this script from CodeDllm/

mkdir -p "projects/sft_dream_py_ast/multitask_stage2_totalmean_0-5_codeonly_2e-6lr_2lm_8ga_4gpus_m7_1000traincases_2epochs/checkpoint-1000"
mkdir -p "data"

huggingface-cli download FrankeMING/sft_multitask_ckpt1000 \
  --local-dir "projects/sft_dream_py_ast/multitask_stage2_totalmean_0-5_codeonly_2e-6lr_2lm_8ga_4gpus_m7_1000traincases_2epochs/checkpoint-1000" \
  --local-dir-use-symlinks False

huggingface-cli download FrankeMING/CodeDllm_data \
  --repo-type dataset \
  --local-dir "data" \
  --local-dir-use-symlinks False
