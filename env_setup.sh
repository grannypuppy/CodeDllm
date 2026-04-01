#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

conda create -n dllm python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate dllm

pip install --upgrade pip setuptools wheel
pip install -e .

if [ -f ".gitmodules" ]; then
  git submodule update --init --recursive
fi

echo "setup done: conda env = dllm"
