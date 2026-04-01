# CodeDllm

Research and engineering codebase for **diffusion language models (Diffusion LM; Dream as the main stack)** on tasks such as **code generation** and **vulnerability repair**. Built around Hugging Face–style `DreamModel` / `DreamTokenizer`, it provides full pipelines for **SFT (supervised fine-tuning)**, **semi-autoregressive sampling with AST or multitask heads**, **RL (policy gradient + reward model + rollout)**, and **SVEN-oriented data and RL variants**.

## Repository map

| Path | Role |
|------|------|
| [`models/`](models/README.md) | Dream and Dream-Multitask modeling, generation, sampling, rollout |
| [`train/`](train/README.md) | SFT / RL training entrypoints and dataset code |
| [`rl.py`](rl.py) / [`rl_sven.py`](rl_sven.py) | RL orchestration (data planning, multiprocess rollout, launching training) |
| [`eval/`](eval/README.md) | Batch generation, evaluation, performance and result analysis |
| [`reward/`](reward/README.md) | Scalar reward models (including SVEN variants) |
| [`utils/`](utils/README.md) | Post-processing, sandbox, AST/data structures, conversion |
| [`configs/`](configs/README.md) | OmegaConf YAML experiment configs |
| [`accelerate_configs/`](accelerate_configs/README.md) | Accelerate + DeepSpeed ZeRO-3 configs |
| [`scripts/`](scripts/README.md) | Shell batch drivers and data/visualization scripts |
| [`data/`](data/README.md) | Datasets and bucket indices (large artifacts often distributed separately) |
| [`docs/`](docs/README.md) | Design notes and pipelines (RL, semi-AR, AST, logging, etc.) |
| [`projects/`](projects/README.md) | Default root for experiment outputs (checkpoints, logs, etc.) |
| [`results/`](results/README.md) | Offline evaluation aggregates |
| [`temp/`](temp/README.md) | Temporary evaluations and intermediates |
| [`temp_utils/`](temp_utils/README.md) | One-off data inspection scripts |
| [`wandb/`](wandb/README.md) | Local Weights & Biases run cache |

## Environment

- **Python**: Match `requirements.txt`; core deps include `torch==2.6.0`, `transformers==4.52.4`, `accelerate`, `deepspeed`, `omegaconf`, `wandb`, etc.
- **Install**: From the repo root, `pip install -r requirements.txt` (align GPU/CUDA with your PyTorch build).

## Typical workflow (conceptual)

1. **Data**: Prepare `jsonl`, bucketed subsets, or `preprocessed` `.pt` files as described under `data/`.
2. **SFT**: Run `train/sft_dream_train*.py` with `configs/sft_*.yaml`; outputs usually go under `projects/sft_*`.
3. **RL**: `rl.py` (or `rl_sven.py` with SVEN data) reads YAML, plans rollout subsets, worker processes run `models/dream/rl_rollout*.py`, then `train/rl_dream_train*.py` updates the policy.
4. **Evaluation**: `eval/batch_eval.py`, `eval/gen_dream_eval*.py`, etc.; outputs land in `results/` or `temp/`.

For equations, tensor shapes, and default hyperparameters, see `docs/` (e.g. `CODEDLLM_RL_PIPELINE_IMPLEMENTATION_EXPLAINED.md`, `RL_DREAM_PY_YAML_PARAMETER_REFERENCE.md`).

## Other root files

| File | Description |
|------|-------------|
| `requirements.txt` | Pinned dependencies |
| `run.sh` | Optional local launcher stub (fill as needed) |
| `*.log` | Training/eval logs (usually not versioned) |

## Notes

- Some paths are listed in `.gitignore` (e.g. `data/`, `projects/`, `wandb/`, `results/`); after cloning, restore data or outputs from backup or internal storage.
- Cursor export notes (e.g. persistent GPU memory in RL) can be copied into `docs*` and linked from here if they live on another machine.

Per-file descriptions for each subdirectory are in that directory’s `README.md`.
