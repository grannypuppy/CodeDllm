# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

CodeDllm is a Python ML research project for code generation using discrete diffusion language models (dLLMs), built on NVIDIA's Dream model architecture. There are no web services, Docker containers, or databases — just Python scripts for training, inference, evaluation, and analysis.

### Key Dependencies

- **transformers must be pinned to `>=4.45,<4.50`** — newer versions (5.x) removed `ROPE_INIT_FUNCTIONS["default"]` which the Dream model code depends on.
- PyTorch CPU is sufficient for development/testing (model instantiation, forward pass, AST parsing, sandbox eval). GPU (CUDA) is required only for actual training and full-scale generation.
- `torchvision` is required by `train/utils.py` even though it's not used in core ML logic.

### Running the Code

- All scripts assume they are run from the **repository root** (`/workspace`). Internal imports use `sys.path.insert(0, ...)` relative to project root.
- **Training**: `accelerate launch --config_file accelerate_configs/<config>.yaml train/<script>.py config=configs/sft_dream_py.yaml` (requires multi-GPU).
- **Generation/Eval**: `torchrun --nproc_per_node=N eval/gen_eval*.py --model_path <path> ...` (requires GPU).
- **Sandbox eval**: `python eval/batch_eval.py --input_file <path> --run_name <name>` (CPU only).
- See `README.md` for detailed command examples.

### Linting

No linter is configured in the repo. Use `flake8` for basic checks:
```
flake8 --select=E9,F63,F7,F82 .
```

### Testing Without GPU

Since training and generation require GPU, development verification on CPU includes:
1. All module imports (internal + external)
2. Dream model instantiation + forward pass (tiny config)
3. AST parsing via tree-sitter
4. Python sandbox code execution (`utils/py_sandbox.py`)
5. Config loading via OmegaConf
6. Training utility functions (`train/utils.py`)

### Gotchas

- The codebase has **no `requirements.txt`** or `pyproject.toml`. Dependencies must be installed manually.
- `PATH` must include `$HOME/.local/bin` for pip-installed scripts like `torchrun` and `accelerate`.
- Pre-trained model weights (Dream-7B) and training/evaluation datasets are external and not included in the repo. Config references `../dLLM-RL/local_models/dream-7b-base`.
