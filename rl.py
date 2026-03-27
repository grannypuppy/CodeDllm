import os
import json
import sys
import subprocess
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


# ---------------------------------------------------------------------------
# Data-planning helpers (unchanged from original, no GPU / dist dependency)
# ---------------------------------------------------------------------------

def _resolve_train_subset_specs(cfg, root: Path):
    subset_dir_cfg = cfg.dataset.get(
        "train_subset_dir",
        "python_splits/test_verified.target_bucket_subsets",
    )
    subset_dir = Path(subset_dir_cfg)
    if not subset_dir.is_absolute():
        subset_dir = root / "data" / subset_dir

    if not subset_dir.exists():
        raise FileNotFoundError(f"train subset dir not found: {subset_dir}")

    default_files = ["1_128.jsonl", "128_256.jsonl", "256_512.jsonl"]
    subset_files_cfg = cfg.dataset.get("train_subset_files", None)
    subset_files = subset_files_cfg if subset_files_cfg is not None else default_files

    specs = []
    for name in subset_files:
        p = subset_dir / str(name)
        if not p.exists():
            raise FileNotFoundError(f"subset file not found: {p}")

        stem = p.stem
        parts = stem.split("_")
        inferred_max_new_tokens = None
        if len(parts) == 2 and parts[1].isdigit():
            inferred_max_new_tokens = int(parts[1])

        max_tokens_map = cfg.rollout.get("subset_max_new_tokens", None)
        if max_tokens_map is not None and stem in max_tokens_map:
            bucket_max_new_tokens = int(max_tokens_map[stem])
        elif inferred_max_new_tokens is not None:
            bucket_max_new_tokens = inferred_max_new_tokens
        else:
            bucket_max_new_tokens = int(cfg.rollout.max_new_tokens)

        steps_map = cfg.rollout.get("subset_steps", None)
        if steps_map is not None and stem in steps_map:
            bucket_steps = int(steps_map[stem])
        else:
            bucket_steps = int(cfg.rollout.steps)

        shrink_map = cfg.training.get("subset_shrink", None)
        if shrink_map is not None and stem in shrink_map:
            bucket_shrink = int(shrink_map[stem])
        else:
            bucket_shrink = int(cfg.training.shrink)

        specs.append(
            {
                "name": stem,
                "path": p,
                "max_new_tokens": bucket_max_new_tokens,
                "steps": bucket_steps,
                "shrink": bucket_shrink,
            }
        )
    return specs


def _count_jsonl(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _materialize_round_subset_chunk(
    cfg,
    run_name: str,
    round_i: int,
    subset_name: str,
    subset_path: Path,
    start_idx: int,
    count: int,
):
    root = Path(__file__).resolve().parent
    out_dir = root / "projects" / str(cfg.experiment.project) / str(run_name) / "round_datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round_{round_i:04d}-{subset_name}.jsonl"

    kept = 0
    with open(subset_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        cur = 0
        for line in fin:
            if not line.strip():
                continue
            if cur < start_idx:
                cur += 1
                continue
            if kept >= count:
                break
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            cur += 1

    if kept != count:
        raise RuntimeError(
            f"Chunk materialization mismatch for {subset_name}: expected {count}, got {kept}."
        )
    return out_path


def _sampling_state_path(cfg, run_name: str) -> Path:
    root = Path(__file__).resolve().parent
    return root / "projects" / str(cfg.experiment.project) / str(run_name) / "round_sampling_state.json"


def _load_sampling_state(cfg, run_name: str):
    state_path = _sampling_state_path(cfg, run_name)
    if not state_path.exists():
        return {
            "subset_cursors": {},
            "round_plan": {},
            "current_subset_idx": 0,
        }
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return state


def _save_sampling_state(cfg, run_name: str, state: dict):
    state_path = _sampling_state_path(cfg, run_name)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, state_path)


def _init_runtime_sampling_state(cfg, run_name: str):
    return _load_sampling_state(cfg, run_name)


def _plan_round_train_data(
    cfg,
    run_name: str,
    round_i: int,
    subset_specs,
    num_task_per_round: int,
    runtime_state: dict,
):
    state = runtime_state
    round_key = str(round_i)

    if round_key in state.get("round_plan", {}):
        return state["round_plan"][round_key]

    subset_cursors = state.setdefault("subset_cursors", {})
    current_subset_idx = int(state.get("current_subset_idx", 0))

    subset_meta = []
    for spec in subset_specs:
        total = _count_jsonl(spec["path"])
        cur = int(subset_cursors.get(spec["name"], 0))
        subset_meta.append({
            "name": spec["name"],
            "path": spec["path"],
            "total": total,
            "cursor": cur,
            "max_new_tokens": int(spec["max_new_tokens"]),
            "steps": int(spec["steps"]),
            "shrink": int(spec["shrink"]),
        })

    chosen = None
    idx = current_subset_idx
    num_subsets = len(subset_meta)
    while idx < num_subsets:
        m = subset_meta[idx]
        remain = m["total"] - m["cursor"]
        if remain < num_task_per_round:
            subset_cursors[m["name"]] = int(m["total"])
            idx += 1
            continue
        take = num_task_per_round
        chosen = (idx, m, take)
        break

    if chosen is None:
        raise RuntimeError(
            "No subset can provide a full round anymore. All configured buckets are exhausted for full-round sampling."
        )

    idx, m, take = chosen
    chunk_path = _materialize_round_subset_chunk(
        cfg=cfg,
        run_name=run_name,
        round_i=round_i,
        subset_name=m["name"],
        subset_path=m["path"],
        start_idx=m["cursor"],
        count=take,
    )

    plan = {
        "subset_name": m["name"],
        "subset_path": str(m["path"]),
        "chunk_path": str(chunk_path),
        "start_idx": int(m["cursor"]),
        "count": int(take),
        "max_new_tokens": int(m["max_new_tokens"]),
        "steps": int(m["steps"]),
        "shrink": int(m["shrink"]),
    }

    state.setdefault("round_plan", {})[round_key] = plan
    next_cursor = int(m["cursor"] + take)
    subset_cursors[m["name"]] = next_cursor
    if next_cursor >= int(m["total"]):
        state["current_subset_idx"] = int(idx + 1)
    else:
        state["current_subset_idx"] = int(idx)
    return plan


# ---------------------------------------------------------------------------
# Subprocess stage runners
# ---------------------------------------------------------------------------

def _get_num_processes(accelerate_config: str) -> int:
    """Read num_processes from an accelerate config YAML."""
    with open(accelerate_config, "r") as f:
        accel_cfg = yaml.safe_load(f)
    return int(accel_cfg.get("num_processes", 1))


def _run_train_stage(script: Path, accelerate_config: str, root: Path, overrides: dict):
    """Launch a DeepSpeed training stage via `accelerate launch`, blocking until completion.

    Each subprocess runs in its own CUDA context; on exit all GPU memory
    (including NCCL buffers and DeepSpeed ZeRO-3 state) is fully reclaimed by the OS.
    """
    cmd = [
        "accelerate", "launch",
        "--config_file", str(accelerate_config),
        str(script),
    ]
    for k, v in overrides.items():
        cmd.append(f"{k}={v}")
    print(f"[RL] >> {' '.join(str(x) for x in cmd)}")
    subprocess.run(cmd, cwd=str(root), check=True)


def _run_rollout_stage(script: Path, nproc: int, root: Path, overrides: dict):
    """Launch a rollout stage via `torchrun`.

    Rollout only reads WORLD_SIZE/RANK/LOCAL_RANK env vars — no Accelerator,
    no NCCL collective ops, no DeepSpeed.  torchrun sets those vars and spawns
    independent per-GPU processes; no accelerate config file is needed.
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        str(script),
    ]
    for k, v in overrides.items():
        cmd.append(f"{k}={v}")
    print(f"[RL] >> {' '.join(str(x) for x in cmd)}")
    subprocess.run(cmd, cwd=str(root), check=True)


def _run_single_stage(script: Path, root: Path, overrides: dict):
    """Launch a single-process stage via the current Python interpreter."""
    cmd = [sys.executable, str(script)]
    for k, v in overrides.items():
        cmd.append(f"{k}={v}")
    print(f"[RL] >> {' '.join(str(x) for x in cmd)}")
    subprocess.run(cmd, cwd=str(root), check=True)


# ---------------------------------------------------------------------------
# Main coordinator (single process, zero GPU / dist state)
# ---------------------------------------------------------------------------

def main():
    cfg = get_config()
    root = Path(__file__).resolve().parent

    # Resolve config path to absolute so subprocesses can always find it.
    raw_config_arg = OmegaConf.from_cli().config
    cfg_path = str(Path(str(raw_config_arg)).resolve())

    run_name = cfg.experiment.get("run_name", None)
    if run_name is None:
        run_name = cfg.experiment.get("wandb_run_name", None)
    if run_name is None:
        raise ValueError("Missing run name: set experiment.run_name or experiment.wandb_run_name in config.")
    run_name = str(run_name)

    start_round = int(cfg.experiment.current_round)
    total_round = int(cfg.experiment.total_round)
    eval_every = int(cfg.experiment.get("eval_every", 0))

    if start_round > total_round:
        raise ValueError(f"current_round({start_round}) > total_round({total_round})")

    # Accelerate config for training (may be relative to root or absolute).
    train_accelerate_config = str(cfg.accelerate.train_config)
    if not Path(train_accelerate_config).is_absolute():
        train_accelerate_config = str(root / train_accelerate_config)

    # Rollout uses torchrun — reuse the same process count as training.
    rollout_nproc = _get_num_processes(train_accelerate_config)

    # Stage script paths (absolute).
    use_ast_rollout = bool(cfg.rollout.get("use_ast_rollout", False))
    rollout_script_name = "rl_rollout_ast.py" if use_ast_rollout else "rl_rollout.py"
    rollout_script = root / "models" / "dream" / rollout_script_name
    if not rollout_script.exists():
        raise FileNotFoundError(f"rollout script not found: {rollout_script}")
    reward_script  = root / "reward" / "rewardmodel.py"
    train_script   = root / "train" / "rl_dream_train.py"

    # Data planning runs entirely in this coordinator process (no GPU needed).
    runtime_sampling_state = _init_runtime_sampling_state(cfg, run_name)
    subset_specs = _resolve_train_subset_specs(cfg, root)
    num_task_per_round = int(cfg.rollout.num_task_per_round)

    model_path = str(cfg.model.pretrained_model)
    resume_ckpt_path = cfg.training.get("resume_from_checkpoint", None)
    if resume_ckpt_path is not None:
        resume_ckpt_path = str(resume_ckpt_path)

    for round_i in range(start_round, total_round + 1):
        print(f"\n[RL] ===== Round {round_i}/{total_round} =====")
        print(f"[RL] model_path={model_path}")
        print(f"[RL] rollout_script={rollout_script_name}")

        round_plan = _plan_round_train_data(
            cfg=cfg,
            run_name=run_name,
            round_i=round_i,
            subset_specs=subset_specs,
            num_task_per_round=num_task_per_round,
            runtime_state=runtime_sampling_state,
        )
        try:
            _save_sampling_state(cfg, run_name, runtime_sampling_state)
        except Exception as e:
            print(f"[RL][warn] failed to write sampling state log: {e}")

        print(
            "[RL] round dataset: "
            f"subset={round_plan['subset_name']} "
            f"start={round_plan['start_idx']} "
            f"count={round_plan['count']} "
            f"max_new_tokens={round_plan['max_new_tokens']} "
            f"steps={round_plan['steps']} "
            f"shrink={round_plan['shrink']}"
        )

        # 1) Rollout — torchrun spawns independent per-GPU processes (no NCCL/DeepSpeed).
        _run_rollout_stage(
            script=rollout_script,
            nproc=rollout_nproc,
            root=root,
            overrides={
                "config": cfg_path,
                "experiment.current_round": round_i,
                "experiment.function": "train",
                "model.pretrained_model": model_path,
                "dataset.train_dataset": str(round_plan["chunk_path"]),
                "rollout.max_new_tokens": round_plan["max_new_tokens"],
                "rollout.steps": round_plan["steps"],
            },
        )

        # 2) Reward — single-process, CPU-only.
        _run_single_stage(
            script=reward_script,
            root=root,
            overrides={
                "config": cfg_path,
                "experiment.current_round": round_i,
                "experiment.function": "train",
                "model.pretrained_model": model_path,
                "dataset.train_dataset": str(round_plan["chunk_path"]),
            },
        )

        # 3) Train — DeepSpeed ZeRO-3 subprocess via accelerate; exits with full GPU cleanup.
        train_overrides = {
            "config": cfg_path,
            "experiment.current_round": round_i,
            "experiment.function": "train",
            "model.pretrained_model": model_path,
            "training.shrink": round_plan["shrink"],
        }
        if resume_ckpt_path is not None:
            train_overrides["training.resume_from_checkpoint"] = resume_ckpt_path
        _run_train_stage(
            script=train_script,
            accelerate_config=train_accelerate_config,
            root=root,
            overrides=train_overrides,
        )

        # Evaluation should use the model produced by the just-finished training round.
        round_ckpt = root / "projects" / str(cfg.experiment.project) / run_name / "ckpt" / f"round_{round_i}"
        round_ckpt_str = str(round_ckpt)

        # 4) Evaluation rollout + reward (optional).
        if eval_every > 0 and (round_i % eval_every == 0):
            _run_rollout_stage(
                script=rollout_script,
                nproc=rollout_nproc,
                root=root,
                overrides={
                    "config": cfg_path,
                    "experiment.current_round": round_i,
                    "experiment.function": "evaluation",
                    "model.pretrained_model": round_ckpt_str,
                },
            )
            _run_single_stage(
                script=reward_script,
                root=root,
                overrides={
                    "config": cfg_path,
                    "experiment.current_round": round_i,
                    "experiment.function": "evaluation",
                    "model.pretrained_model": round_ckpt_str,
                },
            )

        # Next round uses this round's saved checkpoint.
        model_path = round_ckpt_str
        resume_ckpt_path = round_ckpt_str

        print(f"[RL] round {round_i} done, next model_path={model_path}")

    print("[RL] all rounds completed.")


if __name__ == "__main__":
    main()
