import hashlib
import json
import os
import time
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from models import DreamModel, DreamTokenizer
from models.dream.generation_utils import DreamGenerationMixin


SRC_CODE_PROMPTS_TEMPLATE = (
    "Below is a program. Optimize the program and provide a faster version.\n"
    "Program:\n```python\n{{src_code}}\n```"
)
TGT_CODE_RESPONSE_PREFIX = "Here is the optimized code:\n```python\n"


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def _get_wandb_run_id(project_dir: Path) -> str:
    """Return a stable wandb run_id for this experiment, persisted to disk."""
    run_id_file = project_dir / "wandb_run_id.txt"
    if run_id_file.exists():
        return run_id_file.read_text().strip()
    try:
        import wandb as _w
        run_id = _w.util.generate_id()
    except Exception:
        run_id = hashlib.sha1(str(project_dir).encode()).hexdigest()[:16]
    run_id_file.parent.mkdir(parents=True, exist_ok=True)
    run_id_file.write_text(run_id)
    return run_id


def _log_rollout_to_wandb(cfg, root: Path, run_name: str, current_round: int,
                          final_path: Path, elapsed: float):
    """Log rollout statistics to wandb from rank 0 (best-effort)."""
    try:
        import wandb
    except ImportError:
        return
    try:
        project_name = str(cfg.experiment.project)
        function     = str(cfg.experiment.get("function", "train"))

        records = []
        with open(final_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not records:
            return

        n           = len(records)
        gen_lengths = [len(r.get("generated_token_ids", [])) for r in records]
        avg_gen_len = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0.0
        prompt_lens = [len(r.get("prompt_ids", [])) for r in records]
        avg_prompt_len = sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0.0

        prefix   = "eval/rollout" if function != "train" else "rollout/train"
        log_dict = {
            f"{prefix}_num_samples":      n,
            f"{prefix}_avg_gen_length":   avg_gen_len,
            f"{prefix}_avg_prompt_length": avg_prompt_len,
            f"{prefix}_elapsed_seconds":  elapsed,
        }

        project_dir = root / "projects" / project_name / run_name
        run_id      = _get_wandb_run_id(project_dir)

        wrun = wandb.init(
            project=project_name,
            name=run_name,
            id=run_id,
            resume="allow",
            entity=cfg.wandb.get("entity", None),
        )
        wrun.log(log_dict, step=current_round)
        wrun.finish()

        print(
            f"[rollout] wandb logged {prefix}: "
            f"n={n}, avg_gen_len={avg_gen_len:.1f}, elapsed={elapsed:.1f}s"
        )
    except Exception as exc:
        print(f"[rollout] wandb logging failed (non-fatal): {exc}")


def extract_code(text: str) -> str:
    start_tag = "```python"
    text = "```python" + text #根据prompt改的，前面prompt如果已经给了```python`，这里就加上，防止找不到
    start = text.find(start_tag)
    if start == -1:
        print(f"Could not find ```python block in output")
        return ""
    start += len(start_tag)
    end = text.find("```", start)
    if end == -1:
        print(f"Could not find ``` block in output")
        return text[start:].strip()
    return text[start:end].strip()


def resolve_dataset_path(root: Path, name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    for rel in [
        Path("data") / f"{name_or_path}.json",
        Path("data") / f"{name_or_path}.jsonl",
        Path("data") / name_or_path,
    ]:
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot locate dataset: {name_or_path}")


def load_records(dataset_path: Path):
    if dataset_path.suffix == ".jsonl":
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported dataset format; expect list or jsonl")


def build_prompt(record: dict) -> str:
    src = record.get("input") or record.get("src_code") or record.get("question") or ""
    return SRC_CODE_PROMPTS_TEMPLATE.replace("{{src_code}}", src)


def build_chat_ids(tokenizer, prompt: str, use_rsp_prefix: bool = True):
    chat = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, add_special_tokens=False)
    if use_rsp_prefix:
        ids += tokenizer(TGT_CODE_RESPONSE_PREFIX, add_special_tokens=False).input_ids
    return ids


def get_dataset_name(cfg):
    if cfg.experiment.function == "train":
        train_dataset = cfg.dataset.get("train_dataset", None)
        if train_dataset is None:
            raise ValueError("Missing dataset.train_dataset for train rollout.")
        return train_dataset
    validation_dataset = cfg.dataset.get("validation_data", None)
    if validation_dataset is None:
        raise ValueError("Missing dataset.validation_data for evaluation rollout.")
    return validation_dataset


def _sanitize_name_part(s: str) -> str:
    return str(s).replace("/", ".").replace("\\", ".")


def get_outputs_name(model_path: str, config_path: str):
    return f"{_sanitize_name_part(model_path)}-{_sanitize_name_part(config_path)}"


def get_outputs_filename(model_path: str, config_path: str, current_round: int) -> str:
    outputs_name = get_outputs_name(model_path, config_path)
    return f"round_{current_round}-outputs-{outputs_name}.jsonl"


def get_outputs_dirname(function_name: str) -> str:
    if function_name == "evaluation":
        return "eval_rollouts"
    return "rollouts"


def get_run_name(cfg):
    run_name = cfg.experiment.get("run_name", None)
    if run_name is None:
        run_name = cfg.experiment.get("wandb_run_name", None)
    if run_name is None:
        raise ValueError("Missing run name: set experiment.run_name or experiment.wandb_run_name in config.")
    return str(run_name)


def resolve_model_path(cfg, root: Path, project_name: str, run_name: str):
    return str(cfg.model.pretrained_model)
    return str(prev_ckpt)


def get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def get_rank_output_path(out_path: Path, rank: int, world_size: int) -> Path:
    return out_path.with_name(f"{out_path.stem}.rank{rank:02d}-of-{world_size:02d}{out_path.suffix}")


def get_rank_done_path(rank_path: Path) -> Path:
    """Sentinel file written by each rank after fully flushing its shard."""
    return rank_path.with_suffix(".done")


def signal_rank_done(rank_path: Path):
    """Called by each rank after closing its shard file."""
    get_rank_done_path(rank_path).touch()


def merge_rank_outputs(out_path: Path, world_size: int, timeout_sec: int = 7200, poll_sec: float = 2.0):
    rank_paths = [get_rank_output_path(out_path, r, world_size) for r in range(world_size)]
    done_paths = [get_rank_done_path(p) for p in rank_paths]
    deadline = time.time() + timeout_sec
    # Wait for .done sentinels, not just file existence, to avoid reading partial shards.
    while time.time() < deadline:
        if all(p.exists() for p in done_paths):
            break
        time.sleep(poll_sec)
    else:
        missing = [str(rank_paths[i]) for i, p in enumerate(done_paths) if not p.exists()]
        raise TimeoutError(f"Timed out waiting for rollout shards (.done). Missing: {missing}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for p in rank_paths:
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else (line + "\n"))

    for p in done_paths:
        try:
            p.unlink()
        except OSError:
            pass


def write_jsonl_record(fp, record: dict):
    fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    fp.write("\n")


def validate_single_node(cfg):
    num_node = int(cfg.experiment.get("num_node", 1))
    node_index = int(cfg.experiment.get("node_index", 0))
    if num_node != 1:
        raise ValueError(
            f"This rollout currently supports single-node only, but experiment.num_node={num_node}."
        )
    if node_index != 0:
        raise ValueError(
            f"For single-node mode, experiment.node_index must be 0, but got {node_index}."
        )


def run(cfg):
    validate_single_node(cfg)
    world_size, rank, local_rank = get_dist_info()
    root = Path(__file__).resolve().parents[2]
    project_name = cfg.experiment.project
    run_name = get_run_name(cfg)
    current_round = int(cfg.experiment.current_round)
    config_path = str(cfg.config)

    model_path = resolve_model_path(cfg, root, project_name, run_name)

    if cfg.experiment.function == "train":
        dataset_name = get_dataset_name(cfg)
        phase_cfg = cfg.rollout
    elif cfg.experiment.function == "evaluation":
        dataset_name = get_dataset_name(cfg)
        phase_cfg = cfg.evaluation
    else:
        raise ValueError(f"Unsupported experiment.function: {cfg.experiment.function}")

    num_resp = int(phase_cfg.get("num_response_per_task", 1))
    steps = int(phase_cfg.get("steps", 256))
    temperature = float(phase_cfg.get("temperature", 0.0))
    top_p = phase_cfg.get("top_p", 0.9)
    top_p = float(top_p) if top_p is not None else None
    top_k = phase_cfg.get("top_k", None)
    max_new_tokens = int(phase_cfg.get("max_new_tokens", 1024))
    alg = str(phase_cfg.get("alg", "entropy"))
    alg_temp = float(phase_cfg.get("alg_temp", 0.1))
    threshold = phase_cfg.get("threshold", None)
    threshold = float(threshold) if threshold is not None else None
    use_rsp_prefix = bool(phase_cfg.get("use_rsp_prefix", True))
    output_history = bool(phase_cfg.get("output_history", True))
    return_dict_in_generate = True

    if rank == 0:
        print(
            f"[rollout] mode={cfg.experiment.function} "
            f"max_new_tokens={max_new_tokens} steps={steps} num_resp={num_resp}"
        )

    dataset_path = resolve_dataset_path(root, dataset_name)
    data = load_records(dataset_path)

    

    if cfg.experiment.function == "train":
        num_task_per_round = int(cfg.rollout.num_task_per_round)
        pick = min(len(data), num_task_per_round)
        if pick > 0:
            # Keep deterministic order; round-level non-overlap is managed by rl.py.
            data = data[:pick]
    elif cfg.experiment.function == "evaluation":
        dataset_limit = phase_cfg.get("dataset_limit", None)
        if dataset_limit is not None:
            dataset_limit = int(dataset_limit)
            if dataset_limit < 0:
                raise ValueError(f"evaluation.dataset_limit must be >= 0, got {dataset_limit}")
            data = data[:dataset_limit]
    else:
        raise ValueError(f"Unsupported experiment.function: {cfg.experiment.function}")

    if world_size > 1:
        data = data[rank::world_size]
        print(f"[rollout] rank {rank}/{world_size} processes {len(data)} samples")

    tokenizer = DreamTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    model = DreamModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    out_dir = root / "projects" / project_name / run_name / get_outputs_dirname(str(cfg.experiment.function))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / get_outputs_filename(model_path, config_path, current_round)
    write_path = out_path if world_size == 1 else get_rank_output_path(out_path, rank, world_size)

    rollout_start = time.time()
    with open(write_path, "w", encoding="utf-8") as fout:
        for rec in tqdm(data, desc=f"dream rollout (rank {rank})"):
            prompt = build_prompt(rec)
            prompt_ids = build_chat_ids(tokenizer, prompt, use_rsp_prefix=use_rsp_prefix)
            input_ids_single = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask_single = torch.ones_like(input_ids_single)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "output_history": output_history,
                "return_dict_in_generate": return_dict_in_generate,
                "steps": steps,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "alg": alg,
                "alg_temp": alg_temp,
                "threshold": threshold,
                "tokenizer": tokenizer,
            }

            # confidence_threshold in generation_utils currently enforces batch size == 1.
            if alg == "confidence_threshold":
                batch_input_ids = input_ids_single
                batch_attention_mask = attention_mask_single
                batch_repeats = num_resp
            else:
                batch_input_ids = input_ids_single.repeat(num_resp, 1)
                batch_attention_mask = attention_mask_single.repeat(num_resp, 1)
                batch_repeats = 1

            for _ in range(batch_repeats):
                with torch.no_grad():
                    out = model.diffusion_generate(batch_input_ids, attention_mask=batch_attention_mask, **gen_kwargs)

                seq = out.sequences if hasattr(out, "sequences") else out
                full_step_map = out.step_map if (output_history and hasattr(out, "step_map")) else None
                if output_history and full_step_map is None:
                    raise RuntimeError(
                        "diffusion_generate must return step_map when return_dict_in_generate=True and output_history=True"
                    )

                for b in range(seq.shape[0]):
                    seq_ids = seq[b].detach().cpu().tolist()
                    gen_token_ids = seq[b, len(prompt_ids):].detach().cpu().tolist()
                    generated = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
                    if full_step_map is not None:
                        sm = full_step_map[b, len(prompt_ids):].detach().cpu().tolist()
                    else:
                        sm = []

                    ext = extract_code(generated)
                    gen_len = len(gen_token_ids)

                    if full_step_map is not None:
                        assert len(sm) == gen_len, f"Step map length {len(sm)} does not match generated token length {gen_len}"

                    # if len(sm) < gen_len:
                    #     sm += [max(sm) + 1 if sm else 0] * (gen_len - len(sm))
                    # elif len(sm) > gen_len:
                    #     sm = sm[:gen_len]

                    row = {
                        "problem_id": rec.get("problem_id", ""),
                        "input": rec.get("input", ""),
                        "target": rec.get("target", ""),
                        "prompt": prompt,
                        "prompt_ids": prompt_ids,
                        "sequence_ids": seq_ids,
                        "generated_token_ids": gen_token_ids,
                        "generated_output": generated,
                        "step_map": sm,
                        "extracted_output": ext,
                    }
                    write_jsonl_record(fout, row)

    print(f"Saved rollout shard to: {write_path}")

    if world_size > 1:
        signal_rank_done(write_path)
        if rank == 0:
            merge_rank_outputs(out_path, world_size)
            print(f"Merged rollout outputs to: {out_path}")



def main():
    cfg = get_config()
    run(cfg)


if __name__ == "__main__":
    main()
