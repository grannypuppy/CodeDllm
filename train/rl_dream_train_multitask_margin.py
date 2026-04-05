"""RL training for dream_multitask: dual GRPO objectives.

Two policy heads are trained jointly:
  1. **LM head (lm_head)**: same TraceRL/GRPO as rl_dream_train.py – PPO-clip
     surrogate on token log-probabilities at positions unmasked at each step.
  2. **Rank head (rank_head)**: GRPO on the raw rank values at code positions that
     were weighted by the rank head during rollout.  The "old policy" rank values
     are the ones saved in the rollout JSON by rl_rollout_ast.py.

Loss:
    total = lm_grpo_loss + rank_loss_weight * rank_grpo_loss
            + (beta * kl_seq_sum / batch  if beta > 0)

Both LM and rank GRPO use the same per-sequence advantage (from grouped GRPO
computed by rewardmodel.py).

Usage:
    accelerate launch --config_file accelerate_configs/... \
        train/rl_dream_train_multitask.py \
        config=configs/rl_dream_multitask.yaml [overrides...]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import gc
import json
import logging
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from models.dream_multitask import DreamConfig, DreamModel, DreamTokenizer
from models.logging import set_verbosity_error, set_verbosity_info
from models.lr_schedulers import get_scheduler
from train.utils import AverageMeter, flatten_omega_conf, get_config

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

def _get_run_name(cfg) -> str:
    run_name = cfg.experiment.get("run_name", None)
    if run_name is None:
        run_name = cfg.experiment.get("wandb_run_name", None)
    if run_name is None:
        raise ValueError("Missing run name: set experiment.run_name or experiment.wandb_run_name in config.")
    return str(run_name)


def _sanitize_name_part(s: str) -> str:
    return str(s).replace("/", ".").replace("\\", ".")


def _short_outputs_stem(model_path: str, config_path: str) -> str:
    """Must match models/dream_multitask/rl_rollout_ast.get_outputs_filename stem."""
    return f"{_sanitize_name_part(Path(str(model_path)).name)}-{_sanitize_name_part(Path(str(config_path)).name)}"


def _get_wandb_run_id(project_dir: Path) -> str:
    run_id_file = project_dir / "wandb_run_id.txt"
    if run_id_file.exists():
        return run_id_file.read_text().strip()
    run_id = wandb.util.generate_id()
    run_id_file.parent.mkdir(parents=True, exist_ok=True)
    run_id_file.write_text(run_id)
    return run_id


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrainDataset(Dataset):
    """Holds one "training view" per diffusion step per generated sequence.

    Additional fields vs. the standard dream RL dataset:
      - ``rank_step_data``: List[dict | None] – for each training sample,
        contains ``{"positions": [...], "rank_values_old": [...]}`` with
        absolute (left-padded) positions and the rank values from rollout.
        ``None`` or empty ``positions`` means no rank loss for that sample.
    """

    def __init__(self, inputs, labels, pmasks, advantages, start_pos, rank_step_data):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks
        self.advantages = advantages
        self.start_pos = start_pos
        self.rank_step_data = rank_step_data  # List[dict | None]
        self.logp_old_tok = torch.full(
            (len(inputs), labels.shape[1] - 1), float("-inf"), dtype=torch.float32
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            idx,
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx],
            self.advantages[idx],
            self.start_pos[idx],
            self.rank_step_data[idx],
        )


def simple_collate(batch):
    idx, inp, lbl, msk, adv, stp, rsd = zip(*batch)
    input_ids = torch.stack(inp)
    B, T = input_ids.shape
    rank_mask = torch.zeros(B, T, dtype=torch.bool)
    rv_old_dense = torch.zeros(B, T, dtype=torch.float32)
    for b, item in enumerate(rsd):
        if item is None or not item.get("positions"):
            continue
        positions = item["positions"]
        vals = item["rank_values_old"]
        for p, v in zip(positions, vals):
            pi = int(p)
            if 0 <= pi < T:
                rank_mask[b, pi] = True
                rv_old_dense[b, pi] = float(v)
    return {
        "ids": torch.tensor(idx),
        "input_ids": input_ids,
        "labels": torch.stack(lbl),
        "p_mask_lm": torch.stack(msk),
        "advantage": list(adv),
        "start_pos": torch.tensor(stp, dtype=torch.long),
        "rank_mask": rank_mask,
        "rv_old_dense": rv_old_dense,
    }


# ---------------------------------------------------------------------------
# Attention bias
# ---------------------------------------------------------------------------

def make_left_pad_attention_bias(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Additive attention bias that masks left-padding keys with -inf."""
    if pad_id is None:
        raise ValueError("pad_id is None; cannot build attention bias for left padding.")
    pad_keys = input_ids.eq(int(pad_id))
    bias = torch.zeros(
        input_ids.shape[0], 1, 1, input_ids.shape[1],
        device=input_ids.device, dtype=torch.float32,
    )
    bias.masked_fill_(pad_keys[:, None, None, :], float("-inf"))
    return bias


# ---------------------------------------------------------------------------
# step_map helpers (same as rl_dream_train.py)
# ---------------------------------------------------------------------------

def collapse_k_unique(lst, k: int):
    if k <= 1:
        return lst
    uniq = sorted(set(lst))
    if not uniq:
        return lst
    groups = (len(uniq) + k - 1) // k
    mapping = {}
    for g in range(groups):
        st = g * k
        ed = min((g + 1) * k - 1, len(uniq) - 1)
        rep = uniq[ed]
        for val in uniq[st: ed + 1]:
            mapping[val] = rep
    return [mapping[x] for x in lst]


def _sanitize_step_map(step_map, gen_len):
    sm = [int(x) for x in step_map[:gen_len]]
    if len(sm) < gen_len:
        fill = max(sm) + 1 if sm else 0
        sm += [fill] * (gen_len - len(sm))
    return sm


# ---------------------------------------------------------------------------
# Old-policy LM log-prob (same as rl_dream_train.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_logp_old_tok_parallel(accelerator, model, dataset, dataloader, pad_id: int):
    model.eval()
    for batch in dataloader:
        ids = batch["ids"]
        input_ids = batch["input_ids"].to(accelerator.device)
        labels = batch["labels"].to(accelerator.device)
        attn_bias = make_left_pad_attention_bias(input_ids, pad_id)
        outputs = model(input_ids, attention_mask=attn_bias, is_causal=False, return_dict=True)
        logits = outputs.logits
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        tok_lp = logp.gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
        dataset.logp_old_tok[ids] = tok_lp.float().cpu()
    accelerator.wait_for_everyone()
    model.train()


# ---------------------------------------------------------------------------
# Load rollout records
# ---------------------------------------------------------------------------

def resolve_rollout_outputs_path(cfg, root: Path, run_name: str, current_round: int) -> Path:
    model_path = str(cfg.model.pretrained_model)
    config_path = str(cfg.config)
    roll_dir = root / "projects" / cfg.experiment.project / run_name / "rollouts"
    short_file = f"round_{current_round}-outputs-{_short_outputs_stem(model_path, config_path)}.jsonl"
    p = roll_dir / short_file
    if p.exists():
        return p
    legacy_file = f"round_{current_round}-outputs-{_sanitize_name_part(model_path)}-{_sanitize_name_part(config_path)}.jsonl"
    p_legacy = roll_dir / legacy_file
    if p_legacy.exists():
        return p_legacy
    return p


def load_training_records(cfg, root: Path, run_name: str, current_round: int):
    outputs_path = resolve_rollout_outputs_path(cfg, root, run_name, current_round)
    if not outputs_path.exists():
        raise ValueError(f"Missing rollout outputs jsonl: {outputs_path}")

    records = []
    raw_rows = 0
    skipped_zero_adv = 0
    zero_adv_rows = 0  # rows with adv==0 in file (independent of beta / filtering)
    filter_zero_adv = float(cfg.training.get("beta", 0.0)) == 0.0
    with open(outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_rows += 1
            row = json.loads(line)
            if "advantage" not in row:
                raise ValueError(
                    f"Missing `advantage` in rollout output row from {outputs_path}. "
                    "Run rewardmodel.py with grouped GRPO advantage writing enabled."
                )
            if not all(k in row for k in ("prompt_ids", "sequence_ids", "generated_token_ids", "step_map")):
                continue
            advantage = float(row["advantage"])
            if advantage == 0.0:
                zero_adv_rows += 1
            if filter_zero_adv and advantage == 0.0:
                skipped_zero_adv += 1
                continue
            records.append({
                "prompt_ids": row["prompt_ids"],
                "sequence_ids": row["sequence_ids"],
                "generated_token_ids": row["generated_token_ids"],
                "step_map": row.get("step_map", []),
                "advantage": advantage,
                # rank_step_records may be absent in legacy files; default to empty
                "rank_step_records": row.get("rank_step_records", {}),
            })

    if not records:
        raise ValueError(
            "No trainable token-id records found in outputs jsonl. "
            f"raw_rows={raw_rows}, skipped_zero_adv={skipped_zero_adv}, "
            f"beta={float(cfg.training.get('beta', 0.0))}."
        )
    logger.info(
        f"Loaded multitask RL records: path={outputs_path} "
        f"raw_rows={raw_rows} kept={len(records)} skipped_zero_adv={skipped_zero_adv} "
        f"zero_adv_rows={zero_adv_rows}"
    )
    return records, raw_rows, skipped_zero_adv, zero_adv_rows


# ---------------------------------------------------------------------------
# Build training pairs
# ---------------------------------------------------------------------------

def prepare_inputs_and_labels_from_token_ids(cfg, mask_id, pad_id, records):
    """Convert rollout records into per-step training pairs.

    For each record we produce one training sample per unique diffusion step
    (TraceRL masking, shrink=1 recommended for multitask).

    Returns:
        noisy_batch, labels_lm, p_mask, advantage_list, start_pos_tensor,
        rank_step_data_list
    where rank_step_data_list[i] = {"positions": [...], "rank_values_old": [...]}
    with positions already adjusted for the left-padded batch layout.
    """
    lower = float(cfg.training.lower_p)
    upper = float(cfg.training.upper_p)

    processed = []
    for row in records:
        prompt_ids = row.get("prompt_ids")
        sequence_ids = row.get("sequence_ids")
        if prompt_ids is None or sequence_ids is None:
            continue
        prompt_ids = [int(x) for x in prompt_ids]
        sequence_ids = [int(x) for x in sequence_ids]
        start_pos = len(prompt_ids)
        if start_pos <= 0 or len(sequence_ids) <= start_pos:
            continue
        gen_len = len(sequence_ids) - start_pos
        step_map = _sanitize_step_map(row.get("step_map", []), gen_len)

        processed.append({
            "prompt_ids": prompt_ids,
            "generated_ids": sequence_ids[start_pos:],
            "prompt_len": int(start_pos),
            "gen_len": int(gen_len),
            "step_map": step_map,
            "advantage": float(row["advantage"]),
            "rank_step_records": row.get("rank_step_records", {}),
        })

    if not processed:
        raise ValueError(
            "No valid token-id records found. Ensure reward stage outputs rows with "
            "prompt_ids/sequence_ids/step_map/advantage."
        )

    max_prompt_len = max(int(x["prompt_len"]) for x in processed)
    max_gen_len = max(int(x["gen_len"]) for x in processed)
    seq_len = int(max_prompt_len + max_gen_len)

    logger.info(
        f"Left-padding prompts: records={len(processed)} "
        f"max_prompt_len={max_prompt_len} max_gen_len={max_gen_len} seq_len={seq_len}"
    )

    # Build aligned sequences (left-pad prompt, right-pad gen)
    aligned = []
    for item in processed:
        prompt_len = int(item["prompt_len"])
        gen_len = int(item["gen_len"])
        prompt_left_pad = max_prompt_len - prompt_len

        base_ids = torch.full((seq_len,), int(pad_id), dtype=torch.long)
        base_ids[prompt_left_pad: prompt_left_pad + prompt_len] = torch.tensor(item["prompt_ids"], dtype=torch.long)
        gen_begin = max_prompt_len
        if gen_len > 0:
            base_ids[gen_begin: gen_begin + gen_len] = torch.tensor(item["generated_ids"], dtype=torch.long)

        aligned.append({
            "input_ids": base_ids,
            "start_pos": int(max_prompt_len),
            "gen_len": gen_len,
            "step_map": item["step_map"],
            "advantage": float(item["advantage"]),
            "rank_step_records": item["rank_step_records"],
            "prompt_left_pad": prompt_left_pad,
        })

    noisy_list, label_list, pmask_list, advantage_list, start_pos_list = [], [], [], [], []
    rank_step_data_list = []

    if cfg.training.method == "TraceRL":
        for item in aligned:
            base_ids = item["input_ids"]
            labels = base_ids.clone()
            start_pos = item["start_pos"]
            gen_len = int(item["gen_len"])
            if gen_len <= 0:
                continue
            step_map_valid = [int(x) for x in item["step_map"][:gen_len]]
            order_list = collapse_k_unique(step_map_valid, int(cfg.training.shrink))
            order = torch.as_tensor(order_list, dtype=torch.long)
            uniq_steps = torch.unique(order, sorted=True)
            rank_recs = item["rank_step_records"]  # dict str(step) -> {gen_positions, rank_values_old}

            for step_val in uniq_steps.tolist():
                tgt_gen_mask = order == step_val
                tgt_mask = torch.zeros(seq_len, dtype=torch.bool)
                tgt_mask[start_pos: start_pos + gen_len] = tgt_gen_mask
                if not tgt_mask.any():
                    continue
                noisy_ids = base_ids.clone()
                suffix = noisy_ids[start_pos: start_pos + gen_len]
                suffix[order >= step_val] = mask_id
                noisy_list.append(noisy_ids)
                label_list.append(labels)
                pmask_list.append(tgt_mask)
                advantage_list.append(float(item["advantage"]))
                start_pos_list.append(start_pos)

                # Build rank step data for this training pair
                step_key = str(int(step_val))
                rsd_raw = rank_recs.get(step_key, None)
                if rsd_raw is not None and rsd_raw.get("gen_positions"):
                    # Convert gen_positions (0-indexed into generated sequence) to
                    # absolute positions in the left-padded training tensor.
                    prompt_left_pad = item["prompt_left_pad"]
                    prompt_len_orig = int(item["start_pos"])  # = max_prompt_len
                    # gen_position gp -> training position = max_prompt_len + gp
                    training_positions = [int(max_prompt_len + gp) for gp in rsd_raw["gen_positions"]]
                    rank_step_data_list.append({
                        "positions": training_positions,
                        "rank_values_old": [float(v) for v in rsd_raw["rank_values_old"]],
                    })
                else:
                    rank_step_data_list.append({"positions": [], "rank_values_old": []})

    elif cfg.training.method == "random_masking":
        mask_times = int(cfg.training.mask_times_per_sample)
        for item in aligned:
            base_ids = item["input_ids"]
            labels = base_ids.clone()
            start_pos = item["start_pos"]
            gen_len = int(item["gen_len"])
            if gen_len <= 0:
                continue
            for _ in range(mask_times):
                t = (upper - lower) * torch.rand(1) + lower
                rand_gen = torch.rand(gen_len) < t
                if not rand_gen.any():
                    continue
                noisy_ids = base_ids.clone()
                noisy_ids[start_pos: start_pos + gen_len][rand_gen] = mask_id
                rand_mask = torch.zeros(seq_len, dtype=torch.bool)
                rand_mask[start_pos: start_pos + gen_len] = rand_gen
                noisy_list.append(noisy_ids)
                label_list.append(labels)
                pmask_list.append(rand_mask)
                advantage_list.append(float(item["advantage"]))
                start_pos_list.append(start_pos)
                rank_step_data_list.append({"positions": [], "rank_values_old": []})
    else:
        raise ValueError(f"Unsupported training.method: {cfg.training.method}")

    if not noisy_list:
        raise ValueError("Masking produced zero training samples.")

    noisy_batch = torch.stack(noisy_list)
    labels_lm = torch.stack(label_list)
    p_mask = torch.stack(pmask_list)
    start_pos_tensor = torch.tensor(start_pos_list, dtype=torch.long)

    valid_rows = p_mask.any(dim=1)
    noisy_batch = noisy_batch[valid_rows]
    labels_lm = labels_lm[valid_rows]
    p_mask = p_mask[valid_rows]
    start_pos_tensor = start_pos_tensor[valid_rows]
    keep_idx = torch.where(valid_rows)[0].tolist()
    advantage_list = [advantage_list[i] for i in keep_idx]
    rank_step_data_list = [rank_step_data_list[i] for i in keep_idx]

    return noisy_batch, labels_lm, p_mask, advantage_list, start_pos_tensor, rank_step_data_list


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_checkpoint(model, tokenizer, cfg, accelerator, ckpt_name, global_step=None, epoch=None, resumable=True):
    run_name = _get_run_name(cfg)
    output_dir = Path(__file__).resolve().parents[1] / "projects" / cfg.experiment.project / run_name / "ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_total_limit = cfg.training.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1]) if "-" in p.name else 0,
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / ckpt_name
    save_base.mkdir(parents=True, exist_ok=True)
    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        model_to_save.save_pretrained(
            save_base,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_base))
        metadata = {"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if global_step is not None:
            metadata["global_step"] = int(global_step)
        if epoch is not None:
            metadata["epoch"] = int(epoch)
        with (save_base / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        trainer_state = {}
        if global_step is not None:
            trainer_state["global_step"] = int(global_step)
        if epoch is not None:
            trainer_state["epoch"] = int(epoch)
        if trainer_state:
            with (save_base / "trainer_state.json").open("w", encoding="utf-8") as f:
                json.dump(trainer_state, f, indent=2)
        logger.info(f"Saved model + tokenizer to {save_base}" + (" (resumable)" if resumable else " (lightweight)"))

    accelerator.wait_for_everyone()
    if resumable:
        accelerator.save_state(str(save_base))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run(cfg):
    run_name = _get_run_name(cfg)

    num_node = int(cfg.experiment.get("num_node", 1))
    node_index = int(cfg.experiment.get("node_index", 0))
    if num_node != 1:
        raise ValueError(
            f"rl_dream_train_multitask.py supports single-node only, but experiment.num_node={num_node}."
        )
    if node_index != 0:
        raise ValueError(
            f"For single-node mode, experiment.node_index must be 0, but got {node_index}."
        )

    project_name = cfg.experiment.project
    cur_round = int(cfg.experiment.current_round)
    root = Path(__file__).resolve().parents[1]
    pretrained_model = str(cfg.model.pretrained_model)

    resume_from_checkpoint = cfg.training.get("resume_from_checkpoint", None)
    if resume_from_checkpoint:
        resume_from_checkpoint = Path(str(resume_from_checkpoint)).resolve()
        if not resume_from_checkpoint.is_dir():
            raise FileNotFoundError(
                f"resume_from_checkpoint not found or not a directory: {resume_from_checkpoint}"
            )
        load_path = str(resume_from_checkpoint)
        print(f"[train] Resuming RL multitask training from: {load_path}")
    else:
        load_path = pretrained_model

    if cfg.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    cfg.experiment.logging_dir = str(root / "projects" / project_name / run_name / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="wandb",
        project_dir=cfg.experiment.logging_dir,
        split_batches=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        project_dir = root / "projects" / project_name / run_name
        run_id = _get_wandb_run_id(project_dir)
        accelerator.init_trackers(
            project_name=str(project_name),
            config=dict(flatten_omega_conf(cfg, resolve=True)),
            init_kwargs={"wandb": {
                "name": run_name,
                "id": run_id,
                "resume": "allow",
                "entity": cfg.wandb.get("entity", None),
            }},
        )

    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    # ---- Model ----
    model_config = DreamConfig.from_pretrained(load_path)
    tokenizer = DreamTokenizer.from_pretrained(load_path)
    model = DreamModel.from_pretrained(load_path, config=model_config, torch_dtype=torch.bfloat16)
    model = model.to(accelerator.device)

    mask_id = model.config.mask_token_id
    pad_id = model.config.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("pad_token_id is required but is None.")

    # ---- Optimizer ----
    optimizer_cfg = cfg.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        grouped,
        lr=optimizer_cfg.learning_rate,
        betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
        eps=optimizer_cfg.epsilon,
    )

    # ---- Data ----
    records, _raw_rows, _skipped_zero_adv, _zero_adv_rows = load_training_records(cfg, root, run_name, cur_round)
    input_ids_t, labels_t, p_mask_t, advantages, start_pos_t, rank_step_data = \
        prepare_inputs_and_labels_from_token_ids(cfg, mask_id, pad_id, records)

    dataset_lm = TrainDataset(input_ids_t, labels_t, p_mask_t, advantages, start_pos_t, rank_step_data)

    total_batch = (
        cfg.training.batch_size_lm
        * accelerator.num_processes
        * cfg.training.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch)
    num_epochs = int(cfg.training.num_train_epochs)
    max_train_steps = num_update_steps_per_epoch * num_epochs + 1

    warmup_param = cfg.lr_scheduler.params.warmup_steps
    if isinstance(warmup_param, float) and 0 < warmup_param < 1:
        num_warmup_steps = int(max_train_steps * warmup_param)
    else:
        num_warmup_steps = int(warmup_param)

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=num_warmup_steps * accelerator.num_processes,
        min_lr_scale=cfg.lr_scheduler.params.min_lr_scale,
    )

    train_dl = DataLoader(
        dataset_lm,
        batch_size=cfg.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0,
    )

    model, optimizer, lr_scheduler, train_dl = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dl
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    first_epoch = 0
    global_step = 0
    if resume_from_checkpoint:
        accelerator.load_state(str(resume_from_checkpoint))
        trainer_state_path = Path(resume_from_checkpoint) / "trainer_state.json"
        if trainer_state_path.is_file():
            with trainer_state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            first_epoch = int(state.get("epoch", 0))
            global_step = int(state.get("global_step", 0))
        logger.info(f"Loaded trainer state: epoch={first_epoch}, global_step={global_step}")

    # ---- Old LM policy pass ----
    logger.info("Running old LM policy logp pass")
    compute_logp_old_tok_parallel(accelerator, model, dataset_lm, train_dl, pad_id)

    # ---- Loss weights ----
    rank_loss_weight = float(cfg.training.get("rank_loss_weight", 0.1))
    beta = float(cfg.training.get("beta", 0.0))

    logger.info("Running RL multitask optimization")

    # ---- Forward pass ----
    def forward_process(input_ids, labels, p_mask_lm, adv, logp_old_tok, rank_mask, rv_old_dense):
        """Compute total loss = LM GRPO + rank_loss_weight * rank GRPO + KL penalty.

        Args:
            input_ids: [B, T]
            labels: [B, T]
            p_mask_lm: [B, T] bool
            adv: list of floats, length B
            logp_old_tok: [B, T-1] old LM log-probs from the rollout model
            rank_mask: [B, T] bool — True where rollout saved rank supervision
            rv_old_dense: [B, T] float — rollout rank values at those positions (0 elsewhere)

        Returns:
            total_loss, lm_policy_loss_detached, kl_per_token_detached, rank_loss_detached
        """
        adv_t = torch.as_tensor(adv, device=input_ids.device, dtype=torch.float32).detach()
        attn_bias = make_left_pad_attention_bias(input_ids, pad_id)
        outputs = model(input_ids, attention_mask=attn_bias, is_causal=False, return_dict=True)
        logits = outputs.logits                    # [B, T, V]
        rank_values = outputs.rank_values.squeeze(-1)  # [B, T]

        batch = logits.shape[0]
        shift_logits = logits[:, :-1, :]           # [B, T-1, V]
        shift_labels = labels[:, 1:]               # [B, T-1]
        shift_mask = p_mask_lm[:, 1:]              # [B, T-1]

        # ---- LM GRPO (PPO-clip) ----
        logp_new_tok = (
            torch.log_softmax(shift_logits, dim=-1)
            .gather(-1, shift_labels.unsqueeze(-1))
            .squeeze(-1)
        )  # [B, T-1]

        ratio = torch.exp(logp_new_tok - logp_old_tok)
        clipped = torch.clamp(ratio, 1 - cfg.training.eps, 1 + cfg.training.eps)
        adv_tok = adv_t.unsqueeze(1)  # [B, 1]
        surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok) * shift_mask  # [B, T-1]
        num_mask = torch.clamp(shift_mask.sum(dim=1), min=1)
        lm_policy_loss = -(surrogate_tok.sum(dim=1) / num_mask).sum() / batch

        # ---- KL divergence (always computed for monitoring) ----
        log_ratio = logp_new_tok - logp_old_tok
        if cfg.training.use_kl_estimator_k3:
            kl_tok = (-log_ratio).exp() - 1.0 + log_ratio
        else:
            kl_tok = log_ratio
        num_tok = shift_mask.sum().clamp(min=1)
        kl_per_token = (kl_tok * shift_mask).sum() / num_tok

        # ---- Rank head Margin Pairwise Ranking Loss ----
        #
        # Unlike the old Gaussian GRPO (which treated rank values as continuous actions
        # and used PPO-clip to push rv_new → rv_old), this loss focuses on ORDERING:
        #
        # For each PAIR of positions (i, j) in the same sequence:
        #   dp_old = rv_old[i] - rv_old[j]   → old_rel = three_way(dp_old, eps)
        #   dp_new = rv_new[i] - rv_new[j]   → new_rel = three_way(dp_new, eps)
        #
        # The old SFT model's ordering (old_rel) serves as the "ground truth":
        #   - If old_rel = +1: we want dp_new > eps (i should have higher rank than j)
        #   - If old_rel = -1: we want dp_new < -eps (i should have lower rank than j)
        #   - If old_rel =  0: we want |dp_new| ≤ eps (i and j should have same rank)
        #
        # Advantage weighting:
        #   - adv > 0 (good result): reinforce old_rel ordering → push new toward old
        #   - adv < 0 (bad result): penalize old_rel ordering → push new away from old
        #
        # This is fundamentally different from Gaussian GRPO because:
        #   1. It only cares about ranking ORDER, not exact numerical values
        #   2. The "margin" structure tolerates noise in rv_old as long as ordering matches
        #   3. The PPO clipping mechanism is replaced by margin-based saturation

        def _three_way_relation(values, eps):
            """Map value differences to {-1, 0, +1}: +1 if > eps, -1 if < -eps, 0 otherwise."""
            rel = torch.zeros_like(values, dtype=torch.int8)
            rel[values > eps] = 1
            rel[values < -eps] = -1
            return rel

        rank_eps = float(cfg.training.get("rank_eps", 1e-4))
        rank_eps_untie_slack = float(cfg.training.get("rank_eps_untie_slack", 1e-6))
        margin_untie = rank_eps + rank_eps_untie_slack

        valid = rank_mask  # [B, T]
        rv_old = rv_old_dense.to(dtype=rank_values.dtype)
        rv_new = rank_values  # [B, T], rv_new is the current model's prediction

        batch_size = logits.shape[0]

        # Accumulate loss per sample, then weight by advantage.
        # IMPORTANT: for ZeRO-3 compatibility, every sample must keep its rank_values
        # in the autograd graph even when no rank supervision exists.  We use
        # torch.where + "* 0.0" trick (same as the original Gaussian GRPO) to ensure
        # backward passes through rank_head are triggered consistently on all ranks.
        sample_losses = []
        sample_has_rank = []

        for b in range(batch_size):
            valid_idx = torch.where(valid[b])[0]

            # ---- Compute actual margin pairwise loss ----
            # Start with a zero that IS connected to rank_values (for ZeRO-3 sync)
            dummy_zero = rank_values[b].sum() * 0.0  # gradient-connected zero

            if valid_idx.numel() < 2:
                # Cannot compute pairwise loss; use gradient-connected zero
                margin_loss_per_sample = dummy_zero
                sample_losses.append(margin_loss_per_sample)
                sample_has_rank.append(False)
                continue

            p_old = rv_old[b, valid_idx]  # [num_valid]
            p_new = rv_new[b, valid_idx]  # [num_valid]

            # Compute pairwise differences
            dp_old = p_old.unsqueeze(1) - p_old.unsqueeze(0)  # [num_valid, num_valid]
            dp_new = p_new.unsqueeze(1) - p_new.unsqueeze(0)  # [num_valid, num_valid]

            # Take upper triangle (avoid counting each pair twice)
            triu_mask = torch.triu(torch.ones_like(dp_old, dtype=torch.bool), diagonal=1)
            dp_old = dp_old[triu_mask]
            dp_new = dp_new[triu_mask]

            if dp_old.numel() == 0:
                # No pairs; use gradient-connected zero via torch.where
                margin_loss_per_sample = dummy_zero
                sample_losses.append(margin_loss_per_sample)
                sample_has_rank.append(False)
                continue

            # Classify relations
            old_rel = _three_way_relation(dp_old, rank_eps)  # [num_pairs]

            # ---- Compute margin pairwise loss (same structure as SFT pairwise_margin_rank_loss) ----
            # Ordered pairs (old_rel != 0): require sign(dp_new) matches old_rel with margin
            neq_mask = old_rel != 0
            loss_ordered = dummy_zero  # start with connected zero
            if neq_mask.any():
                y = old_rel[neq_mask].float()  # +1 or -1
                loss_ordered = F.relu(margin_untie - y * dp_new[neq_mask]).sum()

            # Tied pairs (old_rel == 0): require |dp_new| ≤ eps
            eq_mask = ~neq_mask
            loss_eq = dummy_zero  # start with connected zero
            if eq_mask.any():
                loss_eq = F.relu(torch.abs(dp_new[eq_mask]) - rank_eps).sum()

            num_pairs = dp_old.numel()
            margin_loss_per_sample = (loss_ordered + loss_eq) / max(num_pairs, 1)

            sample_losses.append(margin_loss_per_sample)
            sample_has_rank.append(True)

        # Stack: [B]
        sample_losses_t = torch.stack(sample_losses)
        sample_has_rank_t = torch.tensor(sample_has_rank, device=rank_values.device, dtype=torch.bool)

        # Weight by advantage: adv_t is [B], sample_losses_t is [B]
        #   adv > 0: reinforce old_rel ordering (push toward satisfying margin)
        #   adv < 0: penalize old_rel ordering (push away from satisfying margin)
        weighted_losses = sample_losses_t * adv_t  # [B], element-wise

        # Sum over samples that have rank supervision, then normalize.
        # Samples without rank supervision contribute zero to the weighted sum,
        # but their rank_values remain in the autograd graph via the dummy_zero trick.
        num_with_rank = sample_has_rank_t.float().sum().clamp(min=1.0)
        rank_policy_loss = -(weighted_losses * sample_has_rank_t.float()).sum() / num_with_rank

        # ---- Total loss ----
        total_loss = lm_policy_loss + rank_loss_weight * rank_policy_loss
        if beta > 0:
            kl_seq_sum = (kl_tok * shift_mask).sum(dim=1)
            total_loss = total_loss + beta * kl_seq_sum.sum() / batch

        return (
            total_loss,
            lm_policy_loss.detach(),
            kl_per_token.detach(),
            rank_policy_loss.detach(),
        )

    # ---- Training loop ----
    epoch = first_epoch
    for epoch in range(first_epoch, num_epochs):
        model.train()
        accumulated_loss = 0.0
        accumulated_lm_loss = 0.0
        accumulated_kl = 0.0
        accumulated_rank_loss = 0.0
        num_steps_in_epoch = len(train_dl)

        for step, batch in enumerate(train_dl, start=1):
            input_ids = batch["input_ids"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)
            old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)
            advantage = batch["advantage"]
            rank_mask = batch["rank_mask"].to(accelerator.device)
            rv_old_dense = batch["rv_old_dense"].to(accelerator.device)

            loss, lm_loss_val, kl_per_token_val, rank_loss_val = forward_process(
                input_ids=input_ids,
                labels=labels,
                p_mask_lm=p_mask_lm,
                adv=advantage,
                logp_old_tok=old_lp,
                rank_mask=rank_mask,
                rv_old_dense=rv_old_dense,
            )
            accumulated_loss += loss.item() / accelerator.gradient_accumulation_steps
            accumulated_lm_loss += lm_loss_val.item() / accelerator.gradient_accumulation_steps
            accumulated_kl += kl_per_token_val.item() / accelerator.gradient_accumulation_steps
            accumulated_rank_loss += rank_loss_val.item() / accelerator.gradient_accumulation_steps

            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)

            should_step = (
                (step % accelerator.gradient_accumulation_steps == 0)
                or (step == num_steps_in_epoch)
            )
            if should_step:
                grad_norm_pre = 0.0
                grad_norm_post = 0.0
                if cfg.training.max_grad_norm is not None:
                    _gn = accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                    grad_norm_pre = float(_gn.item()) if hasattr(_gn, "item") else float(_gn)
                    grad_norm_post = min(grad_norm_pre, float(cfg.training.max_grad_norm))

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                accelerator.log(
                    {
                        "train/loss": accumulated_loss,
                        "train/lm_policy_loss": accumulated_lm_loss,
                        "train/rank_policy_loss": accumulated_rank_loss,
                        "train/kl_per_token": accumulated_kl,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/grad_norm_pre_clip": grad_norm_pre,
                        "train/grad_norm_post_clip": grad_norm_post,
                        "train/round": cur_round,
                        "train/train_num_samples":      len(records),
                        "train/train_raw_samples":      _raw_rows,
                        "train/train_skipped_zero_adv": _skipped_zero_adv,
                        "train/train_zero_adv_rows":    _zero_adv_rows,
                    },
                    step=global_step,
                )

                if accelerator.is_local_main_process:
                    logger.info(
                        f"round={cur_round} epoch={epoch + 1} global_step={global_step} "
                        f"loss={accumulated_loss:.6f} lm_loss={accumulated_lm_loss:.6f} "
                        f"rank_loss={accumulated_rank_loss:.6f} kl={accumulated_kl:.6f} "
                        f"lr={lr_scheduler.get_last_lr()[0]:.3e} "
                        f"grad_norm_pre={grad_norm_pre:.4f} grad_norm_post={grad_norm_post:.4f}"
                    )

                accumulated_loss = 0.0
                accumulated_lm_loss = 0.0
                accumulated_kl = 0.0
                accumulated_rank_loss = 0.0

                lr_scheduler.step()
                global_step += 1

                ckpt_interval = cfg.training.get("ckpt_interval", None)
                if ckpt_interval is not None and global_step % int(ckpt_interval) == 0:
                    resumable_interval = cfg.training.get("resumable_ckpt_interval", None)
                    if resumable_interval is None:
                        is_resumable = False
                    else:
                        resumable_interval = int(resumable_interval)
                        if resumable_interval % int(ckpt_interval) != 0:
                            raise ValueError(
                                f"resumable_ckpt_interval ({resumable_interval}) must be a "
                                f"multiple of ckpt_interval ({ckpt_interval})"
                            )
                        is_resumable = (global_step % resumable_interval == 0)
                    save_checkpoint(
                        model, tokenizer, cfg, accelerator,
                        ckpt_name=f"checkpoint-{global_step}",
                        global_step=global_step,
                        epoch=epoch,
                        resumable=is_resumable,
                    )

        logger.info(f"epoch={epoch + 1} done")

    save_checkpoint(
        model, tokenizer, cfg, accelerator,
        ckpt_name=f"round_{cur_round}",
        global_step=global_step,
        epoch=epoch,
        resumable=True,
    )

    # Match rl_dream_train.py: drop references before teardown so NCCL / destroy_process_group
    # sees lower peak memory after ZeRO-3 training.
    del train_dl, dataset_lm
    del records, input_ids_t, labels_t, p_mask_t, advantages, start_pos_t, rank_step_data
    del optimizer, lr_scheduler
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    accelerator.end_training()


def main():
    cfg = get_config()
    run(cfg)


if __name__ == "__main__":
    main()
