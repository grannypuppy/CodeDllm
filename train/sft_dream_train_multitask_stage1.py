import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from omegaconf import OmegaConf
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader

from models import DreamTokenizer, DreamModel
from models.dream_multitask import DreamConfig as DreamMultitaskConfig
from models.dream_multitask import DreamModel as DreamMultitaskModel
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from train.utils import get_config, flatten_omega_conf, AverageMeter

logger = get_logger(__name__, log_level="INFO")


def _normalize_mismatched_keys(mismatched):
    normalized = []
    for item in mismatched:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            normalized.append(item[0])
        else:
            normalized.append(item)
    return normalized


def _validate_loading_info(loading_info, *, allow_missing_prefixes=()):
    missing_keys = loading_info.get("missing_keys", []) or []
    unexpected_keys = loading_info.get("unexpected_keys", []) or []
    mismatched_keys = _normalize_mismatched_keys(loading_info.get("mismatched_keys", []) or [])
    error_msgs = loading_info.get("error_msgs", []) or []

    unexpected_non_ignored = [k for k in unexpected_keys if "rotary_emb.inv_freq" not in k]
    invalid_missing = [
        k for k in missing_keys
        if not any(k.startswith(prefix) for prefix in allow_missing_prefixes)
    ]

    failures = []
    if invalid_missing:
        failures.append(f"invalid missing_keys={invalid_missing}")
    if unexpected_non_ignored:
        failures.append(f"unexpected_keys={unexpected_non_ignored}")
    if mismatched_keys:
        failures.append(f"mismatched_keys={mismatched_keys}")
    if error_msgs:
        failures.append(f"error_msgs={error_msgs}")

    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(
            "Checkpoint loading validation failed. "
            f"{detail}. "
            "For stage1 we only allow missing rank_head params when bootstrapping from base Dream."
        )

    if missing_keys:
        logger.info(f"Allowed missing keys during load: {missing_keys}")


def _three_way_relation(values, eps):
    """Map values to {-1, 0, +1} with eps tie region."""
    rel = torch.zeros_like(values, dtype=torch.int8)
    rel[values > eps] = 1
    rel[values < -eps] = -1
    return rel


class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks, depth_labels, start_pos):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks
        self.depth_labels = depth_labels
        self.start_pos = start_pos

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx],
            self.depth_labels[idx],
            self.start_pos[idx]
        )


def simple_collate(batch):
    inp, lbl, msk, dlbl, spos = zip(*batch)
    res = {
        "input_ids":  torch.stack(inp),
        "labels":     torch.stack(lbl),
        "p_mask_lm":  torch.stack(msk),
        "depth_labels": torch.stack(dlbl),
        "start_pos":  torch.tensor(spos)
    }
    return res


def main():
    config = get_config()
    pretrained_model = config.model.pretrained_model

    # 预处理数据目录：与 sft_dream_dataset 约定一致
    preprocessed_dir = config.dataset.get(
        "preprocessed_dir",
        Path("data") / "preprocessed" / config.experiment.project / config.experiment.wandb_run_name,
    )
    preprocessed_dir = Path(preprocessed_dir)
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(
            f"Preprocessed data dir not found: {preprocessed_dir}. "
            "Run sft_dream_dataset.py first (e.g. with config.dataset.preprocessed_dir=...)."
        )

    # 加载训练集
    train_pt = preprocessed_dir / "train.pt"
    if not train_pt.is_file():
        raise FileNotFoundError(f"Train data not found: {train_pt}")
    train_data = torch.load(train_pt, map_location="cpu", weights_only=True)
    dataset_lm = TrainDataset(
        train_data["input_ids"],
        train_data["labels"],
        train_data["p_mask_lm"],
        train_data["depth_labels"],
        train_data["start_pos"],
    )
    num_train = len(dataset_lm)

    # 加载验证集（可选）
    eval_dataloader = None
    val_pt = preprocessed_dir / "val.pt"
    if val_pt.is_file():
        val_data = torch.load(val_pt, map_location="cpu", weights_only=True)
        dataset_val = TrainDataset(
            val_data["input_ids"],
            val_data["labels"],
            val_data["p_mask_lm"],
            val_data["depth_labels"],
            val_data["start_pos"],
        )
        eval_dataloader = DataLoader(
            dataset_val,
            batch_size=config.training.batch_size_lm,
            sampler=None,
            collate_fn=simple_collate,
            num_workers=0,
        )
    else:
        logger.info("No val.pt found; skipping validation.")

    # 元数据（用于日志）
    meta_path = preprocessed_dir / "meta.json"
    meta = {}
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    drop_num = meta.get("drop_num", 0)
    num_val = meta.get("num_val", 0)

    # ---------- Accelerator ----------
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Validate checkpoint intervals: resumable_ckpt_interval must be a multiple of ckpt_interval
    ckpt_interval = config.training.get("ckpt_interval")
    resumable_ckpt_interval = config.training.get("resumable_ckpt_interval")
    if resumable_ckpt_interval is not None and ckpt_interval is not None:
        if resumable_ckpt_interval % ckpt_interval != 0:
            raise ValueError(
                f"resumable_ckpt_interval ({resumable_ckpt_interval}) must be a multiple of ckpt_interval ({ckpt_interval})"
            )

    config.experiment.logging_dir = str( Path("projects") / Path(config.experiment.project) / Path(config.experiment.wandb_run_name) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    # ---------- Logging, seed, config ----------
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

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id
        wandb_init_kwargs = dict(
            name=config.experiment.wandb_run_name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(Path("projects") / config.experiment.project / config.experiment.wandb_run_name, exist_ok=True)
        config_path = Path("projects") / config.experiment.project / config.experiment.wandb_run_name / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    resume_from_checkpoint = config.experiment.get("resume_from_checkpoint", None) or config.training.get("resume_from_checkpoint", None)
    if resume_from_checkpoint:
        resume_from_checkpoint = Path(resume_from_checkpoint).resolve()
        if not resume_from_checkpoint.is_dir():
            raise FileNotFoundError(f"resume_from_checkpoint not found or not a directory: {resume_from_checkpoint}")
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        load_path = str(resume_from_checkpoint)
        # Resume: rank_head weights should already exist in checkpoint.
        model_config = DreamMultitaskConfig.from_pretrained(load_path)
        logger.info("Loading models and optimizer")
        tokenizer = DreamTokenizer.from_pretrained(load_path)
        model, loading_info = DreamMultitaskModel.from_pretrained(
            load_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
            output_loading_info=True,
        )
        _validate_loading_info(loading_info, allow_missing_prefixes=())

    else:
        load_path = pretrained_model
        logger.info(f"Loading base model from: {load_path}")
        
        # 1. Load config and enforce num_rank_labels=1
        model_config = DreamMultitaskConfig.from_pretrained(load_path)
        model_config.num_rank_labels = 1
        
        # 2. Set seed for deterministic initialization of the new rank_head
        if config.training.seed is not None:
            set_seed(42)  # Fixed seed as requested for new layer init

        # 3. Initialize model structure (new rank_head remains randomly initialized)
        logger.info("Loading models and optimizer")
        tokenizer = DreamTokenizer.from_pretrained(load_path)
        
        # 重要：严格校验加载，只允许 rank_head 为 missing（从 base Dream 启动时）
        model, loading_info = DreamMultitaskModel.from_pretrained(
            load_path, 
            config=model_config, 
            torch_dtype=torch.bfloat16,
            output_loading_info=True,
        )
        _validate_loading_info(loading_info, allow_missing_prefixes=("rank_head.",))
        
        # 恢复正常的训练 seed (如果之前设了42现在改回来)
        if config.training.seed is not None:
            set_seed(config.training.seed)


    # --- Stage 1: Freeze all parameters except the new rank_head ---
    # We MUST apply freezing logic even when resuming, to ensure the model structure (trainable params)
    # matches the optimizer state saved in the checkpoint.
    logger.info("Stage 1 Training: Freezing all parameters except 'rank_head'...")
    for name, param in model.named_parameters():
        if "rank_head" in name:
            param.requires_grad = True
            if not resume_from_checkpoint:
               logger.info(f"  Training parameter: {name} (shape: {param.shape})")
        else:
            param.requires_grad = False
    
    # In Stage 1, since backbone is frozen, gradient checkpointing is unnecessary and can cause warnings/slowdown
    if config.training.gradient_checkpointing_enable:
        logger.info("Stage 1: Disabling gradient_checkpointing because backbone is frozen.")
        config.training.gradient_checkpointing_enable = False
    # -------------------------------------------------------------

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = model.config.mask_token_id
    pad_id = model.config.pad_token_id

    # ---------- Optimizer ----------
    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # ---------- LR scheduler (基于 num_train) ----------
    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(num_train / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    warmup_param = config.lr_scheduler.params.warmup_steps
    if isinstance(warmup_param, float) and 0 < warmup_param < 1:
        num_warmup_steps = int(max_train_steps * warmup_param)
    else:
        num_warmup_steps = int(warmup_param)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=num_warmup_steps * accelerator.num_processes,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    # ---------- DataLoader ----------
    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    if eval_dataloader:
        model, optimizer, lr_scheduler, train_dataloader_lm, eval_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader_lm, eval_dataloader
        )
    else:
        model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader_lm
        )
    accelerator.register_for_checkpointing(lr_scheduler)

    # Load full training state when resuming (optimizer, scheduler, RNG; model/tokenizer already loaded from load_path)
    first_epoch = 0
    global_step = 0
    if resume_from_checkpoint:
        accelerator.load_state(str(resume_from_checkpoint))
        trainer_state_path = Path(resume_from_checkpoint) / "trainer_state.json"
        if trainer_state_path.is_file():
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)
            global_step = int(trainer_state.get("global_step", 0))
            first_epoch = int(trainer_state.get("epoch", 0))
            logger.info(f"Resumed training state: global_step={global_step}, epoch={first_epoch}")
        else:
            logger.warning(f"No trainer_state.json in {resume_from_checkpoint}; starting from global_step=0, epoch=0")

    # ---------- Attention mask / forward_process：与 sft_dream_chat 完全一致 ----------
    def make_causal_attention_mask(input_ids):
        B, T = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        ).view(1, 1, T, T)

        bias = torch.zeros((B, 1, T, T), device=device)
        bias.masked_fill_(causal_mask, float("-inf"))

        return bias



    def make_attention_mask(input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        bias = torch.zeros(B, 1, 1, T, device=device)
        return bias


    def pairwise_margin_rank_loss(rank_pred, depth_labels_shift, shift_mask, rank_eps=1e-4):
        """Pairwise margin rank loss + 0/1 relation metrics."""
        losses = []
        B = rank_pred.size(0)
        total_pairs = torch.tensor(0.0, device=rank_pred.device, dtype=rank_pred.dtype)
        correct_pairs = torch.tensor(0.0, device=rank_pred.device, dtype=rank_pred.dtype)
        wrong_pairs = torch.tensor(0.0, device=rank_pred.device, dtype=rank_pred.dtype)
        for b in range(B):
            valid_idx = torch.where(shift_mask[b])[0]
            if valid_idx.numel() < 2:
                continue
            p = rank_pred[b, valid_idx]
            d = depth_labels_shift[b, valid_idx]

            dp = p.unsqueeze(1) - p.unsqueeze(0)
            dd = d.unsqueeze(1) - d.unsqueeze(0)
            tri = torch.triu(torch.ones_like(dd, dtype=torch.bool), diagonal=1)
            dp = dp[tri]
            dd = dd[tri]
            if dp.numel() == 0:
                continue

            pred_rel = _three_way_relation(dp, rank_eps)
            label_rel = torch.sign(dd).to(torch.int8)
            match = (pred_rel == label_rel)
            pair_cnt = torch.tensor(float(match.numel()), device=rank_pred.device, dtype=rank_pred.dtype)
            corr_cnt = match.to(rank_pred.dtype).sum()
            wrong_cnt = pair_cnt - corr_cnt
            total_pairs = total_pairs + pair_cnt
            correct_pairs = correct_pairs + corr_cnt
            wrong_pairs = wrong_pairs + wrong_cnt

            neq_mask = dd != 0
            eq_mask = ~neq_mask

            loss_neq = torch.tensor(0.0, device=rank_pred.device, dtype=rank_pred.dtype)
            if neq_mask.any():
                y = torch.sign(dd[neq_mask])
                loss_neq = F.relu(rank_eps - y * dp[neq_mask]).mean()

            loss_eq = torch.tensor(0.0, device=rank_pred.device, dtype=rank_pred.dtype)
            if eq_mask.any():
                loss_eq = F.relu(torch.abs(dp[eq_mask]) - rank_eps).mean()

            losses.append(loss_neq + loss_eq)

        if not losses:
            zero = torch.tensor(0.0, device=rank_pred.device, dtype=rank_pred.dtype)
            return zero, {
                "pair_total": total_pairs,
                "pair_correct": correct_pairs,
                "pair_wrong": wrong_pairs,
            }

        return torch.stack(losses).mean(), {
            "pair_total": total_pairs,
            "pair_correct": correct_pairs,
            "pair_wrong": wrong_pairs,
        }

    def forward_process(input_ids, labels, p_mask_lm, depth_labels, start_pos):
        if config.training.method == "ar":
            #if config.training.batch_size_lm == 1:
            #    logits = model(input_ids, is_causal=True).logits
            attn_mask = make_causal_attention_mask(input_ids)
            outputs = model(input_ids, attention_mask=attn_mask, is_causal=False, return_dict=True)
        else:
            attention_mask = make_attention_mask(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, is_causal=False, return_dict=True)

        logits = outputs.logits
        rank_values = outputs.rank_values.squeeze(-1)

        # --- LM Loss ---
        B, T, V = logits.shape

        shift_mask   = p_mask_lm[:, 1:]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        shift_rank_values = rank_values[:, :-1]
        shift_depth_labels = depth_labels[:, 1:]

        # Create safe labels for gather, replacing -100 with 0
        safe_labels = shift_labels.clone()
        safe_labels[safe_labels == -100] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        log_probs = F.log_softmax(shift_logits, dim=-1)                             # (B, T-1, V)
        logp_tok  = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)     # (B, T-1)

        if False:
            shift_inputs = input_ids[:, 1:]
            # --- Enhanced Logging Start ---
            # Only log for the first sample in the batch to avoid clutter
            b_idx = 0
            logger.info(f"=== Detailed Supervision for Batch Sample {b_idx} ===")

            # 1. Input Tokens
            input_tokens_str = tokenizer.decode(input_ids[b_idx], skip_special_tokens=False)
            logger.info(f"Full Input Sequence: {input_tokens_str}")
            logger.info(f"Input IDs: {input_ids[b_idx].tolist()}")

            # 2. Predictions vs Targets
            # Get predicted token ids (argmax)
            pred_ids = torch.argmax(shift_logits[b_idx], dim=-1) # (T-1)

            # Iterate over sequence
            seq_len = T - 1
            logger.info(f"{'Pos':<5} | {'Input':<15} | {'Target':<15} | {'Pred':<15} | {'Tgt Prob':<10} | {'Mask':<5}")
            logger.info("-" * 80)

            for t in range(seq_len):
                curr_input_id = shift_inputs[b_idx, t].item()
                curr_target_id = shift_labels[b_idx, t].item()
                curr_pred_id = pred_ids[t].item()
                curr_mask = shift_mask[b_idx, t].item()
                curr_prob = torch.exp(logp_tok[b_idx, t]).item()

                # Decode
                s_input = tokenizer.decode([curr_input_id])
                s_target = tokenizer.decode([curr_target_id]) if curr_target_id != -100 else "<IGNORE>"
                s_pred = tokenizer.decode([curr_pred_id])

                logger.info(f"{t:<5} | {s_input:<15} | {s_target:<15} | {s_pred:<15} | {curr_prob:.4f}     | {curr_mask:<5}")

            logger.info("=" * 50)
            # --- Enhanced Logging End ---

        if True:
            # --- Modified Loss Calculation Start ---
            token_losses = -logp_tok

            B_size, T_minus_1 = shift_labels.shape
            device = shift_labels.device

            indices = torch.arange(T_minus_1, device=device).unsqueeze(0).expand(B_size, -1)

            # Find last non-pad token
            is_non_pad = (shift_labels != pad_id)
            # Use a large negative number for pad positions so max picks the last non-pad index
            non_pad_indices = torch.where(is_non_pad, indices, torch.tensor(-1, device=device))
            last_non_pad_idx = non_pad_indices.max(dim=1).values

            cutoff_idx = last_non_pad_idx + 1

            is_normal_mask = indices <= cutoff_idx.unsqueeze(1)
            is_pad_group_mask = indices > cutoff_idx.unsqueeze(1)

            final_normal_mask = shift_mask & is_normal_mask
            final_pad_group_mask = shift_mask & is_pad_group_mask

            loss_normal_sum = (token_losses * final_normal_mask).sum(dim=1)
            count_normal = final_normal_mask.sum(dim=1)

            loss_pad_group_sum = (token_losses * final_pad_group_mask).sum(dim=1)
            count_pad_group = final_pad_group_mask.sum(dim=1)

            avg_pad_loss = torch.zeros_like(loss_pad_group_sum)
            has_pad_group = count_pad_group > 0
            avg_pad_loss[has_pad_group] = loss_pad_group_sum[has_pad_group] / count_pad_group[has_pad_group]

            total_seq_loss = loss_normal_sum + avg_pad_loss
            total_seq_count = count_normal + has_pad_group.float()

            loss_per_seq = total_seq_loss / total_seq_count.clamp(min=1)
            loss_lm = loss_per_seq.sum() / B
            # --- Modified Loss Calculation End ---
        else:
            loss_lm = - (logp_tok * shift_mask).sum(dim=1)

            mask_num = (shift_mask).sum(dim=1).clamp(min=1)
            loss_lm = loss_lm / mask_num
            loss_lm = loss_lm.sum() / B
        
        # --- Rank Loss (Stage 1: only rank branch optimization) ---
        rank_pred = F.softmax(shift_rank_values, dim=-1)
        rank_eps = config.training.get("rank_eps", 1e-4)
        loss_rank, rank_stats = pairwise_margin_rank_loss(
            rank_pred,
            shift_depth_labels,
            shift_mask,
            rank_eps=rank_eps,
        )

        # --- Total Loss (Stage 1: only rank) ---
        loss_total = loss_rank

        return loss_total, loss_lm, loss_rank, rank_stats

    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")

    logger.info(f"  Num response = {meta.get('num_response', num_train)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {num_train}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    data_time_m = AverageMeter()
    end = time.time()
    epoch = first_epoch  # so final save has a defined epoch even if loop runs 0 times

    from tqdm.auto import tqdm

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        # When resuming mid-epoch: skip batches already processed (Accelerate does not save DataLoader state)
        active_dataloader = train_dataloader_lm
        if resume_from_checkpoint and epoch == first_epoch and global_step > 0:
            steps_done_in_prev_epochs = num_update_steps_per_epoch * first_epoch
            steps_done_in_this_epoch = global_step - steps_done_in_prev_epochs
            batches_to_skip = steps_done_in_this_epoch * accelerator.gradient_accumulation_steps
            active_dataloader = accelerator.skip_first_batches(train_dataloader_lm, batches_to_skip)
            logger.info(f"Skipping first {batches_to_skip} batches (step {steps_done_in_this_epoch} in epoch {epoch}) to resume from global_step={global_step}")

        progress_bar = tqdm(
            active_dataloader,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True
        )
        accumulated_loss = 0.0
        accumulated_loss_lm = 0.0
        accumulated_loss_rank = 0.0
        accumulated_pair_total = 0.0
        accumulated_pair_correct = 0.0
        accumulated_pair_wrong = 0.0
        batch_length_list = []
        mask_num_list = []

        for step, batch in enumerate(progress_bar, start=1):

            data_time_m.update(time.time() - end)

            input_ids = batch["input_ids"].to(accelerator.device)
            labels    = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)
            start_pos = batch["start_pos"].to(accelerator.device)

            # Collect metrics
            batch_length_list.append(input_ids.shape[1])
            mask_num_list.append(p_mask_lm.sum().item())
            
            # --- Label extraction (`depth_labels` kept as data key) ---
            depth_labels = batch["depth_labels"].to(accelerator.device).bfloat16()

            # --- Forward Pass ---
            loss, loss_lm, loss_rank, rank_stats = forward_process(
                input_ids=input_ids,
                labels=labels,
                p_mask_lm=p_mask_lm,
                depth_labels=depth_labels,
                start_pos=start_pos
            )

            accumulated_loss += loss.item() / accelerator.gradient_accumulation_steps
            accumulated_loss_lm += loss_lm.item() / accelerator.gradient_accumulation_steps
            accumulated_loss_rank += loss_rank.item() / accelerator.gradient_accumulation_steps
            accumulated_pair_total += rank_stats["pair_total"].item()
            accumulated_pair_correct += rank_stats["pair_correct"].item()
            accumulated_pair_wrong += rank_stats["pair_wrong"].item()
            
            # Normalize for accumulation
            loss_normalized = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss_normalized)

            if step % accelerator.gradient_accumulation_steps == 0:
                # --- Validation Loop (Optional) ---
                if eval_dataloader and config.training.get("val_loss_interval", None) and global_step % config.training.val_loss_interval == 0:
                    logger.info(f"Running validation at step {global_step}...")
                    model.eval()
                    total_val_loss = 0.0
                    total_val_loss_lm = 0.0
                    total_val_loss_rank = 0.0
                    total_val_pair_total = 0.0
                    total_val_pair_correct = 0.0
                    total_val_pair_wrong = 0.0
                    num_val_batches = 0

                    # Disable gradient calculation for validation
                    with torch.no_grad():
                        for val_batch in eval_dataloader:
                            val_input_ids = val_batch["input_ids"].to(accelerator.device)
                            val_labels    = val_batch["labels"].to(accelerator.device)
                            val_p_mask_lm = val_batch["p_mask_lm"].to(accelerator.device)
                            val_start_pos = val_batch["start_pos"].to(accelerator.device)
                            val_depth_labels = val_batch["depth_labels"].to(accelerator.device).bfloat16()

                            val_loss, val_loss_lm, val_loss_rank, val_rank_stats = forward_process(
                                input_ids=val_input_ids,
                                labels=val_labels,
                                p_mask_lm=val_p_mask_lm,
                                depth_labels=val_depth_labels,
                                start_pos=val_start_pos
                            )
                            # Gather loss from all GPUs
                            gathered_val_loss = accelerator.gather(val_loss).mean()
                            gathered_val_loss_lm = accelerator.gather(val_loss_lm).mean()
                            gathered_val_loss_rank = accelerator.gather(val_loss_rank).mean()
                            gathered_val_pair_total = accelerator.gather(val_rank_stats["pair_total"]).sum()
                            gathered_val_pair_correct = accelerator.gather(val_rank_stats["pair_correct"]).sum()
                            gathered_val_pair_wrong = accelerator.gather(val_rank_stats["pair_wrong"]).sum()

                            total_val_loss += gathered_val_loss.item()
                            total_val_loss_lm += gathered_val_loss_lm.item()
                            total_val_loss_rank += gathered_val_loss_rank.item()
                            total_val_pair_total += gathered_val_pair_total.item()
                            total_val_pair_correct += gathered_val_pair_correct.item()
                            total_val_pair_wrong += gathered_val_pair_wrong.item()
                            num_val_batches += 1

                    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
                    avg_val_loss_lm = total_val_loss_lm / num_val_batches if num_val_batches > 0 else 0.0
                    avg_val_loss_rank = total_val_loss_rank / num_val_batches if num_val_batches > 0 else 0.0
                    val_pair_acc = (
                        total_val_pair_correct / total_val_pair_total
                        if total_val_pair_total > 0
                        else 0.0
                    )
                    
                    logger.info(
                        f"Validation Loss at step {global_step}: "
                        f"Total={avg_val_loss:.4f}, LM={avg_val_loss_lm:.4f}, Rank={avg_val_loss_rank:.4f}, "
                        f"PairAcc={val_pair_acc:.4f} ({int(total_val_pair_correct)}/{int(total_val_pair_total)})"
                    )

                    if accelerator.is_main_process:
                        accelerator.log({
                            "val_loss": avg_val_loss,
                            "val_loss_lm": avg_val_loss_lm,
                            "val_loss_rank": avg_val_loss_rank,
                            "val_rank_pair_acc": val_pair_acc,
                            "val_rank_pair_total": total_val_pair_total,
                            "val_rank_pair_correct": total_val_pair_correct,
                            "val_rank_pair_wrong": total_val_pair_wrong,
                        }, step=global_step)

                    model.train()
                # -----------------------

                # Clip gradients
                grad_norm_pre = 0.0
                grad_norm_post = 0.0
                if config.training.max_grad_norm is not None:
                    # clip_grad_norm_ returns the pre-clip norm
                    grad_norm_pre = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    if hasattr(grad_norm_pre, 'item'):
                        grad_norm_pre = grad_norm_pre.item()
                    # Post-clip norm is min(pre, max)
                    grad_norm_post = min(grad_norm_pre, config.training.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Calculate aggregated metrics
                batch_length_max = max(batch_length_list) if batch_length_list else 0
                batch_length_avg = sum(batch_length_list) / len(batch_length_list) if batch_length_list else 0
                mask_num_max = max(mask_num_list) if mask_num_list else 0
                mask_num_avg = sum(mask_num_list) / len(mask_num_list) if mask_num_list else 0
                train_pair_acc = (
                    accumulated_pair_correct / accumulated_pair_total
                    if accumulated_pair_total > 0
                    else 0.0
                )

                accelerator.log({
                    "train_loss": accumulated_loss,
                    "lm_loss": accumulated_loss_lm,
                    "rank_loss": accumulated_loss_rank,
                    "rank_pair_acc": train_pair_acc,
                    "rank_pair_total": accumulated_pair_total,
                    "rank_pair_correct": accumulated_pair_correct,
                    "rank_pair_wrong": accumulated_pair_wrong,
                    "grad_norm_pre_clip": grad_norm_pre,
                    "grad_norm_post_clip": grad_norm_post,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "batch_length_max": batch_length_max,
                    "batch_length_avg": batch_length_avg,
                    "mask_num_max": mask_num_max,
                    "mask_num_avg": mask_num_avg
                }, step=global_step)

                accumulated_loss = 0.0
                accumulated_loss_lm = 0.0
                accumulated_loss_rank = 0.0
                accumulated_pair_total = 0.0
                accumulated_pair_correct = 0.0
                accumulated_pair_wrong = 0.0
                batch_length_list = []
                mask_num_list = []

                lr_scheduler.step()
                global_step += 1

                if config.training.ckpt_interval is not None and global_step % config.training.ckpt_interval == 0:
                    resumable_interval = config.training.get("resumable_ckpt_interval")
                    if resumable_interval is None:
                        is_resumable = False
                    else:
                        if resumable_interval % config.training.ckpt_interval != 0:
                            raise ValueError(
                                f"resumable_ckpt_interval ({resumable_interval}) must be a multiple of ckpt_interval ({config.training.ckpt_interval})"
                            )
                        is_resumable = (global_step % resumable_interval == 0)
                    logger.info(f"Saving checkpoint at global step {global_step} ({'resumable' if is_resumable else 'lightweight'}) ...")
                    accelerator.wait_for_everyone()
                    logger.info("Finish wait_for_everyone")
                    ckpt_name = f"checkpoint-{global_step}"
                    save_checkpoint(model, tokenizer, config, accelerator, ckpt_name, global_step=global_step, epoch=epoch, resumable=is_resumable)

            del input_ids, labels, p_mask_lm, depth_labels
            torch.cuda.empty_cache()


        accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    ckpt_name = "final"
    save_checkpoint(model, tokenizer, config, accelerator, ckpt_name, global_step=global_step, epoch=epoch, resumable=True)

    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, ckpt_name, global_step=None, epoch=None, resumable=True):
    """Save checkpoint in HuggingFace format. Optionally also save optimizer/scheduler/RNG for resume.

    Args:
        resumable: If True, save full state (accelerator.save_state) for resuming. If False, save only
            model + tokenizer (lightweight, cannot resume from this checkpoint).
    """
    output_dir = Path("projects", config.experiment.project, config.experiment.wandb_run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_total_limit = config.training.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")],
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

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if global_step is not None:
            metadata["global_step"] = global_step
        if epoch is not None:
            metadata["epoch"] = epoch
        with (save_base / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        trainer_state = {}
        if global_step is not None:
            trainer_state["global_step"] = global_step
        if epoch is not None:
            trainer_state["epoch"] = epoch
        if trainer_state:
            with (save_base / "trainer_state.json").open("w", encoding="utf-8") as f:
                json.dump(trainer_state, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base}" + (" (resumable)" if resumable else " (lightweight)"))

    # Save optimizer, scheduler, RNG with Accelerator when resumable; all processes must call for DeepSpeed
    accelerator.wait_for_everyone()
    if resumable:
        accelerator.save_state(str(save_base))


if __name__ == "__main__":
    main()
