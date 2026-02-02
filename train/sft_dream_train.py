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
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from train.utils import get_config, flatten_omega_conf, AverageMeter

logger = get_logger(__name__, log_level="INFO")


class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks, start_pos):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks
        self.start_pos = start_pos

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx],
            self.start_pos[idx]
        )


def simple_collate(batch):
    inp, lbl, msk, spos = zip(*batch)
    res = {
        "input_ids":  torch.stack(inp),
        "labels":     torch.stack(lbl),
        "p_mask_lm":  torch.stack(msk),
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
    logger.info("Loading models and optimizer")
    tokenizer = DreamTokenizer.from_pretrained(pretrained_model)
    model = DreamModel.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)

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


    def forward_process(input_ids, labels, p_mask_lm, start_pos):
        if config.training.method == "ar":
            #if config.training.batch_size_lm == 1:
            #    logits = model(input_ids, is_causal=True).logits
            attn_mask = make_causal_attention_mask(input_ids)
            logits = model(input_ids, attention_mask=attn_mask, is_causal=False).logits
        else:
            attention_mask = make_attention_mask(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, is_causal=False).logits

        B, T, V = logits.shape

        shift_mask   = p_mask_lm[:, 1:]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

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

        return loss_lm

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

    first_epoch = 0
    global_step = 0
    data_time_m = AverageMeter()
    end = time.time()

    from tqdm.auto import tqdm

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True
        )
        accumulated_loss = 0.0
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

            loss_lm = forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    p_mask_lm=p_mask_lm,
                    start_pos=start_pos
                )

            accumulated_loss += loss_lm.item() / accelerator.gradient_accumulation_steps
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps
            accelerator.backward(loss_lm)

            if step % accelerator.gradient_accumulation_steps == 0:

                # --- Validation Loop ---
                if eval_dataloader and config.training.get("val_loss_interval", None) and global_step % config.training.val_loss_interval == 0:
                    logger.info(f"Running validation at step {global_step}...")
                    model.eval()
                    total_val_loss = 0.0
                    num_val_batches = 0

                    # Disable gradient calculation for validation
                    with torch.no_grad():
                        for val_batch in eval_dataloader:
                            val_input_ids = val_batch["input_ids"].to(accelerator.device)
                            val_labels    = val_batch["labels"].to(accelerator.device)
                            val_p_mask_lm = val_batch["p_mask_lm"].to(accelerator.device)
                            val_start_pos = val_batch["start_pos"].to(accelerator.device)

                            val_loss = forward_process(
                                input_ids=val_input_ids,
                                labels=val_labels,
                                p_mask_lm=val_p_mask_lm,
                                start_pos=val_start_pos
                            )
                            # Gather loss from all GPUs
                            gathered_val_loss = accelerator.gather(val_loss).mean()
                            total_val_loss += gathered_val_loss.item()
                            num_val_batches += 1

                    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
                    logger.info(f"Validation Loss at step {global_step}: {avg_val_loss:.4f}")

                    if accelerator.is_main_process:
                        accelerator.log({"val_loss": avg_val_loss}, step=global_step)

                    model.train()
                # -----------------------

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

                accelerator.log({
                    "train_loss": accumulated_loss,
                    "grad_norm_pre_clip": grad_norm_pre,
                    "grad_norm_post_clip": grad_norm_post,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "batch_length_max": batch_length_max,
                    "batch_length_avg": batch_length_avg,
                    "mask_num_max": mask_num_max,
                    "mask_num_avg": mask_num_avg
                }, step=global_step)

                accumulated_loss = 0.0
                batch_length_list = []
                mask_num_list = []

                lr_scheduler.step()
                global_step += 1

                if config.training.ckpt_interval is not None and global_step % config.training.ckpt_interval == 0:
                    logger.info(f"Saving checkpoint at global step {global_step} ...")
                    accelerator.wait_for_everyone()
                    logger.info("Finish wait_for_everyone")
                    ckpt_name = f"checkpoint-{global_step}"
                    save_checkpoint(model, tokenizer, config, accelerator, ckpt_name)

                del input_ids, labels, p_mask_lm
                torch.cuda.empty_cache()


        accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    ckpt_name = "final"
    save_checkpoint(model, tokenizer, config, accelerator, ckpt_name)

    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, ckpt_name):
    output_dir = Path("projects", config.experiment.project, config.experiment.wandb_run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / ckpt_name
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
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base}")


if __name__ == "__main__":
    main()
