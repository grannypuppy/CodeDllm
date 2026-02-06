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
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from models import DreamTokenizer, DreamModel
from models.dream_online.generation_utils import sample_tokens
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

SYSTEM_PROMPT_LEN = 28

from train.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

# Online training: diffusion step counts for round-robin. Each seq runs once; steps cycle [16, 32, 64, 128, 16, 32, ...].
TOTAL_STEPS_SETTING = [8, 16, 32]
# Diffusion params (align with gen_eval.sh / gen_eval_online.py: top_p=0.9, temperature=0.1)
DIFFUSION_EPS = 1e-3
DIFFUSION_ALG = "entropy"
DIFFUSION_ALG_TEMP = 0.1
DIFFUSION_TEMPERATURE = 0.1
DIFFUSION_TOP_P = 0.9
DIFFUSION_TOP_K = None
DIFFUSION_THRESHOLD = None

# Loss mode: True = grouped loss (normal + pad_group, same as sft_dream_chat 668-705); False = simple mean over response segment (same as chat else branch)
USE_GROUPED_LOSS = True



class TrainDataset(Dataset):
    """Online training: each item is (input_ids, labels, start_pos). No p_mask."""

    def __init__(self, inputs, labels, start_pos):
        self.inputs = inputs
        self.labels = labels
        self.start_pos = start_pos

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.start_pos[idx],
        )


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    pretrained_model = config.model.pretrained_model

    # Enable TF32 on Ampere GPUs
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

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
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


    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
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




    def collapse_k_unique(lst, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))

        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]
    
    


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    
    @torch.no_grad()
    def prepare_inputs_and_labels_online(input_list, target_list, mask_id=mask_id):
        """Online training: strictly same padding as sft_dream_dataset_m.prepare_inputs_and_labels_for_text.
        Order and values: build curr_input_ids/curr_labels per sample, then right-pad to max_len.
        Input = prompt + [MASK]*len(response); labels = -100*prompt + response.
        Right pad: input 补 pad_id, labels 补 pad_id（与 dataset_m 一致；pad 区域可被 mask 并参与 loss）.
        """
        src_code_prompts_template = '''Below is a program. Optimize the program and provide a faster version.\nProgram:\n```python\n{{src_code}}\n```'''
        eos_token = tokenizer.eos_token
        tgt_code_response_template = '''Here is the optimized code:\n```python\n{{tgt_code}}\n```''' + eos_token
        pad_id = tokenizer.pad_token_id
        max_prompt_len = config.training.max_prompt_len
        max_gen_length = config.training.max_gen_length

        # Same order as dataset_m: list per sample first
        input_ids_lm = []
        labels_lm = []
        start_pos_list = []
        keep_indices = []

        for i, (inp, tgt) in enumerate(zip(input_list, target_list)):
            prompt_text = [{"role": "user", "content": src_code_prompts_template.replace("{{src_code}}", inp)}]
            prompt_ids = tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True, add_special_tokens=False)
            if len(prompt_ids) > max_prompt_len:
                continue

            response_ids = tokenizer(tgt_code_response_template.replace("{{tgt_code}}", tgt), add_special_tokens=False).input_ids
            if len(response_ids) > max_gen_length:
                response_ids = response_ids[:max_gen_length]

            # Same as dataset_m: curr_input_ids = prompt + content_region; curr_labels = -100*prompt + content_region.
            # Online: content_region in input is MASK*len(response); in labels is response_ids.
            curr_input_ids = prompt_ids + [mask_id] * len(response_ids)
            curr_labels = [-100] * len(prompt_ids) + response_ids

            input_ids_lm.append(torch.tensor(curr_input_ids))
            labels_lm.append(torch.tensor(curr_labels))
            start_pos_list.append(len(prompt_ids))
            keep_indices.append(i)

        drop_num = len(input_list) - len(keep_indices)
        if not input_ids_lm:
             raise ValueError("No valid data found in the batch")

        # Right Padding
        max_len = max([x.size(0) for x in input_ids_lm])

        padded_inputs = []
        padded_labels = []
        
        for i in range(len(input_ids_lm)):
            rem = max_len - input_ids_lm[i].size(0)
            if rem > 0:
                r_pad = torch.full((rem,), pad_id, dtype=torch.long)
                r_lbl = torch.full((rem,), pad_id, dtype=torch.long)                
                padded_inputs.append(torch.cat([input_ids_lm[i], r_pad]))
                padded_labels.append(torch.cat([labels_lm[i], r_lbl]))
            else:
                padded_inputs.append(input_ids_lm[i])
                padded_labels.append(labels_lm[i])

        input_ids_lm = torch.stack(padded_inputs)
        labels_lm = torch.stack(padded_labels)
        start_pos_list = torch.tensor(start_pos_list, dtype=torch.long)
        return input_ids_lm, labels_lm, start_pos_list, drop_num

    def simple_collate(batch):
        inp, lbl, spos = zip(*batch)
        return {
            "input_ids": torch.stack(inp),
            "labels": torch.stack(lbl),
            "start_pos": torch.stack(spos) if isinstance(spos[0], torch.Tensor) else torch.tensor(spos),
        }
    

    
    with open("./data/" + config.dataset.optimization_data, 'r') as f:
        dataset_load = json.load(f)
    dataset_load = dataset_load[:1000]

    input_list = [x["input"] for x in dataset_load]
    target_list = [x["target"] for x in dataset_load]

    input_ids, labels, start_pos, drop_num = prepare_inputs_and_labels_online(input_list, target_list)
    dataset_lm = TrainDataset(input_ids, labels, start_pos)

    # --- Validation Data Loading ---
    eval_dataloader = None
    if config.dataset.get("validation_data", None):
        logger.info(f"Loading validation data from {config.dataset.validation_data}")
        with open("./data/" + config.dataset.validation_data, 'r') as f:
            val_dataset_load = json.load(f)
        val_dataset_load = val_dataset_load[-50:]
        val_input_list = [x["input"] for x in val_dataset_load]
        val_target_list = [x["target"] for x in val_dataset_load]
        val_input_ids, val_labels, val_start_pos, _ = prepare_inputs_and_labels_online(val_input_list, val_target_list)
        dataset_val = TrainDataset(val_input_ids, val_labels, val_start_pos)
        eval_dataloader = DataLoader(
            dataset_val,
            batch_size=config.training.batch_size_lm,
            sampler=None,
            collate_fn=simple_collate,
            num_workers=0,
        )
    # -------------------------------
    
    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    # Online: one run per seq, steps round-robin from TOTAL_STEPS_SETTING (seq0→16, seq1→32, seq2→64, seq3→128, seq4→16, ...)
    num_logic_samples_per_step = config.training.batch_size_lm
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    # Handle warmup steps or ratio
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

    

    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Online: TOTAL_STEPS_SETTING = {TOTAL_STEPS_SETTING}, logic samples per step = {num_logic_samples_per_step}")

    first_epoch = 0
    global_step = 0
    data_time_m = AverageMeter()
    end = time.time()

    import torch.nn.functional as F



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

    @torch.no_grad()
    def diffusion_step_update(x, logits, i, steps, timesteps, mask_token_id, alg, alg_temp, temperature, top_p, top_k, threshold, device, state=None):
        """Update x for next diffusion step. Same logic as generation_utils._sample (while loop body). No grad.
        For alg=='confidence_threshold', state is dict with number_transfer_tokens, left_tokens_last_step, steps (mutable); returns updated state."""
        # Same as generation_utils: mask_index, shift logits, sample_tokens
        mask_index = (x == mask_token_id)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        confidence_full, x0_full = sample_tokens(
            logits, temperature=temperature, top_p=top_p, top_k=top_k,
            margin_confidence=(alg == 'topk_margin'), neg_entropy=(alg == 'entropy')
        )
        if not alg == 'confidence_threshold':
            t = timesteps[i]
            s = timesteps[i + 1]

        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + mask_token_id
            transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
            x0[transfer_index_t_s] = x0_full[mask_index][transfer_index_t_s]
            x[mask_index] = x0.clone()
            x[~mask_index] = x0_full[~mask_index]
        elif alg == 'confidence_threshold':
            number_transfer_tokens = state["number_transfer_tokens"]
            left_tokens_last_step = state["left_tokens_last_step"]
            full_confidence = torch.full_like(x, -torch.inf, device=device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence_full[mask_index]
            current_transfer_tokens = number_transfer_tokens + left_tokens_last_step
            state["left_tokens_last_step"] = 0
            selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
            transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
            select_index = select_index.to(x.device)
            transfer_index[0, select_index[0]] = True
            for k in range(1, current_transfer_tokens):
                if selected_confidence[0, k] < threshold:
                    if i < steps - 1:
                        state["left_tokens_last_step"] += 1
                        transfer_index[0, select_index[0, k]] = False
                    else:
                        state["number_transfer_tokens"] = 0
                        state["steps"] = state.get("steps", steps) + 1
                        state["left_tokens_last_step"] += 1
                        transfer_index[0, select_index[0, k]] = False
            x[~mask_index] = x0_full[~mask_index]
            x[transfer_index] = x0_full[transfer_index]
        else:
            if alg not in ('maskgit_plus', 'topk_margin', 'entropy'):
                raise RuntimeError(f"Unknown alg: {alg}")
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            full_confidence = torch.full_like(x, -torch.inf, device=device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence_full[mask_index]
            x[~mask_index] = x0_full[~mask_index]
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices, transfer_index] = x0_full[row_indices, transfer_index]
        return state

    def online_diffusion_forward(input_ids, labels, start_pos, steps, mask_id, pad_id):
        """One diffusion run: align with inference loop; at each step compute loss on response vs GT.
        Supports per-sample start_pos (same as sft_dream_chat / sft_dream_dataset_m).
        """
        device = input_ids.device
        B, max_length = input_ids.shape
        start_pos_1d = start_pos.view(-1)  # (B,)

        x = input_ids.clone()
        attention_mask = make_attention_mask(x)
        tok_idx = None
        timesteps = torch.linspace(1, DIFFUSION_EPS, steps + 1, device=device)

        if DIFFUSION_ALG == 'confidence_threshold':
            mask_index_0 = (x == mask_id)
            assert mask_index_0.sum() % steps == 0, "mask_index.sum() must be divisible by steps"
            assert x.shape[0] == 1, "batch size must be 1"
            state = {"number_transfer_tokens": mask_index_0.sum().item() // steps, "left_tokens_last_step": 0, "steps": steps}
        else:
            state = None

        total_loss = 0.0
        num_steps = 0
        i = 0
        current_steps = steps
        while i < current_steps:
            logits = model(x, attention_mask, tok_idx).logits
            logits_shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            # Per-sample start_pos and gen_length; USE_GROUPED_LOSS ? grouped (sft_dream_chat 668-705) : simple mean (chat else 707-712)
            step_loss_sum = 0.0
            for b in range(B):
                start_pos_b = start_pos_1d[b].item()
                gen_length_b = max_length - start_pos_b
                resp_logits_b = logits_shifted[b, start_pos_b : start_pos_b + gen_length_b, :]   # (gen_length_b, V)
                resp_labels_b = labels[b, start_pos_b : start_pos_b + gen_length_b]              # (gen_length_b,)
                log_probs_b = F.log_softmax(resp_logits_b, dim=-1)
                safe_labels_b = resp_labels_b.clone()
                safe_labels_b[safe_labels_b == -100] = pad_id if pad_id is not None else 0
                logp_tok_b = log_probs_b.gather(dim=-1, index=safe_labels_b.unsqueeze(-1)).squeeze(-1)  # (gen_length_b,)
                token_losses_b = -logp_tok_b
                if USE_GROUPED_LOSS:
                    # --- Modified Loss Calculation (same as sft_dream_chat 668-705) ---
                    T_b = resp_labels_b.shape[0]
                    indices_b = torch.arange(T_b, device=device)
                    is_non_pad = (resp_labels_b != pad_id)
                    non_pad_indices = torch.where(is_non_pad, indices_b, torch.tensor(-1, device=device))
                    last_non_pad_idx = non_pad_indices.max().item()
                    cutoff_idx = last_non_pad_idx + 1
                    is_normal_mask = indices_b <= last_non_pad_idx
                    is_pad_group_mask = indices_b > last_non_pad_idx
                    final_normal_mask = is_normal_mask
                    final_pad_group_mask = is_pad_group_mask
                    loss_normal_sum = (token_losses_b * final_normal_mask.float()).sum()
                    count_normal = final_normal_mask.sum().float()
                    loss_pad_group_sum = (token_losses_b * final_pad_group_mask.float()).sum()
                    count_pad_group = final_pad_group_mask.sum()
                    has_pad_group = count_pad_group > 0
                    avg_pad_loss = torch.tensor(0.0, device=device)
                    if has_pad_group:
                        avg_pad_loss = loss_pad_group_sum / count_pad_group.float()
                    total_seq_loss = loss_normal_sum + avg_pad_loss
                    total_seq_count = count_normal + (1.0 if has_pad_group else 0.0)
                    loss_per_seq_b = total_seq_loss / total_seq_count.clamp(min=1)
                else:
                    # --- Simple mean (same as sft_dream_chat else 707-712) ---
                    mask_num = token_losses_b.numel()
                    loss_per_seq_b = token_losses_b.sum() / max(mask_num, 1)
                step_loss_sum = step_loss_sum + loss_per_seq_b
            total_loss = total_loss + step_loss_sum / B
            num_steps += 1

            state = diffusion_step_update(
                x, logits, i, current_steps, timesteps, mask_id, DIFFUSION_ALG, DIFFUSION_ALG_TEMP,
                DIFFUSION_TEMPERATURE, DIFFUSION_TOP_P, DIFFUSION_TOP_K, DIFFUSION_THRESHOLD, device, state,
            )
            if state is not None and "steps" in state:
                current_steps = state["steps"]
            i += 1
        return total_loss / max(num_steps, 1)


    

    




    from tqdm.auto import tqdm

    steps_setting_idx = 0
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
        step_count_list = []
        for step, batch in enumerate(progress_bar, start=1):

            data_time_m.update(time.time() - end)

            input_ids = batch["input_ids"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            start_pos = batch["start_pos"].to(accelerator.device)

            batch_length_list.append(input_ids.shape[1])
            B = input_ids.shape[0]
            run_losses = []
            for b in range(B):
                steps = TOTAL_STEPS_SETTING[(steps_setting_idx + b) % len(TOTAL_STEPS_SETTING)]
                run_loss = online_diffusion_forward(
                    input_ids[b : b + 1],
                    labels[b : b + 1],
                    start_pos[b : b + 1] if start_pos.dim() > 0 else start_pos.unsqueeze(0),
                    steps,
                    mask_id,
                    pad_id,
                )
                run_losses.append(run_loss)
                step_count_list.append(steps)
            steps_setting_idx += B
            loss_lm = torch.stack(run_losses).mean()
            accumulated_loss += loss_lm.item() / accelerator.gradient_accumulation_steps
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps
            accelerator.backward(loss_lm)

            if step % accelerator.gradient_accumulation_steps == 0:

                # --- Validation Loop (online: one diffusion run per sample with first step count) ---
                if eval_dataloader and config.training.get("val_loss_interval", None) and global_step % config.training.val_loss_interval == 0:
                    logger.info(f"Running validation at step {global_step}...")
                    model.eval()
                    total_val_loss = 0.0
                    num_val_batches = 0
                    val_steps = TOTAL_STEPS_SETTING[0]
                    with torch.no_grad():
                        for val_batch in eval_dataloader:
                            val_input_ids = val_batch["input_ids"].to(accelerator.device)
                            val_labels = val_batch["labels"].to(accelerator.device)
                            val_start_pos = val_batch["start_pos"].to(accelerator.device)
                            for b in range(val_input_ids.shape[0]):
                                val_loss = online_diffusion_forward(
                                    val_input_ids[b : b + 1],
                                    val_labels[b : b + 1],
                                    val_start_pos[b : b + 1] if val_start_pos.dim() > 0 else val_start_pos.unsqueeze(0),
                                    val_steps,
                                    mask_id,
                                    pad_id,
                                )
                                total_val_loss += val_loss.item()
                                num_val_batches += 1
                    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
                    gathered_val_loss = accelerator.gather(torch.tensor(avg_val_loss, device=accelerator.device)).mean().item()
                    logger.info(f"Validation Loss at step {global_step}: {gathered_val_loss:.4f}")
                    if accelerator.is_main_process:
                        accelerator.log({"val_loss": gathered_val_loss}, step=global_step)
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

                batch_length_max = max(batch_length_list) if batch_length_list else 0
                batch_length_avg = sum(batch_length_list) / len(batch_length_list) if batch_length_list else 0
                step_count_avg = sum(step_count_list) / len(step_count_list) if step_count_list else 0

                accelerator.log({
                    "train_loss": accumulated_loss,
                    "grad_norm_pre_clip": grad_norm_pre,
                    "grad_norm_post_clip": grad_norm_post,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "batch_length_max": batch_length_max,
                    "batch_length_avg": batch_length_avg,
                    "diffusion_steps_avg": step_count_avg,
                }, step=global_step)

                accumulated_loss = 0.0
                batch_length_list = []
                step_count_list = []

                lr_scheduler.step()
                global_step += 1
                
                if config.training.ckpt_interval is not None and global_step % config.training.ckpt_interval == 0:
                    logger.info(f"Saving checkpoint at global step {global_step} ...")
                    accelerator.wait_for_everyone()
                    logger.info("Finish wait_for_everyone")
                    ckpt_name = f"checkpoint-{global_step}"
                    save_checkpoint(model, tokenizer, config, accelerator, ckpt_name)

                del input_ids, labels, start_pos
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
