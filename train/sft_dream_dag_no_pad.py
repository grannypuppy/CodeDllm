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
from train.prompting_utils import SegmentedPrompting
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







class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks, start_pos, weights=None, t_values=None):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks
        self.start_pos = start_pos
        self.weights = weights
        self.t_values = t_values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        t_val = self.t_values[idx] if self.t_values is not None else 0.0
        if self.weights is not None:
            return (
                self.inputs[idx],
                self.labels[idx],
                self.pmasks[idx],
                self.start_pos[idx],
                self.weights[idx],
                t_val
            )
        else:
             return (
                self.inputs[idx],
                self.labels[idx],
                self.pmasks[idx],
                self.start_pos[idx],
                torch.empty(0),
                t_val
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
        torch.backends.cudnn.benchmark = False
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
    def prepare_inputs_and_labels_for_text(
        input_list, target_list, step_map_list, target_weights_list=None, target_tokens_list=None, eps=1e-3, mask_id=mask_id
    ):
        # Templates
        src_code_prompts_template = '''<|im start|>system\nYou are a helpful assistant.<|im end|> <|im start|>user\nBelow is a program. Optimize the program and provide a faster version.\nProgram:\n ```python\n{{src_code}}\n``` <|im end|> <|im start|>assistant\nHere is the code:\n```python\n'''
        eos_token = tokenizer.eos_token
        tgt_code_suffix = '''\n```''' + eos_token
        
        max_prompt_len = config.training.max_prompt_len
        max_gen_length = config.training.max_gen_length
        
        # Custom processing: Tokenize without padding
        input_ids_lm = []
        labels_lm = []
        batch_weights = []
        start_pos = [] 
        keep_indices = []
        batch_response_lens = []

        for i, (inp, tgt) in enumerate(zip(input_list, target_list)):
            prompt_text = src_code_prompts_template.replace("{{src_code}}", inp)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            
            if len(prompt_ids) > max_prompt_len:
                continue
                
            target_ids = tokenizer(tgt, add_special_tokens=False).input_ids
            suffix_ids = tokenizer(tgt_code_suffix, add_special_tokens=False).input_ids
            
            w = target_weights_list[i] if target_weights_list else []
            t_weights = torch.ones(len(target_ids), dtype=torch.float32)
            if w:
                l_w = min(len(w), len(target_ids))
                t_weights[:l_w] += torch.tensor(w[:l_w], dtype=torch.float32)
            
            response_ids = target_ids + suffix_ids
            response_weights = torch.cat([t_weights, torch.ones(len(suffix_ids), dtype=torch.float32)])
            
            if len(response_ids) > max_gen_length:
                response_ids = response_ids[:max_gen_length]
                response_weights = response_weights[:max_gen_length]

            curr_input_ids = prompt_ids + response_ids
            curr_labels = [-100] * len(prompt_ids) + response_ids
            curr_weights = [1.0] * len(prompt_ids) + response_weights.tolist()
            
            input_ids_lm.append(torch.tensor(curr_input_ids))
            labels_lm.append(torch.tensor(curr_labels))
            batch_weights.append(torch.tensor(curr_weights))
            start_pos.append(len(prompt_ids))
            keep_indices.append(i)
            batch_response_lens.append(len(response_ids))

        drop_num = len(input_list) - len(keep_indices)
        
        if len(input_ids_lm) > 2:
             logger.info(f"weights: {batch_weights[2]}")
             logger.info(f"input_ids: {input_ids_lm[2]}")
             logger.info(f"input_tokens: {tokenizer.decode(input_ids_lm[2], skip_special_tokens=True)}")
             logger.info(f"labels: {labels_lm[2]}")
             logger.info(f"start_pos: {start_pos[2]}; input_tokens_at_start_pos: {tokenizer.decode(input_ids_lm[2][start_pos[2]], skip_special_tokens=True)}")

        # Step Map filtering
        step_map_list = [step_map_list[i] for i in keep_indices]
        
        batch_step_map = []
        for i, stp in enumerate(step_map_list):
            resp_len = batch_response_lens[i]
            current_step_map = stp[:] if stp else []
             
            if len(current_step_map) < resp_len:
                last_step = current_step_map[-1] if current_step_map else -1
                needed = resp_len - len(current_step_map)
                current_step_map.extend([last_step + 1 + k for k in range(needed)])
             
            current_step_map = current_step_map[:resp_len]
            batch_step_map.append(current_step_map)
             
        # Masking Logic
        pad_id = tokenizer.pad_token_id
        B = len(input_ids_lm)
        device = input_ids_lm[0].device if input_ids_lm else torch.device('cpu')
        
        lower = config.training.lower_p
        upper = config.training.upper_p
        
        noisy_list_out = []
        label_list_out = []
        pmask_list_out = []
        weight_list_out = []
        t_list_out = []
        start_pos_out_list = []

        if config.training.method == "semi-ar":
            for b in range(B):
                base_ids = input_ids_lm[b].to(device)
                L_after = base_ids.size(0)
                current_start_pos = start_pos[b]

                order_list = list(batch_step_map[b])
                order_list = collapse_k_unique(order_list, config.training.block_size)
                order = torch.as_tensor(order_list, device=device)
                
                order_full = torch.full((L_after,), -1, device=device)
                resp_len = len(order)
                order_full[current_start_pos : current_start_pos + resp_len] = order
                
                uniq_steps = torch.unique(order_full[current_start_pos:], sorted=True)
                uniq_steps = uniq_steps[uniq_steps != -1]

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:current_start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L_after, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L_after, dtype=torch.bool, device=device)

                for i in range(0, len(uniq_steps)):
                    block_mask = (order_full == uniq_steps[i])
                    p = torch.empty(L_after, device=device).uniform_(lower, upper)
                    block_mask = (torch.rand(L_after, device=device) < p) & block_mask
                    
                    noisy_ids = base_ids.clone()
                    mask_pos  = (order_full > uniq_steps[i]) | block_mask
                    noisy_ids[mask_pos] = mask_id
                    
                    pmask_this = block_mask & ~tail_pad_b

                    if not pmask_this.any():
                        continue

                    noisy_list_out.append(noisy_ids)
                    label_list_out.append(labels_lm[b])
                    pmask_list_out.append(pmask_this)
                    start_pos_out_list.append(current_start_pos)
                    if batch_weights:
                        weight_list_out.append(batch_weights[b])
                    # t_list_out.append(0.0)
        
        elif config.training.method == "ar":
            for b in range(B):
                base_ids = input_ids_lm[b].to(device)
                L_after = base_ids.size(0)
                current_start_pos = start_pos[b]
                
                p_mask = torch.zeros_like(base_ids, dtype=torch.bool)
                p_mask[current_start_pos:] = True
            
                if config.training.post_num is not None:
                    pad_mask = (base_ids == pad_id)
                    pad_mask[:current_start_pos] = False
                    keep_first_pad = pad_mask & (torch.cumsum(pad_mask.int(), dim=0) <= config.training.post_num)
                    p_mask   = p_mask & (~pad_mask | keep_first_pad)
                
                noisy_list_out.append(base_ids)
                label_list_out.append(labels_lm[b])
                pmask_list_out.append(p_mask)
                start_pos_out_list.append(current_start_pos)
                if batch_weights:
                    weight_list_out.append(batch_weights[b])
                # t_list_out.append(0.0)
        
        elif config.training.method == "random_masking":
            m = config.training.mask_times_per_sample
            
            for b in range(B):
                base_ids  = input_ids_lm[b].to(device)
                label_ids = labels_lm[b]
                L_after = base_ids.size(0)
                current_start_pos = start_pos[b]

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:current_start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L_after, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L_after, dtype=torch.bool, device=device)

                for _ in range(m):
                    t = (upper - lower) * torch.rand(1, device=device) + lower
                    rand_mask = torch.rand(L_after, device=device) < t
                    rand_mask[:current_start_pos] = False
                    rand_mask = rand_mask & ~tail_pad_b

                    if not rand_mask.any():
                        continue

                    noisy_ids = base_ids.clone()
                    noisy_ids[rand_mask]   = mask_id
                    # noisy_ids[tail_pad_b]  = mask_id

                    noisy_list_out.append(noisy_ids)
                    label_list_out.append(label_ids)
                    pmask_list_out.append(rand_mask)
                    start_pos_out_list.append(current_start_pos)
                    if batch_weights:
                        weight_list_out.append(batch_weights[b])
                    t_list_out.append(t.item())

        if not noisy_list_out:
             raise ValueError("No valid data found in the batch")
            
        return noisy_list_out, label_list_out, pmask_list_out, start_pos_out_list, drop_num, weight_list_out, t_list_out
        

    def simple_collate(batch):
        inp, lbl, msk, spos, wgt, t_val = zip(*batch)
        res = {
            "input_ids":  torch.stack(inp),
            "labels":     torch.stack(lbl),
            "p_mask_lm":  torch.stack(msk),
            "start_pos":  torch.tensor(spos),
            "t_values":   torch.tensor(t_val)
        }
        if wgt[0].numel() > 0:
            res["weights"] = torch.stack(wgt)
        return res
    

    
    with open("./data/" + config.dataset.optimization_data, 'r') as f:
        dataset_load = json.load(f)
    dataset_load = dataset_load[:4]
    
    input_list = []
    target_list = []
    step_map_list = []
    target_weights_list = []

    for x in dataset_load:
        input_list.append(x["input"])   # Use 'input' as prompt (src_code)
        target_list.append(x["target"]) # Use 'target' as target (tgt_code)
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(config.training.max_gen_length)])
        else:
            step_map_list.append(x["step_map"])
        target_weights_list.append(x.get("target_token_weights", []))
        # target_tokens_list no longer needed for alignment, but we keep loop structure
       
    # Call new prepare function with lists
    input_ids, labels, p_mask_lm, start_pos, drop_num, final_weights, t_values = prepare_inputs_and_labels_for_text(
        input_list, target_list, step_map_list, target_weights_list
    )

    
    dataset_lm = TrainDataset(input_ids, labels, p_mask_lm, start_pos, final_weights, t_values)

    if config.training.batch_size_lm != 1:
        logger.warning(f"batch_size_lm is not 1, but {config.training.batch_size_lm}. Reset to 1")
        config.training.batch_size_lm = 1

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
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
        num_warmup_steps=num_warmup_steps,
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
    ##################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )

    

    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {len(input_ids)}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    global_step = 0
    data_time_m = AverageMeter()
    end = time.time()

    import torch.nn.functional as F



    def make_causal_attention_mask(input_ids, pad_id, start_pos):
        B, T = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        ).view(1, 1, T, T)

        bias = torch.zeros((B, 1, T, T), device=device)
        bias.masked_fill_(causal_mask, float("-inf"))

        idx = torch.arange(T, device=device)
        pad_prefix_mask = (input_ids == pad_id) & (idx[None, :] < start_pos[:, None])
        bias.masked_fill_(pad_prefix_mask.view(B, 1, 1, T), float("-inf"))

        return bias 



    def make_attention_mask(input_ids, pad_id, pos):
        B, T = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype
        idx = torch.arange(T, device=device)
        keep = ~input_ids.ne(pad_id) & (idx[None, :] <= pos[:, None])  # shape (B, T)
        # Allocate bias of shape (B,1,1,T)
        bias = torch.zeros(B, 1, 1, T, device=device)
        bias.masked_fill_(keep[:, None, None, :], float("-inf"))
        return bias
    

    
    def forward_process(input_ids, labels, p_mask_lm, start_pos, weights=None):
        if config.training.method == "ar":
            #if config.training.batch_size_lm == 1:
            #    logits = model(input_ids, is_causal=True).logits
            attn_mask = make_causal_attention_mask(input_ids, pad_id, start_pos)
            logits = model(input_ids, attention_mask=attn_mask, is_causal=False).logits
        else:
            attention_mask = make_attention_mask(input_ids, pad_id, start_pos)
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
            logger.info(f"{'Pos':<5} | {'Input':<15} | {'Target':<15} | {'Pred':<15} | {'Tgt Prob':<10} | {'Mask':<5} | {'Weight':<6}")
            logger.info("-" * 80)
            
            for t in range(seq_len):
                curr_input_id = shift_inputs[b_idx, t].item()
                curr_target_id = shift_labels[b_idx, t].item()
                curr_pred_id = pred_ids[t].item()
                curr_mask = shift_mask[b_idx, t].item()
                curr_prob = torch.exp(logp_tok[b_idx, t]).item()
                
                curr_weight = 1.0
                if weights is not None:
                    curr_weight = weights[b_idx, t+1].item()

                # Decode
                s_input = tokenizer.decode([curr_input_id])
                s_target = tokenizer.decode([curr_target_id]) if curr_target_id != -100 else "<IGNORE>"
                s_pred = tokenizer.decode([curr_pred_id])

                logger.info(f"{t:<5} | {s_input:<15} | {s_target:<15} | {s_pred:<15} | {curr_prob:.4f}     | {curr_mask:<5} | {curr_weight:<6}")
                
            logger.info("=" * 50)
            # --- Enhanced Logging End ---

        if weights is not None:
            shift_weights = weights[:, 1:]
            loss_lm = - (logp_tok * shift_mask * shift_weights).sum(dim=1)
        else:
            loss_lm = - (logp_tok * shift_mask).sum(dim=1)

        mask_num = (shift_mask).sum(dim=1).clamp(min=1)
        loss_lm = loss_lm / mask_num
        loss_lm = loss_lm.sum() / B
        
        return loss_lm


    

    




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
        mask_ratio_list = []
        t_val_list = []
        for step, batch in enumerate(progress_bar, start=1):

            data_time_m.update(time.time() - end)
            
            input_ids = batch["input_ids"].to(accelerator.device)
            labels    = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)
            start_pos = batch["start_pos"].to(accelerator.device)
            weights   = batch.get("weights", None)
            t_values  = batch.get("t_values", None)
            if weights is not None:
                weights = weights.to(accelerator.device)
            if t_values is not None:
                t_values = t_values.to(accelerator.device)

            # Collect metrics
            batch_length_list.append(input_ids.shape[1])
            mask_num_list.append(p_mask_lm.sum().item())
            
            current_mask_ratio = p_mask_lm.sum().item() / input_ids.numel()
            mask_ratio_list.append(current_mask_ratio)
            
            if t_values is not None:
                t_val_list.extend(t_values.tolist())

            loss_lm = forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    p_mask_lm=p_mask_lm,
                    start_pos=start_pos,
                    weights=weights
                )
            
            accumulated_loss += loss_lm.item() / accelerator.gradient_accumulation_steps
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps
            if step <= 32 and accelerator.is_local_main_process:
                print(f"\n{'='*20} Step {step} Debug Info {'='*20}")
                print(f"Loss: {loss_lm.item()}")
                # Print first sample in the batch
                # print(f"Input IDs (sample 0): {input_ids[0].tolist()}")
                try:
                    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    print(f"Decoded Input (sample 0): {decoded_text}")
                except Exception as e:
                    print(f"Decode failed: {e}")
                # print(f"Labels (sample 0): {labels[0].tolist()}")
                # print(f"P_Mask_LM (sample 0): {p_mask_lm[0].tolist()}")
                print(f"p_mask_lm.sum: {p_mask_lm.sum().item()}")
                print(f"{'='*50}\n")
            accelerator.backward(loss_lm)

            if step % accelerator.gradient_accumulation_steps == 0:
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
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Calculate aggregated metrics
                batch_length_max = max(batch_length_list) if batch_length_list else 0
                batch_length_avg = sum(batch_length_list) / len(batch_length_list) if batch_length_list else 0
                mask_num_max = max(mask_num_list) if mask_num_list else 0
                mask_num_avg = sum(mask_num_list) / len(mask_num_list) if mask_num_list else 0
                
                mask_ratio_max = max(mask_ratio_list) if mask_ratio_list else 0
                mask_ratio_avg = sum(mask_ratio_list) / len(mask_ratio_list) if mask_ratio_list else 0
                t_max = max(t_val_list) if t_val_list else 0
                t_avg = sum(t_val_list) / len(t_val_list) if t_val_list else 0

                accelerator.log({
                    "train_loss": accumulated_loss,
                    "grad_norm_pre_clip": grad_norm_pre,
                    "grad_norm_post_clip": grad_norm_post,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "batch_length_max": batch_length_max,
                    "batch_length_avg": batch_length_avg,
                    "mask_num_max": mask_num_max,
                    "mask_num_avg": mask_num_avg,
                    "mask_ratio_max": mask_ratio_max,
                    "mask_ratio_avg": mask_ratio_avg,
                    "t_max": t_max,
                    "t_avg": t_avg
                }, step=global_step)
                
                accumulated_loss = 0.0
                batch_length_list = []
                mask_num_list = []
                mask_ratio_list = []
                t_val_list = []
                
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

    accelerator.wait_for_everyone()
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
    sys.exit(0)
