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
from train.prompting_utils import UniversalPrompting
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







import pandas as pd
from functools import partial
import random

def compute_position_id_with_mask(attention_mask):
    position_ids = torch.cumsum(attention_mask, dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids

class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(
        self,
        data_files: Union[str, list],
        tokenizer,
        prompt_key="prompt",
        response_key="response",
        max_length=1024,
        truncation="error",
        pad_token_id=None,
        pad_input=False,
    ):
        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(data_files, list):
            data_files = [data_files]

        self.data_files = data_files
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key

        self.max_length = max_length
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        )
        self.pad_input = pad_input
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            # read jsonl files and cache
            if data_file.endswith(".jsonl") or data_file.endswith(".json"):
                dataframe = pd.read_json(data_file, lines=True if data_file.endswith(".jsonl") else False)
            elif data_file.endswith(".parquet"):
                dataframe = pd.read_parquet(data_file)
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

    @staticmethod
    def _tokenize_static(example, tokenizer, prompt_key, response_key, max_length, truncation, pad_token_id):
        prompt = example[prompt_key]
        response = example[response_key]

        # apply chat template
        if not isinstance(prompt, str):
            prompt_chat = list(prompt)
        else:
            prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(
            prompt_chat_str, return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(
            response_chat_str, return_tensors="pt", add_special_tokens=False
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat(
            (prompt_attention_mask, response_attention_mask), dim=-1
        )

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < max_length:
            padded_input_ids = (
                torch.ones(
                    size=(max_length - sequence_length,), dtype=input_ids.dtype
                )
                * pad_token_id
            )
            padded_attention_mask = torch.ones(  # NOTE: we use 1 here
                size=(max_length - sequence_length,), dtype=attention_mask.dtype
            )

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > max_length:
            if truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-max_length :]
                attention_mask = attention_mask[-max_length :]
            elif truncation == "right":
                input_ids = input_ids[: max_length]
                attention_mask = attention_mask[: max_length]
            elif truncation == "error":
                raise NotImplementedError(
                    f"{sequence_length=} is larger than {max_length=}"
                )
            else:
                raise NotImplementedError(
                    f"Unknown truncation method {truncation}"
                )

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0))] = 0

        return {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
            "loss_mask": loss_mask.numpy(),
        }

    def _tokenize(self, example):
        return self._tokenize_static(
            example,
            self.tokenizer,
            self.prompt_key,
            self.response_key,
            self.max_length,
            self.truncation,
            self.pad_token_id
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        example = self.dataframe.iloc[item]
        data = self._tokenize(example)
        return {
            "input_ids": torch.tensor(data["input_ids"]),
            "attention_mask": torch.tensor(data["attention_mask"]),
            "position_ids": torch.tensor(data["position_ids"]),
            "loss_mask": torch.tensor(data["loss_mask"]),
        }

def q_sample(
    input_ids,
    maskable_mask,
    mask_token_id,
    min=0.0,
    max=1.0,
    eos_token_id=None,
    t=None,
    t_mask=None,
):
    x_0 = input_ids

    if t_mask is None:
        if t is None:
            t = torch.rand((x_0.shape[0],), dtype=torch.float, device=input_ids.device)
            t = min + (max - min) * t
        u = torch.rand_like(x_0, dtype=torch.float)  # t/T prob to mask
        t_mask = (u < t[:, None]) & maskable_mask

    x_t = x_0.masked_fill(t_mask, mask_token_id)

    if eos_token_id is not None:
        # get the last non-eos token index
        last_non_eos_token_idx = ((input_ids != eos_token_id) | (~maskable_mask)).sum(
            dim=-1
        ) - 1
        seq_len = x_0.shape[1]

        for i in range(x_0.shape[0]):
            if last_non_eos_token_idx[i] < seq_len - 1:  # with eos tokens
                t_mask_at_eos = t_mask[
                    i, last_non_eos_token_idx[i] + 1
                ]  # use arbitrary eos token
                # t_mask[i, last_non_eos_token_idx[i] + 2:] = False  # only learn the first eos token
                if t_mask_at_eos:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = mask_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = True
                else:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = eos_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = False

    return x_t, t, t_mask  #  True means it's "MASK" token and should have loss


def context_adaptive_reweight(seq_len, distribution="symmetric-geometric", **kwargs):
    position_ids_l = np.arange(seq_len).reshape(-1, 1)
    position_ids_r = np.arange(seq_len).reshape(1, -1)
    distance = position_ids_l - position_ids_r
    distance = torch.from_numpy(distance)

    def geometric_distribution(k, cart_p=0.8, **kwargs):
        if not 0 < cart_p <= 1:
            raise ValueError("p must be between 0 and 1")

        res = (math.log(cart_p) + (k.abs() - 1) * math.log(1 - cart_p)).exp() * 0.5
        res.masked_fill_(k == 0, 0)  # ignore distance=0
        return res

    if distribution == "symmetric-geometric":
        matrix = geometric_distribution(distance, **kwargs)
    else:
        raise ValueError(f"Unknown distribution {distribution}")

    return matrix

class OptimizedCollateFunction:
    """
    Optimized collate function that completes preprocessing during data loading
    Reduces GPU computation overhead and improves training efficiency
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.pad_eos_token_id = (
            config.data.pad_token_id
            if getattr(config.data, "pad_token_id", None) is not None
            else tokenizer.pad_token_id
        )

        # Cache configuration items to avoid repeated access
        self.enable_perbatch_cutoff = getattr(config.data, "perbatch_cutoff", False)
        self.perbatch_cutoff_type = getattr(config.data, "perbatch_cutoff_type", None)
        self.resp_cutoff_ratio = getattr(config.data, "resp_cutoff_ratio", 0.0)

        self.random = random.Random(42)
        self.np_random = np.random.RandomState(42)

    def __call__(self, batch_samples):
        """
        Efficient collate function implementation
        Args:
            batch_samples: List of samples from dataset
        Returns:
            Preprocessed batch dict
        """
        # Use default collate to merge samples into batch tensors
        from torch.utils.data.dataloader import default_collate

        batch = default_collate(batch_samples)

        # Extract tensors
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].bool()
        position_ids = batch["position_ids"]
        loss_mask = batch["loss_mask"].bool()

        # 1. Handle perbatch_cutoff related logic
        if self.enable_perbatch_cutoff:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_perbatch_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        # 2. Handle response truncation
        if self.resp_cutoff_ratio > 0.0:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_resp_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def _apply_perbatch_cutoff(
        self, input_ids, attention_mask, position_ids, loss_mask
    ):
        """Apply perbatch cutoff logic"""
        if self.perbatch_cutoff_type == "random":
            non_pad_lens = (input_ids != self.pad_eos_token_id).sum(-1).cpu()

            # randomly choose a cutoff length from non_pad_lens
            cutoff_seq_len = np.random.choice(non_pad_lens)

            # cutoff
            input_ids = input_ids[:, :cutoff_seq_len]
            attention_mask = attention_mask[:, :cutoff_seq_len]
            position_ids = position_ids[:, :cutoff_seq_len]
            loss_mask = loss_mask[:, :cutoff_seq_len]
        elif self.perbatch_cutoff_type == "random_with_input_pad":
            prompt_mask = loss_mask == 0
            response_mask = (loss_mask == 1) & (input_ids != self.pad_eos_token_id)

            prompt_lens = prompt_mask.sum(-1)
            response_lens = response_mask.sum(-1)
            max_prompt_len = prompt_lens.max()
            pad_lens = max_prompt_len - prompt_lens

            # randomly choose a response length from response_lens
            kept_response_len = np.random.choice(response_lens.cpu())

            # rebuild input_ids, attention_mask, loss_mask
            new_input_ids = (
                torch.ones(
                    input_ids.shape[0],
                    max_prompt_len + kept_response_len,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                * self.pad_eos_token_id
            )
            new_attention_mask = torch.ones_like(
                new_input_ids, dtype=attention_mask.dtype
            )
            new_loss_mask = torch.ones_like(new_input_ids, dtype=loss_mask.dtype)

            for i in range(input_ids.shape[0]):
                kept_response_len_i = min(kept_response_len, response_lens[i])
                new_input_ids[i, pad_lens[i] : pad_lens[i] + prompt_lens[i]] = (
                    input_ids[i][prompt_mask[i]]
                )
                new_input_ids[
                    i,
                    pad_lens[i]
                    + prompt_lens[i] : pad_lens[i]
                    + prompt_lens[i]
                    + kept_response_len_i,
                ] = input_ids[i][response_mask[i]][:kept_response_len_i]

                new_attention_mask[i, : pad_lens[i]] = 0
                new_loss_mask[i, : pad_lens[i] + prompt_lens[i]] = 0

            input_ids = new_input_ids
            attention_mask = new_attention_mask
            position_ids = compute_position_id_with_mask(new_attention_mask)
            loss_mask = new_loss_mask
        else:
            pad_lens = (input_ids == self.pad_eos_token_id).sum(-1)
            # cutoff_len = eos_lens.min() - 1
            # assert cutoff_len > 0, input_ids
            cutoff_len = pad_lens.min()
            assert cutoff_len >= 0

            # cutoff
            seq_len = input_ids.shape[-1]
            input_ids = input_ids[:, : seq_len - cutoff_len]
            attention_mask = attention_mask[:, : seq_len - cutoff_len]
            position_ids = position_ids[:, : seq_len - cutoff_len]
            loss_mask = loss_mask[:, : seq_len - cutoff_len]

        return input_ids, attention_mask, position_ids, loss_mask

    def _apply_resp_cutoff(self, input_ids, attention_mask, position_ids, loss_mask):
        """Apply response truncation logic"""
        import numpy as np

        if self.np_random.rand() < self.resp_cutoff_ratio:
            # Calculate response length for each sample (loss_mask True portion)
            resp_lens = loss_mask.sum(-1)
            min_resp_len = resp_lens.min().item()

            if min_resp_len > 1:
                # Randomly select truncation length
                cutoff_len = self.np_random.randint(1, min_resp_len)

                # Truncate from the end of sequence
                new_seq_len = input_ids.shape[-1] - cutoff_len
                input_ids = input_ids[:, :new_seq_len].contiguous()
                attention_mask = attention_mask[:, :new_seq_len].contiguous()
                position_ids = position_ids[:, :new_seq_len].contiguous()
                loss_mask = loss_mask[:, :new_seq_len].contiguous()

        return input_ids, attention_mask, position_ids, loss_mask


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

    config.experiment.logging_dir = str( Path("projects") / Path(config.experiment.project) / "logs")
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
            config.experiment.wandb_run_name,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(Path("projects") / config.experiment.project, exist_ok=True)
        config_path = Path("projects") / config.experiment.project / "config.yaml"
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
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)


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




    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    # Initialize SFTDataset
    train_files = config.dataset.get("train_files", config.dataset.get("optimization_data", []))
    if isinstance(train_files, str):
        train_files = [train_files]
    
    # Ensure config.data exists for OptimizedCollateFunction
    if not hasattr(config, "data"):
        config.data = config.dataset

    # If train_files are relative paths, we might need to adjust them.
    # Assuming they are correct for now.

    dataset_lm = SFTDataset(
        data_files=train_files,
        tokenizer=tokenizer,
        prompt_key=config.dataset.get("prompt_key", "prompt"),
        response_key=config.dataset.get("response_key", "response"),
        max_length=config.training.max_gen_length + config.training.max_prompt_len,
        truncation="right",
        pad_token_id=pad_id
    )

    collate_fn = OptimizedCollateFunction(config, tokenizer)

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )

    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training data = {len(dataset_lm)}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    global_step = 0
    data_time_m = AverageMeter()
    end = time.time()

    from tqdm.auto import tqdm
    import torch.nn.functional as F

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
        for step, batch in enumerate(progress_bar, start=1):
            data_time_m.update(time.time() - end)
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            loss_mask = batch["loss_mask"]

            # Forward
            pad_eos_token_id = (
                config.data.pad_token_id
                if getattr(config.data, "pad_token_id", None) is not None
                else tokenizer.pad_token_id
            )

            masked_input_ids, t, loss_mask_nonflatten = q_sample(
                input_ids,
                maskable_mask=loss_mask.bool(),
                mask_token_id=mask_id,
                eos_token_id=(
                    pad_eos_token_id
                    if config.data.get("treat_eos_as_one", False)
                    else None
                ),
            )
            
            # Filter out samples with no masked tokens
            valid_samples = loss_mask_nonflatten.sum(dim=-1) > 0
            if not valid_samples.any():
                continue
            
            input_ids = input_ids[valid_samples]
            attention_mask = attention_mask[valid_samples]
            position_ids = position_ids[valid_samples]
            masked_input_ids = masked_input_ids[valid_samples]
            t = t[valid_samples]
            loss_mask_nonflatten = loss_mask_nonflatten[valid_samples]
            
            loss_mask = loss_mask_nonflatten.reshape(-1)

            # 2d -> 4d conversion for attention_mask
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )

            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
            logits = outputs.logits

            # Loss calculation
            shift_logits = torch.cat(
                [logits[:, 0:1], logits[:, :-1]], dim=1
            ).contiguous()
            shift_labels = input_ids.contiguous()
            
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits, shift_labels)
            
            loss_mask = loss_mask.to(loss.device)
            loss = loss.masked_fill(~loss_mask, 0)
            
            if config.diffusion.get("token_reweighting", False):
                loss = (
                    config.diffusion.alpha
                    * (1 - torch.exp(-loss)) ** config.diffusion.gamma
                    * loss
                )

            if config.diffusion.time_reweighting == "original":
                weight = 1 / (t[:, None].float().expand(input_ids.size()) + 1e-6)
            elif config.diffusion.time_reweighting == "linear":
                weight = 1 - t[:, None].float().expand(input_ids.size())
            elif config.diffusion.time_reweighting == "cart":
                seq_len = input_ids.shape[-1]
                weight_matrix = context_adaptive_reweight(
                    seq_len, cart_p=config.diffusion.cart_p
                )
                _weight_matrix = weight_matrix[:seq_len, :seq_len].to(
                    loss_mask.device
                )
                non_mask = ~loss_mask_nonflatten.to(
                    loss.device
                )
                weight = (
                    non_mask.type_as(_weight_matrix)
                    .matmul(_weight_matrix)
                    .masked_fill(non_mask, 0)
                )
            else:
                weight = (
                    t.new_ones((input_ids.shape[0], 1)).float().expand(input_ids.size())
                )

            loss = loss * weight.reshape(-1)
            
            valid_tokens = loss_mask.sum()
            if valid_tokens > 0:
                loss_lm = loss.sum() / valid_tokens
            else:
                loss_lm = torch.tensor(0.0, device=loss.device, requires_grad=True)

            accumulated_loss += loss_lm.item() / accelerator.gradient_accumulation_steps
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps
            
            if step <= 10 and accelerator.is_local_main_process:
                print(f"Step {step} Loss: {loss_lm.item() * accelerator.gradient_accumulation_steps}")

            accelerator.backward(loss_lm)

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                grad_norm = 0.0
                if config.training.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    if hasattr(grad_norm, 'item'):
                        grad_norm = grad_norm.item()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                accelerator.log({
                    "train_loss": accumulated_loss,
                    "grad": grad_norm,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "batch_length": input_ids.shape[1]
                }, step=global_step)
                accumulated_loss = 0.0
                
                global_step += 1
                
                if config.training.ckpt_interval is not None and global_step % config.training.ckpt_interval == 0:
                    logger.info(f"Saving checkpoint at global step {global_step} ...")
                    accelerator.wait_for_everyone()
                    logger.info("Finish wait_for_everyone")
                    ckpt_name = f"checkpoint-{global_step}"
                    save_checkpoint(model, tokenizer, config, accelerator, ckpt_name)

            end = time.time()

    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    ckpt_name = "final"
    save_checkpoint(model, tokenizer, config, accelerator, ckpt_name)

    accelerator.end_training()






def save_checkpoint(model, tokenizer, config, accelerator, ckpt_name):
    output_dir = Path("projects",config.experiment.project)
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