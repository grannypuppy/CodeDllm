# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Block generation + AST Weighted Remasking (FULL version)
#
# Difference from generation_utils_block_ast:
# - AST analysis uses the FULL sequence (all mask positions filled with x0), same as generation_utils_ast
# - Prediction and analysis identical to full-sequence version
# - Unmask/transfer restricted to current block only; weight normalization applied only within block
#
# Note: Full-sequence analysis requires x0 for entire remaining sequence.
# - dual_cache=False: we have x0 for current_block_start to end -> full analysis
# - dual_cache=True: we only have x0 for current block -> fallback to block-only analysis

import warnings
import copy
import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from utils.ast_dag import generate_char_weights_from_dag
except ImportError as e:
    logger.warning(f"Import failed in generation_utils_block_ast_full: {e}")
    generate_char_weights_from_dag = None


def extract_code_from_output(output: str) -> str:
    """Extract Python code block from model output (aligned with prompt prefix ```python)."""
    if not output:
        return ""
    output = "```python" + output
    start_tag = "```python"
    start_idx = output.find(start_tag)
    if start_idx == -1:
        logger.warning("Could not find ```python block in output")
        return ""
    start_idx += len(start_tag)
    end_idx = output.find("```", start_idx)
    if end_idx == -1:
        code = output[start_idx:]
    else:
        code = output[start_idx:end_idx]
    return code.strip()


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


def _apply_ast_weighted_remasking_block_only(
    x, mask_index, x0, confidence, input_ids, tokenizer,
    current_block_start, block_length, dual_cache
):
    """
    Block-only analysis: x_temp = prompt + filled blocks + current block filled.
    Same as generation_utils_block_ast.
    """
    if tokenizer is None or generate_char_weights_from_dag is None or extract_code_from_output is None:
        return
    try:
        full_mask = torch.zeros_like(x, dtype=torch.bool)
        seg_len = mask_index.shape[1]
        full_mask[:, current_block_start:current_block_start + seg_len] = mask_index
        x_temp = x.clone()
        x_temp[full_mask] = x0

        prompt_len = input_ids.shape[1]
        for b in range(x.shape[0]):
            full_text = tokenizer.decode(x_temp[b], skip_special_tokens=True)
            generated_text = tokenizer.decode(x_temp[b][prompt_len:], skip_special_tokens=True)
            code_text = extract_code_from_output(generated_text)
            if not code_text:
                continue
            raw_char_weights = generate_char_weights_from_dag(code_text)
            if raw_char_weights is None:
                continue
            depths_array = np.array(raw_char_weights, dtype=float)
            max_depth = np.max(depths_array)
            inverted_depths = (1 + max_depth - depths_array) / (1 + max_depth)
            inverted_depths = inverted_depths / inverted_depths.mean()

            full_text_reconstructed = ""
            start_idx = generated_text.find(code_text)
            if start_idx == -1:
                continue
            end_idx = start_idx + len(code_text)
            token_ids_list = x_temp[b].tolist()
            b_mask = full_mask[b]
            current_mask_indices = torch.nonzero(b_mask).squeeze(-1)
            if current_mask_indices.dim() == 0:
                current_mask_indices = current_mask_indices.unsqueeze(0)
            offset = full_mask[:b].sum().item()
            mean_conf = confidence.mean()

            for j, token_id in enumerate(token_ids_list[prompt_len:]):
                token_idx = prompt_len + j
                token_str = tokenizer.decode([token_id])
                tok_start = len(full_text_reconstructed)
                tok_end = tok_start + len(token_str)
                full_text_reconstructed += token_str
                overlap_start = max(tok_start, start_idx)
                overlap_end = min(tok_end, end_idx)
                if overlap_start >= overlap_end:
                    continue
                mask_pos = (current_mask_indices == token_idx).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    continue
                local_start = overlap_start - start_idx
                local_end = overlap_end - start_idx
                relevant_weights = inverted_depths[local_start:local_end]
                if len(relevant_weights) == 0:
                    continue
                token_weight = float(np.max(relevant_weights))
                conf_idx = offset + mask_pos[0].item()
                confidence[conf_idx] = mean_conf + token_weight * (confidence[conf_idx] - mean_conf)
    except Exception as e:
        logger.warning(f"Weighted Remasking (block-only) failed: {e}")


def _apply_ast_weighted_remasking_full(
    x, mask_index, x0, confidence, input_ids, tokenizer,
    current_block_start, current_block_end, block_length, dual_cache
):
    """
    Full-sequence AST analysis (same as generation_utils_ast):
    - x_temp = full sequence with ALL mask positions filled by x0
    - Decode full sequence, extract code, AST analysis on entire code
    - Apply weight adjustment ONLY to tokens in [current_block_start, current_block_end)
    - mean_conf for formula: normalize only within current block (weights归一化只在 block 里完成)

    Requires x0 for entire remaining sequence -> only works when dual_cache=False.
    When dual_cache=True, falls back to block-only analysis.
    """
    if tokenizer is None or generate_char_weights_from_dag is None or extract_code_from_output is None:
        return
    if dual_cache:
        _apply_ast_weighted_remasking_block_only(
            x, mask_index, x0, confidence, input_ids, tokenizer,
            current_block_start, block_length, dual_cache,
        )
        return
    try:
        full_mask = torch.zeros_like(x, dtype=torch.bool)
        seg_len = mask_index.shape[1]
        full_mask[:, current_block_start:current_block_start + seg_len] = mask_index
        x_temp = x.clone()
        x_temp[full_mask] = x0

        prompt_len = input_ids.shape[1]
        for b in range(x.shape[0]):
            full_text = tokenizer.decode(x_temp[b], skip_special_tokens=True)
            generated_text = tokenizer.decode(x_temp[b][prompt_len:], skip_special_tokens=True)
            code_text = extract_code_from_output(generated_text)
            if not code_text:
                continue
            raw_char_weights = generate_char_weights_from_dag(code_text)
            if raw_char_weights is None:
                continue
            depths_array = np.array(raw_char_weights, dtype=float)
            max_depth = np.max(depths_array)
            inverted_depths = (1 + max_depth - depths_array) / (1 + max_depth)

            full_text_reconstructed = ""
            start_idx = generated_text.find(code_text)
            if start_idx == -1:
                continue
            end_idx = start_idx + len(code_text)
            token_ids_list = x_temp[b].tolist()
            b_mask = full_mask[b]
            all_mask_indices = torch.nonzero(b_mask).squeeze(-1)
            if all_mask_indices.dim() == 0:
                all_mask_indices = all_mask_indices.unsqueeze(0)
            offset = full_mask[:b].sum().item()

            block_conf_values = []
            adjustments = []
            for j, token_id in enumerate(token_ids_list[prompt_len:]):
                token_idx = prompt_len + j
                token_str = tokenizer.decode([token_id])
                tok_start = len(full_text_reconstructed)
                tok_end = tok_start + len(token_str)
                full_text_reconstructed += token_str
                if token_idx < current_block_start or token_idx >= current_block_end:
                    continue
                overlap_start = max(tok_start, start_idx)
                overlap_end = min(tok_end, end_idx)
                if overlap_start >= overlap_end:
                    continue
                mask_pos = (all_mask_indices == token_idx).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    continue
                local_start = overlap_start - start_idx
                local_end = overlap_end - start_idx
                relevant_weights = inverted_depths[local_start:local_end]
                if len(relevant_weights) == 0:
                    continue
                token_weight = float(np.max(relevant_weights))
                conf_idx = offset + mask_pos[0].item()
                block_conf_values.append(confidence[conf_idx].item())
                adjustments.append((conf_idx, token_weight))

            if not adjustments:
                continue
            mean_conf = np.mean(block_conf_values)
            weights_arr = np.array([adj[1] for adj in adjustments], dtype=float)
            weights_normalized = weights_arr / (weights_arr.mean() + 1e-10)

            for idx, (conf_idx, _) in enumerate(adjustments):
                token_weight = weights_normalized[idx]
                confidence[conf_idx] = mean_conf + token_weight * (confidence[conf_idx].item() - mean_conf)
    except Exception as e:
        logger.warning(f"Weighted Remasking (full) failed: {e}")


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)
        if not self._from_model_config:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass


class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        if is_torchdynamo_compiling():
            return
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            raise ValueError(
                f"Input length of input_ids is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior."
            )

    def _prepare_generated_length(
        self, generation_config, has_default_max_length, input_ids_length,
    ):
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` and `max_length` set. `max_new_tokens` will take precedence."
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)
        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(**kwargs)
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id
        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        def _tensor_or_none(token, device=None):
            if token is None:
                return token
            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor}.")
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        tokenizer=None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                f"input_ids on {input_ids.device.type} vs model on {self.device.type}.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id")
            and torch.any(input_ids == generation_config.pad_token_id)
            and attention_mask is None
        ):
            warnings.warn("Padding detected but no attention mask passed.", UserWarning)

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        threshold = kwargs.get("threshold", 0.9)
        block_length = kwargs.get("block_length", 32)
        dual_cache = kwargs.get("dual_cache", False)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            threshold=threshold,
            block_length=block_length,
            dual_cache=dual_cache,
            tokenizer=tokenizer,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = 32,
        dual_cache: bool = False,
        tokenizer=None,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        eos_token_id = generation_config.eos_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        histories = [] if (return_dict_in_generate and output_history) else None

        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        x = generation_tokens_hook_func(None, x, None)
        gen_length = max_length - input_ids.shape[1]

        if block_length is None:
            block_length = gen_length
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        past_key_values = None
        step_idx = 0

        for num_block in range(num_blocks):
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            model_output = self(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits = generation_logits_hook_func(step_idx, x, logits)
            confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            x[:, current_block_start] = x0[:, current_block_start]
            x = generation_tokens_hook_func(step_idx, x, logits)
            step_idx += 1

            if not dual_cache:
                new_past_key_values = []
                for i in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for j in range(len(past_key_values[i])):
                        new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
                past_key_values = new_past_key_values
            else:
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, current_block_start:current_block_end] = 1

            i = 1
            while True:
                if dual_cache:
                    mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                else:
                    mask_index = (x[:, current_block_start:] == mask_token_id)

                if attention_mask != "full":
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = attention_mask

                if dual_cache:
                    model_output = self(
                        x[:, current_block_start:current_block_end],
                        current_attention_mask,
                        tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                        dual_cache=dual_cache,
                        replace_position=replace_position,
                    )
                else:
                    model_output = self(
                        x[:, current_block_start:],
                        current_attention_mask,
                        tok_idx[:, current_block_start:] if tok_idx is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                logits = model_output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits_len = logits.shape[1]
                full_logits = torch.full(
                    (x.shape[0], x.shape[1], logits.shape[-1]),
                    torch.finfo(logits.dtype).min,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                full_logits[:, current_block_start : current_block_start + logits_len, :] = logits
                full_logits = generation_logits_hook_func(step_idx, x, full_logits)
                logits = full_logits[:, current_block_start : current_block_start + logits_len, :]

                if alg == 'confidence_threshold':
                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    _apply_ast_weighted_remasking_full(
                        x, mask_index, x0, confidence, input_ids, tokenizer,
                        current_block_start, current_block_end, block_length, dual_cache,
                    )
                    if dual_cache:
                        x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    x_[mask_index] = x0.clone()
                    full_confidence[mask_index] = confidence
                    full_confidence[:, block_length:] = -torch.inf
                    current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_confidence[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = False
                    if dual_cache:
                        x[:, current_block_start:current_block_end][transfer_index] = x_[transfer_index]
                    else:
                        x[:, current_block_start:][transfer_index] = x_[transfer_index]
                else:
                    if i == steps_per_block:
                        break
                    t = timesteps[i]
                    s = timesteps[i + 1]
                    mask_index_entropy = mask_index.clone()
                    mask_index_entropy[:, block_length:] = False
                    mask_logits = logits[mask_index_entropy]
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    _apply_ast_weighted_remasking_full(
                        x, mask_index_entropy, x0, confidence, input_ids, tokenizer,
                        current_block_start, current_block_end, block_length, dual_cache,
                    )
                    num_mask_token = mask_index_entropy.sum() / mask_index_entropy.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)
                    if dual_cache:
                        full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index_entropy] = confidence
                    full_confidence[:, block_length:] = -torch.inf
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        if dual_cache:
                            x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        else:
                            x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index_entropy] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        if dual_cache:
                            x[:, current_block_start:current_block_end][row_indices, transfer_index] = x_[row_indices, transfer_index]
                        else:
                            x[:, current_block_start:][row_indices, transfer_index] = x_[row_indices, transfer_index]
                    i += 1

                x = generation_tokens_hook_func(step_idx, x, full_logits)
                step_idx += 1
                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    break

            if eos_token_id is not None:
                eos_id = eos_token_id
                if isinstance(eos_id, torch.Tensor):
                    eos_id = eos_id.item() if eos_id.numel() == 1 else eos_id[0].item()
                elif isinstance(eos_id, (list, tuple)):
                    eos_id = eos_id[0] if eos_id else None
                if eos_id is not None:
                    block_tokens = x[:, current_block_start:current_block_end]
                    if (block_tokens == eos_id).all():
                        remaining_mask = (x[:, current_block_end:] == mask_token_id)
                        x[:, current_block_end:] = torch.where(
                            remaining_mask,
                            torch.full_like(x[:, current_block_end:], eos_id),
                            x[:, current_block_end:],
                        )
                        break

        x = generation_tokens_hook_func(-1, x, None)

        if return_dict_in_generate:
            return DreamModelOutput(sequences=x, history=histories)
        return x
