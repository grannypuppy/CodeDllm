"""
SFT Dream dataset for SVEN security hardening (mode A) with AST-M masking.

This script is a SVEN-aware variant of `train/sft_dream_dataset_ast_m.py`:
  - User prompt: SVEN security hardening instructions + vulnerable code block.
  - Assistant response: fixed prefix + target code inside a single fenced block.
  - AST masking (tree-sitter python) is applied only to Python targets.

Expected input JSON (from scripts/build_sven_sft_json.py):
  [
    { "input": <vulnerable code>, "target": <patched code>, "vul_type": "cwe-089", ... },
    ...
  ]
"""

import os
import sys
import json
import logging
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch

from train.utils import get_config

try:
    from utils import set_seed
except ImportError:
    from CodeDllm.utils import set_seed

from models.dream import DreamTokenizer, DreamConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


RANDOM_MASKING_RATIOS = [0.2, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9]

# ----------------------------
# Templates (must be consistent)
# ----------------------------

def _normalize_lang(lang: str) -> str:
    s = str(lang or "python").strip().lower()
    if s in {"py", "python"}:
        return "python"
    if s in {"c"}:
        return "c"
    if s in {"cpp", "c++", "cc", "cxx"}:
        return "cpp"
    return "python"


def _fence_for_lang(lang: str) -> str:
    return _normalize_lang(lang)


def _get_response_prefix(lang: str) -> str:
    fence = _fence_for_lang(lang)
    return f"Here is the hardened code:\n```{fence}\n"


def _get_response_suffix() -> str:
    return "\n```"


def _get_full_response_string(tgt_code: str, eos_token: str, lang: str) -> str:
    """Build full response = prefix + tgt_code + suffix + eos."""
    return _get_response_prefix(lang) + tgt_code + _get_response_suffix() + eos_token


SVEN_SRC_CODE_PROMPTS_TEMPLATE = (
    "You are hardening code against security vulnerabilities.\n"
    "The following code contains a {vul_type} vulnerability.\n"
    "Rewrite the code to fix this specific vulnerability.\n"
    "Output only the fixed code in a single fenced block.\n\n"
    "Vulnerable code:\n```{fence}\n{{src_code}}\n```\n"
)


try:
    from utils.ast_token_mapping import (
        get_response_token_to_node_mapping,
        get_code_token_indices_set,
    )
except ImportError:
    from CodeDllm.utils.ast_token_mapping import (
        get_response_token_to_node_mapping,
        get_code_token_indices_set,
    )


@torch.no_grad()
def prepare_inputs_and_labels_for_text(
    config,
    tokenizer,
    mask_id,
    input_list,
    target_list,
    vul_type_list,
    lang_list,
    step_map_list,
):
    """
    AST split masking: response = code + non-code (prefix/suffix/pad/eos).
    - num_maskable = L_after - start_pos (full response including padding, same as sft_dream_dataset_ast_m).
    - Code tokens (AST-mapped): random selection within code group.
    - Non-code tokens: random selection like sft_dream_dataset_ast_m.
    """

    eos_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    max_prompt_len = config.training.max_prompt_len
    max_gen_length = config.training.max_gen_length
    input_ids_lm = []
    labels_lm = []
    start_pos_list = []
    keep_indices = []
    batch_response_lens = []
    batch_node_mappings = []
    batch_ast_node_info = []
    ast_success_count_by_lang = defaultdict(int)
    ast_fail_count_by_lang = defaultdict(int)

    # Zip (inp, tgt, vul_type) for consistent prompt rendering.
    for i, (inp, tgt, vul_type, lang_raw) in enumerate(
        zip(input_list, target_list, vul_type_list, lang_list)
    ):
        lang = _normalize_lang(lang_raw)
        fence = _fence_for_lang(lang)
        prompt_text = [
            {
                "role": "user",
                # Note: SVEN_SRC_CODE_PROMPTS_TEMPLATE uses `{{src_code}}` so we don't
                # accidentally let `.format()` treat it as a placeholder. After
                # `.format(vul_type=...)`, it becomes `{src_code}` literal; then we
                # replace that with the actual vulnerable code.
                "content": SVEN_SRC_CODE_PROMPTS_TEMPLATE.format(vul_type=vul_type, fence=fence).replace(
                    "{src_code}", inp
                ),
            }
        ]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_text, add_generation_prompt=True, tokenize=True, add_special_tokens=False
        )
        if len(prompt_ids) > max_prompt_len:
            continue

        full_response_string = _get_full_response_string(tgt_code=tgt, eos_token=eos_token, lang=lang)
        code_start_in_full = len(_get_response_prefix(lang))
        response_ids = tokenizer(full_response_string, add_special_tokens=False).input_ids
        if len(response_ids) > max_gen_length:
            response_ids = response_ids[:max_gen_length]

        curr_input_ids = prompt_ids + response_ids
        curr_labels = [-100] * len(prompt_ids) + response_ids

        input_ids_lm.append(torch.tensor(curr_input_ids))
        labels_lm.append(torch.tensor(curr_labels))
        start_pos_list.append(len(prompt_ids))
        keep_indices.append(i)
        batch_response_lens.append(len(response_ids))

        node_mapping = None
        ast_info = []
        try:
            node_to_tokens, ast_info = get_response_token_to_node_mapping(
                tgt_code=tgt,
                full_response_string=full_response_string,
                code_start_in_full=code_start_in_full,
                tokenizer=tokenizer,
                lang=lang,
                max_response_tokens=len(response_ids),
            )
            node_mapping = node_to_tokens
            if ast_info:
                ast_success_count_by_lang[lang] += 1
            else:
                ast_fail_count_by_lang[lang] += 1
        except Exception as e:
            ast_fail_count_by_lang[lang] += 1
            logger.warning("AST mapping failed for sample %d: %s", i, e)
        batch_node_mappings.append(node_mapping)
        batch_ast_node_info.append(ast_info)

    drop_num = len(input_list) - len(keep_indices)
    if not input_ids_lm:
        raise ValueError("No valid data found in the batch (all prompts too long or empty).")

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

    # Align step_map with kept indices (even if training currently ignores it, keep behavior consistent).
    step_map_list = [step_map_list[j] for j in keep_indices]

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

    ratios = config.training.get("random_masking_ratios", RANDOM_MASKING_RATIOS)

    noisy_list, label_list, pmask_list = [], [], []
    start_pos_out_list = []
    sample_to_original_list = []

    # B, L_after are after padding.
    B, L_after = input_ids_lm.shape
    device = input_ids_lm.device

    # Prepare masked examples.
    for b in range(B):
        base_ids = input_ids_lm[b]
        current_start_pos = start_pos_list[b]
        ast_info = batch_ast_node_info[b]
        resp_len = batch_response_lens[b]

        # full response region including padding/eos
        num_maskable = L_after - current_start_pos
        if num_maskable <= 0:
            continue

        # Split code vs non-code within response region.
        code_token_indices = get_code_token_indices_set(ast_info, resp_len)
        non_code_indices = [i for i in range(num_maskable) if i not in code_token_indices]
        num_code = len(code_token_indices)
        num_noncode = len(non_code_indices)

        for ratio in ratios:
            num_to_mask_code = max(0, int(ratio * num_code))
            num_to_mask_noncode = max(0, int(ratio * num_noncode))
            if num_to_mask_code == 0 and num_to_mask_noncode == 0:
                continue

            code_mask_global = []
            if num_code > 0 and num_to_mask_code > 0:
                code_indices = list(code_token_indices)
                perm = torch.randperm(num_code, device=device)
                for i in range(min(num_to_mask_code, num_code)):
                    resp_local = code_indices[perm[i].item()]
                    code_mask_global.append(current_start_pos + resp_local)

            noncode_mask_global = []
            if num_noncode > 0 and num_to_mask_noncode > 0:
                perm = torch.randperm(num_noncode, device=device)
                for i in range(min(num_to_mask_noncode, num_noncode)):
                    resp_local = non_code_indices[perm[i].item()]
                    noncode_mask_global.append(current_start_pos + resp_local)

            all_mask_indices = list(code_mask_global) + noncode_mask_global
            if not all_mask_indices:
                continue

            rand_mask = torch.zeros(L_after, dtype=torch.bool, device=device)
            for idx in all_mask_indices:
                if idx < L_after:
                    rand_mask[idx] = True

            noisy_ids = base_ids.clone()
            noisy_ids[rand_mask] = mask_id

            noisy_list.append(noisy_ids)
            label_list.append(labels_lm[b])
            pmask_list.append(rand_mask)
            start_pos_out_list.append(current_start_pos)
            sample_to_original_list.append(b)

        # Full mask example: mask all response tokens (same style as original AST-M dataset).
        full_mask = torch.ones(L_after, dtype=torch.bool, device=device)
        full_mask[:current_start_pos] = False
        if full_mask.any():
            noisy_ids_full = base_ids.clone()
            noisy_ids_full[full_mask] = mask_id
            noisy_list.append(noisy_ids_full)
            label_list.append(labels_lm[b])
            pmask_list.append(full_mask)
            start_pos_out_list.append(current_start_pos)
            sample_to_original_list.append(b)

    if noisy_list:
        noisy_batch = torch.stack(noisy_list)
        labels_lm = torch.stack(label_list)
        p_mask = torch.stack(pmask_list)
        start_pos_list = torch.tensor(start_pos_out_list, device=device)
        sample_to_original_idx = sample_to_original_list
    else:
        raise ValueError("No valid samples for split random masking. Check data and config.")

    valid_rows = p_mask.any(dim=1)
    noisy_batch = noisy_batch[valid_rows]
    labels_lm = labels_lm[valid_rows]
    p_mask = p_mask[valid_rows]
    start_pos_list = start_pos_list[valid_rows]
    sample_to_original_idx = [
        sample_to_original_list[i] for i in range(len(sample_to_original_list)) if valid_rows[i].item()
    ]

    return (
        noisy_batch,
        labels_lm,
        p_mask,
        start_pos_list,
        drop_num,
        batch_ast_node_info,
        sample_to_original_idx,
        {
            "ast_success_count_by_lang": dict(ast_success_count_by_lang),
            "ast_fail_count_by_lang": dict(ast_fail_count_by_lang),
        },
    )


def main():
    config = get_config()
    seed = config.training.get("seed", None)
    if seed is not None:
        set_seed(seed)
    pretrained_model = config.model.pretrained_model

    preprocessed_dir = config.dataset.get(
        "preprocessed_dir",
        Path("data") / "preprocessed" / config.experiment.project / config.experiment.wandb_run_name,
    )
    preprocessed_dir = Path(preprocessed_dir)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preprocessed data will be saved to: %s", preprocessed_dir)

    tokenizer = DreamTokenizer.from_pretrained(pretrained_model)
    model_config = DreamConfig.from_pretrained(pretrained_model)
    mask_id = model_config.mask_token_id

    data_path = "./data/" + config.dataset.optimization_data
    if not os.path.isfile(data_path):
        data_path = data_path if data_path.endswith(".json") else data_path + ".json"

    with open(data_path, "r", encoding="utf-8") as f:
        dataset_load = json.load(f)
    if config.dataset.get("train_limit", None) is not None:
        dataset_load = dataset_load[: config.dataset.train_limit]

    input_list = []
    target_list = []
    vul_type_list = []
    lang_list = []
    step_map_list = []

    for x in dataset_load:
        input_list.append(x["input"])
        target_list.append(x["target"])
        vul_type_list.append(x.get("vul_type", "unknown"))
        lang_list.append(x.get("lang", "python"))
        step_map_list.append(
            x["step_map"] if "step_map" in x else list(range(config.training.max_gen_length))
        )

    (
        input_ids,
        labels,
        p_mask_lm,
        start_pos,
        drop_num,
        ast_node_info,
        sample_to_original_idx,
        ast_stats_train,
    ) = prepare_inputs_and_labels_for_text(
        config, tokenizer, mask_id, input_list, target_list, vul_type_list, lang_list, step_map_list
    )

    train_data = {
        "input_ids": input_ids.cpu(),
        "labels": labels.cpu(),
        "p_mask_lm": p_mask_lm.cpu(),
        "start_pos": start_pos.cpu(),
        "ast_node_info": ast_node_info,
        "sample_to_original_idx": sample_to_original_idx,
    }
    torch.save(train_data, preprocessed_dir / "train.pt")
    num_train = len(input_ids)
    logger.info(
        "Train: num_response=%d, drop_num=%d, num_samples=%d",
        len(dataset_load),
        drop_num,
        num_train,
    )

    num_val = 0
    val_data_path = config.dataset.get("validation_data", None)
    if val_data_path:
        val_path = "./data/" + val_data_path
        if not os.path.isfile(val_path):
            val_path = val_path if val_path.endswith(".json") else val_path + ".json"

        with open(val_path, "r", encoding="utf-8") as f:
            val_dataset_load = json.load(f)

        # Keep the same behavior as the original script: only use the last 50.
        val_dataset_load = val_dataset_load[-50:]

        val_input_list = [x["input"] for x in val_dataset_load]
        val_target_list = [x["target"] for x in val_dataset_load]
        val_vul_type_list = [x.get("vul_type", "unknown") for x in val_dataset_load]
        val_lang_list = [x.get("lang", "python") for x in val_dataset_load]
        val_step_map_list = [
            x["step_map"] if "step_map" in x else list(range(config.training.max_gen_length))
            for x in val_dataset_load
        ]

        (
            val_input_ids,
            val_labels,
            val_p_mask_lm,
            val_start_pos,
            val_drop_num,
            val_ast_node_info,
            val_sample_to_original_idx,
            ast_stats_val,
        ) = prepare_inputs_and_labels_for_text(
            config,
            tokenizer,
            mask_id,
            val_input_list,
            val_target_list,
            val_vul_type_list,
            val_lang_list,
            val_step_map_list,
        )

        val_data = {
            "input_ids": val_input_ids.cpu(),
            "labels": val_labels.cpu(),
            "p_mask_lm": val_p_mask_lm.cpu(),
            "start_pos": val_start_pos.cpu(),
            "ast_node_info": val_ast_node_info,
            "sample_to_original_idx": val_sample_to_original_idx,
        }

        torch.save(val_data, preprocessed_dir / "val.pt")
        num_val = len(val_input_ids)
        logger.info("Val: num_response=%d, num_samples=%d", len(val_dataset_load), num_val)

    meta = {
        "drop_num": drop_num,
        "num_train": num_train,
        "num_val": num_val,
        "method": config.training.get("method", "ast_depth_masking"),
        "num_response": len(dataset_load),
        "preprocessed_dir": str(preprocessed_dir),
        "model_path": pretrained_model,
        "dataset_mode": "sven_security_hardening_A",
        "ast_success_count_by_lang": ast_stats_train.get("ast_success_count_by_lang", {}),
        "ast_fail_count_by_lang": ast_stats_train.get("ast_fail_count_by_lang", {}),
        "val_ast_success_count_by_lang": (
            ast_stats_val.get("ast_success_count_by_lang", {}) if val_data_path else {}
        ),
        "val_ast_fail_count_by_lang": (
            ast_stats_val.get("ast_fail_count_by_lang", {}) if val_data_path else {}
        ),
    }
    with open(preprocessed_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved meta.json: %s", meta)
    logger.info("Done. Use config.dataset.preprocessed_dir in sft_dream_train.py.")


if __name__ == "__main__":
    main()

