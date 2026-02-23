import os
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch

from train.utils import get_config

# 使用 dream 子模块，避免加载完整 model
from models.dream import DreamTokenizer, DreamConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


@torch.no_grad()
def prepare_inputs_and_labels_for_text(
    config,
    tokenizer,
    mask_id,
    input_list,
    target_list,
    step_map_list,
):

    # Templates
    src_code_prompts_template = '''Below is a program. Optimize the program and provide a faster version.\nProgram:\n```python\n{{src_code}}\n```'''
    eos_token = tokenizer.eos_token
    tgt_code_response_template = '''Here is the optimized code:\n```python\n{{tgt_code}}\n```''' + eos_token
    pad_id = tokenizer.pad_token_id

    max_prompt_len = config.training.max_prompt_len
    max_gen_length = config.training.max_gen_length

    # Custom Right-Padding Logic
    input_ids_lm = []
    labels_lm = []
    start_pos_list = []
    keep_indices = []
    batch_response_lens = []

    for i, (inp, tgt) in enumerate(zip(input_list, target_list)):
        prompt_text = [{"role": "user", "content": src_code_prompts_template.replace("{{src_code}}", inp)}]
        prompt_ids = tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True, add_special_tokens=False)

        if len(prompt_ids) > max_prompt_len:
            continue

        response_ids = tokenizer(tgt_code_response_template.replace("{{tgt_code}}", tgt), add_special_tokens=False).input_ids

        if len(response_ids) > max_gen_length:
            response_ids = response_ids[:max_gen_length]

        curr_input_ids = prompt_ids + response_ids
        curr_labels = [-100] * len(prompt_ids) + response_ids

        input_ids_lm.append(torch.tensor(curr_input_ids))
        labels_lm.append(torch.tensor(curr_labels))
        start_pos_list.append(len(prompt_ids))
        keep_indices.append(i)
        batch_response_lens.append(len(response_ids))

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

    if len(input_ids_lm) > 2:
        logger.info(f"input_ids: {input_ids_lm[2]}")
        logger.info(f"input_tokens: {tokenizer.decode(input_ids_lm[2], skip_special_tokens=True)}")
        logger.info(f"labels: {labels_lm[2]}")
        logger.info(f"start_pos: {start_pos_list[2]}; input_tokens_at_start_pos: {tokenizer.decode(input_ids_lm[2][start_pos_list[2]], skip_special_tokens=True)}")
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
    B, L_after = input_ids_lm.shape
    device = input_ids_lm.device

    lower = config.training.lower_p
    upper = config.training.upper_p

    if config.training.method == "semi-ar":

        noisy_list, label_list, pmask_list = [], [], []
        start_pos_out_list = []

        for b in range(B):
            current_start_pos = start_pos_list[b]
            order_list = list(batch_step_map[b])
            order_list = collapse_k_unique(order_list, config.training.block_size)
            order = torch.as_tensor(order_list, device=device)

            order_full = torch.full((L_after,), -1, device=device)
            resp_len = len(order)
            order_full[current_start_pos : current_start_pos + resp_len] = order

            uniq_steps = torch.unique(order_full[current_start_pos:], sorted=True)
            uniq_steps = uniq_steps[uniq_steps != -1]

            base_ids = input_ids_lm[b]

            for i in range(0, len(uniq_steps)):
                block_mask = (order_full == uniq_steps[i])
                p = torch.empty(L_after, device=device).uniform_(lower, upper)
                block_mask = (torch.rand(L_after, device=device) < p) & block_mask

                noisy_ids = base_ids.clone()
                mask_pos  = (order_full > uniq_steps[i]) | block_mask
                noisy_ids[mask_pos] = mask_id

                pmask_this = block_mask # & ~tail_pad_b

                if not pmask_this.any():
                    continue

                noisy_list.append(noisy_ids)
                label_list.append(labels_lm[b])
                pmask_list.append(pmask_this)
                start_pos_out_list.append(current_start_pos)

        if noisy_list:
            noisy_batch = torch.stack(noisy_list)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
            start_pos_list = torch.tensor(start_pos_out_list, device=device)
        else:
            raise ValueError("No valid data found in the batch")

    elif config.training.method == "ar":

        noisy_batch = input_ids_lm
        labels_lm   = input_ids_lm
        start_pos_list = torch.tensor(start_pos_list, device=device)

        p_mask_list = []
        for b in range(B):
            current_start_pos = start_pos_list[b]
            pm = torch.zeros(L_after, dtype=torch.bool, device=device)
            pm[current_start_pos:] = True

            p_mask_list.append(pm)
        p_mask = torch.stack(p_mask_list)

    elif config.training.method == "random_masking":
        m = config.training.mask_times_per_sample

        noisy_list, label_list, pmask_list = [], [], []
        start_pos_out_list = []
        for b in range(B):
            base_ids  = input_ids_lm[b]
            label_ids = labels_lm[b]
            current_start_pos = start_pos_list[b]

            for _ in range(m):
                t = (upper - lower) * torch.rand(1, device=device) + lower
                rand_mask = torch.rand(L_after, device=device) < t
                rand_mask[:current_start_pos] = False
                rand_mask = rand_mask # & ~tail_pad_b

                if not rand_mask.any():
                    continue

                noisy_ids = base_ids.clone()
                noisy_ids[rand_mask]   = mask_id

                noisy_list.append(noisy_ids)
                label_list.append(labels_lm[b])
                pmask_list.append(rand_mask)
                start_pos_out_list.append(current_start_pos)

            # Add full mask sample (mask all non-prompt tokens)
            full_mask = torch.ones(L_after, dtype=torch.bool, device=device)
            full_mask[:current_start_pos] = False

            if full_mask.any():
                noisy_ids_full = base_ids.clone()
                noisy_ids_full[full_mask] = mask_id

                noisy_list.append(noisy_ids_full)
                label_list.append(labels_lm[b])
                pmask_list.append(full_mask)
                start_pos_out_list.append(current_start_pos)

        noisy_batch = torch.stack(noisy_list)
        labels_lm   = torch.stack(label_list)
        p_mask      = torch.stack(pmask_list)
        start_pos_list = torch.tensor(start_pos_out_list, device=device)

    # Filter valid rows (needed for Semi-AR and Random Masking)
    if config.training.method != "ar":
        valid_rows = p_mask.any(dim=1)
        noisy_batch = noisy_batch[valid_rows]
        labels_lm   = labels_lm[valid_rows]
        p_mask      = p_mask[valid_rows]
        start_pos_list = start_pos_list[valid_rows]

    return noisy_batch, labels_lm, p_mask, start_pos_list, drop_num


def main():
    config = get_config()
    pretrained_model = config.model.pretrained_model

    # 输出目录：优先 config.dataset.preprocessed_dir，否则 data/preprocessed/<project>/<wandb_run_name>
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

    # 训练集
    data_path = "./data/" + config.dataset.optimization_data
    if not os.path.isfile(data_path):
        data_path = data_path if data_path.endswith(".json") else data_path + ".json"
    with open(data_path, "r", encoding="utf-8") as f:
        dataset_load = json.load(f)
    if config.dataset.get("train_limit", None) is not None:
        dataset_load = dataset_load[: config.dataset.train_limit]

    input_list = []
    target_list = []
    step_map_list = []
    for x in dataset_load:
        input_list.append(x["input"])
        target_list.append(x["target"])
        step_map_list.append(
            x["step_map"] if "step_map" in x else list(range(config.training.max_gen_length))
        )

    input_ids, labels, p_mask_lm, start_pos, drop_num = prepare_inputs_and_labels_for_text(
        config, tokenizer, mask_id, input_list, target_list, step_map_list
    )

    # 转为 CPU 再保存，便于任意环境加载
    train_data = {
        "input_ids": input_ids.cpu(),
        "labels": labels.cpu(),
        "p_mask_lm": p_mask_lm.cpu(),
        "start_pos": start_pos.cpu(),
    }
    torch.save(train_data, preprocessed_dir / "train.pt")
    num_train = len(input_ids)
    logger.info("Train: num_response=%d, drop_num=%d, num_samples=%d", len(dataset_load), drop_num, num_train)

    # 验证集（可选）
    val_data_path = config.dataset.get("validation_data", None)
    num_val = 0
    if val_data_path:
        val_path = "./data/" + val_data_path
        if not os.path.isfile(val_path):
            val_path = val_path if val_path.endswith(".json") else val_path + ".json"
        with open(val_path, "r", encoding="utf-8") as f:
            val_dataset_load = json.load(f)
        val_dataset_load = val_dataset_load[-50:]  # 与 chat 一致：取最后 50
        val_input_list = [x["input"] for x in val_dataset_load]
        val_target_list = [x["target"] for x in val_dataset_load]
        val_step_map_list = [
            x["step_map"] if "step_map" in x else list(range(config.training.max_gen_length))
            for x in val_dataset_load
        ]
        val_input_ids, val_labels, val_p_mask_lm, val_start_pos, val_drop_num = prepare_inputs_and_labels_for_text(
            config, tokenizer, mask_id, val_input_list, val_target_list, val_step_map_list
        )
        val_data = {
            "input_ids": val_input_ids.cpu(),
            "labels": val_labels.cpu(),
            "p_mask_lm": val_p_mask_lm.cpu(),
            "start_pos": val_start_pos.cpu(),
        }
        torch.save(val_data, preprocessed_dir / "val.pt")
        num_val = len(val_input_ids)
        logger.info("Val: num_response=%d, num_samples=%d", len(val_dataset_load), num_val)

    meta = {
        "drop_num": drop_num,
        "num_train": num_train,
        "num_val": num_val,
        "method": config.training.method,
        "num_response": len(dataset_load),
        "preprocessed_dir": str(preprocessed_dir),
    }
    with open(preprocessed_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved meta.json: %s", meta)
    logger.info("Done. Use config.dataset.preprocessed_dir = %s in sft_dream_train.py.", str(preprocessed_dir))


if __name__ == "__main__":
    main()
