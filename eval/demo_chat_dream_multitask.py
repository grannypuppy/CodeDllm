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
# Modified from Dream repos: https://github.com/HKUNLP/Dream
# Demo for Dream multitask model with AST-guided generation (generation_utils_ast_ex).

import torch
import types

# Configure logging so AST detail logs from generation_utils_ast_ex are visible
import logging as sys_logging
sys_logging.basicConfig(level=sys_logging.INFO)
import time
from models.dream_multitask import DreamModel
from models.dream_multitask.tokenization_dream import DreamTokenizer

# Config: aligned with gen_dream_eval_2.sh and gen_dream_eval_multitask_ast.py
MODEL_PATH = "projects/sft_dream_py_ast/multitask_stage2_totalmean_0-5_codeonly_2e-6lr_2lm_8ga_4gpus_m7_1000traincases_2epochs/checkpoint-1000"
MAX_NEW_TOKENS = 512
DIFFUSION_STEPS = 512
TOP_P = 0.9
TEMPERATURE = 0.1
ALG = "entropy"
ALG_TEMP = 0.1
THRESHOLD = None  # set to float (e.g. 0.9) to use confidence_threshold
RANK_EPS = 1e-4   # for code-token rank weighting in ast_ex

model_path = MODEL_PATH
model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = DreamTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
model = model.to("cuda").eval()

# Patch with AST-ex generation (code-token rank weighting, no DAG; uses rank head + extract_code_from_output)
from models.dream_multitask.generation_utils_ast_ex import DreamGenerationMixin
model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
model._sample = types.MethodType(DreamGenerationMixin._sample, model)


print("Single-turn generation with Dream Multitask (AST-ex: code-token rank weighting)")
print("-------------------------------------------------------------------------------")

# Get user input
raw_code = """N,M = list(map(int,input().split()))\n\nif N==1:\n\n    nmax=M\n\nelse:\n\n    nmax = 1\n\n    for i in range(2,int(M**0.5)+1):\n\n        if M%i==0 and M//i>=N:\n\n            nmax = max(nmax,i)\n\n        if M%i==0 and i>=N:\n\n            nmax = max(nmax,M//i)\n\nprint(nmax)"""

# Aligned with gen_dream_eval_multitask_ast: apply_chat_template + tgt_code_response_prefix
USE_RSP_PREFIX = True
src_code_prompts_template = '''Below is a program. Optimize the program and provide a faster version.\nProgram:\n```python\n{{src_code}}\n```'''
tgt_code_response_prefix = '''Here is the optimized code:\n```python\n'''

prompt_text = [{"role": "user", "content": src_code_prompts_template.replace("{{src_code}}", raw_code)}]
prompt_ids = tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True, add_special_tokens=False)
if USE_RSP_PREFIX:
    prefix_ids = tokenizer(tgt_code_response_prefix, add_special_tokens=False).input_ids
    full_prompt_ids = prompt_ids + prefix_ids
else:
    full_prompt_ids = prompt_ids

input_ids = torch.tensor([full_prompt_ids], device="cuda")
attention_mask = torch.ones_like(input_ids, device="cuda")


def generation_tokens_hook_func(step, x, logits):
    print(f"\033[32m############ Step {step} After Remasking ############\033[0m")
    print(tokenizer.decode(x[0].tolist()))
    return x

# Generate response: kwargs aligned with gen_dream_eval_multitask_ast / gen_dream_eval_2.sh
gen_kwargs = {
    "attention_mask": attention_mask,
    "max_new_tokens": MAX_NEW_TOKENS,
    "output_history": True,
    "return_dict_in_generate": True,
    "steps": DIFFUSION_STEPS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": None,
    "alg": ALG,
    "alg_temp": ALG_TEMP,
    "tokenizer": tokenizer,
    "threshold": THRESHOLD,
    "rank_eps": RANK_EPS,
    "generation_tokens_hook_func": generation_tokens_hook_func,
}
if THRESHOLD is not None:
    gen_kwargs["alg"] = "confidence_threshold"

output = model.diffusion_generate(input_ids, **gen_kwargs)

# Decode same as gen_dream_eval_multitask_ast: skip prompt, then decode with skip_special_tokens
sequences = output.sequences if hasattr(output, "sequences") else output
generation = tokenizer.decode(sequences[0, len(input_ids[0]):].tolist(), skip_special_tokens=True)
generation = generation.split(tokenizer.eos_token)[0].strip() if getattr(tokenizer, "eos_token", None) else generation.strip()

# Print response
print("\nModel Output:", generation)
