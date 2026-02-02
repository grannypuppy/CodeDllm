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

import torch
from transformers import AutoModel, AutoTokenizer

# Configure standard logging to ensure logs from local modules (like model.generation_utils_ast) are shown
import logging as sys_logging
sys_logging.basicConfig(level=sys_logging.INFO)
import time
from models.dream.modeling_dream import DreamModel
from models.dream.tokenization_dream import DreamTokenizer

import types
# Load model and tokenizer

# Config: aligned with gen_eval.sh (lines 15-26)
MODEL_PATH = "../dLLM-RL/local_models/dream-7b-instruct"
MAX_NEW_TOKENS = 512
DIFFUSION_STEPS = 512
BLOCK_SIZE = 32
USE_CACHE = True
TOP_P = None
THRESHOLD = None
TEMPERATURE = 0.1
ALG = "entropy"
ALG_TEMP = 0.1

use_cache = USE_CACHE
if use_cache:
    model_path = MODEL_PATH
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = DreamTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    model = model.to("cuda").eval()

    from models.dream.generation_utils_block import DreamGenerationMixin
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
else:
    model_path = MODEL_PATH
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = DreamTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    model = model.to("cuda").eval()
    
    # # Patch with AST-aware generation utils
    # from models.dream.generation_utils_ast import DreamGenerationMixin
    # model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    # model._sample = types.MethodType(DreamGenerationMixin._sample, model)


print("Single-turn generation with Dream-v0-Instruct-7B")
print("----------------------------------------------")

# Get user input
# user_input = input("Please enter raw input (no template will be added): ")
raw_code = """D,G = list(map(int,input().split()))\n\nP = []\n\nnum_p = 0\n\nfor i in range(1,D+1):\n\n    p,c = list(map(int,input().split()))\n\n    num_p += p\n\n    for j in range(1,p+1):\n\n        P += [(j,i*100*j+c*(j==p))]\n\ndp = [0]*(num_p+1)\n\nfor k,pt in P:\n\n    if k==1:\n\n        t_dp = dp[:]\n\n    for cur in range(1,num_p+1):\n\n        if cur>=k:\n\n            dp[cur] = max(dp[cur], t_dp[cur-k]+pt)\n\nfor i in range(num_p+1):\n\n    if dp[i]>=G:\n\n        print(i)\n\n        break"""

# Aligned with gen_eval.py: apply_chat_template + tgt_code_response_prefix
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

# Generate response (params aligned with gen_eval.sh)
output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=MAX_NEW_TOKENS,
    output_history=True,
    return_dict_in_generate=True,
    steps=DIFFUSION_STEPS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=None,
    alg=ALG,
    alg_temp=ALG_TEMP,
    use_cache=USE_CACHE,
    dual_cache=False,
    block_length=BLOCK_SIZE,
    threshold=THRESHOLD,
    tokenizer=tokenizer,
    generation_tokens_hook_func=generation_tokens_hook_func
)

# Process response
generation = tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist())
generation = generation.split(tokenizer.eos_token)[0].strip()

# Print response
print("\nModel Output:", generation)



