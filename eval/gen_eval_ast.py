import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger
import fire
import types

try:
    from models import DreamModel, DreamTokenizer
    from models.dream.generation_utils_ast import DreamGenerationMixin
    from models.dream.generation_utils_block_ast import DreamGenerationMixin as BlockDreamGenerationMixin
except ImportError as e:
    logger.warning(f"Could not import Dream model components: {e}. Ensure you are in the correct environment and 'model' package is available.")
    DreamModel = None
    DreamTokenizer = None

# Global switch: True = use "Here is the optimized code:\n```python\n" prefix (aligned with sft_dream_dataset_m); False = prompt only
USE_RSP_PREFIX = True

class BenchmarkGenerator:
    def __init__(self, 
                 model_path: str, 
                 output_dir: str,
                 test_data_path: str = "data/python_splits/test_input_verified_baseline.jsonl",
                 use_rsp_prefix: bool = None):
        self.model_path = model_path
        self.output_dir = output_dir
        self.test_data_path = test_data_path
        self.use_rsp_prefix = use_rsp_prefix if use_rsp_prefix is not None else USE_RSP_PREFIX
        
        # Distributed setup
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        else:
            self.device = "cpu"
            
        logger.info(f"Initialized Generator on Rank {self.rank}/{self.world_size} (Local: {self.local_rank}), Device: {self.device}")
        
        self.src_code_prompts_template = '''Below is a program. Optimize the program and provide a faster version.\nProgram:\n```python\n{{src_code}}\n```'''
        # Same prefix as sft_dream_dataset_m.py tgt_code_response_template (before {{tgt_code}})
        self.tgt_code_response_prefix = '''Here is the optimized code:\n```python\n'''

    def load_model(self, use_cache: bool = False):
        logger.info(f"Loading model from {self.model_path}...")
        self.model = DreamModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(self.device)
        print(f"Model loaded to device: {self.model.device}")
        self.tokenizer = DreamTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Patch generation methods for Dream model variants (Diffusion/Block)
        if use_cache:
            self.model.diffusion_generate = types.MethodType(BlockDreamGenerationMixin.diffusion_generate, self.model)
            self.model._sample = types.MethodType(BlockDreamGenerationMixin._sample, self.model)
            logger.info("Using block+AST generation cache")
        else:
            logger.info("Using full generation")
            self.model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, self.model)
            self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)
    def generate(self,
                 num_generations: int = 1,
                 max_new_tokens: int = 2048,
                 temperature: float = 0.0,
                 diffusion_steps: int = 256,
                 use_cache: bool = False,
                 dual_cache: bool = False,
                 block_size: int = 32,
                 threshold: float = None,
                 top_p: float = 0.95,
                 alg_temp: float = 0.1,
                 top_k: int = None,
                 alg: str = "entropy"):
        
        def generation_tokens_hook_func(step, x, logits):
            print(f"\033[32m############ Step {step} After Remasking ############\033[0m")
            # print(tokenizer.decode(h[0].tolist()))
            print(self.tokenizer.decode(x[0].tolist()))
            return x

        if not DreamModel:
            logger.error("DreamModel not imported. Cannot proceed.")
            return

        self.load_model(use_cache)
        
        # Load test data
        if not os.path.exists(self.test_data_path):
            logger.error(f"Test data not found at {self.test_data_path}")
            return

        try:
            data = pd.read_json(self.test_data_path, lines=True, orient="records").to_dict(orient="records")
            logger.info(f"Loaded {len(data)} problems from {self.test_data_path}")
        except Exception as e:
            logger.error(f"Error reading test data: {e}")
            return

        data = data[:100]

        # Split data for distributed execution
        if self.world_size > 1:
            total_samples = len(data)
            data = data[self.rank::self.world_size]
            logger.info(f"Rank {self.rank}: Processing {len(data)}/{total_samples} samples")

        os.makedirs(self.output_dir, exist_ok=True)
        
        results = []
        
        for i, record in enumerate(tqdm(data, desc=f"Generating (Rank {self.rank})")):
            problem_id = record.get("problem_id")
            # Handle input field name variations: check 'input' first, then 'src_code'
            src_code = record.get("input") or record.get("src_code")
            
            if not src_code:
                logger.warning(f"Skipping record {i}: No source code found.")
                continue

            # Prepare prompt (same as sft_dream_dataset_m.py: apply_chat_template [+ tgt_code_response_prefix])
            prompt_text = [{"role": "user", "content": self.src_code_prompts_template.replace("{{src_code}}", src_code)}]
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_text, add_generation_prompt=True, tokenize=True, add_special_tokens=False
            )
            if self.use_rsp_prefix:
                # Append "Here is the optimized code:\n```python\n" prefix (aligned with tgt_code_response_template)
                prefix_ids = self.tokenizer(self.tgt_code_response_prefix, add_special_tokens=False).input_ids
                full_prompt_ids = prompt_ids + prefix_ids
            else:
                full_prompt_ids = prompt_ids
            input_ids = torch.tensor([full_prompt_ids], dtype=torch.long).to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

            generated_outputs = []
            
            for _ in range(num_generations):
                gen_kwargs = {
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "output_history": True,
                    "return_dict_in_generate":True,
                    "steps": diffusion_steps,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "alg": alg,
                    "alg_temp": alg_temp,
                    "use_cache": use_cache,
                    "dual_cache": dual_cache,
                    "block_length": block_size if use_cache else None,
                    "threshold": threshold,
                    "tokenizer": self.tokenizer,
                    # "generation_tokens_hook_func": generation_tokens_hook_func
                }
                if threshold is not None:
                    gen_kwargs["alg"] = "confidence_threshold"

                try:
                    with torch.no_grad():
                        output = self.model.diffusion_generate(
                            input_ids,
                            **gen_kwargs
                        )
                    
                    # Decode
                    sequences = output.sequences if hasattr(output, "sequences") else output
                    # Skip prompt tokens
                    generated_text = self.tokenizer.decode(sequences[0, len(input_ids[0]):].tolist(), skip_special_tokens=True)
                    generated_outputs.append(generated_text)
                    # print(f"Generated Text: {generated_text}")
                except Exception as e:
                    logger.error(f"Generation error for problem {problem_id}: {e}")
                    # INSERT_YOUR_CODE
                    import traceback
                    print(traceback.format_exc())
                    generated_outputs.append("")

            # Construct result record for eval.py
            result_record = {
                "problem_id": problem_id,
                "output": generated_outputs,
                # Preserve other fields
                **{k:v for k,v in record.items() if k not in ["output"]}
            }
            results.append(result_record)

            # Periodic save
            if (i + 1) % 10 == 0:
                 self._save_results(results, f"generation_results_rank{self.rank}_partial.jsonl")

        # Final save
        self._save_results(results, f"generation_results_rank{self.rank}.jsonl")
        
    def _save_results(self, results, filename):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res) + "\n")

def main(
    model_path: str,
    output_dir: str,
    data_path: str = "datasets/python_splits/test_input_verified_baseline.jsonl",
    num_generations: int = 4,
    max_new_tokens: int = 1024,
    diffusion_steps: int = 1024,
    use_cache: bool = False,
    dual_cache: bool = False,
    block_size: int = 32,
    threshold: float = None,
    top_p: float = 0.95,
    alg_temp: float = 0.1,
    temperature: float = 0.1,
    top_k: int = None,
    alg: str = "entropy",
    use_rsp_prefix: bool = None,
):
    """
    Generate solutions for Python benchmarks using Dream model.
    
    Args:
        model_path: Path to the Dream model.
        output_dir: Directory to save results.
        data_path: Path to input JSONL file.
        num_generations: Number of samples per problem.
        max_new_tokens: Max tokens to generate.
        diffusion_steps: Number of diffusion steps.
        use_cache: Enable block generation cache (semi-ar).
        dual_cache: Enable dual cache.
        block_size: Block size for cache.
        threshold: Confidence threshold for generation.
        top_p: Nucleus sampling probability.
        alg_temp: Algorithm temperature for diffusion sampling.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        alg: Generation algorithm.
        use_rsp_prefix: Use "Here is the optimized code:\n```python\n" prefix. Default uses USE_RSP_PREFIX.
    """
    generator = BenchmarkGenerator(
        model_path=model_path,
        output_dir=output_dir,
        test_data_path=data_path,
        use_rsp_prefix=use_rsp_prefix
    )
    generator.generate(
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        diffusion_steps=diffusion_steps,
        use_cache=use_cache,
        dual_cache=dual_cache,
        block_size=block_size,
        threshold=threshold,
        top_p=top_p,
        alg_temp=alg_temp,
        temperature=temperature,
        top_k=top_k,
        alg=alg
    )

if __name__ == "__main__":
    fire.Fire(main)

