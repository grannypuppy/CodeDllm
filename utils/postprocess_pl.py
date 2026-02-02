import os
import json
import re
import glob
import fire
from loguru import logger
from tqdm import tqdm

def extract_code_from_output(output: str) -> str:
    """
    Post-process tool to extract Python code from model output.
    Copied from ast-dllm/utils/postprocess.py for standalone usage.
    """
    if not output:
        return ""

    # 1. Prepend the opening tag
    prefix = "```python"
    text_with_header = prefix + output
    
    # 2. Extract content between ```python and the next ```
    # re.DOTALL allows . to match newlines
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text_with_header, re.DOTALL)
    
    if match:
        code = match.group(1)
    else:
        # Fallback or strict? Original script returns "" with warning, but let's be silent here or consistent.
        # logger.warning(f"Could not extract code block from output starting with: {output[:50]}...")
        return ""
        
    # 3. Return the stripped code
    return code.strip()

def merge_and_process(
    results_dir: str,
    output_file: str = "batch_results.jsonl",
    world_size: int = None
):
    """
    Merges parallel generation results from gen_eval.py, processes the code extraction,
    and saves to a single JSONL file with the correct original order.

    Args:
        results_dir: Directory containing generation_results_rank*.jsonl files.
        output_file: Path for the final merged output.
        world_size: Optional. If not provided, inferred from the maximum rank found.
                    Important for correctly interleaving results.
    """
    
    # 1. Find all rank files
    pattern = os.path.join(results_dir, "generation_results_rank*.jsonl")
    files = glob.glob(pattern)
    files = [f for f in files if "_partial" not in f] # Exclude partial files
    
    if not files:
        logger.error(f"No generation_results_rank*.jsonl files found in {results_dir}")
        return

    logger.info(f"Found {len(files)} rank files.")

    # 2. Identify ranks and infer world_size
    rank_files = {}
    for f in files:
        filename = os.path.basename(f)
        match = re.search(r"generation_results_rank(\d+).jsonl", filename)
        if match:
            rank = int(match.group(1))
            rank_files[rank] = f
    
    max_rank = max(rank_files.keys())
    
    if world_size is None:
        world_size = max_rank + 1
        logger.info(f"Inferred world_size: {world_size} (max rank found: {max_rank})")
    else:
        logger.info(f"Using provided world_size: {world_size}")
        
    if len(rank_files) != world_size:
        logger.warning(f"Expected {world_size} files but found {len(rank_files)}. Some ranks might be missing.")

    # 3. Read and place records in global order
    # The generation script splits data using: data[rank::world_size]
    # So the i-th record in rank r corresponds to global index: i * world_size + rank
    
    all_records = {} # global_index -> record
    
    for rank, filepath in rank_files.items():
        logger.info(f"Processing rank {rank} from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                    
                try:
                    record = json.loads(line)
                    
                    # Calculate global index
                    global_index = i * world_size + rank
                    
                    # Process output code extraction
                    if "output" in record:
                        outputs = record["output"]
                        if isinstance(outputs, list):
                            record["output"] = [extract_code_from_output(o) for o in outputs]
                        elif isinstance(outputs, str):
                            record["output"] = extract_code_from_output(outputs)
                    
                    all_records[global_index] = record
                    
                except json.JSONDecodeError:
                    logger.error(f"JSON error in {filepath} line {i+1}")
                    
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")

    # 4. Write merged results
    sorted_indices = sorted(all_records.keys())
    
    if not sorted_indices:
        logger.error("No records found to write.")
        return

    # Check for gaps (optional warning)
    if len(sorted_indices) != (sorted_indices[-1] + 1):
        logger.warning(f"Detected gaps in indices. Max index: {sorted_indices[-1]}, Count: {len(sorted_indices)}")
    
    logger.info(f"Writing {len(all_records)} merged records to {output_file}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in sorted_indices:
            f.write(json.dumps(all_records[idx]) + "\n")
            
    logger.info("Done.")

if __name__ == "__main__":
    fire.Fire(merge_and_process)

