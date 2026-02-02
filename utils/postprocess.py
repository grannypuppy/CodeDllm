import re
import json
import os
from loguru import logger
from tqdm import tqdm

def extract_code_from_output(output: str) -> str:
    """
    Post-process tool to extract Python code from model output.
    
    This function implements the following logic requested by the user:
    1. Prepend "```python" to the output (reconstructing the prompt context).
    2. Extract the content between "```python" and the next "```".
    3. Return the extracted code.
    
    Args:
        output (str): The raw output string from the model (e.g., completion starting after ```python).
        
    Returns:
        str: The extracted and stripped Python code.
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

        logger.warning(f"Could not extract code block from output starting with: {output}...")
        return ""
        
    # 3. Return the stripped code
    return code.strip()

def process_jsonl_file(input_file: str, output_file: str = None):
    """
    Reads a JSONL file, processes the 'output' field of each row using extract_code_from_output,
    and saves the result to a new file.
    
    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str, optional): Path to the output JSONL file. 
                                     If None, overwrites input_file or creates a new one with suffix.
                                     Here we default to saving to input_file if not provided? 
                                     Let's require output_file or assume in-place/suffix to be safe.
                                     Let's use a suffix '_processed' if not provided to avoid accidental overwrite.
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_processed{ext}"
        
    logger.info(f"Processing {input_file} -> {output_file}")
    
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        for line in tqdm(lines, desc="Processing records"):
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
                
                # Handle 'output' field which can be a string or a list of strings
                if "output" in record:
                    outputs = record["output"]
                    if isinstance(outputs, list):
                        record["output"] = [extract_code_from_output(o) for o in outputs]
                    elif isinstance(outputs, str):
                        record["output"] = extract_code_from_output(outputs)
                
                f_out.write(json.dumps(record) + "\n")
                processed_count += 1
                
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON line: {line[:50]}...")
            except Exception as e:
                logger.error(f"Error processing line: {e}")

    logger.info(f"Finished processing {processed_count} records.")

if __name__ == "__main__":
    import fire
    fire.Fire(process_jsonl_file)
