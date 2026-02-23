import os
from eval.evaluator import Evaluator

# 1. 定义文件路径
input_file = "results/test_verified/dream-7b-instruct_len512_steps512_block512/generation_results_processed.jsonl"
output_dir = "results"
run_name = "test_verified/dream-7b-instruct_len512_steps512_block512"

import argparse

parser = argparse.ArgumentParser(description="Batch evaluation for code generation results.")
parser.add_argument('--input_file', type=str, required=True, help='Path to the processed generation results (.jsonl)')
parser.add_argument('--output_dir', type=str, default="results", help='Directory to save batch eval results')
parser.add_argument('--run_name', type=str, required=True, help='Name for this eval run (used as subdirectory)')

args = parser.parse_args()

input_file = args.input_file
output_dir = args.output_dir
run_name = args.run_name

# 2. 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"Error: Input file not found at {input_file}")
    exit(1)

print(f"Starting batch evaluation test...")
print(f"Input: {input_file}")
print(f"Output Dir: {output_dir}/{run_name}")

# 3. 初始化 Evaluator
evaluator = Evaluator(run_name=run_name)

try:
    evaluator.batch_evaluate_jsonl(
        input_file=input_file,
        output_dir=output_dir,
        num_workers=1  # 并行进程数
    )
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"\nAn error occurred during batch evaluation: {e}")

