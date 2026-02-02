import os
from eval.evaluator import Evaluator

# 1. 定义文件路径
input_file = "results/test_verified/dream-7b-base_len512_steps512_block32/generation_results_processed.jsonl"
output_dir = "results"
run_name = "test_verified/dream-7b-base_len512_steps512_block32"

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

