import argparse
import json
import numpy as np
import os
import sys
# from tabulate import tabulate

def load_jsonl(file_path):
    print(f"Loading {file_path}...")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    """
    Write a list of Python objects to JSONL.
    """
    with open(file_path, 'w') as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")

def get_metrics(result_entry):
    """
    Extract correctness, runtime, and compilation status from a single result entry.
    Returns: (correctness, runtime, compiled)
    """
    overview = result_entry.get("completion_results_overview", {})
    if not overview:
        return False, float('inf'), False
    
    correctness = overview.get("correctness", False)
    if isinstance(correctness, str):
        correctness = correctness.lower() == 'true'
    
    compilation_error = overview.get("compilation_error", False)
    if isinstance(compilation_error, str):
        compilation_error = compilation_error.lower() == 'true'
    compiled = not compilation_error
    
    # Try to get runtime, could be 'avg_run_time' or just 'runtime' depending on format variations
    runtime = overview.get("avg_run_time")
    if runtime is None:
        runtime = overview.get("runtime")
    if runtime is None:
        runtime = float('inf')
        
    return correctness, float(runtime), compiled

def geometric_mean(data):
    """
    Calculate geometric mean of a list of numbers.
    Ignores non-positive values.
    """
    a = np.array(data)
    a = a[a > 0]
    if len(a) == 0:
        return 0.0
    return np.exp(np.mean(np.log(a)))

def analyze_results(baseline_file, target_file, nrows2analyze, augmented_output=None):
    baseline_data = load_jsonl(baseline_file)
    target_data = load_jsonl(target_file)
    
    if nrows2analyze is not None:
        baseline_data = baseline_data[:nrows2analyze]
        target_data = target_data[:nrows2analyze]

    if len(baseline_data) != len(target_data):
        print(f"Warning: File lengths differ! Baseline: {len(baseline_data)}, Target: {len(target_data)}")
        print("Analysis will proceed with matching rows row-by-row up to the length of the shorter file.")
    
    # Structure: list of sample metrics for each row (task)
    metrics_all_rows = []
    # Keep aligned augmented entries to facilitate later export by row
    augmented_entries = []
    # mismatched_count = 0
    
    # Determine K from the first valid entry in target_data
    k = 0
    for entry in target_data:
        results = entry.get("eval_results", [])
        if results:
            k = len(results)
            print(f"Detected k={k} from the first valid entry.")
            break
            
    if k == 0:
        print("Error: Could not determine k (no eval_results found in target file).")
        return

    # Iterate simultaneously over baseline and target
    for i, (base_entry, target_entry) in enumerate(zip(baseline_data, target_data)):
        
        # --- Validation Start ---
        base_pid = base_entry.get("problem_id")
        target_pid = target_entry.get("problem_id")
        
        # Check problem_id
        if base_pid != target_pid:
            print(f"[Row {i}] Error: problem_id mismatch! Baseline='{base_pid}' vs Target='{target_pid}'. Skipping.")
            # mismatched_count += 1
            continue
            
        # Check input (first 10 chars)
        base_input = base_entry.get("input", "")
        target_input = target_entry.get("input", "")
        
        # Handle cases where input might not be string (though unlikely in this dataset)
        base_input_prefix = str(base_input)[:10]
        target_input_prefix = str(target_input)[:10]
        
        if base_input_prefix != target_input_prefix:
            print(f"[Row {i}] Error: input mismatch! Baseline='{base_input_prefix}...' vs Target='{target_input_prefix}...'. Skipping.")
            # mismatched_count += 1
            continue
        # --- Validation End ---

        # 标记该行输出是否与输入完全一致，用于后续 speedup 纠正
        raw_outputs = target_entry.get("output", [])
        if not isinstance(raw_outputs, list):
            raw_outputs = [raw_outputs]
        is_identity_output = [str(o) == str(target_input) for o in raw_outputs]
        target_entry["output_equals_input"] = is_identity_output

        # 1. Process Baseline Row
        base_eval_results = base_entry.get("eval_results", [])
        base_correct = False
        base_runtime = float('inf')
        
        if base_eval_results:
            # Assume first result is the baseline reference
            base_correct, base_runtime, _ = get_metrics(base_eval_results[0])
            
        # 2. Process Target Row
        target_eval_results = target_entry.get("eval_results", [])
        
        # Collect metrics for all samples for this row
        row_samples_metrics = []
        for i, res in enumerate(target_eval_results):
            correct, runtime, compiled = get_metrics(res)
            
            speedup = 0.0
            # Calculate speedup only if both are correct and valid runtimes exist
            # 存储原始值到 augmented 文件；聚合统计时再取 max(1, speedup)
            if correct and base_correct and base_runtime > 0 and runtime > 0:
                speedup = base_runtime / runtime
            
            # 如果模型只是原样输出输入，视为无速度提升，强制记为 1 以避免异常统计
            if is_identity_output[i] and correct:
                speedup = 1.0
            
            row_samples_metrics.append({
                "correct": correct,
                "runtime": runtime,
                "compiled": compiled,
                "speedup": speedup
            })
            
        # 将样本级指标写回到当前 target 行，键与 eval_results 同级
        target_entry["row_samples_metrics"] = row_samples_metrics
        metrics_all_rows.append(row_samples_metrics)
        augmented_entries.append(target_entry)
        
    print(f"Analyzed {len(metrics_all_rows)} rows.")
    # if mismatched_count > 0:
    #     print(f"Skipped {mismatched_count} rows due to content mismatches.")

    # 3. Compute Aggregates for the single detected k
    
    pass_counts = 0 # This acts as Pass Best@k count
    
    # Trackers for Pass (Avg@k)
    # Sum of probabilities (fraction correct per problem)
    sum_pass_avg = 0.0
    
    # Trackers for Compilation
    compilation_best_counts = 0
    sum_compilation_avg = 0.0

    # Speedup lists for aggregation (Geometric Mean)
    # We collect the BEST speedup for each solved row (in top k)
    best_speedups = [] 
    # We collect the AVG speedup for each solved row (in top k)
    avg_speedups = []  
    
    # Counters for speedup thresholds
    # We track two counts for each threshold: based on BEST speedup and AVG speedup
    count_best_gt_1_1 = 0
    count_best_gt_1_2 = 0
    count_best_gt_1_5 = 0
    
    count_avg_gt_1_1 = 0
    count_avg_gt_1_2 = 0
    count_avg_gt_1_5 = 0

    # Collect rows exceeding speedup thresholds (based on BEST speedup per row)
    rows_best_gt = {
        "1.1": [],
        "1.2": [],
        "1.5": [],
    }

    for idx, samples in enumerate(metrics_all_rows):
        # Use all available samples since k is uniform
        k_samples = samples
        current_k = len(k_samples)
        if current_k == 0:
            continue
            
        # --- Compilation Stats ---
        # Best@k: Did any compile?
        if any(s["compiled"] for s in k_samples):
            compilation_best_counts += 1
        
        # Avg@k: Fraction compiled
        num_compiled = sum(1 for s in k_samples if s["compiled"])
        sum_compilation_avg += (num_compiled / current_k)

        # --- Correctness (Pass@k) Stats ---
        # Best@k: Is ANY sample correct?
        if any(s["correct"] for s in k_samples):
            pass_counts += 1
            
            # Analyze Speedup for this solved row
            # Filter for correct samples
            correct_samples_speedups = [s["speedup"] for s in k_samples if s["correct"]]
            
            # Filter out 0.0 speedups just in case (e.g. baseline failed or runtime invalid)
            valid_speedups = [s for s in correct_samples_speedups if s > 0]
            
            if valid_speedups:
                # 聚合时对 speedup 取 max(1, x)，augmented 文件中仍存原始值
                capped = [max(1.0, s) for s in valid_speedups]
                best_s = max(capped)
                avg_s = np.mean(capped)
                
                best_speedups.append(best_s)
                avg_speedups.append(avg_s)

                # Collect rows by BEST speedup threshold
                if best_s > 1.1:
                    rows_best_gt["1.1"].append(augmented_entries[idx])
                if best_s > 1.2:
                    rows_best_gt["1.2"].append(augmented_entries[idx])
                if best_s > 1.5:
                    rows_best_gt["1.5"].append(augmented_entries[idx])

                # Update Counters for Best Speedup
                if best_s > 1.1:
                    count_best_gt_1_1 += 1
                if best_s > 1.2:
                    count_best_gt_1_2 += 1
                if best_s > 1.5:
                    count_best_gt_1_5 += 1
                
                # Update Counters for Avg Speedup
                if avg_s > 1.1:
                    count_avg_gt_1_1 += 1
                if avg_s > 1.2:
                    count_avg_gt_1_2 += 1
                if avg_s > 1.5:
                    count_avg_gt_1_5 += 1
        
        # Avg@k: Fraction correct
        num_correct = sum(1 for s in k_samples if s["correct"])
        sum_pass_avg += (num_correct / current_k)
    
    total_rows = len(metrics_all_rows)
    
    # Calculate Rates
    if total_rows > 0:
        pass_best_rate = (pass_counts / total_rows * 100)
        pass_avg_rate = (sum_pass_avg / total_rows * 100)
        
        compilation_best_rate = (compilation_best_counts / total_rows * 100)
        compilation_avg_rate = (sum_compilation_avg / total_rows * 100)
    else:
        pass_best_rate = 0.0
        pass_avg_rate = 0.0
        compilation_best_rate = 0.0
        compilation_avg_rate = 0.0
    
    # Calculate percentages for thresholds
    def calc_pct(count, total):
        return (count / total * 100) if total > 0 else 0.0

    pct_best_gt_1_1 = calc_pct(count_best_gt_1_1, total_rows)
    pct_best_gt_1_2 = calc_pct(count_best_gt_1_2, total_rows)
    pct_best_gt_1_5 = calc_pct(count_best_gt_1_5, total_rows)

    pct_avg_gt_1_1 = calc_pct(count_avg_gt_1_1, total_rows)
    pct_avg_gt_1_2 = calc_pct(count_avg_gt_1_2, total_rows)
    pct_avg_gt_1_5 = calc_pct(count_avg_gt_1_5, total_rows)

    # Arithmetic mean: sum / count
    def arithmetic_mean(data):
        if not data:
            return 0.0
        return sum(data) / len(data)
    amean_best = arithmetic_mean(best_speedups)
    amean_avg = arithmetic_mean(avg_speedups)
    
    # 输出增广数据与不同 speedup 阈值的子集
    output_base = augmented_output or f"{target_file}.augmented.jsonl"
    write_jsonl(output_base, target_data)
    print(f"Augmented target written to: {output_base}")

    for th, rows in rows_best_gt.items():
        out_path = f"{output_base}.best_gt{th}.jsonl"
        write_jsonl(out_path, rows)
        print(f"Rows with BEST speedup > {th} written to: {out_path}")

    print("\nEvaluation Analysis Results:")
    print("=" * 40)
    print(f"Compilation Rate (%):")
    print(f"  Best@{k}: {compilation_best_rate:.2f}%")
    print(f"  Avg@{k}:  {compilation_avg_rate:.2f}%")
    print("-" * 40)
    print(f"Pass@{k} (%):")
    print(f"  Best@{k}: {pass_best_rate:.2f}%")
    print(f"  Avg@{k}:  {pass_avg_rate:.2f}%")
    print("-" * 40)
    print(f"AMean Speedup (算术平均 = 和/总个数):")
    print(f"  Best@{k}: {amean_best:.4f}")
    print(f"  Avg@{k}:  {amean_avg:.4f}")
    print("-" * 40)
    print(f"Speedup > 1.1 (%):")
    print(f"  Best@{k}: {pct_best_gt_1_1:.2f}%")
    print(f"  Avg@{k}:  {pct_avg_gt_1_1:.2f}%")
    print("-" * 40)
    print(f"Speedup > 1.2 (%):")
    print(f"  Best@{k}: {pct_best_gt_1_2:.2f}%")
    print(f"  Avg@{k}:  {pct_avg_gt_1_2:.2f}%")
    print("-" * 40)
    print(f"Speedup > 1.5 (%):")
    print(f"  Best@{k}: {pct_best_gt_1_5:.2f}%")
    print(f"  Avg@{k}:  {pct_avg_gt_1_5:.2f}%")
    print("=" * 40)
    print(f"Solved Count:   {pass_counts}")
    print(f"Total Analyzed: {total_rows}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze code generation results benchmark.")
    parser.add_argument("--baseline", type=str, default="results/test_input_verified_baseline_batch_eval/batch_results.jsonl", help="Path to baseline jsonl file")
    parser.add_argument("--target", type=str, required=True, help="Path to target jsonl file (with outputs)")
    parser.add_argument("--nrows2analyze", type=int, default=569, help="Number of rows to analyze")
    parser.add_argument("--augmented_output", type=str, default=None, help="Path to write target entries augmented with row_samples_metrics. Default: <target>.augmented.jsonl")
    # Removed --k argument as it's now auto-detected
    
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline):
        print(f"Error: Baseline file not found: {args.baseline}")
        sys.exit(1)
    if not os.path.exists(args.target):
        print(f"Error: Target file not found: {args.target}")
        sys.exit(1)
    
    augmented_output = args.augmented_output if args.augmented_output is not None else f"{args.target}.augmented.jsonl"
    analyze_results(args.baseline, args.target, args.nrows2analyze, augmented_output)
