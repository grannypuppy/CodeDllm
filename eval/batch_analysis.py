"""
batch_analysis.py

Analyze multiple eval runs against a baseline and compile results into a single CSV.
Speedup is calculated from instruction count only.

Usage:
    # Analyze specific target files:
    uv run python eval/batch_analysis.py \
        --baseline results/baseline/baseline/batch_results.jsonl \
        --targets results/test_verified/run1/batch_results.jsonl results/test_verified/run2/batch_results.jsonl \
        --output results/analysis.csv

    # Auto-discover all batch_results.jsonl under a directory:
    uv run python eval/batch_analysis.py \
        --baseline results/baseline/baseline/batch_results.jsonl \
        --runs_dir results/test_verified \
        --output results/analysis.csv
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Per-sample metric extraction
# ---------------------------------------------------------------------------

def get_metrics(result_entry):
    """
    Extract per-sample metrics from a single eval_result entry.
    Returns: (correctness, compiled, instruction_count)
    instruction_count may be None if not measured.
    """
    overview = result_entry.get("completion_results_overview", {})
    if not overview:
        return False, False, None

    correctness = overview.get("correctness", False)
    if isinstance(correctness, str):
        correctness = correctness.lower() == "true"

    compilation_error = overview.get("compilation_error", False)
    if isinstance(compilation_error, str):
        compilation_error = compilation_error.lower() == "true"
    compiled = not compilation_error

    instruction_count = overview.get("instruction_count")

    return correctness, compiled, instruction_count


# ---------------------------------------------------------------------------
# Core analysis for a single run
# ---------------------------------------------------------------------------

def analyze_run(baseline_data, target_data, nrows2analyze):
    """
    Compare target_data against baseline_data.

    Speedup is computed from instruction_count only.

    Returns a dict of aggregate metrics, or None if k cannot be determined.
    """
    if nrows2analyze is not None:
        baseline_data = baseline_data[:nrows2analyze]
        target_data = target_data[:nrows2analyze]

    # Detect k from first valid target entry
    k = 0
    for entry in target_data:
        results = entry.get("eval_results", [])
        if results:
            k = len(results)
            break
    if k == 0:
        return None

    # -----------------------------------------------------------------------
    # Row-level processing
    # -----------------------------------------------------------------------
    metrics_all_rows = []

    for i, (base_entry, target_entry) in enumerate(zip(baseline_data, target_data)):
        # Validate alignment
        if base_entry.get("problem_id") != target_entry.get("problem_id"):
            continue
        base_input = str(base_entry.get("input", ""))
        target_input = str(target_entry.get("input", ""))
        if base_input[:10] != target_input[:10]:
            continue

        # Detect identity outputs (model just copied the input)
        raw_outputs = target_entry.get("output", [])
        if not isinstance(raw_outputs, list):
            raw_outputs = [raw_outputs]
        is_identity = [str(o) == target_input for o in raw_outputs]

        # Baseline reference (first eval_result)
        base_eval_results = base_entry.get("eval_results", [])
        base_correct = False
        base_ic = None
        if base_eval_results:
            base_correct, _, base_ic = get_metrics(base_eval_results[0])

        # Target samples
        row_samples = []
        for j, res in enumerate(target_entry.get("eval_results", [])):
            correct, compiled, ic = get_metrics(res)

            speedup = 0.0
            if correct and base_correct:
                if ic is not None and base_ic is not None and base_ic > 0 and ic > 0:
                    speedup = base_ic / ic

            # Identity output → no speedup
            if j < len(is_identity) and is_identity[j] and correct:
                speedup = 1.0

            row_samples.append({
                "correct": correct,
                "compiled": compiled,
                "speedup": speedup,
            })

        metrics_all_rows.append(row_samples)

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    pass_counts = 0
    sum_pass_avg = 0.0
    compilation_best_counts = 0
    sum_compilation_avg = 0.0

    best_speedups = []
    avg_speedups = []

    count_best_gt = {1.1: 0, 1.2: 0, 1.5: 0}
    count_avg_gt  = {1.1: 0, 1.2: 0, 1.5: 0}

    for samples in metrics_all_rows:
        cur_k = len(samples)
        if cur_k == 0:
            continue

        # Compilation
        if any(s["compiled"] for s in samples):
            compilation_best_counts += 1
        sum_compilation_avg += sum(s["compiled"] for s in samples) / cur_k

        # Correctness
        num_correct = sum(s["correct"] for s in samples)
        sum_pass_avg += num_correct / cur_k

        if any(s["correct"] for s in samples):
            pass_counts += 1

            valid_speedups = [
                s["speedup"] for s in samples if s["correct"] and s["speedup"] > 0
            ]
            if valid_speedups:
                capped = [max(1.0, s) for s in valid_speedups]
                best_s = max(capped)
                avg_s  = float(np.mean(capped))

                best_speedups.append(best_s)
                avg_speedups.append(avg_s)

                for th in (1.1, 1.2, 1.5):
                    if best_s > th:
                        count_best_gt[th] += 1
                    if avg_s > th:
                        count_avg_gt[th] += 1

    total = len(metrics_all_rows)

    def rate(n):
        return n / total * 100 if total > 0 else 0.0

    def amean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "k": k,
        "total_rows": total,
        "solved_count": pass_counts,
        "pass_best_pct": rate(pass_counts),
        "pass_avg_pct": sum_pass_avg / total * 100 if total > 0 else 0.0,
        "compile_best_pct": rate(compilation_best_counts),
        "compile_avg_pct": sum_compilation_avg / total * 100 if total > 0 else 0.0,
        "speedup_amean_best": amean(best_speedups),
        "speedup_amean_avg":  amean(avg_speedups),
        "speedup_best_gt1.1_pct": rate(count_best_gt[1.1]),
        "speedup_best_gt1.2_pct": rate(count_best_gt[1.2]),
        "speedup_best_gt1.5_pct": rate(count_best_gt[1.5]),
        "speedup_avg_gt1.1_pct":  rate(count_avg_gt[1.1]),
        "speedup_avg_gt1.2_pct":  rate(count_avg_gt[1.2]),
        "speedup_avg_gt1.5_pct":  rate(count_avg_gt[1.5]),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze multiple eval runs and write results to a single CSV."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="results/baseline/baseline/batch_results.jsonl",
        help="Path to the baseline batch_results.jsonl",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=[],
        help="One or more target batch_results.jsonl files to analyze",
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default=None,
        help="Directory to auto-discover batch_results.jsonl files (one level deep)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit analysis to the first N rows (default: all rows)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/analysis.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Validate baseline
    if not os.path.exists(args.baseline):
        print(f"Error: baseline file not found: {args.baseline}")
        sys.exit(1)

    # Collect target files
    target_files = list(args.targets)
    if args.runs_dir:
        discovered = sorted(glob.glob(os.path.join(args.runs_dir, "*/batch_results.jsonl")))
        print(f"Discovered {len(discovered)} runs under {args.runs_dir}")
        target_files.extend(discovered)

    if not target_files:
        print("Error: no target files specified. Use --targets or --runs_dir.")
        sys.exit(1)

    # Load baseline once
    print(f"Loading baseline: {args.baseline}")
    baseline_data = load_jsonl(args.baseline)

    # CSV columns
    columns = [
        "run_name",
        "k",
        "total_rows",
        "solved_count",
        "pass_best_pct",
        "pass_avg_pct",
        "compile_best_pct",
        "compile_avg_pct",
        "speedup_amean_best",
        "speedup_amean_avg",
        "speedup_best_gt1.1_pct",
        "speedup_best_gt1.2_pct",
        "speedup_best_gt1.5_pct",
        "speedup_avg_gt1.1_pct",
        "speedup_avg_gt1.2_pct",
        "speedup_avg_gt1.5_pct",
    ]

    rows = []
    for target_file in target_files:
        if not os.path.exists(target_file):
            print(f"  [skip] not found: {target_file}")
            continue

        # Derive a human-readable run name from the file path
        parts = os.path.normpath(target_file).split(os.sep)
        # Use the parent directory name as run_name
        run_name = parts[-2] if len(parts) >= 2 else target_file

        print(f"  Analyzing: {run_name} ...")
        target_data = load_jsonl(target_file)
        metrics = analyze_run(baseline_data, target_data, args.nrows)

        if metrics is None:
            print(f"    [skip] could not determine k for {run_name}")
            continue

        row = {"run_name": run_name, **metrics}
        rows.append(row)

        print(
            f"    k={metrics['k']}  solved={metrics['solved_count']}/{metrics['total_rows']}"
            f"  pass_best={metrics['pass_best_pct']:.1f}%"
            f"  speedup_best={metrics['speedup_amean_best']:.4f}"
            f"  speedup_avg={metrics['speedup_amean_avg']:.4f}"
        )

    if not rows:
        print("No results to write.")
        sys.exit(0)

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
