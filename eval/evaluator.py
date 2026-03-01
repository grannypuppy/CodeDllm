import os
from typing import List, Dict, Any
from loguru import logger
from utils.py_sandbox import PySandBox
from utils import summarize_results
import json
import pandas as pd
import multiprocessing
from tqdm import tqdm
from eval.evalperf_runner import profile_script_with_evalperf

class Evaluator:
    def __init__(
        self,
        tmp_dir: str = "temp/eval",
        run_name: str = "dream_codenet_demo",
        num_runs: int = 10,
        use_evalperf: bool = False,
    ):
        """
        Initialize the Evaluator with a temporary directory for sandbox execution.
        
        Args:
            tmp_dir (str): Directory to store temporary files (e.g., compiled binaries).
            run_name (str): Name of the current evaluation run, used for organizing results.
            num_runs (int): Number of times to run each task for stable evaluation (default: 5).
            use_evalperf (bool): Use EvalPerf for instruction-count-based performance evaluation.
                Requires: pip install "evalplus[perf]" and Linux perf enabled.
        """
        self.run_name = run_name
        self.tmp_dir = tmp_dir
        self.num_runs = num_runs
        self.use_evalperf = use_evalperf
        if use_evalperf and not self.use_evalperf:
            logger.warning(
                "use_evalperf=True but EvalPerf not available. "
                "Install with: pip install 'evalplus[perf]' and enable perf (Linux)."
            )
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.sandbox = PySandBox(tmp_dir=tmp_dir)
        self.testcases_cache = {}

    def get_testcases(self, problem_id: str) -> List[Dict]:
        """
        Retrieve test cases for a specific problem ID, using a cache to avoid redundant I/O.
        
        Args:
            problem_id (str): The identifier of the problem.
            
        Returns:
            List[Dict]: A list of test cases, each containing 'input' and 'output'.
        """
        if problem_id not in self.testcases_cache:
            try:
                testcases = self.sandbox.get_testcases(problem_id)
                self.testcases_cache[problem_id] = testcases
                # logger.debug(f"Loaded {len(testcases)} testcases for problem {problem_id}")
            except Exception as e:
                logger.error(f"Failed to load testcases for problem {problem_id}: {e}")
                self.testcases_cache[problem_id] = []
        return self.testcases_cache[problem_id]

    def evaluate(self, code: str, problem_id: str, idx: str, row_index: int = None) -> Dict[str, Any]:
        """
        Execute the provided code against the problem's test cases.
        
        Args:
            code (str): The source code to execute (e.g., Python).
            problem_id (str): The problem identifier.
            idx (str): Unique identifier for this execution (used for file naming).
            
        Returns:
            Dict[str, Any]: A dictionary containing the execution results and overview.
                            Keys: 'completion_results_details', 'completion_results_overview', 'idx'.
        """
        testcases = self.get_testcases(problem_id)
        
        if not testcases:
            logger.warning(f"No testcases found for problem {problem_id}. Skipping execution.")
            return {
                "completion_results_details": [],
                "completion_results_overview": {"correctness": False, "error": "No testcases found"},
                "idx": idx
            }

        # Define path for the temporary source file directory based on problem_id
        # Add row_index to avoid collisions when the same problem_id appears in multiple rows
        row_suffix = f"row_{row_index}" if row_index is not None else "row_unknown"
        problem_dir = os.path.join(self.tmp_dir, self.run_name, str(problem_id), row_suffix)
        os.makedirs(problem_dir, exist_ok=True)
        
        py_file_path = os.path.join(problem_dir, f"{idx}.py")

        try:
            # Run code in sandbox for correctness
            completion_results = self.sandbox.run_python_code(
                code=code,
                testcases=testcases,
                py_file_path=py_file_path,
                time_out=30
            )
            
            # Summarize results
            completion_results_overview = summarize_results(completion_results)
            
            # If correct: (1) multi-run for stability, (2) EvalPerf instruction count when enabled
            evalperf_instruction_count = None
            if completion_results_overview["correctness"]:
                # EvalPerf: CPU instruction count per task (one-by-one)
                if self.use_evalperf and profile_script_with_evalperf:
                    evalperf_instruction_count = profile_script_with_evalperf(
                        code=code,
                        testcases=testcases,
                        timeout_per_test=60.0,
                        num_rounds=self.num_runs,
                    )
                    if evalperf_instruction_count is not None:
                        completion_results_overview["evalperf_instruction_count"] = evalperf_instruction_count

            return {
                "completion_results_details": completion_results,
                "completion_results_overview": completion_results_overview,
                "idx": idx
            }
            
        except Exception as e:
            logger.error(f"Execution failed for {idx} ({problem_id}): {e}")
            return {
                "completion_results_details": [],
                "completion_results_overview": {"correctness": False, "error": str(e)},
                "idx": idx
            }

    def _process_single_row(self, row_data: Dict, row_index: int) -> Dict:
        """
        Process a single row from the jsonl input.
        Helper function for multiprocessing.
        """
        problem_id = row_data.get("problem_id")
        outputs = row_data.get("output", [])
        
        # If output is not a list, wrap it (though spec says it's a list)
        if not isinstance(outputs, list):
            outputs = [outputs]

        results = []
        for i, code in enumerate(outputs):
            # Construct a unique idx for this execution: problem_id + batch_index
            # Using a combination to be unique within this run
            current_idx = f"{problem_id}_row{row_index}_code{i}"
            result = self.evaluate(code, problem_id, current_idx, row_index=row_index)
            results.append(result)
        
        # Return the original row data augmented with evaluation results
        # We can choose to return just the results or the full row + results
        return {
            **row_data,
            "eval_results": results
        }

    def batch_evaluate_jsonl(self, input_file: str, output_dir: str = "results", num_workers: int = 8):
        """
        Evaluate a batch of problems from a jsonl file in parallel.
        
        Args:
            input_file (str): Path to the input .jsonl file.
            output_dir (str): Base directory to save results.
            num_workers (int): Number of parallel worker processes.
        """
        # 1. Read input file
        try:
            data = pd.read_json(input_file, orient="records", lines=True).to_dict(orient="records")
            logger.info(f"Loaded {len(data)} records from {input_file}")
        except Exception as e:
            logger.error(f"Failed to read input file {input_file}: {e}")
            return

        # 2. Prepare output directory
        # Structure: results/run_name/
        run_output_dir = os.path.join(output_dir, self.run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        
        # 3. Parallel Processing
        # We use a pool of workers to process rows in parallel
        # Note: 'evaluate' method calls 'sandbox.run_python_code' which might spawn its own subprocesses.
        # Nested parallelism (multiprocessing inside multiprocessing) can be tricky.
        # Since sandbox uses subprocess or multiprocessing.Pool internally depending on implementation,
        # we need to be careful.
        # PySandBox.run_python_code uses multiprocessing.Pool if parallel=True.
        # To avoid "daemonic processes are not allowed to have children" error, 
        # we should either set parallel=False in sandbox for this batch mode 
        # OR use a better way to distribute tasks.
        # Given PySandBox implementation, it spawns a Pool for testcases. 
        # It's better to iterate over problems here and let each problem validation run sequentially 
        # OR let each problem run in parallel but turn off sandbox parallelism if it conflicts.
        # However, for maximum throughput, parallelizing at the problem level (here) is usually better 
        # because one problem might have few test cases but we have many problems.
        
        # Let's try using a Pool here. If PySandBox uses 'spawn' or 'forkserver', it might be okay, 
        # but standard 'fork' (default on Linux) with nested pools is problematic.
        # SAFE APPROACH: We will modify evaluate call implicitly or explicitly to not use nested pool if possible,
        # OR just rely on the fact that we are running on Linux and if PySandBox manages its pool correctly it might work,
        # but usually 'daemon' flag is the issue.
        # A common workaround is to use 'spawn' context or just process sequentially here if sandbox is already parallel.
        # BUT, usually test cases are few (10-50), while we might have 1000 problems. 
        # So parallelizing HERE (problem level) is more efficient.
        # I will assume PySandBox handles testcases. If we want to parallelize THIS loop, 
        # we must ensure PySandBox doesn't use a Pool or use a compatible one.
        # Checking PySandBox again: it uses `with multiprocessing.Pool...`. 
        # This will likely crash if called from a worker process.
        # 
        # SOLUTION: We'll keep it simple for now. If num_workers > 1, we assume we want to parallelize at THIS level.
        # To avoid crash, we might need to set sandbox.NUM_PROCESSES = 1 or similar if we run this in parallel workers.
        # Let's try to process with a Pool, but catch potential issues.
        
        # Actually, a better pattern for 'batch_evaluate' where we have many tasks is to flatten everything 
        # or just run sequential loop here if sandbox is already highly parallel. 
        # If sandbox spawns 16 processes per problem, and we spawn 8 workers here, we have 128 processes.
        # That might overload the machine.
        # 
        # Let's implement with a Pool but maybe suggest setting sandbox parallelism to False/Low if calling this.
        # Or, we can just run a sequential loop with tqdm if we trust sandbox's internal parallelism.
        # So I will implement multiprocessing at this level.
        
        processed_results = []
        
        # EvalPerf's profile() spawns child processes; Pool workers are daemonic and cannot have children.
        # When use_evalperf, run in main process (no Pool).
        if self.use_evalperf:
            logger.info("EvalPerf enabled; running sequentially in main process (Pool workers cannot spawn children).")
            for idx, row in enumerate(tqdm(data, desc="Evaluating")):
                result = _worker_func((idx, row, self.tmp_dir, self.run_name, self.num_runs, self.use_evalperf))
                processed_results.append(result)
        else:
            logger.info(f"Starting batch evaluation with {num_workers} workers...")
            tasks = [
                (idx, row, self.tmp_dir, self.run_name, self.num_runs, self.use_evalperf)
                for idx, row in enumerate(data)
            ]
            with multiprocessing.Pool(processes=num_workers) as pool:
                results_iter = pool.imap(_worker_func, tasks)
                processed_results = list(tqdm(results_iter, total=len(data), desc="Evaluating"))

        # 4. Save results
        output_file = os.path.join(run_output_dir, "batch_results.jsonl")
        logger.info(f"Saving results to {output_file}")
        
        df = pd.DataFrame(processed_results)
        df.to_json(output_file, orient="records", lines=True)
        logger.success(f"Batch evaluation completed. Results saved to {output_file}")

# Standalone helper function for multiprocessing
def _worker_func(args):
    row_index, row_data, tmp_dir, run_name, num_runs, use_evalperf = args

    evaluator = Evaluator(
        tmp_dir=tmp_dir, run_name=run_name, num_runs=num_runs, use_evalperf=use_evalperf
    )
    
    original_run_python_code = evaluator.sandbox.run_python_code
    
    def serialized_run_python_code(*args, **kwargs):
        kwargs['parallel'] = False # Force serial
        return original_run_python_code(*args, **kwargs)
    
    evaluator.sandbox.run_python_code = serialized_run_python_code
    
    return evaluator._process_single_row(row_data, row_index)
