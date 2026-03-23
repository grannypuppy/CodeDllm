from typing import List, Dict
import tempfile
import subprocess
import time
import os
import re
from utils.data_structure import RunResult
import resource
import multiprocessing
import sys

class PySandBox:
    def __init__(
            self,
            tmp_dir: str = 'temp/PySemRep',
            MAX_VIRTUAL_MEMORY: int = 10 * 1024 * 1024 * 50, # 500MB by default
            MAX_INSTRUCTION_COUNT: int = 1e12, # Not directly used for Python, but kept for consistency in interface
            NUM_PROCESSES: int = 16,
            evalperf: bool = False,
            num_runs: int = 5,
            num_warmup: int = 1
    ):
        self.tmp_dir = tmp_dir if tmp_dir else tempfile.gettempdir()
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.MAX_VIRTUAL_MEMORY = MAX_VIRTUAL_MEMORY
        self.MAX_INSTRUCTION_COUNT = MAX_INSTRUCTION_COUNT
        self.NUM_PROCESSES = NUM_PROCESSES
        self.evalperf = evalperf
        self.num_runs = num_runs
        self.num_warmup = num_warmup

    def limit_virtual_memory(self):
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (self.MAX_VIRTUAL_MEMORY, self.MAX_VIRTUAL_MEMORY * 10))

    @staticmethod
    def _parse_perf_instructions(perf_stderr: str) -> int:
        """Parse instruction count from perf stat stderr output."""
        match = re.search(r'([\d,]+)\s+instructions', perf_stderr)
        if match:
            return int(match.group(1).replace(',', ''))
        return -1

    def _run_once(self, command: list, testcase_input: str, time_out: int):
        """Run command once, return (execution_time_ms, stdout, stderr_decoded, returncode) or raise TimeoutExpired."""
        start_time = time.time()
        proc = subprocess.run(
            command,
            input=testcase_input.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=time_out,
            preexec_fn=self.limit_virtual_memory,
        )
        execution_time = (time.time() - start_time) * 1000
        return execution_time, proc.stdout, proc.stderr.decode(), proc.returncode

    def execute_single_python_code_with_single_testcase(
            self,
            py_file_path: str,
            testcase: Dict,
            time_out: int = 10,
            testcase_input: str = None,
            testcase_output: str = None
    ) -> RunResult:
        if self.evalperf:
            command = ['perf', 'stat', '-e', 'instructions', '--', sys.executable, py_file_path]
        else:
            command = [sys.executable, py_file_path]

        # Warm-up runs (results discarded)
        for _ in range(self.num_warmup):
            try:
                self._run_once(command, testcase_input, time_out)
            except subprocess.TimeoutExpired:
                pass

        # Measured runs
        execution_times = []
        instruction_counts = []
        stdout = b""
        for _ in range(self.num_runs):
            try:
                execution_time, stdout, stderr_decoded, returncode = self._run_once(command, testcase_input, time_out)
            except subprocess.TimeoutExpired:
                print(f"Timeout in {py_file_path} with testcase {testcase_input[:20]}...")
                return RunResult(
                    compilation_error=False,
                    execution_error=False,
                    correct=False,
                    execution_time=time_out * 1000,
                    time_limit_error=True,
                    output=""
                )

            if returncode != 0:
                prog_stderr = re.sub(r'\n\s*Performance counter stats.*', '', stderr_decoded, flags=re.DOTALL).strip() if self.evalperf else stderr_decoded
                return RunResult(
                    compilation_error=False,
                    execution_error=f"Execution failed: {prog_stderr}",
                )

            execution_times.append(execution_time)
            if self.evalperf:
                instruction_counts.append(self._parse_perf_instructions(stderr_decoded))

        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_instruction_count = int(sum(instruction_counts) / len(instruction_counts)) if instruction_counts else None

        return RunResult(
            compilation_error=False,
            execution_error=False,
            correct=(stdout.decode().strip() == testcase_output.strip()),
            execution_time=avg_execution_time,
            instruction_count=avg_instruction_count,
            execution_time_list=execution_times,
            instruction_count_list=instruction_counts if instruction_counts else None,
            output=stdout.decode()
        )

    def execute_python_code(self, py_file_path: str, testcases: List[Dict], time_out: int = 10, parallel: bool = True) -> List[RunResult]:
        results = []
        if parallel:
            with multiprocessing.Pool(processes=self.NUM_PROCESSES) as pool:
                tasks = [(py_file_path, testcase, time_out, testcase['input'], testcase['output']) for testcase in testcases]
                results = pool.starmap(self.execute_single_python_code_with_single_testcase, tasks)
        else:
            for testcase in testcases:
                result = self.execute_single_python_code_with_single_testcase(
                    py_file_path,
                    testcase,
                    time_out=time_out,
                    testcase_input=testcase['input'],
                    testcase_output=testcase['output']
                )
                results.append(result)

        return results

    def run_python_code(self, code: str, testcases: List[Dict], py_file_path = None, time_out: int = 10, parallel: bool = True) -> List[RunResult]:
        if py_file_path is None:
            pid = os.getpid()
            py_file_path = os.path.join(self.tmp_dir, f"{pid}.py")
        
        with open(py_file_path, "w") as f:
            f.write(code)
        
        # Python is interpreted, so no compilation step. 
        # However, we could optionally check for syntax errors before running.
        try:
            py_compile_command = [sys.executable, "-m", "py_compile", py_file_path]
            subprocess.check_output(py_compile_command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
             return [RunResult(compilation_error=e.output.decode())]

        try:
            return self.execute_python_code(py_file_path, testcases, time_out=time_out, parallel=parallel)
        except Exception as execution_stderr:
            return [RunResult(compilation_error=False, execution_error=str(execution_stderr))]

    def get_testcases(self, problem_id: str) -> List[Dict]:
        # Reusing the logic from SandBox since testcase format is likely shared
        testcase_dir = os.path.join("data/codenet/public_test_cases", problem_id)
        testcases = []
        if os.path.exists(testcase_dir):
            for input_file in os.listdir(testcase_dir):
                # if match input.*.txt
                if input_file.startswith("input.") and input_file.endswith(".txt"):
                    output_file = input_file.replace("input.", "output.")
                    with open(os.path.join(testcase_dir, input_file), 'r') as f:
                        input_data = f.read()
                    with open(os.path.join(testcase_dir, output_file), 'r') as f:
                        output_data = f.read()
                    testcases.append({
                        "input": input_data,
                        "output": output_data
                    })
        return testcases

if __name__ == "__main__":
    # Simple test
    code = "import sys\nprint(sys.stdin.read().strip())"
    sandbox = PySandBox()
    testcases = [{"input": "hello", "output": "hello"}, {"input": "world", "output": "world"}]
    results = sandbox.run_python_code(code, testcases, parallel=False)
    for res in results:
        print(res)

