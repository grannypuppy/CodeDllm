from typing import List, Dict
import tempfile
import subprocess
import time
import os
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
            NUM_PROCESSES: int = 16
    ):
        self.tmp_dir = tmp_dir if tmp_dir else tempfile.gettempdir()
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.MAX_VIRTUAL_MEMORY = MAX_VIRTUAL_MEMORY
        self.MAX_INSTRUCTION_COUNT = MAX_INSTRUCTION_COUNT
        self.NUM_PROCESSES = NUM_PROCESSES

    def limit_virtual_memory(self):
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (self.MAX_VIRTUAL_MEMORY, self.MAX_VIRTUAL_MEMORY * 10))

    def execute_single_python_code_with_single_testcase(
            self,
            py_file_path: str,
            testcase: Dict,
            time_out: int = 10,
            testcase_input: str = None,
            testcase_output: str = None
    ) -> RunResult:
        # Python execution command
        command = [sys.executable, py_file_path]

        try:
            start_time = time.time()
            execution_process = subprocess.run(
                command,
                input=testcase_input.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=time_out,
                preexec_fn=self.limit_virtual_memory,
            )
            end_time = time.time()
            stdout, stderr = execution_process.stdout, execution_process.stderr
            execution_time = (end_time - start_time) * 1000 # Convert to ms to be somewhat comparable to instructions/perf metrics
        except subprocess.TimeoutExpired:
            # clean up is handled by subprocess.run killing the process on timeout
            print(f"Timeout in {py_file_path} with testcase {testcase_input[:20]}...")
            return RunResult(
                compilation_error=False,
                execution_error=False,
                correct=False,
                execution_time=time_out * 1000, # Max time
                time_limit_error=True,
                output=""
            )

        # Check if the execution was successful (return code 0)
        if execution_process.returncode != 0:
            return RunResult(
                compilation_error=False,
                execution_error=f"Execution failed: {stderr.decode()}",
            )

        # For Python, we don't have 'perf' instruction count easily available per process without overhead.
        # We use wall clock time or a placeholder.
        # We also check correctness.
        
        return RunResult(
            compilation_error=False,
            execution_error=False,
            correct=(stdout.decode().strip() == testcase_output.strip()),
            execution_time=execution_time, 
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

