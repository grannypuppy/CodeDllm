"""
EvalPerf integration for profiling stdin-based scripts (e.g., CodeNet format).
Wraps script execution as a callable and uses evalplus.perf.profile for CPU instruction counting.
Requires: pip install "evalplus[perf]" and Linux with perf enabled (sudo echo 0 > /proc/sys/kernel/perf_event_paranoid).
"""
import base64
from typing import List, Dict, Any, Optional

from evalplus.perf.profile import profile, are_profiles_broken


def _make_profilable_script_code(user_code: str) -> str:
    """Wrap stdin-based script as a callable for EvalPerf's profile()."""
    # Encode to avoid escaping issues with quotes/newlines in user code
    code_b64 = base64.b64encode(user_code.encode("utf-8")).decode("ascii")
    return f'''
import base64
import io
import sys

__USER_CODE__ = base64.b64decode("{code_b64}").decode("utf-8")

def __evalperf_run__(input_str):
    _stdin_bak = sys.stdin
    _stdout_bak = sys.stdout
    # TextIOWrapper(BytesIO(...)) provides .buffer for code that uses sys.stdin.buffer
    sys.stdin = io.TextIOWrapper(io.BytesIO(input_str.encode("utf-8")), encoding="utf-8")
    _stdin_stream = sys.stdin
    sys.stdout = io.StringIO()
    # reliability_guard sets builtins.open=None; restore open(0) for stdin (common in CodeNet)
    import builtins
    _b = dict(builtins.__dict__)
    def _open(file, mode="r", *a, **kw):
        if file == 0:
            return _stdin_stream
        raise OSError("Only stdin (open(0)) allowed during profiling")
    _b["open"] = _open
    _globals = {{"__name__": "__main__", "__builtins__": _b}}
    try:
        exec(__USER_CODE__, _globals)
    finally:
        sys.stdin = _stdin_bak
        sys.stdout = _stdout_bak
'''


def profile_script_with_evalperf(
    code: str,
    testcases: List[Dict[str, str]],
    timeout_per_test: float = 45.0,
    num_rounds: int = 5,
) -> Optional[int]:
    """
    Profile a stdin-based script using EvalPerf's CPU instruction counter.
    
    Args:
        code: Python script that reads stdin and prints to stdout.
        testcases: List of {"input": str, "output": str}.
        timeout_per_test: Timeout in seconds per test input.
        num_rounds: Number of profiling rounds (use >1 for stability).
    
    Returns:
        Total instruction count across all testcases, or None if profiling failed.
    """
    if not testcases:
        return None

    try:
        test_inputs = [(tc["input"],) for tc in testcases]
        func_code = _make_profilable_script_code(code)

        profiles = profile(
            func_code,
            "__evalperf_run__",
            test_inputs,
            timeout_second_per_test=timeout_per_test,
            profile_rounds=num_rounds,
        )

        if are_profiles_broken(profiles):
            return None

        # profiles is List[int|float] - one per round
        return int(sum(profiles) / len(profiles)) if profiles else None
    except Exception:
        return None