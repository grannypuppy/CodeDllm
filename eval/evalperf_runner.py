"""
CPU instruction counting via `perf stat` subprocess.
No evalplus/C++ wrapper needed — requires Linux with perf available
and perf_event_paranoid <= 1:
  sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
"""
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from loguru import logger


def _run_perf_stat(script_path: str, stdin_data: str, timeout: float) -> Optional[int]:
    """Run script under perf stat and return user-space instruction count."""
    try:
        result = subprocess.run(
            ["perf", "stat", "-e", "instructions:u", sys.executable, script_path],
            input=stdin_data.encode(),
            capture_output=True,
            timeout=timeout,
        )
        m = re.search(r"([\d,]+)\s+instructions", result.stderr.decode())
        if m:
            return int(m.group(1).replace(",", ""))
        logger.warning(f"perf stat: no instruction count in output for {script_path}")
    except subprocess.TimeoutExpired:
        logger.warning(f"perf stat timed out ({timeout}s) for {script_path}")
    except FileNotFoundError:
        logger.error("perf not found — install linux-perf and ensure it's on PATH")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.warning(f"perf stat error: {e}")
    return None


def profile_script_with_evalperf(
    code: str,
    testcases: List[Dict[str, str]],
    timeout_per_test: float = 45.0,
    num_rounds: int = 5,
) -> Tuple[Optional[List[int]], Optional[int]]:
    """
    Profile a stdin-based script using perf stat CPU instruction counter.

    Args:
        code: Python script that reads stdin and writes to stdout.
        testcases: List of {"input": str, "output": str}.
        timeout_per_test: Per-testcase timeout in seconds.
        num_rounds: Profiling rounds (averaged for stability).

    Returns:
        (profiles, avg_count): profiles is a list of per-round total instruction
        counts (summed across testcases); avg_count is their mean.
        Returns (None, None) on any failure.
    """
    if not testcases:
        return None, None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        profiles = []
        for _ in range(num_rounds):
            round_total = 0
            for tc in testcases:
                count = _run_perf_stat(script_path, tc["input"], timeout_per_test)
                if count is None:
                    print(f"Error profiling script with evalperf")
                    return None, None
                round_total += count
            profiles.append(round_total)

        avg = int(sum(profiles) / len(profiles))
        return profiles, avg
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        os.unlink(script_path)
