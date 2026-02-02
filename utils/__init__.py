from .data_structure import RunResult

def summarize_results(run_results: list[RunResult]) -> dict:
    compilation_error = any([result.compilation_error for result in run_results])
    execution_error = any([result.execution_error for result in run_results])
    wrong_output = any([not result.correct for result in run_results])
    correctness = all([result.correct for result in run_results])
    accuracy = sum([result.correct for result in run_results if result.correct is not None]) / len(run_results)
    avg_run_time = sum([result.execution_time for result in run_results]) / len(run_results)
    return {
        "compilation_error": compilation_error,
        "execution_error": execution_error,
        "wrong_output": wrong_output,
        "correctness": correctness,
        "accuracy": accuracy,
        "avg_run_time": avg_run_time
    }

def sanitize_for_json(df):
    def safe_str(x):
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")  # replaces invalid bytes with �
        elif isinstance(x, str):
            try:
                x.encode("utf-8")  # test encoding
                return x
            except UnicodeEncodeError:
                return x.encode("utf-8", errors="replace").decode("utf-8")
        else:
            return x

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(safe_str)
    return df

def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
