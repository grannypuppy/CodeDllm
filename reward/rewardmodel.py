import json
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import OmegaConf

from eval.evaluator import Evaluator


def _get_wandb_run_id(project_dir: Path) -> str:
    """Return a stable wandb run_id for this experiment, persisted to disk."""
    run_id_file = project_dir / "wandb_run_id.txt"
    if run_id_file.exists():
        return run_id_file.read_text().strip()
    try:
        import wandb as _w
        run_id = _w.util.generate_id()
    except Exception:
        run_id = hashlib.sha1(str(project_dir).encode()).hexdigest()[:16]
    run_id_file.parent.mkdir(parents=True, exist_ok=True)
    run_id_file.write_text(run_id)
    return run_id


def _compute_stats_from_data(data: list, function: str) -> dict:
    """Compute reward-related statistics from scored rollout records."""
    if not data:
        return {}
    is_seq = isinstance(data[0].get("generated_output", None), str)

    if is_seq:
        rewards      = [float(r.get("reward", 0.0)) for r in data]
        correct      = [bool(r.get("correctness", False)) for r in data]
        compiled     = [bool(r.get("compilation", False)) for r in data]
        speedups     = [float(r.get("speedup", 0.0)) for r in data]
    else:
        rewards  = [float(v) for r in data for v in r.get("rewards", [])]
        correct  = [bool(v) for r in data for v in r.get("correctness", [])]
        compiled = [bool(v) for r in data for v in r.get("compilation", [])]
        speedups = [float(v) for r in data for v in r.get("speedup", [])]

    if not rewards:
        return {}
    n = len(rewards)
    mean_r = sum(rewards) / n
    var_r  = sum((x - mean_r) ** 2 for x in rewards) / max(n - 1, 1)
    return {
        "mean":             mean_r,
        "std":              var_r ** 0.5,
        "max":              max(rewards),
        "min":              min(rewards),
        "correctness_rate": sum(correct)  / n,
        "compilation_rate": sum(compiled) / n,
        "mean_speedup":     sum(speedups) / n,
    }


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def grouped_grpo_advantages(values: List[float]) -> Tuple[List[float], float, float]:
    """Return GRPO-style within-group normalized advantages.

    For a single response group (same task / same prompt), compute:
        adv_i = (r_i - mean(r)) / std(r)
    and fall back to zeros when std == 0.
    """
    if not values:
        return [], 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = var ** 0.5
    if std == 0:
        return [0.0 for _ in values], float(mean), 0.0
    adv = [(x - mean) / std for x in values]
    return adv, float(mean), float(std)


class RewardModel:
    """Mock reward model interface.

    You can replace `score_response` with real reward model inference later.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._baseline_cache = {}
        self._candidate_cache = {}
        self._row_counter = 0
        self._baseline_exact = {}
        self._baseline_prefix = {}

        run_name = cfg.experiment.get("run_name", None)
        if run_name is None:
            run_name = cfg.experiment.get("wandb_run_name", "reward_eval")
        run_name = str(run_name)

        reward_tmp_dir = str(cfg.get("reward", {}).get("tmp_dir", "temp/eval_reward"))
        reward_num_runs = int(cfg.get("reward", {}).get("num_runs", 5))
        reward_use_evalperf = bool(cfg.get("reward", {}).get("use_evalperf", True))

        self.evaluator = Evaluator(
            tmp_dir=reward_tmp_dir,
            run_name=f"{run_name}_reward",
            use_evalperf=reward_use_evalperf,
            num_runs=reward_num_runs,
        )

        root = Path(__file__).resolve().parents[1]
        default_baseline = root / "results" / "baseline" / "baseline" / "batch_results.jsonl"
        self.baseline_results_path = str(cfg.get("reward", {}).get("baseline_results_path", str(default_baseline)))
        self._load_baseline_results(self.baseline_results_path)

    @staticmethod
    def _to_bool(v) -> bool:
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    @staticmethod
    def _normalize_code(text: str) -> str:
        return (text or "").strip()

    @staticmethod
    def _baseline_key(problem_id, input_text: str) -> Tuple[str, str]:
        return str(problem_id), str(input_text)

    @staticmethod
    def _baseline_prefix_key(problem_id, input_text: str) -> Tuple[str, str]:
        return str(problem_id), str(input_text)[:10]

    @staticmethod
    def _extract_metrics(eval_result: dict):
        overview = eval_result.get("completion_results_overview", {}) or {}
        correctness = RewardModel._to_bool(overview.get("correctness", False))
        compilation_error = RewardModel._to_bool(overview.get("compilation_error", False))
        compiled = not compilation_error
        instruction_count = overview.get("instruction_count", None)
        instruction_count = float(instruction_count) if instruction_count is not None else None
        return correctness, compiled, instruction_count

    def _load_baseline_results(self, baseline_path: str):
        p = Path(baseline_path)
        if not p.exists():
            print(f"[RewardModel] Baseline results file not found, fallback to live baseline eval: {p}")
            return

        n = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = row.get("problem_id", "")
                inp = row.get("input", "")
                eval_results = row.get("eval_results", [])
                if not eval_results:
                    continue
                first_eval = eval_results[0]
                corr, comp, instr = self._extract_metrics(first_eval)
                val = {
                    "correctness": bool(corr),
                    "compilation": bool(comp),
                    "instruction_count": instr,
                }
                self._baseline_exact[self._baseline_key(pid, inp)] = val
                self._baseline_prefix[self._baseline_prefix_key(pid, inp)] = val
                n += 1
        print(f"[RewardModel] Loaded {n} baseline rows from: {p}")

    def _get_baseline_metrics(self, item: dict, baseline_code: str) -> Optional[dict]:
        pid = item.get("problem_id", "")
        k_exact = self._baseline_key(pid, baseline_code)
        if k_exact in self._baseline_exact:
            return self._baseline_exact[k_exact]

        k_prefix = self._baseline_prefix_key(pid, baseline_code)
        if k_prefix in self._baseline_prefix:
            return self._baseline_prefix[k_prefix]
        return None

    def _evaluate_code(self, code: str, problem_id: str, cache: dict, key_prefix: str):
        code = self._normalize_code(code)
        key = (problem_id, hashlib.sha1(code.encode("utf-8")).hexdigest())
        if key in cache:
            return cache[key]

        idx = f"{key_prefix}_{problem_id}_{self._row_counter}"
        self._row_counter += 1
        result = self.evaluator.evaluate(code=code, problem_id=str(problem_id), idx=idx, row_index=self._row_counter)
        cache[key] = result
        return result

    def score_response_components(self, item: dict, response_text: str, extracted_output: str) -> dict:
        problem_id = item.get("problem_id", None)
        if problem_id is None:
            return {
                "reward": 0.0,
                "correctness": False,
                "compilation": False,
                "speedup": 0.0,
                "baseline_instruction_count": None,
                "candidate_instruction_count": None,
            }

        baseline_code = item.get("input", "") or ""
        candidate_code = self._normalize_code(extracted_output) or self._normalize_code(response_text)

        if not str(baseline_code).strip() or not candidate_code:
            return {
                "reward": 0.0,
                "correctness": False,
                "compilation": False,
                "speedup": 0.0,
                "baseline_instruction_count": None,
                "candidate_instruction_count": None,
            }

        baseline_metrics = self._get_baseline_metrics(item, baseline_code)
        if baseline_metrics is not None:
            base_correct = bool(baseline_metrics["correctness"])
            base_instr = baseline_metrics["instruction_count"]
        else:
            assert False, f"Baseline metrics not found for problem {problem_id}"
            base_eval = self._evaluate_code(baseline_code, problem_id, self._baseline_cache, "base")
            base_correct, _, base_instr = self._extract_metrics(base_eval)

        cand_eval = self._evaluate_code(candidate_code, problem_id, self._candidate_cache, "cand")
        cand_correct, cand_compiled, cand_instr = self._extract_metrics(cand_eval)

        speedup = 0.0
        if cand_correct and base_correct and base_instr is not None and cand_instr is not None and base_instr > 0 and cand_instr > 0:
            speedup = float(base_instr) / float(cand_instr)

        # If candidate is exactly baseline and passes, treat speedup as 1.0 (same program).
        if cand_correct and self._normalize_code(candidate_code) == self._normalize_code(baseline_code):
            speedup = 1.0

        reward = 0.1 * float(cand_compiled) + 0.2 * float(cand_correct) + 1.0 * float(speedup)
        return {
            "reward": float(reward),
            "correctness": bool(cand_correct),
            "compilation": bool(cand_compiled),
            "speedup": float(speedup),
            "baseline_instruction_count": base_instr,
            "candidate_instruction_count": cand_instr,
        }

    def score_response(self, item: dict, response_text: str, extracted_output: str) -> float:
        return float(self.score_response_components(item, response_text, extracted_output)["reward"])


class RewardBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = RewardModel(cfg)

    def _run_name(self) -> str:
        run_name = self.cfg.experiment.get("run_name", None)
        if run_name is None:
            run_name = self.cfg.experiment.get("wandb_run_name", None)
        if run_name is None:
            raise ValueError("Missing run name: set experiment.run_name or experiment.wandb_run_name in config.")
        return str(run_name)

    def _resolve_model_path(self) -> str:
        return str(self.cfg.model.pretrained_model)

    def _sanitize_name_part(self, s: str) -> str:
        return str(s).replace("/", ".").replace("\\", ".")

    def _outputs_name(self, model_path: str, config_path: str) -> str:
        return f"{self._sanitize_name_part(model_path)}-{self._sanitize_name_part(config_path)}"

    def _outputs_filename(self, model_path: str, config_path: str, current_round: int) -> str:
        outputs_name = self._outputs_name(model_path, config_path)
        return f"round_{current_round}-outputs-{outputs_name}.jsonl"

    def _outputs_filename_legacy_json(self, model_path: str, config_path: str, current_round: int) -> str:
        outputs_name = self._outputs_name(model_path, config_path)
        return f"round_{current_round}-outputs-{outputs_name}.json"

    def _outputs_dirname(self) -> str:
        function_name = str(self.cfg.experiment.get("function", "train"))
        if function_name == "evaluation":
            return "eval_rollouts"
        return "rollouts"

    def _load_outputs(self):
        root = Path(__file__).resolve().parents[1]
        model_path = self._resolve_model_path()
        config_path = str(self.cfg.config)
        current_round = int(self.cfg.experiment.current_round)
        run_name = self._run_name()
        outputs_file = self._outputs_filename(model_path, config_path, current_round)
        outputs_dir = self._outputs_dirname()
        path = root / "projects" / self.cfg.experiment.project / run_name / outputs_dir / outputs_file
        if path.exists():
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data, path

        legacy_file = self._outputs_filename_legacy_json(model_path, config_path, current_round)
        legacy_path = root / "projects" / self.cfg.experiment.project / run_name / outputs_dir / legacy_file
        with open(legacy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        path = legacy_path
        return data, path

    def _save_outputs(self, data, path: Path):
        if path.suffix == ".jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def build(self):
        data, outputs_path = self._load_outputs()

        is_seq_jsonl = bool(data) and isinstance(data[0].get("generated_output", None), str)

        if is_seq_jsonl:
            groups = {}
            for idx, item in enumerate(data):
                key = (item.get("problem_id", ""), item.get("prompt", ""))
                groups.setdefault(key, []).append(idx)

            for idx_list in groups.values():
                comps = []
                for idx in idx_list:
                    item = data[idx]
                    resp = item.get("generated_output", "")
                    ext = item.get("extracted_output", "")
                    comps.append(self.model.score_response_components(item, resp, ext))

                raw_rewards = [float(x["reward"]) for x in comps]
                advantages, group_mean, group_std = grouped_grpo_advantages(raw_rewards)

                for j, idx in enumerate(idx_list):
                    item = data[idx]
                    item["reward"] = float(comps[j]["reward"])
                    item["advantage"] = float(advantages[j])
                    item["correctness"] = bool(comps[j]["correctness"])
                    item["compilation"] = bool(comps[j]["compilation"])
                    item["speedup"] = float(comps[j]["speedup"])
                    item["baseline_instruction_count"] = comps[j]["baseline_instruction_count"]
                    item["candidate_instruction_count"] = comps[j]["candidate_instruction_count"]
                    item["group_reward_mean"] = float(group_mean)
                    item["group_reward_std"] = float(group_std)
                    item["group_size"] = int(len(idx_list))

                    if self.cfg.experiment.function != "train":
                        item["step_map"] = []
        else:
            for item in data:
                outputs = item.get("generated_output", item.get("full_output", []))
                comps = []
                for resp, ext in zip(outputs, item.get("extracted_output", [])):
                    comps.append(self.model.score_response_components(item, resp, ext))

                item["rewards"] = [float(x["reward"]) for x in comps]
                item["correctness"] = [bool(x["correctness"]) for x in comps]
                item["compilation"] = [bool(x["compilation"]) for x in comps]
                item["speedup"] = [float(x["speedup"]) for x in comps]

                if self.cfg.experiment.function != "train":
                    item["step_map"] = []

        outputs_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_outputs(data, outputs_path)

        print(f"Updated outputs with rewards: {outputs_path}")
        self._log_to_wandb(data)

    def _log_to_wandb(self, data: list):
        """Log reward statistics to wandb (single-process, best-effort)."""
        try:
            import wandb
        except ImportError:
            return
        try:
            cfg = self.cfg
            root = Path(__file__).resolve().parents[1]
            project_name = str(cfg.experiment.project)
            run_name     = self._run_name()
            current_round = int(cfg.experiment.current_round)
            function      = str(cfg.experiment.get("function", "train"))

            project_dir = root / "projects" / project_name / run_name
            run_id      = _get_wandb_run_id(project_dir)

            stats = _compute_stats_from_data(data, function)
            if not stats:
                return

            prefix = "reward/eval" if function != "train" else "reward/train"
            log_dict = {
                "reward/round": current_round,
                **{f"{prefix}_{k}": v for k, v in stats.items()},
            }

            run = wandb.init(
                project=project_name,
                name=run_name,
                id=run_id,
                resume="allow",
                entity=cfg.wandb.get("entity", None),
            )
            run.define_metric("reward/round")
            run.define_metric("reward/*", step_metric="reward/round")
            run.log(log_dict)
            run.finish()

            print(
                f"[RewardModel] wandb logged {prefix}: "
                + ", ".join(
                    f"{k}={v:.4f}" for k, v in stats.items() if isinstance(v, float)
                )
            )
        except Exception as exc:
            print(f"[RewardModel] wandb logging failed (non-fatal): {exc}")


def run(cfg):
    builder = RewardBuilder(cfg)
    builder.build()


def main():
    cfg = get_config()
    run(cfg)


if __name__ == "__main__":
    main()
