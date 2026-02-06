from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent_rl_core.runner import RolloutRunner


@dataclass
class TrainConfig:
    total_updates: int = 1000
    batch_size: int = 32
    learning_rate: float = 2e-2
    cost_penalty: float = 0.02
    seed: int = 42
    log_interval: int = 20
    eval_interval: int = 100
    save_interval: int = 100
    output_dir: str = "outputs/baseline"
    checkpoint_name: str = "policy_latest.json"
    extra: Dict[str, Any] = field(default_factory=dict)


class BaselineTrainer:
    def __init__(
        self,
        config: TrainConfig,
        runner: RolloutRunner,
        policy: Any,
        train_tasks: List[Dict[str, Any]],
        eval_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.policy = policy
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks or []
        self.rng = random.Random(config.seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.eval_path = self.output_dir / "eval.jsonl"

    def _sample_tasks(self, tasks: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        if not tasks:
            raise ValueError("task list is empty")
        return [tasks[self.rng.randrange(len(tasks))] for _ in range(n)]

    def _run_batch(self, tasks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.runner.run_episode(task) for task in tasks]

    @staticmethod
    def _aggregate_rollouts(rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        if not rollouts:
            return {}

        n = float(len(rollouts))
        reward = sum(float(r.get("final_reward", 0.0)) for r in rollouts) / n

        verifier_scores = []
        successes = 0.0
        costs: Dict[str, float] = {}
        for rollout in rollouts:
            verifier = rollout.get("verifier", {})
            verifier_scores.append(float(verifier.get("score", 0.0)))
            if bool(verifier.get("success", False)):
                successes += 1.0
            for key, value in dict(rollout.get("final_costs", {})).items():
                costs[key] = costs.get(key, 0.0) + float(value)

        for key in list(costs):
            costs[key] /= n

        metrics: Dict[str, float] = {
            "avg_reward": reward,
            "avg_verifier_score": (sum(verifier_scores) / n) if verifier_scores else 0.0,
            "success_rate": successes / n,
        }
        for key, value in costs.items():
            metrics[f"avg_cost_{key}"] = value
        return metrics

    def _write_jsonl(self, path: Path, row: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _save_checkpoint(self, step: int) -> Path:
        ckpt = self.output_dir / self.config.checkpoint_name
        payload = {
            "step": step,
            "policy_state": self.policy.state_dict() if hasattr(self.policy, "state_dict") else {},
            "trainer_config": {
                "total_updates": self.config.total_updates,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "cost_penalty": self.config.cost_penalty,
            },
        }
        ckpt.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return ckpt

    def evaluate(self, tasks: Optional[List[Dict[str, Any]]] = None, step: int = 0) -> Dict[str, float]:
        eval_tasks = tasks if tasks is not None else self.eval_tasks
        if not eval_tasks:
            return {}
        rollouts = self._run_batch(eval_tasks)
        metrics = self._aggregate_rollouts(rollouts)
        row = {"step": step, "split": "eval", **metrics}
        self._write_jsonl(self.eval_path, row)
        return metrics

    def train(self) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        latest_eval: Dict[str, float] = {}

        for step in range(1, self.config.total_updates + 1):
            tasks = self._sample_tasks(self.train_tasks, self.config.batch_size)
            rollouts = self._run_batch(tasks)
            train_metrics = self._aggregate_rollouts(rollouts)

            if hasattr(self.policy, "update_from_rollouts"):
                self.policy.update_from_rollouts(
                    rollouts=rollouts,
                    learning_rate=self.config.learning_rate,
                    cost_penalty=self.config.cost_penalty,
                )

            row: Dict[str, Any] = {"step": step, "split": "train", **train_metrics}
            history.append(row)
            self._write_jsonl(self.metrics_path, row)

            if step % self.config.eval_interval == 0:
                latest_eval = self.evaluate(step=step)
            if step % self.config.save_interval == 0:
                self._save_checkpoint(step)

        final_ckpt = self._save_checkpoint(self.config.total_updates)
        return {
            "history": history,
            "latest_eval": latest_eval,
            "checkpoint": str(final_ckpt),
            "output_dir": str(self.output_dir),
        }
