"""
Experiment Tracker - 实验追踪器
自动创建实验文档，记录参数、输入输出
"""

import os
import json
import time
import hashlib
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str
    project: str  # "paper_a" or "paper_b"
    description: str = ""
    
    # 模型配置
    model_name: str = "Qwen2.5-7B"
    use_lora: bool = True
    lora_rank: int = 64
    
    # 训练配置
    algorithm: str = "GRPO"
    learning_rate: float = 1e-5
    batch_size: int = 32
    max_steps: int = 10000
    
    # 环境配置
    env_type: str = "code"
    max_trajectory_length: int = 20
    
    # RL配置
    gamma: float = 1.0
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    
    # Paper A 配置
    use_counterfactual_credit: bool = False
    counterfactual_k: int = 4
    intervention_types: List[str] = field(default_factory=lambda: ["delete", "truncate"])
    
    # Paper B 配置
    use_conflict_aware: bool = False
    num_groups: int = 4
    surgery_method: str = "pcgrad"
    
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetrics:
    """运行时指标"""
    step: int = 0
    train_loss: float = 0.0
    success_rate: float = 0.0
    pass_at_1: float = 0.0
    pass_at_k: Dict[int, float] = field(default_factory=dict)
    avg_trajectory_length: float = 0.0
    wall_time: float = 0.0
    # Paper B
    conflict_ratio: float = 0.0
    solution_diversity: float = 0.0
    # Paper A
    avg_credit_spread: float = 0.0
    # Extra scalars for analysis (e.g., AGOP stats)
    extra: Dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """实验追踪器"""
    
    def __init__(self, config: ExperimentConfig, base_dir: str = "./experiments"):
        self.config = config
        self.base_dir = Path(base_dir)
        self.start_time = datetime.now()
        self.metrics_history: List[RunMetrics] = []
        self.experiment_dir = self._create_experiment_dir()
        self._save_config()
        self._create_experiment_doc()
    
    def _create_experiment_dir(self) -> Path:
        """创建实验目录"""
        # Include microseconds to avoid collisions when launching multiple runs quickly.
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S_%f")
        config_hash = hashlib.md5(
            json.dumps(asdict(self.config), sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        
        dir_name_base = f"{self.config.project}_{self.config.experiment_name}_{timestamp}_{config_hash}"
        exp_dir = self.base_dir / dir_name_base
        if exp_dir.exists():
            suffix = 1
            while (self.base_dir / f"{dir_name_base}_{suffix}").exists():
                suffix += 1
            exp_dir = self.base_dir / f"{dir_name_base}_{suffix}"
        
        for subdir in ["checkpoints", "logs", "artifacts", "analysis"]:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return exp_dir
    
    def _save_config(self):
        """保存配置"""
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def _create_experiment_doc(self):
        """创建实验文档"""
        doc = f"""# 实验记录: {self.config.experiment_name}

## 基本信息
- 项目: {self.config.project}
- 开始时间: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}
- 描述: {self.config.description}

## 模型配置
- 模型: {self.config.model_name}
- LoRA: {self.config.use_lora}, rank={self.config.lora_rank}

## 训练配置
- 算法: {self.config.algorithm}
- 学习率: {self.config.learning_rate}
- Batch: {self.config.batch_size}
- 种子: {self.config.seed}

## 运行日志

"""
        with open(self.experiment_dir / "EXPERIMENT.md", 'w') as f:
            f.write(doc)
    
    def log_metrics(self, metrics: RunMetrics):
        """记录指标"""
        self.metrics_history.append(metrics)
        with open(self.experiment_dir / "logs" / "metrics.jsonl", 'a') as f:
            record = asdict(metrics)
            record['timestamp'] = datetime.now().isoformat()
            f.write(json.dumps(record) + '\n')
    
    def log_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """记录事件"""
        with open(self.experiment_dir / "logs" / "events.jsonl", 'a') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'type': event_type,
                'message': message,
                'data': data or {}
            }) + '\n')
        
        with open(self.experiment_dir / "EXPERIMENT.md", 'a') as f:
            f.write(f"\n`{datetime.now().strftime('%H:%M:%S')}` [{event_type}] {message}\n")
    
    def save_artifact(self, name: str, data: Any):
        """保存artifact"""
        path = self.experiment_dir / "artifacts" / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path
    
    def finalize(self):
        """实验结束"""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            summary = f"""
## 最终结果
- 步数: {latest.step}
- 成功率: {latest.success_rate:.4f}
- Pass@1: {latest.pass_at_1:.4f}
"""
            with open(self.experiment_dir / "EXPERIMENT.md", 'a') as f:
                f.write(summary)
        
        self.log_event("finished", "Experiment completed")
        print(f"Results: {self.experiment_dir}")
