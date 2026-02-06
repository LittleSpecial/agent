"""
Config Manager - 配置管理器
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import asdict

from .tracker import ExperimentConfig

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_config(path: str) -> ExperimentConfig:
        """从文件加载配置"""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            if yaml is None:
                raise ModuleNotFoundError(
                    "PyYAML is required to load .yaml/.yml configs. Install with `pip install pyyaml`."
                )
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return ExperimentConfig(**data)
    
    @staticmethod
    def save_config(config: ExperimentConfig, path: str):
        """保存配置到文件"""
        path = Path(path)
        data = asdict(config)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif path.suffix in ['.yaml', '.yml']:
            if yaml is None:
                raise ModuleNotFoundError(
                    "PyYAML is required to save .yaml/.yml configs. Install with `pip install pyyaml`."
                )
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def merge_configs(base: ExperimentConfig, override: Dict[str, Any]) -> ExperimentConfig:
        """合并配置"""
        data = asdict(base)
        data.update(override)
        return ExperimentConfig(**data)
