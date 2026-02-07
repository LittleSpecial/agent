"""
Advantage Mapper
将credit map映射为RL训练可用的advantage
"""

import math
from typing import List, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from shared.envs.base import Trajectory
from .credit_estimator import CreditMap


class AdvantageMapper:
    """
    将credit映射为advantage
    支持step-level和token-level映射
    """
    
    def __init__(
        self,
        level: str = "step",  # step, token, action
        length_normalize: bool = True,
        clip_range: Optional[float] = 3.0
    ):
        self.level = level
        self.length_normalize = length_normalize
        self.clip_range = clip_range
    
    def map_to_step_advantages(
        self,
        trajectory: Trajectory,
        credit_map: CreditMap
    ):
        """
        将credit映射为step-level advantage
        """
        advantages_list = [float(x) for x in credit_map.normalized_credits]
        
        # 长度校正
        if self.length_normalize:
            n_steps = len(trajectory.steps)
            if n_steps > 1:
                # 防止长轨迹credit被稀释
                scale = math.sqrt(n_steps)
                advantages_list = [x * scale for x in advantages_list]
        
        # 裁剪
        if self.clip_range is not None:
            lo, hi = -self.clip_range, self.clip_range
            advantages_list = [min(max(x, lo), hi) for x in advantages_list]
        
        if torch is None:
            return advantages_list
        return torch.tensor(advantages_list, dtype=torch.float)
    
    def map_to_token_advantages(
        self,
        trajectory: Trajectory,
        credit_map: CreditMap
    ):
        """
        将credit映射为token-level advantage
        每个step的credit平均分配到该step的token上
        """
        token_advantages = []
        
        for i, step in enumerate(trajectory.steps):
            step_credit = credit_map.normalized_credits[i]
            
            if step.tokens is not None:
                n_tokens = len(step.tokens)
                # 均匀分配到每个token
                token_credit = step_credit / max(1, n_tokens)
                token_advantages.extend([token_credit] * n_tokens)
            else:
                # 如果没有token信息，当作一个token处理
                token_advantages.append(step_credit)
        
        advantages_list = [float(x) for x in token_advantages]
        
        if self.clip_range is not None:
            lo, hi = -self.clip_range, self.clip_range
            advantages_list = [min(max(x, lo), hi) for x in advantages_list]
        
        if torch is None:
            return advantages_list
        return torch.tensor(advantages_list, dtype=torch.float)
    
    def map_batch(
        self,
        trajectories: List[Trajectory],
        credit_maps: List[CreditMap]
    ) -> List:
        """批量映射"""
        if self.level == "step":
            return [
                self.map_to_step_advantages(traj, cm)
                for traj, cm in zip(trajectories, credit_maps)
            ]
        elif self.level == "token":
            return [
                self.map_to_token_advantages(traj, cm)
                for traj, cm in zip(trajectories, credit_maps)
            ]
        else:
            raise ValueError(f"Unknown level: {self.level}")
