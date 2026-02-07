"""
Credit Estimator
从反事实结果估计每个步骤的credit分数
"""

import math
import statistics
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass

from shared.envs.base import Trajectory
from .counterfactual import CounterfactualResult, InterventionType


@dataclass
class CreditMap:
    """Credit分布"""
    trajectory_id: str
    step_credits: List[float]  # 每步的credit
    normalized_credits: List[float]  # 归一化后的credit
    earliest_success_step: Optional[int]  # 最早成功点
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def entropy(self) -> float:
        """计算credit分布的熵"""
        # 用 |credit| 作为贡献强度分布，避免负值导致的“非概率”问题
        mags = [abs(x) for x in self.step_credits]
        total = sum(mags)
        if total <= 1e-12:
            return 0.0

        entropy = 0.0
        for m in mags:
            p = (m / total) + 1e-12
            entropy -= p * math.log(p)
        return float(entropy)
    
    @property
    def spread(self) -> float:
        """计算credit的spread（标准差）"""
        if len(self.step_credits) <= 1:
            return 0.0
        # population std 更符合“分布离散度”直觉
        return float(statistics.pstdev(self.step_credits))


class CreditEstimator:
    """
    Credit估计器
    从反事实结果计算每步的边际贡献
    """
    
    def __init__(
        self,
        aggregation: str = "mean",  # mean, max, weighted
        normalization: str = "signed",  # signed, minmax, zscore, softmax, none
        use_block_credit: bool = False,
        block_size: int = 2
    ):
        self.aggregation = aggregation
        self.normalization = normalization
        self.use_block_credit = use_block_credit
        self.block_size = block_size
    
    def estimate(
        self,
        trajectory: Trajectory,
        cf_results: List[CounterfactualResult],
        *,
        base_reward: Optional[float] = None,
        cf_reward_fn: Optional[Callable[[CounterfactualResult], float]] = None,
    ) -> CreditMap:
        """
        从反事实结果估计credit
        
        核心公式: c_t = E[r(τ) - r(τ \\ a_t)]
        即删除第t步后成功率的下降
        """
        n_steps = len(trajectory.steps)
        
        # 初始化credit
        step_credits = [0.0] * n_steps
        step_counts = [0] * n_steps
        step_max_credits: List[Optional[float]] = [None] * n_steps
        
        # 寻找最早成功点
        earliest_success = None
        
        if base_reward is None:
            base_reward = float(trajectory.r_final)
        else:
            base_reward = float(base_reward)

        def _cf_reward(cf: CounterfactualResult) -> float:
            if cf_reward_fn is not None:
                return float(cf_reward_fn(cf))
            return float(cf.r_final_cf)

        def _accumulate_step_credit(step_idx: int, credit_value: float) -> None:
            if self.aggregation == "max":
                prev = step_max_credits[step_idx]
                if prev is None or abs(float(credit_value)) > abs(float(prev)):
                    step_max_credits[step_idx] = float(credit_value)
            else:
                step_credits[step_idx] += float(credit_value)
            step_counts[step_idx] += 1

        for cf in cf_results:
            if not cf.is_valid:
                continue
            
            intv = cf.intervention
            
            # 根据干预类型计算credit
            if intv.intervention_type == InterventionType.DELETE_STEP:
                # 删除单步：credit = 原始reward - 反事实reward
                t = intv.target_step
                if 0 <= t < n_steps:
                    credit = base_reward - _cf_reward(cf)
                    _accumulate_step_credit(t, credit)
            
            elif intv.intervention_type == InterventionType.DELETE_BLOCK:
                # 删除block：credit分配给block内的步骤
                start = intv.target_step
                end = intv.end_step or (start + 1)
                credit = base_reward - _cf_reward(cf)
                block_credit = credit / (end - start)
                for t in range(start, min(end, n_steps)):
                    _accumulate_step_credit(t, block_credit)
            
            elif intv.intervention_type == InterventionType.TRUNCATE:
                # 截断：找最早成功点
                trunc_point = intv.target_step
                if cf.r_final_cf > 0.5:  # 截断后仍成功
                    if earliest_success is None or trunc_point < earliest_success:
                        earliest_success = trunc_point
            
            elif intv.intervention_type == InterventionType.SWAP_STEP:
                # 替换：credit = 原始reward - 反事实reward
                t = intv.target_step
                if 0 <= t < n_steps:
                    credit = base_reward - _cf_reward(cf)
                    _accumulate_step_credit(t, credit)
        
        # 聚合
        for t in range(n_steps):
            if step_counts[t] > 0:
                if self.aggregation == "mean":
                    step_credits[t] /= step_counts[t]
                elif self.aggregation == "max":
                    step_credits[t] = float(step_max_credits[t]) if step_max_credits[t] is not None else 0.0
        
        # 如果有最早成功点，将credit集中到该点附近
        if earliest_success is not None:
            # 给最早成功点之前的步骤加权
            for t in range(earliest_success):
                step_credits[t] *= 1.5
            # 给之后的步骤降权
            for t in range(earliest_success, n_steps):
                step_credits[t] *= 0.5
        
        # 归一化
        normalized = self._normalize(step_credits)
        
        return CreditMap(
            trajectory_id=trajectory.trajectory_id,
            step_credits=step_credits,
            normalized_credits=normalized,
            earliest_success_step=earliest_success,
            metadata={
                "n_cf_results": len(cf_results),
                "n_valid": sum(1 for cf in cf_results if cf.is_valid)
            }
        )
    
    def _normalize(self, credits: List[float]) -> List[float]:
        """归一化credit"""
        if not credits:
            return []

        if self.normalization == "none":
            return credits

        if self.normalization == "signed":
            max_abs = max(abs(x) for x in credits)
            if max_abs <= 1e-12:
                return [0.0 for _ in credits]
            return [float(x / max_abs) for x in credits]

        if self.normalization == "minmax":
            min_val = min(credits)
            max_val = max(credits)
            if max_val - min_val <= 1e-12:
                return [0.0 for _ in credits]
            return [float((x - min_val) / (max_val - min_val)) for x in credits]

        if self.normalization == "zscore":
            if len(credits) <= 1:
                return [0.0 for _ in credits]
            mean = statistics.fmean(credits)
            std = statistics.pstdev(credits)
            if std <= 1e-12:
                return [0.0 for _ in credits]
            return [float((x - mean) / std) for x in credits]

        if self.normalization == "softmax":
            m = max(credits)
            exp_vals = [math.exp(x - m) for x in credits]
            denom = sum(exp_vals)
            if denom <= 1e-12:
                return [0.0 for _ in credits]
            return [float(v / denom) for v in exp_vals]

        raise ValueError(f"Unknown normalization: {self.normalization}")
    
    def estimate_batch(
        self,
        trajectories: List[Trajectory],
        cf_results_list: List[List[CounterfactualResult]]
    ) -> List[CreditMap]:
        """批量估计"""
        return [
            self.estimate(traj, cf_results)
            for traj, cf_results in zip(trajectories, cf_results_list)
        ]
