"""
Counterfactual Generator and Executor
反事实干预生成器与执行器

核心思想：通过删除/替换/截断步骤来估计每个步骤对最终成功的边际贡献
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import random
import copy

from shared.envs.base import Trajectory, Step, Action, ActionType, Observation, VerifierInfo


class InterventionType(Enum):
    """干预类型"""
    DELETE_STEP = "delete"      # 删除某一步
    DELETE_BLOCK = "delete_block"  # 删除连续几步
    SWAP_STEP = "swap"          # 替换某一步的动作
    TRUNCATE = "truncate"       # 截断到某一步
    PERTURB = "perturb"         # 轻微扰动


@dataclass
class InterventionSpec:
    """干预规格"""
    intervention_type: InterventionType
    target_step: int  # 目标步骤索引
    end_step: Optional[int] = None  # 对于block操作
    replacement_action: Optional[Action] = None  # 对于swap操作
    perturbation_params: Optional[Dict[str, Any]] = None
    
    def __repr__(self):
        if self.intervention_type == InterventionType.DELETE_BLOCK:
            return f"DeleteBlock({self.target_step}:{self.end_step})"
        elif self.intervention_type == InterventionType.SWAP_STEP:
            repl = self.replacement_action.get_hash() if self.replacement_action is not None else "none"
            return f"Swap({self.target_step}, repl={repl})"
        elif self.intervention_type == InterventionType.TRUNCATE:
            return f"Truncate({self.target_step})"
        return f"{self.intervention_type.value}({self.target_step})"


@dataclass
class CounterfactualResult:
    """反事实结果"""
    base_trajectory_id: str
    intervention: InterventionSpec
    cf_trajectory: Optional[Trajectory]
    r_final_cf: float
    is_valid: bool  # 反事实轨迹是否合法可执行
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CounterfactualGenerator:
    """
    反事实干预生成器
    从一条轨迹生成K个反事实干预
    """
    
    def __init__(
        self,
        intervention_types: List[str] = None,
        k: int = 4,
        block_size: int = 2,
        prioritize_high_value: bool = True,
        seed: int = 42
    ):
        """
        Args:
            intervention_types: 要使用的干预类型列表
            k: 每条轨迹生成的反事实数量
            block_size: block删除的大小
            prioritize_high_value: 是否优先对高价值轨迹生成更多反事实
        """
        if intervention_types is None:
            intervention_types = ["delete", "truncate"]
        
        self.intervention_types = [
            InterventionType(t) if isinstance(t, str) else t 
            for t in intervention_types
        ]
        self.k = k
        self.block_size = block_size
        self.prioritize_high_value = prioritize_high_value
        self.rng = random.Random(seed)
    
    def generate(self, trajectory: Trajectory) -> List[InterventionSpec]:
        """
        为一条轨迹生成K个干预规格
        """
        interventions = []
        n_steps = len(trajectory.steps)
        
        if n_steps == 0:
            return []
        
        # 确定每种干预类型的数量
        k_per_type = max(1, self.k // len(self.intervention_types))
        
        for int_type in self.intervention_types:
            if int_type == InterventionType.DELETE_STEP:
                # 随机选择要删除的步骤
                candidates = list(range(n_steps))
                self.rng.shuffle(candidates)
                for step_idx in candidates[:k_per_type]:
                    interventions.append(InterventionSpec(
                        intervention_type=InterventionType.DELETE_STEP,
                        target_step=step_idx
                    ))
            
            elif int_type == InterventionType.DELETE_BLOCK:
                # 删除连续的block
                if n_steps >= self.block_size:
                    max_start = n_steps - self.block_size
                    candidates = list(range(max_start + 1))
                    self.rng.shuffle(candidates)
                    for start_idx in candidates[:k_per_type]:
                        interventions.append(InterventionSpec(
                            intervention_type=InterventionType.DELETE_BLOCK,
                            target_step=start_idx,
                            end_step=start_idx + self.block_size
                        ))
            
            elif int_type == InterventionType.TRUNCATE:
                # 截断到不同位置，找最早成功点
                truncate_points = []
                for i in range(1, n_steps + 1):
                    truncate_points.append(i)
                self.rng.shuffle(truncate_points)
                for trunc_point in truncate_points[:k_per_type]:
                    interventions.append(InterventionSpec(
                        intervention_type=InterventionType.TRUNCATE,
                        target_step=trunc_point
                    ))
            
            elif int_type == InterventionType.SWAP_STEP:
                # swap需要知道可替换的动作，这里只生成规格
                candidates = list(range(n_steps))
                self.rng.shuffle(candidates)
                for step_idx in candidates[:k_per_type]:
                    interventions.append(InterventionSpec(
                        intervention_type=InterventionType.SWAP_STEP,
                        target_step=step_idx,
                        replacement_action=None  # 由executor填充
                    ))
        
        # 限制总数为k
        if len(interventions) > self.k:
            self.rng.shuffle(interventions)
            interventions = interventions[:self.k]
        
        return interventions
    
    def should_generate_counterfactuals(self, trajectory: Trajectory) -> bool:
        """
        判断是否应该为此轨迹生成反事实
        只对高价值轨迹（成功或接近成功）生成
        """
        if not self.prioritize_high_value:
            return True
        
        # 成功轨迹一定生成
        if trajectory.success:
            return True
        
        # 接近成功的轨迹也生成（如部分测试通过）
        if hasattr(trajectory.verifier_info, 'score') and trajectory.verifier_info.score > 0.5:
            return True
        
        return False


class CounterfactualExecutor:
    """
    反事实执行器
    执行反事实轨迹并获取新的验证结果
    """
    
    def __init__(self, env, use_cache: bool = True, seed: int = 42):
        """
        Args:
            env: 环境实例
            use_cache: 是否使用缓存
        """
        self.env = env
        self.use_cache = use_cache
        self._cache: Dict[str, CounterfactualResult] = {}
        self.rng = random.Random(seed)
    
    def execute(
        self,
        trajectory: Trajectory,
        intervention: InterventionSpec
    ) -> CounterfactualResult:
        """
        执行一个反事实并返回结果
        """
        # 检查缓存
        cache_key = f"{trajectory.trajectory_id}_{intervention}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # SWAP_STEP: 如果未提供 replacement_action，则自动构造一个“相似但不同”的替代动作
            intervention = copy.deepcopy(intervention)
            if (
                intervention.intervention_type == InterventionType.SWAP_STEP
                and intervention.replacement_action is None
            ):
                replacement = self._propose_replacement_action(trajectory, intervention.target_step)
                if replacement is None:
                    result = CounterfactualResult(
                        base_trajectory_id=trajectory.trajectory_id,
                        intervention=intervention,
                        cf_trajectory=None,
                        r_final_cf=0.0,
                        is_valid=False,
                        error_message="Unable to propose replacement action for swap",
                    )
                    if self.use_cache:
                        self._cache[cache_key] = result
                    return result
                intervention.replacement_action = replacement

            cf_trajectory = self._apply_intervention(trajectory, intervention)
            
            if cf_trajectory is None:
                result = CounterfactualResult(
                    base_trajectory_id=trajectory.trajectory_id,
                    intervention=intervention,
                    cf_trajectory=None,
                    r_final_cf=0.0,
                    is_valid=False,
                    error_message="Failed to apply intervention"
                )
            else:
                # 重新执行并验证
                r_final_cf, verifier_info = self._replay_and_verify(cf_trajectory)
                cf_trajectory.r_final = float(r_final_cf)
                cf_trajectory.verifier_info = verifier_info
                
                result = CounterfactualResult(
                    base_trajectory_id=trajectory.trajectory_id,
                    intervention=intervention,
                    cf_trajectory=cf_trajectory,
                    r_final_cf=r_final_cf,
                    is_valid=True
                )
        
        except Exception as e:
            result = CounterfactualResult(
                base_trajectory_id=trajectory.trajectory_id,
                intervention=intervention,
                cf_trajectory=None,
                r_final_cf=0.0,
                is_valid=False,
                error_message=str(e)
            )
        
        if self.use_cache:
            self._cache[cache_key] = result
        
        return result

    def _propose_replacement_action(self, trajectory: Trajectory, target_step: int) -> Optional[Action]:
        """
        为 SWAP_STEP 生成一个替代动作。

        规则（尽量确定性、低工程风险）：
        1) 优先从同一轨迹中找“同 action_type 但内容不同”的其他动作进行替换；
        2) 否则根据 task 元信息（initial_code 等）或对原 action 做轻微扰动。
        """
        if not (0 <= target_step < len(trajectory.steps)):
            return None

        original = trajectory.steps[target_step].action

        # 1) 从轨迹内找同类型替代动作（最近邻优先）
        for offset in range(1, len(trajectory.steps)):
            for idx in (target_step - offset, target_step + offset):
                if 0 <= idx < len(trajectory.steps):
                    candidate = trajectory.steps[idx].action
                    if candidate.action_type != original.action_type:
                        continue
                    if (
                        candidate.content != original.content
                        or candidate.tool_name != original.tool_name
                        or (candidate.tool_args or {}) != (original.tool_args or {})
                    ):
                        return copy.deepcopy(candidate)

        task = (trajectory.metadata or {}).get("task") or {}

        # 2) 回退：按类型构造轻微扰动
        if original.action_type == ActionType.CODE_WRITE:
            initial_code = task.get("initial_code")
            if isinstance(initial_code, str) and initial_code and initial_code != original.content:
                return Action(
                    action_type=ActionType.CODE_WRITE,
                    content=initial_code,
                    metadata={"swap_from": "initial_code"},
                )

            lines = original.content.splitlines()
            if len(lines) > 1:
                perturbed = "\n".join(lines[:-1]).rstrip() + "\n"
                if perturbed != original.content:
                    return Action(
                        action_type=ActionType.CODE_WRITE,
                        content=perturbed,
                        metadata={"swap_from": "drop_last_line"},
                    )
            return None

        if original.action_type == ActionType.TOOL_CALL:
            tool_name = original.tool_name or ""
            tool_swap = {
                # code env
                "run_tests": "get_code",
                "get_code": "run_tests",
                # sql env
                "execute_sql": "show_schema",
                "show_schema": "execute_sql",
                "submit_query": "execute_sql",
            }.get(tool_name)
            if tool_swap:
                return Action(
                    action_type=ActionType.TOOL_CALL,
                    content=original.content,
                    tool_name=tool_swap,
                    tool_args=copy.deepcopy(original.tool_args),
                    metadata={"swap_from": tool_name},
                )
            return None

        # SQL: 非 TOOL_CALL 的 query / TEXT_RESPONSE 等，做轻微 query 变体
        if isinstance(original.content, str) and original.content.strip():
            q = original.content.strip()
            lower = q.lower()
            if " where " in lower:
                idx = lower.find(" where ")
                perturbed = q[:idx].rstrip()
            else:
                perturbed = f"{q} LIMIT 1"
            if perturbed != original.content:
                return Action(
                    action_type=original.action_type,
                    content=perturbed,
                    tool_name=original.tool_name,
                    tool_args=copy.deepcopy(original.tool_args),
                    metadata={"swap_from": "query_perturb"},
                )

        return None
    
    def _apply_intervention(
        self,
        trajectory: Trajectory,
        intervention: InterventionSpec
    ) -> Optional[Trajectory]:
        """应用干预生成新轨迹"""
        
        new_steps = copy.deepcopy(trajectory.steps)
        
        if intervention.intervention_type == InterventionType.DELETE_STEP:
            if 0 <= intervention.target_step < len(new_steps):
                del new_steps[intervention.target_step]
            else:
                return None
        
        elif intervention.intervention_type == InterventionType.DELETE_BLOCK:
            start = intervention.target_step
            end = intervention.end_step or (start + 1)
            if 0 <= start < end <= len(new_steps):
                new_steps = new_steps[:start] + new_steps[end:]
            else:
                return None
        
        elif intervention.intervention_type == InterventionType.TRUNCATE:
            trunc_point = intervention.target_step
            if 0 < trunc_point <= len(new_steps):
                new_steps = new_steps[:trunc_point]
            else:
                return None
        
        elif intervention.intervention_type == InterventionType.SWAP_STEP:
            if intervention.replacement_action is None:
                return None
            if 0 <= intervention.target_step < len(new_steps):
                new_steps[intervention.target_step].action = intervention.replacement_action
            else:
                return None
        
        # 创建新轨迹
        cf_trajectory = Trajectory(
            trajectory_id=f"{trajectory.trajectory_id}_cf",
            task_id=trajectory.task_id,
            prompt=trajectory.prompt,
            steps=new_steps,
            r_final=0.0,  # 待重新计算
            verifier_info=VerifierInfo(success=False, score=0.0),
            metadata=dict(trajectory.metadata) | {
                "base_trajectory_id": trajectory.trajectory_id,
                "intervention": str(intervention),
            },
        )
        
        return cf_trajectory
    
    def _replay_and_verify(
        self,
        cf_trajectory: Trajectory
    ) -> Tuple[float, VerifierInfo]:
        """
        重放反事实轨迹并验证
        这需要环境支持确定性重放
        """
        task = (cf_trajectory.metadata or {}).get("task")
        if task is None:
            raise ValueError(
                "Missing task spec in trajectory.metadata['task']; "
                "store the original env.reset() input to enable replay."
            )

        # 重置环境并顺序执行动作（确定性 replay）
        self.env.reset(task)
        for step in cf_trajectory.steps:
            _, _, done, _ = self.env.step(step.action)
            if done:
                break

        verifier_info = self.env.verify() if hasattr(self.env, "verify") else VerifierInfo(success=False, score=0.0)
        r_final = 1.0 if verifier_info.success else 0.0
        return r_final, verifier_info
    
    def batch_execute(
        self,
        trajectory: Trajectory,
        interventions: List[InterventionSpec]
    ) -> List[CounterfactualResult]:
        """批量执行反事实"""
        return [self.execute(trajectory, intv) for intv in interventions]
