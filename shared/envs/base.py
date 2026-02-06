"""
Base Environment Interface for RLVR
统一的可验证奖励环境接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import json


class ActionType(Enum):
    """动作类型枚举"""
    TOOL_CALL = "tool_call"
    CODE_WRITE = "code_write"
    CODE_EXECUTE = "code_execute"
    TEXT_RESPONSE = "text_response"


@dataclass
class Action:
    """统一的动作表示"""
    action_type: ActionType
    content: str  # 动作内容（代码、工具调用JSON等）
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hash(self) -> str:
        """获取动作的唯一哈希"""
        hash_content = json.dumps({
            "type": self.action_type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args
        }, sort_keys=True)
        return hashlib.md5(hash_content.encode()).hexdigest()[:16]


@dataclass
class Observation:
    """环境观察"""
    content: str
    obs_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hash(self) -> str:
        """获取观察的唯一哈希"""
        return hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class Step:
    """轨迹中的一步"""
    step_id: int
    observation: Observation
    action: Action
    logprob: Optional[float] = None
    tokens: Optional[List[str]] = None
    token_logprobs: Optional[List[float]] = None
    
    def get_hash(self) -> str:
        """获取步骤的唯一哈希"""
        return f"{self.observation.get_hash()}_{self.action.get_hash()}"


@dataclass
class VerifierInfo:
    """验证器输出的详细信息"""
    success: bool
    score: float  # 0.0 - 1.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    passed_tests: int = 0
    total_tests: int = 0
    diff_info: Optional[str] = None  # SQL/JSON等的diff信息
    earliest_success_step: Optional[int] = None  # 最早成功的步骤
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """完整轨迹"""
    trajectory_id: str
    task_id: str
    prompt: str
    steps: List[Step]
    r_final: float  # 最终奖励 (通常是0或1)
    verifier_info: VerifierInfo
    total_tokens: int = 0
    total_tool_calls: int = 0
    wall_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return len(self.steps)
    
    @property
    def success(self) -> bool:
        return self.r_final > 0.5
    
    def get_step_hashes(self) -> List[str]:
        """获取所有步骤的哈希列表"""
        return [step.get_hash() for step in self.steps]


@dataclass
class EnvConfig:
    """环境配置"""
    name: str
    max_steps: int = 20
    timeout_seconds: float = 60.0
    deterministic: bool = True
    seed: int = 42
    cache_tool_outputs: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseEnv(ABC):
    """
    可验证奖励环境的基类
    所有环境（Code/SQL/JSON等）都继承此类
    """
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self._tool_cache: Dict[str, Any] = {}
        self._current_trajectory: Optional[Trajectory] = None
        
    @abstractmethod
    def reset(self, task: Dict[str, Any]) -> Observation:
        """
        重置环境到新任务
        Args:
            task: 任务描述（包含prompt、expected_output等）
        Returns:
            初始观察
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        执行一步
        Args:
            action: 要执行的动作
        Returns:
            (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def verify(self) -> VerifierInfo:
        """
        验证当前状态是否成功
        Returns:
            验证结果信息
        """
        pass
    
    def get_trajectory(self) -> Optional[Trajectory]:
        """获取当前轨迹"""
        return self._current_trajectory
    
    def cache_tool_output(self, key: str, output: Any):
        """缓存工具输出（用于反事实重放）"""
        if self.config.cache_tool_outputs:
            self._tool_cache[key] = output
    
    def get_cached_tool_output(self, key: str) -> Optional[Any]:
        """获取缓存的工具输出"""
        return self._tool_cache.get(key)
    
    def clear_cache(self):
        """清空缓存"""
        self._tool_cache.clear()
