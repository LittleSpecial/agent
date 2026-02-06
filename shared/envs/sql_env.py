"""
SQL Environment for RLVR
SQL生成环境，使用查询结果匹配作为验证器
"""

import sqlite3
import tempfile
import os
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .base import (
    BaseEnv, EnvConfig, Action, ActionType,
    Observation, Step, Trajectory, VerifierInfo
)


@dataclass
class SQLTask:
    """SQL任务定义"""
    task_id: str
    prompt: str
    database_schema: str
    sample_data: Optional[str] = None  # INSERT语句
    expected_result: Optional[List[Tuple]] = None
    expected_query: Optional[str] = None  # 用于参考
    metadata: Dict[str, Any] = field(default_factory=dict)


class SQLEnv(BaseEnv):
    """
    SQL环境：支持SQL生成、执行、结果匹配验证
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.current_task: Optional[SQLTask] = None
        self.current_query: str = ""
        self.db_path: Optional[str] = None
        self.connection: Optional[sqlite3.Connection] = None
        self._steps: List[Step] = []
        self._step_counter: int = 0
        self._start_time: float = 0.0
        self._total_tokens: int = 0
        self._total_tool_calls: int = 0
        
    def reset(self, task: Dict[str, Any]) -> Observation:
        """重置环境到新任务"""
        # 清理旧数据库
        self._cleanup_db()
        self.clear_cache()
        
        self.current_task = SQLTask(
            task_id=task.get("task_id", str(uuid.uuid4())),
            prompt=task["prompt"],
            database_schema=task["database_schema"],
            sample_data=task.get("sample_data"),
            expected_result=task.get("expected_result"),
            expected_query=task.get("expected_query"),
            metadata=task.get("metadata", {}),
        )
        
        self.current_query = ""
        self._steps = []
        self._step_counter = 0
        self._start_time = time.time()
        self._total_tokens = 0
        self._total_tool_calls = 0
        
        # 创建数据库
        self._setup_database()
        
        # 构建初始观察
        obs_content = f"Task: {self.current_task.prompt}\n\n"
        obs_content += f"Database Schema:\n```sql\n{self.current_task.database_schema}\n```\n\n"
        
        if self.current_task.sample_data:
            obs_content += "Sample data has been loaded into the database.\n"
        
        return Observation(
            content=obs_content,
            obs_type="task_description",
            metadata={"task_id": self.current_task.task_id}
        )
    
    def _setup_database(self):
        """设置SQLite数据库"""
        # 创建临时数据库文件
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        # 执行schema
        cursor.executescript(self.current_task.database_schema)
        
        # 插入示例数据
        if self.current_task.sample_data:
            cursor.executescript(self.current_task.sample_data)
        
        self.connection.commit()
    
    def _cleanup_db(self):
        """清理数据库"""
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.db_path and os.path.exists(self.db_path):
            os.unlink(self.db_path)
            self.db_path = None
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """执行一步动作"""
        self._step_counter += 1
        if action.action_type == ActionType.TOOL_CALL:
            self._total_tool_calls += 1
        
        if action.action_type == ActionType.TOOL_CALL:
            if action.tool_name == "execute_sql":
                obs, reward, done, info = self._execute_sql(action.content)
            elif action.tool_name == "submit_query":
                obs, reward, done, info = self._submit_query(action.content)
            elif action.tool_name == "show_schema":
                obs = Observation(
                    content=f"```sql\n{self.current_task.database_schema}\n```",
                    obs_type="schema"
                )
                reward, done, info = 0.0, False, {"action": "show_schema"}
            else:
                obs = Observation(
                    content=f"Unknown tool: {action.tool_name}",
                    obs_type="error"
                )
                reward, done, info = 0.0, False, {"error": "unknown_tool"}
        else:
            # 默认当作SQL查询处理
            obs, reward, done, info = self._execute_sql(action.content)
        
        # 记录步骤
        step = Step(
            step_id=self._step_counter,
            observation=obs,
            action=action,
            logprob=action.metadata.get("logprob"),
            tokens=action.metadata.get("tokens"),
            token_logprobs=action.metadata.get("token_logprobs")
        )
        self._steps.append(step)
        
        # 检查是否达到最大步数
        if self._step_counter >= self.config.max_steps:
            done = True
            info["terminated_reason"] = "max_steps"
        
        return obs, reward, done, info
    
    def _execute_sql(self, query: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """执行SQL查询"""
        self.current_query = query
        
        # 检查缓存
        cache_key = f"sql_{hash(query)}"
        cached_result = self.get_cached_tool_output(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # 格式化输出
            if results:
                output = f"Columns: {', '.join(columns)}\n"
                output += "Results:\n"
                for row in results[:20]:  # 限制显示行数
                    output += f"  {row}\n"
                if len(results) > 20:
                    output += f"  ... ({len(results) - 20} more rows)\n"
            else:
                output = "Query executed successfully. No results returned."
            
            obs = Observation(
                content=output,
                obs_type="query_result",
                metadata={"columns": columns, "row_count": len(results)}
            )
            
            response = (obs, 0.0, False, {
                "action": "execute_sql",
                "success": True,
                "results": results,
                "columns": columns
            })
            
        except Exception as e:
            obs = Observation(
                content=f"SQL Error: {str(e)}",
                obs_type="error"
            )
            response = (obs, 0.0, False, {
                "action": "execute_sql",
                "success": False,
                "error": str(e)
            })
        
        self.cache_tool_output(cache_key, response)
        return response
    
    def _submit_query(self, query: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """提交最终查询并验证"""
        self.current_query = query
        verifier_info = self.verify()
        
        if verifier_info.success:
            obs = Observation(
                content="Query submitted successfully! Result matches expected output.",
                obs_type="success"
            )
            return obs, 1.0, True, {"action": "submit_query", "verifier_info": verifier_info}
        else:
            obs = Observation(
                content=f"Query result does not match expected output.\n{verifier_info.diff_info or verifier_info.error_message}",
                obs_type="failure"
            )
            return obs, 0.0, True, {"action": "submit_query", "verifier_info": verifier_info}
    
    def verify(self) -> VerifierInfo:
        """验证当前查询结果"""
        if not self.current_query.strip():
            return VerifierInfo(
                success=False,
                score=0.0,
                error_message="No query submitted"
            )
        
        if self.current_task.expected_result is None:
            return VerifierInfo(
                success=False,
                score=0.0,
                error_message="No expected result to compare"
            )
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(self.current_query)
            results = cursor.fetchall()
            
            # 比较结果
            expected = self.current_task.expected_result
            
            # 多重集合比较（忽略顺序但保留重复行）
            result_counter = Counter(results)
            expected_counter = Counter(expected)
            if result_counter == expected_counter:
                return VerifierInfo(
                    success=True,
                    score=1.0,
                    passed_tests=1,
                    total_tests=1
                )
            else:
                # 计算部分匹配分数
                expected_total = sum(expected_counter.values())
                matched_total = sum((result_counter & expected_counter).values())
                score = matched_total / max(1, expected_total)

                missing = expected_counter - result_counter
                extra = result_counter - expected_counter

                def _fmt_counter(counter: Counter, limit: int = 20) -> str:
                    items = list(counter.items())
                    parts: List[str] = []
                    for row, count in items[:limit]:
                        parts.append(f"{row} x{count}" if count != 1 else f"{row}")
                    suffix = "" if len(items) <= limit else f" ... (+{len(items) - limit} more)"
                    return ", ".join(parts) + suffix

                diff = f"Expected {len(expected)} rows, got {len(results)} rows.\n"
                diff += f"Matched: {matched_total}/{expected_total}\n"
                diff += f"Missing: {_fmt_counter(missing)}\n"
                diff += f"Extra: {_fmt_counter(extra)}"
                
                return VerifierInfo(
                    success=False,
                    score=score,
                    error_message="Result mismatch",
                    diff_info=diff
                )
                
        except Exception as e:
            return VerifierInfo(
                success=False,
                score=0.0,
                error_message=str(e),
                error_type="sql_error"
            )
    
    def get_trajectory(self) -> Trajectory:
        """获取当前轨迹"""
        verifier_info = self.verify()
        
        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            task_id=self.current_task.task_id,
            prompt=self.current_task.prompt,
            steps=self._steps.copy(),
            r_final=1.0 if verifier_info.success else 0.0,
            verifier_info=verifier_info,
            total_tokens=self._total_tokens,
            total_tool_calls=self._total_tool_calls,
            wall_time_seconds=time.time() - self._start_time,
            metadata={
                **({"task_type": self.current_task.metadata.get("task_type")} if self.current_task.metadata else {}),
                "database_schema": self.current_task.database_schema,
                "final_query": self.current_query,
                "task": {
                    "task_id": self.current_task.task_id,
                    "prompt": self.current_task.prompt,
                    "database_schema": self.current_task.database_schema,
                    "sample_data": self.current_task.sample_data,
                    "expected_result": self.current_task.expected_result,
                    "expected_query": self.current_task.expected_query,
                    "metadata": self.current_task.metadata,
                },
            }
        )
    
    def __del__(self):
        """析构时清理数据库"""
        self._cleanup_db()
