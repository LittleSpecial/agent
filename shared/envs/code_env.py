"""
Code Environment for RLVR
代码修复/生成环境，使用单元测试作为验证器
"""

import subprocess
import tempfile
import os
import sys
import time
import uuid
import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .base import (
    BaseEnv, EnvConfig, Action, ActionType, 
    Observation, Step, Trajectory, VerifierInfo
)


@dataclass
class CodeTask:
    """代码任务定义"""
    task_id: str
    prompt: str
    initial_code: str = ""
    test_code: str = ""
    expected_output: Optional[str] = None
    language: str = "python"
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeEnv(BaseEnv):
    """
    代码环境：支持代码编写、执行、单元测试验证
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.current_task: Optional[CodeTask] = None
        self.current_code: str = ""
        self.execution_history: List[Dict[str, Any]] = []
        self._steps: List[Step] = []
        self._step_counter: int = 0
        self._start_time: float = 0.0
        self._total_tokens: int = 0
        self._total_tool_calls: int = 0
        
    def reset(self, task: Dict[str, Any]) -> Observation:
        """重置环境到新任务"""
        self.clear_cache()
        self.current_task = CodeTask(
            task_id=task.get("task_id", str(uuid.uuid4())),
            prompt=task["prompt"],
            initial_code=task.get("initial_code", ""),
            test_code=task.get("test_code", ""),
            expected_output=task.get("expected_output"),
            language=task.get("language", "python"),
            timeout=task.get("timeout", self.config.extra.get("default_timeout", 30.0)),
            metadata=task.get("metadata", {}),
        )
        
        self.current_code = self.current_task.initial_code
        self.execution_history = []
        self._steps = []
        self._step_counter = 0
        self._start_time = time.time()
        self._total_tokens = 0
        self._total_tool_calls = 0
        
        # 构建初始观察
        obs_content = f"Task: {self.current_task.prompt}\n\n"
        if self.current_task.initial_code:
            obs_content += f"Initial Code:\n```{self.current_task.language}\n{self.current_task.initial_code}\n```\n\n"
        show_tests = bool(self.config.extra.get("show_tests", True))
        if show_tests and self.current_task.test_code:
            obs_content += f"Test Cases:\n```{self.current_task.language}\n{self.current_task.test_code}\n```"
        
        return Observation(
            content=obs_content,
            obs_type="task_description",
            metadata={"task_id": self.current_task.task_id}
        )
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """执行一步动作"""
        self._step_counter += 1
        if action.action_type == ActionType.TOOL_CALL:
            self._total_tool_calls += 1
        
        if action.action_type == ActionType.CODE_WRITE:
            obs, reward, done, info = self._handle_code_write(action)
        elif action.action_type == ActionType.CODE_EXECUTE:
            obs, reward, done, info = self._handle_code_execute(action)
        elif action.action_type == ActionType.TOOL_CALL:
            obs, reward, done, info = self._handle_tool_call(action)
        else:
            obs = Observation(
                content=f"Unknown action type: {action.action_type}",
                obs_type="error"
            )
            reward, done, info = 0.0, False, {"error": "unknown_action"}
        
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
    
    def _handle_code_write(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """处理代码编写动作"""
        self.current_code = action.content
        
        obs = Observation(
            content=f"Code updated successfully.\n\nCurrent code:\n```{self.current_task.language}\n{self.current_code}\n```",
            obs_type="code_update"
        )
        
        return obs, 0.0, False, {"action": "code_write"}
    
    def _handle_code_execute(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """处理代码执行动作"""
        code_to_run = action.content if action.content else self.current_code
        
        # 检查缓存
        cache_key = f"exec_{hash(code_to_run)}"
        cached_result = self.get_cached_tool_output(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = self._execute_code(code_to_run)
        self.execution_history.append(result)
        
        if result["success"]:
            obs_content = f"Execution successful.\nOutput:\n{result['stdout']}"
        else:
            obs_content = f"Execution failed.\nError:\n{result['stderr']}"
        
        obs = Observation(
            content=obs_content,
            obs_type="execution_result",
            metadata=result
        )
        
        # 缓存结果
        response = (obs, 0.0, False, {"action": "code_execute", "result": result})
        self.cache_tool_output(cache_key, response)
        
        return response
    
    def _handle_tool_call(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """处理工具调用"""
        tool_name = action.tool_name
        tool_args = action.tool_args or {}
        
        if tool_name == "run_tests":
            return self._run_tests()
        elif tool_name == "get_code":
            obs = Observation(
                content=f"```{self.current_task.language}\n{self.current_code}\n```",
                obs_type="code_content"
            )
            return obs, 0.0, False, {"action": "get_code"}
        else:
            obs = Observation(
                content=f"Unknown tool: {tool_name}",
                obs_type="error"
            )
            return obs, 0.0, False, {"error": "unknown_tool"}
    
    def _run_tests(self) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """运行测试用例"""
        if not self.current_task.test_code:
            obs = Observation(content="No test cases available.", obs_type="info")
            return obs, 0.0, False, {"action": "run_tests", "no_tests": True}
        
        # 组合代码和测试
        full_code = f"{self.current_code}\n\n{self.current_task.test_code}"
        
        # 检查缓存
        cache_key = f"test_{hash(full_code)}"
        cached_result = self.get_cached_tool_output(cache_key)
        if cached_result is not None:
            return cached_result
        
        suite_output = self._execute_assert_suite(self.current_code, self.current_task.test_code)
        if suite_output is not None:
            result, verifier_info = suite_output
        else:
            result = self._execute_code(full_code)
            verifier_info = self._parse_test_result(result)
        
        if verifier_info.success:
            obs_content = f"All tests passed! ({verifier_info.passed_tests}/{verifier_info.total_tests})"
            reward = float(verifier_info.score)
            done = True
        else:
            obs_content = f"Tests failed ({verifier_info.passed_tests}/{verifier_info.total_tests}).\n"
            if result["stderr"]:
                obs_content += f"Error:\n{result['stderr']}"
            # Dense reward for sparse-feedback settings: keep verifier score even on failure.
            reward = float(verifier_info.score)
            done = False
        
        obs = Observation(
            content=obs_content,
            obs_type="test_result",
            metadata={"verifier_info": verifier_info.__dict__}
        )
        
        response = (obs, reward, done, {
            "action": "run_tests", 
            "verifier_info": verifier_info,
            "result": result
        })
        self.cache_tool_output(cache_key, response)
        
        return response

    def _extract_top_level_asserts(self, test_code: str) -> Optional[Tuple[str, List[str]]]:
        """
        Try to split test_code into:
          - setup/preamble code
          - a list of top-level assert statements
        Returns None when no suitable assert suite is found.
        """
        try:
            module = ast.parse(test_code)
        except SyntaxError:
            return None

        assert_srcs: List[str] = []
        preamble_nodes: List[ast.stmt] = []
        for node in module.body:
            if isinstance(node, ast.Assert):
                try:
                    assert_srcs.append(ast.unparse(node))
                except Exception:
                    return None
            else:
                preamble_nodes.append(node)

        # Fallback to old executor for single assert / opaque harnesses.
        if len(assert_srcs) < 2:
            return None

        try:
            preamble = ast.unparse(ast.Module(body=preamble_nodes, type_ignores=[])) if preamble_nodes else ""
        except Exception:
            return None
        return preamble, assert_srcs

    def _execute_assert_suite(
        self,
        code: str,
        test_code: str,
    ) -> Optional[Tuple[Dict[str, Any], VerifierInfo]]:
        """
        Execute top-level asserts one-by-one to obtain partial pass ratio.
        This gives a dense verifier score under sparse binary rewards.
        """
        parsed = self._extract_top_level_asserts(test_code)
        if parsed is None:
            return None
        preamble, assert_srcs = parsed

        script_parts: List[str] = [code, "", preamble, ""]
        script_parts += [
            "import json",
            "__passed = 0",
            "__failed = 0",
            "__errors = []",
            f"__assert_srcs = {repr(assert_srcs)}",
            "for __idx, __src in enumerate(__assert_srcs):",
            "    try:",
            "        exec(__src, globals(), globals())",
            "        __passed += 1",
            "    except AssertionError as _e:",
            "        __failed += 1",
            "        __errors.append(f'assert_{__idx}: {str(_e)}')",
            "    except Exception as _e:",
            "        __failed += 1",
            "        __errors.append(f'assert_{__idx}: {type(_e).__name__}: {_e}')",
            "print('__ASSERT_SUMMARY__' + json.dumps({",
            "    'passed': __passed,",
            "    'failed': __failed,",
            "    'total': __passed + __failed,",
            "    'errors': __errors[:20],",
            "}))",
        ]
        harness = "\n".join(part for part in script_parts if part is not None)
        result = self._execute_code(harness)

        marker = "__ASSERT_SUMMARY__"
        summary_line = None
        for line in result.get("stdout", "").splitlines()[::-1]:
            if line.startswith(marker):
                summary_line = line[len(marker):]
                break
        if summary_line is None:
            return None

        try:
            summary = json.loads(summary_line)
            passed = int(summary.get("passed", 0))
            total = int(summary.get("total", 0))
            failed = int(summary.get("failed", max(0, total - passed)))
            errors = list(summary.get("errors", []))
        except Exception:
            return None

        score = (float(passed) / float(total)) if total > 0 else 0.0
        verifier_info = VerifierInfo(
            success=(failed == 0 and total > 0),
            score=score,
            error_message=("; ".join(errors[:5]) if failed > 0 else None),
            error_type=("test_failure" if failed > 0 else None),
            passed_tests=passed,
            total_tests=total,
        )

        # Remove summary marker from displayed stdout for cleaner logs.
        cleaned_stdout_lines = [
            line for line in result.get("stdout", "").splitlines() if not line.startswith(marker)
        ]
        result["stdout"] = "\n".join(cleaned_stdout_lines).strip()
        return result, verifier_info
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """执行代码"""
        temp_path: Optional[str] = None
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.current_task.timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    # On some clusters / cancellation paths the temp file may already be gone.
                    pass
    
    def _parse_test_result(self, result: Dict[str, Any]) -> VerifierInfo:
        """解析测试结果"""
        # Fast-path success
        if result["success"]:
            return VerifierInfo(
                success=True,
                score=1.0,
                passed_tests=1,
                total_tests=1
            )

        # Best-effort parse for pytest/unittest summaries (dense score on failures).
        text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
        passed = 0
        failed = 0
        m_pass = re.search(r"(\d+)\s+passed", text)
        m_fail = re.search(r"(\d+)\s+failed", text)
        if m_pass:
            passed = int(m_pass.group(1))
        if m_fail:
            failed = int(m_fail.group(1))
        total = passed + failed

        if total > 0:
            return VerifierInfo(
                success=(failed == 0),
                score=float(passed) / float(total),
                error_message=result.get("stderr"),
                error_type=("test_failure" if failed > 0 else None),
                passed_tests=passed,
                total_tests=total,
            )

        return VerifierInfo(
            success=False,
            score=0.0,
            error_message=result.get("stderr"),
            error_type="test_failure",
            passed_tests=0,
            total_tests=1,
        )
    
    def verify(self) -> VerifierInfo:
        """验证当前代码是否通过所有测试"""
        if not self.current_task.test_code:
            # 没有测试用例，检查是否有期望输出
            if self.current_task.expected_output:
                result = self._execute_code(self.current_code)
                if result["success"] and result["stdout"].strip() == self.current_task.expected_output.strip():
                    return VerifierInfo(success=True, score=1.0)
                else:
                    return VerifierInfo(
                        success=False, 
                        score=0.0,
                        error_message="Output mismatch",
                        diff_info=f"Expected: {self.current_task.expected_output}\nGot: {result['stdout']}"
                    )
            return VerifierInfo(success=False, score=0.0, error_message="No verification criteria")
        
        # 运行测试
        _, _, _, info = self._run_tests()
        return info.get("verifier_info", VerifierInfo(success=False, score=0.0))
    
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
                "language": self.current_task.language,
                "final_code": self.current_code,
                "task": {
                    "task_id": self.current_task.task_id,
                    "prompt": self.current_task.prompt,
                    "initial_code": self.current_task.initial_code,
                    "test_code": self.current_task.test_code,
                    "expected_output": self.current_task.expected_output,
                    "language": self.current_task.language,
                    "timeout": self.current_task.timeout,
                    "metadata": self.current_task.metadata,
                },
            }
        )
