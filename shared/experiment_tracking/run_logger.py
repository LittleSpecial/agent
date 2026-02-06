"""
Run Logger - 单次运行的详细日志记录器
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from ..envs.base import Trajectory


class RunLogger:
    """单次运行的日志记录器"""
    
    def __init__(self, log_dir: Path, run_id: str):
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.trajectory_file = self.log_dir / f"trajectories_{run_id}.jsonl"
        self.step_file = self.log_dir / f"steps_{run_id}.jsonl"
    
    def log_trajectory(self, trajectory: Trajectory, extra: Optional[Dict] = None):
        """记录完整轨迹"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'trajectory_id': trajectory.trajectory_id,
            'task_id': trajectory.task_id,
            'success': trajectory.success,
            'r_final': trajectory.r_final,
            'length': trajectory.length,
            'total_tokens': trajectory.total_tokens,
            'wall_time': trajectory.wall_time_seconds,
        }
        if extra:
            record.update(extra)
        
        with open(self.trajectory_file, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    
    def log_step(self, trajectory_id: str, step_id: int, 
                 action: str, observation: str, 
                 credit: Optional[float] = None,
                 extra: Optional[Dict] = None):
        """记录单步"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'trajectory_id': trajectory_id,
            'step_id': step_id,
            'action': action[:500],  # 截断
            'observation': observation[:500],
            'credit': credit,
        }
        if extra:
            record.update(extra)
        
        with open(self.step_file, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
