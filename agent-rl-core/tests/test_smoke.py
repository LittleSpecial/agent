from agent_rl_core.interfaces import StepRecord
from agent_rl_core.runner import RolloutConfig, RolloutRunner
from agent_rl_core.toy import ToyEnvironment, ToyPolicy, ToyVerifier, build_toy_tasks
from agent_rl_core.trainer import BaselineTrainer, TrainConfig


def test_step_record_defaults() -> None:
    record = StepRecord(state={}, action={})
    assert record.reward == 0.0


def test_runner_episode_smoke() -> None:
    tasks = build_toy_tasks(1, seed=11)
    policy = ToyPolicy(action_space=["inspect", "search", "tool_call", "reason", "answer"], seed=11)
    runner = RolloutRunner(
        config=RolloutConfig(max_steps=8),
        environment=ToyEnvironment(seed=11),
        policy=policy,
        verifier=ToyVerifier(),
    )
    output = runner.run_episode(tasks[0])
    assert output["task_id"] == tasks[0]["task_id"]
    assert "trajectory" in output
    assert output["trajectory"].task_id == tasks[0]["task_id"]


def test_trainer_smoke(tmp_path) -> None:
    tasks = build_toy_tasks(8, seed=12)
    policy = ToyPolicy(action_space=["inspect", "search", "tool_call", "reason", "answer"], seed=12)
    runner = RolloutRunner(
        config=RolloutConfig(max_steps=8),
        environment=ToyEnvironment(seed=12),
        policy=policy,
        verifier=ToyVerifier(),
    )
    trainer = BaselineTrainer(
        config=TrainConfig(total_updates=3, batch_size=4, eval_interval=2, save_interval=2, output_dir=str(tmp_path)),
        runner=runner,
        policy=policy,
        train_tasks=tasks,
        eval_tasks=tasks[:4],
    )
    result = trainer.train()
    assert "checkpoint" in result
