# agent-rl

Unified workspace for:

- `agent-rl-core`: shared rollout/training framework
- `paper-c3rl`: mainline `A -> C3` upgrade (counterfactual cost-credit constraints)
- `paper-pirl`: robustness via prompt/interface randomization + invariance
- `paper-vera-rl`: verifier-uncertainty auxiliary module

## Local Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ./agent-rl-core
python3 -m pip install -e ./paper-c3rl
python3 -m pip install -e ./paper-pirl
python3 -m pip install -e ./paper-vera-rl

python3 agent-rl-core/scripts/train_baseline.py --config agent-rl-core/configs/base.yaml
python3 paper-c3rl/scripts/train.py --config paper-c3rl/configs/train_c3rl.yaml
python3 paper-pirl/scripts/train.py --config paper-pirl/configs/train_pirl.yaml
python3 paper-vera-rl/scripts/train.py --config paper-vera-rl/configs/train_vera.yaml
```

## Cluster / Slurm

Login node:

```bash
bash setup_cluster.sh
mkdir -p logs outputs
```

Submit jobs:

```bash
sbatch scripts/slurm/setup_env.sh
sbatch scripts/slurm/run_baseline.sh
sbatch scripts/slurm/run_c3_main.sh  # default: strict C3 config
sbatch scripts/slurm/run_pirl_main.sh
sbatch scripts/slurm/run_vera_aux.sh
```

To override C3 config:

```bash
C3_CONFIG=paper-c3rl/configs/train_c3rl.yaml sbatch scripts/slurm/run_c3_main.sh
```

One-shot suite:

```bash
sbatch run_job.sh
```
