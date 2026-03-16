# Empathetic Trajectory Planning

## Environment Setup on the Explorer Cluster

### 1. Log Into Explorer

Log into the Explorer cluster via command line or OOD (Open OnDemand).

Then request a GPU node:

```bash
srun -p courses-gpu --gres=gpu:p100:1 --time=05:00:00 --pty bash
```

# Commands
```bash
module load cuda/12.1.1 anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate affectplan
```

# Run a script
```bash
python -m experiments.script
```