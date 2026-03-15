# empathetic-trajectory-planning

# Environment Setup on the Explorer cluster

## Quick Start

First of all, log into the Explorer cluster via either commandline or OOD (Open OnDemand). Request a node, for example, by running 
```bash
srun -p courses-gpu --gres=gpu:p100:1 --time=05:00:00 --pty bash
```

# Commands
module load cuda/12.1.1 anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate affectplan

# Run a script
python -m experiments.script